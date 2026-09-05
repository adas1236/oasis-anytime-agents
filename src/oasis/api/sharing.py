"""Opt-in, password-protected public UI sharing; never enabled by app imports."""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib.util
import json
import os
import secrets
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send


def _require_gradio() -> None:
    # Do not import Gradio (or start its analytics) in the model-serving process.
    if importlib.util.find_spec("gradio") is None:
        raise ValueError(
            "--share requires Gradio: uv sync --extra share "
            "(for CUDA, add --no-group cpu --group gpu)."
        )


def _password() -> str:
    configured = os.environ.get("OASIS_SHARE_PASSWORD")
    if configured is None:
        return secrets.token_urlsafe(24)
    if len(configured) < 12 or not configured.isascii() or not configured.isprintable():
        raise ValueError(
            "OASIS_SHARE_PASSWORD must contain at least 12 printable ASCII characters."
        )
    return configured


def validate_share(*, host: str, serve_ui: bool) -> None:
    """Fail before binding a socket, loading a model, or contacting the relay."""
    if not serve_ui:
        raise ValueError("--share requires --serve-ui (and cannot be used with --no-serve-ui).")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("--share requires a loopback --host, e.g. 127.0.0.1; do not use 0.0.0.0.")
    _password()
    _require_gradio()


class ShareAuthMiddleware:
    """Authenticate every HTTP route without consuming or buffering SSE streams."""

    def __init__(self, app: ASGIApp, *, password: str) -> None:
        self.app = app
        self.expected = f"oasis:{password}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        scheme, _, encoded = headers.get("authorization", "").partition(" ")
        try:
            credentials = (
                base64.b64decode(encoded, validate=True) if scheme.lower() == "basic" else b""
            )
        except (ValueError, binascii.Error):
            credentials = b""
        if not secrets.compare_digest(credentials, self.expected):
            response = PlainTextResponse(
                "Authentication required",
                status_code=401,
                headers={
                    "WWW-Authenticate": 'Basic realm="OASIS", charset="UTF-8"',
                    "Cache-Control": "no-store",
                },
            )
            await response(scope, receive, send)
            return
        # Basic credentials are cached by browsers: reject cross-origin writes so
        # another website cannot spend an authenticated visitor's GPU budget.
        origin = headers.get("origin")
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"} and (
            headers.get("sec-fetch-site") == "cross-site"
            or (
                origin is not None
                and origin
                not in {f"https://{headers.get('host')}", f"http://{headers.get('host')}"}
            )
        ):
            await PlainTextResponse("Cross-origin request rejected", status_code=403)(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)


def _signal_worker(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    try:
        if os.name == "posix":
            # The worker owns this new process group, including Gradio's FRP child.
            os.killpg(proc.pid, sig)
        elif proc.returncode is None:
            proc.terminate() if sig == signal.SIGTERM else proc.kill()
    except ProcessLookupError:
        pass


async def _stop_worker(proc: asyncio.subprocess.Process) -> None:
    if proc.stdin is not None:
        proc.stdin.close()  # EOF asks the worker to stop and reap its own FRP child.
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        _signal_worker(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            _signal_worker(proc, signal.SIGKILL)
            await proc.wait()
    if proc.returncode not in {None, 0} and os.name == "posix":
        _signal_worker(proc, signal.SIGKILL)


async def _read_share_url(proc: asyncio.subprocess.Process) -> str:
    if proc.stdout is None:
        raise ValueError("Gradio share worker has no output stream.")
    while line := await proc.stdout.readline():
        try:
            message = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue  # Ignore dependency notices; never relay raw logs or secrets.
        if not isinstance(message, dict):
            continue
        if message.get("status") == "error":
            raise ValueError("Gradio could not create a share link.")
        if message.get("status") == "ready":
            url = message.get("url")
            if isinstance(url, str) and urlsplit(url).scheme == "https" and urlsplit(url).hostname:
                return url
            raise ValueError("Gradio did not return an HTTPS share URL.")
    raise ValueError("Gradio share worker exited before creating a link.")


@asynccontextmanager
async def public_tunnel(host: str, port: int) -> AsyncIterator[str]:
    """Use the same tunnel helper as Gradio launch(share=True), isolated for cleanup."""
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED="False")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "oasis.api.share_worker",
        "--host",
        host,
        "--port",
        str(port),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=os.name == "posix",
        env=env,
    )
    try:
        try:
            # Gradio may first download its checksum-verified FRP binary. A worker
            # process lets us enforce a real deadline even if its blocking I/O stalls.
            url = await asyncio.wait_for(_read_share_url(proc), timeout=90)
        except (TimeoutError, ValueError, OSError):
            raise ValueError(
                "Could not create a Gradio share link within 90 seconds. Check outbound network "
                "access to api.gradio.app and cdn-media.huggingface.co, writable HF_HOME and "
                "working directory, and https://status.gradio.app. No account/token is needed."
            ) from None
        yield url
    finally:
        await _stop_worker(proc)


def serve_shared(app: FastAPI, *, host: str, port: int) -> None:
    """Run the existing app with auth and a tunnel tied to its ASGI lifespan."""
    if host == "localhost":
        host = "127.0.0.1"
    password = _password()
    app.add_middleware(ShareAuthMiddleware, password=password)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with original_lifespan(application), public_tunnel(host, port) as url:
            print(f"Public UI: {url}", flush=True)
            print("Username: oasis", flush=True)
            if "OASIS_SHARE_PASSWORD" in os.environ:
                print("Password: the value of OASIS_SHARE_PASSWORD", flush=True)
            else:
                print(f"Password: {password}", flush=True)
            print(
                "Anyone with these credentials can use the GPU and access run data. "
                "Ctrl+C stops sharing. Do not publish this terminal output.",
                flush=True,
            )
            yield

    app.router.lifespan_context = lifespan
    # Reserve the exact port BEFORE publishing. An occupied port must never cause
    # us to forward public requests to a different, potentially unprotected app.
    config = uvicorn.Config(app, host=host, port=port, timeout_graceful_shutdown=10)
    server = uvicorn.Server(config)
    sock = config.bind_socket()
    try:
        server.run(sockets=[sock])
        if not server.started:
            raise ValueError("Shared UI startup failed; the server was not started.")
    finally:
        sock.close()
