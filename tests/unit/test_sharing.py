from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from oasis.api import share_worker, sharing
from oasis.cli import _parser, _run, _serve


@pytest.mark.parametrize("path", ["/", "/src/app.js", "/api/v1/runs", "/api/v1/docs"])
@pytest.mark.parametrize("authorization", ["", "Bearer secret", "Basic !!!", "Basic /w=="])
async def test_all_routes_require_auth(path: str, authorization: str) -> None:
    upstream = AsyncMock()
    app = sharing.ShareAuthMiddleware(upstream, password="a-long-password")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://ui.test"
    ) as client:
        response = await client.get(path, headers={"Authorization": authorization})
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Basic")
    upstream.assert_not_called()


async def test_authenticated_stream_and_cross_origin_protection() -> None:
    app = FastAPI()

    @app.get("/events")
    async def events() -> StreamingResponse:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"data: first\n\n"
            yield b"data: second\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    @app.post("/api/v1/runs")
    async def create() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(sharing.ShareAuthMiddleware, password="a-long-password")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="https://ui.test",
        auth=("oasis", "a-long-password"),
    ) as client:
        response = await client.get("/events")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.content == b"data: first\n\ndata: second\n\n"
        assert (
            await client.post("/api/v1/runs", headers={"Origin": "https://evil.test"})
        ).status_code == 403
        assert (
            await client.post("/api/v1/runs", headers={"Origin": "https://ui.test"})
        ).status_code == 200
        assert (await client.get("/events", auth=("oasis", "wrong"))).status_code == 401


@pytest.mark.parametrize(
    ("host", "ui", "message"),
    [
        ("127.0.0.1", False, "requires --serve-ui"),
        ("0.0.0.0", True, "loopback"),
    ],
)
def test_share_validation(
    monkeypatch: pytest.MonkeyPatch, host: str, ui: bool, message: str
) -> None:
    sdk = MagicMock()
    monkeypatch.setattr(sharing, "_require_gradio", sdk)
    with pytest.raises(ValueError, match=message):
        sharing.validate_share(host=host, serve_ui=ui)
    sdk.assert_not_called()


def test_password_and_missing_gradio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OASIS_SHARE_PASSWORD", raising=False)
    assert len(sharing._password()) >= 24
    monkeypatch.setenv("OASIS_SHARE_PASSWORD", "short")
    with pytest.raises(ValueError, match="12 printable"):
        sharing._password()
    monkeypatch.setattr(sharing.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="--extra share"):
        sharing._require_gradio()


def test_share_needs_no_account_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NGROK_AUTHTOKEN", raising=False)
    monkeypatch.delenv("OASIS_SHARE_PASSWORD", raising=False)
    monkeypatch.setattr(sharing.importlib.util, "find_spec", lambda name: object())
    sharing.validate_share(host="127.0.0.1", serve_ui=True)


def mock_worker(monkeypatch: pytest.MonkeyPatch, message: dict[str, str]) -> SimpleNamespace:
    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps(message).encode() + b"\n")
    reader.feed_eof()
    proc = SimpleNamespace(
        stdin=MagicMock(), stdout=reader, wait=AsyncMock(return_value=0), returncode=0, pid=1234
    )
    monkeypatch.setattr(sharing.asyncio, "create_subprocess_exec", AsyncMock(return_value=proc))
    return proc


@pytest.mark.parametrize("fail_body", [False, True])
async def test_tunnel_https_and_cleanup(monkeypatch: pytest.MonkeyPatch, fail_body: bool) -> None:
    proc = mock_worker(monkeypatch, {"status": "ready", "url": "https://shared.gradio.live"})
    try:
        async with sharing.public_tunnel("127.0.0.1", 8765) as url:
            assert url == "https://shared.gradio.live"
            if fail_body:
                raise RuntimeError("server failed")
    except RuntimeError:
        assert fail_body
    args, kwargs = sharing.asyncio.create_subprocess_exec.call_args
    assert args[1:] == ("-m", "oasis.api.share_worker", "--host", "127.0.0.1", "--port", "8765")
    assert kwargs["env"]["GRADIO_ANALYTICS_ENABLED"] == "False"
    proc.stdin.close.assert_called_once()
    proc.wait.assert_awaited_once()


async def test_provider_errors_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = mock_worker(monkeypatch, {"status": "error", "detail": "secret-token"})
    with pytest.raises(ValueError) as error:
        async with sharing.public_tunnel("127.0.0.1", 8000):
            pytest.fail("must not start")
    assert "secret-token" not in str(error.value)
    assert "No account/token is needed" in str(error.value)
    proc.stdin.close.assert_called_once()


async def test_non_https_url_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = mock_worker(monkeypatch, {"status": "ready", "url": "http://unsafe.test"})
    with pytest.raises(ValueError, match="Gradio share link"):
        async with sharing.public_tunnel("127.0.0.1", 8000):
            pytest.fail("must not publish")
    proc.stdin.close.assert_called_once()


async def test_startup_timeout_closes_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = mock_worker(monkeypatch, {})
    monkeypatch.setattr(sharing, "_read_share_url", AsyncMock(side_effect=TimeoutError))
    with pytest.raises(ValueError, match="90 seconds"):
        async with sharing.public_tunnel("127.0.0.1", 8000):
            pytest.fail("must not publish")
    proc.stdin.close.assert_called_once()
    proc.wait.assert_awaited_once()


async def test_hung_worker_escalates_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = mock_worker(monkeypatch, {})
    proc.wait.side_effect = [TimeoutError, TimeoutError, 0]
    terminate = MagicMock()
    monkeypatch.setattr(sharing, "_signal_worker", terminate)
    await sharing._stop_worker(proc)
    assert [call.args[1] for call in terminate.call_args_list] == [
        sharing.signal.SIGTERM,
        sharing.signal.SIGKILL,
    ]


@pytest.mark.parametrize("fail_startup", [False, True])
def test_worker_uses_gradio_launch_helper_and_cleans_own_tunnel(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fail_startup: bool
) -> None:
    unrelated = SimpleNamespace(share_token="someone-else", kill=MagicMock())
    tunnels = [unrelated]
    child = MagicMock()
    own = SimpleNamespace(share_token=None, proc=child, kill=MagicMock())

    def setup(**kwargs: object) -> str:
        assert kwargs["local_host"] == "127.0.0.1"
        assert kwargs["local_port"] == 8765
        assert kwargs["share_server_address"] is None
        assert kwargs["share_server_tls_certificate"] is None
        own.share_token = kwargs["share_token"]
        assert len(own.share_token) >= 32
        tunnels.append(own)
        if fail_startup:
            raise RuntimeError("setup failed")
        return "https://example.gradio.live"

    modules = {
        "gradio.networking": SimpleNamespace(setup_tunnel=setup),
        "gradio.tunneling": SimpleNamespace(CURRENT_TUNNELS=tunnels),
    }
    monkeypatch.setattr(share_worker.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(share_worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"")))
    if fail_startup:
        with pytest.raises(RuntimeError, match="setup failed"):
            share_worker.run("127.0.0.1", 8765)
    else:
        assert share_worker.run("127.0.0.1", 8765) == 0
        assert json.loads(capsys.readouterr().out)["url"] == "https://example.gradio.live"
    own.kill.assert_called_once()
    child.wait.assert_called_once_with(timeout=3)
    unrelated.kill.assert_not_called()
    assert tunnels == [unrelated]


def test_shared_server_lifecycle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    monkeypatch.setenv("OASIS_SHARE_PASSWORD", "do-not-print-this-password")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        order.append("app-start")
        yield
        order.append("app-stop")

    app = FastAPI(lifespan=lifespan)

    @asynccontextmanager
    async def tunnel(host: str, port: int) -> AsyncIterator[str]:
        assert (host, port) == ("127.0.0.1", 8765)
        order.append("tunnel-start")
        yield "https://shared.test"
        order.append("tunnel-stop")

    socket = MagicMock()

    def bind(config: object) -> MagicMock:
        order.append("bind")
        return socket

    class Server:
        started = True

        def __init__(self, config: object) -> None:
            pass

        def run(self, *, sockets: list[object]) -> None:
            assert sockets == [socket]

            async def run_lifespan() -> None:
                async with app.router.lifespan_context(app):
                    order.append("serve")

            asyncio.run(run_lifespan())

    monkeypatch.setattr(sharing, "public_tunnel", tunnel)
    monkeypatch.setattr(sharing.uvicorn.Config, "bind_socket", bind)
    monkeypatch.setattr(sharing.uvicorn, "Server", Server)
    sharing.serve_shared(app, host="localhost", port=8765)
    assert order == ["bind", "app-start", "tunnel-start", "serve", "tunnel-stop", "app-stop"]
    socket.close.assert_called_once()
    assert app.user_middleware[0].cls is sharing.ShareAuthMiddleware
    output = capsys.readouterr().out
    assert "https://shared.test" in output
    assert "Username: oasis" in output
    assert "do-not-print-this-password" not in output


def test_cli_routes_sharing_without_real_server(monkeypatch: pytest.MonkeyPatch) -> None:
    import oasis.api

    validate = MagicMock()
    shared = MagicMock()
    create = MagicMock()
    monkeypatch.setattr(sharing, "validate_share", validate)
    monkeypatch.setattr(sharing, "serve_shared", shared)
    monkeypatch.setattr(oasis.api, "create_app", create)
    args = _parser().parse_args(
        ["serve", "--backend", "fake", "--serve-ui", "--share", "--port", "8765"]
    )
    assert _serve(args) == 0
    validate.assert_called_once_with(host="127.0.0.1", serve_ui=True)
    shared.assert_called_once_with(create.return_value, host="127.0.0.1", port=8765)


def test_missing_gradio_returns_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sharing.importlib.util, "find_spec", lambda name: None)
    assert _run(["serve", "--backend", "fake", "--serve-ui", "--share"]) == 2


def test_occupied_port_never_opens_tunnel(monkeypatch: pytest.MonkeyPatch) -> None:
    tunnel = MagicMock()
    monkeypatch.setattr(sharing, "public_tunnel", tunnel)
    monkeypatch.setattr(sharing.uvicorn.Config, "bind_socket", MagicMock(side_effect=SystemExit(1)))
    with pytest.raises(SystemExit):
        sharing.serve_shared(FastAPI(), host="127.0.0.1", port=8000)
    tunnel.assert_not_called()


def test_server_failure_closes_reserved_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = MagicMock()
    server = MagicMock()
    server.run.side_effect = RuntimeError("server failure")
    monkeypatch.setattr(sharing.uvicorn.Config, "bind_socket", MagicMock(return_value=socket))
    monkeypatch.setattr(sharing.uvicorn, "Server", MagicMock(return_value=server))
    with pytest.raises(RuntimeError, match="server failure"):
        sharing.serve_shared(FastAPI(), host="127.0.0.1", port=8000)
    socket.close.assert_called_once()
