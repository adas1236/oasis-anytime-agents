from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from integration.test_anytime_controller import published_problem
from integration.test_anytime_controller import request as controller_request
from oasis.config import RuntimePolicy
from oasis.controller import AnytimeController, BudgetSpec, LocalRunStore
from oasis.llm import (
    ChatMessage,
    FakeModelBackend,
    ModelCapabilities,
    ModelRequest,
    RuntimeModelBackend,
)
from oasis.llm.profiles import resolve_model_profile
from oasis.model_worker import create_model_worker_app
from oasis.model_worker.schemas import MODEL_WORKER_SCHEMA_VERSION, WorkerStreamEnvelope
from oasis.runtimes import (
    ConservativeRuntimePlanner,
    RemoteModelRuntime,
    RemoteRuntimeError,
    fake_inventory,
)


@pytest.mark.asyncio
async def test_authenticated_worker_and_remote_runtime_complete_normal_backend_contract() -> None:
    profile = resolve_model_profile("gemma4_e2b_it")
    worker_backend = FakeModelBackend(["remote reply"], profile=profile)
    app = create_model_worker_app(worker_backend, auth_token="test-token")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        plan = ConservativeRuntimePlanner().plan(
            profile,
            fake_inventory(),
            RuntimePolicy(
                device="auto",
                engine="remote",
                remote_endpoint="http://worker",
            ),
        )
        runtime = RemoteModelRuntime("http://worker", auth_token="test-token", client=client)
        backend = RuntimeModelBackend(
            profile=profile,
            capabilities=worker_backend.capabilities,
            runtime=runtime,
            plan=plan,
        )
        await backend.load()
        request = ModelRequest(
            request_id="remote-complete",
            messages=(ChatMessage(role="user", content="hello"),),
            max_generated_tokens=10,
        )
        turn = await backend.generate(request)
        usage = await runtime.usage(request.request_id)

    assert turn.message.content == "remote reply"
    assert turn.usage.total_tokens > 0
    assert usage.found and usage.usage == turn.usage
    assert backend.runtime_plan.requested_model_id == profile.model_id


@pytest.mark.asyncio
async def test_worker_rejects_authentication_without_exposing_token() -> None:
    app = create_model_worker_app(FakeModelBackend(), auth_token="secret-value")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        response = await client.get(
            "/api/v1/capabilities", headers={"Authorization": "Bearer wrong"}
        )
        version_response = await client.post(
            "/api/v1/tokens/count",
            headers={"Authorization": "Bearer secret-value"},
            json={
                "schema_version": "0.0.0",
                "request": {
                    "request_id": "bad-version",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            },
        )
    assert response.status_code == 401
    assert "secret-value" not in response.text
    assert version_response.status_code == 409
    assert version_response.json()["code"] == "version_mismatch"


class _Lines(httpx.AsyncByteStream):
    def __init__(self, lines: list[bytes], *, disconnect: bool = False) -> None:
        self.lines = lines
        self.disconnect = disconnect

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for line in self.lines:
            yield line
            await asyncio.sleep(0)
        if self.disconnect:
            raise httpx.ReadError("simulated disconnect")


class FakeWorkerTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        *,
        version: str = MODEL_WORKER_SCHEMA_VERSION,
        disconnect: bool = False,
        model_revision: str | None = None,
    ) -> None:
        self.version = version
        self.disconnect = disconnect
        self.aborted: list[str] = []
        self.profile = resolve_model_profile("gemma4_e2b_it")
        default_backend = FakeModelBackend(profile=self.profile)
        self.worker_backend = FakeModelBackend(
            profile=self.profile,
            runtime_plan=default_backend.runtime_plan.model_copy(
                update={"model_revision": model_revision}
            ),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.headers.get("Authorization") != "Bearer test-token":
            return httpx.Response(401, request=request, json={"code": "authentication_failed"})
        if path.endswith("/health"):
            return httpx.Response(
                200, request=request, json={"schema_version": self.version, "status": "ok"}
            )
        if path.endswith("/capabilities"):
            return httpx.Response(
                200,
                request=request,
                json={
                    "schema_version": self.version,
                    "model_profile": self.profile.name,
                    "model_id": self.profile.model_id,
                    "capabilities": self.worker_backend.capabilities.model_dump(mode="json"),
                    "runtime_plan": self.worker_backend.runtime_plan.model_dump(mode="json"),
                    "inventory": self.worker_backend.compute_inventory.model_dump(mode="json"),
                },
            )
        if "/abort/" in path:
            request_id = path.rsplit("/", 1)[1]
            self.aborted.append(request_id)
            return httpx.Response(
                200,
                request=request,
                json={
                    "schema_version": self.version,
                    "request_id": request_id,
                    "abort_requested": True,
                },
            )
        if path.endswith("/generate"):
            payload = json.loads(request.content)
            request_id = payload["request"]["request_id"]
            deltas = [
                WorkerStreamEnvelope(type="delta", delta={"text": "partial"}),
            ]
            if not self.disconnect:
                deltas.append(
                    WorkerStreamEnvelope(
                        type="delta",
                        delta={
                            "usage": {"input_tokens": 1, "generated_tokens": 1},
                            "finish_reason": (
                                "cancelled" if request_id in self.aborted else "stop"
                            ),
                        },
                    )
                )
            lines = [(item.model_dump_json() + "\n").encode() for item in deltas]
            return httpx.Response(
                200,
                request=request,
                headers={"Content-Type": "application/x-ndjson"},
                stream=_Lines(lines, disconnect=self.disconnect),
            )
        if path.endswith("/tokens/count"):
            return httpx.Response(
                200,
                request=request,
                json={"schema_version": self.version, "input_tokens": 1},
            )
        raise AssertionError(path)


def remote_backend(
    transport: httpx.AsyncBaseTransport,
    *,
    model_revision: str | None = None,
) -> tuple[RuntimeModelBackend, RemoteModelRuntime]:
    profile = resolve_model_profile("gemma4_e2b_it")
    client = httpx.AsyncClient(transport=transport, base_url="http://worker")
    runtime = RemoteModelRuntime("http://worker", auth_token="test-token", client=client)
    plan = ConservativeRuntimePlanner().plan(
        profile,
        fake_inventory(),
        RuntimePolicy(device="auto", engine="remote", remote_endpoint="http://worker"),
        revision=model_revision,
    )
    backend = RuntimeModelBackend(
        profile=profile,
        capabilities=ModelCapabilities(
            generative=True,
            chat_template=True,
            native_tools=True,
            structured_fallback=True,
            reasoning_channels=True,
            streaming_abort=True,
        ),
        runtime=runtime,
        plan=plan,
    )
    return backend, runtime


@pytest.mark.asyncio
async def test_remote_abort_and_disconnect_are_structured() -> None:
    transport = FakeWorkerTransport()
    backend, runtime = remote_backend(transport)
    await backend.load()
    await backend.abort("cancel-me")
    assert transport.aborted == ["cancel-me"]
    request = ModelRequest(
        request_id="cancel-me",
        messages=(ChatMessage(role="user", content="hello"),),
    )
    deltas = [delta async for delta in backend.stream(request)]
    assert deltas[-1].finish_reason == "cancelled"
    await runtime._client.aclose()

    disconnected, disconnected_runtime = remote_backend(FakeWorkerTransport(disconnect=True))
    await disconnected.load()
    with pytest.raises(RemoteRuntimeError, match="disconnected"):
        _ = [delta async for delta in disconnected.stream(request)]
    await disconnected_runtime._client.aclose()


@pytest.mark.asyncio
async def test_remote_version_mismatch_and_authentication_failure_are_clear() -> None:
    mismatched, mismatched_runtime = remote_backend(FakeWorkerTransport(version="9.9.9"))
    with pytest.raises(RemoteRuntimeError, match="version mismatch"):
        await mismatched.load()
    await mismatched_runtime._client.aclose()

    profile = resolve_model_profile("gemma4_e2b_it")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request, json={"code": "authentication_failed"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime = RemoteModelRuntime("http://worker", auth_token="wrong", client=client)
    plan = ConservativeRuntimePlanner().plan(
        profile,
        fake_inventory(),
        RuntimePolicy(device="auto", engine="remote", remote_endpoint="http://worker"),
    )
    with pytest.raises(RemoteRuntimeError, match="authentication failed"):
        await runtime.load(profile, plan)
    await client.aclose()


@pytest.mark.asyncio
async def test_remote_runtime_rejects_a_different_model_revision() -> None:
    backend, runtime = remote_backend(FakeWorkerTransport(), model_revision="expected-revision")
    with pytest.raises(RemoteRuntimeError, match="different model revision"):
        await backend.load()
    await runtime._client.aclose()


@pytest.mark.asyncio
async def test_remote_worker_failure_cannot_erase_controller_incumbent(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    backend, runtime = remote_backend(FakeWorkerTransport(disconnect=True))
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
        backend=backend,
    )

    result = await controller.run(
        controller_request(
            run_id="remote-worker-failure",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=2_000,
                max_total_model_tokens=100,
                max_generated_tokens=20,
                max_tool_calls=1,
            ),
            enable_fallback=False,
        )
    )

    assert any("model action failed" in failure for failure in result.failures)
    assert result.best_plan is not None
    assert result.best_scorecard is not None and result.best_scorecard.feasible
    assert len(result.incumbent_timeline) == 1
    await runtime._client.aclose()
