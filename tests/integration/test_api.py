from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from unit.test_location_allocation import problem_fixture, provenance

from oasis.api import create_app
from oasis.api.schemas import (
    ChatResponse,
    HealthResponse,
    ModelCatalogResponse,
    ProblemCatalogResponse,
    RunCreatedResponse,
    RunInspectionResponse,
    RuntimeResponse,
)
from oasis.artifacts import LocalArtifactStore, put_json
from oasis.config import BackendKind, OasisSettings
from oasis.controller import EventKind, LocalRunStore
from oasis.evidence import run_evidence_demo
from oasis.llm import FakeModelBackend, ToolCall
from oasis.problems import LocationAllocationPolicy, LocationProblemType
from oasis.schemas import (
    ArtifactKind,
    ArtifactMetadata,
    DeterminismClassification,
    Plan,
    SideEffectClassification,
    ToolEvent,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools import ToolContext, ToolRegistry
from oasis.tools.decision import RenderMapTool


def published_problem(tmp_path: Path) -> tuple[LocalArtifactStore, str, str, Any]:
    artifacts, problem = problem_fixture(
        tmp_path / "artifacts",
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    problem_ref = put_json(
        artifacts,
        problem.model_dump(mode="json"),
        kind=ArtifactKind.JSON_SPECIFICATION,
        units="unitless",
        provenance=provenance("phase8-problem"),
        data_schema={"type": "LocationAllocationProblem", "version": problem.schema_version},
    )
    baseline = Plan(problem_type=problem.type_id.value, selected_site_ids=("s1",))
    baseline_ref = put_json(
        artifacts,
        baseline.model_dump(mode="json"),
        kind=ArtifactKind.PLAN,
        units="unitless",
        provenance=provenance("phase8-baseline"),
        data_schema={"type": "Plan", "version": baseline.schema_version},
    )
    return artifacts, problem_ref.id, baseline_ref.id, problem


def settings(tmp_path: Path, **overrides: object) -> OasisSettings:
    return OasisSettings(
        backend=BackendKind.FAKE,
        artifact_root=tmp_path / "artifacts",
        run_root=tmp_path / "runs",
        api_sse_heartbeat_seconds=0.01,
        **overrides,
    )


@asynccontextmanager
async def api_client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def run_body(problem_id: str, baseline_id: str, run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "source": {
            "kind": "artifact",
            "problem_artifact_id": problem_id,
            "baseline_plan_artifact_id": baseline_id,
        },
        "budget": {"wall_time_ms": 2_000},
        "enable_model": False,
    }


async def wait_for_result(client: httpx.AsyncClient, run_id: str) -> RunInspectionResponse:
    for _ in range(200):
        response = await client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        inspection = RunInspectionResponse.model_validate(response.json())
        if inspection.result is not None:
            return inspection
        await asyncio.sleep(0.005)
    raise AssertionError("run did not finalize")


def event_ids(body: str) -> list[int]:
    return [int(line.removeprefix("id: ")) for line in body.splitlines() if line.startswith("id: ")]


class LoadingFakeBackend(FakeModelBackend):
    def __init__(
        self, responses: list[str | ToolCall] | None = None, *, delay: float = 0.02
    ) -> None:
        super().__init__(responses or ())
        self.delay = delay
        self.load_calls = 0

    async def load(self) -> None:
        self.load_calls += 1
        await asyncio.sleep(self.delay)


@pytest.mark.asyncio
async def test_openapi_discovery_chat_and_runtime_are_versioned_and_non_probing(
    tmp_path: Path,
) -> None:
    backend = LoadingFakeBackend()
    app = create_app(settings(tmp_path), backend=backend)

    async with api_client(app) as client:
        health = HealthResponse.model_validate((await client.get("/api/v1/health")).json())
        models = ModelCatalogResponse.model_validate((await client.get("/api/v1/models")).json())
        tools = (await client.get("/api/v1/tools")).json()
        problems = (await client.get("/api/v1/problems")).json()
        schema = (await client.get("/api/v1/openapi.json")).json()
        chat_response = await client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_generated_tokens": 8,
            },
        )
        second_chat = await client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "again"}]},
        )
        runtime = RuntimeResponse.model_validate((await client.get("/api/v1/runtime")).json())

    chat = ChatResponse.model_validate(chat_response.json())
    second = ChatResponse.model_validate(second_chat.json())
    assert health.api_version == "v1"
    assert len(models.models) == 5
    assert runtime.inventory_probed is False
    assert runtime.inventory["accelerator_count"] == 0
    assert runtime.inventory["discovery_mode"] == "fake"
    assert runtime.inventory["accelerators"] == []
    assert runtime.resolved_plan["runtime"] == "fake"
    assert runtime.resolved_plan["requested_model_id"] == models.active_model_id
    assert runtime.capabilities.remote is True
    assert chat.content == "[fake] hello"
    assert chat.model_startup_ms == runtime.model_startup_ms
    assert second.model_startup_ms == chat.model_startup_ms
    assert backend.load_calls == 1
    assert chat.model_startup_ms >= 10
    assert chat.usage.total_tokens > 0
    assert any(tool["name"] == "improve" for tool in tools["tools"])
    assert any(problem["type_id"] == "mobile_service_route" for problem in problems["problems"])
    assert "/api/v1/runs/{run_id}/events" in schema["paths"]
    assert schema["info"]["version"] == "1.1.0"
    assert "ApiErrorResponse" in schema["components"]["schemas"]


@pytest.mark.asyncio
async def test_lazy_model_loading_is_outside_run_budget_accounting(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    backend = LoadingFakeBackend(
        ['{"type":"stop","rationale":"enough"}'],
        delay=0.2,
    )
    runs = LocalRunStore(tmp_path / "runs")
    app = create_app(
        settings(tmp_path),
        backend=backend,
        artifact_store=artifacts,
        run_store=runs,
    )

    async with api_client(app) as client:
        created = await client.post(
            "/api/v1/runs",
            json={
                **run_body(problem_id, baseline_id, "startup-separated"),
                "budget": {
                    "wall_time_ms": 2_000,
                    "max_total_model_tokens": 500,
                    "max_generated_tokens": 50,
                    "max_tool_calls": 1,
                },
                "enable_model": True,
                "enable_deterministic_fallback": False,
            },
        )
        assert created.status_code == 202
        inspection = await wait_for_result(client, "startup-separated")
        runtime = RuntimeResponse.model_validate((await client.get("/api/v1/runtime")).json())

    assert inspection.result is not None
    assert inspection.result.terminal_reason == "model_stopped"
    assert inspection.result.best_plan is not None
    assert backend.load_calls == 1
    assert runtime.model_startup_ms >= 150
    assert inspection.result.runtime_plan["model_startup_ms"] == runtime.model_startup_ms
    assert runs.read_events("startup-separated")[0].relative_monotonic_ms < 150


@pytest.mark.asyncio
async def test_create_inspect_replay_reconnect_map_and_restart(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    runs = LocalRunStore(tmp_path / "runs")
    first_app = create_app(
        settings(tmp_path),
        backend=FakeModelBackend(),
        artifact_store=artifacts,
        run_store=runs,
    )

    async with api_client(first_app) as client:
        created_response = await client.post(
            "/api/v1/runs", json=run_body(problem_id, baseline_id, "api-replay")
        )
        assert created_response.status_code == 202
        created = RunCreatedResponse.model_validate(created_response.json())
        inspection = await wait_for_result(client, created.run_id)
        assert inspection.result is not None and inspection.result.best_plan is not None
        complete_stream = await client.get(created.links.events)
        ids = event_ids(complete_stream.text)
        assert ids == list(range(len(ids)))
        assert "event: baseline_committed" in complete_stream.text
        assert "event: run_finalized" in complete_stream.text

        reconnected = await client.get(created.links.events, headers={"Last-Event-ID": "2"})
        reconnect_ids = event_ids(reconnected.text)
        assert reconnect_ids and min(reconnect_ids) > 2

        plan_response = await client.get(
            f"/api/v1/artifacts/{inspection.result.best_plan_artifact_id}"
        )
        map_response = await client.get(created.links.map)
        assert plan_response.status_code == 200
        assert plan_response.headers["x-content-type-options"] == "nosniff"
        assert map_response.status_code == 200
        assert map_response.headers["content-type"].startswith("application/geo+json")
        assert json.loads(map_response.content)["type"] == "FeatureCollection"

    restarted_app = create_app(
        settings(tmp_path),
        backend=FakeModelBackend(),
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
    )
    async with api_client(restarted_app) as client:
        recovered = RunInspectionResponse.model_validate(
            (await client.get("/api/v1/runs/api-replay")).json()
        )
        replay = await client.get("/api/v1/runs/api-replay/events")

    assert recovered.result == inspection.result
    assert event_ids(replay.text) == ids


@pytest.mark.asyncio
async def test_inline_problem_and_structured_compilation_sources_round_trip(tmp_path: Path) -> None:
    artifacts, _problem_id, _baseline_id, problem = published_problem(tmp_path)
    evidence = await run_evidence_demo(artifacts.root)
    app = create_app(
        settings(tmp_path, api_max_concurrent_runs=2),
        backend=FakeModelBackend(),
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
    )
    inline_baseline = Plan(problem_type=problem.type_id.value, selected_site_ids=("s1",))

    async with api_client(app) as client:
        inline = await client.post(
            "/api/v1/runs",
            json={
                "run_id": "inline-problem",
                "source": {
                    "kind": "inline",
                    "problem": problem.model_dump(mode="json"),
                    "baseline_plan": inline_baseline.model_dump(mode="json"),
                },
                "budget": {"wall_time_ms": 2_000},
                "enable_model": False,
            },
        )
        compiled = await client.post(
            "/api/v1/runs",
            json={
                "run_id": "structured-compilation",
                "source": {
                    "kind": "compile_problem",
                    "arguments": {
                        "type_id": "max_weighted_coverage",
                        "demand_spec_artifact_id": evidence.demand_spec_artifact_id,
                        "candidate_spec_artifact_id": evidence.candidate_spec_artifact_id,
                        "access_matrix_artifact_id": evidence.access_matrix_artifact_id,
                        "service_matrix_artifact_ids": {
                            "normal": evidence.service_matrix_artifact_id
                        },
                        "need_field": "population",
                        "policy": {"site_limit": 1},
                    },
                },
                "budget": {"wall_time_ms": 5_000, "max_tool_calls": 1},
                "enable_model": False,
            },
        )
        assert inline.status_code == compiled.status_code == 202
        inline_result, compiled_result = await asyncio.gather(
            wait_for_result(client, "inline-problem"),
            wait_for_result(client, "structured-compilation"),
        )

    assert inline_result.result is not None and inline_result.result.best_plan is not None
    assert compiled_result.result is not None and compiled_result.result.best_plan is not None
    assert compiled_result.result.consumed_budget.tool_calls == 1
    assert inline_result.result.run_id != compiled_result.result.run_id
    assert all(
        event.run_id == "inline-problem"
        for event in app.state.run_store.read_events("inline-problem")
    )
    assert all(
        event.run_id == "structured-compilation"
        for event in app.state.run_store.read_events("structured-compilation")
    )
    assert (
        inline_result.result.runtime_plan["model_startup_ms"]
        == compiled_result.result.runtime_plan["model_startup_ms"]
    )


@pytest.mark.asyncio
async def test_advertised_ui_example_and_per_run_model_selection_round_trip(
    tmp_path: Path,
) -> None:
    app = create_app(settings(tmp_path), backend=FakeModelBackend())

    async with api_client(app) as client:
        catalog = ProblemCatalogResponse.model_validate(
            (await client.get("/api/v1/problems")).json()
        )
        runtime = RuntimeResponse.model_validate((await client.get("/api/v1/runtime")).json())
        completed: list[RunInspectionResponse] = []
        for index, example in enumerate(catalog.examples):
            run_id = f"ui-example-{index}"
            response = await client.post(
                "/api/v1/runs",
                json={
                    "run_id": run_id,
                    "source": {
                        "kind": "example",
                        "example_id": example.id,
                        "equity_template": example.default_equity_template,
                        "group_floors": example.default_group_floors,
                    },
                    "budget": {"wall_time_ms": 5_000, "max_tool_calls": 5},
                    "enable_model": False,
                    "model_profile": "gemma4_e2b_it",
                    "runtime_policy": runtime.requested_policy.model_dump(mode="json"),
                },
            )
            assert response.status_code == 202
            inspection = await wait_for_result(client, run_id)
            assert inspection.result is not None
            assert inspection.result.best_scorecard is not None
            map_response = await client.get(f"/api/v1/runs/{run_id}/map")
            assert map_response.status_code == 200, (example.id, map_response.text)
            if index == 0:
                svg_response = await client.get(f"/api/v1/runs/{run_id}/map?format=svg")
                assert svg_response.status_code == 200
                assert svg_response.headers["content-type"].startswith("image/svg+xml")
            completed.append(inspection)
        custom = catalog.examples[0]
        custom_response = await client.post(
            "/api/v1/runs",
            json={
                "run_id": "ui-custom-model",
                "source": {
                    "kind": "example",
                    "example_id": custom.id,
                    "equity_template": custom.default_equity_template,
                    "group_floors": custom.default_group_floors,
                },
                "budget": {
                    "wall_time_ms": 5_000,
                    "max_total_model_tokens": 2_000,
                    "max_generated_tokens": 256,
                    "max_tool_calls": 6,
                },
                "enable_model": True,
                "enable_deterministic_fallback": False,
                "model_profile": "gemma4_e4b_it",
                "model_id": "organization/compatible-chat-model",
            },
        )
        assert custom_response.status_code == 202
        custom_result = await wait_for_result(client, "ui-custom-model")

    assert len(catalog.examples) >= 4
    assert runtime.options.devices
    assert runtime.options.engines
    assert all(item.result is not None for item in completed)
    assert all(item.result.model_profile == "gemma4_e2b_it" for item in completed if item.result)
    assert all(item.result.model_id == "google/gemma-4-E2B-it" for item in completed if item.result)
    assert all(item.result.consumed_budget.tool_calls == 5 for item in completed if item.result)
    assert custom_result.result is not None
    assert custom_result.result.model_profile == "custom"
    assert custom_result.result.model_id == "organization/compatible-chat-model"
    assert custom_result.result.consumed_budget.model_usage.total_tokens > 0


class CancellableImproveTool:
    spec = ToolSpec(
        name="improve",
        version="1.0.0",
        description="Wait for API cancellation while preserving the committed baseline.",
        input_schema={"type": "object", "additionalProperties": True},
        output_schema={"type": "object"},
        capability_tags=frozenset({"decision", "search", "offline"}),
        problem_tags=frozenset({"location_allocation"}),
        side_effects=SideEffectClassification.NONE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=100, p95_ms=1_000, time_to_first_candidate_ms=10),
        streams_candidates=True,
        smoke_input={},
    )

    async def stream(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        del arguments, context
        await asyncio.Event().wait()
        if False:
            yield ToolEvent.model_construct()


@pytest.mark.asyncio
async def test_api_cancellation_returns_the_retained_incumbent(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    runs = LocalRunStore(tmp_path / "runs")
    app = create_app(
        settings(tmp_path),
        backend=FakeModelBackend(
            [ToolCall(id="wait", name="improve", arguments={"strategy": "add_swap"})]
        ),
        artifact_store=artifacts,
        run_store=runs,
        tool_registry=ToolRegistry((CancellableImproveTool(), RenderMapTool())),
    )

    async with api_client(app) as client:
        response = await client.post(
            "/api/v1/runs",
            json={
                **run_body(problem_id, baseline_id, "api-cancel"),
                "budget": {
                    "wall_time_ms": 2_000,
                    "max_total_model_tokens": 500,
                    "max_generated_tokens": 50,
                    "max_tool_calls": 1,
                },
                "enable_model": True,
                "enable_deterministic_fallback": False,
            },
        )
        assert response.status_code == 202
        for _ in range(200):
            kinds = (
                {event.kind for event in runs.read_events("api-cancel")}
                if runs.read_metadata("api-cancel")
                else set()
            )
            if EventKind.TOOL_STARTED in kinds:
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("tool did not start")
        live_map = await client.get("/api/v1/runs/api-cancel/map")
        cancelled = await client.post("/api/v1/runs/api-cancel/cancel")
        repeated = await client.post("/api/v1/runs/api-cancel/cancel")

    assert cancelled.status_code == 200
    assert live_map.status_code == 200
    assert json.loads(live_map.content)["type"] == "FeatureCollection"
    result = cancelled.json()["result"]
    assert result["status"] == "cancelled"
    assert result["terminal_reason"] == "user_cancelled"
    assert result["best_plan"]["selected_site_ids"] == ["s1"]
    assert repeated.json()["already_finalized"] is True
    assert repeated.json()["result"] == result
    assert EventKind.TOOL_CANCELLED in {event.kind for event in runs.read_events("api-cancel")}


@pytest.mark.asyncio
async def test_slow_subscriber_does_not_block_run_and_capacity_rejects_cleanly(
    tmp_path: Path,
) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    app = create_app(
        settings(tmp_path, api_max_concurrent_runs=1),
        backend=FakeModelBackend(
            [ToolCall(id="wait", name="improve", arguments={"strategy": "add_swap"})]
        ),
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
        tool_registry=ToolRegistry((CancellableImproveTool(),)),
    )

    async with api_client(app) as client:
        first = await client.post(
            "/api/v1/runs",
            json={
                **run_body(problem_id, baseline_id, "capacity-one"),
                "budget": {
                    "wall_time_ms": 2_000,
                    "max_total_model_tokens": 500,
                    "max_generated_tokens": 50,
                    "max_tool_calls": 1,
                },
                "enable_model": True,
                "enable_deterministic_fallback": False,
            },
        )
        assert first.status_code == 202
        queue = await app.state.run_manager.subscribe("capacity-one")
        second = await client.post(
            "/api/v1/runs", json=run_body(problem_id, baseline_id, "capacity-two")
        )
        assert second.status_code == 503
        assert second.json()["error"]["code"] == "run_capacity_unavailable"
        cancelled = await client.post("/api/v1/runs/capacity-one/cancel")
        assert cancelled.status_code == 200
        assert queue.qsize() == 1
        await app.state.run_manager.unsubscribe("capacity-one", queue)


@pytest.mark.asyncio
async def test_artifact_and_request_limits_and_invalid_ids_are_structured(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    oversized = artifacts.put_bytes(
        b"0123456789",
        ArtifactMetadata(
            kind=ArtifactKind.MAP,
            media_type="application/geo+json",
            privacy="public",
        ),
    )
    disallowed = artifacts.put_bytes(
        b'{"trace":true}',
        ArtifactMetadata(
            kind=ArtifactKind.TRACE_ATTACHMENT,
            media_type="application/json",
            privacy="public",
        ),
    )
    private = artifacts.put_bytes(
        b'{"type":"FeatureCollection","features":[]}',
        ArtifactMetadata(
            kind=ArtifactKind.MAP,
            media_type="application/geo+json",
            privacy="internal",
        ),
    )
    app = create_app(
        settings(
            tmp_path,
            api_max_artifact_response_bytes=5,
            api_max_request_bytes=64,
        ),
        backend=FakeModelBackend(),
        artifact_store=artifacts,
    )

    async with api_client(app) as client:
        invalid = await client.get("/api/v1/artifacts/not-an-artifact")
        traversal = await client.get("/api/v1/artifacts/%2E%2E%2Fsecret")
        too_large = await client.get(f"/api/v1/artifacts/{oversized.id}")
        blocked_type = await client.get(f"/api/v1/artifacts/{disallowed.id}")
        blocked_privacy = await client.get(f"/api/v1/artifacts/{private.id}")
        bad_run = await client.get("/api/v1/runs/not-present")
        large_request = await client.post(
            "/api/v1/chat",
            content=b"x" * 65,
            headers={"content-type": "application/json"},
        )

    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_artifact_id"
    assert traversal.status_code == 404
    assert "traceback" not in traversal.text.lower()
    assert too_large.status_code == 413
    assert blocked_type.status_code == 415
    assert blocked_privacy.status_code == 403
    assert bad_run.status_code == 404
    assert large_request.status_code == 413
    assert all(
        "traceback" not in response.text.lower() for response in (invalid, too_large, bad_run)
    )


@pytest.mark.asyncio
async def test_optional_static_ui_mount_does_not_change_api_routes(tmp_path: Path) -> None:
    ui_root = tmp_path / "ui"
    ui_root.mkdir()
    (ui_root / "index.html").write_text("<!doctype html><title>future OASIS UI</title>")
    app = create_app(settings(tmp_path, serve_ui=True, ui_root=ui_root), backend=FakeModelBackend())

    async with api_client(app) as client:
        page = await client.get("/")
        health = await client.get("/api/v1/health")

    assert page.status_code == 200
    assert "future OASIS UI" in page.text
    assert health.status_code == 200
