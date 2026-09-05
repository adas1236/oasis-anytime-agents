from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from integration.test_api import api_client, settings, wait_for_result
from oasis.api import create_app
from oasis.artifacts import read_json
from oasis.llm import FakeModelBackend, ToolCall
from oasis.llm.schemas import ChatRole, ModelRequest, ModelTurn
from oasis.mock_experiments import DatasetKind, load_dataset
from oasis.prompts import AGENT_SYSTEM_PROMPT
from oasis.providers.cache import MemorySnapshotCache
from oasis.providers.mock_dataset import DatasetEvidenceProvider
from oasis.registry_experiments import RegistrySmokeBackend
from oasis.schemas import ToolEvent, ToolEventKind
from oasis.tools import ToolRegistry, create_public_tool_registry
from oasis.tools.decision.common import read_plan
from oasis.tools.public import CompactImproveTool


class RecordingBackend(FakeModelBackend):
    def __init__(self, responses=()):
        super().__init__(responses)
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        return await super().generate(request)


@pytest.mark.asyncio
async def test_message_only_ask_answers_without_any_problem(tmp_path: Path) -> None:
    backend = RecordingBackend(["Hello. How can I help?"])
    app = create_app(settings(tmp_path), backend=backend, providers={})
    async with api_client(app) as client:
        response = await client.post("/api/v1/ask", json={"message": "Hello"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["answer"] == "Hello. How can I help?"
        assert result["status"] == "complete"
        assert result["answer_source"] == "model"
        assert result["problem_artifact_id"] is None
        assert result["best_plan"] is None
        assert result["incumbent_timeline"] == []
        assert result["consumed_budget"]["tool_calls"] == 0
        assert len(backend.requests[0].messages) == 2
        assert {tool["name"] for tool in (await client.get("/api/v1/tools")).json()["tools"]} == {
            spec.name for spec in create_public_tool_registry(discover_entry_points=False).list()
        }
        assert backend.requests[0].messages[0].content == AGENT_SYSTEM_PROMPT
        assert backend.requests[0].messages[1].content == "Hello"
        assert {
            "compile_max_coverage",
            "compile_tsp",
            "improve",
            "inspect_artifact",
            "calculator",
        } <= {tool.name for tool in backend.requests[0].tools}
        replay = await client.get(f"/api/v1/runs/{result['run_id']}")
        assert replay.json()["result"]["answer"] == result["answer"]


@pytest.mark.asyncio
async def test_run_message_tool_results_feed_back_and_sse_replays(tmp_path: Path) -> None:
    backend = RecordingBackend(
        [
            ToolCall(
                id="c1", name="calculator", arguments={"operation": "multiply", "operands": [6, 7]}
            ),
            "6 times 7 is 42.",
        ]
    )
    app = create_app(settings(tmp_path), backend=backend, providers={})
    async with api_client(app) as client:
        created = await client.post("/api/v1/runs", json={"message": "What is 6 times 7?"})
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        inspection = await wait_for_result(client, run_id)
        assert inspection.result is not None
        assert inspection.result.answer == "6 times 7 is 42."
        assert inspection.result.consumed_budget.tool_calls == 1
        observation = backend.requests[1].messages[-1]
        assert observation.role is ChatRole.TOOL
        assert json.loads(observation.content)["metrics"]["value"] == 42
        events = (await client.get(f"/api/v1/runs/{run_id}/events")).text
        assert "tool_started" in events
        assert "tool_completed" in events
        assert "run_finalized" in events
        # The new run survives a fresh service instance using the same stores.
    restarted = create_app(settings(tmp_path), backend=FakeModelBackend(), providers={})
    async with api_client(restarted) as client:
        inspection = await client.get(f"/api/v1/runs/{run_id}")
        assert inspection.json()["result"]["answer"] == "6 times 7 is 42."


@pytest.mark.asyncio
async def test_message_to_evidence_compilation_and_improvement(tmp_path: Path) -> None:
    case = load_dataset(DatasetKind.MAX_COVERAGE, Path("data/max_coverage.json"))[0]
    provider = DatasetEvidenceProvider(case.locations, case.region)
    app = create_app(
        settings(tmp_path),
        backend=RegistrySmokeBackend(),
        providers={
            "place_resolution": provider,
            "catalog_search": provider,
            "source_snapshot": provider,
        },
        resources={"snapshot_cache": MemorySnapshotCache()},
    )
    async with api_client(app) as client:
        response = await client.post("/api/v1/ask", json={"message": case.prompt})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "complete", result
        assert result["best_scorecard"]["feasible"]
        assert result["best_plan"]["problem_type"] == "max_weighted_coverage"
        assert result["consumed_budget"]["tool_calls"] == 9
        transcript = read_json(app.state.artifact_store, result["conversation_artifact_id"])
        assert len([m for m in transcript if m["role"] == "user"]) == 1
        assert transcript[1]["content"] == case.prompt
        calls = [c for m in transcript for c in m.get("tool_calls", [])]
        assert calls[0]["name"] == "resolve_locations"
        compilation = next(c for c in calls if c["name"] == "compile_max_coverage")
        assert compilation["arguments"]["site_limit"] == case.centers_to_place
        assert any(c["name"] == "improve" for c in calls)
        events = app.state.run_store.read_events(result["run_id"])
        first_plan = next(e for e in events if e.kind == "incumbent_committed")
        assert first_plan.budget_after.tool_calls >= 7
        assert first_plan.payload["problem_artifact_id"] == result["problem_artifact_id"]
        mapped = await client.get(f"/api/v1/runs/{result['run_id']}/map")
        assert mapped.status_code == 200


@pytest.mark.asyncio
async def test_empty_budget_has_no_injected_answer_or_baseline(tmp_path: Path) -> None:
    backend = RecordingBackend()
    app = create_app(settings(tmp_path, agent_total_tokens=0), backend=backend, providers={})
    async with api_client(app) as client:
        response = await client.post(
            "/api/v1/ask", json={"message": "Choose the best clinic sites."}
        )
        result = response.json()
        assert result["status"] == "partial"
        assert result["answer_source"] == "status"
        assert result["best_plan"] is None
        assert result["problem_artifact_id"] is None
        assert backend.requests == []


@pytest.mark.asyncio
async def test_tool_limit_returns_the_agent_compiled_plan(tmp_path: Path) -> None:
    case = load_dataset(DatasetKind.MAX_COVERAGE, Path("data/max_coverage.json"))[0]
    provider = DatasetEvidenceProvider(case.locations, case.region)
    app = create_app(
        settings(tmp_path, agent_tool_calls=7),
        backend=RegistrySmokeBackend(),
        providers={"place_resolution": provider},
    )
    async with api_client(app) as client:
        response = await client.post("/api/v1/ask", json={"message": case.prompt})
        result = response.json()
        assert result["terminal_reason"] == "tool_call_limit", result
        assert result["answer_source"] == "plan"
        assert result["best_scorecard"]["feasible"]
        assert result["consumed_budget"]["tool_calls"] == 7
        assert "Selected sites:" in result["answer"]
        assert any(place.name in result["answer"] for place in case.locations)


class WaitingImproveTool(CompactImproveTool):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def stream(self, arguments, context):
        _, plan = read_plan(context, arguments["resume_from"])
        yield ToolEvent(sequence=0, kind=ToolEventKind.CANDIDATE, candidate=plan)
        self.started.set()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_cancel_streaming_tool_preserves_checked_plan_and_map(tmp_path: Path) -> None:
    case = load_dataset(DatasetKind.MAX_COVERAGE, Path("data/max_coverage.json"))[0]
    provider = DatasetEvidenceProvider(case.locations, case.region)
    waiting = WaitingImproveTool()
    base = create_public_tool_registry(discover_entry_points=False)
    registry = ToolRegistry(
        [waiting if s.name == "improve" else base.get(s.name) for s in base.list()]
    )
    app = create_app(
        settings(tmp_path),
        backend=RegistrySmokeBackend(),
        providers={"place_resolution": provider},
        tool_registry=registry,
    )
    async with api_client(app) as client:
        created = await client.post("/api/v1/runs", json={"message": case.prompt})
        run_id = created.json()["run_id"]
        await asyncio.wait_for(waiting.started.wait(), 5)
        assert (await client.get(f"/api/v1/runs/{run_id}/map")).status_code == 200
        await client.post(f"/api/v1/runs/{run_id}/cancel")
        inspection = await wait_for_result(client, run_id)
        assert inspection.result.status == "cancelled"
        assert inspection.result.answer_source == "plan"
        assert inspection.result.best_scorecard.feasible
        assert (await client.get(f"/api/v1/runs/{run_id}/map")).status_code == 200


@pytest.mark.asyncio
async def test_aggregate_token_budget_limits_generation(tmp_path: Path) -> None:
    class FixedCounterBackend(RecordingBackend):
        async def count_input_tokens(self, request: ModelRequest) -> int:
            return 99

    backend = FixedCounterBackend(["One two three four five."])
    app = create_app(settings(tmp_path, agent_total_tokens=100), backend=backend, providers={})
    async with api_client(app) as client:
        await client.post("/api/v1/ask", json={"message": "Explain."})
        assert backend.requests[0].max_generated_tokens == 1


class WaitingBackend(RecordingBackend):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.aborted: list[str] = []

    async def generate(self, request: ModelRequest) -> ModelTurn:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        await super().abort(request_id)


@pytest.mark.asyncio
async def test_cancel_during_generation_finalizes_without_a_plan(tmp_path: Path) -> None:
    backend = WaitingBackend()
    app = create_app(settings(tmp_path), backend=backend, providers={})
    async with api_client(app) as client:
        created = await client.post("/api/v1/runs", json={"message": "Help plan a visit."})
        run_id = created.json()["run_id"]
        await asyncio.wait_for(backend.started.wait(), 2)
        response = await client.post(f"/api/v1/runs/{run_id}/cancel")
        assert response.status_code == 200
        inspection = await wait_for_result(client, run_id)
        assert inspection.result.status == "cancelled"
        assert inspection.result.terminal_reason == "user_cancelled"
        assert inspection.result.best_plan is None
        assert backend.aborted
        assert not inspection.result.usage_complete


@pytest.mark.asyncio
async def test_model_timeout_is_persisted(tmp_path: Path) -> None:
    backend = WaitingBackend()
    app = create_app(
        settings(tmp_path, agent_model_timeout_seconds=0.02), backend=backend, providers={}
    )
    async with api_client(app) as client:
        response = await client.post("/api/v1/ask", json={"message": "Help."})
        assert response.json()["terminal_reason"] == "model_call_timeout"
        assert response.json()["status"] == "partial"
        assert backend.aborted


@pytest.mark.asyncio
async def test_tool_error_is_visible_to_model_and_recoverable(tmp_path: Path) -> None:
    backend = RecordingBackend(
        [
            ToolCall(
                id="bad", name="calculator", arguments={"operation": "divide", "operands": [4, 0]}
            ),
            "Division by zero is undefined.",
        ]
    )
    app = create_app(settings(tmp_path), backend=backend, providers={})
    async with api_client(app) as client:
        response = await client.post("/api/v1/ask", json={"message": "Divide 4 by zero."})
        assert response.json()["status"] == "complete"
        assert json.loads(backend.requests[1].messages[-1].content)["error"]


@pytest.mark.asyncio
async def test_wall_budget_cancels_a_running_model(tmp_path: Path) -> None:
    backend = WaitingBackend()
    app = create_app(settings(tmp_path, agent_wall_time_ms=200), backend=backend, providers={})
    async with api_client(app) as client:
        response = await client.post("/api/v1/ask", json={"message": "Help."})
        assert response.json()["terminal_reason"] == "time_exhausted"
        assert response.json()["answer_source"] == "status"
        assert backend.aborted


@pytest.mark.asyncio
async def test_model_can_revise_formulation_without_comparing_unrelated_objectives(tmp_path: Path):
    class RevisingBackend(RegistrySmokeBackend):
        def _next_response(self, request):
            compiled = [
                c
                for m in request.messages
                for c in m.tool_calls
                if c.name == "compile_max_coverage"
            ]
            if len(compiled) == 1:
                arguments = dict(compiled[0].arguments)
                arguments["site_limit"] = 1
                return ToolCall(id="revised", name="compile_max_coverage", arguments=arguments)
            if len(compiled) == 2:
                return "I have revised the plan to use one site."
            return super()._next_response(request)

    case = load_dataset(DatasetKind.MAX_COVERAGE, Path("data/max_coverage.json"))[0]
    provider = DatasetEvidenceProvider(case.locations, case.region)
    app = create_app(
        settings(tmp_path), backend=RevisingBackend(), providers={"place_resolution": provider}
    )
    async with api_client(app) as client:
        result = (await client.post("/api/v1/ask", json={"message": case.prompt})).json()
        assert result["status"] == "complete"
        assert len(result["best_plan"]["selected_site_ids"]) == 1
        problem = read_json(app.state.artifact_store, result["problem_artifact_id"])
        assert problem["policy"]["site_limit"] == 1
        hashes = {p["problem_hash"] for p in result["incumbent_timeline"]}
        assert hashes == {problem["problem_hash"]}
        events = app.state.run_store.read_events(result["run_id"])
        assert (
            len({e.payload["problem_hash"] for e in events if e.kind == "incumbent_committed"}) == 2
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"message": "  "}, {"message": "hi", "type_id": "tsp"}])
async def test_ask_accepts_only_a_nonempty_message(tmp_path: Path, payload) -> None:
    app = create_app(settings(tmp_path), backend=FakeModelBackend(), providers={})
    async with api_client(app) as client:
        assert (await client.post("/api/v1/ask", json=payload)).status_code == 422
