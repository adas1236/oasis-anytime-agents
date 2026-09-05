from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import pytest

from oasis.artifacts import read_vector
from oasis.llm.fake import FakeModelBackend
from oasis.llm.schemas import ChatRole, ToolCall
from oasis.mock_experiments import (
    AgentRun,
    BudgetPoint,
    DatasetKind,
    OsrmMatrixStore,
    _BudgetClock,
    _build_region_catalog,
    _objective_score,
    _run_agent_case,
    _score,
    load_dataset,
    run_experiment,
)
from oasis.providers.mock_dataset import DatasetEvidenceProvider, DatasetRoutingProvider
from oasis.providers.models import (
    PlaceResolveRequest,
    ProviderError,
    ProviderRequestContext,
    RouteAnnotation,
    RouteMatrixRequest,
)
from oasis.registry_experiments import RegistrySession, RegistrySmokeBackend, system_prompt
from oasis.schemas import Plan
from oasis.tools import CancellationToken, create_tool_registry
from unit.test_mock_experiments import DATA_ROOT, _config


def config(tmp_path: Path):
    result = _config(tmp_path)
    result.tool_mode = "registry"
    result.max_tool_rounds = 20
    result.max_generated_tokens = 768
    result.case_timeout_seconds = None
    result.osrm_cache = DATA_ROOT.parent / "infra/runpod/osrm-cache"
    return result


def roads() -> OsrmMatrixStore:
    return OsrmMatrixStore(
        endpoint="https://router.project-osrm.org",
        cache_dir=DATA_ROOT.parent / "infra/runpod/osrm-cache",
        cache_only=True,
        timeout_seconds=5,
        region_locations=_build_region_catalog(
            load_dataset(DatasetKind.TSP, DATA_ROOT / "tsp.json")
        ),
    )


def provider_context() -> ProviderRequestContext:
    return ProviderRequestContext(
        deadline_monotonic=math.inf,
        cancellation=CancellationToken(),
        monotonic=time.monotonic,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset", list(DatasetKind))
async def test_real_registry_end_to_end_and_prompt_only_model_input(tmp_path: Path, dataset):
    case = load_dataset(dataset, DATA_ROOT / f"{dataset.value}.json")[0]
    backend = RegistrySmokeBackend()
    result = await _run_agent_case(case, backend, config(tmp_path), roads())
    assert result.error is None
    assert result.agent_plan_found
    assert result.terminal_reason == "completed"
    assert _score(case, result.prediction, 1)
    assert result.protocol == "live_registry_v1"
    assert result.tool_names == [
        d.name for d in create_tool_registry(discover_entry_points=False).model_definitions()
    ]
    assert "solve_current_problem" not in result.tool_names
    assert "search_locations" not in result.tool_names
    trace = [
        json.loads(line)
        for line in (Path(result.artifacts_directory) / "trace.jsonl").read_text().splitlines()
    ]
    assert trace[0]["messages"][0]["content"] == system_prompt()
    assert trace[0]["messages"][1]["content"] == case.prompt
    assert len(trace[0]["messages"]) == 2
    assert "expected" not in trace[0] and "locations" not in trace[0]
    assert any(event["event"] == "incumbent" for event in trace)
    assert any(event["event"] == "tool_event" for event in trace)
    assert result.initial_input_tokens > 0
    assert result.calls[0]["arguments"]["queries"] == [p.name for p in case.locations]


@pytest.mark.asyncio
async def test_lookup_requires_query_and_preserves_explicit_selection(tmp_path: Path):
    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    session = RegistrySession(case, roads(), tmp_path, 42, lambda event: None)
    run, clock = AgentRun(), _BudgetClock(BudgetPoint("u", None, None, None))
    unknown = await session.call(
        ToolCall(
            id="1",
            name="resolve_locations",
            arguments={"queries": ["definitely not a real location xyzzzz"]},
        ),
        run,
        clock,
        1,
        {},
    )
    assert unknown.metrics["result_count"] == 0
    resolved = await session.call(
        ToolCall(
            id="2",
            name="resolve_locations",
            arguments={"queries": [case.locations[0].name, case.locations[1].name]},
        ),
        run,
        clock,
        1,
        {},
    )
    resolution = resolved.metrics["resolution_artifact_id"]
    ids = [case.locations[1].location_id, case.locations[0].location_id]
    materialized = await session.call(
        ToolCall(
            id="3",
            name="materialize_locations",
            arguments={
                "resolution_artifact_id": resolution,
                "provider_ids": ids,
                "metadata_fields": ["population"],
            },
        ),
        run,
        clock,
        1,
        {},
    )
    assert materialized.error is None
    frame = read_vector(session.store, str(materialized.metrics["artifact_id"]))
    assert list(frame["id"]) == ids
    assert list(frame["population"]) == [case.locations[i].population for i in (1, 0)]
    for forbidden in ([case.locations[2].location_id], [ids[0], ids[0]]):
        result = await session.call(
            ToolCall(
                id="4",
                name="materialize_locations",
                arguments={"resolution_artifact_id": resolution, "provider_ids": forbidden},
            ),
            run,
            clock,
            1,
            {},
        )
        assert result.error is not None


@pytest.mark.asyncio
async def test_country_filter_and_directed_distance_submatrix():
    case = next(
        c
        for c in load_dataset(DatasetKind.TSP, DATA_ROOT / "tsp.json")
        if c.region == "Tokyo, Japan"
    )
    source = DatasetEvidenceProvider(case.locations, case.region)
    resolved = await source.resolve(
        PlaceResolveRequest(query=case.locations[0].name, country_codes=("jp",)), provider_context()
    )
    assert resolved.candidates[0].provider_id == case.locations[0].location_id
    store = roads()
    master = store.region_locations[case.region]
    matrix = await store.region_matrix(case.region)
    provider = DatasetRoutingProvider(store, case.region)
    request = RouteMatrixRequest(
        coordinates=tuple((master[i].longitude, master[i].latitude) for i in (2, 0, 1)),
        source_ids=("a", "b"),
        destination_ids=("c", "a"),
        source_indices=(0, 1),
        destination_indices=(2, 0),
        profile="driving",
        annotation=RouteAnnotation.DISTANCE,
    )
    result = await provider.matrix(request, provider_context())
    assert result.values == ((matrix[2][1], matrix[2][2]), (matrix[0][1], matrix[0][2]))
    assert result.units == "meters"
    with pytest.raises(ProviderError, match="not durations"):
        await provider.matrix(
            request.model_copy(update={"annotation": RouteAnnotation.DURATION}), provider_context()
        )


@pytest.mark.asyncio
async def test_grader_rejects_wrong_problem_depot_and_missing_locations(tmp_path: Path):
    case = load_dataset(DatasetKind.TSP, DATA_ROOT / "tsp.json")[0]
    session = RegistrySession(case, roads(), tmp_path, 42, lambda event: None)
    ids = [p.location_id for p in case.locations]
    for tour in ([ids[0], ids[1], ids[0]], [*ids[1:], ids[0], ids[1]]):
        with pytest.raises(ValueError):
            await session.prediction(Plan(problem_type="tsp", routes=({"node_ids": tour},)))
    with pytest.raises(ValueError, match="exactly one tour"):
        await session.prediction(Plan(problem_type="max_weighted_coverage"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,reason",
    [
        (BudgetPoint("zero-time", 0, None, None), "time_budget_exhausted"),
        (BudgetPoint("zero-tokens", None, 0, None), "token_budget_exhausted"),
        (BudgetPoint("tiny-tokens", None, 1, None), "token_budget_exhausted"),
    ],
)
async def test_limits_stop_before_model_or_tools(tmp_path: Path, budget, reason):
    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    result = await _run_agent_case(case, RegistrySmokeBackend(), config(tmp_path), roads(), budget)
    assert result.terminal_reason == reason
    assert result.generations == result.tool_calls_used == 0
    assert result.prediction_source == "baseline"
    assert not result.agent_plan_found


@pytest.mark.asyncio
async def test_tool_failure_and_later_model_failure_preserve_history(tmp_path: Path):
    class BrokenBackend(FakeModelBackend):
        async def generate(self, request):
            if any(message.role is ChatRole.TOOL for message in request.messages):
                raise RuntimeError("deliberate backend crash")
            return await super().generate(request)

    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    backend = BrokenBackend(responses=[ToolCall(id="bad", name="resolve_locations", arguments={})])
    result = await _run_agent_case(case, backend, config(tmp_path), roads())
    assert result.terminal_reason == "model_runtime_error"
    assert len(result.calls) == 1
    assert [failure["category"] for failure in result.failures] == [
        "invalid_arguments",
        "model_runtime_error",
    ]
    assert result.generations == 1
    assert not result.usage_complete
    assert result.prediction is not None


@pytest.mark.asyncio
async def test_grid_resume_and_protocol_separation(tmp_path: Path):
    options = config(tmp_path)
    options.time_budgets_seconds = (0, None)
    options.token_budgets = (0, None)
    first = await run_experiment(options)
    assert first["completed_cells"] == 4
    options.resume = True
    second = await run_experiment(options)
    assert second["resumed_cells"] == 4
    options.tool_mode = "legacy"
    with pytest.raises(ValueError, match="configuration differs"):
        await run_experiment(options)


@pytest.mark.asyncio
async def test_model_timeout_is_separate_from_global_deadline(tmp_path: Path):
    class SlowBackend(FakeModelBackend):
        async def generate(self, request):
            await asyncio.sleep(1)
            return await super().generate(request)

    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    options = config(tmp_path)
    options.case_timeout_seconds = 0.01
    result = await _run_agent_case(case, SlowBackend(), options, roads())
    assert result.terminal_reason == "model_call_timeout"
    assert not result.usage_complete


@pytest.mark.asyncio
async def test_catalog_snapshot_and_inspection_are_real_registry_tools(tmp_path: Path):
    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    session = RegistrySession(case, roads(), tmp_path, 42, lambda event: None)
    run, clock = AgentRun(), _BudgetClock(BudgetPoint("u", None, None, None))
    catalog = await session.call(
        ToolCall(
            id="catalog",
            name="search_sources",
            arguments={
                "query": {"names": [case.locations[0].name]},
            },
        ),
        run,
        clock,
        1,
        {},
    )
    assert catalog.error is None
    from oasis.artifacts import read_json

    payload = read_json(session.store, catalog.artifacts[0])
    url = payload["items"][0]["assets"][0]["href"]
    snapshot = await session.call(
        ToolCall(
            id="snapshot",
            name="snapshot_source",
            arguments={
                "url": url,
                "format": "geojson",
                "units": "location_attributes",
                "license": "mock example; no license supplied",
            },
        ),
        run,
        clock,
        1,
        {},
    )
    assert snapshot.error is None
    inspected = await session.call(
        ToolCall(
            id="inspect",
            name="inspect_artifact",
            arguments={
                "artifact_id": snapshot.metrics["artifact_id"],
                "limit": 1,
            },
        ),
        run,
        clock,
        1,
        {},
    )
    assert inspected.error is None
    assert inspected.metrics["value"][0]["population"] == case.locations[0].population
    assert inspected.metrics["total_items"] == 1
    forbidden = await session.call(
        ToolCall(
            id="forbidden",
            name="snapshot_source",
            arguments={
                "url": "https://example.com/not-in-the-dataset.geojson",
                "format": "geojson",
                "units": "location_attributes",
                "license": "mock example; no license supplied",
            },
        ),
        run,
        clock,
        1,
        {},
    )
    assert forbidden.error is not None


@pytest.mark.asyncio
async def test_optimal_coverage_with_fewer_centers_preserves_both_scores(tmp_path: Path):
    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[776]
    result = await _run_agent_case(case, RegistrySmokeBackend(), config(tmp_path), roads())
    assert result.agent_plan_found
    assert len(result.prediction["center_locations"]) < case.centers_to_place
    assert not _score(case, result.prediction, 1)
    assert _objective_score(case, result.prediction, 1)


@pytest.mark.asyncio
async def test_retry_preserves_previous_attempt_trace(tmp_path: Path):
    case = load_dataset(DatasetKind.MAX_COVERAGE, DATA_ROOT / "max_coverage.json")[0]
    budget = BudgetPoint("zero", 0, None, None)
    first = await _run_agent_case(case, RegistrySmokeBackend(), config(tmp_path), roads(), budget)
    trace = Path(first.artifacts_directory) / "trace.jsonl"
    original = trace.read_bytes()
    second = await _run_agent_case(case, RegistrySmokeBackend(), config(tmp_path), roads(), budget)
    assert first.artifacts_directory != second.artifacts_directory
    assert trace.read_bytes() == original
    events = [json.loads(line) for line in original.splitlines()]
    assert events[1]["event"] == "incumbent"
    assert events[1]["source"] == "baseline"
