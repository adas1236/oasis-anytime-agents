from __future__ import annotations

import math
import time
from pathlib import Path

import pytest
from unit.test_routing import route_fixture, route_provenance

from oasis.artifacts import ArtifactStore, LocalArtifactStore, put_json, read_json
from oasis.problems import (
    Comparison,
    Deadline,
    RouteProblemType,
    RouteServicePolicy,
    RouteServiceProblem,
    Scorecard,
    create_builtin_problem_registry,
)
from oasis.routing import run_routing_demo
from oasis.schemas import ArtifactKind, Plan, ToolEventKind, ToolResultStatus
from oasis.tools import (
    CancellationToken,
    ToolContext,
    create_tool_registry,
    invoke_tool,
    stream_tool,
)


def context(store: ArtifactStore) -> ToolContext:
    return ToolContext(
        run_id="routing-integration",
        artifact_store=store,
        deadline_monotonic=time.monotonic() + 10,
        cancellation=CancellationToken(),
        seed=23,
    )


def publish_route_problem(
    tmp_path: Path,
    *,
    type_id: RouteProblemType = RouteProblemType.MOBILE_SERVICE,
) -> tuple[LocalArtifactStore, str, str, RouteServiceProblem]:
    policy = RouteServicePolicy(
        depot_ids=("depot",),
        shift_length=100,
        time_units="minutes",
        vehicle_capacity=8 if type_id is RouteProblemType.MOBILE_SERVICE else None,
        capacity_units="people" if type_id is RouteProblemType.MOBILE_SERVICE else None,
    )
    store, problem = route_fixture(tmp_path, type_id, policy)
    problem_ref = put_json(
        store,
        problem.model_dump(mode="json"),
        kind=ArtifactKind.JSON_SPECIFICATION,
        units="unitless",
        provenance=route_provenance("compiled-problem"),
        data_schema={"type": "RouteServiceProblem", "version": "1.0.0"},
    )
    depot_route = Plan(
        problem_type=problem.type_id.value,
        routes=({"vehicle_id": "vehicle-1", "node_ids": ["depot", "depot"]},),
    )
    if type_id is RouteProblemType.TSP:
        plugin = create_builtin_problem_registry().get(problem.type_id.value)
        depot_route = plugin.make_baseline(
            problem,
            store,
            Deadline(time.monotonic() + 5),
        )
    plan_ref = put_json(
        store,
        depot_route.model_dump(mode="json"),
        kind=ArtifactKind.PLAN,
        units="unitless",
        provenance=route_provenance("starting-plan"),
        data_schema={"type": "Plan", "version": "1.0.0"},
    )
    return store, problem_ref.id, plan_ref.id, problem


@pytest.mark.asyncio
async def test_mobile_vaccination_demo_improves_and_independently_scores(tmp_path: Path) -> None:
    result = await run_routing_demo(tmp_path)
    store = LocalArtifactStore(tmp_path)
    problem = RouteServiceProblem.model_validate(read_json(store, result.problem_artifact_id))
    baseline = Scorecard.model_validate(read_json(store, result.baseline_scorecard_artifact_id))
    best = Scorecard.model_validate(read_json(store, result.best_scorecard_artifact_id))
    plugin = create_builtin_problem_registry().get(problem.type_id.value)

    assert best.feasible
    assert result.overall_metrics["served_value"] == 8.0
    assert plugin.compare(best, baseline) is Comparison.BETTER
    assert result.scenario_metrics["normal"]["coverage"] == pytest.approx(8 / 15)


@pytest.mark.asyncio
async def test_unlimited_ortools_and_model_visible_resume_artifact(tmp_path: Path) -> None:
    store, problem_id, plan_id, _ = publish_route_problem(tmp_path, type_id=RouteProblemType.TSP)
    registry = create_tool_registry(discover_entry_points=False)
    unlimited = ToolContext(
        run_id="unlimited-routing",
        artifact_store=store,
        deadline_monotonic=math.inf,
        cancellation=CancellationToken(),
        seed=42,
    )
    solved = await invoke_tool(
        registry.get("improve"),
        {
            "problem_artifact_id": problem_id,
            "strategy": "ortools_routing",
        },
        unlimited,
    )
    assert solved.error is None
    assert solved.candidate is not None
    partial = await invoke_tool(
        registry.get("improve"),
        {
            "problem_artifact_id": problem_id,
            "starting_plan_artifact_id": plan_id,
            "strategy": "exact_enumeration",
            "max_candidates": 1,
        },
        unlimited,
    )
    assert partial.error is None
    token_id = partial.metrics["resume_token_artifact_id"]
    assert isinstance(token_id, str) and token_id in partial.model_summary()
    resumed = await invoke_tool(
        registry.get("improve"),
        {
            "problem_artifact_id": problem_id,
            "strategy": "exact_enumeration",
            "resume_token_artifact_id": token_id,
            "max_candidates": 1000,
        },
        unlimited,
    )
    assert resumed.error is None
    assert resumed.metrics["complete"]


@pytest.mark.asyncio
async def test_routing_search_cancels_safely_and_resumes_deterministically(tmp_path: Path) -> None:
    store, problem_id, plan_id, problem = publish_route_problem(tmp_path)
    registry = create_tool_registry(discover_entry_points=False)
    first = [
        event
        async for event in stream_tool(
            registry.get("improve"),
            {
                "problem_artifact_id": problem_id,
                "starting_plan_artifact_id": plan_id,
                "strategy": "exact_enumeration",
                "max_candidates": 1,
            },
            context(store),
        )
    ]
    partial = first[-1].result
    assert partial is not None and partial.status is ToolResultStatus.PARTIAL
    assert partial.resume_token is not None
    resumed = [
        event
        async for event in stream_tool(
            registry.get("improve"),
            {
                "problem_artifact_id": problem_id,
                "strategy": "exact_enumeration",
                "max_candidates": 100,
                "resume_token": partial.resume_token,
            },
            context(store),
        )
    ]
    assert resumed[-1].result is not None
    assert resumed[-1].result.status is ToolResultStatus.COMPLETE

    cancellation_context = context(store)
    retained = None
    terminal = None
    async for event in stream_tool(
        registry.get("improve"),
        {
            "problem_artifact_id": problem_id,
            "starting_plan_artifact_id": plan_id,
            "strategy": "exact_enumeration",
            "max_candidates": 100,
        },
        cancellation_context,
    ):
        if event.kind is ToolEventKind.CANDIDATE:
            retained = event.candidate
            cancellation_context.cancellation.cancel("retain first route")
        elif event.kind is ToolEventKind.RESULT:
            terminal = event.result
    assert retained is not None
    assert terminal is not None and terminal.status is ToolResultStatus.EXPIRED
    plugin = create_builtin_problem_registry().get(problem.type_id.value)
    assert plugin.validate_plan(problem, retained, store).valid


@pytest.mark.asyncio
async def test_scenario_sweep_isolates_route_policy_variants(tmp_path: Path) -> None:
    store, problem_id, _, problem = publish_route_problem(tmp_path)
    registry = create_tool_registry(discover_entry_points=False)
    result = await invoke_tool(
        registry.get("scenario_sweep"),
        {
            "problem_artifact_id": problem_id,
            "variants": [
                {
                    "name": "longer_shift",
                    "policy": {
                        **problem.policy.model_dump(mode="json"),
                        "shift_length": 120,
                    },
                },
                {
                    "name": "larger_vehicle",
                    "policy": {
                        **problem.policy.model_dump(mode="json"),
                        "vehicle_capacity": 10,
                    },
                },
            ],
        },
        context(store),
    )

    assert result.status is ToolResultStatus.COMPLETE
    records = result.metrics["variants"]
    hashes = {record["problem_hash"] for record in records}
    assert len(hashes) == 2
    assert problem.problem_hash not in hashes
    for record in records:
        variant = RouteServiceProblem.model_validate(
            read_json(store, str(record["problem_artifact_id"]))
        )
        assert variant.problem_hash == record["problem_hash"]


@pytest.mark.asyncio
async def test_ortools_routing_candidate_is_independently_validated(tmp_path: Path) -> None:
    store, problem_id, plan_id, problem = publish_route_problem(
        tmp_path, type_id=RouteProblemType.TSP
    )
    events = [
        event
        async for event in stream_tool(
            create_tool_registry(discover_entry_points=False).get("improve"),
            {
                "problem_artifact_id": problem_id,
                "starting_plan_artifact_id": plan_id,
                "strategy": "ortools_routing",
            },
            context(store),
        )
    ]
    terminal = events[-1].result

    assert terminal is not None and terminal.candidate is not None
    assert (
        create_builtin_problem_registry()
        .get(problem.type_id.value)
        .validate_plan(problem, terminal.candidate, store)
        .valid
    )
    assert terminal.bound is not None
