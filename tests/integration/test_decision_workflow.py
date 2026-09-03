from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from unit.test_location_allocation import problem_fixture, provenance

from oasis.artifacts import ArtifactStore, LocalArtifactStore, put_json, read_json
from oasis.decision import run_decision_demo
from oasis.evidence import run_evidence_demo
from oasis.problems import (
    LocationAllocationPolicy,
    LocationAllocationProblem,
    LocationProblemType,
    Scorecard,
    VerifiedBound,
    create_problem_registry,
)
from oasis.schemas import ArtifactKind, Plan, ToolEventKind, ToolResultStatus
from oasis.tools import (
    CancellationToken,
    ToolContext,
    create_tool_registry,
    invoke_tool,
    stream_tool,
)


def context(store: ArtifactStore, *, seed: int = 17) -> ToolContext:
    return ToolContext(
        run_id="decision-integration",
        artifact_store=store,
        deadline_monotonic=time.monotonic() + 10,
        cancellation=CancellationToken(),
        seed=seed,
    )


def publish_problem_and_plan(
    tmp_path: Path,
) -> tuple[LocalArtifactStore, str, str, LocationAllocationProblem]:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    problem_ref = put_json(
        store,
        problem.model_dump(mode="json"),
        kind=ArtifactKind.JSON_SPECIFICATION,
        units="unitless",
        provenance=provenance("compiled-problem"),
        data_schema={"type": "LocationAllocationProblem", "version": "1.0.0"},
    )
    starting = Plan(problem_type=problem.type_id.value, selected_site_ids=("s1",))
    plan_ref = put_json(
        store,
        starting.model_dump(mode="json"),
        kind=ArtifactKind.PLAN,
        units="unitless",
        provenance=provenance("starting-plan"),
        data_schema={"type": "Plan", "version": "1.0.0"},
    )
    return store, problem_ref.id, plan_ref.id, problem


@pytest.mark.asyncio
async def test_frozen_cooling_center_demo_compiles_scores_and_renders(tmp_path: Path) -> None:
    result = await run_decision_demo(tmp_path)
    store = LocalArtifactStore(tmp_path)

    assert result.overall_metrics["coverage"] == pytest.approx(2 / 3)
    assert result.group_metrics["older_adults"]["coverage"] == pytest.approx(4 / 9)
    assert result.geojson_map_artifact_id.startswith("sha256-")
    assert result.svg_map_artifact_id.startswith("sha256-")
    assert store.get_metadata(result.geojson_map_artifact_id).media_type == "application/geo+json"
    assert (
        json.loads(store.read_bytes(result.geojson_map_artifact_id))["type"] == "FeatureCollection"
    )
    assert store.get_metadata(result.svg_map_artifact_id).media_type == "image/svg+xml"
    assert store.read_bytes(result.svg_map_artifact_id).startswith(b"<svg")


@pytest.mark.asyncio
async def test_exact_search_streams_improvement_bound_and_authoritative_score(
    tmp_path: Path,
) -> None:
    store, problem_id, plan_id, problem = publish_problem_and_plan(tmp_path)
    registry = create_tool_registry(discover_entry_points=False)
    events = [
        event
        async for event in stream_tool(
            registry.get("improve"),
            {
                "problem_artifact_id": problem_id,
                "starting_plan_artifact_id": plan_id,
                "strategy": "exact_enumeration",
                "max_candidates": 100,
            },
            context(store),
        )
    ]

    assert [event.kind for event in events] == [
        ToolEventKind.CANDIDATE,
        ToolEventKind.BOUND,
        ToolEventKind.RESULT,
    ]
    assert events[0].candidate is not None
    assert events[0].candidate.selected_site_ids in {("s2",), ("s3",)}
    terminal = events[-1].result
    assert terminal is not None and terminal.status is ToolResultStatus.COMPLETE
    assert terminal.bound is not None
    assert terminal.candidate is not None
    score = Scorecard.model_validate(
        read_json(store, str(terminal.metrics["best_scorecard_artifact_id"]))
    )
    plugin = create_problem_registry().get(problem.type_id.value)
    rescored = plugin.measure(problem, terminal.candidate, store)
    assert score.comparator_key == rescored.comparator_key
    assert score.raw_objective == rescored.raw_objective
    assert score.optimality_gap == 0.0
    bound = VerifiedBound.model_validate(read_json(store, terminal.bound.id))
    assert bound.complete
    assert bound.best_comparator_key == score.comparator_key


@pytest.mark.asyncio
async def test_exact_search_resume_and_fixed_seed_reproduce_event_order(tmp_path: Path) -> None:
    store, problem_id, plan_id, _ = publish_problem_and_plan(tmp_path)
    registry = create_tool_registry(discover_entry_points=False)
    first_context = context(store)
    first_events = [
        event
        async for event in stream_tool(
            registry.get("improve"),
            {
                "problem_artifact_id": problem_id,
                "starting_plan_artifact_id": plan_id,
                "strategy": "exact_enumeration",
                "max_candidates": 1,
            },
            first_context,
        )
    ]
    partial = first_events[-1].result
    assert partial is not None and partial.status is ToolResultStatus.PARTIAL
    assert partial.resume_token is not None

    resumed_events = [
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
    replayed_events = [
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

    assert resumed_events == replayed_events
    assert resumed_events[-1].result is not None
    assert resumed_events[-1].result.status is ToolResultStatus.COMPLETE


@pytest.mark.asyncio
async def test_cancellation_after_streamed_candidate_retains_last_valid_plan(
    tmp_path: Path,
) -> None:
    store, problem_id, plan_id, problem = publish_problem_and_plan(tmp_path)
    registry = create_tool_registry(discover_entry_points=False)
    tool_context = context(store)
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
        tool_context,
    ):
        if event.kind is ToolEventKind.CANDIDATE:
            retained = event.candidate
            tool_context.cancellation.cancel("stop after first verified improvement")
        elif event.kind is ToolEventKind.RESULT:
            terminal = event.result

    assert retained is not None
    assert terminal is not None and terminal.status is ToolResultStatus.EXPIRED
    plugin = create_problem_registry().get(problem.type_id.value)
    assert plugin.validate_plan(problem, retained, store).valid


@pytest.mark.asyncio
async def test_ortools_candidate_and_bound_are_independently_validated(tmp_path: Path) -> None:
    store, problem_id, plan_id, problem = publish_problem_and_plan(tmp_path)
    registry = create_tool_registry(discover_entry_points=False)
    events = [
        event
        async for event in stream_tool(
            registry.get("improve"),
            {
                "problem_artifact_id": problem_id,
                "starting_plan_artifact_id": plan_id,
                "strategy": "ortools_cp_sat",
            },
            context(store),
        )
    ]

    terminal = events[-1].result
    assert terminal is not None and terminal.status is ToolResultStatus.COMPLETE
    assert terminal.bound is not None
    assert terminal.candidate is not None
    plugin = create_problem_registry().get(problem.type_id.value)
    assert plugin.validate_plan(problem, terminal.candidate, store).valid
    bound = VerifiedBound.model_validate(read_json(store, terminal.bound.id))
    assert bound.complete
    assert bound.certificate["solver"] == "ortools_cp_sat"


@pytest.mark.asyncio
async def test_compile_problem_rejects_a_policy_with_no_feasible_baseline(tmp_path: Path) -> None:
    evidence = await run_evidence_demo(tmp_path)
    store = LocalArtifactStore(tmp_path)
    registry = create_tool_registry(discover_entry_points=False)
    result = await invoke_tool(
        registry.get("compile_problem"),
        {
            "type_id": "max_weighted_coverage",
            "demand_spec_artifact_id": evidence.demand_spec_artifact_id,
            "candidate_spec_artifact_id": evidence.candidate_spec_artifact_id,
            "access_matrix_artifact_id": evidence.access_matrix_artifact_id,
            "service_matrix_artifact_ids": {"base": evidence.service_matrix_artifact_id},
            "need_field": "population",
            "policy": {"site_limit": 1, "coverage_target": 0.9},
        },
        context(store),
    )

    assert result.status is ToolResultStatus.INFEASIBLE
    assert result.metrics["issue_codes"] == ["no_feasible_baseline"]
