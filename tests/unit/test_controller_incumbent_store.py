from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from oasis.controller import IncumbentStore
from oasis.problems import LocationAllocationPolicy, LocationProblemType, create_problem_registry
from oasis.schemas import Plan
from unit.test_location_allocation import problem_fixture


@pytest.mark.asyncio
async def test_concurrent_incumbent_commits_are_atomic_and_monotone(tmp_path: Path) -> None:
    artifact_store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    plugin = create_problem_registry().get(problem.type_id.value)
    weak = Plan(problem_type=problem.type_id.value, selected_site_ids=("s1",))
    strong = Plan(problem_type=problem.type_id.value, selected_site_ids=("s2",))
    weak_score = plugin.measure(problem, weak, artifact_store)
    strong_score = plugin.measure(problem, strong, artifact_store)
    assert plugin.compare(strong_score, weak_score).value == "better"
    incumbents = IncumbentStore(plugin.compare)

    await asyncio.gather(
        incumbents.try_commit(
            plan=strong,
            scorecard=strong_score,
            plan_artifact_id="strong-plan",
            scorecard_artifact_id="strong-score",
            source_action_id="strong",
            committed_at_ms=1,
            seed=5,
        ),
        incumbents.try_commit(
            plan=weak,
            scorecard=weak_score,
            plan_artifact_id="weak-plan",
            scorecard_artifact_id="weak-score",
            source_action_id="weak",
            committed_at_ms=1,
            seed=5,
        ),
    )

    assert incumbents.current is not None
    assert incumbents.current.plan == strong
    keys = [record.comparator_key for record in incumbents.timeline]
    assert keys == sorted(keys)


@pytest.mark.asyncio
async def test_infeasible_candidate_cannot_become_incumbent(tmp_path: Path) -> None:
    artifact_store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    plugin = create_problem_registry().get(problem.type_id.value)
    invalid = Plan(problem_type=problem.type_id.value, selected_site_ids=("missing",))
    score = plugin.measure(problem, invalid, artifact_store)
    incumbents = IncumbentStore(plugin.compare)

    committed = await incumbents.try_commit(
        plan=invalid,
        scorecard=score,
        plan_artifact_id="invalid-plan",
        scorecard_artifact_id="invalid-score",
        source_action_id="bad",
        committed_at_ms=0,
        seed=0,
    )

    assert committed is None
    assert incumbents.current is None
