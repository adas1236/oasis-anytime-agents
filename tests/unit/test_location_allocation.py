from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from oasis.artifacts import (
    ArtifactProvenance,
    LocalArtifactStore,
    MatrixData,
    put_matrix,
    put_vector,
    read_matrix,
)
from oasis.problems import (
    Comparison,
    Deadline,
    EquityGroup,
    EquityObjective,
    LocationAllocationPolicy,
    LocationAllocationProblem,
    LocationProblemType,
    ProblemRegistryError,
    ScenarioAggregation,
    SearchStrategy,
    ServiceScenario,
    create_problem_registry,
    load_problem_data,
    problem_hashes,
)
from oasis.problems.search import all_candidate_plans, candidate_space
from oasis.schemas import CandidateSpec, DemandSpec, MissingDataPolicy, Plan


def provenance(name: str) -> ArtifactProvenance:
    return ArtifactProvenance(source_uri=f"fixture://phase5/{name}", license="CC0-1.0")


def problem_fixture(
    tmp_path: Path,
    type_id: LocationProblemType,
    policy: LocationAllocationPolicy,
) -> tuple[LocalArtifactStore, LocationAllocationProblem]:
    store = LocalArtifactStore(tmp_path / type_id.value)
    demand = gpd.GeoDataFrame(
        {
            "demand_id": ["d1", "d2", "d3"],
            "need": [4.0, 3.0, 3.0],
            "group_a": [1.0, 0.0, 0.0],
            "group_b": [0.0, 1.0, 1.0],
        },
        geometry=[Point(0, 0), Point(9, 0), Point(10, 0)],
        crs="EPSG:3857",
    )
    candidates = gpd.GeoDataFrame(
        {
            "site_id": ["s1", "s2", "s3"],
            "cost": [1.0, 2.0, 2.0],
            "capacity": [4.0, 3.0, 3.0],
            "eligible": [True, True, True],
            "existing": [True, False, False],
        },
        geometry=[Point(0, 0), Point(9, 0), Point(10, 0)],
        crs="EPSG:3857",
    )
    demand_ref = put_vector(store, demand, units="people", provenance=provenance("demand"))
    candidate_ref = put_vector(
        store, candidates, units="meters", provenance=provenance("candidates")
    )
    access_values = np.array([[0, 9, 10], [10, 0, 4], [8, 4, 0]], dtype=np.float64)
    access_ref = put_matrix(
        store,
        MatrixData(
            values=access_values,
            row_ids=("d1", "d2", "d3"),
            column_ids=("s1", "s2", "s3"),
        ),
        crs=None,
        units="minutes",
        provenance=provenance("access"),
    )
    service_ref = put_matrix(
        store,
        MatrixData(
            values=(access_values <= 4).astype(np.float64),
            row_ids=("d1", "d2", "d3"),
            column_ids=("s1", "s2", "s3"),
        ),
        crs=None,
        units="unitless",
        provenance=provenance("service"),
    )
    demand_spec = DemandSpec(
        artifact=demand_ref,
        location_id_field="demand_id",
        need_fields=("need",),
        group_fields=("group_a", "group_b"),
        missing_data_policy=MissingDataPolicy.ERROR,
    )
    candidate_spec = CandidateSpec(
        artifact=candidate_ref,
        candidate_id_field="site_id",
        opening_cost_field="cost",
        capacity_field="capacity",
        eligibility_field="eligible",
        existing_site_field=(
            "existing" if type_id is LocationProblemType.INCREMENTAL_COVERAGE else None
        ),
        generation_method="supplied",
    )
    blank = "0" * 64
    problem = LocationAllocationProblem(
        type_id=type_id,
        demand=demand_spec,
        candidates=candidate_spec,
        access_matrix=access_ref,
        service_scenarios=(ServiceScenario(name="base", service_matrix=service_ref),),
        need_field="need",
        groups=(EquityGroup(name="a", field="group_a"), EquityGroup(name="b", field="group_b")),
        policy=policy,
        evidence_hash=blank,
        policy_hash=blank,
        problem_hash=blank,
    )
    evidence_hash, policy_hash, full_hash = problem_hashes(problem)
    return store, problem.model_copy(
        update={
            "evidence_hash": evidence_hash,
            "policy_hash": policy_hash,
            "problem_hash": full_hash,
        }
    )


def exact_best(
    store: LocalArtifactStore, problem: LocationAllocationProblem
) -> tuple[Plan, object]:
    plugin = create_problem_registry().get(problem.type_id.value)
    data = load_problem_data(problem, store)
    best_plan = None
    best_score = None
    for plan in all_candidate_plans(problem, data).plans:
        score = plugin.measure(problem, plan, store)
        if score.feasible and (
            best_score is None or plugin.compare(score, best_score) is Comparison.BETTER
        ):
            best_plan, best_score = plan, score
    assert best_plan is not None and best_score is not None
    return best_plan, best_score


@pytest.mark.parametrize(
    ("type_id", "policy", "metric", "expected"),
    [
        (
            LocationProblemType.MAX_WEIGHTED_COVERAGE,
            LocationAllocationPolicy(site_limit=1),
            "coverage",
            0.6,
        ),
        (
            LocationProblemType.MIN_COST_TARGET_COVERAGE,
            LocationAllocationPolicy(site_limit=3, coverage_target=0.6),
            "opening_cost",
            2.0,
        ),
        (
            LocationProblemType.WEIGHTED_P_MEDIAN,
            LocationAllocationPolicy(site_limit=1),
            "average_access",
            4.8,
        ),
        (
            LocationProblemType.P_CENTER,
            LocationAllocationPolicy(site_limit=1),
            "maximum_access",
            9.0,
        ),
        (
            LocationProblemType.QUANTILE_ACCESS,
            LocationAllocationPolicy(site_limit=1, quantile=0.7),
            "quantile_access",
            8.0,
        ),
        (
            LocationProblemType.CAPACITATED_ALLOCATION,
            LocationAllocationPolicy(site_limit=2),
            "coverage",
            0.7,
        ),
        (
            LocationProblemType.EQUITY_COVERAGE,
            LocationAllocationPolicy(site_limit=2, equity_objective=EquityObjective.MAX_MIN),
            "coverage",
            1.0,
        ),
        (
            LocationProblemType.INCREMENTAL_COVERAGE,
            LocationAllocationPolicy(site_limit=3, new_site_limit=1),
            "coverage",
            1.0,
        ),
        (
            LocationProblemType.RESILIENT_COVERAGE,
            LocationAllocationPolicy(site_limit=3, redundancy=2),
            "one_failure_coverage",
            0.6,
        ),
    ],
)
def test_hand_calculated_tiny_optimum_for_every_family(
    tmp_path: Path,
    type_id: LocationProblemType,
    policy: LocationAllocationPolicy,
    metric: str,
    expected: float,
) -> None:
    store, problem = problem_fixture(tmp_path, type_id, policy)
    plugin = create_problem_registry().get(type_id.value)

    assert plugin.validate_spec(problem, store).valid
    baseline = plugin.make_baseline(problem, store, Deadline(time.monotonic() + 5))
    assert plugin.validate_plan(problem, baseline, store).valid
    expected_baseline_strategy = {
        LocationProblemType.MIN_COST_TARGET_COVERAGE: "cost_aware_greedy_cover",
        LocationProblemType.WEIGHTED_P_MEDIAN: "greedy_p_median",
        LocationProblemType.P_CENTER: "farthest_first",
        LocationProblemType.QUANTILE_ACCESS: "farthest_first",
        LocationProblemType.CAPACITATED_ALLOCATION: "feasible_flow",
    }.get(type_id, "greedy_coverage")
    assert baseline.metadata["strategy"] == expected_baseline_strategy
    _, optimum = exact_best(store, problem)

    assert optimum.overall_metrics[metric] == pytest.approx(expected)
    if type_id in {
        LocationProblemType.WEIGHTED_P_MEDIAN,
        LocationProblemType.P_CENTER,
        LocationProblemType.QUANTILE_ACCESS,
    }:
        assert set(optimum.group_metrics) == {"a", "b"}
        assert "average_access" in optimum.group_metrics["a"]
        assert "maximum_access" in optimum.group_metrics["b"]
        assert "quantile_access" in optimum.group_metrics["b"]
    if type_id is LocationProblemType.CAPACITATED_ALLOCATION:
        assert "served_demand" in optimum.group_metrics["a"]
        assert "average_access" in optimum.group_metrics["b"]
    if type_id is LocationProblemType.RESILIENT_COVERAGE:
        assert optimum.overall_metrics["redundant_coverage"] == pytest.approx(0.6)
        assert "one_failure_coverage" in optimum.group_metrics["a"]
        assert "redundant_coverage" in optimum.group_metrics["b"]


def test_multiple_optima_and_binary_demand_coverage_are_not_double_counted(
    tmp_path: Path,
) -> None:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=2),
    )
    plugin = create_problem_registry().get(problem.type_id.value)
    s1_s2 = Plan(problem_type=problem.type_id.value, selected_site_ids=("s1", "s2"))
    s1_s3 = Plan(problem_type=problem.type_id.value, selected_site_ids=("s1", "s3"))
    duplicate_cover = Plan(problem_type=problem.type_id.value, selected_site_ids=("s2", "s3"))

    left = plugin.measure(problem, s1_s2, store)
    right = plugin.measure(problem, s1_s3, store)
    overlap = plugin.measure(problem, duplicate_cover, store)

    assert left.feasible and right.feasible
    assert plugin.compare(left, right) is Comparison.EQUAL
    assert overlap.overall_metrics["coverage"] == pytest.approx(0.6)


def test_capacity_and_equity_infeasibility_are_reported_without_repair(
    tmp_path: Path,
) -> None:
    capacity_store, capacity_problem = problem_fixture(
        tmp_path,
        LocationProblemType.CAPACITATED_ALLOCATION,
        LocationAllocationPolicy(site_limit=1),
    )
    capacity_plugin = create_problem_registry().get(capacity_problem.type_id.value)
    invalid_allocation = Plan(
        problem_type=capacity_problem.type_id.value,
        selected_site_ids=("s2",),
        allocations=({"demand_id": "d1", "site_id": "s2", "amount": 5.0},),
    )
    report = capacity_plugin.validate_plan(capacity_problem, invalid_allocation, capacity_store)
    assert {issue.code for issue in report.issues} == {"demand_overallocation", "capacity_exceeded"}
    tight_store, tight_capacity = problem_fixture(
        tmp_path,
        LocationProblemType.CAPACITATED_ALLOCATION,
        LocationAllocationPolicy(site_limit=1, coverage_target=0.8),
    )
    tight_plugin = create_problem_registry().get(tight_capacity.type_id.value)
    with pytest.raises(ValueError, match="baseline admission"):
        tight_plugin.make_baseline(tight_capacity, tight_store, Deadline(time.monotonic() + 5))

    equity_store, equity_problem = problem_fixture(
        tmp_path,
        LocationProblemType.EQUITY_COVERAGE,
        LocationAllocationPolicy(
            site_limit=1,
            equity_objective=EquityObjective.FLOORS,
            group_floors={"a": 1.0, "b": 1.0},
        ),
    )
    equity_plugin = create_problem_registry().get(equity_problem.type_id.value)
    with pytest.raises(ValueError, match="baseline admission"):
        equity_plugin.make_baseline(equity_problem, equity_store, Deadline(time.monotonic() + 5))


def test_capacity_assignments_conserve_supply_and_demand(tmp_path: Path) -> None:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.CAPACITATED_ALLOCATION,
        LocationAllocationPolicy(site_limit=2),
    )
    plan, score = exact_best(store, problem)
    plugin = create_problem_registry().get(problem.type_id.value)

    assert plugin.validate_plan(problem, plan, store).valid
    assert sum(float(record["amount"]) for record in plan.allocations) == pytest.approx(7.0)
    assert score.overall_metrics["served_demand"] == pytest.approx(7.0)
    assert score.overall_metrics["unmet_demand"] == pytest.approx(3.0)


def test_group_floor_and_lexicographic_max_min_comparators(tmp_path: Path) -> None:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.EQUITY_COVERAGE,
        LocationAllocationPolicy(site_limit=2, equity_objective=EquityObjective.MAX_MIN),
    )
    plugin = create_problem_registry().get(problem.type_id.value)
    inequitable = plugin.measure(
        problem,
        Plan(problem_type=problem.type_id.value, selected_site_ids=("s2", "s3")),
        store,
    )
    equitable = plugin.measure(
        problem,
        Plan(problem_type=problem.type_id.value, selected_site_ids=("s1", "s2")),
        store,
    )

    assert inequitable.group_metrics["a"]["coverage"] == 0.0
    assert equitable.group_metrics["a"]["coverage"] == 1.0
    assert plugin.compare(equitable, inequitable) is Comparison.BETTER


def test_problem_hash_changes_with_policy_and_registry_rejects_cross_problem_comparison(
    tmp_path: Path,
) -> None:
    store, first = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    _, second = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=2),
    )
    plugin = create_problem_registry().get(first.type_id.value)
    first_score = plugin.measure(
        first, Plan(problem_type=first.type_id.value, selected_site_ids=("s1",)), store
    )
    second_score = plugin.measure(
        second, Plan(problem_type=second.type_id.value, selected_site_ids=("s1",)), store
    )

    assert first.problem_hash != second.problem_hash
    with pytest.raises(ValueError, match="different immutable problems"):
        plugin.compare(first_score, second_score)


def test_public_problem_and_scorecard_schemas_round_trip(tmp_path: Path) -> None:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    plugin = create_problem_registry().get(problem.type_id.value)
    score = plugin.measure(
        problem, Plan(problem_type=problem.type_id.value, selected_site_ids=("s2",)), store
    )

    assert LocationAllocationProblem.model_validate_json(problem.model_dump_json()) == problem
    assert type(score).model_validate_json(score.model_dump_json()) == score


def test_scenario_aggregation_reports_each_scenario_and_uses_worst_case(tmp_path: Path) -> None:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    normal = read_matrix(store, problem.service_scenarios[0].service_matrix)
    outage_values = normal.values.copy()
    outage_values[:, 1] = 0.0
    outage_ref = put_matrix(
        store,
        MatrixData(
            values=outage_values,
            row_ids=normal.row_ids,
            column_ids=normal.column_ids,
        ),
        crs=None,
        units="unitless",
        provenance=provenance("outage"),
    )
    policy = problem.policy.model_copy(
        update={"scenario_aggregation": ScenarioAggregation.WORST_CASE}
    )
    modified = problem.model_copy(
        update={
            "service_scenarios": (
                problem.service_scenarios[0],
                ServiceScenario(name="outage", service_matrix=outage_ref),
            ),
            "policy": policy,
        }
    )
    evidence_hash, policy_hash, full_hash = problem_hashes(modified)
    modified = modified.model_copy(
        update={
            "evidence_hash": evidence_hash,
            "policy_hash": policy_hash,
            "problem_hash": full_hash,
        }
    )
    plugin = create_problem_registry().get(modified.type_id.value)
    score = plugin.measure(
        modified,
        Plan(problem_type=modified.type_id.value, selected_site_ids=("s2",)),
        store,
    )

    assert score.scenario_metrics["base"]["coverage"] == pytest.approx(0.6)
    assert score.scenario_metrics["outage"]["coverage"] == 0.0
    assert score.overall_metrics["coverage"] == 0.0


def test_matrix_dimension_and_unit_mismatches_fail_problem_validation(tmp_path: Path) -> None:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    bad_service = put_matrix(
        store,
        MatrixData(
            values=np.ones((3, 3)),
            row_ids=("d1", "d2", "d3"),
            column_ids=("s3", "s2", "s1"),
        ),
        crs=None,
        units="people",
        provenance=provenance("bad-service"),
    )
    modified = problem.model_copy(
        update={"service_scenarios": (ServiceScenario(name="base", service_matrix=bad_service),)}
    )
    evidence_hash, policy_hash, full_hash = problem_hashes(modified)
    modified = modified.model_copy(
        update={
            "evidence_hash": evidence_hash,
            "policy_hash": policy_hash,
            "problem_hash": full_hash,
        }
    )
    plugin = create_problem_registry().get(modified.type_id.value)
    report = plugin.validate_spec(modified, store)

    assert not report.valid
    assert report.issues[0].code == "invalid_evidence"
    assert "unitless" in report.issues[0].message


def test_greedy_and_local_candidates_independently_rescore_identically(tmp_path: Path) -> None:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    plugin = create_problem_registry().get(problem.type_id.value)
    baseline = plugin.make_baseline(problem, store, Deadline(time.monotonic() + 5))
    assert plugin.measure(problem, baseline, store) == plugin.measure(problem, baseline, store)

    starting = Plan(problem_type=problem.type_id.value, selected_site_ids=("s1",))
    local = candidate_space(
        problem, load_problem_data(problem, store), starting, SearchStrategy.ADD_SWAP
    )
    rescored = [plugin.measure(problem, plan, store) for plan in local.plans]
    assert any(score.feasible and score.overall_metrics["coverage"] == 0.6 for score in rescored)

    capacity_store, capacity_problem = problem_fixture(
        tmp_path,
        LocationProblemType.CAPACITATED_ALLOCATION,
        LocationAllocationPolicy(site_limit=2),
    )
    capacity_plugin = create_problem_registry().get(capacity_problem.type_id.value)
    capacity_baseline = capacity_plugin.make_baseline(
        capacity_problem, capacity_store, Deadline(time.monotonic() + 5)
    )
    reassigned = tuple(
        candidate_space(
            capacity_problem,
            load_problem_data(capacity_problem, capacity_store),
            capacity_baseline,
            SearchStrategy.LOCAL_ASSIGNMENT,
        ).plans
    )
    assert len(reassigned) == 1
    assert (
        capacity_plugin.measure(capacity_problem, reassigned[0], capacity_store).comparator_key
        == capacity_plugin.measure(
            capacity_problem, capacity_baseline, capacity_store
        ).comparator_key
    )


def test_tampered_policy_is_rejected_by_immutable_problem_hash(tmp_path: Path) -> None:
    store, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    tampered = problem.model_copy(update={"policy": LocationAllocationPolicy(site_limit=2)})
    plugin = create_problem_registry().get(problem.type_id.value)

    report = plugin.validate_spec(tampered, store)

    assert not report.valid
    assert report.issues[0].code == "hash_mismatch"


def test_problem_registry_contains_every_family_and_rejects_duplicates() -> None:
    registry = create_problem_registry()

    assert {plugin.type_id for plugin in registry.list()} == {
        problem_type.value for problem_type in LocationProblemType
    }
    with pytest.raises(ProblemRegistryError, match="duplicate"):
        registry.register(registry.get(LocationProblemType.P_CENTER.value))
