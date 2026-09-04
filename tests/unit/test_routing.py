from __future__ import annotations

import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
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
    LocationAllocationPolicy,
    LocationProblemType,
    RouteProblemType,
    RouteScenario,
    RouteServicePolicy,
    RouteServiceProblem,
    ScenarioAggregation,
    ServiceScenario,
    create_builtin_problem_registry,
    load_route_data,
    problem_hashes,
    route_problem_hashes,
)
from oasis.problems.routing_search import all_route_plans, route_candidate_space
from oasis.problems.schemas import SearchStrategy
from oasis.schemas import Plan
from unit.test_location_allocation import problem_fixture, provenance


def route_provenance(name: str) -> ArtifactProvenance:
    return ArtifactProvenance(source_uri=f"fixture://phase6/{name}", license="CC0-1.0")


def route_fixture(
    tmp_path: Path,
    type_id: RouteProblemType,
    policy: RouteServicePolicy,
    *,
    travel: np.ndarray | None = None,
    window_ends: tuple[float, ...] = (100.0, 100.0, 100.0, 100.0),
    scenarios: tuple[tuple[str, np.ndarray, float], ...] | None = None,
) -> tuple[LocalArtifactStore, RouteServiceProblem]:
    store = LocalArtifactStore(tmp_path / type_id.value)
    node_ids = ("depot", "a", "b", "c")
    nodes = gpd.GeoDataFrame(
        {
            "node_id": node_ids,
            "prize": [0.0, 2.0, 5.0, 4.0],
            "demand": [0.0, 2.0, 4.0, 3.0],
            "service": [0.0, 0.0, 0.0, 0.0],
            "window_start": [0.0, 0.0, 0.0, 0.0],
            "window_end": window_ends,
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        crs="EPSG:3857",
    )
    node_ref = put_vector(store, nodes, units="people", provenance=route_provenance("nodes"))
    base = (
        travel
        if travel is not None
        else np.array(
            [[0, 1, 4, 3], [2, 0, 2, 5], [4, 2, 0, 1], [3, 5, 1, 0]],
            dtype=np.float64,
        )
    )
    scenario_values = scenarios or (("normal", base, 1.0),)
    route_scenarios = []
    for name, values, weight in scenario_values:
        reference = put_matrix(
            store,
            MatrixData(values=values, row_ids=node_ids, column_ids=node_ids),
            crs=None,
            units="minutes",
            provenance=route_provenance(name),
        )
        route_scenarios.append(RouteScenario(name=name, travel_matrix=reference, weight=weight))
    blank = "0" * 64
    problem = RouteServiceProblem(
        type_id=type_id,
        nodes=node_ref,
        node_id_field="node_id",
        prize_field="prize" if type_id is not RouteProblemType.TSP else None,
        demand_field="demand" if policy.vehicle_capacity is not None else None,
        service_time_field="service",
        window_start_field="window_start",
        window_end_field="window_end",
        travel_scenarios=tuple(route_scenarios),
        policy=policy,
        evidence_hash=blank,
        policy_hash=blank,
        problem_hash=blank,
    )
    evidence_hash, policy_hash, full_hash = route_problem_hashes(problem)
    return store, problem.model_copy(
        update={
            "evidence_hash": evidence_hash,
            "policy_hash": policy_hash,
            "problem_hash": full_hash,
        }
    )


def best_route(store: LocalArtifactStore, problem: RouteServiceProblem) -> tuple[Plan, object]:
    plugin = create_builtin_problem_registry().get(problem.type_id.value)
    data = load_route_data(problem, store)
    best_plan = None
    best_score = None
    for plan in all_route_plans(problem, data).plans:
        score = plugin.measure(problem, plan, store)
        if score.feasible and (
            best_score is None or plugin.compare(score, best_score) is Comparison.BETTER
        ):
            best_plan, best_score = plan, score
    assert best_plan is not None and best_score is not None
    return best_plan, best_score


def test_exact_small_tsp_and_orienteering_optima(tmp_path: Path) -> None:
    tsp_store, tsp = route_fixture(
        tmp_path,
        RouteProblemType.TSP,
        RouteServicePolicy(depot_ids=("depot",), shift_length=100, time_units="minutes"),
    )
    tsp_plugin = create_builtin_problem_registry().get(tsp.type_id.value)
    tsp_baseline = tsp_plugin.make_baseline(tsp, tsp_store, Deadline(time.monotonic() + 5))
    tsp_plan, tsp_score = best_route(tsp_store, tsp)

    assert tsp_plugin.validate_plan(tsp, tsp_baseline, tsp_store).valid
    assert tsp_score.raw_objective["total_route_time"] == 7.0
    assert tsp_plan.routes[0]["node_ids"] == ["depot", "a", "b", "c", "depot"]

    route_store, orienteering = route_fixture(
        tmp_path,
        RouteProblemType.ORIENTEERING,
        RouteServicePolicy(depot_ids=("depot",), shift_length=6, time_units="minutes"),
    )
    route_plan, route_score = best_route(route_store, orienteering)

    assert route_score.raw_objective["collected_prize"] == 4.0
    assert route_plan.routes[0]["node_ids"] == ["depot", "c", "depot"]


def test_directed_unreachable_arc_and_time_capacity_violations(tmp_path: Path) -> None:
    travel = np.array(
        [[0, 1, 4, 3], [2, 0, 2, 5], [4, math.inf, 0, 1], [3, 5, 1, 0]],
        dtype=np.float64,
    )
    store, problem = route_fixture(
        tmp_path,
        RouteProblemType.MOBILE_SERVICE,
        RouteServicePolicy(
            depot_ids=("depot",),
            shift_length=3,
            time_units="minutes",
            vehicle_capacity=5,
            capacity_units="people",
        ),
        travel=travel,
        window_ends=(100.0, 100.0, 1.0, 100.0),
    )
    plugin = create_builtin_problem_registry().get(problem.type_id.value)
    plan = Plan(
        problem_type=problem.type_id.value,
        routes=({"vehicle_id": "vehicle-1", "node_ids": ["depot", "b", "a", "depot"]},),
    )
    report = plugin.validate_plan(problem, plan, store)

    assert {issue.code for issue in report.issues} >= {
        "unreachable_arc",
        "time_window",
        "shift_length",
        "capacity_exceeded",
    }


def test_unique_demand_across_vehicles_and_route_neighborhoods(tmp_path: Path) -> None:
    store, problem = route_fixture(
        tmp_path,
        RouteProblemType.MOBILE_SERVICE,
        RouteServicePolicy(
            depot_ids=("depot",),
            vehicle_count=2,
            shift_length=100,
            time_units="minutes",
            vehicle_capacity=10,
            capacity_units="people",
        ),
    )
    plugin = create_builtin_problem_registry().get(problem.type_id.value)
    valid = Plan(
        problem_type=problem.type_id.value,
        routes=(
            {"vehicle_id": "vehicle-1", "node_ids": ["depot", "a", "depot"]},
            {"vehicle_id": "vehicle-2", "node_ids": ["depot", "b", "depot"]},
        ),
    )
    duplicate = valid.model_copy(
        update={
            "routes": (
                valid.routes[0],
                {"vehicle_id": "vehicle-2", "node_ids": ["depot", "a", "depot"]},
            )
        }
    )

    assert plugin.measure(problem, valid, store).overall_metrics["served_value"] == 6.0
    assert "duplicate_visit" in {
        issue.code for issue in plugin.validate_plan(problem, duplicate, store).issues
    }
    data = load_route_data(problem, store)
    for strategy in (SearchStrategy.TWO_OPT, SearchStrategy.RELOCATE, SearchStrategy.SWAP):
        space = route_candidate_space(problem, data, valid, strategy)
        assert space.total is not None


def test_route_scenarios_use_worst_case_comparator(tmp_path: Path) -> None:
    normal = np.array(
        [[0, 1, 4, 3], [2, 0, 2, 5], [4, 2, 0, 1], [3, 5, 1, 0]],
        dtype=np.float64,
    )
    congested = normal.copy()
    congested[0, 1] = 8
    store, problem = route_fixture(
        tmp_path,
        RouteProblemType.ORIENTEERING,
        RouteServicePolicy(
            depot_ids=("depot",),
            shift_length=100,
            time_units="minutes",
            scenario_aggregation=ScenarioAggregation.WORST_CASE,
        ),
        scenarios=(("normal", normal, 1.0), ("congested", congested, 1.0)),
    )
    plugin = create_builtin_problem_registry().get(problem.type_id.value)
    plan = Plan(
        problem_type=problem.type_id.value,
        routes=({"vehicle_id": "vehicle-1", "node_ids": ["depot", "a", "depot"]},),
    )
    score = plugin.measure(problem, plan, store)

    assert score.scenario_metrics["normal"]["total_route_time"] == 3.0
    assert score.scenario_metrics["congested"]["total_route_time"] == 10.0
    assert score.overall_metrics["total_route_time"] == 10.0


def test_location_scenario_wrappers_cover_demand_travel_and_failures(tmp_path: Path) -> None:
    store, coverage = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1, scenario_aggregation=ScenarioAggregation.WORST_CASE),
    )
    multiplier = put_matrix(
        store,
        MatrixData(
            values=np.array([[2.0], [1.0], [1.0]]),
            row_ids=("d1", "d2", "d3"),
            column_ids=("multiplier",),
        ),
        crs=None,
        units="unitless",
        provenance=provenance("demand-scenario"),
    )
    modified = coverage.model_copy(
        update={
            "service_scenarios": (
                coverage.service_scenarios[0].model_copy(update={"demand_multiplier": multiplier}),
                ServiceScenario(
                    name="failure",
                    service_matrix=coverage.service_scenarios[0].service_matrix,
                    failed_site_ids=("s1",),
                ),
            )
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
    plugin = create_builtin_problem_registry().get(modified.type_id.value)
    score = plugin.measure(
        modified,
        Plan(problem_type=modified.type_id.value, selected_site_ids=("s1",)),
        store,
    )

    assert score.scenario_metrics["base"]["demand"] == 14.0
    assert score.scenario_metrics["failure"]["coverage"] == 0.0
    assert score.overall_metrics["coverage"] == 0.0

    access_store, access_problem = problem_fixture(
        tmp_path,
        LocationProblemType.WEIGHTED_P_MEDIAN,
        LocationAllocationPolicy(site_limit=1, scenario_aggregation=ScenarioAggregation.WORST_CASE),
    )
    base_access = read_matrix(access_store, access_problem.access_matrix)
    congested_access = put_matrix(
        access_store,
        MatrixData(
            values=base_access.values * 2,
            row_ids=base_access.row_ids,
            column_ids=base_access.column_ids,
        ),
        crs=None,
        units="minutes",
        provenance=provenance("travel-scenario"),
    )
    access_modified = access_problem.model_copy(
        update={
            "service_scenarios": (
                access_problem.service_scenarios[0],
                ServiceScenario(
                    name="congested",
                    service_matrix=access_problem.service_scenarios[0].service_matrix,
                    access_matrix=congested_access,
                ),
            )
        }
    )
    evidence_hash, policy_hash, full_hash = problem_hashes(access_modified)
    access_modified = access_modified.model_copy(
        update={
            "evidence_hash": evidence_hash,
            "policy_hash": policy_hash,
            "problem_hash": full_hash,
        }
    )
    access_score = (
        create_builtin_problem_registry()
        .get(access_modified.type_id.value)
        .measure(
            access_modified,
            Plan(problem_type=access_modified.type_id.value, selected_site_ids=("s2",)),
            access_store,
        )
    )

    assert access_score.scenario_metrics["base"]["average_access"] == 4.8
    assert access_score.scenario_metrics["congested"]["average_access"] == 9.6
    assert access_score.overall_metrics["average_access"] == 9.6


def test_route_problem_schema_round_trip_and_full_registry(tmp_path: Path) -> None:
    store, problem = route_fixture(
        tmp_path,
        RouteProblemType.TSP,
        RouteServicePolicy(depot_ids=("depot",), shift_length=100, time_units="minutes"),
    )
    plugin = create_builtin_problem_registry().get(problem.type_id.value)
    plan = plugin.make_baseline(problem, store, Deadline(time.monotonic() + 5))
    score = plugin.measure(problem, plan, store)

    assert RouteServiceProblem.model_validate_json(problem.model_dump_json()) == problem
    assert type(score).model_validate_json(score.model_dump_json()) == score
    assert {plugin.type_id for plugin in create_builtin_problem_registry().list()} >= {
        value.value for value in RouteProblemType
    }


def test_phase5_location_problem_json_remains_compatible(tmp_path: Path) -> None:
    _, problem = problem_fixture(
        tmp_path,
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationAllocationPolicy(site_limit=1),
    )
    payload = problem.model_dump(mode="json")
    payload["schema_version"] = "1.0.0"
    for scenario in payload["service_scenarios"]:
        scenario.pop("access_matrix")
        scenario.pop("demand_multiplier")
        scenario.pop("failed_site_ids")

    restored = type(problem).model_validate(payload)
    evidence_hash, policy_hash, full_hash = problem_hashes(restored)
    restored = restored.model_copy(
        update={
            "evidence_hash": evidence_hash,
            "policy_hash": policy_hash,
            "problem_hash": full_hash,
        }
    )

    assert restored.schema_version == "1.0.0"
    assert restored.service_scenarios[0].access_matrix is None
    assert restored.service_scenarios[0].failed_site_ids == ()
    assert (
        create_builtin_problem_registry()
        .get(restored.type_id.value)
        .validate_spec(restored, LocalArtifactStore(tmp_path / restored.type_id.value))
        .valid
    )
