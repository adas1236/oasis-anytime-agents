"""Deterministic synthetic location-allocation and routing instance generators."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from oasis.artifacts import (
    ArtifactProvenance,
    ArtifactStore,
    MatrixData,
    put_json,
    put_matrix,
    put_vector,
)
from oasis.evaluation.models import (
    ConstraintRegime,
    DatasetSplit,
    EquityStructure,
    GeneratedInstance,
    ProblemFamily,
    SpatialDistribution,
    SyntheticInstanceSpec,
)
from oasis.schemas import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactTransformation,
    CandidateSpec,
    DemandSpec,
    MissingDataPolicy,
    ToolResultStatus,
)
from oasis.tools import CancellationToken, ToolContext, create_tool_registry, invoke_tool


def effective_seed(spec: SyntheticInstanceSpec) -> int:
    """Derive disjoint development/held-out RNG namespaces from an explicit seed."""

    payload = f"{spec.generator_version}:{spec.split.value}:{spec.seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _points(
    distribution: SpatialDistribution,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if distribution is SpatialDistribution.UNIFORM:
        return rng.uniform(0.0, 100.0, size=(count, 2))
    if distribution is SpatialDistribution.CLUSTERED:
        centers = np.array(((25.0, 25.0), (75.0, 75.0)))
        labels = rng.integers(0, len(centers), size=count)
        return centers[labels] + rng.normal(0.0, 8.0, size=(count, 2))
    if distribution is SpatialDistribution.GRID:
        width = math.ceil(math.sqrt(count))
        grid = np.array(
            [(x, y) for y in np.linspace(0.0, 100.0, width) for x in np.linspace(0.0, 100.0, width)]
        )
        return grid[:count]
    if distribution is SpatialDistribution.RING:
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        return np.column_stack((50.0 + 40.0 * np.cos(angles), 50.0 + 40.0 * np.sin(angles)))
    if distribution is SpatialDistribution.CORRIDOR:
        x = np.linspace(0.0, 100.0, count)
        return np.column_stack((x, 50.0 + rng.normal(0.0, 2.0, size=count)))
    if distribution is SpatialDistribution.ISLANDS:
        labels = np.arange(count) % 2
        centers = np.array(((20.0, 50.0), (80.0, 50.0)))
        return centers[labels] + rng.normal(0.0, 5.0, size=(count, 2))
    points = rng.uniform(15.0, 85.0, size=(count, 2))
    points[-1] = (150.0, 150.0)
    return points


def _provenance(spec: SyntheticInstanceSpec, role: str) -> ArtifactProvenance:
    fingerprint = hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:16]
    return ArtifactProvenance(
        source_uri=f"oasis://evaluation/{spec.generator_version}/{fingerprint}/{role}",
        source_provider="oasis-synthetic-generator",
        source_version=spec.generator_version,
        license="CC0-1.0",
        lineage=ArtifactLineage(
            transformations=(
                ArtifactTransformation(
                    name="generate_synthetic_instance",
                    version=spec.generator_version,
                    parameters={
                        "role": role,
                        "seed": spec.seed,
                        "split": spec.split.value,
                        "distribution": spec.distribution.value,
                    },
                ),
            )
        ),
    )


def _json_spec(
    store: ArtifactStore,
    value: DemandSpec | CandidateSpec,
    spec: SyntheticInstanceSpec,
    role: str,
) -> str:
    return put_json(
        store,
        value.model_dump(mode="json"),
        kind=ArtifactKind.JSON_SPECIFICATION,
        units="unitless",
        provenance=_provenance(spec, role),
        data_schema={"type": type(value).__name__, "version": value.schema_version},
    ).id


def _distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distances: np.ndarray = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return distances


def _group_membership(
    points: np.ndarray,
    structure: EquityStructure,
) -> np.ndarray:
    if structure is EquityStructure.EMPTY:
        return np.zeros(len(points), dtype=np.int64)
    if structure is EquityStructure.ISOLATED:
        threshold = float(np.quantile(points[:, 0], 0.70))
        return (points[:, 0] >= threshold).astype(np.int64)
    return (np.arange(len(points)) % 2).astype(np.int64)


def _apply_boundaries_and_duplicates(
    spec: SyntheticInstanceSpec,
    demand_points: np.ndarray,
    candidate_points: np.ndarray,
) -> None:
    demand_duplicates = min(
        len(demand_points) // 2, int(len(demand_points) * spec.duplicate_fraction)
    )
    candidate_duplicates = min(
        len(candidate_points) // 2, int(len(candidate_points) * spec.duplicate_fraction)
    )
    if demand_duplicates:
        demand_points[-demand_duplicates:] = demand_points[:demand_duplicates]
    if candidate_duplicates:
        candidate_points[-candidate_duplicates:] = candidate_points[:candidate_duplicates]
    if spec.force_distance_ties and len(candidate_points) >= 2:
        demand_points[0] = (50.0, 50.0)
        candidate_points[0] = (40.0, 50.0)
        candidate_points[1] = (60.0, 50.0)


def _location_inputs(
    spec: SyntheticInstanceSpec,
    store: ArtifactStore,
    rng: np.random.Generator,
) -> dict[str, object]:
    demand_points = _points(spec.distribution, spec.demand_count, rng)
    candidate_points = _points(spec.distribution, spec.candidate_count, rng)
    if spec.distribution in {SpatialDistribution.CORRIDOR, SpatialDistribution.GRID}:
        candidate_points = demand_points[
            np.linspace(0, len(demand_points) - 1, spec.candidate_count).astype(int)
        ].copy()
    _apply_boundaries_and_duplicates(spec, demand_points, candidate_points)
    demand_ids = tuple(f"demand-{index:04d}" for index in range(spec.demand_count))
    candidate_ids = tuple(f"site-{index:04d}" for index in range(spec.candidate_count))
    need = rng.integers(5, 101, size=spec.demand_count).astype(np.float64)
    membership = _group_membership(demand_points, spec.equity_structure)
    demand_frame = gpd.GeoDataFrame(
        {
            "demand_id": demand_ids,
            "need": need,
            "underserved": membership,
            "suppressed": np.zeros(spec.demand_count, dtype=np.bool_),
        },
        geometry=[Point(float(x), float(y)) for x, y in demand_points],
        crs="EPSG:3857",
    )
    total_need = float(need.sum())
    capacity_scale = 0.65 if spec.constraint_regime is ConstraintRegime.TIGHT else 1.0
    base_capacity = max(1.0, total_need * capacity_scale / max(1, spec.site_limit))
    capacities = base_capacity * rng.uniform(0.75, 1.25, size=spec.candidate_count)
    candidates_frame = gpd.GeoDataFrame(
        {
            "site_id": candidate_ids,
            "opening_cost": np.round(rng.uniform(1.0, 10.0, spec.candidate_count), 6),
            "capacity": capacities,
            "eligible": np.ones(spec.candidate_count, dtype=np.bool_),
            "existing": np.arange(spec.candidate_count) == 0,
        },
        geometry=[Point(float(x), float(y)) for x, y in candidate_points],
        crs="EPSG:3857",
    )
    demand_ref = put_vector(
        store, demand_frame, units="people", provenance=_provenance(spec, "demand")
    )
    candidate_ref = put_vector(
        store, candidates_frame, units="meters", provenance=_provenance(spec, "candidates")
    )
    group_fields = (
        ()
        if spec.equity_structure in {EquityStructure.NONE, EquityStructure.EMPTY}
        else ("underserved",)
    )
    demand_spec = DemandSpec(
        artifact=demand_ref,
        location_id_field="demand_id",
        need_fields=("need",),
        group_fields=group_fields,
        suppression_fields=("suppressed",),
        spatial_resolution="synthetic projected points",
        missing_data_policy=MissingDataPolicy.ERROR,
    )
    candidate_spec = CandidateSpec(
        artifact=candidate_ref,
        candidate_id_field="site_id",
        opening_cost_field="opening_cost",
        capacity_field="capacity",
        eligibility_field="eligible",
        existing_site_field=("existing" if spec.problem_type == "incremental_coverage" else None),
        generation_method=f"synthetic_{spec.distribution.value}",
    )
    demand_spec_id = _json_spec(store, demand_spec, spec, "demand_spec")
    candidate_spec_id = _json_spec(store, candidate_spec, spec, "candidate_spec")
    access = _distance(demand_points, candidate_points)
    if spec.directed_travel:
        access *= np.where(candidate_points[None, :, 0] < demand_points[:, None, 0], 1.2, 1.0)
    if spec.unreachable_fraction > 0:
        finite_access = access.copy()
        unreachable = rng.random(access.shape) < spec.unreachable_fraction
        access[unreachable] = math.inf
        for row in range(len(access)):
            if np.isinf(access[row]).all():
                nearest_column = int(np.argmin(finite_access[row]))
                access[row, nearest_column] = finite_access[row, nearest_column]
    access_ref = put_matrix(
        store,
        MatrixData(access, demand_ids, candidate_ids),
        crs="EPSG:3857",
        units="minutes",
        provenance=_provenance(spec, "access"),
    )
    nearest = access.min(axis=1)
    threshold = (
        float(access[0, 0])
        if spec.service_threshold_boundary
        else max(1.0, float(np.quantile(nearest, 0.75)) * 1.35)
    )
    service_ids: dict[str, str] = {}
    access_scenario_ids: dict[str, str] = {}
    failed_sites: dict[str, list[str]] = {}
    for index in range(spec.scenario_count):
        name = "normal" if index == 0 else f"scenario_{index}"
        scenario_access = access * (1.0 + 0.1 * index)
        service = (scenario_access <= threshold).astype(np.float64)
        if spec.constraint_regime is ConstraintRegime.INFEASIBLE and group_fields:
            service[membership.astype(bool), :] = 0.0
        service_ids[name] = put_matrix(
            store,
            MatrixData(service, demand_ids, candidate_ids),
            crs=None,
            units="unitless",
            provenance=_provenance(spec, f"service_{name}"),
        ).id
        access_scenario_ids[name] = put_matrix(
            store,
            MatrixData(scenario_access, demand_ids, candidate_ids),
            crs="EPSG:3857",
            units="minutes",
            provenance=_provenance(spec, f"access_{name}"),
        ).id
        if spec.problem_type == "resilient_coverage" and index > 0:
            failed_sites[name] = [candidate_ids[(index - 1) % len(candidate_ids)]]

    policy: dict[str, object] = {"site_limit": spec.site_limit}
    if spec.problem_type == "min_cost_target_coverage":
        base_service = (access <= threshold).astype(np.float64)
        best_single_coverage = float(np.max(need @ base_service) / need.sum())
        policy["coverage_target"] = min(0.5, best_single_coverage)
    if spec.problem_type == "incremental_coverage":
        policy["new_site_limit"] = max(0, spec.site_limit - 1)
    if spec.problem_type == "resilient_coverage":
        policy["redundancy"] = min(2, spec.site_limit)
        policy["scenario_aggregation"] = "worst_case"
    groups: list[dict[str, object]] = []
    coverage_types = {
        "max_weighted_coverage",
        "min_cost_target_coverage",
        "capacitated_allocation",
        "equity_coverage",
        "incremental_coverage",
        "resilient_coverage",
    }
    if group_fields:
        groups.append({"name": "underserved", "field": "underserved", "match_value": 1})
        if spec.problem_type in coverage_types:
            floor = {
                ConstraintRegime.FEASIBLE: 0.0,
                ConstraintRegime.TIGHT: 0.25,
                ConstraintRegime.INFEASIBLE: 1.0,
            }[spec.constraint_regime]
            policy.update(
                {
                    "equity_objective": "floors",
                    "group_floors": {"underserved": floor},
                }
            )
    if spec.problem_type == "equity_coverage":
        policy["equity_objective"] = "max_min"
        policy["group_floors"] = {}
    return {
        "type_id": spec.problem_type,
        "demand_spec_artifact_id": demand_spec_id,
        "candidate_spec_artifact_id": candidate_spec_id,
        "access_matrix_artifact_id": access_ref.id,
        "service_matrix_artifact_ids": service_ids,
        "access_scenario_artifact_ids": (
            {} if spec.problem_type == "capacitated_allocation" else access_scenario_ids
        ),
        "failed_site_ids_by_scenario": failed_sites,
        "need_field": "need",
        "groups": groups,
        "policy": policy,
    }


def _route_inputs(
    spec: SyntheticInstanceSpec,
    store: ArtifactStore,
    rng: np.random.Generator,
) -> dict[str, object]:
    service_count = spec.demand_count
    points = _points(spec.distribution, service_count + 1, rng)
    points[0] = (50.0, 50.0)
    duplicate_count = min(len(points) // 2, int(len(points) * spec.duplicate_fraction))
    if duplicate_count:
        points[-duplicate_count:] = points[1 : duplicate_count + 1]
    if spec.force_distance_ties and len(points) >= 3:
        points[0] = (50.0, 50.0)
        points[1] = (40.0, 50.0)
        points[2] = (60.0, 50.0)
    node_ids = ("depot", *(f"stop-{index:04d}" for index in range(service_count)))
    prizes = np.concatenate(([0.0], rng.integers(1, 20, size=service_count))).astype(float)
    demands = np.concatenate(([0.0], rng.integers(1, 8, size=service_count))).astype(float)
    service_times = np.concatenate(([0.0], np.full(service_count, 2.0)))
    base_travel = _distance(points, points)
    if spec.directed_travel:
        base_travel *= np.where(points[None, :, 0] < points[:, None, 0], 1.15, 1.0)
    if spec.unreachable_fraction > 0:
        mask = rng.random(base_travel.shape) < spec.unreachable_fraction
        np.fill_diagonal(mask, False)
        base_travel[mask] = math.inf
    if spec.constraint_regime is ConstraintRegime.INFEASIBLE and spec.problem_type == "tsp":
        base_travel[0, -1] = math.inf
        base_travel[-1, 0] = math.inf
    generous_shift = float(
        np.sum(np.max(np.where(np.isfinite(base_travel), base_travel, 0), axis=1))
    )
    shift_length = max(100.0, generous_shift * 2.0 + float(service_times.sum()))
    if spec.constraint_regime is ConstraintRegime.TIGHT:
        shift_length *= 0.55
    if spec.constraint_regime is ConstraintRegime.INFEASIBLE and spec.problem_type == "tsp":
        shift_length = 1.0
    nodes = gpd.GeoDataFrame(
        {
            "node_id": node_ids,
            "prize": prizes,
            "demand": demands,
            "service_minutes": service_times,
            "window_start": np.zeros(service_count + 1),
            "window_end": np.full(service_count + 1, shift_length),
        },
        geometry=[Point(float(x), float(y)) for x, y in points],
        crs="EPSG:3857",
    )
    node_ref = put_vector(store, nodes, units="people", provenance=_provenance(spec, "route_nodes"))
    travel_ids: dict[str, str] = {}
    for index in range(spec.scenario_count):
        name = "normal" if index == 0 else f"scenario_{index}"
        travel = base_travel * (1.0 + 0.1 * index)
        travel_ids[name] = put_matrix(
            store,
            MatrixData(travel, node_ids, node_ids),
            crs=None,
            units="minutes",
            provenance=_provenance(spec, f"route_travel_{name}"),
        ).id
    capacity = max(
        1.0,
        float(demands.sum()) * (0.5 if spec.constraint_regime is ConstraintRegime.TIGHT else 0.8),
    )
    policy: dict[str, object] = {
        "depot_ids": ["depot"],
        "vehicle_count": 1,
        "shift_length": shift_length,
        "time_units": "minutes",
        "require_return": True,
        "scenario_aggregation": "worst_case" if spec.scenario_count > 1 else "expected",
    }
    if spec.problem_type == "mobile_service_route":
        policy.update({"vehicle_capacity": capacity, "capacity_units": "people"})
    return {
        "type_id": spec.problem_type,
        "nodes_artifact_id": node_ref.id,
        "node_id_field": "node_id",
        "prize_field": "prize",
        "demand_field": "demand",
        "service_time_field": "service_minutes",
        "window_start_field": "window_start",
        "window_end_field": "window_end",
        "travel_matrix_artifact_ids": travel_ids,
        "policy": policy,
    }


async def generate_instance(
    instance_id: str,
    spec: SyntheticInstanceSpec,
    store: ArtifactStore,
) -> GeneratedInstance:
    """Generate immutable evidence and compile it through the normal decision-tool contract."""

    seed = effective_seed(spec)
    rng = np.random.default_rng(seed)
    arguments = (
        _location_inputs(spec, store, rng)
        if spec.family is ProblemFamily.LOCATION_ALLOCATION
        else _route_inputs(spec, store, rng)
    )
    context = ToolContext(
        run_id=f"generate-{instance_id}",
        artifact_store=store,
        deadline_monotonic=time.monotonic() + 120.0,
        cancellation=CancellationToken(),
        seed=seed,
    )
    result = await invoke_tool(
        create_tool_registry(discover_entry_points=False).get("compile_problem"),
        arguments,
        context,
    )
    metrics: Mapping[str, object] = result.metrics
    raw_issue_codes = metrics.get("issue_codes", ())
    if not isinstance(raw_issue_codes, (list, tuple)):
        raise RuntimeError("compile_problem returned malformed issue_codes")
    admitted = result.status is ToolResultStatus.COMPLETE
    if result.status not in {ToolResultStatus.COMPLETE, ToolResultStatus.INFEASIBLE}:
        detail = result.error.message if result.error is not None else str(result.summary)
        raise RuntimeError(f"synthetic instance compilation failed: {detail}")
    return GeneratedInstance(
        id=instance_id,
        generator_spec=spec,
        effective_seed=seed,
        problem_artifact_id=(
            str(metrics["problem_artifact_id"]) if metrics.get("problem_artifact_id") else None
        ),
        baseline_plan_artifact_id=(
            str(metrics["baseline_plan_artifact_id"])
            if metrics.get("baseline_plan_artifact_id")
            else None
        ),
        baseline_scorecard_artifact_id=(
            str(metrics["baseline_scorecard_artifact_id"])
            if metrics.get("baseline_scorecard_artifact_id")
            else None
        ),
        problem_hash=(str(metrics["problem_hash"]) if metrics.get("problem_hash") else None),
        problem_type=spec.problem_type,
        evaluator_version="1.0.0",
        admitted=admitted,
        issue_codes=tuple(str(value) for value in raw_issue_codes),
    )


def split_seeds_are_disjoint(seed: int) -> bool:
    """Expose the held-out separation invariant for contract and regression tests."""

    base = SyntheticInstanceSpec(
        family=ProblemFamily.LOCATION_ALLOCATION,
        problem_type="max_weighted_coverage",
        seed=seed,
        split=DatasetSplit.DEVELOPMENT,
    )
    return effective_seed(base) != effective_seed(
        base.model_copy(update={"split": DatasetSplit.HELD_OUT})
    )
