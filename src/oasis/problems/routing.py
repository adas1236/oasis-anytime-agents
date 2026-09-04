"""Independent validation and scoring for directed mobile-service routing problems."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from oasis.artifacts import (
    ArtifactStore,
    canonical_json_bytes,
    read_matrix,
    read_table,
    read_vector,
)
from oasis.artifacts.protocols import ArtifactStoreError
from oasis.problems.protocols import Deadline
from oasis.problems.schemas import (
    Comparison,
    ResultView,
    RouteProblemType,
    RouteServiceProblem,
    ScenarioAggregation,
    Scorecard,
    SearchStrategy,
    ValidationIssue,
    ValidationReport,
)
from oasis.schemas import ArtifactKind, Plan

FLOAT_TOLERANCE = 1e-9
INFEASIBLE_KEY = (-1e300,)


class RouteDataError(ValueError):
    """Raised when immutable route evidence violates its declared contract."""


@dataclass(frozen=True, slots=True)
class RouteServiceData:
    """Validated numerical view of nodes and scenario-specific route inputs."""

    node_ids: tuple[str, ...]
    prizes: np.ndarray
    demands: np.ndarray
    service_times: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    travel: dict[str, np.ndarray]
    demand_multipliers: dict[str, np.ndarray]
    scenario_weights: dict[str, float]
    depot_indices: tuple[int, ...]
    service_indices: tuple[int, ...]


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def route_problem_hashes(problem: RouteServiceProblem) -> tuple[str, str, str]:
    """Recompute stable evidence, policy, and complete route-problem hashes."""

    evidence_payload = {
        "nodes": {
            "id": problem.nodes.id,
            "kind": problem.nodes.kind.value,
            "crs": problem.nodes.crs,
            "units": problem.nodes.units,
        },
        "fields": {
            "node_id": problem.node_id_field,
            "prize": problem.prize_field,
            "demand": problem.demand_field,
            "service_time": problem.service_time_field,
            "window_start": problem.window_start_field,
            "window_end": problem.window_end_field,
        },
        "travel_scenarios": [
            {
                "name": scenario.name,
                "travel_matrix": scenario.travel_matrix.id,
                "travel_units": scenario.travel_matrix.units,
                "demand_multiplier": (
                    scenario.demand_multiplier.id
                    if scenario.demand_multiplier is not None
                    else None
                ),
                "weight": scenario.weight,
                "directed": scenario.directed,
            }
            for scenario in problem.travel_scenarios
        ],
    }
    evidence_hash = _hash(evidence_payload)
    policy_hash = _hash(problem.policy.model_dump(mode="json"))
    problem_hash = _hash(
        {
            "schema_version": problem.schema_version,
            "type_id": problem.type_id.value,
            "plugin_version": problem.plugin_version,
            "evaluator_version": problem.evaluator_version,
            "evidence_hash": evidence_hash,
            "policy_hash": policy_hash,
        }
    )
    return evidence_hash, policy_hash, problem_hash


def route_plan_hash(plan: Plan) -> str:
    """Hash every route-plan decision field, excluding any claimed score."""

    return _hash(plan.model_dump(mode="json"))


def _issue(code: str, message: str, **context: Any) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, context=context)


def _frame(problem: RouteServiceProblem, store: ArtifactStore) -> pd.DataFrame:
    reference = store.get_metadata(problem.nodes.id)
    if reference.kind is ArtifactKind.VECTOR:
        return read_vector(store, reference)
    if reference.kind is ArtifactKind.TABLE:
        return read_table(store, reference)
    raise RouteDataError("route nodes must be stored as a vector or table artifact")


def _numeric_column(
    frame: pd.DataFrame,
    field: str | None,
    *,
    default: float,
    label: str,
    finite: bool = True,
) -> np.ndarray:
    if field is None:
        return np.full(len(frame), default, dtype=np.float64)
    if field not in frame:
        raise RouteDataError(f"{label} field {field!r} is missing")
    try:
        values = pd.to_numeric(frame[field], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RouteDataError(f"{label} field {field!r} must be numeric") from error
    if np.isnan(values).any() or (values < 0).any() or (finite and not np.isfinite(values).all()):
        raise RouteDataError(f"{label} values must be non-negative and finite")
    return cast(np.ndarray, values)


def load_route_data(problem: RouteServiceProblem, store: ArtifactStore) -> RouteServiceData:
    """Load and cross-check all decision-relevant immutable routing artifacts."""

    node_ref = store.get_metadata(problem.nodes.id)
    if node_ref.kind not in {ArtifactKind.VECTOR, ArtifactKind.TABLE}:
        raise RouteDataError("route nodes must be a vector or table")
    if (node_ref.kind, node_ref.crs, node_ref.units) != (
        problem.nodes.kind,
        problem.nodes.crs,
        problem.nodes.units,
    ):
        raise RouteDataError("committed node metadata does not match the problem reference")
    frame = _frame(problem, store)
    if problem.node_id_field not in frame:
        raise RouteDataError(f"node ID field {problem.node_id_field!r} is missing")
    ids = frame[problem.node_id_field]
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise RouteDataError("route node IDs must be non-missing and unique")
    node_ids = tuple(ids.astype(str))
    if len(node_ids) < 2:
        raise RouteDataError("a route problem requires at least two nodes")
    unknown_depots = sorted(set(problem.policy.depot_ids) - set(node_ids))
    if unknown_depots:
        raise RouteDataError(f"unknown depot IDs: {unknown_depots}")
    depot_indices = tuple(node_ids.index(identifier) for identifier in problem.policy.depot_ids)
    depot_set = set(depot_indices)
    service_indices = tuple(index for index in range(len(node_ids)) if index not in depot_set)

    prizes = _numeric_column(frame, problem.prize_field, default=1.0, label="prize")
    demands = _numeric_column(frame, problem.demand_field, default=0.0, label="demand")
    service_times = _numeric_column(
        frame, problem.service_time_field, default=0.0, label="service time"
    )
    window_starts = _numeric_column(
        frame, problem.window_start_field, default=0.0, label="window start"
    )
    window_ends = _numeric_column(
        frame,
        problem.window_end_field,
        default=math.inf,
        label="window end",
        finite=problem.window_end_field is not None,
    )
    if np.any(window_ends + FLOAT_TOLERANCE < window_starts):
        raise RouteDataError("every time-window end must be at or after its start")
    prizes[list(depot_indices)] = 0.0
    demands[list(depot_indices)] = 0.0
    service_times[list(depot_indices)] = 0.0

    configured_weights = problem.policy.scenario_weights
    scenario_names = {scenario.name for scenario in problem.travel_scenarios}
    if configured_weights and set(configured_weights) != scenario_names:
        raise RouteDataError("configured scenario weights must name every route scenario exactly")
    travel: dict[str, np.ndarray] = {}
    multipliers: dict[str, np.ndarray] = {}
    scenario_weights: dict[str, float] = {}
    for scenario in problem.travel_scenarios:
        reference = store.get_metadata(scenario.travel_matrix.id)
        if reference.kind is not ArtifactKind.MATRIX:
            raise RouteDataError("route travel scenarios must be matrix artifacts")
        if reference.units != scenario.travel_matrix.units:
            raise RouteDataError("committed travel units do not match the problem reference")
        if reference.units != problem.policy.time_units:
            raise RouteDataError("travel and shift-length units must match")
        matrix = read_matrix(store, reference)
        if matrix.row_ids != node_ids or matrix.column_ids != node_ids:
            raise RouteDataError("route matrix labels must exactly match the node IDs")
        if np.isnan(matrix.values).any() or (matrix.values < 0).any():
            raise RouteDataError("route travel values must be non-negative and cannot be NaN")
        if np.any(np.diag(matrix.values) > FLOAT_TOLERANCE):
            raise RouteDataError("route travel matrix diagonal must be zero")
        travel[scenario.name] = matrix.values
        multiplier = np.ones(len(node_ids), dtype=np.float64)
        if scenario.demand_multiplier is not None:
            multiplier_ref = store.get_metadata(scenario.demand_multiplier.id)
            if multiplier_ref.kind is not ArtifactKind.MATRIX:
                raise RouteDataError("demand multipliers must be matrix artifacts")
            if multiplier_ref.units != "unitless":
                raise RouteDataError("demand multipliers must use unitless values")
            multiplier_matrix = read_matrix(store, multiplier_ref)
            if multiplier_matrix.row_ids != node_ids or multiplier_matrix.column_ids != (
                "multiplier",
            ):
                raise RouteDataError("demand multiplier labels must be node IDs by multiplier")
            multiplier = multiplier_matrix.values[:, 0].copy()
            if not np.isfinite(multiplier).all() or (multiplier < 0).any():
                raise RouteDataError("demand multipliers must be finite and non-negative")
        multiplier[list(depot_indices)] = 0.0
        multipliers[scenario.name] = multiplier
        scenario_weights[scenario.name] = configured_weights.get(scenario.name, scenario.weight)
    return RouteServiceData(
        node_ids=node_ids,
        prizes=prizes,
        demands=demands,
        service_times=service_times,
        window_starts=window_starts,
        window_ends=window_ends,
        travel=travel,
        demand_multipliers=multipliers,
        scenario_weights=scenario_weights,
        depot_indices=depot_indices,
        service_indices=service_indices,
    )


def vehicle_depot_index(data: RouteServiceData, vehicle_index: int) -> int:
    """Return the shared or vehicle-specific depot index."""

    return (
        data.depot_indices[0] if len(data.depot_indices) == 1 else data.depot_indices[vehicle_index]
    )


def route_node_sequences(plan: Plan) -> tuple[tuple[str, ...], ...]:
    """Read route node sequences conservatively from the neutral plan envelope."""

    sequences: list[tuple[str, ...]] = []
    for record in plan.routes:
        raw = record.get("node_ids")
        if not isinstance(raw, (list, tuple)) or any(not isinstance(value, str) for value in raw):
            sequences.append(())
        else:
            sequences.append(tuple(cast(list[str] | tuple[str, ...], raw)))
    return tuple(sequences)


def make_route_plan(
    problem: RouteServiceProblem,
    data: RouteServiceData,
    routes: tuple[tuple[int, ...], ...],
    *,
    strategy: str,
) -> Plan:
    """Create the canonical problem-neutral plan encoding for indexed vehicle routes."""

    records: tuple[dict[str, Any], ...] = tuple(
        {
            "vehicle_id": f"vehicle-{index + 1}",
            "node_ids": [data.node_ids[node] for node in route],
        }
        for index, route in enumerate(routes)
    )
    return Plan(problem_type=problem.type_id.value, routes=records, metadata={"strategy": strategy})


def indexed_routes(plan: Plan, data: RouteServiceData) -> tuple[tuple[int, ...], ...]:
    """Convert known node IDs to indices; unknown IDs are omitted only for safe diagnostics."""

    by_id = {identifier: index for index, identifier in enumerate(data.node_ids)}
    return tuple(
        tuple(by_id[value] for value in route if value in by_id)
        for route in route_node_sequences(plan)
    )


def _simulate_route(
    route: tuple[int, ...], data: RouteServiceData, travel: np.ndarray
) -> tuple[float, bool, bool]:
    elapsed = 0.0
    reachable = True
    windows_met = True
    for left, right in itertools.pairwise(route):
        arc = float(travel[left, right])
        if not math.isfinite(arc):
            reachable = False
            break
        elapsed += arc
        elapsed = max(elapsed, float(data.window_starts[right]))
        if elapsed > float(data.window_ends[right]) + FLOAT_TOLERANCE:
            windows_met = False
        elapsed += float(data.service_times[right])
    return elapsed, reachable, windows_met


def _aggregate(
    values: dict[str, float],
    problem: RouteServiceProblem,
    data: RouteServiceData,
    *,
    worst: Callable[[Iterable[float]], float] = max,
) -> float:
    if problem.policy.scenario_aggregation is ScenarioAggregation.WORST_CASE:
        return float(worst(values.values()))
    total = sum(data.scenario_weights.values())
    return sum(values[name] * data.scenario_weights[name] for name in values) / total


class RouteServicePlugin:
    """Shared protocol implementation for TSP, orienteering, and mobile-service routing."""

    version = "1.0.0"

    def __init__(self, type_id: RouteProblemType) -> None:
        self.problem_type = type_id
        self.type_id = type_id.value

    def validate_spec(self, spec: object, store: ArtifactStore) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if not isinstance(spec, RouteServiceProblem):
            return ValidationReport(
                valid=False,
                issues=(_issue("wrong_schema", "expected a RouteServiceProblem"),),
            )
        if spec.type_id is not self.problem_type:
            issues.append(_issue("wrong_plugin", "route problem type does not match plugin"))
        if route_problem_hashes(spec) != (
            spec.evidence_hash,
            spec.policy_hash,
            spec.problem_hash,
        ):
            issues.append(_issue("hash_mismatch", "problem hashes do not match immutable inputs"))
        try:
            data = load_route_data(spec, store)
        except (ArtifactStoreError, RouteDataError, ValueError, KeyError) as error:
            issues.append(_issue("invalid_evidence", str(error)))
            return ValidationReport(valid=False, issues=tuple(issues))
        if not all(scenario.directed for scenario in spec.travel_scenarios):
            issues.append(_issue("undirected_matrix", "route travel scenarios must be directed"))
        if self.problem_type in {RouteProblemType.ORIENTEERING, RouteProblemType.MOBILE_SERVICE}:
            if spec.prize_field is None:
                issues.append(_issue("missing_prize", "route coverage objectives require prizes"))
            elif float(data.prizes.sum()) <= FLOAT_TOLERANCE:
                issues.append(_issue("zero_prize", "total non-depot prize must be positive"))
        if self.problem_type is RouteProblemType.MOBILE_SERVICE:
            if spec.demand_field is None:
                issues.append(_issue("missing_demand", "mobile-service routing requires demand"))
            elif float(data.demands.sum()) <= FLOAT_TOLERANCE:
                issues.append(_issue("zero_demand", "total non-depot demand must be positive"))
            if spec.policy.vehicle_capacity is None:
                issues.append(
                    _issue("missing_capacity", "mobile-service routing requires vehicle capacity")
                )
        if spec.policy.vehicle_capacity is not None and spec.demand_field is None:
            issues.append(_issue("missing_demand", "vehicle capacity requires a demand field"))
        if not data.service_indices:
            issues.append(_issue("no_service_nodes", "at least one non-depot node is required"))
        return ValidationReport(valid=not issues, issues=tuple(issues))

    def make_baseline(self, spec: object, store: ArtifactStore, deadline: Deadline) -> Plan:
        if not isinstance(spec, RouteServiceProblem):
            raise RouteDataError("expected a RouteServiceProblem")
        report = self.validate_spec(spec, store)
        if not report.valid:
            raise RouteDataError(report.issues[0].message)
        from oasis.problems.routing_search import make_route_baseline

        return make_route_baseline(self, spec, store, deadline)

    def _structural_issues(
        self,
        problem: RouteServiceProblem,
        plan: Plan,
        data: RouteServiceData,
        *,
        require_all: bool = True,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if plan.problem_type != problem.type_id.value:
            issues.append(_issue("wrong_problem_type", "plan problem type does not match"))
        if plan.selected_site_ids or plan.assignments or plan.allocations or plan.schedules:
            issues.append(
                _issue("unsupported_fields", "route plans may contain only route records")
            )
        if len(plan.routes) > problem.policy.vehicle_count:
            issues.append(_issue("vehicle_count", "plan exceeds the configured vehicle count"))
        expected_vehicles = {
            f"vehicle-{index + 1}" for index in range(problem.policy.vehicle_count)
        }
        seen_vehicles: set[str] = set()
        seen_service: set[int] = set()
        by_id = {identifier: index for index, identifier in enumerate(data.node_ids)}
        for record in plan.routes:
            if set(record) != {"vehicle_id", "node_ids"}:
                issues.append(
                    _issue(
                        "invalid_route_fields",
                        "route records require exactly vehicle_id and node_ids",
                    )
                )
            vehicle_id = record.get("vehicle_id")
            if not isinstance(vehicle_id, str) or vehicle_id not in expected_vehicles:
                issues.append(_issue("unknown_vehicle", "route has an unknown vehicle ID"))
                continue
            if vehicle_id in seen_vehicles:
                issues.append(_issue("duplicate_vehicle", "vehicle IDs must be unique"))
            seen_vehicles.add(vehicle_id)
            vehicle_index = int(vehicle_id.removeprefix("vehicle-")) - 1
            raw_nodes = record.get("node_ids")
            if not isinstance(raw_nodes, (list, tuple)) or any(
                not isinstance(value, str) for value in raw_nodes
            ):
                issues.append(_issue("invalid_route_nodes", "node_ids must be a string array"))
                continue
            node_ids = tuple(cast(list[str] | tuple[str, ...], raw_nodes))
            unknown = sorted({value for value in node_ids if value not in by_id})
            if unknown:
                issues.append(_issue("unknown_node", "route contains unknown nodes", nodes=unknown))
                continue
            route = tuple(by_id[value] for value in node_ids)
            depot = vehicle_depot_index(data, vehicle_index)
            minimum_length = 2 if problem.policy.require_return else 1
            if len(route) < minimum_length or not route or route[0] != depot:
                issues.append(_issue("depot_start", "every route must start at its vehicle depot"))
                continue
            if problem.policy.require_return and route[-1] != depot:
                issues.append(_issue("depot_return", "every route must return to its depot"))
            interior = route[1:-1] if problem.policy.require_return else route[1:]
            if any(node in data.depot_indices for node in interior):
                issues.append(_issue("interior_depot", "depots may appear only at route endpoints"))
            duplicate = seen_service.intersection(interior)
            if duplicate:
                issues.append(
                    _issue(
                        "duplicate_visit",
                        "service nodes may be visited only once across all vehicles",
                        nodes=sorted(data.node_ids[node] for node in duplicate),
                    )
                )
            if len(interior) != len(set(interior)):
                issues.append(_issue("duplicate_visit", "a route repeats a service node"))
            seen_service.update(interior)
            load = float(data.demands[list(interior)].sum()) if interior else 0.0
            capacity = problem.policy.vehicle_capacity
            if capacity is not None and load > capacity + FLOAT_TOLERANCE:
                issues.append(_issue("capacity_exceeded", "route demand exceeds vehicle capacity"))
            for name, travel in data.travel.items():
                duration, reachable, windows_met = _simulate_route(route, data, travel)
                if not reachable:
                    issues.append(
                        _issue("unreachable_arc", f"route uses an unreachable arc in {name}")
                    )
                if not windows_met:
                    issues.append(
                        _issue("time_window", f"route violates a service window in {name}")
                    )
                if duration > problem.policy.shift_length + FLOAT_TOLERANCE:
                    issues.append(_issue("shift_length", f"route exceeds shift length in {name}"))
        if require_all and self.problem_type is RouteProblemType.TSP:
            missing = set(data.service_indices) - seen_service
            if missing:
                issues.append(
                    _issue(
                        "missing_required_visit",
                        "TSP routes must visit every non-depot node",
                        nodes=sorted(data.node_ids[node] for node in missing),
                    )
                )
        return issues

    def validate_plan(self, spec: object, plan: Plan, store: ArtifactStore) -> ValidationReport:
        if not isinstance(spec, RouteServiceProblem):
            return ValidationReport(
                valid=False,
                issues=(_issue("wrong_schema", "expected a RouteServiceProblem"),),
            )
        try:
            data = load_route_data(spec, store)
        except (ArtifactStoreError, RouteDataError, ValueError, KeyError) as error:
            return ValidationReport(valid=False, issues=(_issue("invalid_evidence", str(error)),))
        issues = self._structural_issues(spec, plan, data)
        return ValidationReport(valid=not issues, issues=tuple(issues))

    def partial_plan_is_feasible(
        self, problem: RouteServiceProblem, plan: Plan, data: RouteServiceData
    ) -> bool:
        """Check insertion feasibility while allowing an incomplete TSP construction."""

        return not self._structural_issues(problem, plan, data, require_all=False)

    def _metrics(
        self, problem: RouteServiceProblem, plan: Plan, data: RouteServiceData
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        routes = indexed_routes(plan, data)
        visited = {
            node
            for route in routes
            for node in (route[1:-1] if problem.policy.require_return else route[1:])
            if node in data.service_indices
        }
        scenario_metrics: dict[str, dict[str, float]] = {}
        travel_values: dict[str, float] = {}
        max_route_values: dict[str, float] = {}
        coverage_values: dict[str, float] = {}
        served_values: dict[str, float] = {}
        for name, travel in data.travel.items():
            durations = [_simulate_route(route, data, travel)[0] for route in routes]
            multiplier = data.demand_multipliers[name]
            base = (
                data.demands
                if self.problem_type is RouteProblemType.MOBILE_SERVICE
                else data.prizes
            )
            weighted = base * multiplier
            total = float(weighted[list(data.service_indices)].sum())
            served = float(weighted[list(visited)].sum()) if visited else 0.0
            coverage = served / total if total > FLOAT_TOLERANCE else 1.0
            total_travel = float(sum(durations))
            maximum_route = max(durations, default=0.0)
            scenario_metrics[name] = {
                "coverage": coverage,
                "served_value": served,
                "total_value": total,
                "total_route_time": total_travel,
                "maximum_route_time": maximum_route,
            }
            coverage_values[name] = coverage
            served_values[name] = served
            travel_values[name] = total_travel
            max_route_values[name] = maximum_route
        overall = {
            "coverage": _aggregate(coverage_values, problem, data, worst=min),
            "served_value": _aggregate(served_values, problem, data, worst=min),
            "total_route_time": _aggregate(travel_values, problem, data, worst=max),
            "maximum_route_time": _aggregate(max_route_values, problem, data, worst=max),
            "visited_nodes": float(len(visited)),
            "vehicle_count": float(len(routes)),
        }
        return overall, scenario_metrics

    def measure(self, spec: object, plan: Plan, store: ArtifactStore) -> Scorecard:
        if not isinstance(spec, RouteServiceProblem):
            raise RouteDataError("expected a RouteServiceProblem")
        data = load_route_data(spec, store)
        issues = self._structural_issues(spec, plan, data)
        if issues:
            return Scorecard(
                feasible=False,
                violations=tuple(issues),
                comparator_key=INFEASIBLE_KEY,
                problem_hash=spec.problem_hash,
                evidence_hash=spec.evidence_hash,
                policy_hash=spec.policy_hash,
                evaluator_version=spec.evaluator_version,
                plan_hash=route_plan_hash(plan),
            )
        overall, scenarios = self._metrics(spec, plan, data)
        comparator: tuple[float, ...]
        if self.problem_type is RouteProblemType.TSP:
            raw = {"total_route_time": overall["total_route_time"]}
            comparator = (-overall["total_route_time"], -overall["maximum_route_time"])
        elif self.problem_type is RouteProblemType.ORIENTEERING:
            raw = {
                "prize_coverage": overall["coverage"],
                "collected_prize": overall["served_value"],
            }
            comparator = (
                overall["coverage"],
                overall["served_value"],
                -overall["total_route_time"],
            )
        else:
            raw = {
                "demand_coverage": overall["coverage"],
                "served_demand": overall["served_value"],
            }
            comparator = (
                overall["coverage"],
                overall["served_value"],
                -overall["total_route_time"],
            )
        return Scorecard(
            feasible=True,
            raw_objective=raw,
            comparator_key=comparator,
            overall_metrics=overall,
            scenario_metrics=scenarios,
            problem_hash=spec.problem_hash,
            evidence_hash=spec.evidence_hash,
            policy_hash=spec.policy_hash,
            evaluator_version=spec.evaluator_version,
            plan_hash=route_plan_hash(plan),
            assumptions=("routes are evaluated against every immutable scenario",),
        )

    def compare(self, left: Scorecard, right: Scorecard) -> Comparison:
        if left.problem_hash != right.problem_hash:
            raise ValueError("cannot compare scorecards from different immutable problems")
        if left.feasible != right.feasible:
            return Comparison.BETTER if left.feasible else Comparison.WORSE
        for left_value, right_value in zip(left.comparator_key, right.comparator_key, strict=True):
            if left_value > right_value + FLOAT_TOLERANCE:
                return Comparison.BETTER
            if left_value < right_value - FLOAT_TOLERANCE:
                return Comparison.WORSE
        return Comparison.EQUAL

    def fallback_actions(self) -> tuple[SearchStrategy, ...]:
        return (
            SearchStrategy.RELOCATE,
            SearchStrategy.TWO_OPT,
            SearchStrategy.SWAP,
            SearchStrategy.ORTOOLS_ROUTING,
            SearchStrategy.EXACT_ENUMERATION,
        )

    def render_result(self, spec: object, plan: Plan, scorecard: Scorecard) -> ResultView:
        if not isinstance(spec, RouteServiceProblem):
            raise RouteDataError("expected a RouteServiceProblem")
        if scorecard.problem_hash != spec.problem_hash:
            raise ValueError("scorecard does not belong to this immutable problem")
        if scorecard.plan_hash != route_plan_hash(plan):
            raise ValueError("scorecard does not belong to this plan")
        primary_name = next(iter(scorecard.raw_objective), None)
        return ResultView(
            problem_type=spec.type_id.value,
            feasible=scorecard.feasible,
            selected_site_ids=(),
            route_count=len(plan.routes),
            primary_metric_name=primary_name,
            primary_metric_value=(
                scorecard.raw_objective[primary_name] if primary_name is not None else None
            ),
            overall_metrics=scorecard.overall_metrics,
            scenario_metrics=scorecard.scenario_metrics,
            violations=scorecard.violations,
            warnings=scorecard.warnings,
        )
