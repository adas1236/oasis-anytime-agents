"""Independent validation and scoring for location-allocation problem families."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, cast

import geopandas as gpd
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
from oasis.problems.registry import ProblemRegistry
from oasis.problems.schemas import (
    Comparison,
    EquityGroup,
    EquityObjective,
    LocationAllocationProblem,
    LocationProblemType,
    ResultView,
    ScenarioAggregation,
    Scorecard,
    SearchStrategy,
    ValidationIssue,
    ValidationReport,
)
from oasis.schemas import ArtifactKind, Plan

FLOAT_TOLERANCE = 1e-9
INFEASIBLE_KEY = (-1e300,)


class ProblemDataError(ValueError):
    """Raised when immutable problem evidence violates its declared contract."""


@dataclass(frozen=True, slots=True)
class LocationAllocationData:
    """Validated numerical view loaded from immutable problem artifacts."""

    demand_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    weights: np.ndarray
    access: np.ndarray
    services: dict[str, np.ndarray]
    scenario_weights: dict[str, float]
    costs: np.ndarray
    capacities: np.ndarray
    eligible: np.ndarray
    existing: np.ndarray
    group_membership: dict[str, np.ndarray]
    candidates_frame: gpd.GeoDataFrame


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def problem_hashes(problem: LocationAllocationProblem) -> tuple[str, str, str]:
    """Recompute stable evidence, policy, and complete problem hashes."""

    evidence_payload = {
        "demand_artifact": {
            "id": problem.demand.artifact.id,
            "kind": problem.demand.artifact.kind.value,
            "crs": problem.demand.artifact.crs,
            "units": problem.demand.artifact.units,
        },
        "demand_fields": {
            "location_id": problem.demand.location_id_field,
            "need": list(problem.demand.need_fields),
            "groups": list(problem.demand.group_fields),
            "times": list(problem.demand.time_fields),
            "suppression": list(problem.demand.suppression_fields),
            "missing_data_policy": problem.demand.missing_data_policy.value,
            "weighting_rules": problem.demand.weighting_rules,
        },
        "candidate_artifact": {
            "id": problem.candidates.artifact.id,
            "kind": problem.candidates.artifact.kind.value,
            "crs": problem.candidates.artifact.crs,
            "units": problem.candidates.artifact.units,
        },
        "candidate_fields": {
            "id": problem.candidates.candidate_id_field,
            "cost": problem.candidates.opening_cost_field,
            "capacity": problem.candidates.capacity_field,
            "eligibility": problem.candidates.eligibility_field,
            "existing": problem.candidates.existing_site_field,
            "minimum_spacing": problem.candidates.minimum_spacing,
            "spacing_units": problem.candidates.spacing_units,
        },
        "access_matrix": {
            "id": problem.access_matrix.id,
            "units": problem.access_matrix.units,
        },
        "service_scenarios": [
            {
                "name": scenario.name,
                "artifact_id": scenario.service_matrix.id,
                "weight": scenario.weight,
                "units": scenario.service_matrix.units,
            }
            for scenario in problem.service_scenarios
        ],
        "need_field": problem.need_field,
        "groups": [group.model_dump(mode="json") for group in problem.groups],
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


def plan_hash(plan: Plan) -> str:
    """Hash every decision field, including allocations, but never claimed metrics."""

    return _hash(plan.model_dump(mode="json"))


def _frame(store: ArtifactStore, kind: ArtifactKind, artifact_id: str) -> pd.DataFrame:
    if kind is ArtifactKind.VECTOR:
        return read_vector(store, artifact_id)
    if kind is ArtifactKind.TABLE:
        return read_table(store, artifact_id)
    raise ProblemDataError(f"expected vector or table artifact, received {kind.value}")


def _boolean_column(frame: pd.DataFrame, field: str | None, *, default: bool) -> np.ndarray:
    if field is None:
        return np.full(len(frame), default, dtype=np.bool_)
    if field not in frame:
        raise ProblemDataError(f"candidate field {field!r} is missing")
    values = frame[field]
    if values.isna().any():
        raise ProblemDataError(f"candidate field {field!r} cannot contain missing values")
    return cast(np.ndarray, values.astype(bool).to_numpy(dtype=np.bool_))


def _numeric_column(
    frame: pd.DataFrame,
    field: str | None,
    *,
    default: float,
    label: str,
) -> np.ndarray:
    if field is None:
        return np.full(len(frame), default, dtype=np.float64)
    if field not in frame:
        raise ProblemDataError(f"{label} field {field!r} is missing")
    try:
        values = pd.to_numeric(frame[field], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ProblemDataError(f"{label} field {field!r} must be numeric") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ProblemDataError(f"{label} values must be finite and non-negative")
    return cast(np.ndarray, values)


def _group_membership(frame: pd.DataFrame, group: EquityGroup) -> np.ndarray:
    if group.field not in frame:
        raise ProblemDataError(f"group field {group.field!r} is missing")
    column = frame[group.field]
    if group.match_value is not None:
        membership = (column == group.match_value).to_numpy(dtype=np.float64)
    elif pd.api.types.is_bool_dtype(column.dtype):
        if column.isna().any():
            raise ProblemDataError(f"group field {group.field!r} cannot contain missing values")
        membership = column.astype(float).to_numpy(dtype=np.float64)
    else:
        try:
            membership = pd.to_numeric(column, errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ProblemDataError(
                f"group {group.name!r} needs match_value or numeric membership values"
            ) from error
        if not np.isfinite(membership).all() or (membership < 0).any() or (membership > 1).any():
            raise ProblemDataError("numeric group membership values must lie between zero and one")
    return cast(np.ndarray, membership)


def _allocation_amount(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("allocation amount must be numeric")
    return float(value)


def load_problem_data(
    problem: LocationAllocationProblem, store: ArtifactStore
) -> LocationAllocationData:
    """Load and cross-check every decision-relevant immutable artifact."""

    demand_ref = store.get_metadata(problem.demand.artifact.id)
    candidate_ref = store.get_metadata(problem.candidates.artifact.id)
    access_ref = store.get_metadata(problem.access_matrix.id)
    if demand_ref.kind not in {ArtifactKind.VECTOR, ArtifactKind.TABLE}:
        raise ProblemDataError("demand artifact must be a vector or table")
    if candidate_ref.kind is not ArtifactKind.VECTOR:
        raise ProblemDataError("candidate artifact must be a vector")
    if access_ref.kind is not ArtifactKind.MATRIX:
        raise ProblemDataError("access artifact must be a matrix")
    if (demand_ref.kind, demand_ref.crs, demand_ref.units) != (
        problem.demand.artifact.kind,
        problem.demand.artifact.crs,
        problem.demand.artifact.units,
    ):
        raise ProblemDataError("committed demand metadata does not match the problem reference")
    if (candidate_ref.kind, candidate_ref.crs, candidate_ref.units) != (
        problem.candidates.artifact.kind,
        problem.candidates.artifact.crs,
        problem.candidates.artifact.units,
    ):
        raise ProblemDataError("committed candidate metadata does not match the problem reference")
    if access_ref.units != problem.access_matrix.units:
        raise ProblemDataError("committed access units do not match the problem reference")
    if not access_ref.units or access_ref.units == "unitless":
        raise ProblemDataError("access matrix must declare distance or time units")
    if (
        problem.candidates.minimum_spacing is not None
        and candidate_ref.units != problem.candidates.spacing_units
    ):
        raise ProblemDataError("candidate spacing units must match the candidate artifact units")
    if (
        demand_ref.id != problem.demand.artifact.id
        or candidate_ref.id != problem.candidates.artifact.id
    ):
        raise ProblemDataError("problem references do not match committed artifact identities")

    demand = _frame(store, demand_ref.kind, demand_ref.id)
    candidate_frame = read_vector(store, candidate_ref.id)
    if candidate_frame.geometry.is_empty.any() or candidate_frame.geometry.isna().any():
        raise ProblemDataError("candidate geometries must be non-empty points")
    if not candidate_frame.geometry.geom_type.eq("Point").all():
        raise ProblemDataError("candidate geometries must be points")
    demand_id_field = problem.demand.location_id_field
    candidate_id_field = problem.candidates.candidate_id_field
    for frame, field, label in (
        (demand, demand_id_field, "demand"),
        (candidate_frame, candidate_id_field, "candidate"),
    ):
        if field not in frame:
            raise ProblemDataError(f"{label} ID field {field!r} is missing")
        if frame[field].isna().any() or frame[field].astype(str).duplicated().any():
            raise ProblemDataError(f"{label} IDs must be non-missing and unique")
    demand_ids = tuple(demand[demand_id_field].astype(str))
    candidate_ids = tuple(candidate_frame[candidate_id_field].astype(str))

    if problem.need_field not in problem.demand.need_fields:
        raise ProblemDataError("selected need field is not declared by DemandSpec")
    weights = _numeric_column(demand, problem.need_field, default=0.0, label="demand need")
    if float(weights.sum()) <= FLOAT_TOLERANCE:
        raise ProblemDataError("total demand need must be positive")

    access = read_matrix(store, access_ref)
    if access.row_ids != demand_ids or access.column_ids != candidate_ids:
        raise ProblemDataError("access matrix labels must exactly match demand and candidate IDs")
    if np.isnan(access.values).any() or (access.values < 0).any():
        raise ProblemDataError("access values must be non-negative and cannot be NaN")

    services: dict[str, np.ndarray] = {}
    scenario_weights: dict[str, float] = {}
    configured_weights = problem.policy.scenario_weights
    scenario_names = {scenario.name for scenario in problem.service_scenarios}
    if configured_weights and set(configured_weights) != scenario_names:
        raise ProblemDataError(
            "configured scenario weights must name every service scenario exactly"
        )
    for scenario in problem.service_scenarios:
        service_ref = store.get_metadata(scenario.service_matrix.id)
        if service_ref.kind is not ArtifactKind.MATRIX:
            raise ProblemDataError("service scenario artifacts must be matrices")
        if service_ref.units != scenario.service_matrix.units:
            raise ProblemDataError("committed service units do not match the problem reference")
        if service_ref.units != "unitless":
            raise ProblemDataError("service matrices must use unitless benefit values")
        matrix = read_matrix(store, service_ref)
        if matrix.row_ids != demand_ids or matrix.column_ids != candidate_ids:
            raise ProblemDataError(
                f"service scenario {scenario.name!r} labels do not match demand/candidates"
            )
        if (
            not np.isfinite(matrix.values).all()
            or (matrix.values < 0).any()
            or (matrix.values > 1).any()
        ):
            raise ProblemDataError("service benefits must be finite and between zero and one")
        services[scenario.name] = matrix.values
        scenario_weights[scenario.name] = configured_weights.get(scenario.name, scenario.weight)

    costs = _numeric_column(
        candidate_frame,
        problem.candidates.opening_cost_field,
        default=0.0,
        label="opening cost",
    )
    capacities = _numeric_column(
        candidate_frame,
        problem.candidates.capacity_field,
        default=math.inf,
        label="capacity",
    )
    eligible = _boolean_column(candidate_frame, problem.candidates.eligibility_field, default=True)
    existing = _boolean_column(
        candidate_frame, problem.candidates.existing_site_field, default=False
    )
    group_membership = {group.name: _group_membership(demand, group) for group in problem.groups}
    for name, membership in group_membership.items():
        if float(np.dot(weights, membership)) <= FLOAT_TOLERANCE:
            raise ProblemDataError(f"group {name!r} has no positive weighted demand")
    return LocationAllocationData(
        demand_ids=demand_ids,
        candidate_ids=candidate_ids,
        weights=weights,
        access=access.values,
        services=services,
        scenario_weights=scenario_weights,
        costs=costs,
        capacities=capacities,
        eligible=eligible,
        existing=existing,
        group_membership=group_membership,
        candidates_frame=candidate_frame,
    )


def _issue(code: str, message: str, **context: Any) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, context=context)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(values, weights) / weights.sum())


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile * float(weights.sum()), side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _selected_indices(plan: Plan, data: LocationAllocationData) -> tuple[int, ...]:
    by_id = {candidate_id: index for index, candidate_id in enumerate(data.candidate_ids)}
    return tuple(
        by_id[candidate_id] for candidate_id in plan.selected_site_ids if candidate_id in by_id
    )


def _coverage_by_scenario(
    selected: tuple[int, ...], data: LocationAllocationData
) -> dict[str, np.ndarray]:
    if not selected:
        return {name: np.zeros(len(data.demand_ids)) for name in data.services}
    columns = np.array(selected, dtype=np.int64)
    return {name: values[:, columns].max(axis=1) for name, values in data.services.items()}


def _aggregate_scenarios(
    values: dict[str, float], problem: LocationAllocationProblem, data: LocationAllocationData
) -> float:
    if problem.policy.scenario_aggregation is ScenarioAggregation.WORST_CASE:
        return min(values.values())
    total_weight = sum(data.scenario_weights.values())
    return sum(values[name] * data.scenario_weights[name] for name in values) / total_weight


def _plan_cost(selected: tuple[int, ...], data: LocationAllocationData) -> float:
    return float(sum(data.costs[index] for index in selected if not data.existing[index]))


def _coverage_metrics(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    selected: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    per_demand = _coverage_by_scenario(selected, data)
    scenario_coverage = {
        name: _weighted_mean(coverage, data.weights) for name, coverage in per_demand.items()
    }
    overall = _aggregate_scenarios(scenario_coverage, problem, data)
    scenario_metrics = {
        name: {"coverage": coverage} for name, coverage in scenario_coverage.items()
    }
    group_metrics: dict[str, dict[str, float]] = {}
    for name, membership in data.group_membership.items():
        group_weights = data.weights * membership
        by_scenario = {
            scenario: _weighted_mean(coverage, group_weights)
            for scenario, coverage in per_demand.items()
        }
        group_metrics[name] = {
            "coverage": _aggregate_scenarios(by_scenario, problem, data),
            "demand": float(group_weights.sum()),
        }
    return {"coverage": overall}, group_metrics, scenario_metrics


def _allocation_totals(
    plan: Plan, data: LocationAllocationData
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    demand_index = {identifier: index for index, identifier in enumerate(data.demand_ids)}
    site_index = {identifier: index for index, identifier in enumerate(data.candidate_ids)}
    served = np.zeros(len(data.demand_ids), dtype=np.float64)
    used = np.zeros(len(data.candidate_ids), dtype=np.float64)
    demand_access = np.zeros(len(data.demand_ids), dtype=np.float64)
    weighted_access = 0.0
    for allocation in plan.allocations:
        demand_id = str(allocation.get("demand_id", ""))
        site_id = str(allocation.get("site_id", ""))
        try:
            amount = _allocation_amount(allocation.get("amount", 0.0))
        except (TypeError, ValueError):
            continue
        if demand_id not in demand_index or site_id not in site_index or not math.isfinite(amount):
            continue
        row = demand_index[demand_id]
        column = site_index[site_id]
        served[row] += amount
        used[column] += amount
        arc_access = amount * float(data.access[row, column])
        demand_access[row] += arc_access
        weighted_access += arc_access
    return served, used, weighted_access, demand_access


class LocationAllocationPlugin:
    """One family-specific facade over the shared independent evaluator."""

    version = "1.0.0"

    def __init__(self, type_id: LocationProblemType) -> None:
        self.problem_type = type_id
        self.type_id = type_id.value

    def validate_spec(self, spec: object, store: ArtifactStore) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if not isinstance(spec, LocationAllocationProblem):
            return ValidationReport(
                valid=False,
                issues=(_issue("wrong_schema", "expected a LocationAllocationProblem"),),
            )
        if spec.type_id is not self.problem_type:
            issues.append(_issue("wrong_plugin", "problem type does not match plugin"))
        expected = problem_hashes(spec)
        if expected != (spec.evidence_hash, spec.policy_hash, spec.problem_hash):
            issues.append(_issue("hash_mismatch", "problem hashes do not match immutable inputs"))
        try:
            data = load_problem_data(spec, store)
        except (ArtifactStoreError, ProblemDataError, ValueError, KeyError) as error:
            issues.append(_issue("invalid_evidence", str(error)))
            return ValidationReport(valid=False, issues=tuple(issues))
        policy = spec.policy
        if (
            policy.site_limit is None
            and policy.new_site_limit is None
            and policy.financial_budget is None
        ):
            issues.append(_issue("missing_budget", "a site limit or financial budget is required"))
        if policy.financial_budget is not None and spec.candidates.opening_cost_field is None:
            issues.append(_issue("missing_cost", "financial budget requires an opening-cost field"))
        if policy.group_floors and not set(policy.group_floors) <= set(data.group_membership):
            issues.append(_issue("unknown_group_floor", "group floors name undeclared groups"))
        if policy.equity_objective is not EquityObjective.NONE and not data.group_membership:
            issues.append(_issue("missing_groups", "equity objectives require named groups"))
        if (
            policy.equity_objective is EquityObjective.MAX_MIN
            and self.problem_type is not LocationProblemType.EQUITY_COVERAGE
        ):
            issues.append(
                _issue("unsupported_max_min", "max-min is supported by equity coverage only")
            )
        if self.problem_type in {
            LocationProblemType.WEIGHTED_P_MEDIAN,
            LocationProblemType.P_CENTER,
            LocationProblemType.QUANTILE_ACCESS,
        } and (
            policy.coverage_target is not None
            or policy.group_floors
            or policy.equity_objective is not EquityObjective.NONE
        ):
            issues.append(
                _issue(
                    "coverage_policy_on_access_problem",
                    "coverage targets and floors do not apply to access objectives",
                )
            )
        if (
            policy.new_site_limit is not None
            and self.problem_type is not LocationProblemType.INCREMENTAL_COVERAGE
        ):
            issues.append(
                _issue(
                    "unsupported_new_site_limit",
                    "new_site_limit applies only to incremental siting",
                )
            )
        if self.problem_type is LocationProblemType.MIN_COST_TARGET_COVERAGE:
            if policy.coverage_target is None:
                issues.append(_issue("missing_target", "minimum-cost coverage requires a target"))
            if spec.candidates.opening_cost_field is None:
                issues.append(
                    _issue("missing_cost", "minimum-cost coverage requires opening costs")
                )
        if (
            self.problem_type
            in {
                LocationProblemType.WEIGHTED_P_MEDIAN,
                LocationProblemType.P_CENTER,
                LocationProblemType.QUANTILE_ACCESS,
            }
            and policy.site_limit is None
        ):
            issues.append(_issue("missing_site_limit", "access objectives require a site limit"))
        if self.problem_type is LocationProblemType.CAPACITATED_ALLOCATION:
            if spec.candidates.capacity_field is None:
                issues.append(
                    _issue("missing_capacity", "capacitated allocation requires capacities")
                )
        if self.problem_type is LocationProblemType.EQUITY_COVERAGE:
            if policy.equity_objective is EquityObjective.NONE:
                issues.append(
                    _issue("missing_equity_policy", "equity coverage needs floors or max-min")
                )
        if self.problem_type is LocationProblemType.INCREMENTAL_COVERAGE:
            if spec.candidates.existing_site_field is None:
                issues.append(
                    _issue("missing_existing_sites", "incremental siting needs existing sites")
                )
            if policy.new_site_limit is None:
                issues.append(
                    _issue("missing_new_site_limit", "incremental siting needs new_site_limit")
                )
        if self.problem_type is LocationProblemType.RESILIENT_COVERAGE:
            limit = policy.site_limit if policy.site_limit is not None else int(data.eligible.sum())
            if limit < policy.redundancy:
                issues.append(
                    _issue(
                        "insufficient_sites", "resilient coverage needs at least redundancy sites"
                    )
                )
        if np.any(data.existing & ~data.eligible):
            issues.append(_issue("ineligible_existing", "existing sites must be eligible"))
        existing_count = int(data.existing.sum())
        if policy.site_limit is not None and existing_count > policy.site_limit:
            issues.append(_issue("existing_over_limit", "existing sites exceed the site limit"))
        if not data.eligible.any():
            issues.append(_issue("no_candidates", "at least one eligible candidate is required"))
        if self.problem_type in {
            LocationProblemType.WEIGHTED_P_MEDIAN,
            LocationProblemType.P_CENTER,
            LocationProblemType.QUANTILE_ACCESS,
        }:
            usable = data.eligible | data.existing
            unreachable_rows = np.isinf(data.access[:, usable]).all(axis=1)
            if np.any(unreachable_rows & (data.weights > 0)):
                issues.append(
                    _issue("unreachable_demand", "positive demand lacks any reachable site")
                )
        return ValidationReport(valid=not issues, issues=tuple(issues))

    def make_baseline(self, spec: object, store: ArtifactStore, deadline: Deadline) -> Plan:
        if not isinstance(spec, LocationAllocationProblem):
            raise ProblemDataError("expected a LocationAllocationProblem")
        report = self.validate_spec(spec, store)
        if not report.valid:
            raise ProblemDataError(report.issues[0].message)
        from oasis.problems.search import make_baseline

        return make_baseline(self, spec, store, deadline)

    def _structural_issues(
        self,
        problem: LocationAllocationProblem,
        plan: Plan,
        data: LocationAllocationData,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if plan.problem_type != problem.type_id.value:
            issues.append(_issue("wrong_problem_type", "plan problem type does not match"))
        if len(plan.selected_site_ids) != len(set(plan.selected_site_ids)):
            issues.append(_issue("duplicate_site", "selected site IDs must be unique"))
        by_id = {identifier: index for index, identifier in enumerate(data.candidate_ids)}
        unknown = sorted(set(plan.selected_site_ids) - set(by_id))
        if unknown:
            issues.append(_issue("unknown_site", "plan selects unknown sites", site_ids=unknown))
        selected = _selected_indices(plan, data)
        if any(not data.eligible[index] for index in selected):
            issues.append(_issue("ineligible_site", "plan selects an ineligible site"))
        missing_existing = [
            data.candidate_ids[index]
            for index in np.flatnonzero(data.existing)
            if data.candidate_ids[index] not in plan.selected_site_ids
        ]
        if missing_existing:
            issues.append(
                _issue(
                    "missing_existing",
                    "plan omits mandatory existing sites",
                    site_ids=missing_existing,
                )
            )
        policy = problem.policy
        if self.problem_type is LocationProblemType.INCREMENTAL_COVERAGE:
            new_count = sum(not data.existing[index] for index in selected)
            if policy.new_site_limit is not None and new_count > policy.new_site_limit:
                issues.append(_issue("new_site_limit", "plan exceeds the new-site limit"))
            if policy.site_limit is not None and len(selected) > policy.site_limit:
                issues.append(_issue("site_limit", "plan exceeds the total site limit"))
        elif policy.site_limit is not None and len(selected) > policy.site_limit:
            issues.append(_issue("site_limit", "plan exceeds the site limit"))
        cost = _plan_cost(selected, data)
        if policy.financial_budget is not None and cost > policy.financial_budget + FLOAT_TOLERANCE:
            issues.append(
                _issue("financial_budget", "plan exceeds the financial budget", cost=cost)
            )
        if problem.candidates.minimum_spacing is not None and len(selected) > 1:
            geometries = tuple(data.candidates_frame.geometry.iloc[index] for index in selected)
            for left in range(len(geometries)):
                for right in range(left + 1, len(geometries)):
                    if (
                        geometries[left].distance(geometries[right]) + FLOAT_TOLERANCE
                        < problem.candidates.minimum_spacing
                    ):
                        issues.append(_issue("minimum_spacing", "selected sites violate spacing"))
                        break
                if issues and issues[-1].code == "minimum_spacing":
                    break
        if plan.assignments or plan.routes or plan.schedules:
            issues.append(
                _issue(
                    "unsupported_fields",
                    "location plans cannot contain assignment, route, or schedule records",
                )
            )
        if self.problem_type is not LocationProblemType.CAPACITATED_ALLOCATION and plan.allocations:
            issues.append(
                _issue("unsupported_allocations", "allocations apply only to capacitated problems")
            )
        if self.problem_type in {
            LocationProblemType.WEIGHTED_P_MEDIAN,
            LocationProblemType.P_CENTER,
            LocationProblemType.QUANTILE_ACCESS,
        }:
            if not selected:
                issues.append(
                    _issue("no_selected_site", "access plans must select at least one site")
                )
            elif np.any(np.isinf(data.access[:, selected]).all(axis=1) & (data.weights > 0)):
                issues.append(
                    _issue("unserved_access", "selected sites leave positive demand unreachable")
                )
        if (
            self.problem_type is LocationProblemType.RESILIENT_COVERAGE
            and len(selected) < policy.redundancy
        ):
            issues.append(
                _issue("insufficient_redundancy", "plan selects fewer than redundancy sites")
            )
        if self.problem_type is LocationProblemType.CAPACITATED_ALLOCATION:
            issues.extend(self._allocation_issues(plan, data, selected))
        return issues

    def _allocation_issues(
        self, plan: Plan, data: LocationAllocationData, selected: tuple[int, ...]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        demand_ids = set(data.demand_ids)
        selected_ids = {data.candidate_ids[index] for index in selected}
        seen: set[tuple[str, str]] = set()
        for allocation in plan.allocations:
            if set(allocation) != {"demand_id", "site_id", "amount"}:
                issues.append(
                    _issue(
                        "invalid_allocation_fields",
                        "allocation records require exactly demand_id, site_id, and amount",
                    )
                )
            demand_id = str(allocation.get("demand_id", ""))
            site_id = str(allocation.get("site_id", ""))
            try:
                amount = _allocation_amount(allocation.get("amount", 0.0))
            except (TypeError, ValueError):
                amount = math.nan
            if demand_id not in demand_ids or site_id not in selected_ids:
                issues.append(
                    _issue(
                        "invalid_allocation_arc",
                        "allocation IDs must name demand and a selected site",
                    )
                )
                continue
            if (demand_id, site_id) in seen:
                issues.append(
                    _issue("duplicate_allocation", "allocation demand/site pairs must be unique")
                )
            seen.add((demand_id, site_id))
            if not math.isfinite(amount) or amount <= 0:
                issues.append(
                    _issue(
                        "invalid_allocation_amount",
                        "allocation amounts must be finite and positive",
                    )
                )
            row = data.demand_ids.index(demand_id)
            column = data.candidate_ids.index(site_id)
            if math.isinf(float(data.access[row, column])):
                issues.append(
                    _issue(
                        "unreachable_allocation", "allocation uses an unreachable demand/site arc"
                    )
                )
        served, used, _, _ = _allocation_totals(plan, data)
        if np.any(served > data.weights + FLOAT_TOLERANCE):
            issues.append(_issue("demand_overallocation", "allocations exceed demand"))
        if np.any(used > data.capacities + FLOAT_TOLERANCE):
            issues.append(_issue("capacity_exceeded", "allocations exceed selected-site capacity"))
        return issues

    def validate_plan(self, spec: object, plan: Plan, store: ArtifactStore) -> ValidationReport:
        if not isinstance(spec, LocationAllocationProblem):
            return ValidationReport(
                valid=False,
                issues=(_issue("wrong_schema", "expected a LocationAllocationProblem"),),
            )
        try:
            data = load_problem_data(spec, store)
        except (ArtifactStoreError, ProblemDataError, ValueError, KeyError) as error:
            return ValidationReport(valid=False, issues=(_issue("invalid_evidence", str(error)),))
        issues = self._structural_issues(spec, plan, data)
        if not issues:
            overall, groups, _ = self._measure_metrics(spec, plan, data)
            target = spec.policy.coverage_target
            if target is not None and overall.get("coverage", 0.0) + FLOAT_TOLERANCE < target:
                issues.append(_issue("coverage_target", "plan does not reach the coverage target"))
            for name, floor in spec.policy.group_floors.items():
                if groups.get(name, {}).get("coverage", 0.0) + FLOAT_TOLERANCE < floor:
                    issues.append(
                        _issue("group_floor", f"plan does not reach the coverage floor for {name}")
                    )
        return ValidationReport(valid=not issues, issues=tuple(issues))

    def _measure_metrics(
        self,
        problem: LocationAllocationProblem,
        plan: Plan,
        data: LocationAllocationData,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        selected = _selected_indices(plan, data)
        if self.problem_type is LocationProblemType.CAPACITATED_ALLOCATION:
            served, _, access_sum, demand_access = _allocation_totals(plan, data)
            served_total = float(served.sum())
            coverage = served_total / float(data.weights.sum())
            groups: dict[str, dict[str, float]] = {}
            for name, membership in data.group_membership.items():
                group_need = float(np.dot(data.weights, membership))
                group_served = float(np.dot(served, membership))
                groups[name] = {
                    "coverage": group_served / group_need,
                    "demand": group_need,
                    "served_demand": group_served,
                    "unmet_demand": group_need - group_served,
                    "average_access": (
                        float(np.dot(demand_access, membership)) / group_served
                        if group_served
                        else 0.0
                    ),
                }
            return (
                {
                    "coverage": coverage,
                    "served_demand": served_total,
                    "unmet_demand": float(data.weights.sum()) - served_total,
                    "average_access": access_sum / served_total if served_total else 0.0,
                },
                groups,
                {},
            )
        if self.problem_type in {
            LocationProblemType.WEIGHTED_P_MEDIAN,
            LocationProblemType.P_CENTER,
            LocationProblemType.QUANTILE_ACCESS,
        }:
            nearest = data.access[:, selected].min(axis=1)
            average = _weighted_mean(nearest, data.weights)
            group_access: dict[str, dict[str, float]] = {}
            for name, membership in data.group_membership.items():
                group_weights = data.weights * membership
                positive = group_weights > 0
                group_access[name] = {
                    "demand": float(group_weights.sum()),
                    "average_access": _weighted_mean(nearest, group_weights),
                    "maximum_access": float(nearest[positive].max()),
                    "quantile_access": _weighted_quantile(
                        nearest, group_weights, problem.policy.quantile
                    ),
                }
            return (
                {
                    "average_access": average,
                    "maximum_access": float(nearest.max()),
                    "quantile_access": _weighted_quantile(
                        nearest, data.weights, problem.policy.quantile
                    ),
                },
                group_access,
                {},
            )
        overall, groups, scenarios = _coverage_metrics(problem, data, selected)
        if self.problem_type is LocationProblemType.RESILIENT_COVERAGE:
            failure_values: list[float] = []
            failure_group_values: dict[str, list[float]] = {
                name: [] for name in data.group_membership
            }
            for failed in selected:
                remaining = tuple(index for index in selected if index != failed)
                failed_overall, failed_groups, failed_scenarios = _coverage_metrics(
                    problem, data, remaining
                )
                coverage = failed_overall["coverage"]
                failure_values.append(coverage)
                for name, metrics in failed_groups.items():
                    failure_group_values[name].append(metrics["coverage"])
                scenarios[f"failure:{data.candidate_ids[failed]}"] = {"coverage": coverage}
                for name, metrics in failed_scenarios.items():
                    scenarios[f"{name}:failure:{data.candidate_ids[failed]}"] = metrics
            overall["normal_coverage"] = overall["coverage"]
            overall["one_failure_coverage"] = min(failure_values) if failure_values else 0.0
            redundant_by_scenario: dict[str, float] = {}
            redundant_group_scenarios: dict[str, dict[str, float]] = {
                group: {} for group in data.group_membership
            }
            for name, service in data.services.items():
                if selected:
                    counts = (service[:, selected] > 0.0).sum(axis=1)
                    redundant = (counts >= problem.policy.redundancy).astype(np.float64)
                else:
                    redundant = np.zeros(len(data.demand_ids), dtype=np.float64)
                redundant_by_scenario[name] = _weighted_mean(redundant, data.weights)
                for group, membership in data.group_membership.items():
                    redundant_group_scenarios[group][name] = _weighted_mean(
                        redundant, data.weights * membership
                    )
            overall["redundant_coverage"] = _aggregate_scenarios(
                redundant_by_scenario, problem, data
            )
            for name, metrics in groups.items():
                metrics["normal_coverage"] = metrics["coverage"]
                metrics["one_failure_coverage"] = min(failure_group_values[name])
                metrics["redundant_coverage"] = _aggregate_scenarios(
                    redundant_group_scenarios[name], problem, data
                )
        return overall, groups, scenarios

    def measure(self, spec: object, plan: Plan, store: ArtifactStore) -> Scorecard:
        if not isinstance(spec, LocationAllocationProblem):
            raise ProblemDataError("expected a LocationAllocationProblem")
        data = load_problem_data(spec, store)
        structural = self._structural_issues(spec, plan, data)
        if structural:
            return Scorecard(
                feasible=False,
                violations=tuple(structural),
                comparator_key=INFEASIBLE_KEY,
                problem_hash=spec.problem_hash,
                evidence_hash=spec.evidence_hash,
                policy_hash=spec.policy_hash,
                evaluator_version=spec.evaluator_version,
                plan_hash=plan_hash(plan),
            )
        overall, groups, scenarios = self._measure_metrics(spec, plan, data)
        violations: list[ValidationIssue] = []
        if spec.policy.coverage_target is not None:
            if overall.get("coverage", 0.0) + FLOAT_TOLERANCE < spec.policy.coverage_target:
                violations.append(_issue("coverage_target", "plan does not reach coverage target"))
        for name, floor in spec.policy.group_floors.items():
            if groups.get(name, {}).get("coverage", 0.0) + FLOAT_TOLERANCE < floor:
                violations.append(_issue("group_floor", f"coverage floor is not met for {name}"))
        selected = _selected_indices(plan, data)
        cost = _plan_cost(selected, data)
        overall["opening_cost"] = cost
        comparator = self._comparator_key(spec, overall, groups, cost)
        raw = self._raw_objective(spec, overall, groups, cost)
        return Scorecard(
            feasible=not violations,
            violations=tuple(violations),
            raw_objective=raw,
            comparator_key=comparator if not violations else INFEASIBLE_KEY,
            overall_metrics=overall,
            group_metrics=groups,
            scenario_metrics=scenarios,
            problem_hash=spec.problem_hash,
            evidence_hash=spec.evidence_hash,
            policy_hash=spec.policy_hash,
            evaluator_version=spec.evaluator_version,
            plan_hash=plan_hash(plan),
            assumptions=("unmet demand is permitted and reported",)
            if self.problem_type is LocationProblemType.CAPACITATED_ALLOCATION
            else (),
        )

    def _comparator_key(
        self,
        problem: LocationAllocationProblem,
        overall: dict[str, float],
        groups: dict[str, dict[str, float]],
        cost: float,
    ) -> tuple[float, ...]:
        if self.problem_type is LocationProblemType.MIN_COST_TARGET_COVERAGE:
            return (-cost, overall["coverage"])
        if self.problem_type is LocationProblemType.WEIGHTED_P_MEDIAN:
            return (-overall["average_access"], -cost)
        if self.problem_type is LocationProblemType.P_CENTER:
            return (-overall["maximum_access"], -overall["average_access"], -cost)
        if self.problem_type is LocationProblemType.QUANTILE_ACCESS:
            return (-overall["quantile_access"], -overall["average_access"], -cost)
        if self.problem_type is LocationProblemType.CAPACITATED_ALLOCATION:
            return (overall["coverage"], -overall["average_access"], -cost)
        if self.problem_type is LocationProblemType.RESILIENT_COVERAGE:
            return (
                overall["one_failure_coverage"],
                overall["redundant_coverage"],
                overall["normal_coverage"],
                -cost,
            )
        worst_group = min((metrics["coverage"] for metrics in groups.values()), default=1.0)
        if problem.policy.equity_objective is EquityObjective.MAX_MIN:
            return (worst_group, overall["coverage"], -cost)
        return (overall["coverage"], worst_group, -cost)

    def _raw_objective(
        self,
        problem: LocationAllocationProblem,
        overall: dict[str, float],
        groups: dict[str, dict[str, float]],
        cost: float,
    ) -> dict[str, float]:
        if self.problem_type is LocationProblemType.WEIGHTED_P_MEDIAN:
            return {"weighted_average_access": overall["average_access"]}
        if self.problem_type is LocationProblemType.P_CENTER:
            return {"maximum_access": overall["maximum_access"]}
        if self.problem_type is LocationProblemType.QUANTILE_ACCESS:
            return {
                "quantile_access": overall["quantile_access"],
                "quantile": problem.policy.quantile,
            }
        if self.problem_type is LocationProblemType.MIN_COST_TARGET_COVERAGE:
            return {"opening_cost": cost, "coverage": overall["coverage"]}
        if self.problem_type is LocationProblemType.CAPACITATED_ALLOCATION:
            return {
                "served_demand": overall["served_demand"],
                "unmet_demand": overall["unmet_demand"],
                "average_access": overall["average_access"],
            }
        if self.problem_type is LocationProblemType.RESILIENT_COVERAGE:
            return {
                "one_failure_coverage": overall["one_failure_coverage"],
                "redundant_coverage": overall["redundant_coverage"],
            }
        if problem.policy.equity_objective is EquityObjective.MAX_MIN:
            return {"worst_group_coverage": min(value["coverage"] for value in groups.values())}
        return {"coverage": overall["coverage"]}

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
        if self.problem_type is LocationProblemType.CAPACITATED_ALLOCATION:
            return (
                SearchStrategy.LOCAL_ASSIGNMENT,
                SearchStrategy.ADD_SWAP,
                SearchStrategy.EXACT_ENUMERATION,
            )
        return (
            SearchStrategy.ADD_SWAP,
            SearchStrategy.MULTI_SWAP,
            SearchStrategy.SCENARIO_AWARE,
            SearchStrategy.EXACT_ENUMERATION,
        )

    def render_result(self, spec: object, plan: Plan, scorecard: Scorecard) -> ResultView:
        if not isinstance(spec, LocationAllocationProblem):
            raise ProblemDataError("expected a LocationAllocationProblem")
        if scorecard.problem_hash != spec.problem_hash:
            raise ValueError("scorecard does not belong to this immutable problem")
        if scorecard.plan_hash != plan_hash(plan):
            raise ValueError("scorecard does not belong to this plan")
        primary_name = next(iter(scorecard.raw_objective), None)
        return ResultView(
            problem_type=spec.type_id.value,
            feasible=scorecard.feasible,
            selected_site_ids=plan.selected_site_ids,
            primary_metric_name=primary_name,
            primary_metric_value=(
                scorecard.raw_objective[primary_name] if primary_name is not None else None
            ),
            overall_metrics=scorecard.overall_metrics,
            group_metrics=scorecard.group_metrics,
            scenario_metrics=scorecard.scenario_metrics,
            violations=scorecard.violations,
            warnings=scorecard.warnings,
        )


def create_problem_registry() -> ProblemRegistry:
    """Create all built-in location problem plugins without import side effects."""

    return ProblemRegistry(LocationAllocationPlugin(type_id) for type_id in LocationProblemType)
