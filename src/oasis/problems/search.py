"""Deterministic baseline and resumable search strategies for location allocation."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from pydantic import JsonValue
from scipy.optimize import linprog

from oasis.artifacts import ArtifactStore
from oasis.problems.location_allocation import (
    FLOAT_TOLERANCE,
    LocationAllocationData,
    LocationAllocationPlugin,
    load_problem_data,
)
from oasis.problems.protocols import Deadline
from oasis.problems.schemas import (
    Comparison,
    LocationAllocationProblem,
    LocationProblemType,
    Scorecard,
    SearchStrategy,
    VerifiedBound,
)
from oasis.schemas import Plan

BASELINE_EXACT_LIMIT = 16


@dataclass(frozen=True, slots=True)
class CandidateSpace:
    """Deterministically ordered plan candidates plus an optional known size."""

    plans: Iterable[Plan]
    total: int | None


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Candidate and independently serializable bound from an external solver."""

    plan: Plan
    bound: VerifiedBound


def _new_site_count(indices: tuple[int, ...], data: LocationAllocationData) -> int:
    return sum(not data.existing[index] for index in indices)


def _selection_within_limits(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    indices: tuple[int, ...],
) -> bool:
    policy = problem.policy
    if problem.type_id is LocationProblemType.INCREMENTAL_COVERAGE:
        if (
            policy.new_site_limit is not None
            and _new_site_count(indices, data) > policy.new_site_limit
        ):
            return False
        if policy.site_limit is not None and len(indices) > policy.site_limit:
            return False
    elif policy.site_limit is not None and len(indices) > policy.site_limit:
        return False
    if policy.financial_budget is not None:
        cost = sum(data.costs[index] for index in indices if not data.existing[index])
        if cost > policy.financial_budget + FLOAT_TOLERANCE:
            return False
    spacing = problem.candidates.minimum_spacing
    if spacing is not None:
        for left, right in itertools.combinations(indices, 2):
            left_geometry = data.candidates_frame.geometry.iloc[left]
            right_geometry = data.candidates_frame.geometry.iloc[right]
            if left_geometry.distance(right_geometry) + FLOAT_TOLERANCE < spacing:
                return False
    return True


def _allocation_plan(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    indices: tuple[int, ...],
    *,
    strategy: str,
) -> Plan:
    arcs = [
        (demand_index, site_index)
        for demand_index in range(len(data.demand_ids))
        for site_index in indices
        if math.isfinite(float(data.access[demand_index, site_index]))
    ]
    if not arcs:
        return Plan(
            problem_type=problem.type_id.value,
            selected_site_ids=tuple(data.candidate_ids[index] for index in indices),
            metadata={"strategy": strategy},
        )
    demand_count = len(data.demand_ids)
    variable_count = len(arcs) + demand_count
    finite_access = [float(data.access[row, column]) for row, column in arcs]
    unmet_penalty = (max(finite_access, default=0.0) + 1.0) * (float(data.weights.sum()) + 1.0)
    objective = np.array([*finite_access, *([unmet_penalty] * demand_count)], dtype=np.float64)
    equalities = np.zeros((demand_count, variable_count), dtype=np.float64)
    for variable, (row, _) in enumerate(arcs):
        equalities[row, variable] = 1.0
    for row in range(demand_count):
        equalities[row, len(arcs) + row] = 1.0
    capacities = np.zeros((len(indices), variable_count), dtype=np.float64)
    for variable, (_, column) in enumerate(arcs):
        capacities[indices.index(column), variable] = 1.0
    result = linprog(
        objective,
        A_ub=capacities,
        b_ub=data.capacities[np.array(indices, dtype=np.int64)],
        A_eq=equalities,
        b_eq=data.weights,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None:
        raise ValueError(f"allocation linear program failed: {result.message}")
    allocations: tuple[dict[str, JsonValue], ...] = tuple(
        {
            "demand_id": data.demand_ids[row],
            "site_id": data.candidate_ids[column],
            "amount": float(result.x[variable]),
        }
        for variable, (row, column) in enumerate(arcs)
        if result.x[variable] > FLOAT_TOLERANCE
    )
    return Plan(
        problem_type=problem.type_id.value,
        selected_site_ids=tuple(data.candidate_ids[index] for index in indices),
        allocations=allocations,
        metadata={"strategy": strategy},
    )


def plan_for_indices(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    indices: Iterable[int],
    *,
    strategy: str,
) -> Plan:
    """Build a canonical plan, independently recomputing capacity allocations if needed."""

    ordered = tuple(sorted(set(indices)))
    if problem.type_id is LocationProblemType.CAPACITATED_ALLOCATION:
        return _allocation_plan(problem, data, ordered, strategy=strategy)
    return Plan(
        problem_type=problem.type_id.value,
        selected_site_ids=tuple(data.candidate_ids[index] for index in ordered),
        metadata={"strategy": strategy},
    )


def _max_optional_sites(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    mandatory: tuple[int, ...],
    optional: tuple[int, ...],
) -> int:
    if problem.type_id is LocationProblemType.INCREMENTAL_COVERAGE:
        maximum = problem.policy.new_site_limit or 0
        if problem.policy.site_limit is not None:
            maximum = min(maximum, max(0, problem.policy.site_limit - len(mandatory)))
        return min(maximum, len(optional))
    if problem.policy.site_limit is None:
        return len(optional)
    return max(0, min(problem.policy.site_limit - len(mandatory), len(optional)))


def _selection_components(
    problem: LocationAllocationProblem, data: LocationAllocationData
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    mandatory = tuple(int(index) for index in np.flatnonzero(data.existing))
    optional = tuple(int(index) for index in np.flatnonzero(data.eligible & ~data.existing))
    return mandatory, optional, _max_optional_sites(problem, data, mandatory, optional)


def all_candidate_plans(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    *,
    strategy: str = SearchStrategy.EXACT_ENUMERATION.value,
) -> CandidateSpace:
    """Enumerate every selection satisfying cheap, plan-independent constraints."""

    mandatory, optional, maximum = _selection_components(problem, data)
    minimum = 0
    if (
        problem.type_id
        in {
            LocationProblemType.WEIGHTED_P_MEDIAN,
            LocationProblemType.P_CENTER,
            LocationProblemType.QUANTILE_ACCESS,
        }
        and not mandatory
    ):
        minimum = 1
    if problem.type_id is LocationProblemType.RESILIENT_COVERAGE:
        minimum = max(minimum, problem.policy.redundancy - len(mandatory))
    total = sum(math.comb(len(optional), size) for size in range(minimum, maximum + 1))

    def generate() -> Iterator[Plan]:
        for size in range(minimum, maximum + 1):
            for addition in itertools.combinations(optional, size):
                indices = tuple(sorted((*mandatory, *addition)))
                yield plan_for_indices(problem, data, indices, strategy=strategy)

    return CandidateSpace(plans=generate(), total=total)


def _relaxed_key(
    problem: LocationAllocationProblem, score: Scorecard, data: LocationAllocationData
) -> tuple[float, ...]:
    metrics = score.overall_metrics
    cost = metrics.get("opening_cost", 0.0)
    if problem.type_id is LocationProblemType.MIN_COST_TARGET_COVERAGE:
        return (metrics.get("coverage", 0.0), -cost)
    if problem.type_id is LocationProblemType.WEIGHTED_P_MEDIAN:
        return (-metrics.get("average_access", math.inf), -cost)
    if problem.type_id is LocationProblemType.P_CENTER:
        return (-metrics.get("maximum_access", math.inf), -metrics.get("average_access", math.inf))
    if problem.type_id is LocationProblemType.QUANTILE_ACCESS:
        return (-metrics.get("quantile_access", math.inf), -metrics.get("average_access", math.inf))
    if problem.type_id is LocationProblemType.CAPACITATED_ALLOCATION:
        return (metrics.get("coverage", 0.0), -metrics.get("average_access", math.inf), -cost)
    if problem.type_id is LocationProblemType.RESILIENT_COVERAGE:
        return (
            metrics.get("one_failure_coverage", 0.0),
            metrics.get("normal_coverage", 0.0),
            -cost,
        )
    if problem.policy.equity_objective.value == "max_min":
        worst = min(
            (values.get("coverage", 0.0) for values in score.group_metrics.values()),
            default=0.0,
        )
        return (worst, metrics.get("coverage", 0.0), -cost)
    return (metrics.get("coverage", 0.0), -cost, float(len(data.candidate_ids)))


def _baseline_strategy(problem: LocationAllocationProblem) -> str:
    return {
        LocationProblemType.MIN_COST_TARGET_COVERAGE: "cost_aware_greedy_cover",
        LocationProblemType.WEIGHTED_P_MEDIAN: "greedy_p_median",
        LocationProblemType.P_CENTER: "farthest_first",
        LocationProblemType.QUANTILE_ACCESS: "farthest_first",
        LocationProblemType.CAPACITATED_ALLOCATION: "feasible_flow",
    }.get(problem.type_id, "greedy_coverage")


def _greedy_baseline(
    plugin: LocationAllocationPlugin,
    problem: LocationAllocationProblem,
    store: ArtifactStore,
    data: LocationAllocationData,
    deadline: Deadline,
) -> Plan:
    mandatory, optional, maximum = _selection_components(problem, data)
    baseline_strategy = _baseline_strategy(problem)
    chosen = list(mandatory)
    remaining = list(optional)
    if problem.type_id is LocationProblemType.RESILIENT_COVERAGE:
        required_additions = max(0, problem.policy.redundancy - len(chosen))
    elif (
        problem.type_id
        in {
            LocationProblemType.WEIGHTED_P_MEDIAN,
            LocationProblemType.P_CENTER,
            LocationProblemType.QUANTILE_ACCESS,
        }
        and not chosen
    ):
        required_additions = 1
    else:
        required_additions = 0
    current = (
        plan_for_indices(problem, data, chosen, strategy=baseline_strategy) if chosen else None
    )
    current_score = plugin.measure(problem, current, store) if current is not None else None
    current_coverage = (
        current_score.overall_metrics.get("coverage", 0.0) if current_score is not None else 0.0
    )
    if (
        problem.type_id is LocationProblemType.MIN_COST_TARGET_COVERAGE
        and current_score is not None
        and current_score.feasible
    ):
        assert current is not None
        return current
    while remaining and len(chosen) - len(mandatory) < maximum:
        if deadline.expired:
            break
        candidates: list[tuple[tuple[float, ...], str, int, Plan, Scorecard]] = []
        farthest_demand = None
        if chosen and problem.type_id in {
            LocationProblemType.P_CENTER,
            LocationProblemType.QUANTILE_ACCESS,
        }:
            nearest = data.access[:, chosen].min(axis=1)
            farthest_demand = max(
                (index for index, weight in enumerate(data.weights) if weight > 0),
                key=lambda index: (nearest[index], data.weights[index], data.demand_ids[index]),
            )
        for index in remaining:
            indices = tuple(sorted((*chosen, index)))
            if not _selection_within_limits(problem, data, indices):
                continue
            plan = plan_for_indices(problem, data, indices, strategy=baseline_strategy)
            score = plugin.measure(problem, plan, store)
            key = _relaxed_key(problem, score, data)
            if problem.type_id is LocationProblemType.MIN_COST_TARGET_COVERAGE:
                coverage = score.overall_metrics.get("coverage", 0.0)
                marginal = max(0.0, coverage - current_coverage)
                incremental_cost = data.costs[index]
                ratio = marginal / incremental_cost if incremental_cost > 0 else math.inf
                key = (ratio, coverage, -incremental_cost)
            elif farthest_demand is not None:
                key = (-float(data.access[farthest_demand, index]), *key)
            candidates.append((key, data.candidate_ids[index], index, plan, score))
        if not candidates:
            break
        _, _, selected_index, selected_plan, selected_score = max(
            candidates, key=lambda item: (item[0], tuple(-ord(char) for char in item[1]))
        )
        additions = len(chosen) - len(mandatory)
        if (
            additions >= required_additions
            and current_score is not None
            and current_score.feasible
            and _relaxed_key(problem, selected_score, data)
            <= _relaxed_key(problem, current_score, data)
        ):
            break
        chosen.append(selected_index)
        remaining.remove(selected_index)
        current = selected_plan
        current_score = selected_score
        current_coverage = selected_score.overall_metrics.get("coverage", 0.0)
        additions = len(chosen) - len(mandatory)
        if additions >= required_additions and selected_score.feasible:
            if problem.type_id is LocationProblemType.MIN_COST_TARGET_COVERAGE:
                break
    if current is None:
        current = plan_for_indices(problem, data, chosen, strategy=baseline_strategy)
    return current


def make_baseline(
    plugin: LocationAllocationPlugin,
    problem: LocationAllocationProblem,
    store: ArtifactStore,
    deadline: Deadline,
) -> Plan:
    """Construct a cheap deterministic plan and exactly repair small infeasible instances."""

    data = load_problem_data(problem, store)
    baseline = _greedy_baseline(plugin, problem, store, data, deadline)
    baseline_score = plugin.measure(problem, baseline, store)
    if baseline_score.feasible:
        return baseline
    if len(data.candidate_ids) <= BASELINE_EXACT_LIMIT:
        best: tuple[Plan, Scorecard] | None = None
        for plan in all_candidate_plans(problem, data, strategy="baseline_repair").plans:
            if deadline.expired:
                break
            score = plugin.measure(problem, plan, store)
            if not score.feasible:
                continue
            if best is None or plugin.compare(score, best[1]) is Comparison.BETTER:
                best = (plan, score)
        if best is not None:
            return best[0]
    report = plugin.validate_plan(problem, baseline, store)
    detail = report.issues[0].message if report.issues else "no feasible baseline found"
    raise ValueError(f"problem failed baseline admission: {detail}")


def neighborhood_plans(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    incumbent: Plan,
    strategy: SearchStrategy,
) -> CandidateSpace:
    """Generate deterministic add/swap or bounded two-swap neighbors."""

    by_id = {identifier: index for index, identifier in enumerate(data.candidate_ids)}
    selected = tuple(sorted(by_id[identifier] for identifier in incumbent.selected_site_ids))
    selected_set = set(selected)
    removable = tuple(index for index in selected if not data.existing[index])
    unselected = tuple(
        index
        for index in range(len(data.candidate_ids))
        if data.eligible[index] and index not in selected_set
    )

    def generate() -> Iterator[Plan]:
        selections: Iterator[tuple[int, ...]] = itertools.chain(
            (tuple(sorted((*selected, addition))) for addition in unselected),
            (
                tuple(sorted((selected_set - {removal}) | {addition}))
                for removal in removable
                for addition in unselected
            ),
            (
                tuple(sorted((selected_set - set(removals)) | set(additions)))
                for removals in itertools.combinations(removable, 2)
                for additions in itertools.combinations(unselected, 2)
            )
            if strategy is SearchStrategy.MULTI_SWAP
            else (),
        )
        for indices in selections:
            yield plan_for_indices(problem, data, indices, strategy=strategy.value)

    total = len(unselected) + len(removable) * len(unselected)
    if strategy is SearchStrategy.MULTI_SWAP:
        total += math.comb(len(removable), 2) * math.comb(len(unselected), 2)
    return CandidateSpace(plans=generate(), total=total)


def candidate_space(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    incumbent: Plan,
    strategy: SearchStrategy,
) -> CandidateSpace:
    """Resolve a stable strategy value to its deterministic candidate sequence."""

    if strategy is SearchStrategy.EXACT_ENUMERATION:
        return all_candidate_plans(problem, data)
    if strategy is SearchStrategy.LOCAL_ASSIGNMENT:
        indices = tuple(data.candidate_ids.index(value) for value in incumbent.selected_site_ids)
        return CandidateSpace(
            plans=(plan_for_indices(problem, data, indices, strategy=strategy.value),), total=1
        )
    if strategy in {
        SearchStrategy.ADD_SWAP,
        SearchStrategy.MULTI_SWAP,
        SearchStrategy.SCENARIO_AWARE,
    }:
        return neighborhood_plans(problem, data, incumbent, strategy)
    raise ValueError("OR-Tools strategy is solved directly rather than enumerated")


def solve_ortools(
    problem: LocationAllocationProblem,
    data: LocationAllocationData,
    *,
    max_time_seconds: float | None = None,
) -> SolverResult:
    """Solve binary single-scenario coverage models with CP-SAT and return its bound."""

    if problem.type_id not in {
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        LocationProblemType.MIN_COST_TARGET_COVERAGE,
        LocationProblemType.EQUITY_COVERAGE,
        LocationProblemType.INCREMENTAL_COVERAGE,
    }:
        raise ValueError("ortools_cp_sat currently supports binary coverage families")
    if len(data.services) != 1:
        raise ValueError(
            "ortools_cp_sat requires one service scenario; use scenario_aware otherwise"
        )
    scenario = problem.service_scenarios[0]
    if (
        scenario.access_matrix is not None
        or scenario.demand_multiplier is not None
        or scenario.failed_site_ids
    ):
        raise ValueError("ortools_cp_sat does not support scenario overrides")
    service = next(iter(data.services.values()))
    if not np.isin(service, (0.0, 1.0)).all():
        raise ValueError("ortools_cp_sat requires a binary service matrix")

    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    sites = [model.new_bool_var(f"site_{index}") for index in range(len(data.candidate_ids))]
    covered = [model.new_bool_var(f"covered_{index}") for index in range(len(data.demand_ids))]
    for index, variable in enumerate(sites):
        if data.existing[index]:
            model.add(variable == 1)
        elif not data.eligible[index]:
            model.add(variable == 0)
    for row, variable in enumerate(covered):
        covering = [sites[column] for column in range(len(sites)) if service[row, column] >= 1.0]
        if covering:
            model.add(variable <= sum(covering))
            for site in covering:
                model.add(variable >= site)
        else:
            model.add(variable == 0)
    policy = problem.policy
    if problem.type_id is LocationProblemType.INCREMENTAL_COVERAGE:
        assert policy.new_site_limit is not None
        model.add(
            sum(sites[index] for index in range(len(sites)) if not data.existing[index])
            <= policy.new_site_limit
        )
    elif policy.site_limit is not None:
        model.add(sum(sites) <= policy.site_limit)
    scale = 1_000_000
    scaled_costs = [round(value * scale) for value in data.costs]
    scaled_weights = [round(value * scale) for value in data.weights]
    if policy.financial_budget is not None:
        model.add(
            sum(
                scaled_costs[index] * sites[index]
                for index in range(len(sites))
                if not data.existing[index]
            )
            <= round(policy.financial_budget * scale)
        )
    weighted_coverage = sum(scaled_weights[index] * covered[index] for index in range(len(covered)))
    total_weight = sum(scaled_weights)
    if policy.coverage_target is not None:
        model.add(weighted_coverage >= math.ceil(policy.coverage_target * total_weight))
    group_expressions: dict[str, tuple[object, int]] = {}
    for name, membership in data.group_membership.items():
        coefficients = [
            round(data.weights[index] * membership[index] * scale)
            for index in range(len(data.weights))
        ]
        expression = sum(coefficients[index] * covered[index] for index in range(len(covered)))
        group_expressions[name] = (expression, sum(coefficients))
    for name, floor in policy.group_floors.items():
        expression, denominator = group_expressions[name]
        model.add(
            expression >= math.ceil(floor * denominator)  # type: ignore[operator]
        )
    cost_expression = sum(
        scaled_costs[index] * sites[index]
        for index in range(len(sites))
        if not data.existing[index]
    )
    worst_group = None
    if group_expressions:
        worst_group = model.new_int_var(0, scale, "worst_group_coverage")
        for expression, denominator in group_expressions.values():
            model.add(
                worst_group * denominator <= expression * scale  # type: ignore[operator]
            )

    stages: list[tuple[str, object, bool]]
    if problem.type_id is LocationProblemType.MIN_COST_TARGET_COVERAGE:
        stages = [("opening_cost", cost_expression, False), ("coverage", weighted_coverage, True)]
    elif policy.equity_objective.value == "max_min":
        assert worst_group is not None
        stages = [
            ("worst_group_coverage", worst_group, True),
            ("coverage", weighted_coverage, True),
            ("opening_cost", cost_expression, False),
        ]
    else:
        stages = [("coverage", weighted_coverage, True)]
        if worst_group is not None:
            stages.append(("worst_group_coverage", worst_group, True))
        stages.append(("opening_cost", cost_expression, False))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    if max_time_seconds is not None:
        solver.parameters.max_time_in_seconds = max(0.001, max_time_seconds / len(stages))
    stage_results: list[JsonValue] = []
    all_optimal = True
    status = cp_model.UNKNOWN
    for stage_index, (name, expression, maximize) in enumerate(stages):
        if maximize:
            model.maximize(expression)  # type: ignore[arg-type]
        else:
            model.minimize(expression)  # type: ignore[arg-type]
        status = solver.solve(model)
        if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            raise ValueError("OR-Tools found no feasible coverage plan")
        objective_value = round(solver.objective_value)
        stage_result: dict[str, JsonValue] = {
            "name": name,
            "direction": "maximize" if maximize else "minimize",
            "value": objective_value,
            "best_bound": solver.best_objective_bound,
            "status": solver.status_name(status),
        }
        stage_results.append(stage_result)
        if status != cp_model.OPTIMAL:
            all_optimal = False
            break
        if stage_index < len(stages) - 1:
            model.add(expression == objective_value)
    indices = tuple(index for index, variable in enumerate(sites) if solver.value(variable))
    plan = plan_for_indices(problem, data, indices, strategy=SearchStrategy.ORTOOLS_CP_SAT.value)
    bound = VerifiedBound(
        problem_hash=problem.problem_hash,
        strategy=SearchStrategy.ORTOOLS_CP_SAT,
        complete=all_optimal and len(stage_results) == len(stages),
        explored_candidates=0,
        best_comparator_key=(),
        certificate={
            "solver": "ortools_cp_sat",
            "status": solver.status_name(status),
            "lexicographic_stages": stage_results,
            "integer_scale": scale,
        },
    )
    return SolverResult(plan=plan, bound=bound)
