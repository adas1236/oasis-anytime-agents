"""Deterministic baselines and bounded search for route-service problems."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import numpy as np

from oasis.artifacts import ArtifactStore
from oasis.problems.protocols import Deadline
from oasis.problems.routing import (
    FLOAT_TOLERANCE,
    RouteServiceData,
    RouteServicePlugin,
    indexed_routes,
    load_route_data,
    make_route_plan,
    vehicle_depot_index,
)
from oasis.problems.schemas import (
    Comparison,
    RouteProblemType,
    RouteServiceProblem,
    SearchStrategy,
    VerifiedBound,
)
from oasis.problems.search import CandidateSpace, SolverResult
from oasis.schemas import Plan

BASELINE_EXACT_LIMIT = 9


def _route_duration(route: tuple[int, ...], data: RouteServiceData, travel: np.ndarray) -> float:
    elapsed = 0.0
    for left, right in itertools.pairwise(route):
        arc = float(travel[left, right])
        if not math.isfinite(arc):
            return math.inf
        elapsed += arc
        elapsed = max(elapsed, float(data.window_starts[right]))
        elapsed += float(data.service_times[right])
    return elapsed


def _initial_routes(
    problem: RouteServiceProblem, data: RouteServiceData
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        ((depot, depot) if problem.policy.require_return else (depot,))
        for depot in (
            vehicle_depot_index(data, vehicle) for vehicle in range(problem.policy.vehicle_count)
        )
    )


def _insert(route: tuple[int, ...], position: int, node: int) -> tuple[int, ...]:
    return (*route[:position], node, *route[position:])


def _insertion_candidates(
    problem: RouteServiceProblem,
    data: RouteServiceData,
    routes: tuple[tuple[int, ...], ...],
    node: int,
) -> Iterator[tuple[float, int, int, tuple[tuple[int, ...], ...]]]:
    base_travel = next(iter(data.travel.values()))
    for vehicle, route in enumerate(routes):
        upper = len(route) if not problem.policy.require_return else len(route) - 1
        for position in range(1, upper + 1):
            candidate_route = _insert(route, position, node)
            candidate_routes = (*routes[:vehicle], candidate_route, *routes[vehicle + 1 :])
            plan = make_route_plan(problem, data, candidate_routes, strategy="nearest_insertion")
            plugin = RouteServicePlugin(problem.type_id)
            if not plugin.partial_plan_is_feasible(problem, plan, data):
                continue
            delta = _route_duration(candidate_route, data, base_travel) - _route_duration(
                route, data, base_travel
            )
            yield delta, vehicle, position, candidate_routes


def _greedy_route_baseline(
    plugin: RouteServicePlugin,
    problem: RouteServiceProblem,
    data: RouteServiceData,
    deadline: Deadline,
) -> Plan:
    routes = _initial_routes(problem, data)
    remaining = list(data.service_indices)
    if problem.type_id is RouteProblemType.TSP:
        while remaining and not deadline.expired:
            tsp_options = [
                (*candidate[:3], node, candidate[3])
                for node in remaining
                for candidate in _insertion_candidates(problem, data, routes, node)
            ]
            if not tsp_options:
                break
            _, _, _, node, routes = min(
                tsp_options,
                key=lambda item: (item[0], data.node_ids[item[3]], item[1], item[2]),
            )
            remaining.remove(node)
        return make_route_plan(problem, data, routes, strategy="nearest_insertion")

    base_value = data.demands if problem.type_id is RouteProblemType.MOBILE_SERVICE else data.prizes
    while remaining and not deadline.expired:
        insertion_options: list[
            tuple[float, float, str, int, int, int, tuple[tuple[int, ...], ...]]
        ] = []
        for node in remaining:
            for delta, vehicle, position, candidate_routes in _insertion_candidates(
                problem, data, routes, node
            ):
                value = float(base_value[node])
                ratio = value / delta if delta > FLOAT_TOLERANCE else math.inf
                insertion_options.append(
                    (ratio, value, data.node_ids[node], vehicle, position, node, candidate_routes)
                )
        if not insertion_options:
            break
        _, _, _, _, _, node, routes = max(
            insertion_options,
            key=lambda item: (
                item[0],
                item[1],
                tuple(-ord(char) for char in item[2]),
                -item[3],
                -item[4],
            ),
        )
        remaining.remove(node)
    return make_route_plan(problem, data, routes, strategy="prize_insertion")


def make_route_baseline(
    plugin: RouteServicePlugin,
    problem: RouteServiceProblem,
    store: ArtifactStore,
    deadline: Deadline,
) -> Plan:
    """Construct nearest/insertion routes and exactly repair small failed constructions."""

    data = load_route_data(problem, store)
    baseline = _greedy_route_baseline(plugin, problem, data, deadline)
    if plugin.validate_plan(problem, baseline, store).valid:
        return baseline
    if len(data.service_indices) <= BASELINE_EXACT_LIMIT and problem.policy.vehicle_count == 1:
        best_plan: Plan | None = None
        best_score = None
        for plan in all_route_plans(problem, data, strategy="baseline_repair").plans:
            if deadline.expired:
                break
            score = plugin.measure(problem, plan, store)
            if score.feasible and (
                best_score is None or plugin.compare(score, best_score) is Comparison.BETTER
            ):
                best_plan, best_score = plan, score
        if best_plan is not None:
            return best_plan
    report = plugin.validate_plan(problem, baseline, store)
    detail = report.issues[0].message if report.issues else "no feasible route baseline found"
    raise ValueError(f"problem failed baseline admission: {detail}")


def all_route_plans(
    problem: RouteServiceProblem,
    data: RouteServiceData,
    *,
    strategy: str = SearchStrategy.EXACT_ENUMERATION.value,
) -> CandidateSpace:
    """Enumerate all single-vehicle visit subsets and orders deterministically."""

    if problem.policy.vehicle_count != 1:
        raise ValueError("exact route enumeration currently supports one vehicle")
    service = data.service_indices
    if problem.type_id is RouteProblemType.TSP:
        sizes: tuple[int, ...] = (len(service),)
        total = math.factorial(len(service))
    else:
        sizes = tuple(range(len(service) + 1))
        total = sum(math.factorial(len(service)) // math.factorial(len(service) - k) for k in sizes)
    depot = vehicle_depot_index(data, 0)

    def generate() -> Iterator[Plan]:
        for size in sizes:
            for chosen in itertools.combinations(service, size):
                for ordered in itertools.permutations(chosen):
                    route = (
                        (depot, *ordered, depot)
                        if problem.policy.require_return
                        else (depot, *ordered)
                    )
                    yield make_route_plan(problem, data, (route,), strategy=strategy)

    return CandidateSpace(plans=generate(), total=total)


def _route_positions(
    routes: tuple[tuple[int, ...], ...], require_return: bool
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (vehicle, position, route[position])
        for vehicle, route in enumerate(routes)
        for position in range(1, len(route) - (1 if require_return else 0))
    )


def route_neighborhood_plans(
    problem: RouteServiceProblem,
    data: RouteServiceData,
    incumbent: Plan,
    strategy: SearchStrategy,
) -> CandidateSpace:
    """Generate deterministic 2-opt, relocate/insertion/removal, or swap neighbors."""

    routes = indexed_routes(incumbent, data)
    positions = _route_positions(routes, problem.policy.require_return)
    visited = {node for _, _, node in positions}
    unvisited = tuple(node for node in data.service_indices if node not in visited)
    plans: list[Plan] = []
    if strategy is SearchStrategy.TWO_OPT:
        for vehicle, route in enumerate(routes):
            stop = len(route) - (1 if problem.policy.require_return else 0)
            for left in range(1, stop):
                for right in range(left + 1, stop):
                    reversed_route = (
                        *route[:left],
                        *reversed(route[left : right + 1]),
                        *route[right + 1 :],
                    )
                    candidate = (*routes[:vehicle], reversed_route, *routes[vehicle + 1 :])
                    plans.append(make_route_plan(problem, data, candidate, strategy=strategy.value))
    elif strategy is SearchStrategy.RELOCATE:
        for source_vehicle, source_position, node in positions:
            removed_route = (
                *routes[source_vehicle][:source_position],
                *routes[source_vehicle][source_position + 1 :],
            )
            removed = (*routes[:source_vehicle], removed_route, *routes[source_vehicle + 1 :])
            if problem.type_id is not RouteProblemType.TSP:
                plans.append(make_route_plan(problem, data, removed, strategy=strategy.value))
            for target_vehicle, target_route in enumerate(removed):
                upper = (
                    len(target_route)
                    if not problem.policy.require_return
                    else len(target_route) - 1
                )
                for target_position in range(1, upper + 1):
                    if target_vehicle == source_vehicle and target_position == source_position:
                        continue
                    inserted = _insert(target_route, target_position, node)
                    candidate = (
                        *removed[:target_vehicle],
                        inserted,
                        *removed[target_vehicle + 1 :],
                    )
                    plans.append(make_route_plan(problem, data, candidate, strategy=strategy.value))
        for node in unvisited:
            for vehicle, route in enumerate(routes):
                upper = len(route) if not problem.policy.require_return else len(route) - 1
                for position in range(1, upper + 1):
                    inserted = _insert(route, position, node)
                    candidate = (*routes[:vehicle], inserted, *routes[vehicle + 1 :])
                    plans.append(make_route_plan(problem, data, candidate, strategy=strategy.value))
    elif strategy is SearchStrategy.SWAP:
        for left_index, left_position in enumerate(positions):
            for right_position in positions[left_index + 1 :]:
                route_lists = [list(route) for route in routes]
                (
                    route_lists[left_position[0]][left_position[1]],
                    route_lists[right_position[0]][right_position[1]],
                ) = (
                    route_lists[right_position[0]][right_position[1]],
                    route_lists[left_position[0]][left_position[1]],
                )
                plans.append(
                    make_route_plan(
                        problem,
                        data,
                        tuple(tuple(route) for route in route_lists),
                        strategy=strategy.value,
                    )
                )
        for vehicle, position, _ in positions:
            for replacement in unvisited:
                route_lists = [list(route) for route in routes]
                route_lists[vehicle][position] = replacement
                plans.append(
                    make_route_plan(
                        problem,
                        data,
                        tuple(tuple(route) for route in route_lists),
                        strategy=strategy.value,
                    )
                )
    else:
        raise ValueError(f"unsupported route neighborhood strategy {strategy.value!r}")
    return CandidateSpace(plans=tuple(plans), total=len(plans))


def route_candidate_space(
    problem: RouteServiceProblem,
    data: RouteServiceData,
    incumbent: Plan,
    strategy: SearchStrategy,
) -> CandidateSpace:
    """Resolve a stable route strategy to its deterministic candidate sequence."""

    if strategy is SearchStrategy.EXACT_ENUMERATION:
        return all_route_plans(problem, data)
    if strategy in {SearchStrategy.TWO_OPT, SearchStrategy.RELOCATE, SearchStrategy.SWAP}:
        return route_neighborhood_plans(problem, data, incumbent, strategy)
    raise ValueError("ortools_routing is solved directly rather than enumerated")


def solve_route_ortools(
    problem: RouteServiceProblem,
    data: RouteServiceData,
    *,
    max_time_seconds: float | None = None,
) -> SolverResult:
    """Run a bounded OR-Tools routing strategy; its plan is always independently rescored."""

    if len(data.travel) != 1:
        raise ValueError("ortools_routing currently requires one travel scenario")
    if not problem.policy.require_return:
        raise ValueError("ortools_routing currently requires depot return")
    from ortools.constraint_solver import (  # type: ignore[import-untyped]
        pywrapcp,
        routing_enums_pb2,
    )

    starts = [vehicle_depot_index(data, vehicle) for vehicle in range(problem.policy.vehicle_count)]
    manager = pywrapcp.RoutingIndexManager(
        len(data.node_ids), problem.policy.vehicle_count, starts, starts
    )
    routing = pywrapcp.RoutingModel(manager)
    travel = next(iter(data.travel.values()))
    scale = 1_000

    def transit(from_index: int, to_index: int) -> int:
        left = manager.IndexToNode(from_index)
        right = manager.IndexToNode(to_index)
        value = float(travel[left, right]) + float(data.service_times[left])
        return round(value * scale) if math.isfinite(value) else 10**12

    transit_index = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    routing.AddDimension(
        transit_index,
        round(problem.policy.shift_length * scale),
        round(problem.policy.shift_length * scale),
        True,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    for node in data.service_indices:
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(
            round(float(data.window_starts[node]) * scale),
            round(min(float(data.window_ends[node]), problem.policy.shift_length) * scale),
        )
    if problem.policy.vehicle_capacity is not None:

        def demand(index: int) -> int:
            return round(float(data.demands[manager.IndexToNode(index)]) * scale)

        demand_index = routing.RegisterUnaryTransitCallback(demand)
        routing.AddDimensionWithVehicleCapacity(
            demand_index,
            0,
            [round(problem.policy.vehicle_capacity * scale)] * problem.policy.vehicle_count,
            True,
            "Capacity",
        )
    if problem.type_id is not RouteProblemType.TSP:
        values = data.demands if problem.type_id is RouteProblemType.MOBILE_SERVICE else data.prizes
        for node in data.service_indices:
            penalty = max(1, round(float(values[node]) * scale * 10_000))
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    bounded = max_time_seconds is not None and math.isfinite(max_time_seconds)
    # Guided local search does not terminate by itself. An unlimited invocation
    # uses descent to a local optimum, without imposing a hidden time budget.
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        if bounded
        else routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    )
    if max_time_seconds is not None and bounded:
        milliseconds = max(1, math.floor(max_time_seconds * 1_000))
        parameters.time_limit.seconds = milliseconds // 1_000
        parameters.time_limit.nanos = (milliseconds % 1_000) * 1_000_000
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise ValueError("OR-Tools found no feasible route plan")
    routes: list[tuple[int, ...]] = []
    for vehicle in range(problem.policy.vehicle_count):
        index = routing.Start(vehicle)
        nodes: list[int] = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            nodes.append(manager.IndexToNode(index))
        routes.append(tuple(nodes))
    plan = make_route_plan(
        problem, data, tuple(routes), strategy=SearchStrategy.ORTOOLS_ROUTING.value
    )
    bound = VerifiedBound(
        problem_hash=problem.problem_hash,
        strategy=SearchStrategy.ORTOOLS_ROUTING,
        complete=False,
        explored_candidates=0,
        best_comparator_key=(),
        certificate={
            "solver": "ortools_routing",
            "status": "bounded_solution" if bounded else "local_optimum",
            "objective": int(solution.ObjectiveValue()),
            "integer_scale": scale,
        },
    )
    return SolverResult(plan=plan, bound=bound)
