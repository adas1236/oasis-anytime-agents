"""Independent exact or bounded reference generation for benchmark instances."""

from __future__ import annotations

import time
from collections.abc import Iterable
from itertools import islice

from pydantic import TypeAdapter

from oasis.artifacts import ArtifactStore, put_json, read_json
from oasis.evaluation.models import (
    GeneratedInstance,
    InstanceScale,
    ReferenceKind,
    ReferenceSolution,
)
from oasis.problems import (
    Comparison,
    LocationAllocationProblem,
    RouteServiceProblem,
    Scorecard,
    SearchStrategy,
    VerifiedBound,
    create_builtin_problem_registry,
    load_problem_data,
    load_route_data,
)
from oasis.problems.routing_search import all_route_plans
from oasis.problems.search import CandidateSpace, all_candidate_plans
from oasis.schemas import ArtifactKind, Plan
from oasis.tools import CancellationToken, ToolContext
from oasis.tools.decision.common import decision_provenance, put_plan_and_scorecard

_PROBLEM_ADAPTER: TypeAdapter[LocationAllocationProblem | RouteServiceProblem] = TypeAdapter(
    LocationAllocationProblem | RouteServiceProblem
)


def _candidate_space(
    problem: LocationAllocationProblem | RouteServiceProblem,
    store: ArtifactStore,
) -> CandidateSpace:
    if isinstance(problem, LocationAllocationProblem):
        return all_candidate_plans(problem, load_problem_data(problem, store))
    return all_route_plans(problem, load_route_data(problem, store))


def _bounded(plans: Iterable[Plan], limit: int) -> Iterable[Plan]:
    return islice(plans, limit)


def build_reference(
    instance: GeneratedInstance,
    store: ArtifactStore,
    *,
    max_exact_candidates: int,
    max_reference_candidates: int,
) -> ReferenceSolution | None:
    """Build an exact small oracle or independently scored medium/stress best-known plan."""

    if instance.problem_artifact_id is None:
        return None
    problem_ref = store.get_metadata(instance.problem_artifact_id)
    problem = _PROBLEM_ADAPTER.validate_python(read_json(store, problem_ref))
    plugin = create_builtin_problem_registry().get(problem.type_id.value)
    space = _candidate_space(problem, store)
    exact = space.total is not None and space.total <= max_exact_candidates
    limit = space.total if exact else max_reference_candidates
    assert limit is not None
    best_plan: Plan | None = None
    best_score: Scorecard | None = None
    evaluated = 0
    for plan in _bounded(space.plans, limit):
        evaluated += 1
        score = plugin.measure(problem, plan, store)
        if not score.feasible:
            continue
        if best_score is None or plugin.compare(score, best_score) is Comparison.BETTER:
            best_plan, best_score = plan, score

    complete = exact and evaluated == space.total
    if best_plan is None or best_score is None:
        if complete:
            return ReferenceSolution(
                kind=ReferenceKind.INFEASIBLE,
                evaluated_candidates=evaluated,
                total_candidates=space.total,
                method="complete_independent_enumeration",
            )
        return None

    context = ToolContext(
        run_id=f"reference-{instance.id}",
        artifact_store=store,
        deadline_monotonic=time.monotonic() + 60.0,
        cancellation=CancellationToken(),
        seed=instance.effective_seed,
    )
    plan_ref, score_ref = put_plan_and_scorecard(
        context,
        best_plan,
        best_score,
        name="evaluation_reference",
        version="1.0.0",
        parents=(problem_ref,),
        parameters={
            "scale": instance.generator_spec.scale.value,
            "evaluated_candidates": evaluated,
            "complete": complete,
        },
    )
    bound_ref = None
    if complete:
        bound = VerifiedBound(
            problem_hash=problem.problem_hash,
            strategy=SearchStrategy.EXACT_ENUMERATION,
            complete=True,
            explored_candidates=evaluated,
            total_candidates=space.total,
            best_comparator_key=best_score.comparator_key,
            certificate={
                "method": "complete_independent_enumeration",
                "evaluator_version": problem.evaluator_version,
            },
        )
        bound_ref = put_json(
            store,
            bound.model_dump(mode="json"),
            kind=ArtifactKind.TRACE_ATTACHMENT,
            units="unitless",
            provenance=decision_provenance(
                "evaluation_exact_oracle",
                "1.0.0",
                (problem_ref, plan_ref, score_ref),
                {"evaluated_candidates": evaluated},
            ),
            data_schema={"type": "VerifiedBound", "version": "1.0.0"},
        )
    kind = ReferenceKind.EXACT_OPTIMUM if complete else ReferenceKind.BEST_KNOWN
    method = (
        "complete_independent_enumeration"
        if complete
        else f"first_{evaluated}_independently_evaluated_candidates"
    )
    return ReferenceSolution(
        kind=kind,
        plan=best_plan,
        scorecard=best_score,
        plan_artifact_id=plan_ref.id,
        scorecard_artifact_id=score_ref.id,
        bound_artifact_id=bound_ref.id if bound_ref is not None else None,
        evaluated_candidates=evaluated,
        total_candidates=space.total,
        method=method,
    )


def attach_reference(
    instance: GeneratedInstance,
    store: ArtifactStore,
    *,
    max_exact_candidates: int,
    max_reference_candidates: int,
) -> GeneratedInstance:
    """Return an instance with the scale-appropriate independently evaluated reference."""

    reference_limit = (
        max_reference_candidates
        if instance.generator_spec.scale in {InstanceScale.MEDIUM, InstanceScale.STRESS}
        else max_exact_candidates
    )
    reference = build_reference(
        instance,
        store,
        max_exact_candidates=max_exact_candidates,
        max_reference_candidates=reference_limit,
    )
    return instance.model_copy(update={"reference": reference})
