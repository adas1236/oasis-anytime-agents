"""Stable resumable improvement tool backed by registered search strategies."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oasis.artifacts import put_json
from oasis.problems.location_allocation import create_problem_registry, load_problem_data
from oasis.problems.protocols import Deadline
from oasis.problems.registry import ProblemRegistry
from oasis.problems.schemas import (
    Comparison,
    SearchResumeToken,
    SearchStrategy,
    VerifiedBound,
)
from oasis.problems.search import candidate_space, solve_ortools
from oasis.schemas import (
    ArtifactKind,
    DeterminismClassification,
    SideEffectClassification,
    ToolEvent,
    ToolEventKind,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.decision.common import (
    decision_provenance,
    put_plan_and_scorecard,
    read_plan,
    read_problem,
)
from oasis.tools.evidence.common import MISSING_ARTIFACT_ID, invalid
from oasis.tools.protocols import ToolContext


class ImproveInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    starting_plan_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    strategy: SearchStrategy = SearchStrategy.ADD_SWAP
    max_candidates: int = Field(default=1_000, ge=1, le=1_000_000)
    resume_token: SearchResumeToken | None = None

    @model_validator(mode="after")
    def one_starting_state(self) -> Self:
        if self.starting_plan_artifact_id is not None and self.resume_token is not None:
            raise ValueError("starting_plan_artifact_id and resume_token are mutually exclusive")
        return self


class ImproveOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_hash: str
    strategy: SearchStrategy
    examined_candidates: int = Field(ge=0)
    emitted_improvements: int = Field(ge=0)
    complete: bool
    best_plan_artifact_id: str
    best_scorecard_artifact_id: str
    bound_artifact_id: str | None = None


class ImproveTool:
    """Search one bounded slice while streaming only independently verified improvements."""

    version = "1.0.0"
    spec = ToolSpec(
        name="improve",
        version=version,
        description=(
            "Run a registered location-allocation improvement strategy, streaming every "
            "independently rescored feasible improvement and any verified search bound."
        ),
        input_schema=ImproveInput.model_json_schema(),
        output_schema=ImproveOutput.model_json_schema(),
        capability_tags=frozenset({"decision", "search", "location_allocation", "offline"}),
        problem_tags=frozenset({"location_allocation"}),
        artifact_tags=frozenset(
            {
                ArtifactKind.JSON_SPECIFICATION,
                ArtifactKind.PLAN,
                ArtifactKind.SCORECARD,
                ArtifactKind.TRACE_ATTACHMENT,
            }
        ),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        seed_description=(
            "all strategy and tie-breaking order is deterministic; seed is recorded only"
        ),
        runtime=ToolRuntimeEstimate(p50_ms=20, p95_ms=10_000, time_to_first_candidate_ms=20),
        streams_progress=True,
        streams_candidates=True,
        streams_bounds=True,
        cooperative_cancellation=False,
        resumable=True,
        resume_token_schema=SearchResumeToken.model_json_schema(),
        smoke_input={
            "problem_artifact_id": MISSING_ARTIFACT_ID,
            "strategy": "add_swap",
            "max_candidates": 1,
        },
    )

    def __init__(self, registry: ProblemRegistry | None = None) -> None:
        self._registry = registry or create_problem_registry()

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        """Non-streaming convenience path returning the terminal envelope."""

        terminal: ToolResult | None = None
        async for event in self.stream(arguments, context):
            if event.kind is ToolEventKind.RESULT:
                terminal = event.result
        if terminal is None:
            raise RuntimeError("improvement stream ended without a result")
        return terminal

    async def stream(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        request = ImproveInput.model_validate(arguments)
        problem_ref, problem = read_problem(context, request.problem_artifact_id)
        plugin = self._registry.get(problem.type_id.value)
        report = plugin.validate_spec(problem, context.artifact_store)
        if not report.valid:
            invalid(f"invalid problem: {report.issues[0].message}")
        next_index = 0
        plan_parent = None
        if request.resume_token is not None:
            token = request.resume_token
            if token.problem_hash != problem.problem_hash or token.strategy is not request.strategy:
                invalid("resume token does not belong to this problem and strategy")
            incumbent = token.incumbent
            next_index = token.next_index
        elif request.starting_plan_artifact_id is not None:
            plan_parent, incumbent = read_plan(context, request.starting_plan_artifact_id)
        else:
            incumbent = plugin.make_baseline(
                problem,
                context.artifact_store,
                Deadline(context.deadline_monotonic, context.monotonic),
            )
        incumbent_report = plugin.validate_plan(problem, incumbent, context.artifact_store)
        if not incumbent_report.valid:
            invalid(f"starting plan is invalid: {incumbent_report.issues[0].message}")
        incumbent_score = plugin.measure(problem, incumbent, context.artifact_store)
        sequence = 0
        examined = 0
        improvements = 0
        complete = True
        bound: VerifiedBound | None = None

        if request.strategy is SearchStrategy.ORTOOLS_CP_SAT:
            context.cancellation.raise_if_cancelled()
            data = load_problem_data(problem, context.artifact_store)
            solved = solve_ortools(problem, data, max_time_seconds=context.remaining_seconds)
            score = plugin.measure(problem, solved.plan, context.artifact_store)
            if not score.feasible:
                raise ValueError("OR-Tools candidate failed independent evaluation")
            examined = 1
            if plugin.compare(score, incumbent_score) is Comparison.BETTER:
                incumbent, incumbent_score = solved.plan, score
                improvements = 1
                yield ToolEvent(
                    sequence=sequence,
                    kind=ToolEventKind.CANDIDATE,
                    message="verified OR-Tools improvement",
                    candidate=incumbent,
                )
                sequence += 1
            bound = solved.bound.model_copy(
                update={"best_comparator_key": incumbent_score.comparator_key}
            )
            complete = bound.complete
        else:
            data = load_problem_data(problem, context.artifact_store)
            space = candidate_space(problem, data, incumbent, request.strategy)
            exhausted = True
            resume_cursor = next_index
            for cursor, candidate in enumerate(space.plans):
                if cursor < next_index:
                    continue
                if examined >= request.max_candidates or context.remaining_seconds <= 0.001:
                    exhausted = False
                    resume_cursor = cursor
                    break
                context.cancellation.raise_if_cancelled()
                score = plugin.measure(problem, candidate, context.artifact_store)
                examined += 1
                resume_cursor = cursor + 1
                if score.feasible and plugin.compare(score, incumbent_score) is Comparison.BETTER:
                    incumbent, incumbent_score = candidate, score
                    improvements += 1
                    yield ToolEvent(
                        sequence=sequence,
                        kind=ToolEventKind.CANDIDATE,
                        message="independently verified improving plan",
                        candidate=candidate,
                    )
                    sequence += 1
                if examined % 50 == 0:
                    progress = (
                        min(1.0, (cursor + 1) / space.total)
                        if space.total is not None and space.total
                        else 0.0
                    )
                    yield ToolEvent(
                        sequence=sequence,
                        kind=ToolEventKind.PROGRESS,
                        message=f"examined {examined} candidates",
                        progress=progress,
                    )
                    sequence += 1
            complete = exhausted
            next_index = resume_cursor
            if (
                request.strategy is SearchStrategy.EXACT_ENUMERATION
                and complete
                and request.resume_token is None
            ):
                bound = VerifiedBound(
                    problem_hash=problem.problem_hash,
                    strategy=request.strategy,
                    complete=True,
                    explored_candidates=next_index,
                    total_candidates=space.total,
                    best_comparator_key=incumbent_score.comparator_key,
                    certificate={
                        "method": "complete_enumeration",
                        "candidate_order": "site-count then lexicographic candidate index",
                    },
                )

        parents = (problem_ref,) if plan_parent is None else (problem_ref, plan_parent)
        if bound is not None and bound.complete:
            incumbent_score = incumbent_score.model_copy(
                update={
                    "verified_lower_bound": incumbent_score.comparator_key,
                    "verified_upper_bound": incumbent_score.comparator_key,
                    "optimality_gap": 0.0,
                }
            )
        plan_ref, score_ref = put_plan_and_scorecard(
            context,
            incumbent,
            incumbent_score,
            name=self.spec.name,
            version=self.version,
            parents=parents,
            parameters={
                "strategy": request.strategy.value,
                "examined_candidates": examined,
                "seed": context.seed,
            },
        )
        bound_ref = None
        if bound is not None:
            bound_ref = put_json(
                context.artifact_store,
                bound.model_dump(mode="json"),
                kind=ArtifactKind.TRACE_ATTACHMENT,
                units="unitless",
                provenance=decision_provenance(
                    self.spec.name,
                    self.version,
                    (problem_ref, plan_ref, score_ref),
                    {"strategy": request.strategy.value, "role": "verified_bound"},
                ),
                data_schema={"type": "VerifiedBound", "version": "1.0.0"},
            )
            yield ToolEvent(
                sequence=sequence,
                kind=ToolEventKind.BOUND,
                message="verified search bound",
                bound=bound_ref,
            )
            sequence += 1
        resume = None
        if not complete:
            resume = SearchResumeToken(
                problem_hash=problem.problem_hash,
                strategy=request.strategy,
                next_index=next_index,
                incumbent=incumbent,
            )
        output = ImproveOutput(
            problem_hash=problem.problem_hash,
            strategy=request.strategy,
            examined_candidates=examined,
            emitted_improvements=improvements,
            complete=complete,
            best_plan_artifact_id=plan_ref.id,
            best_scorecard_artifact_id=score_ref.id,
            bound_artifact_id=bound_ref.id if bound_ref is not None else None,
        )
        status = ToolResultStatus.COMPLETE if complete else ToolResultStatus.PARTIAL
        yield ToolEvent(
            sequence=sequence,
            kind=ToolEventKind.RESULT,
            result=ToolResult(
                status=status,
                summary={
                    "strategy": request.strategy.value,
                    "examined": examined,
                    "improvements": improvements,
                    "complete": complete,
                    "best_plan": plan_ref.id,
                    "best_scorecard": score_ref.id,
                },
                artifacts=tuple(
                    reference
                    for reference in (plan_ref, score_ref, bound_ref)
                    if reference is not None
                ),
                metrics=output.model_dump(mode="json"),
                candidate=incumbent,
                bound=bound_ref,
                resume_token=resume.model_dump(mode="json") if resume is not None else None,
            ),
        )
