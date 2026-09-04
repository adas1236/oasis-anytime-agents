"""Framework-neutral anytime controller with authoritative incumbent ownership."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from oasis.artifacts import ArtifactStore, read_json
from oasis.controller.budget import BudgetAccount, BudgetExceededError, Deadline
from oasis.controller.incumbent import IncumbentStore
from oasis.controller.schemas import (
    ActionRecord,
    ActionStatus,
    BudgetTier,
    CallToolAction,
    CompactModelState,
    CompactTool,
    ControllerAction,
    ControllerPolicy,
    ControllerState,
    EventActor,
    EventKind,
    IncumbentRecord,
    RunRequest,
    RunResult,
    RunStatus,
    StopAction,
    SubmitCandidateAction,
    TerminalReason,
)
from oasis.controller.state import (
    ActionLedger,
    EventCallback,
    EventJournal,
    StateMachine,
    action_fingerprint,
)
from oasis.controller.store import RunMetadata, RunStore
from oasis.errors import ModelBackendError, ToolCallParseError
from oasis.llm import ChatMessage, ChatRole, ModelBackend, ModelRequest, ModelTurn, TokenUsage
from oasis.problems import (
    Deadline as ProblemDeadline,
)
from oasis.problems import (
    ProblemPlugin,
    ProblemRegistry,
    VerifiedBound,
    create_builtin_problem_registry,
)
from oasis.runtimes import ComputeInventory, RuntimePlan
from oasis.schemas import ArtifactRef, Plan, ToolEventKind, ToolResult, ToolResultStatus
from oasis.schemas.artifacts import PrivacyClassification
from oasis.tools import (
    CancellationToken,
    StreamingTool,
    ToolContext,
    ToolRegistry,
    ToolRegistryError,
    create_tool_registry,
    invoke_tool,
    stream_tool,
)
from oasis.tools.decision.common import put_plan_and_scorecard, read_plan, read_problem
from oasis.tools.protocols import ToolExecutionError
from oasis.tools.registry import validate_arguments

CONTROLLER_VERSION = "1.0.0"
_ACTION_ADAPTER: TypeAdapter[ControllerAction] = TypeAdapter(ControllerAction)


class ModelBudgetUnavailable(RuntimeError):
    """Raised when another model request cannot fit the aggregate token budget."""


class UnsupportedProblemVersion(RuntimeError):
    """Raised when a persisted problem names a different plugin implementation version."""


def _natural_tier(request: RunRequest, backend: ModelBackend | None) -> BudgetTier:
    if request.budget.max_tool_calls == 0:
        return BudgetTier.BASELINE_ONLY
    model_available = (
        request.enable_model
        and backend is not None
        and backend.capabilities.generative
        and (backend.capabilities.native_tools or backend.capabilities.structured_fallback)
        and request.budget.max_total_model_tokens > 0
        and request.budget.max_generated_tokens > 0
    )
    if not model_available:
        return (
            BudgetTier.DETERMINISTIC_IMPROVEMENT
            if request.enable_deterministic_fallback
            else BudgetTier.BASELINE_ONLY
        )
    return BudgetTier.ITERATIVE_MODEL


_TIER_ORDER = {
    BudgetTier.BASELINE_ONLY: 0,
    BudgetTier.DETERMINISTIC_IMPROVEMENT: 1,
    BudgetTier.ONE_SHOT_MODEL: 2,
    BudgetTier.ITERATIVE_MODEL: 3,
}


class AnytimeController:
    """Run one immutable problem while retaining a feasible baseline-or-better plan."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        run_store: RunStore,
        backend: ModelBackend | None = None,
        tool_registry: ToolRegistry | None = None,
        problem_registry: ProblemRegistry | None = None,
        policy: ControllerPolicy | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        event_callback: EventCallback | None = None,
        providers: Mapping[str, object] | None = None,
        resources: Mapping[str, object] | None = None,
        allowed_privacy: frozenset[PrivacyClassification] = frozenset(
            {PrivacyClassification.PUBLIC}
        ),
    ) -> None:
        self._artifacts = artifact_store
        self._runs = run_store
        self._backend = backend
        self._tools = tool_registry or create_tool_registry(discover_entry_points=False)
        self._problems = problem_registry or create_builtin_problem_registry()
        self._policy = policy or ControllerPolicy()
        self._monotonic = monotonic
        self._utcnow = utcnow
        self._event_callback = event_callback
        self._providers = dict(providers or {})
        self._resources = dict(resources or {})
        self._allowed_privacy = allowed_privacy

    def _tier(self, request: RunRequest) -> BudgetTier:
        natural = _natural_tier(request, self._backend)
        if natural is BudgetTier.ITERATIVE_MODEL and (
            request.budget.max_total_model_tokens <= self._policy.one_shot_total_token_threshold
        ):
            natural = BudgetTier.ONE_SHOT_MODEL
        if request.requested_tier is None:
            return natural
        return min((natural, request.requested_tier), key=_TIER_ORDER.__getitem__)

    async def run(
        self,
        request: RunRequest,
        *,
        cancellation: CancellationToken | None = None,
        started_at_monotonic: float | None = None,
        initial_tool_calls: int = 0,
    ) -> RunResult:
        """Execute and persist a response-anytime run under absolute aggregate budgets."""

        run_cancellation = cancellation or CancellationToken()
        state = StateMachine()
        deadline = Deadline(
            request.budget,
            self._policy,
            monotonic=self._monotonic,
            started_at=started_at_monotonic,
        )
        budget = BudgetAccount(request.budget, deadline)
        for _ in range(initial_tool_calls):
            if not budget.tools.admit():
                raise ValueError("initial tool calls exceed the declared tool-call budget")
        tier = self._tier(request)
        self._runs.create(
            RunMetadata(
                run_id=request.run_id,
                run_generation=request.run_generation,
                problem_artifact_id=request.problem_artifact_id,
                seed=request.seed,
                metadata={
                    "budget_tier": tier.value,
                    "controller_version": CONTROLLER_VERSION,
                    "runtime_plan": request.runtime_plan.model_dump(mode="json"),
                    "compute_inventory": request.compute_inventory.sanitized().model_dump(
                        mode="json"
                    ),
                },
            )
        )
        journal = EventJournal(
            run_id=request.run_id,
            run_generation=request.run_generation,
            deadline=deadline,
            budget=budget,
            append=self._runs.append_event,
            state=state,
            utcnow=self._utcnow,
            callback=self._event_callback,
        )
        action_ledger = ActionLedger()
        failures: list[str] = []
        warnings: list[str] = []
        problem: Any | None = None
        problem_ref: ArtifactRef | None = None
        plugin: ProblemPlugin | None = None
        incumbents: IncumbentStore | None = None
        baseline: IncumbentRecord | None = None
        verified_bound_id: str | None = None
        reason = TerminalReason.INTERNAL_FAILURE

        await journal.emit(
            EventKind.RUN_CREATED,
            payload={"budget_tier": tier.value, "seed": request.seed},
        )
        try:
            state.transition(ControllerState.GROUNDING)
            if run_cancellation.cancelled:
                reason = TerminalReason.USER_CANCELLED
            else:
                context = self._tool_context(
                    request,
                    deadline.at_monotonic,
                    run_cancellation,
                )
                problem_ref, problem = read_problem(context, request.problem_artifact_id)
                plugin = self._problems.get(problem.type_id.value)
                if problem.plugin_version != plugin.version:
                    raise UnsupportedProblemVersion(
                        f"problem requires plugin version {problem.plugin_version}, "
                        f"but {plugin.version} is installed"
                    )
                report = plugin.validate_spec(problem, self._artifacts)
                if not report.valid:
                    failures.extend(issue.message for issue in report.issues)
                    reason = TerminalReason.INFEASIBLE_PROBLEM
                else:
                    await journal.emit(
                        EventKind.EVIDENCE_SNAPSHOT_LOCKED,
                        artifact_ids=(problem_ref.id,),
                        payload={
                            "evidence_hash": problem.evidence_hash,
                            "policy_hash": problem.policy_hash,
                        },
                    )
                    state.transition(ControllerState.PROBLEM_LOCKED)
                    await journal.emit(
                        EventKind.PROBLEM_COMPILED,
                        artifact_ids=(problem_ref.id,),
                        payload={
                            "problem_type": problem.type_id.value,
                            "problem_hash": problem.problem_hash,
                            "evaluator_version": problem.evaluator_version,
                        },
                    )
                    if (
                        deadline.search_remaining_ms
                        < self._policy.minimum_baseline_budget_ms
                        + self._policy.validation_reserve_ms
                    ):
                        reason = TerminalReason.BUDGET_TOO_SMALL
                    elif self._cancelled(run_cancellation):
                        reason = TerminalReason.USER_CANCELLED
                    else:
                        state.transition(ControllerState.ADMITTED)
                        await journal.emit(
                            EventKind.BUDGET_CHECKPOINT,
                            payload={"checkpoint": "admitted"},
                        )
                        incumbents = IncumbentStore(plugin.compare)
                        baseline = await self._commit_baseline(
                            request=request,
                            problem=problem,
                            problem_ref=problem_ref,
                            plugin=plugin,
                            incumbents=incumbents,
                            deadline=deadline,
                        )
                        state.transition(ControllerState.BASELINE_COMMITTED)
                        await journal.emit(
                            EventKind.BASELINE_COMMITTED,
                            actor=EventActor.EVALUATOR,
                            artifact_ids=(
                                baseline.plan_artifact_id,
                                baseline.scorecard_artifact_id,
                            ),
                            payload={"comparator_key": list(baseline.comparator_key)},
                        )
                        if self._cancelled(run_cancellation):
                            reason = TerminalReason.USER_CANCELLED
                        else:
                            state.transition(ControllerState.SEARCHING)
                            await journal.emit(
                                EventKind.BUDGET_CHECKPOINT,
                                payload={"checkpoint": "search_started"},
                            )
                            reason, verified_bound_id = await self._search(
                                request=request,
                                tier=tier,
                                problem=problem,
                                problem_ref=problem_ref,
                                plugin=plugin,
                                incumbents=incumbents,
                                deadline=deadline,
                                budget=budget,
                                journal=journal,
                                action_ledger=action_ledger,
                                run_cancellation=run_cancellation,
                                failures=failures,
                            )
        except (ToolExecutionError, ToolRegistryError, KeyError) as error:
            failures.append(str(error))
            reason = (
                TerminalReason.MISSING_EVIDENCE
                if "not found" in str(error).lower()
                else TerminalReason.INVALID_REQUEST
            )
        except (ValueError, ValidationError) as error:
            failures.append(str(error))
            reason = TerminalReason.INVALID_REQUEST
        except UnsupportedProblemVersion as error:
            failures.append(str(error))
            reason = TerminalReason.UNSUPPORTED_CAPABILITY
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
            reason = TerminalReason.INTERNAL_FAILURE

        if state.state is not ControllerState.FINALIZED:
            if state.state is not ControllerState.QUIESCING:
                state.transition(ControllerState.QUIESCING)
            await journal.emit(
                EventKind.BUDGET_CHECKPOINT,
                payload={"checkpoint": "quiescing", "reason": reason.value},
            )
            state.transition(ControllerState.FINALIZED)

        current = incumbents.current if incumbents is not None else None
        view = (
            plugin.render_result(problem, current.plan, current.scorecard)
            if plugin is not None and problem is not None and current is not None
            else None
        )
        comparison = (
            plugin.compare(current.scorecard, baseline.scorecard)
            if plugin is not None and current is not None and baseline is not None
            else None
        )
        await journal.emit(
            EventKind.RUN_FINALIZED,
            artifact_ids=(
                tuple(
                    value
                    for value in (
                        current.plan_artifact_id if current is not None else None,
                        current.scorecard_artifact_id if current is not None else None,
                        verified_bound_id,
                    )
                    if value is not None
                )
            ),
            payload={"reason": reason.value, "has_incumbent": current is not None},
        )
        result = RunResult(
            run_id=request.run_id,
            run_generation=request.run_generation,
            status=self._result_status(reason, current is not None),
            terminal_reason=reason,
            budget_tier=tier,
            problem_artifact_id=request.problem_artifact_id,
            problem_hash=problem.problem_hash if problem is not None else None,
            evidence_hash=problem.evidence_hash if problem is not None else None,
            policy_hash=problem.policy_hash if problem is not None else None,
            best_plan=current.plan if current is not None else None,
            best_scorecard=current.scorecard if current is not None else None,
            best_plan_artifact_id=current.plan_artifact_id if current is not None else None,
            best_scorecard_artifact_id=(
                current.scorecard_artifact_id if current is not None else None
            ),
            result_view=view,
            baseline_comparison=comparison,
            verified_bound_artifact_id=verified_bound_id,
            requested_budget=request.budget,
            consumed_budget=budget.snapshot(),
            deadline_overshoot_ms=deadline.overshoot_ms,
            time_to_first_feasible_ms=(baseline.committed_at_ms if baseline is not None else None),
            incumbent_timeline=incumbents.timeline if incumbents is not None else (),
            warnings=tuple(warnings),
            failures=tuple(failures),
            runtime_plan=self._final_runtime_plan(request),
            compute_inventory=self._final_compute_inventory(request),
            hardware_validation=self._final_runtime_plan(request).hardware_validation.value,
            model_profile=self._backend.profile.name if self._backend is not None else None,
            model_id=self._backend.profile.model_id if self._backend is not None else None,
            problem_plugin_version=(problem.plugin_version if problem is not None else None),
            evaluator_version=(problem.evaluator_version if problem is not None else None),
            tool_versions={spec.name: spec.version for spec in self._tools.list()},
            controller_version=CONTROLLER_VERSION,
            seed=request.seed,
            event_count=journal.count,
        )
        self._runs.write_result(result)
        return result

    def _final_runtime_plan(self, request: RunRequest) -> RuntimePlan:
        plan = getattr(self._backend, "runtime_plan", None)
        if not isinstance(plan, RuntimePlan):
            return request.runtime_plan
        metrics = plan.metrics.model_copy(
            update={
                "startup_ms": max(
                    plan.metrics.startup_ms,
                    request.runtime_plan.metrics.startup_ms,
                )
            }
        )
        return plan.model_copy(update={"metrics": metrics})

    def _final_compute_inventory(self, request: RunRequest) -> ComputeInventory:
        inventory = getattr(self._backend, "compute_inventory", None)
        resolved = (
            inventory if isinstance(inventory, ComputeInventory) else request.compute_inventory
        )
        return resolved.sanitized()

    def _tool_context(
        self,
        request: RunRequest,
        at_monotonic: float,
        cancellation: CancellationToken,
    ) -> ToolContext:
        return ToolContext(
            run_id=request.run_id,
            artifact_store=self._artifacts,
            deadline_monotonic=at_monotonic,
            cancellation=cancellation,
            seed=request.seed,
            providers=self._providers,
            resources=self._resources,
            allowed_privacy=self._allowed_privacy,
            monotonic=self._monotonic,
        )

    async def _commit_baseline(
        self,
        *,
        request: RunRequest,
        problem: Any,
        problem_ref: ArtifactRef,
        plugin: ProblemPlugin,
        incumbents: IncumbentStore,
        deadline: Deadline,
    ) -> IncumbentRecord:
        context = self._tool_context(
            request,
            deadline.search_deadline_monotonic,
            CancellationToken(),
        )
        if request.baseline_plan_artifact_id is None:
            plan = plugin.make_baseline(
                problem,
                self._artifacts,
                ProblemDeadline(deadline.search_deadline_monotonic, self._monotonic),
            )
        else:
            _, plan = read_plan(context, request.baseline_plan_artifact_id)
        report = plugin.validate_plan(problem, plan, self._artifacts)
        if not report.valid:
            raise ValueError(f"baseline is invalid: {report.issues[0].message}")
        score = plugin.measure(problem, plan, self._artifacts)
        if not score.feasible:
            raise ValueError("baseline evaluator did not produce a feasible scorecard")
        plan_ref, score_ref = put_plan_and_scorecard(
            context,
            plan,
            score,
            name="anytime_controller",
            version=CONTROLLER_VERSION,
            parents=(problem_ref,),
            parameters={"role": "baseline", "seed": request.seed},
        )
        committed = await incumbents.try_commit(
            plan=plan,
            scorecard=score,
            plan_artifact_id=plan_ref.id,
            scorecard_artifact_id=score_ref.id,
            source_action_id="baseline",
            committed_at_ms=deadline.elapsed_ms,
            seed=request.seed,
            committed_at=self._utcnow(),
        )
        if committed is None:
            raise RuntimeError("failed to atomically commit a feasible baseline")
        return committed

    async def _search(
        self,
        *,
        request: RunRequest,
        tier: BudgetTier,
        problem: Any,
        problem_ref: ArtifactRef,
        plugin: ProblemPlugin,
        incumbents: IncumbentStore,
        deadline: Deadline,
        budget: BudgetAccount,
        journal: EventJournal,
        action_ledger: ActionLedger,
        run_cancellation: CancellationToken,
        failures: list[str],
    ) -> tuple[TerminalReason, str | None]:
        if tier is BudgetTier.BASELINE_ONLY:
            return TerminalReason.BASELINE_ONLY, None
        if tier is BudgetTier.DETERMINISTIC_IMPROVEMENT:
            return await self._fallback(
                request=request,
                problem=problem,
                problem_ref=problem_ref,
                plugin=plugin,
                incumbents=incumbents,
                deadline=deadline,
                budget=budget,
                journal=journal,
                action_ledger=action_ledger,
                run_cancellation=run_cancellation,
                failures=failures,
            )

        model_limit = 1 if tier is BudgetTier.ONE_SHOT_MODEL else self._policy.max_model_actions
        recent: list[str] = []
        no_progress = 0
        verified_bound_id: str | None = None
        for _ in range(model_limit):
            terminal = self._budget_terminal(deadline, budget, run_cancellation)
            if terminal is not None:
                if (
                    terminal is TerminalReason.TOKEN_EXHAUSTED_NO_DETERMINISTIC_WORK
                    and request.enable_deterministic_fallback
                ):
                    return await self._fallback(
                        request=request,
                        problem=problem,
                        problem_ref=problem_ref,
                        plugin=plugin,
                        incumbents=incumbents,
                        deadline=deadline,
                        budget=budget,
                        journal=journal,
                        action_ledger=action_ledger,
                        run_cancellation=run_cancellation,
                        failures=failures,
                    )
                return terminal, verified_bound_id
            try:
                action = await self._propose_action(
                    request=request,
                    problem=problem,
                    incumbents=incumbents,
                    deadline=deadline,
                    budget=budget,
                    journal=journal,
                    recent=recent,
                    failures=failures,
                )
            except ModelBudgetUnavailable:
                if request.enable_deterministic_fallback:
                    return await self._fallback(
                        request=request,
                        problem=problem,
                        problem_ref=problem_ref,
                        plugin=plugin,
                        incumbents=incumbents,
                        deadline=deadline,
                        budget=budget,
                        journal=journal,
                        action_ledger=action_ledger,
                        run_cancellation=run_cancellation,
                        failures=failures,
                    )
                return TerminalReason.TOKEN_EXHAUSTED_NO_DETERMINISTIC_WORK, None
            if action is None:
                no_progress += 1
                recent.append("malformed model action rejected")
            elif isinstance(action, StopAction):
                return TerminalReason.MODEL_STOPPED, verified_bound_id
            else:
                fingerprint = action_fingerprint(
                    action.model_dump(mode="json", exclude={"rationale"})
                )
                if action_ledger.is_duplicate(fingerprint):
                    no_progress += 1
                    await journal.emit(
                        EventKind.ACTION_REJECTED,
                        actor=EventActor.CONTROLLER,
                        payload={"reason": "duplicate_action", "action_type": action.type},
                    )
                    recent.append("duplicate action rejected")
                elif isinstance(action, SubmitCandidateAction):
                    record = action_ledger.admit(
                        fingerprint=fingerprint,
                        admitted_at_ms=deadline.elapsed_ms,
                    )
                    await journal.emit(
                        EventKind.ACTION_ADMITTED,
                        actor=EventActor.CONTROLLER,
                        action=record,
                        payload={"action_type": action.type},
                    )
                    committed = await self._evaluate_candidate(
                        request=request,
                        problem=problem,
                        problem_ref=problem_ref,
                        plugin=plugin,
                        incumbents=incumbents,
                        candidate=action.candidate,
                        deadline=deadline,
                        journal=journal,
                        action=record,
                    )
                    action_ledger.mark(record, ActionStatus.COMPLETED)
                    no_progress = 0 if committed else no_progress + 1
                    recent.append(
                        "direct candidate improved" if committed else "candidate rejected"
                    )
                else:
                    improved, bound_id, optimal = await self._execute_tool_action(
                        request=request,
                        problem=problem,
                        problem_ref=problem_ref,
                        plugin=plugin,
                        incumbents=incumbents,
                        action=action,
                        fingerprint=fingerprint,
                        deadline=deadline,
                        budget=budget,
                        journal=journal,
                        action_ledger=action_ledger,
                        run_cancellation=run_cancellation,
                        failures=failures,
                    )
                    verified_bound_id = bound_id or verified_bound_id
                    if optimal:
                        return TerminalReason.PROVEN_OPTIMAL, verified_bound_id
                    no_progress = 0 if improved else no_progress + 1
                    recent.append(
                        f"tool {action.tool} improved"
                        if improved
                        else f"tool {action.tool} stalled"
                    )
            if run_cancellation.cancelled:
                return TerminalReason.USER_CANCELLED, verified_bound_id
            if deadline.search_expired:
                return TerminalReason.TIME_EXHAUSTED, verified_bound_id
            recent[:] = recent[-self._policy.recent_action_limit :]
            if no_progress >= self._policy.max_no_progress_actions:
                if request.enable_deterministic_fallback:
                    return await self._fallback(
                        request=request,
                        problem=problem,
                        problem_ref=problem_ref,
                        plugin=plugin,
                        incumbents=incumbents,
                        deadline=deadline,
                        budget=budget,
                        journal=journal,
                        action_ledger=action_ledger,
                        run_cancellation=run_cancellation,
                        failures=failures,
                    )
                return TerminalReason.PLATEAU, verified_bound_id

        if request.enable_deterministic_fallback:
            return await self._fallback(
                request=request,
                problem=problem,
                problem_ref=problem_ref,
                plugin=plugin,
                incumbents=incumbents,
                deadline=deadline,
                budget=budget,
                journal=journal,
                action_ledger=action_ledger,
                run_cancellation=run_cancellation,
                failures=failures,
            )
        return TerminalReason.BUDGET_TIER_COMPLETE, verified_bound_id

    async def _propose_action(
        self,
        *,
        request: RunRequest,
        problem: Any,
        incumbents: IncumbentStore,
        deadline: Deadline,
        budget: BudgetAccount,
        journal: EventJournal,
        recent: list[str],
        failures: list[str],
    ) -> ControllerAction | None:
        if self._backend is None or incumbents.current is None:
            raise ModelBudgetUnavailable("no model backend or incumbent is available")
        repair_message: str | None = None
        for attempt in range(self._policy.schema_repair_attempts + 1):
            model_budget_before = budget.snapshot()
            compact = self._compact_state(
                request=request,
                problem=problem,
                incumbent=incumbents.current,
                deadline=deadline,
                budget=budget,
                recent=recent,
            )
            prompt = compact.model_dump_json(exclude_none=True)
            if repair_message is not None:
                prompt += "\n" + repair_message
            compact_context_bytes = len(prompt.encode("utf-8"))
            messages = (
                ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=(
                        "Return exactly one controller action: call_tool, submit_candidate, or "
                        "stop. Use one native tool call or one JSON object and a short rationale."
                    ),
                ),
                ChatMessage(role=ChatRole.USER, content=prompt),
            )
            definitions = self._tools.model_definitions(
                self._tools.get(name).spec for name in request.allowed_tools
            )
            provisional = ModelRequest(
                request_id=f"{request.run_id}-model-{budget.tokens.usage.total_tokens}-{attempt}",
                messages=messages,
                max_generated_tokens=1,
                thinking_enabled=request.thinking_enabled,
                tools=definitions,
                seed=request.seed,
            )
            try:
                estimated_input = await self._count_input_with_deadline(provisional, deadline)
                allowance = budget.tokens.generation_allowance(
                    estimated_input_tokens=estimated_input,
                    requested=min(512, budget.tokens.remaining_generated),
                )
                if allowance < 1:
                    raise ModelBudgetUnavailable("another model action cannot fit the token budget")
                model_request = provisional.model_copy(update={"max_generated_tokens": allowance})
                turn = await self._generate_with_deadline(model_request, deadline)
                budget.tokens.record(turn.usage)
                action = self._parse_model_turn(turn)
            except (ToolCallParseError, ValidationError, ValueError, json.JSONDecodeError) as error:
                if isinstance(error, ToolCallParseError):
                    raw_usage = error.detail.context.get("token_usage")
                    if isinstance(raw_usage, dict):
                        try:
                            budget.tokens.record(TokenUsage.model_validate(raw_usage))
                        except (BudgetExceededError, ValidationError):
                            pass
                failures.append(f"malformed model action: {error}")
                await journal.emit(
                    EventKind.ACTION_REJECTED,
                    actor=EventActor.MODEL,
                    payload={
                        "reason": "malformed_action",
                        "repair_attempt": attempt,
                        "compact_context_bytes": compact_context_bytes,
                    },
                    budget_before=model_budget_before,
                )
                if attempt >= self._policy.schema_repair_attempts:
                    return None
                repair_message = (
                    "The prior action was malformed. Return exactly one valid action object; do "
                    "not include prose or more than one tool call."
                )
                continue
            except BudgetExceededError as error:
                failures.append(str(error))
                raise ModelBudgetUnavailable(str(error)) from error
            except ModelBudgetUnavailable:
                raise
            except (TimeoutError, ModelBackendError, RuntimeError) as error:
                failures.append(f"model action failed: {error}")
                await journal.emit(
                    EventKind.ACTION_REJECTED,
                    actor=EventActor.MODEL,
                    payload={
                        "reason": "model_failure",
                        "compact_context_bytes": compact_context_bytes,
                    },
                    budget_before=model_budget_before,
                )
                return None
            await journal.emit(
                EventKind.MODEL_ACTION_PROPOSED,
                actor=EventActor.MODEL,
                payload={
                    "action_type": action.type,
                    "rationale": action.rationale,
                    "repair_attempt": attempt,
                    "compact_context_bytes": compact_context_bytes,
                },
                budget_before=model_budget_before,
            )
            return action
        return None

    async def _count_input_with_deadline(self, request: ModelRequest, deadline: Deadline) -> int:
        assert self._backend is not None
        if deadline.search_expired:
            raise TimeoutError("token counting cannot start after the search deadline")
        task = asyncio.create_task(self._backend.count_input_tokens(request))
        try:
            return await asyncio.wait_for(
                task,
                timeout=max(0.001, deadline.search_remaining_ms / 1_000),
            )
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise TimeoutError("model token counting exceeded the search deadline") from None

    async def _generate_with_deadline(self, request: ModelRequest, deadline: Deadline) -> ModelTurn:
        assert self._backend is not None
        if deadline.search_expired:
            raise TimeoutError("model action cannot start after the search deadline")
        task = asyncio.create_task(self._backend.generate(request))
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=max(0.001, deadline.search_remaining_ms / 1_000)
            )
        except TimeoutError:
            await self._backend.abort(request.request_id)
            try:
                await asyncio.wait_for(
                    task,
                    timeout=max(
                        0.001,
                        min(self._policy.cancellation_grace_ms, deadline.remaining_ms) / 1_000,
                    ),
                )
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise TimeoutError("model action exceeded its hard deadline") from None

    @staticmethod
    def _parse_model_turn(turn: ModelTurn) -> ControllerAction:
        calls = turn.message.tool_calls
        if calls:
            if len(calls) != 1:
                raise ValueError("a model turn must contain exactly one action")
            call = calls[0]
            return CallToolAction(
                type="call_tool",
                tool=call.name,
                arguments=call.arguments,
                rationale="native model tool selection",
            )
        payload = json.loads(turn.message.content)
        return _ACTION_ADAPTER.validate_python(payload)

    def _compact_state(
        self,
        *,
        request: RunRequest,
        problem: Any,
        incumbent: IncumbentRecord,
        deadline: Deadline,
        budget: BudgetAccount,
        recent: list[str],
    ) -> CompactModelState:
        specs = tuple(self._tools.get(name).spec for name in request.allowed_tools)
        return CompactModelState(
            run_id=request.run_id,
            problem_type=problem.type_id.value,
            problem_hash=problem.problem_hash,
            incumbent_plan_artifact_id=incumbent.plan_artifact_id,
            incumbent_comparator_key=incumbent.comparator_key,
            incumbent_metrics=incumbent.scorecard.overall_metrics,
            recent_actions=tuple(recent[-self._policy.recent_action_limit :]),
            available_tools=tuple(
                CompactTool(
                    name=spec.name,
                    description=spec.description,
                    p95_ms=spec.runtime.p95_ms,
                    streams_candidates=spec.streams_candidates,
                    resumable=spec.resumable,
                )
                for spec in specs
            ),
            remaining_total_model_tokens=budget.tokens.remaining_total,
            remaining_generated_tokens=budget.tokens.remaining_generated,
            remaining_tool_calls=budget.tools.remaining,
            remaining_wall_ms=deadline.search_remaining_ms,
        )

    async def _execute_tool_action(
        self,
        *,
        request: RunRequest,
        problem: Any,
        problem_ref: ArtifactRef,
        plugin: ProblemPlugin,
        incumbents: IncumbentStore,
        action: CallToolAction,
        fingerprint: str,
        deadline: Deadline,
        budget: BudgetAccount,
        journal: EventJournal,
        action_ledger: ActionLedger,
        run_cancellation: CancellationToken,
        failures: list[str],
        controller_owned: bool = False,
    ) -> tuple[bool, str | None, bool]:
        before = budget.snapshot()
        if not controller_owned and action.tool not in request.allowed_tools:
            await journal.emit(
                EventKind.ACTION_REJECTED,
                payload={"reason": "tool_not_exposed", "tool": action.tool},
            )
            return False, None, False
        try:
            tool = self._tools.get(action.tool)
            arguments = self._controller_arguments(
                action.tool,
                action.arguments,
                request=request,
                incumbent=incumbents.current,
            )
            validate_arguments(tool.spec, arguments)
            missing_providers = tool.spec.required_providers - self._providers.keys()
            missing_resources = tool.spec.required_resources - self._resources.keys()
            if (
                missing_providers
                or missing_resources
                or tool.spec.privacy not in self._allowed_privacy
            ):
                raise ValueError("tool prerequisites or privacy permission are unavailable")
        except (ToolRegistryError, ValueError) as error:
            failures.append(str(error))
            await journal.emit(
                EventKind.ACTION_REJECTED,
                payload={"reason": "invalid_tool_action", "tool": action.tool},
            )
            return False, None, False
        estimate_ms = (
            tool.spec.runtime.time_to_first_candidate_ms
            if tool.spec.streams_candidates
            and tool.spec.runtime.time_to_first_candidate_ms is not None
            else tool.spec.runtime.p95_ms
        )
        if not deadline.admits(estimate_ms, self._policy.validation_reserve_ms):
            await journal.emit(
                EventKind.ACTION_REJECTED,
                payload={"reason": "insufficient_wall_time", "tool": action.tool},
            )
            return False, None, False
        if not budget.tools.admit():
            await journal.emit(
                EventKind.ACTION_REJECTED,
                payload={"reason": "tool_call_budget_exhausted", "tool": action.tool},
            )
            return False, None, False
        subdeadline = deadline.action_subdeadline(tool.spec.runtime.p95_ms)
        record = action_ledger.admit(
            fingerprint=fingerprint,
            admitted_at_ms=deadline.elapsed_ms,
            tool_name=action.tool,
            subdeadline_monotonic=subdeadline,
        )
        await journal.emit(
            EventKind.ACTION_ADMITTED,
            action=record,
            payload={"tool": action.tool, "subdeadline_monotonic": subdeadline},
            budget_before=before,
        )
        action_ledger.mark(record, ActionStatus.RUNNING)
        await journal.emit(
            EventKind.TOOL_STARTED,
            actor=EventActor.TOOL,
            action=record,
            payload={
                "tool": action.tool,
                "estimated_p50_ms": tool.spec.runtime.p50_ms,
                "estimated_p95_ms": tool.spec.runtime.p95_ms,
                "estimated_time_to_first_candidate_ms": (
                    tool.spec.runtime.time_to_first_candidate_ms
                ),
            },
        )
        action_cancellation = CancellationToken()
        relay = asyncio.create_task(self._relay_cancellation(run_cancellation, action_cancellation))
        context = self._tool_context(request, subdeadline, action_cancellation)
        improved = False
        bound_id: str | None = None
        optimal = False
        result: ToolResult | None = None
        try:
            if isinstance(tool, StreamingTool):
                async for event in stream_tool(tool, arguments, context):
                    if not action_ledger.accepts(record.action_id, record.generation):
                        continue
                    if event.kind is ToolEventKind.PROGRESS:
                        await journal.emit(
                            EventKind.TOOL_PROGRESS,
                            actor=EventActor.TOOL,
                            action=record,
                            payload={"message": event.message, "progress": event.progress},
                        )
                    elif event.kind is ToolEventKind.CANDIDATE:
                        assert event.candidate is not None
                        improved = (
                            await self._evaluate_candidate(
                                request=request,
                                problem=problem,
                                problem_ref=problem_ref,
                                plugin=plugin,
                                incumbents=incumbents,
                                candidate=event.candidate,
                                deadline=deadline,
                                journal=journal,
                                action=record,
                            )
                            or improved
                        )
                    elif event.kind is ToolEventKind.BOUND:
                        assert event.bound is not None
                        valid, is_optimal = await self._verify_bound(
                            event.bound,
                            problem=problem,
                            incumbents=incumbents,
                            journal=journal,
                            action=record,
                            failures=failures,
                        )
                        if valid:
                            bound_id = event.bound.id
                            optimal = optimal or is_optimal
                    else:
                        result = event.result
            else:
                result = await invoke_tool(tool, arguments, context)
            if result is not None and result.candidate is not None and not improved:
                improved = await self._evaluate_candidate(
                    request=request,
                    problem=problem,
                    problem_ref=problem_ref,
                    plugin=plugin,
                    incumbents=incumbents,
                    candidate=result.candidate,
                    deadline=deadline,
                    journal=journal,
                    action=record,
                )
            if result is not None and result.bound is not None and result.bound.id != bound_id:
                valid, is_optimal = await self._verify_bound(
                    result.bound,
                    problem=problem,
                    incumbents=incumbents,
                    journal=journal,
                    action=record,
                    failures=failures,
                )
                if valid:
                    bound_id = result.bound.id
                    optimal = optimal or is_optimal
            if result is None:
                raise RuntimeError("tool execution ended without a terminal result")
            if result.status is ToolResultStatus.EXPIRED:
                status = ActionStatus.CANCELLED
                kind = EventKind.TOOL_CANCELLED
            elif result.status is ToolResultStatus.FAILED:
                status = ActionStatus.FAILED
                kind = EventKind.TOOL_FAILED
                if result.error is not None:
                    failures.append(result.error.message)
            else:
                status = ActionStatus.COMPLETED
                kind = EventKind.TOOL_COMPLETED
            action_ledger.mark(record, status)
            await journal.emit(
                kind,
                actor=EventActor.TOOL,
                action=record,
                artifact_ids=tuple(reference.id for reference in result.artifacts),
                payload={"tool": action.tool, "status": result.status.value},
            )
        except Exception as error:
            action_cancellation.cancel("tool failed")
            action_ledger.mark(record, ActionStatus.FAILED)
            failures.append(f"tool {action.tool} crashed: {error}")
            await journal.emit(
                EventKind.TOOL_FAILED,
                actor=EventActor.TOOL,
                action=record,
                payload={"tool": action.tool, "reason": "exception"},
            )
        finally:
            relay.cancel()
            await asyncio.gather(relay, return_exceptions=True)
        return improved, bound_id, optimal

    def _controller_arguments(
        self,
        tool_name: str,
        proposed: Mapping[str, Any],
        *,
        request: RunRequest,
        incumbent: IncumbentRecord | None,
    ) -> dict[str, Any]:
        arguments = dict(proposed)
        if tool_name == "improve":
            if incumbent is None:
                raise ValueError("improvement requires a committed incumbent")
            arguments.pop("resume_token", None)
            arguments["problem_artifact_id"] = request.problem_artifact_id
            arguments["starting_plan_artifact_id"] = incumbent.plan_artifact_id
            maximum = int(arguments.get("max_candidates", self._policy.max_candidates_per_action))
            arguments["max_candidates"] = min(maximum, self._policy.max_candidates_per_action)
        return arguments

    async def _evaluate_candidate(
        self,
        *,
        request: RunRequest,
        problem: Any,
        problem_ref: ArtifactRef,
        plugin: ProblemPlugin,
        incumbents: IncumbentStore,
        candidate: Plan,
        deadline: Deadline,
        journal: EventJournal,
        action: ActionRecord,
    ) -> bool:
        report = plugin.validate_plan(problem, candidate, self._artifacts)
        if not report.valid:
            await journal.emit(
                EventKind.CANDIDATE_REJECTED,
                actor=EventActor.EVALUATOR,
                action=action,
                payload={
                    "reason": "validation_failed",
                    "issue_codes": [issue.code for issue in report.issues],
                },
            )
            return False
        score = plugin.measure(problem, candidate, self._artifacts)
        if not score.feasible:
            await journal.emit(
                EventKind.CANDIDATE_REJECTED,
                actor=EventActor.EVALUATOR,
                action=action,
                payload={"reason": "evaluator_rejected"},
            )
            return False
        context = self._tool_context(
            request,
            deadline.at_monotonic,
            CancellationToken(),
        )
        plan_ref, score_ref = put_plan_and_scorecard(
            context,
            candidate,
            score,
            name="anytime_controller",
            version=CONTROLLER_VERSION,
            parents=(problem_ref,),
            parameters={"role": "candidate", "source_action": action.action_id},
        )
        committed = await incumbents.try_commit(
            plan=candidate,
            scorecard=score,
            plan_artifact_id=plan_ref.id,
            scorecard_artifact_id=score_ref.id,
            source_action_id=action.action_id,
            committed_at_ms=deadline.elapsed_ms,
            seed=request.seed,
            committed_at=self._utcnow(),
        )
        if committed is None:
            await journal.emit(
                EventKind.CANDIDATE_REJECTED,
                actor=EventActor.EVALUATOR,
                action=action,
                artifact_ids=(plan_ref.id, score_ref.id),
                payload={"reason": "not_better_than_incumbent"},
            )
            return False
        await journal.emit(
            EventKind.INCUMBENT_COMMITTED,
            actor=EventActor.EVALUATOR,
            action=action,
            artifact_ids=(plan_ref.id, score_ref.id),
            payload={"comparator_key": list(score.comparator_key)},
        )
        return True

    async def _verify_bound(
        self,
        reference: ArtifactRef,
        *,
        problem: Any,
        incumbents: IncumbentStore,
        journal: EventJournal,
        action: ActionRecord,
        failures: list[str],
    ) -> tuple[bool, bool]:
        try:
            bound = VerifiedBound.model_validate(read_json(self._artifacts, reference))
            if bound.problem_hash != problem.problem_hash:
                raise ValueError("bound belongs to a different immutable problem")
        except (ValueError, OSError) as error:
            failures.append(f"invalid bound: {error}")
            return False, False
        await journal.emit(
            EventKind.BOUND_VERIFIED,
            actor=EventActor.EVALUATOR,
            action=action,
            artifact_ids=(reference.id,),
            payload={
                "complete": bound.complete,
                "best_comparator_key": list(bound.best_comparator_key),
            },
        )
        current = incumbents.current
        optimal = (
            bound.complete
            and current is not None
            and bound.best_comparator_key == current.comparator_key
        )
        return True, optimal

    async def _fallback(
        self,
        *,
        request: RunRequest,
        problem: Any,
        problem_ref: ArtifactRef,
        plugin: ProblemPlugin,
        incumbents: IncumbentStore,
        deadline: Deadline,
        budget: BudgetAccount,
        journal: EventJournal,
        action_ledger: ActionLedger,
        run_cancellation: CancellationToken,
        failures: list[str],
    ) -> tuple[TerminalReason, str | None]:
        verified_bound_id: str | None = None
        attempted = False
        improved_any = False
        for strategy in plugin.fallback_actions():
            terminal = self._budget_terminal(
                deadline, budget, run_cancellation, include_tokens=False
            )
            if terminal is not None:
                return terminal, verified_bound_id
            await journal.emit(
                EventKind.FALLBACK_INVOKED,
                payload={"strategy": strategy.value},
            )
            action = CallToolAction(
                type="call_tool",
                tool="improve",
                arguments={
                    "strategy": strategy.value,
                    "max_candidates": self._policy.max_candidates_per_action,
                },
                rationale="deterministic problem-plugin fallback",
            )
            controlled = self._controller_arguments(
                action.tool,
                action.arguments,
                request=request,
                incumbent=incumbents.current,
            )
            fingerprint = action_fingerprint(
                {"type": action.type, "tool": action.tool, "arguments": controlled}
            )
            if action_ledger.is_duplicate(fingerprint):
                continue
            attempted = True
            improved, bound_id, optimal = await self._execute_tool_action(
                request=request,
                problem=problem,
                problem_ref=problem_ref,
                plugin=plugin,
                incumbents=incumbents,
                action=action,
                fingerprint=fingerprint,
                deadline=deadline,
                budget=budget,
                journal=journal,
                action_ledger=action_ledger,
                run_cancellation=run_cancellation,
                failures=failures,
                controller_owned=True,
            )
            improved_any = improved_any or improved
            verified_bound_id = bound_id or verified_bound_id
            if optimal:
                return TerminalReason.PROVEN_OPTIMAL, verified_bound_id
            if budget.tools.remaining == 0:
                break
        if run_cancellation.cancelled:
            return TerminalReason.USER_CANCELLED, verified_bound_id
        if deadline.search_expired:
            return TerminalReason.TIME_EXHAUSTED, verified_bound_id
        if not attempted and budget.tools.remaining == 0:
            return TerminalReason.BUDGET_TIER_COMPLETE, verified_bound_id
        return (
            TerminalReason.BUDGET_TIER_COMPLETE if improved_any else TerminalReason.PLATEAU,
            verified_bound_id,
        )

    @staticmethod
    async def _relay_cancellation(
        source: CancellationToken, destination: CancellationToken
    ) -> None:
        reason = await source.wait()
        destination.cancel(reason)

    @staticmethod
    def _cancelled(token: CancellationToken) -> bool:
        """Read cancellation without encouraging stale type narrowing across awaits."""

        return token.cancelled

    @staticmethod
    def _budget_terminal(
        deadline: Deadline,
        budget: BudgetAccount,
        cancellation: CancellationToken,
        *,
        include_tokens: bool = True,
    ) -> TerminalReason | None:
        if cancellation.cancelled:
            return TerminalReason.USER_CANCELLED
        if deadline.search_expired:
            return TerminalReason.TIME_EXHAUSTED
        if budget.tools.remaining == 0:
            return TerminalReason.BUDGET_TIER_COMPLETE
        if include_tokens and (
            budget.tokens.remaining_total == 0 or budget.tokens.remaining_generated == 0
        ):
            return TerminalReason.TOKEN_EXHAUSTED_NO_DETERMINISTIC_WORK
        return None

    @staticmethod
    def _result_status(reason: TerminalReason, has_incumbent: bool) -> RunStatus:
        if reason is TerminalReason.USER_CANCELLED:
            return RunStatus.CANCELLED
        if not has_incumbent:
            if reason is TerminalReason.INTERNAL_FAILURE:
                return RunStatus.FAILED
            return RunStatus.REJECTED
        if reason is TerminalReason.INTERNAL_FAILURE:
            return RunStatus.PARTIAL
        if reason in {TerminalReason.TIME_EXHAUSTED, TerminalReason.PLATEAU}:
            return RunStatus.PARTIAL
        if reason in {
            TerminalReason.BASELINE_ONLY,
            TerminalReason.BUDGET_TIER_COMPLETE,
            TerminalReason.TARGET_REACHED,
            TerminalReason.PROVEN_OPTIMAL,
            TerminalReason.MODEL_STOPPED,
        }:
            return RunStatus.COMPLETE
        return RunStatus.PARTIAL
