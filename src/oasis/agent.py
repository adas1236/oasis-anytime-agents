"""Message-to-answer tool orchestration, with no prepared problem or benchmark fallback.

The model owns interpretation, evidence gathering, compilation and strategy selection.
The host enforces budgets and independently validates any plans produced along the way.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar

from oasis.artifacts import ArtifactProvenance, ArtifactStore, put_json
from oasis.config import OasisSettings
from oasis.controller import (
    BudgetAccount,
    BudgetSpec,
    BudgetTier,
    ControllerPolicy,
    ControllerState,
    Deadline,
    EventActor,
    EventJournal,
    EventKind,
    RunMetadata,
    RunResult,
    RunStatus,
    RunStore,
    StateMachine,
    TerminalReason,
)
from oasis.controller.budget import BudgetExceededError
from oasis.controller.incumbent import IncumbentStore
from oasis.controller.state import EventCallback
from oasis.errors import ToolCallParseError
from oasis.llm import ModelBackend
from oasis.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    ModelRequest,
    TokenUsage,
    ToolCall,
)
from oasis.problems import LocationAllocationProblem, ProblemRegistry, RouteServiceProblem
from oasis.problems.protocols import ProblemPlugin
from oasis.prompts import AGENT_SYSTEM_PROMPT
from oasis.schemas import ArtifactKind, ArtifactRef, Plan, ToolError, ToolErrorCode, ToolEventKind
from oasis.schemas.tools import ToolResult, ToolResultStatus, ToolSpec
from oasis.tools import CancellationToken, ToolContext, ToolRegistry, invoke_tool, stream_tool
from oasis.tools.decision.common import put_plan_and_scorecard, read_plan, read_problem
from oasis.tools.evidence.common import read_frame
from oasis.tools.protocols import ToolCancelledError

T = TypeVar("T")


@dataclass
class _ProblemState:
    reference: ArtifactRef
    problem: LocationAllocationProblem | RouteServiceProblem
    plugin: ProblemPlugin
    incumbents: IncumbentStore


class MessageAgent:
    """One run's model/tool loop; shared stores and provider handles are injected."""

    def __init__(
        self,
        *,
        backend: ModelBackend,
        tools: ToolRegistry,
        problems: ProblemRegistry,
        artifacts: ArtifactStore,
        runs: RunStore,
        settings: OasisSettings,
        providers: Mapping[str, object],
        resources: Mapping[str, object],
        callback: EventCallback | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.problems = problems
        self.artifacts = artifacts
        self.runs = runs
        self.settings = settings
        self.providers = providers
        self.resources = resources
        self.callback = callback
        self.monotonic = monotonic
        self.formulations: dict[str, _ProblemState] = {}
        self.active_problem_id: str | None = None
        self.failures: list[str] = []
        self.observed_usage = TokenUsage()
        self.usage_complete = True
        self.labels: dict[str, dict[str, str]] = {}

    async def run(
        self,
        *,
        run_id: str,
        message: str,
        budget: BudgetSpec,
        cancellation: CancellationToken,
        seed: int = 0,
        thinking: bool = False,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> RunResult:
        self.deadline = Deadline(budget, ControllerPolicy(), monotonic=self.monotonic)
        self.account = BudgetAccount(budget, self.deadline)
        self.context = ToolContext(
            run_id=run_id,
            artifact_store=self.artifacts,
            deadline_monotonic=self.deadline.search_deadline_monotonic,
            cancellation=cancellation,
            seed=seed,
            providers=self.providers,
            resources=self.resources,
            monotonic=self.monotonic,
        )
        specs = tuple(
            spec
            for spec in self.tools.list()
            if (allowed_tools is None or spec.name in allowed_tools)
            and spec.required_providers <= self.providers.keys()
            and spec.required_resources <= self.resources.keys()
            and spec.privacy in self.context.allowed_privacy
        )
        self.exposed = frozenset(spec.name for spec in specs)
        self.history = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=(self.settings.agent_system_prompt or AGENT_SYSTEM_PROMPT),
            ),
            ChatMessage(role=ChatRole.USER, content=message),
        ]
        self.runs.create(
            RunMetadata(
                run_id=run_id,
                seed=seed,
                metadata={"mode": "message", "message": message},
            )
        )
        state = StateMachine()
        self.journal = EventJournal(
            run_id=run_id,
            deadline=self.deadline,
            budget=self.account,
            append=self.runs.append_event,
            state=state,
            callback=self.callback,
        )
        await self.journal.emit(EventKind.RUN_CREATED, payload={"mode": "message"})
        state.transition(ControllerState.GROUNDING)
        state.transition(ControllerState.REASONING)
        answer: str | None = None
        reason = TerminalReason.INTERNAL_FAILURE
        try:
            reason, answer = await self._loop(run_id, thinking, seed, specs)
        except ToolCancelledError:
            reason = TerminalReason.USER_CANCELLED
        except TimeoutError:
            reason = (
                TerminalReason.TIME_EXHAUSTED
                if self.deadline.search_expired
                else TerminalReason.MODEL_CALL_TIMEOUT
            )
        except BudgetExceededError as error:
            self.failures.append(str(error))
            self.usage_complete = False
            reason = TerminalReason.TOKEN_EXHAUSTED_NO_DETERMINISTIC_WORK
        except Exception as error:
            self.failures.append(f"Agent execution failed ({type(error).__name__}).")
            reason = TerminalReason.INTERNAL_FAILURE

        current_problem = self.formulations.get(self.active_problem_id or "")
        incumbent = current_problem.incumbents.current if current_problem else None
        view = (
            current_problem.plugin.render_result(
                current_problem.problem,
                incumbent.plan,
                incumbent.scorecard,
            )
            if current_problem and incumbent
            else None
        )
        source: Literal["model", "plan", "status"] = "model" if answer else "status"
        if not answer and current_problem and incumbent:
            answer = self._plan_answer(current_problem)
            source = "plan"
        if not answer:
            answer = {
                TerminalReason.USER_CANCELLED: "Stopped before an answer was ready.",
                TerminalReason.TIME_EXHAUSTED: "I ran out of time before I could finish an answer.",
                TerminalReason.TOKEN_EXHAUSTED_NO_DETERMINISTIC_WORK: (
                    "I reached the response budget before I could finish an answer."
                ),
                TerminalReason.TOOL_CALL_LIMIT: (
                    "I reached the tool limit before I could finish an answer."
                ),
                TerminalReason.TOOL_ROUND_LIMIT: (
                    "I reached the step limit before I could finish an answer."
                ),
                TerminalReason.MODEL_CALL_TIMEOUT: (
                    "The model timed out before an answer was ready."
                ),
                TerminalReason.CONTEXT_LIMIT: "The available context was too small to continue.",
            }.get(reason, "I couldn't complete an answer. Please try again.")
        transcript = put_json(
            self.artifacts,
            [item.model_dump(mode="json") for item in self.history],
            kind=ArtifactKind.JSON_SPECIFICATION,
            units="conversation",
            data_schema={"type": "AgentConversation", "version": "1.0.0"},
            provenance=ArtifactProvenance(
                source_uri=f"oasis://runs/{run_id}/conversation",
                license="user-provided",
            ),
        )
        state.transition(ControllerState.QUIESCING)
        state.transition(ControllerState.FINALIZED)
        await self.journal.emit(
            EventKind.RUN_FINALIZED,
            payload={
                "reason": reason.value,
                "has_incumbent": incumbent is not None,
                "answer_source": source,
            },
        )
        status = RunStatus.PARTIAL
        if reason is TerminalReason.MODEL_STOPPED and source == "model":
            status = RunStatus.COMPLETE
        elif reason is TerminalReason.USER_CANCELLED:
            status = RunStatus.CANCELLED
        elif reason is TerminalReason.INTERNAL_FAILURE and incumbent is None:
            status = RunStatus.FAILED
        snapshot = self.account.snapshot().model_copy(
            update={
                "model_usage": self.observed_usage,
                "remaining_total_model_tokens": max(
                    0, budget.max_total_model_tokens - self.observed_usage.total_tokens
                ),
                "remaining_generated_tokens": max(
                    0, budget.max_generated_tokens - self.observed_usage.generated_tokens
                ),
            }
        )
        result = RunResult(
            run_id=run_id,
            status=status,
            terminal_reason=reason,
            budget_tier=BudgetTier.ITERATIVE_MODEL,
            answer=answer,
            answer_source=source,
            conversation_artifact_id=transcript.id,
            usage_complete=self.usage_complete,
            problem_artifact_id=self.active_problem_id,
            problem_hash=current_problem.problem.problem_hash if current_problem else None,
            evidence_hash=current_problem.problem.evidence_hash if current_problem else None,
            policy_hash=current_problem.problem.policy_hash if current_problem else None,
            best_plan=incumbent.plan if incumbent else None,
            best_scorecard=incumbent.scorecard if incumbent else None,
            best_plan_artifact_id=incumbent.plan_artifact_id if incumbent else None,
            best_scorecard_artifact_id=incumbent.scorecard_artifact_id if incumbent else None,
            result_view=view,
            requested_budget=budget,
            consumed_budget=snapshot,
            deadline_overshoot_ms=self.deadline.overshoot_ms,
            time_to_first_feasible_ms=(
                current_problem.incumbents.timeline[0].committed_at_ms
                if current_problem and incumbent
                else None
            ),
            incumbent_timeline=current_problem.incumbents.timeline if current_problem else (),
            failures=tuple(self.failures),
            model_profile=self.backend.profile.name,
            model_id=self.backend.profile.model_id,
            tool_versions={spec.name: spec.version for spec in specs},
            controller_version="message-agent-1.0.0",
            seed=seed,
            event_count=self.journal.count,
        )
        self.runs.write_result(result)
        return result

    def _stop_reason(self) -> TerminalReason | None:
        if self.deadline.search_expired:
            return TerminalReason.TIME_EXHAUSTED
        if self.context.cancellation.cancelled:
            return TerminalReason.USER_CANCELLED
        return None

    async def _wait_model(self, awaitable: Awaitable[T], request_id: str) -> T:
        task = asyncio.ensure_future(awaitable)
        cancelled = asyncio.create_task(self.context.cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {task, cancelled},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=min(
                    self.context.remaining_seconds, self.settings.agent_model_timeout_seconds
                ),
            )
            if cancelled in done or self.context.cancellation.cancelled:
                raise ToolCancelledError("run cancelled")
            if task in done:
                return task.result()
            raise TimeoutError
        except (TimeoutError, asyncio.CancelledError):
            self.usage_complete = False
            task.cancel()
            try:
                await self.backend.abort(request_id)
            except Exception:
                self.failures.append("The backend could not confirm that generation stopped.")
            raise
        finally:
            task.cancel()
            cancelled.cancel()
            await asyncio.gather(task, cancelled, return_exceptions=True)

    def _record_usage(self, usage: TokenUsage) -> None:
        self.observed_usage += usage
        self.account.tokens.record(usage)

    async def _loop(
        self,
        run_id: str,
        thinking: bool,
        seed: int,
        specs: tuple[ToolSpec, ...],
    ) -> tuple[TerminalReason, str | None]:
        repairs = 0
        for round_index in range(self.settings.agent_tool_rounds + 1):
            reason = self._stop_reason()
            if reason:
                return reason, None
            if (
                self.account.tokens.remaining_total < 1
                or self.account.tokens.remaining_generated < 1
            ):
                return TerminalReason.TOKEN_EXHAUSTED_NO_DETERMINISTIC_WORK, None
            request = ModelRequest(
                request_id=f"{run_id}-{round_index}",
                messages=tuple(self.history),
                tools=self.tools.model_definitions(specs),
                seed=seed,
                thinking_enabled=thinking,
                max_generated_tokens=self.settings.agent_generation_tokens,
            )
            estimated = await self._wait_model(
                self.backend.count_input_tokens(request),
                request.request_id,
            )
            allowance = self.account.tokens.generation_allowance(
                estimated_input_tokens=estimated,
                requested=self.settings.agent_generation_tokens,
            )
            context_limit = self.backend.capabilities.context_limit
            if context_limit is not None and estimated >= context_limit:
                return TerminalReason.CONTEXT_LIMIT, None
            if context_limit is not None:
                allowance = min(allowance, context_limit - estimated)
            if allowance < 1:
                return TerminalReason.TOKEN_EXHAUSTED_NO_DETERMINISTIC_WORK, None
            request = request.model_copy(update={"max_generated_tokens": allowance})
            reason = self._stop_reason()
            if reason:
                return reason, None
            try:
                turn = await self._wait_model(self.backend.generate(request), request.request_id)
            except ToolCallParseError as error:
                raw = error.detail.context.get("token_usage")
                usage = (
                    TokenUsage.model_validate(raw)
                    if isinstance(raw, dict)
                    else TokenUsage(
                        input_tokens=estimated,
                    )
                )
                self.usage_complete = self.usage_complete and isinstance(raw, dict)
                self._record_usage(usage)
                if repairs >= 1:
                    raise
                repairs += 1
                self.history.append(
                    ChatMessage(
                        role=ChatRole.USER,
                        content=(
                            "The previous tool call was malformed. Use a provided tool name and "
                            "its argument schema, or answer the original request in plain text."
                        ),
                    )
                )
                continue
            except (TimeoutError, asyncio.CancelledError):
                self._record_usage(TokenUsage(input_tokens=estimated))
                raise
            except Exception:
                self.usage_complete = False
                self._record_usage(TokenUsage(input_tokens=estimated))
                raise
            self._record_usage(turn.usage)
            self.history.append(turn.message)
            await self.journal.emit(
                EventKind.MODEL_ACTION_PROPOSED,
                actor=EventActor.MODEL,
                payload={"tools": [c.name for c in turn.message.tool_calls]},
            )
            reason = self._stop_reason()
            if reason:
                return reason, None
            if not turn.message.tool_calls:
                reason = {
                    FinishReason.LENGTH: TerminalReason.MODEL_OUTPUT_LIMIT,
                    FinishReason.CANCELLED: TerminalReason.USER_CANCELLED,
                    FinishReason.ERROR: TerminalReason.INTERNAL_FAILURE,
                }.get(turn.finish_reason, TerminalReason.MODEL_STOPPED)
                return reason, turn.message.content or None
            if round_index >= self.settings.agent_tool_rounds:
                return TerminalReason.TOOL_ROUND_LIMIT, None
            for call in turn.message.tool_calls:
                reason = self._stop_reason()
                if reason:
                    return reason, None
                if not self.account.tools.admit():
                    return TerminalReason.TOOL_CALL_LIMIT, None
                await self.journal.emit(
                    EventKind.TOOL_STARTED,
                    actor=EventActor.TOOL,
                    payload={"tool": call.name, "call_id": call.id},
                )
                result = await self._call(call)
                self.history.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        name=call.name,
                        tool_call_id=call.id,
                        content=result.model_summary(),
                    )
                )
                await self.journal.emit(
                    EventKind.TOOL_FAILED if result.error else EventKind.TOOL_COMPLETED,
                    actor=EventActor.TOOL,
                    artifact_ids=tuple(a.id for a in result.artifacts),
                    payload={
                        "tool": call.name,
                        "call_id": call.id,
                        "status": result.status.value,
                        "result": result.model_summary(),
                    },
                )
        return TerminalReason.TOOL_ROUND_LIMIT, None

    async def _call(self, call: ToolCall) -> ToolResult:
        if call.name not in self.exposed:
            return ToolResult(
                status=ToolResultStatus.FAILED,
                summary="Tool is unavailable.",
                error=ToolError(
                    code=ToolErrorCode.CAPABILITY_DENIED,
                    message=f"Tool {call.name!r} is unavailable.",
                ),
            )
        try:
            tool = self.tools.get(call.name)
            result = None
            if (
                tool.spec.streams_candidates
                or tool.spec.streams_progress
                or tool.spec.streams_bounds
            ):
                async for event in stream_tool(tool, call.arguments, self.context):
                    if event.candidate is not None:
                        await self._observe(
                            call.arguments.get("problem_artifact_id"), event.candidate, call.id
                        )
                    if event.kind is ToolEventKind.PROGRESS:
                        await self.journal.emit(
                            EventKind.TOOL_PROGRESS,
                            actor=EventActor.TOOL,
                            payload={"tool": call.name, "progress": event.progress},
                        )
                    if event.result is not None:
                        result = event.result
            else:
                result = await invoke_tool(tool, call.arguments, self.context)
            if result is None:
                raise RuntimeError("tool produced no result")
            problem_id = call.arguments.get("problem_artifact_id")
            if call.name == "compile_problem" and result.metrics.get("problem_artifact_id"):
                problem_id = result.metrics["problem_artifact_id"]
                if isinstance(problem_id, str):
                    self._problem(problem_id)
                    self.active_problem_id = problem_id
                    await self.journal.emit(EventKind.PROBLEM_COMPILED, artifact_ids=(problem_id,))
            if result.candidate is not None:
                await self._observe(problem_id, result.candidate, call.id)
            plan_id = result.metrics.get("baseline_plan_artifact_id")
            if call.name == "summarize_plan" and result.metrics.get("feasible"):
                plan_id = call.arguments.get("plan_artifact_id")
            if isinstance(plan_id, str):
                _, plan = read_plan(self.context, plan_id)
                await self._observe(problem_id, plan, call.id)
            return result
        except ToolCancelledError:
            raise
        except Exception as error:
            return ToolResult(
                status=ToolResultStatus.FAILED,
                summary="Tool execution failed.",
                error=ToolError(
                    code=ToolErrorCode.INTERNAL_ERROR,
                    message=f"Tool execution failed ({type(error).__name__}).",
                ),
            )

    def _problem(self, problem_id: str) -> _ProblemState:
        if problem_id not in self.formulations:
            reference, problem = read_problem(self.context, problem_id)
            plugin = self.problems.get(problem.type_id.value)
            if not plugin.validate_spec(problem, self.artifacts).valid:
                raise ValueError("invalid compiled problem")
            self.formulations[problem_id] = _ProblemState(
                reference,
                problem,
                plugin,
                IncumbentStore(plugin.compare),
            )
        return self.formulations[problem_id]

    async def _observe(self, problem_id: object, plan: Plan, call_id: str) -> None:
        if not isinstance(problem_id, str) or self._stop_reason():
            return
        problem = self._problem(problem_id)
        report = problem.plugin.validate_plan(problem.problem, plan, self.artifacts)
        if not report.valid:
            await self.journal.emit(
                EventKind.CANDIDATE_REJECTED,
                actor=EventActor.EVALUATOR,
                payload={"reason": "invalid_plan"},
            )
            return
        score = problem.plugin.measure(problem.problem, plan, self.artifacts)
        if not score.feasible or self._stop_reason():
            return
        plan_ref, score_ref = put_plan_and_scorecard(
            self.context,
            plan,
            score,
            name="message_agent",
            version="1.0.0",
            parents=(problem.reference,),
            parameters={"call_id": call_id},
        )
        committed = await problem.incumbents.try_commit(
            plan=plan,
            scorecard=score,
            plan_artifact_id=plan_ref.id,
            scorecard_artifact_id=score_ref.id,
            source_action_id=call_id,
            committed_at_ms=self.deadline.elapsed_ms,
            seed=self.context.seed,
        )
        self.active_problem_id = problem_id
        if committed:
            await self.journal.emit(
                EventKind.INCUMBENT_COMMITTED,
                actor=EventActor.EVALUATOR,
                artifact_ids=(plan_ref.id, score_ref.id),
                payload={
                    "problem_artifact_id": problem_id,
                    "problem_hash": problem.problem.problem_hash,
                    "answer": self._plan_answer(problem, stopped=False),
                    "comparator_key": list(score.comparator_key),
                },
            )

    def _plan_answer(self, state: _ProblemState, *, stopped: bool = True) -> str:
        incumbent = state.incumbents.current
        assert incumbent is not None
        problem = state.problem
        reference = (
            problem.candidates.artifact
            if isinstance(problem, LocationAllocationProblem)
            else problem.nodes
        )
        id_field = (
            problem.candidates.candidate_id_field
            if isinstance(problem, LocationAllocationProblem)
            else problem.node_id_field
        )
        names = self._location_names(reference, id_field)
        if incumbent.plan.routes:
            choices = []
            for route in incumbent.plan.routes:
                nodes = route.get("node_ids", [])
                if isinstance(nodes, list):
                    choices.append(" → ".join(names.get(str(i), str(i)) for i in nodes))
            description = "Routes: " + "; ".join(choices)
        else:
            description = "Selected sites: " + ", ".join(
                names.get(i, i) for i in incumbent.plan.selected_site_ids
            )
        metrics = "; ".join(
            f"{k.replace('_', ' ')}: {v:g}" for k, v in incumbent.scorecard.raw_objective.items()
        )
        answer = f"Best plan found so far. {description}. {metrics}."
        if stopped:
            answer += " Search stopped before a final explanation was ready."
        return answer

    def _location_names(self, reference: ArtifactRef, id_field: str) -> dict[str, str]:
        """Recover display labels even when canonical solver evidence omits name columns."""
        if reference.id in self.labels:
            return self.labels[reference.id]
        names: dict[str, str] = {}
        pending = [reference.id]
        visited: set[str] = set()
        while pending and len(visited) < 32:
            identifier = pending.pop(0)
            if identifier in visited:
                continue
            visited.add(identifier)
            item = self.artifacts.get_metadata(identifier)
            if item.kind in {ArtifactKind.TABLE, ArtifactKind.VECTOR}:
                frame = read_frame(self.context, item)
                if id_field in frame.columns and "name" in frame.columns:
                    for key, name in zip(frame[id_field], frame["name"], strict=True):
                        names.setdefault(str(key), str(name))
            pending.extend(item.lineage.parent_ids)
        self.labels[reference.id] = names
        return names
