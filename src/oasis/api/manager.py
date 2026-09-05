"""Asynchronous run lifecycle, capacity, cancellation, and event subscriptions."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from oasis.agent import MessageAgent
from oasis.api.examples import prepare_example
from oasis.api.lifecycle import ModelService
from oasis.api.schemas import (
    CancelRunResponse,
    CompileProblemSource,
    ExampleProblemSource,
    InlineProblemSource,
    ManagedRunPhase,
    RunCreatedResponse,
    RunCreateRequest,
    RunInspectionResponse,
    RunLinks,
    StoredProblemSource,
)
from oasis.artifacts import ArtifactNotFoundError, ArtifactProvenance, ArtifactStore, put_json
from oasis.controller import (
    AnytimeController,
    BudgetAccount,
    BudgetSpec,
    BudgetTier,
    ControllerEvent,
    ControllerPolicy,
    ControllerState,
    Deadline,
    EventActor,
    EventJournal,
    EventKind,
    RunMetadata,
    RunRequest,
    RunResult,
    RunStatus,
    RunStore,
    StateMachine,
    TerminalReason,
)
from oasis.errors import ModelBackendError
from oasis.llm import ModelBackend
from oasis.problems import (
    LocationAllocationProblem,
    ProblemRegistry,
    RouteServiceProblem,
    create_builtin_problem_registry,
)
from oasis.schemas import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactRef,
    ArtifactTransformation,
    PrivacyClassification,
    ToolResultStatus,
)
from oasis.tools import (
    CancellationToken,
    ToolContext,
    ToolRegistry,
    create_tool_registry,
    invoke_tool,
)
from oasis.tools.registry import ToolRegistryError

_PENDING_ARTIFACT_ID = "sha256-" + "0" * 64
_PRIVACY_ORDER = {
    PrivacyClassification.PUBLIC: 0,
    PrivacyClassification.INTERNAL: 1,
    PrivacyClassification.SENSITIVE: 2,
    PrivacyClassification.RESTRICTED: 3,
}


class RunManagerError(RuntimeError):
    """A structured, expected service-layer run error."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(slots=True)
class _ActiveRun:
    request: RunCreateRequest
    cancellation: CancellationToken
    started_at_monotonic: float
    task: asyncio.Task[RunResult] | None = None
    phase: ManagedRunPhase = ManagedRunPhase.PREPARING
    backend: ModelBackend | None = None


def run_links(run_id: str) -> RunLinks:
    base = f"/api/v1/runs/{run_id}"
    return RunLinks(self=base, events=f"{base}/events", cancel=f"{base}/cancel", map=f"{base}/map")


def _problem_references(
    problem: LocationAllocationProblem | RouteServiceProblem,
) -> tuple[ArtifactRef, ...]:
    if isinstance(problem, LocationAllocationProblem):
        references = [problem.demand.artifact, problem.candidates.artifact, problem.access_matrix]
        for service_scenario in problem.service_scenarios:
            references.append(service_scenario.service_matrix)
            if service_scenario.access_matrix is not None:
                references.append(service_scenario.access_matrix)
            if service_scenario.demand_multiplier is not None:
                references.append(service_scenario.demand_multiplier)
    else:
        references = [problem.nodes]
        for route_scenario in problem.travel_scenarios:
            references.append(route_scenario.travel_matrix)
            if route_scenario.demand_multiplier is not None:
                references.append(route_scenario.demand_multiplier)
    unique = {reference.id: reference for reference in references}
    return tuple(unique[key] for key in sorted(unique))


class RunManager:
    """Own controller tasks while persisted stores remain authoritative for replay."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        run_store: RunStore,
        model_service: ModelService,
        max_concurrent_runs: int,
        cancel_wait_seconds: float = 1.0,
        tool_registry: ToolRegistry | None = None,
        problem_registry: ProblemRegistry | None = None,
        controller_policy: ControllerPolicy | None = None,
        providers: Mapping[str, object] | None = None,
        resources: Mapping[str, object] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.artifact_store = artifact_store
        self.run_store = run_store
        self.model_service = model_service
        self.tool_registry = tool_registry or create_tool_registry(discover_entry_points=False)
        self.problem_registry = problem_registry or create_builtin_problem_registry()
        self.controller_policy = controller_policy or ControllerPolicy()
        self.providers = dict(providers or {})
        self.resources = dict(resources or {})
        self.max_concurrent_runs = max_concurrent_runs
        self.cancel_wait_seconds = cancel_wait_seconds
        self._monotonic = monotonic
        self._active: dict[str, _ActiveRun] = {}
        self._subscribers: dict[str, set[asyncio.Queue[None]]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _new_run_id() -> str:
        return f"run-{uuid.uuid4().hex}"

    async def start(self, request: RunCreateRequest) -> RunCreatedResponse:
        if "budget" not in request.model_fields_set:
            settings = self.model_service.settings
            request = request.model_copy(
                update={
                    "budget": BudgetSpec(
                        wall_time_ms=settings.agent_wall_time_ms,
                        max_total_model_tokens=settings.agent_total_tokens,
                        max_generated_tokens=min(
                            settings.agent_generated_tokens, settings.agent_total_tokens
                        ),
                        max_tool_calls=settings.agent_tool_calls,
                    )
                }
            )
        if "thinking_enabled" not in request.model_fields_set:
            request = request.model_copy(
                update={"thinking_enabled": self.model_service.settings.thinking}
            )
        for name in request.allowed_tools or ():
            try:
                self.tool_registry.get(name)
            except ToolRegistryError as error:
                raise RunManagerError(422, "unknown_tool", str(error)) from error
        run_id = request.run_id or self._new_run_id()
        async with self._lock:
            if run_id in self._active or self.run_store.read_metadata(run_id) is not None:
                raise RunManagerError(409, "run_exists", "The requested run ID already exists.")
            if len(self._active) >= self.max_concurrent_runs:
                raise RunManagerError(
                    503,
                    "run_capacity_unavailable",
                    "The service has no run capacity available; retry later.",
                )
            active = _ActiveRun(
                request=request,
                cancellation=CancellationToken(),
                started_at_monotonic=self._monotonic(),
            )
            self._active[run_id] = active
            active.task = asyncio.create_task(
                self._execute(run_id, active), name=f"oasis-run-{run_id}"
            )
        return RunCreatedResponse(run_id=run_id, links=run_links(run_id))

    async def _execute(self, run_id: str, active: _ActiveRun) -> RunResult:
        try:
            backend = self.model_service.backend_for(
                model_profile=active.request.model_profile,
                model_id=active.request.model_id,
                runtime_policy=active.request.runtime_policy,
            )
            active.backend = backend
            if self._uses_model(active.request):
                try:
                    await self.model_service.ensure_ready(backend)
                finally:
                    active.started_at_monotonic = self._monotonic()
            if active.request.message is not None:
                active.phase = ManagedRunPhase.RUNNING
                agent = MessageAgent(
                    backend=backend,
                    tools=self.tool_registry,
                    problems=self.problem_registry,
                    artifacts=self.artifact_store,
                    runs=self.run_store,
                    settings=self.model_service.settings,
                    providers=self.providers,
                    resources=self.resources,
                    callback=self.publish,
                    monotonic=self._monotonic,
                )
                result = await agent.run(
                    run_id=run_id,
                    message=active.request.message,
                    budget=active.request.budget,
                    cancellation=active.cancellation,
                    seed=active.request.seed,
                    thinking=active.request.thinking_enabled,
                    allowed_tools=active.request.allowed_tools,
                )
                plan = self.model_service.runtime_plan_model(backend)
                result = result.model_copy(
                    update={
                        "runtime_plan": plan,
                        "compute_inventory": self.model_service.inventory_model(
                            backend
                        ).sanitized(),
                        "hardware_validation": plan.hardware_validation.value,
                    }
                )
                self.run_store.write_result(result)
                return result
            problem_id, baseline_id, preparation_tool_calls = await self._resolve_source(
                run_id, active
            )
            active.phase = ManagedRunPhase.RUNNING
            controller_request = RunRequest(
                run_id=run_id,
                problem_artifact_id=problem_id,
                baseline_plan_artifact_id=baseline_id,
                budget=active.request.budget,
                seed=active.request.seed,
                enable_model=active.request.enable_model,
                enable_deterministic_fallback=active.request.enable_deterministic_fallback,
                allowed_tools=(
                    active.request.allowed_tools
                    if active.request.allowed_tools is not None
                    else ("improve",)
                ),
                thinking_enabled=active.request.thinking_enabled,
                requested_tier=active.request.requested_tier,
                runtime_plan=self.model_service.runtime_plan_model(backend),
                compute_inventory=self.model_service.inventory_model(backend),
            )
            controller = AnytimeController(
                artifact_store=self.artifact_store,
                run_store=self.run_store,
                backend=backend,
                tool_registry=self.tool_registry,
                problem_registry=self.problem_registry,
                policy=self.controller_policy,
                monotonic=self._monotonic,
                event_callback=self.publish,
            )
            return await controller.run(
                controller_request,
                cancellation=active.cancellation,
                started_at_monotonic=active.started_at_monotonic,
                initial_tool_calls=preparation_tool_calls,
            )
        except Exception as error:
            if self.run_store.read_metadata(run_id) is not None:
                raise
            return await self._persist_preparation_failure(run_id, active, error)
        finally:
            active.phase = ManagedRunPhase.FINALIZED
            self.publish_signal(run_id)
            async with self._lock:
                self._active.pop(run_id, None)

    @staticmethod
    def _uses_model(request: RunCreateRequest) -> bool:
        if request.message is not None:
            return (
                request.budget.max_total_model_tokens > 0
                and request.budget.max_generated_tokens > 0
            )
        return (
            request.enable_model
            and request.budget.max_tool_calls > 0
            and request.budget.max_total_model_tokens > 0
            and request.budget.max_generated_tokens > 0
            and request.requested_tier
            not in {BudgetTier.BASELINE_ONLY, BudgetTier.DETERMINISTIC_IMPROVEMENT}
        )

    async def _resolve_source(self, run_id: str, active: _ActiveRun) -> tuple[str, str | None, int]:
        source = active.request.source
        if isinstance(source, StoredProblemSource):
            self.artifact_store.get_metadata(source.problem_artifact_id)
            if source.baseline_plan_artifact_id is not None:
                self.artifact_store.get_metadata(source.baseline_plan_artifact_id)
            return source.problem_artifact_id, source.baseline_plan_artifact_id, 0
        if isinstance(source, InlineProblemSource):
            inline_problem_id, inline_baseline_id = self._publish_inline_source(source)
            return inline_problem_id, inline_baseline_id, 0
        if isinstance(source, ExampleProblemSource):
            required_tool_calls = 5
            if active.request.budget.max_tool_calls < required_tool_calls:
                raise ValueError(
                    f"frozen example preparation requires {required_tool_calls} tool calls"
                )
            deadline = Deadline(
                active.request.budget,
                self.controller_policy,
                monotonic=self._monotonic,
                started_at=active.started_at_monotonic,
            )
            prepared = await prepare_example(
                source,
                store=self.artifact_store,
                registry=self.tool_registry,
                context=ToolContext(
                    run_id=run_id,
                    artifact_store=self.artifact_store,
                    deadline_monotonic=deadline.search_deadline_monotonic,
                    cancellation=active.cancellation,
                    seed=active.request.seed,
                    monotonic=self._monotonic,
                ),
            )
            return (
                prepared.problem_artifact_id,
                prepared.baseline_plan_artifact_id,
                prepared.tool_calls,
            )
        if isinstance(source, CompileProblemSource):
            if active.request.budget.max_tool_calls < 1:
                raise ValueError("structured problem compilation requires one tool call")
            deadline = Deadline(
                active.request.budget,
                self.controller_policy,
                monotonic=self._monotonic,
                started_at=active.started_at_monotonic,
            )
            context = ToolContext(
                run_id=run_id,
                artifact_store=self.artifact_store,
                deadline_monotonic=deadline.search_deadline_monotonic,
                cancellation=active.cancellation,
                seed=active.request.seed,
                monotonic=self._monotonic,
            )
            result = await invoke_tool(
                self.tool_registry.get("compile_problem"), source.arguments, context
            )
            if result.status is not ToolResultStatus.COMPLETE:
                message = result.error.message if result.error is not None else "compilation failed"
                raise ValueError(message)
            problem_id = result.metrics.get("problem_artifact_id")
            baseline_id = result.metrics.get("baseline_plan_artifact_id")
            if not isinstance(problem_id, str) or not isinstance(baseline_id, str):
                raise ValueError("compile_problem did not return the required artifact IDs")
            return problem_id, baseline_id, 1
        raise TypeError("unsupported run source")

    def _publish_inline_source(self, source: InlineProblemSource) -> tuple[str, str | None]:
        references = _problem_references(source.problem)
        for reference in references:
            stored = self.artifact_store.get_metadata(reference.id)
            if stored.content_hash != reference.content_hash:
                raise ValueError("inline problem contains an artifact identity mismatch")
        privacy = max((ref.privacy for ref in references), key=_PRIVACY_ORDER.__getitem__)
        licenses = " AND ".join(sorted({ref.license or "unknown" for ref in references}))
        problem_ref = put_json(
            self.artifact_store,
            source.problem.model_dump(mode="json"),
            kind=ArtifactKind.JSON_SPECIFICATION,
            units="unitless",
            provenance=ArtifactProvenance(
                source_uri=f"oasis://api/v1/inline-problem/{source.problem.problem_hash}",
                source_provider="oasis-api",
                source_version="1.0.0",
                license=licenses,
                privacy=privacy,
                lineage=ArtifactLineage(
                    parent_ids=tuple(reference.id for reference in references),
                    transformations=(
                        ArtifactTransformation(
                            name="publish_inline_problem",
                            version="1.0.0",
                            parameters={"problem_hash": source.problem.problem_hash},
                        ),
                    ),
                ),
            ),
            data_schema={
                "type": type(source.problem).__name__,
                "version": source.problem.schema_version,
            },
        )
        if source.baseline_plan is None:
            return problem_ref.id, None
        plan_ref = put_json(
            self.artifact_store,
            source.baseline_plan.model_dump(mode="json"),
            kind=ArtifactKind.PLAN,
            units="unitless",
            provenance=ArtifactProvenance(
                source_uri=f"oasis://api/v1/inline-baseline/{source.problem.problem_hash}",
                source_provider="oasis-api",
                source_version="1.0.0",
                license=licenses,
                privacy=privacy,
                lineage=ArtifactLineage(
                    parent_ids=(problem_ref.id,),
                    transformations=(
                        ArtifactTransformation(
                            name="publish_inline_baseline",
                            version="1.0.0",
                            parameters={"problem_hash": source.problem.problem_hash},
                        ),
                    ),
                ),
            ),
            data_schema={"type": "Plan", "version": source.baseline_plan.schema_version},
        )
        return problem_ref.id, plan_ref.id

    async def _persist_preparation_failure(
        self, run_id: str, active: _ActiveRun, error: Exception
    ) -> RunResult:
        deadline = Deadline(
            active.request.budget,
            self.controller_policy,
            monotonic=self._monotonic,
            started_at=active.started_at_monotonic,
        )
        budget = BudgetAccount(active.request.budget, deadline)
        state = StateMachine()
        self.run_store.create(
            RunMetadata(
                run_id=run_id,
                problem_artifact_id=_PENDING_ARTIFACT_ID,
                seed=active.request.seed,
                metadata={"preparation_failed": True, "controller_version": "1.0.0"},
            )
        )
        journal = EventJournal(
            run_id=run_id,
            deadline=deadline,
            budget=budget,
            append=self.run_store.append_event,
            state=state,
            callback=self.publish,
        )
        await journal.emit(EventKind.RUN_CREATED, payload={"budget_tier": "baseline_only"})
        state.transition(ControllerState.GROUNDING)
        if active.cancellation.cancelled:
            reason = TerminalReason.USER_CANCELLED
            status = RunStatus.CANCELLED
        elif deadline.expired:
            reason = TerminalReason.TIME_EXHAUSTED
            status = RunStatus.REJECTED
        elif isinstance(error, ArtifactNotFoundError):
            reason = TerminalReason.MISSING_EVIDENCE
            status = RunStatus.REJECTED
        elif isinstance(error, ModelBackendError):
            reason = TerminalReason.UNSUPPORTED_CAPABILITY
            status = RunStatus.REJECTED
        else:
            reason = TerminalReason.INVALID_REQUEST
            status = RunStatus.REJECTED
        state.transition(ControllerState.QUIESCING)
        await journal.emit(
            EventKind.BUDGET_CHECKPOINT,
            payload={"checkpoint": "quiescing", "reason": reason.value},
        )
        state.transition(ControllerState.FINALIZED)
        await journal.emit(
            EventKind.RUN_FINALIZED,
            actor=EventActor.CONTROLLER,
            payload={"reason": reason.value, "has_incumbent": False},
        )
        result = RunResult(
            run_id=run_id,
            status=status,
            terminal_reason=reason,
            budget_tier=BudgetTier.BASELINE_ONLY,
            problem_artifact_id=_PENDING_ARTIFACT_ID,
            answer=(
                "The model could not start. Please try again." if active.request.message else None
            ),
            answer_source="status" if active.request.message else None,
            requested_budget=active.request.budget,
            consumed_budget=budget.snapshot(),
            deadline_overshoot_ms=deadline.overshoot_ms,
            failures=(
                f"Run preparation failed before controller admission ({type(error).__name__}).",
            ),
            runtime_plan=self.model_service.runtime_plan_model(active.backend),
            compute_inventory=self.model_service.inventory_model(active.backend).sanitized(),
            hardware_validation=self.model_service.runtime_plan_model(
                active.backend
            ).hardware_validation.value,
            model_profile=(active.backend or self.model_service.backend).profile.name,
            model_id=(active.backend or self.model_service.backend).profile.model_id,
            tool_versions={spec.name: spec.version for spec in self.tool_registry.list()},
            seed=active.request.seed,
            event_count=journal.count,
        )
        self.run_store.write_result(result)
        return result

    def publish(self, event: ControllerEvent) -> None:
        """Wake subscribers without awaiting them or retaining event payload copies."""

        self.publish_signal(event.run_id)

    def publish_signal(self, run_id: str) -> None:
        for queue in tuple(self._subscribers.get(run_id, ())):
            if queue.empty():
                queue.put_nowait(None)

    async def subscribe(self, run_id: str) -> asyncio.Queue[None]:
        if not await self.exists(run_id):
            raise RunManagerError(404, "run_not_found", "The requested run does not exist.")
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        async with self._lock:
            self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue[None]) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(run_id)
            if subscribers is None:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(run_id, None)

    async def exists(self, run_id: str) -> bool:
        async with self._lock:
            if run_id in self._active:
                return True
        return self.run_store.read_metadata(run_id) is not None

    def read_events(self, run_id: str, *, after_sequence: int) -> tuple[ControllerEvent, ...]:
        metadata = self.run_store.read_metadata(run_id)
        if metadata is None:
            return ()
        return self.run_store.read_events(run_id, after_sequence=after_sequence)

    async def is_active(self, run_id: str) -> bool:
        async with self._lock:
            return run_id in self._active

    async def wait(self, run_id: str) -> RunResult:
        """Await a run for the message-to-answer HTTP and CLI convenience interfaces."""
        async with self._lock:
            active = self._active.get(run_id)
        if active is not None and active.task is not None:
            return await asyncio.shield(active.task)
        result = self.run_store.read_result(run_id)
        if result is None:
            raise RunManagerError(404, "run_not_found", "The requested run has no result.")
        return result

    async def inspect(self, run_id: str) -> RunInspectionResponse:
        async with self._lock:
            active = self._active.get(run_id)
        metadata = self.run_store.read_metadata(run_id)
        if active is None and metadata is None:
            raise RunManagerError(404, "run_not_found", "The requested run does not exist.")
        result = self.run_store.read_result(run_id) if metadata is not None else None
        events = self.run_store.read_events(run_id) if metadata is not None else ()
        if result is not None:
            phase = ManagedRunPhase.FINALIZED
            state = ControllerState.FINALIZED
        elif active is not None:
            phase = active.phase
            state = events[-1].state if events else ControllerState.RECEIVED
        else:
            phase = ManagedRunPhase.RUNNING
            state = events[-1].state if events else ControllerState.RECEIVED
        return RunInspectionResponse(
            run_id=run_id,
            phase=phase,
            controller_state=state,
            last_event_id=events[-1].sequence if events else None,
            cancellation_requested=active.cancellation.cancelled if active is not None else False,
            result=result,
            links=run_links(run_id),
        )

    async def cancel(self, run_id: str) -> CancelRunResponse:
        async with self._lock:
            active = self._active.get(run_id)
        if active is None:
            metadata = self.run_store.read_metadata(run_id)
            if metadata is None:
                raise RunManagerError(404, "run_not_found", "The requested run does not exist.")
            result = self.run_store.read_result(run_id)
            return CancelRunResponse(
                run_id=run_id,
                cancellation_requested=result is not None
                and result.terminal_reason is TerminalReason.USER_CANCELLED,
                already_finalized=result is not None,
                result=result,
            )
        active.cancellation.cancel("user requested cancellation through the service API")
        task = active.task
        result = None
        if task is not None and self.cancel_wait_seconds > 0:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task), timeout=self.cancel_wait_seconds
                )
            except TimeoutError:
                pass
        return CancelRunResponse(
            run_id=run_id,
            cancellation_requested=True,
            already_finalized=False,
            result=result,
        )

    async def render_map(self, run_id: str, *, format_name: str) -> ArtifactRef:
        inspection = await self.inspect(run_id)
        result = inspection.result
        if result is not None:
            problem_artifact_id = result.problem_artifact_id
            plan_artifact_id = result.best_plan_artifact_id
            seed = result.seed
        else:
            metadata = self.run_store.read_metadata(run_id)
            events = self.run_store.read_events(run_id) if metadata is not None else ()
            incumbent_event = next(
                (
                    event
                    for event in reversed(events)
                    if event.kind in {EventKind.BASELINE_COMMITTED, EventKind.INCUMBENT_COMMITTED}
                    and event.artifact_ids
                ),
                None,
            )
            if metadata is None or incumbent_event is None:
                raise RunManagerError(
                    409,
                    "map_not_ready",
                    "The run has not committed a validated plan yet.",
                )
            event_problem_id = incumbent_event.payload.get("problem_artifact_id")
            problem_artifact_id = (
                event_problem_id
                if isinstance(event_problem_id, str)
                else metadata.problem_artifact_id
            )
            plan_artifact_id = incumbent_event.artifact_ids[0]
            seed = metadata.seed
        if plan_artifact_id is None or problem_artifact_id is None:
            raise RunManagerError(422, "map_unavailable", "The run has no validated plan to map.")
        context = ToolContext(
            run_id=f"{run_id}-map",
            artifact_store=self.artifact_store,
            deadline_monotonic=self._monotonic() + 5.0,
            cancellation=CancellationToken(),
            seed=seed,
            monotonic=self._monotonic,
        )
        rendered = await invoke_tool(
            self.tool_registry.get("render_map"),
            {
                "problem_artifact_id": problem_artifact_id,
                "plan_artifact_id": plan_artifact_id,
                "format": format_name,
            },
            context,
        )
        artifact_id = rendered.metrics.get("map_artifact_id")
        if rendered.status is not ToolResultStatus.COMPLETE or not isinstance(artifact_id, str):
            raise RunManagerError(422, "map_unavailable", "This run cannot be rendered as a map.")
        return self.artifact_store.get_metadata(artifact_id)

    async def close(self) -> None:
        async with self._lock:
            active = tuple(self._active.values())
        for record in active:
            record.cancellation.cancel("service shutdown")
        tasks = tuple(record.task for record in active if record.task is not None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["RunManager", "RunManagerError", "run_links"]
