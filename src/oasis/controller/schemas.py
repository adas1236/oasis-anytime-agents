"""Serializable contracts for anytime runs, actions, traces, and results."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from oasis.llm.schemas import TokenUsage
from oasis.problems.schemas import Comparison, ResultView, Scorecard
from oasis.runtimes import (
    ComputeInventory,
    HardwareValidationStatus,
    RuntimeKind,
    RuntimePlan,
    fake_inventory,
)
from oasis.schemas import Plan


class ControllerState(StrEnum):
    """Validated lifecycle states for one immutable run."""

    RECEIVED = "received"
    GROUNDING = "grounding"
    PROBLEM_LOCKED = "problem_locked"
    ADMITTED = "admitted"
    BASELINE_COMMITTED = "baseline_committed"
    SEARCHING = "searching"
    QUIESCING = "quiescing"
    FINALIZED = "finalized"


class TerminalReason(StrEnum):
    """Stable explanations for why controller work stopped."""

    BASELINE_ONLY = "baseline_only"
    BUDGET_TIER_COMPLETE = "budget_tier_complete"
    INVALID_REQUEST = "invalid_request"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    BUDGET_TOO_SMALL = "budget_too_small"
    MISSING_EVIDENCE = "missing_evidence"
    INFEASIBLE_PROBLEM = "infeasible_problem"
    USER_CANCELLED = "user_cancelled"
    TIME_EXHAUSTED = "time_exhausted"
    TOKEN_EXHAUSTED_NO_DETERMINISTIC_WORK = "token_exhausted_no_deterministic_work"
    TARGET_REACHED = "target_reached"
    PROVEN_OPTIMAL = "proven_optimal"
    PLATEAU = "plateau"
    MODEL_STOPPED = "model_stopped"
    INTERNAL_FAILURE = "internal_failure"


class RunStatus(StrEnum):
    """Coarse public outcome independent of the detailed terminal reason."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BudgetTier(StrEnum):
    """Increasing levels of work enabled by the supplied budgets."""

    BASELINE_ONLY = "baseline_only"
    DETERMINISTIC_IMPROVEMENT = "deterministic_improvement"
    ONE_SHOT_MODEL = "one_shot_model"
    ITERATIVE_MODEL = "iterative_model"


class BudgetSpec(BaseModel):
    """Hard aggregate wall, model-token, generation, and tool-call limits."""

    model_config = ConfigDict(frozen=True)

    wall_time_ms: int = Field(gt=0)
    max_total_model_tokens: int = Field(default=0, ge=0)
    max_generated_tokens: int = Field(default=0, ge=0)
    max_tool_calls: int = Field(default=0, ge=0)
    finalization_reserve_ms: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def reserve_fits_wall_budget(self) -> Self:
        if (
            self.finalization_reserve_ms is not None
            and self.finalization_reserve_ms >= self.wall_time_ms
        ):
            raise ValueError("finalization reserve must be smaller than the wall-time budget")
        if self.max_generated_tokens > self.max_total_model_tokens:
            raise ValueError("generated-token budget cannot exceed total model-token budget")
        return self


class BudgetSnapshot(BaseModel):
    """Reproducible accounting snapshot attached to every trace event."""

    model_config = ConfigDict(frozen=True)

    wall_elapsed_ms: int = Field(ge=0)
    wall_remaining_ms: int = Field(ge=0)
    search_remaining_ms: int = Field(ge=0)
    model_usage: TokenUsage = Field(default_factory=TokenUsage)
    remaining_total_model_tokens: int = Field(ge=0)
    remaining_generated_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    remaining_tool_calls: int = Field(ge=0)


class EventKind(StrEnum):
    """Controller event names required by the anytime trace contract."""

    RUN_CREATED = "run_created"
    EVIDENCE_SNAPSHOT_LOCKED = "evidence_snapshot_locked"
    PROBLEM_COMPILED = "problem_compiled"
    BASELINE_COMMITTED = "baseline_committed"
    MODEL_ACTION_PROPOSED = "model_action_proposed"
    ACTION_ADMITTED = "action_admitted"
    ACTION_REJECTED = "action_rejected"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_COMPLETED = "tool_completed"
    TOOL_CANCELLED = "tool_cancelled"
    TOOL_FAILED = "tool_failed"
    CANDIDATE_REJECTED = "candidate_rejected"
    INCUMBENT_COMMITTED = "incumbent_committed"
    BOUND_VERIFIED = "bound_verified"
    FALLBACK_INVOKED = "fallback_invoked"
    BUDGET_CHECKPOINT = "budget_checkpoint"
    RUN_FINALIZED = "run_finalized"


class EventActor(StrEnum):
    """Originator of a persisted event."""

    CONTROLLER = "controller"
    MODEL = "model"
    TOOL = "tool"
    EVALUATOR = "evaluator"
    USER = "user"


class ControllerEvent(BaseModel):
    """One append-only, ordered, redacted run event."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=0)
    kind: EventKind
    state: ControllerState
    relative_monotonic_ms: int = Field(ge=0)
    timestamp: datetime
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    run_generation: int = Field(default=1, ge=1)
    action_id: str | None = None
    action_generation: int | None = Field(default=None, ge=1)
    actor: EventActor
    budget_before: BudgetSnapshot
    budget_after: BudgetSnapshot
    artifact_ids: tuple[str, ...] = ()
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def timestamp_and_action_are_consistent(self) -> Self:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("event timestamps must be timezone-aware UTC values")
        if (self.action_id is None) != (self.action_generation is None):
            raise ValueError("action ID and generation must be supplied together")
        return self


class CallToolAction(BaseModel):
    """One model-proposed invocation of an exposed search tool."""

    model_config = ConfigDict(frozen=True)

    type: Literal["call_tool"]
    tool: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=500)


class SubmitCandidateAction(BaseModel):
    """One model-authored candidate which still requires independent evaluation."""

    model_config = ConfigDict(frozen=True)

    type: Literal["submit_candidate"]
    candidate: Plan
    rationale: str = Field(min_length=1, max_length=500)


class StopAction(BaseModel):
    """A request to stop model-directed exploration and retain the incumbent."""

    model_config = ConfigDict(frozen=True)

    type: Literal["stop"]
    rationale: str = Field(min_length=1, max_length=500)


ControllerAction = Annotated[
    CallToolAction | SubmitCandidateAction | StopAction,
    Field(discriminator="type"),
]


class CompactTool(BaseModel):
    """Small tool description included in model state without full trace replay."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    p95_ms: int = Field(ge=0)
    streams_candidates: bool
    resumable: bool


class CompactModelState(BaseModel):
    """Bounded decision context for one model action."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    run_id: str
    problem_type: str
    problem_hash: str
    incumbent_plan_artifact_id: str
    incumbent_comparator_key: tuple[float, ...]
    incumbent_metrics: dict[str, float]
    verified_bound_artifact_id: str | None = None
    recent_actions: tuple[str, ...] = Field(default=(), max_length=8)
    available_tools: tuple[CompactTool, ...] = Field(default=(), max_length=16)
    remaining_total_model_tokens: int = Field(ge=0)
    remaining_generated_tokens: int = Field(ge=0)
    remaining_tool_calls: int = Field(ge=0)
    remaining_wall_ms: int = Field(ge=0)


class ActionStatus(StrEnum):
    """Lifecycle status of a controller-admitted action."""

    ADMITTED = "admitted"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"


class ActionRecord(BaseModel):
    """Ledger record used to reject duplicates and stale action generations."""

    model_config = ConfigDict(frozen=True)

    action_id: str
    generation: int = Field(ge=1)
    tool_name: str | None = None
    fingerprint: str
    status: ActionStatus
    admitted_at_ms: int = Field(ge=0)
    subdeadline_monotonic: float | None = None


class IncumbentRecord(BaseModel):
    """The sole returnable, independently evaluated plan for an immutable problem."""

    model_config = ConfigDict(frozen=True)

    plan: Plan
    scorecard: Scorecard
    plan_artifact_id: str
    scorecard_artifact_id: str
    problem_hash: str
    evidence_hash: str
    policy_hash: str
    comparator_key: tuple[float, ...]
    source_action_id: str
    committed_at: datetime
    committed_at_ms: int = Field(ge=0)
    seed: int

    @model_validator(mode="after")
    def identities_match_scorecard(self) -> Self:
        score = self.scorecard
        if not score.feasible:
            raise ValueError("an incumbent must be feasible")
        if (
            self.problem_hash != score.problem_hash
            or self.evidence_hash != score.evidence_hash
            or self.policy_hash != score.policy_hash
            or self.comparator_key != score.comparator_key
        ):
            raise ValueError("incumbent identities must match its authoritative scorecard")
        if self.committed_at.tzinfo is None or self.committed_at.utcoffset() != timedelta(0):
            raise ValueError("incumbent commit timestamps must use UTC")
        return self


class ControllerPolicy(BaseModel):
    """Small, testable controller timing and circuit-breaker configuration."""

    model_config = ConfigDict(frozen=True)

    reserve_fraction: float = Field(default=0.05, ge=0.0, le=0.5)
    minimum_finalization_reserve_ms: int = Field(default=50, ge=1)
    maximum_finalization_reserve_ms: int = Field(default=2_000, ge=1)
    minimum_baseline_budget_ms: int = Field(default=10, ge=0)
    validation_reserve_ms: int = Field(default=2, ge=0)
    cancellation_grace_ms: int = Field(default=25, ge=0, le=5_000)
    max_no_progress_actions: int = Field(default=2, ge=1)
    one_shot_total_token_threshold: int = Field(default=1_024, ge=1)
    max_model_actions: int = Field(default=8, ge=1)
    max_candidates_per_action: int = Field(default=1_000, ge=1, le=1_000_000)
    recent_action_limit: int = Field(default=6, ge=1, le=8)
    schema_repair_attempts: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def reserve_bounds_are_ordered(self) -> Self:
        if self.minimum_finalization_reserve_ms > self.maximum_finalization_reserve_ms:
            raise ValueError("minimum finalization reserve cannot exceed its maximum")
        return self


class RunRequest(BaseModel):
    """Framework-neutral request to search one already immutable problem."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    run_generation: int = Field(default=1, ge=1)
    problem_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    baseline_plan_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    budget: BudgetSpec
    seed: int = 0
    enable_model: bool = True
    enable_deterministic_fallback: bool = True
    allowed_tools: tuple[str, ...] = Field(default=("improve",), max_length=16)
    thinking_enabled: bool = False
    requested_tier: BudgetTier | None = None
    runtime_plan: RuntimePlan = Field(default_factory=lambda: _default_runtime_plan())
    compute_inventory: ComputeInventory = Field(default_factory=fake_inventory)

    @model_validator(mode="after")
    def tools_are_unique(self) -> Self:
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("allowed tool names must be unique")
        return self


def _default_runtime_plan() -> RuntimePlan:
    return RuntimePlan(
        requested_profile="unspecified",
        requested_model_id="unspecified",
        runtime=RuntimeKind.FAKE,
        device_placement=("cpu",),
        dtype="fake",
        attention_backend="fake",
        rationale=("No model runtime was supplied for this deterministic run.",),
        hardware_validation=HardwareValidationStatus.NOT_APPLICABLE,
    )


class RunResult(BaseModel):
    """Deterministic final response that never requires another model call."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.1.0"
    run_id: str
    run_generation: int = Field(default=1, ge=1)
    status: RunStatus
    terminal_reason: TerminalReason
    final_state: Literal[ControllerState.FINALIZED] = ControllerState.FINALIZED
    budget_tier: BudgetTier
    problem_artifact_id: str
    problem_hash: str | None = None
    evidence_hash: str | None = None
    policy_hash: str | None = None
    best_plan: Plan | None = None
    best_scorecard: Scorecard | None = None
    best_plan_artifact_id: str | None = None
    best_scorecard_artifact_id: str | None = None
    result_view: ResultView | None = None
    baseline_comparison: Comparison | None = None
    verified_bound_artifact_id: str | None = None
    requested_budget: BudgetSpec
    consumed_budget: BudgetSnapshot
    deadline_overshoot_ms: int = Field(ge=0)
    time_to_first_feasible_ms: int | None = Field(default=None, ge=0)
    incumbent_timeline: tuple[IncumbentRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    runtime_plan: RuntimePlan = Field(default_factory=_default_runtime_plan)
    compute_inventory: ComputeInventory = Field(default_factory=fake_inventory)
    hardware_validation: Literal["not_applicable", "pending", "passed", "failed"] = "not_applicable"
    model_profile: str | None = None
    model_id: str | None = None
    problem_plugin_version: str | None = None
    evaluator_version: str | None = None
    tool_versions: dict[str, str] = Field(default_factory=dict)
    controller_version: str = "1.0.0"
    seed: int
    event_count: int = Field(ge=1)

    @model_validator(mode="after")
    def result_artifacts_are_consistent(self) -> Self:
        plan_fields = (
            self.best_plan,
            self.best_scorecard,
            self.best_plan_artifact_id,
            self.best_scorecard_artifact_id,
            self.result_view,
        )
        if any(value is not None for value in plan_fields) and any(
            value is None for value in plan_fields
        ):
            raise ValueError("a final incumbent requires plan, scorecard, IDs, and result view")
        if self.model_id is not None and self.runtime_plan.requested_model_id not in {
            self.model_id,
            "unspecified",
        }:
            raise ValueError("runtime plan must preserve the result model ID")
        return self
