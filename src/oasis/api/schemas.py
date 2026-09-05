"""Versioned HTTP wire contracts for the OASIS service."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from oasis.config import DevicePolicy, RuntimeConfig, RuntimeEngine
from oasis.controller import BudgetSpec, BudgetTier, ControllerState, RunResult
from oasis.llm import ChatMessage, FinishReason, ModelCapabilities, TokenUsage, ToolCall
from oasis.problems import LocationAllocationProblem, RouteServiceProblem
from oasis.schemas import Plan, ToolSpec

API_SCHEMA_VERSION = "1.2.0"


class HealthResponse(BaseModel):
    """Small liveness response with no hardware or model probing."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    status: Literal["ok"] = "ok"
    api_version: Literal["v1"] = "v1"


class ModelCatalogEntry(BaseModel):
    """One configured model profile and its declared capabilities."""

    model_config = ConfigDict(frozen=True)

    name: str
    model_id: str
    family: str
    context_limit: int | None = None
    is_default: bool = False
    capabilities: ModelCapabilities


class ModelCatalogResponse(BaseModel):
    """Stable model discovery envelope."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    active_profile: str
    active_model_id: str
    models: tuple[ModelCatalogEntry, ...]


class RuntimeCapabilities(BaseModel):
    """Installed runtime capabilities without active device discovery."""

    model_config = ConfigDict(frozen=True)

    fake: bool = True
    transformers: bool
    accelerate: bool
    remote: bool = False


class RuntimeOptions(BaseModel):
    """Server-authorized choices used to build runtime controls dynamically."""

    model_config = ConfigDict(frozen=True)

    devices: tuple[DevicePolicy, ...]
    engines: tuple[RuntimeEngine, ...]
    dtypes: tuple[str, ...]
    quantizations: tuple[str, ...]
    attention_backends: tuple[str, ...]


class RuntimeResponse(BaseModel):
    """Non-probing runtime report extended in Phase 9 without a wire break."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    requested_policy: RuntimeConfig
    resolved_plan: dict[str, JsonValue]
    capabilities: RuntimeCapabilities
    options: RuntimeOptions
    inventory: dict[str, JsonValue]
    inventory_probed: bool = False
    model_loaded: bool = False
    model_lifecycle: Literal["lazy", "warmup"] = "lazy"
    model_startup_ms: int = Field(default=0, ge=0)


class ToolCatalogEntry(BaseModel):
    """API-safe projection of one registry tool specification."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    capability_tags: frozenset[str]
    problem_tags: frozenset[str]
    side_effects: str
    privacy: str
    p50_ms: int = Field(ge=0)
    p95_ms: int = Field(ge=0)
    streams_progress: bool
    streams_candidates: bool
    streams_bounds: bool
    cooperative_cancellation: bool
    resumable: bool

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> ToolCatalogEntry:
        return cls(
            name=spec.name,
            version=spec.version,
            description=spec.description,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
            capability_tags=spec.capability_tags,
            problem_tags=spec.problem_tags,
            side_effects=spec.side_effects.value,
            privacy=spec.privacy.value,
            p50_ms=spec.runtime.p50_ms,
            p95_ms=spec.runtime.p95_ms,
            streams_progress=spec.streams_progress,
            streams_candidates=spec.streams_candidates,
            streams_bounds=spec.streams_bounds,
            cooperative_cancellation=spec.cooperative_cancellation,
            resumable=spec.resumable,
        )


class ToolCatalogResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    tools: tuple[ToolCatalogEntry, ...]


class ProblemCatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    type_id: str
    version: str


class ProblemExampleEntry(BaseModel):
    """One server-owned public-health example that can be launched without artifact IDs."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    name: str
    description: str
    problem_type: str
    evidence_summary: str
    group_names: tuple[str, ...] = ()
    equity_templates: tuple[Literal["overall", "floors", "max_min"], ...]
    default_equity_template: Literal["overall", "floors", "max_min"]
    default_group_floors: dict[str, float] = Field(default_factory=dict)
    preparation_tool_calls: int = Field(default=5, ge=0)


class ProblemCatalogResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    problems: tuple[ProblemCatalogEntry, ...]
    examples: tuple[ProblemExampleEntry, ...] = ()


class ChatRequest(BaseModel):
    """One bounded raw chat request using the service-owned model lifecycle."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=128)
    max_generated_tokens: int = Field(default=512, ge=1, le=32_768)
    thinking_enabled: bool = False
    seed: int = 0


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    model_profile: str
    model_id: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage
    finish_reason: FinishReason
    model_startup_ms: int = Field(ge=0)


class StoredProblemSource(BaseModel):
    """Reference an immutable problem already present in the artifact store."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["artifact"]
    problem_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    baseline_plan_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")


class InlineProblemSource(BaseModel):
    """Publish a complete compiled problem, and optionally a baseline, before admission."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["inline"]
    problem: LocationAllocationProblem | RouteServiceProblem
    baseline_plan: Plan | None = None

    @model_validator(mode="after")
    def baseline_matches_problem(self) -> Self:
        if (
            self.baseline_plan is not None
            and self.baseline_plan.problem_type != self.problem.type_id
        ):
            raise ValueError("inline baseline problem type must match the compiled problem")
        return self


class CompileProblemSource(BaseModel):
    """Run the stable compile_problem tool from structured evidence and policy arguments."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["compile_problem"]
    arguments: dict[str, JsonValue]


class ExampleProblemSource(BaseModel):
    """Ask the service to materialize one advertised frozen demonstration problem."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["example"]
    example_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    equity_template: Literal["overall", "floors", "max_min"] = "overall"
    group_floors: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def floors_are_probabilities(self) -> Self:
        if any(value < 0.0 or value > 1.0 for value in self.group_floors.values()):
            raise ValueError("example group floors must lie between zero and one")
        return self


RunSource = Annotated[
    StoredProblemSource | InlineProblemSource | CompileProblemSource | ExampleProblemSource,
    Field(discriminator="kind"),
]


class RunCreateRequest(BaseModel):
    """Start from a message, or explicitly use the legacy prepared-problem interface."""

    model_config = ConfigDict(frozen=True)

    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None = None
    source: RunSource | None = None
    budget: BudgetSpec = Field(
        default_factory=lambda: BudgetSpec(
            wall_time_ms=120_000,
            max_total_model_tokens=512_000,
            max_generated_tokens=32_768,
            max_tool_calls=64,
        )
    )
    seed: int = 0
    enable_model: bool = True
    enable_deterministic_fallback: bool = True
    allowed_tools: tuple[str, ...] | None = None
    thinking_enabled: bool = False
    requested_tier: BudgetTier | None = None
    model_profile: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    model_id: str | None = Field(default=None, min_length=1, max_length=512)
    runtime_policy: RuntimeConfig | None = None

    @model_validator(mode="after")
    def one_input(self) -> Self:
        if (self.message is None) == (self.source is None):
            raise ValueError("supply exactly one of message or source")
        if self.allowed_tools is not None and len(set(self.allowed_tools)) != len(
            self.allowed_tools
        ):
            raise ValueError("allowed tool names must be unique")
        if self.message is not None and not self.enable_model:
            raise ValueError("message runs require the model")
        return self


class MessageRequest(BaseModel):
    """The public message-to-answer interface; configuration belongs to the server."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RunLinks(BaseModel):
    model_config = ConfigDict(frozen=True)

    self: str
    events: str
    cancel: str
    map: str


class RunCreatedResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    run_id: str
    status: Literal["accepted"] = "accepted"
    links: RunLinks


class ManagedRunPhase(StrEnum):
    PREPARING = "preparing"
    RUNNING = "running"
    FINALIZED = "finalized"


class RunInspectionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    run_id: str
    phase: ManagedRunPhase
    controller_state: ControllerState
    last_event_id: int | None = Field(default=None, ge=0)
    cancellation_requested: bool = False
    result: RunResult | None = None
    links: RunLinks


class CancelRunResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    run_id: str
    cancellation_requested: bool
    already_finalized: bool
    result: RunResult | None = None


class ApiErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    fields: tuple[str, ...] = ()


class ApiErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = API_SCHEMA_VERSION
    error: ApiErrorDetail
