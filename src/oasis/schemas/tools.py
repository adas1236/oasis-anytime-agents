"""Serializable contracts for registry tools, results, and streamed events."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from oasis.schemas.artifacts import ArtifactKind, ArtifactRef, PrivacyClassification
from oasis.schemas.plans import Plan

MAX_TOOL_SUMMARY_BYTES = 8_192


class SideEffectClassification(StrEnum):
    """Externally observable mutations a tool may perform."""

    NONE = "none"
    LOCAL_WRITE = "local_write"
    EXTERNAL_READ = "external_read"
    EXTERNAL_WRITE = "external_write"


class DeterminismClassification(StrEnum):
    """How repeatability relates to inputs, evidence, and seed."""

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    EXTERNAL = "external"


class ToolResultStatus(StrEnum):
    """Portable terminal outcomes for a tool invocation."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INFEASIBLE = "infeasible"
    AMBIGUOUS = "ambiguous"
    EXPIRED = "expired"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"


class ToolErrorCode(StrEnum):
    """Stable tool failure categories suitable for traces and model summaries."""

    INVALID_ARGUMENTS = "invalid_arguments"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    NOT_FOUND = "not_found"
    CAPABILITY_DENIED = "capability_denied"
    PROVIDER_FAILURE = "provider_failure"
    INTERNAL_ERROR = "internal_error"


class ToolError(BaseModel):
    """Compact, structured error safe to return across process boundaries."""

    model_config = ConfigDict(frozen=True)

    code: ToolErrorCode
    message: str = Field(min_length=1, max_length=1_000)
    retryable: bool = False
    context: dict[str, JsonValue] = Field(default_factory=dict)


class ToolCostEstimate(BaseModel):
    """Estimated abstract cost units derived from numeric instance features."""

    model_config = ConfigDict(frozen=True)

    units: float = Field(ge=0.0)
    features: dict[str, float] = Field(default_factory=dict)


class ToolCostModel(BaseModel):
    """Serializable linear estimator usable before a tool is admitted."""

    model_config = ConfigDict(frozen=True)

    base_units: float = Field(default=0.0, ge=0.0)
    feature_weights: dict[str, Annotated[float, Field(ge=0.0)]] = Field(default_factory=dict)

    def estimate(self, features: dict[str, float]) -> ToolCostEstimate:
        """Apply the declared estimator to non-negative instance features."""

        if any(value < 0 for value in features.values()):
            raise ValueError("cost-estimation features must be non-negative")
        units = self.base_units + sum(
            weight * features.get(name, 0.0) for name, weight in self.feature_weights.items()
        )
        return ToolCostEstimate(units=units, features=features)


class ToolRuntimeEstimate(BaseModel):
    """Runtime quantiles and first-useful-output estimate in milliseconds."""

    model_config = ConfigDict(frozen=True)

    p50_ms: int = Field(ge=0)
    p95_ms: int = Field(ge=0)
    time_to_first_candidate_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def quantiles_are_ordered(self) -> Self:
        if self.p50_ms > self.p95_ms:
            raise ValueError("p50 runtime must not exceed p95 runtime")
        if (
            self.time_to_first_candidate_ms is not None
            and self.time_to_first_candidate_ms > self.p95_ms
        ):
            raise ValueError("time to first candidate must not exceed p95 runtime")
        return self


class ToolSpec(BaseModel):
    """Stable model-facing and controller-facing declaration of a tool."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    version: str
    description: str = Field(min_length=1, max_length=500)
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    capability_tags: frozenset[str] = frozenset()
    problem_tags: frozenset[str] = frozenset()
    artifact_tags: frozenset[ArtifactKind] = frozenset()
    side_effects: SideEffectClassification = SideEffectClassification.NONE
    privacy: PrivacyClassification = PrivacyClassification.PUBLIC
    required_providers: frozenset[str] = frozenset()
    required_resources: frozenset[str] = frozenset()
    determinism: DeterminismClassification = DeterminismClassification.DETERMINISTIC
    seed_description: str = Field(
        default="seed is ignored because the tool is deterministic", min_length=1
    )
    cost_model: ToolCostModel = Field(default_factory=ToolCostModel)
    runtime: ToolRuntimeEstimate
    streams_progress: bool = False
    streams_candidates: bool = False
    streams_bounds: bool = False
    cooperative_cancellation: bool = True
    safe_hard_kill: bool = False
    resumable: bool = False
    resume_token_schema: dict[str, JsonValue] | None = None
    smoke_input: dict[str, JsonValue]

    @model_validator(mode="after")
    def valid_semantics(self) -> Self:
        if re.fullmatch(r"0|[1-9]\d*\.(0|[1-9]\d*)\.(0|[1-9]\d*)", self.version) is None:
            raise ValueError("tool version must be semantic MAJOR.MINOR.PATCH")
        if self.resumable != (self.resume_token_schema is not None):
            raise ValueError("resumable tools must declare exactly one resume-token schema")
        string_tags = (
            self.capability_tags
            | self.problem_tags
            | self.required_providers
            | self.required_resources
        )
        if any(re.fullmatch(r"[a-z][a-z0-9_.:-]*", tag) is None for tag in string_tags):
            raise ValueError("tool tags and requirement names must be stable lowercase identifiers")
        return self

    def model_definition(self) -> dict[str, JsonValue]:
        """Return the standard Transformers JSON-schema function declaration."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolResult(BaseModel):
    """Compact common envelope returned by every tool."""

    model_config = ConfigDict(frozen=True)

    status: ToolResultStatus
    summary: str | dict[str, JsonValue]
    artifacts: tuple[ArtifactRef, ...] = Field(default=(), max_length=32)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    candidate: Plan | None = None
    bound: ArtifactRef | None = None
    resume_token: JsonValue | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def valid_envelope(self) -> Self:
        encoded = json.dumps(
            self._model_payload(), separators=(",", ":"), ensure_ascii=False
        ).encode()
        if len(encoded) > MAX_TOOL_SUMMARY_BYTES:
            raise ValueError(
                f"tool summary exceeds {MAX_TOOL_SUMMARY_BYTES} bytes; "
                "store large output as an artifact"
            )
        if self.status is ToolResultStatus.FAILED and self.error is None:
            raise ValueError("failed tool results require a structured error")
        if self.resume_token is not None and self.status not in {
            ToolResultStatus.PARTIAL,
            ToolResultStatus.EXPIRED,
        }:
            raise ValueError("resume tokens are only valid for partial or expired results")
        return self

    def model_summary(self) -> str:
        """Serialize only bounded summaries and artifact references for model context."""

        return json.dumps(self._model_payload(), separators=(",", ":"), ensure_ascii=False)

    def _model_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "status": self.status.value,
            "summary": self.summary,
            "artifacts": [
                {
                    "id": artifact.id,
                    "kind": artifact.kind.value,
                    "media_type": artifact.media_type,
                    "byte_size": artifact.byte_size,
                }
                for artifact in self.artifacts
            ],
            "metrics": self.metrics,
        }
        if self.error is not None:
            payload["error"] = self.error.model_dump(mode="json")
        return payload


class ToolEventKind(StrEnum):
    """Kinds emitted by an optional streaming tool handler."""

    PROGRESS = "progress"
    CANDIDATE = "candidate"
    BOUND = "bound"
    RESULT = "result"


class ToolEvent(BaseModel):
    """One ordered progress, candidate, bound, or terminal result event."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=0)
    kind: ToolEventKind
    message: str = ""
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate: Plan | None = None
    bound: ArtifactRef | None = None
    result: ToolResult | None = None

    @model_validator(mode="after")
    def payload_matches_kind(self) -> Self:
        expected = {
            ToolEventKind.PROGRESS: self.progress,
            ToolEventKind.CANDIDATE: self.candidate,
            ToolEventKind.BOUND: self.bound,
            ToolEventKind.RESULT: self.result,
        }[self.kind]
        if expected is None:
            raise ValueError(f"{self.kind.value} event is missing its payload")
        return self
