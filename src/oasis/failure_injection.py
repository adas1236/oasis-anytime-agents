"""Deterministic, opt-in failures for hardening controller and provider workflows."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from oasis.controller import ControllerEvent, EventKind
from oasis.errors import ModelBackendError, ModelErrorCode, ModelErrorDetail
from oasis.llm import FakeModelBackend
from oasis.llm.schemas import FinishReason, ModelDelta, ModelRequest, TokenUsage
from oasis.providers import (
    ProviderError,
    ProviderErrorCode,
    ProviderRequestContext,
    RetrievedSource,
    SourceSnapshotRequest,
)
from oasis.schemas import (
    DeterminismClassification,
    SideEffectClassification,
    ToolEvent,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools import CancellationToken, ToolContext


class FailureMode(StrEnum):
    """Named release scenarios that never activate unless explicitly constructed."""

    MALFORMED_MODEL_CALL = "malformed_model_call"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_OOM = "model_oom"
    MODEL_ERROR = "model_error"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_ERROR = "tool_error"
    PROVIDER_OUTAGE = "provider_outage"
    USER_CANCELLATION = "user_cancellation"


class FailureInjection(BaseModel):
    """Serializable identity and safe public message for one deterministic failure."""

    model_config = ConfigDict(frozen=True)

    mode: FailureMode
    message: str = "injected release-hardening failure"


class FailureInjectingModelBackend(FakeModelBackend):
    """Fake backend which injects malformed output, unavailability, OOM, or failure."""

    def __init__(self, injection: FailureInjection) -> None:
        if injection.mode not in {
            FailureMode.MALFORMED_MODEL_CALL,
            FailureMode.MODEL_UNAVAILABLE,
            FailureMode.MODEL_OOM,
            FailureMode.MODEL_ERROR,
        }:
            raise ValueError("model failure injector requires a model failure mode")
        super().__init__()
        self.injection = injection

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        if self.injection.mode is FailureMode.MALFORMED_MODEL_CALL:
            malformed = '{"type":"call_tool","tool":'
            yield ModelDelta(text=malformed)
            yield ModelDelta(
                usage=TokenUsage(
                    input_tokens=await self.count_input_tokens(request),
                    generated_tokens=1,
                ),
                finish_reason=FinishReason.STOP,
            )
            return
        detail = ModelErrorDetail(
            code=(
                ModelErrorCode.MODEL_LOAD_FAILED
                if self.injection.mode is FailureMode.MODEL_OOM
                else (
                    ModelErrorCode.MODEL_UNAVAILABLE
                    if self.injection.mode is FailureMode.MODEL_UNAVAILABLE
                    else ModelErrorCode.GENERATION_FAILED
                )
            ),
            message=(
                "injected model out-of-memory condition"
                if self.injection.mode is FailureMode.MODEL_OOM
                else (
                    "injected model unavailable condition"
                    if self.injection.mode is FailureMode.MODEL_UNAVAILABLE
                    else self.injection.message
                )
            ),
            model_id=self.profile.model_id,
            context={"injected": True},
        )
        raise ModelBackendError(detail)


class FailureInjectingImproveTool:
    """Controller-compatible streamed search tool that times out or fails safely."""

    spec = ToolSpec(
        name="improve",
        version="1.0.0",
        description="Inject a declared timeout or failure for release hardening.",
        input_schema={"type": "object", "additionalProperties": True},
        output_schema={"type": "object"},
        capability_tags=frozenset({"decision", "search", "failure_injection"}),
        problem_tags=frozenset({"location_allocation", "routing"}),
        side_effects=SideEffectClassification.NONE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=1, p95_ms=25, time_to_first_candidate_ms=1),
        streams_candidates=True,
        smoke_input={},
    )

    def __init__(self, injection: FailureInjection) -> None:
        if injection.mode not in {FailureMode.TOOL_TIMEOUT, FailureMode.TOOL_ERROR}:
            raise ValueError("tool failure injector requires a tool failure mode")
        self.injection = injection

    async def stream(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> AsyncIterator[ToolEvent]:
        del arguments
        if self.injection.mode is FailureMode.TOOL_ERROR:
            raise RuntimeError(self.injection.message)
        if self.injection.mode is FailureMode.TOOL_TIMEOUT:
            await asyncio.Event().wait()
            context.cancellation.raise_if_cancelled()
        else:
            yield ToolEvent.model_construct()


class ScriptedSourceProvider:
    """Frozen source sequence used to reproduce success, staleness, and outage paths."""

    def __init__(self, outcomes: Sequence[RetrievedSource | ProviderError]) -> None:
        if not outcomes:
            raise ValueError("scripted source provider requires at least one outcome")
        self._outcomes = tuple(outcomes)
        self.calls = 0

    async def fetch(
        self,
        request: SourceSnapshotRequest,
        context: ProviderRequestContext,
    ) -> RetrievedSource:
        del request
        context.cancellation.raise_if_cancelled()
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


def provider_outage(message: str = "injected provider outage") -> ProviderError:
    """Create the typed retryable outage used by the release failure matrix."""

    return ProviderError(ProviderErrorCode.UNAVAILABLE, message, retryable=True)


def cancel_on_tool_start(
    cancellation: CancellationToken,
    injection: FailureInjection | None = None,
) -> Callable[[ControllerEvent], None]:
    """Create an event callback that reproducibly injects mid-tool user cancellation."""

    resolved = injection or FailureInjection(
        mode=FailureMode.USER_CANCELLATION,
        message="injected mid-tool user cancellation",
    )
    if resolved.mode is not FailureMode.USER_CANCELLATION:
        raise ValueError("cancellation injector requires the user cancellation failure mode")

    def callback(event: ControllerEvent) -> None:
        if event.kind is EventKind.TOOL_STARTED:
            cancellation.cancel(resolved.message)

    return callback


__all__ = [
    "FailureInjectingImproveTool",
    "FailureInjectingModelBackend",
    "FailureInjection",
    "FailureMode",
    "ScriptedSourceProvider",
    "cancel_on_tool_start",
    "provider_outage",
]
