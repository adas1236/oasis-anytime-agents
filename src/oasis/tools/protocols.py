"""Execution protocols and cooperative cancellation for tools."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from oasis.artifacts.protocols import ArtifactStore
from oasis.schemas.artifacts import PrivacyClassification
from oasis.schemas.tools import ToolError, ToolEvent, ToolResult, ToolSpec


class ToolLogger(Protocol):
    """Small logger surface that avoids binding tools to a logging framework."""

    def info(self, message: str, **context: object) -> None: ...

    def warning(self, message: str, **context: object) -> None: ...


class NullToolLogger:
    """Default logger for tests and embeddings that do not need emitted logs."""

    def info(self, message: str, **context: object) -> None:
        del message, context

    def warning(self, message: str, **context: object) -> None:
        del message, context


class CancellationToken:
    """One-way cooperative cancellation signal shared with a tool invocation."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    async def wait(self) -> str:
        await self._event.wait()
        return self._reason or "cancelled"

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ToolCancelledError(self._reason or "cancelled")


class ToolCancelledError(asyncio.CancelledError):
    """Raised by cooperative tools after observing a cancellation token."""


class ToolExecutionError(RuntimeError):
    """Expected tool failure carrying a safe structured error."""

    def __init__(self, detail: ToolError) -> None:
        super().__init__(detail.message)
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Per-invocation dependencies and limits supplied by the controller."""

    run_id: str
    artifact_store: ArtifactStore
    deadline_monotonic: float
    cancellation: CancellationToken
    seed: int
    logger: ToolLogger = field(default_factory=NullToolLogger)
    providers: Mapping[str, object] = field(default_factory=dict)
    resources: Mapping[str, object] = field(default_factory=dict)
    allowed_privacy: frozenset[PrivacyClassification] = frozenset({PrivacyClassification.PUBLIC})
    monotonic: Callable[[], float] = time.monotonic

    @property
    def remaining_seconds(self) -> float:
        """Non-negative time remaining at the injectable monotonic clock."""

        return max(0.0, self.deadline_monotonic - self.monotonic())


@runtime_checkable
class Tool(Protocol):
    """Typed asynchronous tool handler."""

    @property
    def spec(self) -> ToolSpec: ...

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult: ...


@runtime_checkable
class StreamingTool(Protocol):
    """Optional event stream for progress, candidates, bounds, and final results."""

    @property
    def spec(self) -> ToolSpec: ...

    def stream(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> AsyncIterator[ToolEvent]: ...
