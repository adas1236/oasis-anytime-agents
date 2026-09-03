"""Protocols separating model execution from conversation formatting."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from oasis.llm.adapters import StreamParser
from oasis.llm.schemas import (
    ChatMessage,
    ChatRole,
    ModelCapabilities,
    ModelDelta,
    ModelProfile,
    ModelRequest,
    ModelTurn,
    ToolCall,
    ToolDefinition,
)


@runtime_checkable
class ConversationAdapter(Protocol):
    """Converts portable chat messages into processor-ready model input."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    def prepare_inputs(
        self,
        processor: Any,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDefinition],
        thinking_enabled: bool,
    ) -> Mapping[str, Any]: ...

    def stream_parser(
        self, *, thinking_enabled: bool, tools_enabled: bool = False
    ) -> StreamParser: ...

    def preserve_special_tokens(
        self, *, thinking_enabled: bool, tools_enabled: bool = False
    ) -> bool: ...


@runtime_checkable
class ModelBackend(Protocol):
    """Token-counting, cancellable, streaming model backend."""

    @property
    def profile(self) -> ModelProfile: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]: ...

    async def generate(self, request: ModelRequest) -> ModelTurn: ...

    async def abort(self, request_id: str) -> None: ...

    async def close(self) -> None: ...


async def collect_turn(backend: ModelBackend, request: ModelRequest) -> ModelTurn:
    """Collect any conforming backend stream into one assistant turn."""

    chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    usage = None
    finish_reason = None
    async for delta in backend.stream(request):
        chunks.append(delta.text)
        tool_calls.extend(delta.tool_calls)
        if delta.usage is not None:
            usage = delta.usage
        if delta.finish_reason is not None:
            finish_reason = delta.finish_reason
    if usage is None or finish_reason is None:
        raise RuntimeError("model stream ended without terminal usage and finish reason")
    return ModelTurn(
        message=ChatMessage(
            role=ChatRole.ASSISTANT,
            content="".join(chunks),
            tool_calls=tuple(tool_calls),
        ),
        usage=usage,
        finish_reason=finish_reason,
    )
