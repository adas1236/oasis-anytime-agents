"""Deterministic streaming backend used by tests and offline examples."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence

from oasis.llm.adapters import parse_tagged_tool_calls
from oasis.llm.profiles import resolve_model_profile
from oasis.llm.protocols import collect_turn
from oasis.llm.schemas import (
    ChatRole,
    FinishReason,
    ModelCapabilities,
    ModelDelta,
    ModelProfile,
    ModelRequest,
    ModelTurn,
    TokenUsage,
    ToolCall,
)
from oasis.runtimes.inventory import fake_inventory
from oasis.runtimes.schemas import (
    ComputeInventory,
    HardwareValidationStatus,
    RuntimeKind,
    RuntimePlan,
)


def _count_tokens(text: str) -> int:
    """Stable fake token counter; it deliberately does not imitate a real tokenizer."""

    return len(re.findall(r"\S+", text))


def _truncate_to_tokens(text: str, limit: int) -> tuple[str, bool]:
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= limit:
        return text, False
    cutoff = matches[limit - 1].end() if limit else 0
    return text[:cutoff], True


class FakeModelBackend:
    """Scripted backend whose chunks, token counts, and cancellation are reproducible."""

    def __init__(
        self,
        responses: Sequence[str | ToolCall] = (),
        *,
        chunk_size: int = 5,
        profile: ModelProfile | None = None,
        inventory: ComputeInventory | None = None,
        runtime_plan: RuntimePlan | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._responses = tuple(responses)
        self._chunk_size = chunk_size
        self._profile = profile or resolve_model_profile()
        self._inventory = inventory or fake_inventory()
        self._runtime_plan = runtime_plan or RuntimePlan(
            requested_profile=self._profile.name,
            requested_model_id=self._profile.model_id,
            runtime=RuntimeKind.FAKE,
            device_placement=("cpu",),
            dtype="fake",
            attention_backend="fake",
            rationale=("Deterministic fake runtime; no model weights or accelerators are used.",),
            hardware_validation=HardwareValidationStatus.NOT_APPLICABLE,
        )
        self._response_index = 0
        self._abort_events: dict[str, asyncio.Event] = {}
        self._closed = False
        self._capabilities = ModelCapabilities(
            generative=True,
            chat_template=True,
            native_tools=True,
            structured_fallback=True,
            reasoning_channels=True,
            streaming_abort=True,
            context_limit=self._profile.context_limit,
        )

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def runtime_plan(self) -> RuntimePlan:
        return self._runtime_plan

    @property
    def compute_inventory(self) -> ComputeInventory:
        return self._inventory

    async def load(self) -> None:
        """Satisfy the shared lifecycle hook without performing work."""

    async def count_input_tokens(self, request: ModelRequest) -> int:
        """Return the exact deterministic count that will appear in terminal usage."""

        count = sum(_count_tokens(message.model_dump_json()) for message in request.messages)
        return count + sum(_count_tokens(tool.model_dump_json()) for tool in request.tools)

    def _next_response(self, request: ModelRequest) -> str | ToolCall:
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
        else:
            last_user = next(
                (
                    message.content
                    for message in reversed(request.messages)
                    if message.role is ChatRole.USER
                ),
                "",
            )
            response = f"[fake] {last_user}"
        self._response_index += 1
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        if self._closed:
            raise RuntimeError("fake backend is closed")
        response = self._next_response(request)
        tool_calls: tuple[ToolCall, ...] = ()
        if isinstance(response, ToolCall):
            if not request.tools:
                raise ValueError("scripted fake tool call requires request tool definitions")
            tool_calls = (response,)
            response_text = ""
        else:
            response_text = response
            if request.tools and "<tool_call>" in response_text:
                response_text, tool_calls = parse_tagged_tool_calls(
                    response_text, model_id=self.profile.model_id
                )
        abort_event = asyncio.Event()
        self._abort_events[request.request_id] = abort_event
        emitted = ""
        requested_thought = "scripted-reasoning" if request.thinking_enabled else ""
        thought, thought_truncated = _truncate_to_tokens(
            requested_thought, request.max_generated_tokens
        )
        response_budget = request.max_generated_tokens - _count_tokens(thought)
        response_text, response_truncated = _truncate_to_tokens(response_text, response_budget)
        if thought:
            yield ModelDelta(thought=thought)

        try:
            for start in range(0, len(response_text), self._chunk_size):
                if abort_event.is_set():
                    break
                chunk = response_text[start : start + self._chunk_size]
                emitted += chunk
                yield ModelDelta(text=chunk)
                await asyncio.sleep(0)
            if tool_calls and not abort_event.is_set():
                yield ModelDelta(tool_calls=tool_calls)
            cancelled = abort_event.is_set()
            input_tokens = await self.count_input_tokens(request)
            generated_tokens = _count_tokens(emitted) + _count_tokens(thought)
            if tool_calls:
                generated_tokens += _count_tokens(
                    json.dumps([call.model_dump(mode="json") for call in tool_calls])
                )
            finish_reason = FinishReason.STOP
            if cancelled:
                finish_reason = FinishReason.CANCELLED
            elif thought_truncated or response_truncated:
                finish_reason = FinishReason.LENGTH
            elif tool_calls:
                finish_reason = FinishReason.TOOL_CALL
            yield ModelDelta(
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    reasoning_tokens=_count_tokens(thought),
                ),
                finish_reason=finish_reason,
            )
        finally:
            self._abort_events.pop(request.request_id, None)

    async def generate(self, request: ModelRequest) -> ModelTurn:
        return await collect_turn(self, request)

    async def abort(self, request_id: str) -> None:
        event = self._abort_events.get(request_id)
        if event is not None:
            event.set()

    async def close(self) -> None:
        self._closed = True
        for event in self._abort_events.values():
            event.set()
