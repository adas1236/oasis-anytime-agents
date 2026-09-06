"""Hosted-model backends so real-model evaluation does not require a local accelerator.

Two providers are supported behind the same ``ModelBackend`` protocol:

``anthropic``
    The official Anthropic SDK against the Messages API. Input tokens are counted
    exactly through ``/v1/messages/count_tokens`` rather than estimated.

``openai``
    Any OpenAI-compatible ``/chat/completions`` endpoint, which also covers
    OpenRouter through ``--api-base-url``. That protocol has no token-counting
    endpoint, so input tokens are estimated before the call and the provider's own
    reported usage is what the budget ledger records afterwards.

Both providers use native tool calling, so no tag parsing or repair is involved:
the evaluation's compact tool schemas are sent as provider tool definitions and
tool calls come back structured.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterable, Sequence
from enum import StrEnum
from typing import Any

import httpx

from oasis.errors import ModelBackendError, ModelErrorCode, ModelErrorDetail
from oasis.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    ModelCapabilities,
    ModelDelta,
    ModelProfile,
    ModelRequest,
    ModelTurn,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Only used before a call, to size the per-turn generation allowance on providers
# without a counting endpoint. Recorded usage always comes from the provider.
_ESTIMATED_CHARS_PER_TOKEN = 4
_ESTIMATED_MESSAGE_OVERHEAD_TOKENS = 4


class ApiProvider(StrEnum):
    """Hosted providers reachable without local model weights."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


def _api_error(message: str, code: ModelErrorCode, **context: Any) -> ModelBackendError:
    return ModelBackendError(ModelErrorDetail(code=code, message=message, context=context))


def _estimate_tokens(text: str) -> int:
    return (len(text) + _ESTIMATED_CHARS_PER_TOKEN - 1) // _ESTIMATED_CHARS_PER_TOKEN


def _merge_tool_messages(messages: Sequence[ChatMessage]) -> list[list[ChatMessage]]:
    """Group runs of consecutive tool results so parallel calls stay in one turn."""

    groups: list[list[ChatMessage]] = []
    for message in messages:
        if message.role is ChatRole.TOOL and groups and groups[-1][0].role is ChatRole.TOOL:
            groups[-1].append(message)
        else:
            groups.append([message])
    return groups


def _tool_result_text(message: ChatMessage) -> str:
    # Tool results are already canonical JSON strings from the evaluation harness;
    # an empty result would be rejected by both providers.
    return message.content or "{}"


class _HostedBackendBase:
    """Shared profile/capability surface and abort bookkeeping."""

    def __init__(self, profile: ModelProfile, capabilities: ModelCapabilities) -> None:
        self._profile = profile
        self._capabilities = capabilities
        self._abort_events: dict[str, asyncio.Event] = {}
        self._closed = False

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def load(self) -> None:
        """Hosted models need no local load step; kept for protocol parity."""

    async def abort(self, request_id: str) -> None:
        event = self._abort_events.get(request_id)
        if event is not None:
            event.set()

    def _register(self, request_id: str) -> asyncio.Event:
        event = asyncio.Event()
        self._abort_events[request_id] = event
        return event

    def _release(self, request_id: str) -> None:
        self._abort_events.pop(request_id, None)


class AnthropicModelBackend(_HostedBackendBase):
    """Anthropic Messages API backend with exact token counting and native tools."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        context_limit: int | None = None,
        effort: str | None = None,
        thinking: bool = True,
        cache_prompt: bool = True,
        max_retries: int = 2,
        timeout_seconds: float | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise _api_error(
                "the anthropic provider requires the 'api' dependency group (uv sync --group api)",
                ModelErrorCode.MODEL_UNAVAILABLE,
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        options: dict[str, Any] = {"max_retries": max_retries}
        if resolved_key:
            options["api_key"] = resolved_key
        if base_url:
            options["base_url"] = base_url
        if timeout_seconds is not None:
            options["timeout"] = timeout_seconds
        # A bare client still resolves an `ant auth login` profile, so an unset
        # ANTHROPIC_API_KEY is not by itself a missing credential.
        self._client = AsyncAnthropic(**options)
        self._effort = effort
        self._thinking = thinking
        self._cache_prompt = cache_prompt
        # Thinking blocks must be replayed unchanged on the same model, but the
        # portable ChatMessage has nowhere to carry them. Keep the provider's own
        # assistant blocks, keyed by the tool-call IDs the harness does preserve.
        self._assistant_blocks: dict[str, list[dict[str, Any]]] = {}
        super().__init__(
            profile=ModelProfile(
                name=f"anthropic:{model_id}",
                model_id=model_id,
                family="anthropic",
                context_limit=context_limit,
                supports_thinking=thinking,
                supports_native_tools=True,
                is_custom=True,
            ),
            capabilities=ModelCapabilities(
                generative=True,
                chat_template=True,
                native_tools=True,
                structured_fallback=False,
                reasoning_channels=thinking,
                streaming_abort=True,
                context_limit=context_limit,
            ),
        )

    def _tools(
        self, tools: Sequence[ToolDefinition], *, cache: bool = False
    ) -> list[dict[str, Any]]:
        rendered = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
        if cache and rendered:
            # Tools render before system and messages, so one breakpoint on the last
            # tool caches the whole static prefix that is otherwise re-sent every turn.
            rendered[-1]["cache_control"] = {"type": "ephemeral"}
        return rendered

    def _assistant_content(self, message: ChatMessage) -> list[dict[str, Any]]:
        for call in message.tool_calls:
            cached = self._assistant_blocks.get(call.id)
            if cached is not None:
                return cached
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        blocks.extend(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": dict(call.arguments),
            }
            for call in message.tool_calls
        )
        return blocks or [{"type": "text", "text": "(no content)"}]

    def _payload(self, request: ModelRequest, *, cache: bool = False) -> dict[str, Any]:
        system_parts: list[str] = []
        conversation: list[dict[str, Any]] = []
        for group in _merge_tool_messages(request.messages):
            first = group[0]
            if first.role is ChatRole.SYSTEM:
                system_parts.append(first.content)
            elif first.role is ChatRole.USER:
                conversation.append({"role": "user", "content": first.content})
            elif first.role is ChatRole.ASSISTANT:
                conversation.append(
                    {"role": "assistant", "content": self._assistant_content(first)}
                )
            else:
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": _tool_result_text(message),
                            }
                            for message in group
                        ],
                    }
                )
        payload: dict[str, Any] = {
            "model": self._profile.model_id,
            "max_tokens": request.max_generated_tokens,
            "messages": conversation,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.tools:
            payload["tools"] = self._tools(request.tools, cache=cache)
        if cache:
            # Also roll a breakpoint onto the last cacheable block so the growing
            # tool-result history is cached turn over turn, not just the prefix.
            payload["cache_control"] = {"type": "ephemeral"}
        return payload

    def _generation_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self._thinking:
            options["thinking"] = {"type": "adaptive"}
        if self._effort is not None:
            options["output_config"] = {"effort": self._effort}
        return options

    async def count_input_tokens(self, request: ModelRequest) -> int:
        payload = self._payload(request)
        payload.pop("max_tokens", None)
        response = await self._client.messages.count_tokens(**payload, **self._generation_options())
        return int(response.input_tokens)

    async def generate(self, request: ModelRequest) -> ModelTurn:
        abort_event = self._register(request.request_id)
        payload = self._payload(request, cache=self._cache_prompt) | self._generation_options()
        try:
            task = asyncio.ensure_future(self._client.messages.create(**payload))
            aborted = asyncio.ensure_future(abort_event.wait())
            done, _ = await asyncio.wait({task, aborted}, return_when=asyncio.FIRST_COMPLETED)
            aborted.cancel()
            if task not in done:
                task.cancel()
                return ModelTurn(
                    message=ChatMessage(role=ChatRole.ASSISTANT, content=""),
                    usage=TokenUsage(),
                    finish_reason=FinishReason.CANCELLED,
                )
            message = task.result()
        finally:
            self._release(request.request_id)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_blocks: list[dict[str, Any]] = []
        for block in message.content:
            raw_blocks.append(block.model_dump(mode="json", exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(arguments)))
        for call in tool_calls:
            self._assistant_blocks[call.id] = raw_blocks

        usage = message.usage
        # Cached reads still occupy the context window and are billed, so they
        # belong in the aggregate input total the budget ledger enforces.
        input_tokens = (
            int(usage.input_tokens)
            + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        )
        return ModelTurn(
            message=ChatMessage(
                role=ChatRole.ASSISTANT,
                content="".join(text_parts),
                tool_calls=tuple(tool_calls),
            ),
            usage=TokenUsage(
                input_tokens=input_tokens,
                generated_tokens=int(usage.output_tokens),
                # Anthropic bills thinking inside output_tokens and does not
                # report it separately; leave the split unclaimed rather than guess.
                reasoning_tokens=0,
            ),
            finish_reason=_ANTHROPIC_FINISH.get(message.stop_reason or "", FinishReason.STOP),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        """Single-delta stream; the evaluation loop only consumes whole turns."""

        turn = await self.generate(request)
        yield ModelDelta(
            text=turn.message.content,
            tool_calls=turn.message.tool_calls,
            usage=turn.usage,
            finish_reason=turn.finish_reason,
        )

    async def close(self) -> None:
        self._closed = True
        for event in self._abort_events.values():
            event.set()
        await self._client.close()


_ANTHROPIC_FINISH = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALL,
    "pause_turn": FinishReason.STOP,
    "refusal": FinishReason.ERROR,
}

_OPENAI_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALL,
    "function_call": FinishReason.TOOL_CALL,
    "content_filter": FinishReason.ERROR,
}


class OpenAICompatibleModelBackend(_HostedBackendBase):
    """Chat-completions backend covering OpenAI and OpenRouter deployments."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        context_limit: int | None = None,
        timeout_seconds: float = 600.0,
        client: httpx.AsyncClient | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        resolved_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        )
        if resolved_key is None and client is None:
            raise _api_error(
                "the openai provider requires OPENAI_API_KEY or OPENROUTER_API_KEY",
                ModelErrorCode.MODEL_UNAVAILABLE,
            )
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if resolved_key is not None:
            headers["Authorization"] = f"Bearer {resolved_key}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
        )
        super().__init__(
            profile=ModelProfile(
                name=f"openai:{model_id}",
                model_id=model_id,
                family="openai",
                context_limit=context_limit,
                supports_native_tools=True,
                is_custom=True,
            ),
            capabilities=ModelCapabilities(
                generative=True,
                chat_template=True,
                native_tools=True,
                structured_fallback=False,
                reasoning_channels=False,
                streaming_abort=True,
                context_limit=context_limit,
            ),
        )

    def _messages(self, messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            if message.role is ChatRole.TOOL:
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": _tool_result_text(message),
                    }
                )
            elif message.role is ChatRole.ASSISTANT:
                entry: dict[str, Any] = {"role": "assistant", "content": message.content or None}
                if message.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, sort_keys=True),
                            },
                        }
                        for call in message.tool_calls
                    ]
                payload.append(entry)
            else:
                payload.append({"role": message.role.value, "content": message.content})
        return payload

    async def count_input_tokens(self, request: ModelRequest) -> int:
        """Estimate only; the chat-completions protocol has no counting endpoint."""

        total = 0
        for message in self._messages(request.messages):
            total += _ESTIMATED_MESSAGE_OVERHEAD_TOKENS
            total += _estimate_tokens(json.dumps(message, sort_keys=True))
        for tool in request.tools:
            total += _estimate_tokens(
                tool.name + tool.description + json.dumps(tool.input_schema, sort_keys=True)
            )
        return total

    async def generate(self, request: ModelRequest) -> ModelTurn:
        abort_event = self._register(request.request_id)
        payload: dict[str, Any] = {
            "model": self._profile.model_id,
            "messages": self._messages(request.messages),
            "max_completion_tokens": request.max_generated_tokens,
        }
        if request.tools:
            payload["tools"] = [tool.transformers_schema() for tool in request.tools]
        try:
            task = asyncio.ensure_future(self._client.post("/chat/completions", json=payload))
            aborted = asyncio.ensure_future(abort_event.wait())
            done, _ = await asyncio.wait({task, aborted}, return_when=asyncio.FIRST_COMPLETED)
            aborted.cancel()
            if task not in done:
                task.cancel()
                return ModelTurn(
                    message=ChatMessage(role=ChatRole.ASSISTANT, content=""),
                    usage=TokenUsage(),
                    finish_reason=FinishReason.CANCELLED,
                )
            response = task.result()
        finally:
            self._release(request.request_id)

        if response.status_code >= 400:
            raise _api_error(
                f"chat completions request failed with status {response.status_code}",
                ModelErrorCode.GENERATION_FAILED,
                status_code=response.status_code,
                body_excerpt=response.text[:500],
            )
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise _api_error(
                "chat completions response contained no choices",
                ModelErrorCode.GENERATION_FAILED,
                body_excerpt=json.dumps(body)[:500],
            )
        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls: list[ToolCall] = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"{request.request_id}-{index}"),
                    name=str(function.get("name") or "unknown"),
                    arguments=arguments,
                )
            )
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return ModelTurn(
            message=ChatMessage(
                role=ChatRole.ASSISTANT,
                content=message.get("content") or "",
                tool_calls=tuple(tool_calls),
            ),
            usage=TokenUsage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                generated_tokens=int(usage.get("completion_tokens") or 0),
                reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            ),
            finish_reason=_OPENAI_FINISH.get(
                choice.get("finish_reason") or "",
                FinishReason.TOOL_CALL if tool_calls else FinishReason.STOP,
            ),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        """Single-delta stream; the evaluation loop only consumes whole turns."""

        turn = await self.generate(request)
        yield ModelDelta(
            text=turn.message.content,
            tool_calls=turn.message.tool_calls,
            usage=turn.usage,
            finish_reason=turn.finish_reason,
        )

    async def close(self) -> None:
        self._closed = True
        for event in self._abort_events.values():
            event.set()
        if self._owns_client:
            await self._client.aclose()


def create_api_backend(
    *,
    provider: str,
    model_id: str | None = None,
    base_url: str | None = None,
    context_limit: int | None = None,
    effort: str | None = None,
    thinking: bool = True,
    cache_prompt: bool = True,
    timeout_seconds: float | None = None,
) -> AnthropicModelBackend | OpenAICompatibleModelBackend:
    """Build a hosted backend from evaluation-runner settings."""

    kind = ApiProvider(provider)
    if kind is ApiProvider.ANTHROPIC:
        return AnthropicModelBackend(
            model_id=model_id or DEFAULT_ANTHROPIC_MODEL,
            base_url=base_url,
            context_limit=context_limit,
            effort=effort,
            thinking=thinking,
            cache_prompt=cache_prompt,
            timeout_seconds=timeout_seconds,
        )
    return OpenAICompatibleModelBackend(
        model_id=model_id or DEFAULT_OPENAI_MODEL,
        base_url=base_url or DEFAULT_OPENAI_BASE_URL,
        context_limit=context_limit,
        timeout_seconds=timeout_seconds if timeout_seconds is not None else 600.0,
    )
