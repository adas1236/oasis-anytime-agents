"""Offline checks for the hosted-model backends; no network request is made."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from oasis.errors import ModelBackendError
from oasis.llm.api_backend import (
    AnthropicModelBackend,
    OpenAICompatibleModelBackend,
    create_api_backend,
)
from oasis.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    ModelRequest,
    ToolCall,
    ToolDefinition,
)

TOOL = ToolDefinition(
    name="build_demand",
    description="Preserve typed need dimensions.",
    input_schema={"type": "object", "properties": {"artifact_id": {"type": "string"}}},
)

CONVERSATION = (
    ChatMessage(role=ChatRole.SYSTEM, content="You plan facilities."),
    ChatMessage(role=ChatRole.USER, content="Where should two centers go?"),
    ChatMessage(
        role=ChatRole.ASSISTANT,
        content="Looking at demand.",
        tool_calls=(ToolCall(id="call-1", name="build_demand", arguments={"artifact_id": "a"}),),
    ),
    ChatMessage(
        role=ChatRole.TOOL, content='{"ok":true}', tool_call_id="call-1", name="build_demand"
    ),
    ChatMessage(
        role=ChatRole.TOOL, content='{"ok":false}', tool_call_id="call-2", name="build_demand"
    ),
)


def _request(**overrides: Any) -> ModelRequest:
    fields: dict[str, Any] = {
        "request_id": "registry-1",
        "messages": CONVERSATION,
        "tools": (TOOL,),
        "max_generated_tokens": 512,
    }
    return ModelRequest(**(fields | overrides))


def _openai_backend(handler: Any) -> OpenAICompatibleModelBackend:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://api.test/v1")
    return OpenAICompatibleModelBackend(model_id="test-model", client=client)


def _openai_body(
    *, content: str | None, tool_calls: list[dict[str, Any]] | None, finish_reason: str
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 64,
            "completion_tokens_details": {"reasoning_tokens": 16},
        },
    }


async def test_openai_backend_sends_tool_history_and_parses_tool_calls() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json=_openai_body(
                content="",
                tool_calls=[
                    {
                        "id": "call-9",
                        "type": "function",
                        "function": {
                            "name": "build_demand",
                            "arguments": '{"artifact_id": "b"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
        )

    backend = _openai_backend(handler)
    turn = await backend.generate(_request())
    await backend.close()

    assert captured["url"] == "https://api.test/v1/chat/completions"
    payload = captured["payload"]
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert payload["messages"][2]["tool_calls"][0]["function"]["name"] == "build_demand"
    assert payload["messages"][3]["tool_call_id"] == "call-1"
    assert payload["tools"][0]["function"]["name"] == "build_demand"
    assert payload["max_completion_tokens"] == 512

    assert turn.finish_reason is FinishReason.TOOL_CALL
    assert turn.message.tool_calls == (
        ToolCall(id="call-9", name="build_demand", arguments={"artifact_id": "b"}),
    )
    # Budget accounting must use the provider's reported usage, never the estimate.
    assert turn.usage.input_tokens == 1200
    assert turn.usage.generated_tokens == 64
    assert turn.usage.reasoning_tokens == 16


async def test_openai_backend_tolerates_unparseable_tool_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_body(
                content=None,
                tool_calls=[
                    {"id": "c", "function": {"name": "build_demand", "arguments": "{not json"}}
                ],
                finish_reason="tool_calls",
            ),
        )

    backend = _openai_backend(handler)
    turn = await backend.generate(_request())
    await backend.close()
    assert turn.message.tool_calls[0].arguments == {}


async def test_openai_backend_raises_a_typed_error_for_a_failed_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    backend = _openai_backend(handler)
    with pytest.raises(ModelBackendError) as error:
        await backend.generate(_request())
    await backend.close()
    assert error.value.detail.context["status_code"] == 429


async def test_openai_estimate_grows_with_the_conversation() -> None:
    backend = _openai_backend(lambda request: httpx.Response(200, json={}))
    short = await backend.count_input_tokens(
        _request(messages=(ChatMessage(role=ChatRole.USER, content="hi"),), tools=())
    )
    long = await backend.count_input_tokens(_request())
    await backend.close()
    assert 0 < short < long


def test_anthropic_payload_merges_parallel_tool_results_and_caches_the_prefix() -> None:
    backend = AnthropicModelBackend(model_id="claude-opus-5", api_key="test-key")
    payload = backend._payload(_request(), cache=True)

    assert payload["system"] == "You plan facilities."
    assert payload["model"] == "claude-opus-5"
    roles = [message["role"] for message in payload["messages"]]
    # Both tool results belong to one user turn, or the API rejects the request.
    assert roles == ["user", "assistant", "user"]
    results = payload["messages"][2]["content"]
    assert [block["tool_use_id"] for block in results] == ["call-1", "call-2"]
    assert payload["messages"][1]["content"][1] == {
        "type": "tool_use",
        "id": "call-1",
        "name": "build_demand",
        "input": {"artifact_id": "a"},
    }
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["cache_control"] == {"type": "ephemeral"}


def test_anthropic_payload_replays_stored_thinking_blocks_verbatim() -> None:
    backend = AnthropicModelBackend(model_id="claude-opus-5", api_key="test-key")
    stored = [
        {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
        {"type": "tool_use", "id": "call-1", "name": "build_demand", "input": {"artifact_id": "a"}},
    ]
    backend._assistant_blocks["call-1"] = stored

    payload = backend._payload(_request())
    assert payload["messages"][1]["content"] == stored


def test_anthropic_payload_omits_cache_control_when_caching_is_disabled() -> None:
    backend = AnthropicModelBackend(model_id="claude-opus-5", api_key="test-key")
    payload = backend._payload(_request(), cache=False)
    assert "cache_control" not in payload
    assert "cache_control" not in payload["tools"][-1]


def test_create_api_backend_selects_the_requested_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    anthropic_backend = create_api_backend(provider="anthropic", model_id="claude-opus-5")
    assert isinstance(anthropic_backend, AnthropicModelBackend)
    assert anthropic_backend.profile.model_id == "claude-opus-5"
    assert anthropic_backend.capabilities.native_tools

    openai_backend = create_api_backend(
        provider="openai",
        model_id="some/model",
        base_url="https://openrouter.ai/api/v1",
    )
    assert isinstance(openai_backend, OpenAICompatibleModelBackend)
    assert openai_backend.profile.model_id == "some/model"


def test_create_api_backend_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValueError):
        create_api_backend(provider="not-a-provider")
