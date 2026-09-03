from __future__ import annotations

import pytest

from oasis.llm.fake import FakeModelBackend
from oasis.llm.schemas import ChatMessage, FinishReason, ModelRequest, ToolCall, ToolDefinition


def request(request_id: str = "request-1") -> ModelRequest:
    return ModelRequest(
        request_id=request_id,
        messages=(ChatMessage(role="user", content="one two"),),
        max_generated_tokens=20,
    )


@pytest.mark.asyncio
async def test_fake_backend_streams_deterministic_deltas() -> None:
    backend = FakeModelBackend(["abcdefgh"], chunk_size=3)

    deltas = [delta async for delta in backend.stream(request())]

    assert [delta.text for delta in deltas[:-1]] == ["abc", "def", "gh"]
    assert deltas[-1].finish_reason is FinishReason.STOP
    assert deltas[-1].usage is not None
    assert deltas[-1].usage.input_tokens == 2
    assert deltas[-1].usage.generated_tokens == 1


@pytest.mark.asyncio
async def test_fake_generate_collects_a_turn() -> None:
    backend = FakeModelBackend(["scripted response"], chunk_size=2)

    turn = await backend.generate(request())

    assert turn.message.content == "scripted response"
    assert turn.message.role.value == "assistant"
    assert turn.usage.total_tokens == 4


@pytest.mark.asyncio
async def test_fake_backend_abort_is_cooperative() -> None:
    backend = FakeModelBackend(["a response long enough to cancel"], chunk_size=2)
    stream = backend.stream(request("cancel-me"))

    first = await anext(stream)
    await backend.abort("cancel-me")
    remaining = [delta async for delta in stream]

    assert first.text == "a "
    assert remaining[-1].finish_reason is FinishReason.CANCELLED


@pytest.mark.asyncio
async def test_fake_backend_honors_generated_token_limit() -> None:
    backend = FakeModelBackend(["one two three"], chunk_size=20)
    limited_request = request().model_copy(update={"max_generated_tokens": 2})

    turn = await backend.generate(limited_request)

    assert turn.message.content == "one two"
    assert turn.usage.generated_tokens == 2
    assert turn.finish_reason is FinishReason.LENGTH


@pytest.mark.asyncio
async def test_fake_backend_emits_scripted_tool_call() -> None:
    tool = ToolDefinition(name="x", description="x", input_schema={"type": "object"})
    call = ToolCall(id="call-x", name="x", arguments={})
    backend = FakeModelBackend([call])
    tool_request = request().model_copy(update={"tools": (tool,)})

    turn = await backend.generate(tool_request)

    assert turn.message.tool_calls == (call,)
    assert turn.finish_reason is FinishReason.TOOL_CALL
    assert backend.capabilities.native_tools
