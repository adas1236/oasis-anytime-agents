from __future__ import annotations

from typing import Any

import pytest

from oasis.errors import (
    ModelBackendError,
    ModelErrorCode,
    ToolCallParseError,
    UnsupportedCapabilityError,
)
from oasis.llm.adapters import (
    Gemma4ChatAdapter,
    PlainChatAdapter,
    parse_gemma_tool_calls,
    parse_tagged_tool_calls,
)
from oasis.llm.schemas import ChatMessage, ToolCall, ToolDefinition


class StubProcessor:
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, str]:
        self.messages = messages
        self.kwargs = kwargs
        return {"input_ids": "tokens"}


def test_gemma_adapter_uses_official_thinking_template_switch() -> None:
    processor = StubProcessor()
    adapter = Gemma4ChatAdapter("google/gemma-4-E4B-it", 131_072)

    result = adapter.prepare_inputs(
        processor,
        [ChatMessage(role="user", content="hello")],
        tools=[],
        thinking_enabled=True,
    )

    assert result == {"input_ids": "tokens"}
    assert processor.kwargs["enable_thinking"] is True
    assert processor.kwargs["add_generation_prompt"] is True


def test_plain_adapter_rejects_reasoning_as_typed_capability_error() -> None:
    adapter = PlainChatAdapter("organization/plain-model")

    with pytest.raises(UnsupportedCapabilityError) as caught:
        adapter.prepare_inputs(
            StubProcessor(),
            [ChatMessage(role="user", content="hello")],
            tools=[],
            thinking_enabled=True,
        )

    assert caught.value.detail.code is ModelErrorCode.UNSUPPORTED_CAPABILITY
    assert caught.value.detail.capability == "reasoning channels"


def test_gemma_adapter_passes_native_tool_schema_to_template() -> None:
    processor = StubProcessor()
    adapter = Gemma4ChatAdapter("google/gemma-4-E4B-it")
    tool = ToolDefinition(
        name="later",
        description="Implemented in phase 2.",
        input_schema={"type": "object"},
    )

    adapter.prepare_inputs(
        processor,
        [ChatMessage(role="user", content="hello")],
        tools=[tool],
        thinking_enabled=False,
    )

    assert processor.kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "later",
                "description": "Implemented in phase 2.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert adapter.capabilities.native_tools


def test_gemma_adapter_merges_tool_response_into_official_assistant_turn() -> None:
    processor = StubProcessor()
    adapter = Gemma4ChatAdapter("google/gemma-4-E4B-it")
    call = ToolCall(id="call-1", name="calculator", arguments={"operands": [2, 3]})

    adapter.prepare_inputs(
        processor,
        [
            ChatMessage(role="user", content="add"),
            ChatMessage(role="assistant", tool_calls=(call,)),
            ChatMessage(
                role="tool",
                name="calculator",
                tool_call_id="call-1",
                content='{"value":5}',
            ),
        ],
        tools=[
            ToolDefinition(
                name="calculator",
                description="Calculate.",
                input_schema={"type": "object"},
            )
        ],
        thinking_enabled=False,
    )

    assistant = processor.messages[-1]
    assert assistant["tool_calls"] == [
        {"function": {"name": "calculator", "arguments": {"operands": [2, 3]}}}
    ]
    assert assistant["tool_responses"] == [{"name": "calculator", "response": {"value": 5}}]


def test_missing_chat_template_is_typed() -> None:
    processor = StubProcessor()
    processor.chat_template = None

    with pytest.raises(ModelBackendError) as caught:
        PlainChatAdapter("organization/no-template").prepare_inputs(
            processor,
            [ChatMessage(role="user", content="hello")],
            tools=[],
            thinking_enabled=False,
        )

    assert caught.value.detail.code is ModelErrorCode.MISSING_CHAT_TEMPLATE


def test_gemma_thought_parser_separates_split_markers() -> None:
    parser = Gemma4ChatAdapter("google/gemma-4-E4B-it").stream_parser(thinking_enabled=True)
    parts = ["<|chan", "nel>thought\nprivate ", "work<chan", "nel|>Public answer<turn|>"]
    answer = ""
    thought = ""
    for part in parts:
        public, private = parser.feed(part)
        answer += public
        thought += private
    public, private = parser.finish()

    assert answer + public == "Public answer"
    assert thought + private == "private work"


def test_gemma_native_call_parser_handles_nested_values_and_string_delimiter() -> None:
    public, calls = parse_gemma_tool_calls(
        '<|tool_call>call:calculator{operation:<|"|>add<|"|>,'
        'operands:[2,3.5],options:{label:<|"|>a,{b}<|"|>}}<tool_call|>'
        "<|tool_response>",
        model_id="google/gemma-4-E4B-it",
    )

    assert public == ""
    assert calls == (
        ToolCall(
            id="call-1",
            name="calculator",
            arguments={
                "operation": "add",
                "operands": [2, 3.5],
                "options": {"label": "a,{b}"},
            },
        ),
    )


def test_gemma_native_call_parser_rejects_malformed_output() -> None:
    with pytest.raises(ToolCallParseError):
        parse_gemma_tool_calls(
            '<|tool_call>call:calculator{operation:<|"|>add<|"|>} ',
            model_id="google/gemma-4-E4B-it",
        )


def test_generic_fallback_formats_instruction_and_parses_tagged_json() -> None:
    processor = StubProcessor()
    adapter = PlainChatAdapter("organization/chat-model")
    tool = ToolDefinition(
        name="calculator",
        description="Calculate.",
        input_schema={"type": "object", "properties": {}},
    )

    adapter.prepare_inputs(
        processor,
        [ChatMessage(role="user", content="two plus three")],
        tools=[tool],
        thinking_enabled=False,
    )
    public, calls = parse_tagged_tool_calls(
        '<tool_call>{"name":"calculator","arguments":{"operation":"add",'
        '"operands":[2,3]}}</tool_call>',
        model_id="organization/chat-model",
    )

    assert "Tools are available" in processor.messages[0]["content"]
    assert public == ""
    assert calls[0].arguments["operands"] == [2, 3]
    assert adapter.capabilities.structured_fallback


def test_generic_fallback_combines_existing_system_instruction() -> None:
    processor = StubProcessor()
    tool = ToolDefinition(
        name="calculator", description="Calculate.", input_schema={"type": "object"}
    )

    PlainChatAdapter("organization/chat-model").prepare_inputs(
        processor,
        [
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="hello"),
        ],
        tools=[tool],
        thinking_enabled=False,
    )

    assert [message["role"] for message in processor.messages].count("system") == 1
    assert "Be concise." in processor.messages[0]["content"]


def test_generic_fallback_rejects_invalid_tagged_json() -> None:
    with pytest.raises(ToolCallParseError):
        parse_tagged_tool_calls(
            "<tool_call>{not-json}</tool_call>", model_id="organization/chat-model"
        )
