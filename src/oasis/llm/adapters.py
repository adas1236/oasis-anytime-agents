"""Conversation formatting and tool-call parsing for supported chat models."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from oasis.errors import ModelBackendError, ModelErrorCode, ModelErrorDetail, ToolCallParseError
from oasis.llm.gemma_schema import gemma_tool_schema
from oasis.llm.schemas import ChatMessage, ChatRole, ModelCapabilities, ToolCall, ToolDefinition


@dataclass(frozen=True, slots=True)
class ParsedModelChunk:
    """Adapter-normalized public text, private thought, and tool calls."""

    text: str = ""
    thought: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    def __iter__(self) -> Iterator[str]:
        """Keep two-value text/thought unpacking compatible with the Phase 1 parser API."""

        yield self.text
        yield self.thought


class StreamParser(Protocol):
    """Incrementally normalize model-family output."""

    def feed(self, text: str) -> ParsedModelChunk: ...

    def finish(self) -> ParsedModelChunk: ...


class _PassthroughParser:
    def feed(self, text: str) -> ParsedModelChunk:
        return ParsedModelChunk(text=text)

    def finish(self) -> ParsedModelChunk:
        return ParsedModelChunk()


class _GemmaThinkingParser:
    """Incremental parser for Gemma 4's documented thought channel delimiters."""

    _START = "<|channel>thought"
    _END = "<channel|>"
    _CONTROL_TOKENS = ("<turn|>", "<|turn>", "<bos>", "<eos>")

    def __init__(self, *, thought_prefilled: bool = False) -> None:
        self._buffer = ""
        self._state = "thought" if thought_prefilled else "before_thought"
        self._skip_thought_newline = False

    @staticmethod
    def _safe_prefix(buffer: str, marker: str) -> tuple[str, str]:
        for length in range(min(len(buffer), len(marker) - 1), 0, -1):
            if marker.startswith(buffer[-length:]):
                return buffer[:-length], buffer[-length:]
        return buffer, ""

    @classmethod
    def strip_control_tokens(cls, text: str) -> str:
        for token in cls._CONTROL_TOKENS:
            text = text.replace(token, "")
        return text

    def feed(self, text: str) -> ParsedModelChunk:
        self._buffer += text
        answer_parts: list[str] = []
        thought_parts: list[str] = []

        while self._buffer:
            if self._skip_thought_newline:
                self._buffer = self._buffer.lstrip("\n")
                if not self._buffer:
                    break
                self._skip_thought_newline = False
            parts = thought_parts if self._state == "thought" else answer_parts
            markers: tuple[str, ...] = self._CONTROL_TOKENS
            if self._state == "before_thought":
                markers += (self._START,)
            elif self._state == "thought":
                markers += (self._END,)
            found = [
                (offset, marker) for marker in markers if (offset := self._buffer.find(marker)) >= 0
            ]
            if found:
                offset, marker = min(found)
                parts.append(self._buffer[:offset])
                self._buffer = self._buffer[offset + len(marker) :]
                if marker == self._START:
                    self._state = "thought"
                    self._skip_thought_newline = True
                elif marker == self._END:
                    self._state = "answer"
                continue
            # Hold partial channel AND control tokens across arbitrary chunk boundaries.
            safe_length = min(len(self._safe_prefix(self._buffer, m)[0]) for m in markers)
            parts.append(self._buffer[:safe_length])
            self._buffer = self._buffer[safe_length:]
            break

        return ParsedModelChunk(text="".join(answer_parts), thought="".join(thought_parts))

    def finish(self) -> ParsedModelChunk:
        remaining = self.strip_control_tokens(self._buffer)
        self._buffer = ""
        if self._state == "thought":
            return ParsedModelChunk(thought=remaining)
        return ParsedModelChunk(text=remaining)


class _GemmaValueParser:
    """Parser for Gemma's JSON-like blocks with ``<|\"|>`` string delimiters."""

    _STRING = '<|"|>'
    _NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
    _IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")

    def __init__(self, source: str) -> None:
        self.source = source
        self.index = 0

    def _skip_space(self) -> None:
        while self.index < len(self.source) and self.source[self.index].isspace():
            self.index += 1

    def _consume(self, expected: str) -> None:
        self._skip_space()
        if not self.source.startswith(expected, self.index):
            raise ValueError(f"expected {expected!r} at offset {self.index}")
        self.index += len(expected)

    def _identifier(self) -> str:
        self._skip_space()
        match = self._IDENTIFIER.match(self.source, self.index)
        if match is None:
            raise ValueError(f"expected identifier at offset {self.index}")
        self.index = match.end()
        return match.group()

    def _string(self) -> str:
        self._consume(self._STRING)
        end = self.source.find(self._STRING, self.index)
        if end < 0:
            raise ValueError("unterminated Gemma string value")
        value = self.source[self.index : end]
        self.index = end + len(self._STRING)
        return value

    def value(self) -> Any:
        self._skip_space()
        if self.source.startswith(self._STRING, self.index):
            return self._string()
        if self.source.startswith("{", self.index):
            return self.object()
        if self.source.startswith("[", self.index):
            return self.array()
        for literal, value in (("true", True), ("false", False), ("null", None)):
            if self.source.startswith(literal, self.index):
                self.index += len(literal)
                return value
        number = self._NUMBER.match(self.source, self.index)
        if number is not None:
            token = number.group()
            self.index = number.end()
            return float(token) if any(character in token for character in ".eE") else int(token)
        raise ValueError(f"expected value at offset {self.index}")

    def object(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self._consume("{")
        self._skip_space()
        if self.source.startswith("}", self.index):
            self.index += 1
            return result
        while True:
            self._skip_space()
            key = (
                self._string()
                if self.source.startswith(self._STRING, self.index)
                else self._identifier()
            )
            self._consume(":")
            result[key] = self.value()
            self._skip_space()
            if self.source.startswith("}", self.index):
                self.index += 1
                return result
            self._consume(",")

    def array(self) -> list[Any]:
        result: list[Any] = []
        self._consume("[")
        self._skip_space()
        if self.source.startswith("]", self.index):
            self.index += 1
            return result
        while True:
            result.append(self.value())
            self._skip_space()
            if self.source.startswith("]", self.index):
                self.index += 1
                return result
            self._consume(",")

    def complete_object(self) -> dict[str, Any]:
        value = self.object()
        self._skip_space()
        if self.index != len(self.source):
            raise ValueError(f"unexpected data at offset {self.index}")
        return value


def _tool_call_from_json(payload: Any, *, sequence: int) -> ToolCall:
    if not isinstance(payload, dict):
        raise ValueError("tool-call payload must be an object")
    function = payload.get("function", payload)
    if not isinstance(function, dict):
        raise ValueError("tool-call function must be an object")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("tool call requires string name and object arguments")
    call_id = payload.get("id", f"call-{sequence}")
    if not isinstance(call_id, str):
        raise ValueError("tool call id must be a string")
    return ToolCall(id=call_id, name=name, arguments=arguments)


def parse_tagged_tool_calls(text: str, *, model_id: str) -> tuple[str, tuple[ToolCall, ...]]:
    """Parse conservative ``<tool_call>`` JSON blocks used by generic models."""

    def strip_control_tokens(value: str) -> str:
        for token in ("<|im_end|>", "<|endoftext|>", "</s>", "<eos>"):
            value = value.replace(token, "")
        return value

    start_marker = "<tool_call>"
    end_marker = "</tool_call>"
    if start_marker not in text:
        if end_marker in text:
            raise ToolCallParseError(
                model_id, "tool-call output has an unmatched end tag", output=text
            )
        return strip_control_tokens(text), ()
    calls: list[ToolCall] = []
    public: list[str] = []
    cursor = 0
    while True:
        start = text.find(start_marker, cursor)
        if start < 0:
            public.append(text[cursor:])
            break
        public.append(text[cursor:start])
        end = text.find(end_marker, start + len(start_marker))
        if end < 0:
            raise ToolCallParseError(
                model_id, "tool-call output is missing its end tag", output=text
            )
        body = text[start + len(start_marker) : end].strip()
        try:
            calls.append(_tool_call_from_json(json.loads(body), sequence=len(calls) + 1))
        except (ValueError, TypeError) as error:
            raise ToolCallParseError(
                model_id, f"invalid tagged JSON tool call: {error}", output=text
            ) from error
        cursor = end + len(end_marker)
    return strip_control_tokens("".join(public)).strip(), tuple(calls)


def parse_gemma_tool_calls(text: str, *, model_id: str) -> tuple[str, tuple[ToolCall, ...]]:
    """Parse official Gemma 4 call blocks into canonical tool calls."""

    start_marker = "<|tool_call>"
    end_marker = "<tool_call|>"
    response_marker = "<|tool_response>"
    if start_marker not in text:
        if end_marker in text:
            raise ToolCallParseError(
                model_id, "Gemma tool call has an unmatched end token", output=text
            )
        return _GemmaThinkingParser.strip_control_tokens(text), ()
    calls: list[ToolCall] = []
    public: list[str] = []
    cursor = 0
    while True:
        start = text.find(start_marker, cursor)
        if start < 0:
            public.append(text[cursor:])
            break
        public.append(text[cursor:start])
        end = text.find(end_marker, start + len(start_marker))
        if end < 0:
            raise ToolCallParseError(
                model_id, "Gemma tool call is missing its end token", output=text
            )
        body = text[start + len(start_marker) : end].strip()
        if not body.startswith("call:"):
            raise ToolCallParseError(
                model_id, "Gemma tool call must start with 'call:'", output=text
            )
        declaration = body[len("call:") :]
        brace = declaration.find("{")
        if brace <= 0:
            raise ToolCallParseError(
                model_id, "Gemma tool call has no name or arguments", output=text
            )
        name = declaration[:brace].strip()
        try:
            arguments = _GemmaValueParser(declaration[brace:]).complete_object()
            calls.append(ToolCall(id=f"call-{len(calls) + 1}", name=name, arguments=arguments))
        except (ValueError, TypeError) as error:
            raise ToolCallParseError(
                model_id, f"invalid Gemma tool-call arguments: {error}", output=text
            ) from error
        cursor = end + len(end_marker)
        if text.startswith(response_marker, cursor):
            cursor += len(response_marker)
    rendered = _GemmaThinkingParser.strip_control_tokens("".join(public)).replace(
        response_marker, ""
    )
    return rendered.strip(), tuple(calls)


class _BufferedAgenticParser:
    def __init__(self, model_id: str, *, gemma: bool, thought_prefilled: bool = False) -> None:
        self._model_id = model_id
        self._gemma = gemma
        self._thought_parser = (
            _GemmaThinkingParser(thought_prefilled=thought_prefilled) if gemma else None
        )
        self._buffer = ""

    def feed(self, text: str) -> ParsedModelChunk:
        if self._thought_parser is not None:
            parsed = self._thought_parser.feed(text)
            self._buffer += parsed.text
            # Account for reasoning even when the following tool call is malformed.
            return ParsedModelChunk(thought=parsed.thought)
        self._buffer += text
        return ParsedModelChunk()

    def finish(self) -> ParsedModelChunk:
        raw = self._buffer
        self._buffer = ""
        if self._thought_parser is not None:
            last = self._thought_parser.finish()
            answer = raw + last.text
            thought = last.thought
            public, calls = parse_gemma_tool_calls(answer, model_id=self._model_id)
        else:
            thought = ""
            public, calls = parse_tagged_tool_calls(raw, model_id=self._model_id)
        return ParsedModelChunk(text=public, thought=thought, tool_calls=calls)


def has_chat_template(processor: Any) -> bool:
    """Return whether a processor/tokenizer exposes a usable chat template."""

    if getattr(processor, "chat_template", None):
        return True
    tokenizer = getattr(processor, "tokenizer", None)
    return bool(getattr(tokenizer, "chat_template", None))


def _standard_message_dicts(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "type": "function",
                    "id": call.id,
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        if message.role is ChatRole.TOOL:
            item["tool_call_id"] = message.tool_call_id
            item["name"] = message.name
        rendered.append(item)
    return rendered


def _gemma_message_dicts(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        if message.role is ChatRole.TOOL:
            if not rendered or rendered[-1]["role"] != "assistant":
                raise ValueError("a Gemma tool response must follow its assistant tool call")
            try:
                response: Any = json.loads(message.content)
            except json.JSONDecodeError:
                response = message.content
            rendered[-1].setdefault("tool_responses", []).append(
                {"name": message.name, "response": response}
            )
            continue
        item: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            if message.content:
                # The native template renders content AFTER calls/results on the same
                # message. Separate the preamble to preserve chronological order.
                rendered.append({"role": "assistant", "content": message.content})
                item["content"] = ""
            item["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in message.tool_calls
            ]
        rendered.append(item)
    return rendered


def _fallback_message_dicts(
    messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition]
) -> list[dict[str, str]]:
    definitions = json.dumps(
        [tool.transformers_schema() for tool in tools], separators=(",", ":"), sort_keys=True
    )
    instruction = (
        "Tools are available below. To call exactly one, output only "
        '<tool_call>{"name":"tool_name","arguments":{}}</tool_call> with valid JSON. '
        f"Otherwise answer normally. Tools: {definitions}"
    )
    rendered: list[dict[str, str]] = [{"role": "system", "content": instruction}]
    for message in messages:
        if message.role is ChatRole.SYSTEM:
            rendered[0]["content"] += f"\n\n{message.content}"
        elif message.role is ChatRole.TOOL:
            rendered.append(
                {
                    "role": "user",
                    "content": (
                        f'<tool_response name="{message.name}" call_id="{message.tool_call_id}">'
                        f"{message.content}</tool_response>"
                    ),
                }
            )
        elif message.tool_calls:
            calls = "".join(
                "<tool_call>"
                + json.dumps(
                    {"id": call.id, "name": call.name, "arguments": call.arguments},
                    separators=(",", ":"),
                )
                + "</tool_call>"
                for call in message.tool_calls
            )
            rendered.append({"role": message.role.value, "content": message.content + calls})
        else:
            rendered.append({"role": message.role.value, "content": message.content})
    return rendered


def _apply_template(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    extra: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if not has_chat_template(processor):
        raise ModelBackendError(
            ModelErrorDetail(
                code=ModelErrorCode.MISSING_CHAT_TEMPLATE,
                message=f"Model {model_id!r} does not provide a chat template.",
                model_id=model_id,
                capability="chat_template",
            )
        )
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    if extra:
        kwargs.update(extra)
    try:
        prepared = processor.apply_chat_template(list(messages), **kwargs)
    except (TypeError, ValueError) as error:
        raise ModelBackendError(
            ModelErrorDetail(
                code=ModelErrorCode.MISSING_CHAT_TEMPLATE,
                message=f"The chat template for {model_id!r} could not format this request.",
                model_id=model_id,
                capability="chat_template",
                context={"reason": str(error)},
            )
        ) from error
    if not isinstance(prepared, Mapping):
        raise ModelBackendError(
            ModelErrorDetail(
                code=ModelErrorCode.INVALID_MODEL,
                message=f"The processor for {model_id!r} returned unsupported chat inputs.",
                model_id=model_id,
            )
        )
    return prepared


class PlainChatAdapter:
    """Use plain chat, with a conservative tagged fallback when tools are supplied."""

    def __init__(self, model_id: str, context_limit: int | None = None) -> None:
        self._model_id = model_id
        self._capabilities = ModelCapabilities(
            generative=True,
            chat_template=True,
            native_tools=False,
            structured_fallback=True,
            reasoning_channels=False,
            streaming_abort=True,
            context_limit=context_limit,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def prepare_inputs(
        self,
        processor: Any,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDefinition],
        thinking_enabled: bool,
    ) -> Mapping[str, Any]:
        if thinking_enabled:
            from oasis.errors import UnsupportedCapabilityError

            raise UnsupportedCapabilityError("reasoning channels", self._model_id)
        rendered: Sequence[Mapping[str, Any]] = (
            _fallback_message_dicts(messages, tools) if tools else _standard_message_dicts(messages)
        )
        return _apply_template(processor, rendered, model_id=self._model_id)

    def stream_parser(
        self, *, thinking_enabled: bool, tools_enabled: bool = False, generation_prefix: str = ""
    ) -> StreamParser:
        del thinking_enabled, generation_prefix
        return (
            _BufferedAgenticParser(self._model_id, gemma=False)
            if tools_enabled
            else _PassthroughParser()
        )

    def preserve_special_tokens(
        self, *, thinking_enabled: bool, tools_enabled: bool = False
    ) -> bool:
        del thinking_enabled
        return tools_enabled


class Gemma4ChatAdapter:
    """Gemma 4 native tool lifecycle and opt-in thought-channel separation."""

    def __init__(self, model_id: str, context_limit: int | None = None) -> None:
        self._model_id = model_id
        self._capabilities = ModelCapabilities(
            generative=True,
            chat_template=True,
            native_tools=True,
            structured_fallback=False,
            reasoning_channels=True,
            streaming_abort=True,
            context_limit=context_limit,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def prepare_inputs(
        self,
        processor: Any,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDefinition],
        thinking_enabled: bool,
    ) -> Mapping[str, Any]:
        extra: dict[str, Any] = {"enable_thinking": thinking_enabled}
        if tools:
            extra["tools"] = [gemma_tool_schema(tool) for tool in tools]
        return _apply_template(
            processor,
            _gemma_message_dicts(messages),
            model_id=self._model_id,
            extra=extra,
        )

    def stream_parser(
        self, *, thinking_enabled: bool, tools_enabled: bool = False, generation_prefix: str = ""
    ) -> StreamParser:
        # The native template opens thought itself after tool results. skip_prompt=True
        # means generation omits that marker; inspect this request's actual prompt tail.
        thought_prefilled = generation_prefix.rstrip().endswith(_GemmaThinkingParser._START)
        if tools_enabled:
            return _BufferedAgenticParser(
                self._model_id, gemma=True, thought_prefilled=thought_prefilled
            )
        return (
            _GemmaThinkingParser(thought_prefilled=thought_prefilled)
            if thinking_enabled or thought_prefilled
            else _PassthroughParser()
        )

    def preserve_special_tokens(
        self, *, thinking_enabled: bool, tools_enabled: bool = False
    ) -> bool:
        return thinking_enabled or tools_enabled
