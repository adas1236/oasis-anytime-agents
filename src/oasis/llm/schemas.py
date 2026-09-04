"""Stable, implementation-independent schemas for model conversations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, computed_field, model_validator


class ChatRole(StrEnum):
    """Roles accepted in a raw chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    """Portable generation termination reasons."""

    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"
    TOOL_CALL = "tool_call"


class ToolCall(BaseModel):
    """One canonical model request to invoke a named tool."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,127}$")
    arguments: dict[str, JsonValue]


class ToolDefinition(BaseModel):
    """Portable function declaration projected from a registry ToolSpec."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]

    def transformers_schema(self) -> dict[str, Any]:
        """Return the standard shape consumed by Transformers chat templates."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ChatMessage(BaseModel):
    """One chat message, optionally carrying canonical tool-call lifecycle data."""

    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def role_fields_are_consistent(self) -> Self:
        if self.tool_calls and self.role is not ChatRole.ASSISTANT:
            raise ValueError("only assistant messages may contain tool calls")
        if self.role is ChatRole.TOOL and (self.tool_call_id is None or self.name is None):
            raise ValueError("tool messages require tool_call_id and name")
        if self.role is not ChatRole.TOOL and (
            self.tool_call_id is not None or self.name is not None
        ):
            raise ValueError("tool_call_id and name are only valid on tool messages")
        if not self.content and not self.tool_calls and self.role is not ChatRole.TOOL:
            raise ValueError("a non-tool message requires content or a tool call")
        return self


class ModelCapabilities(BaseModel):
    """Capabilities available through a concrete backend/adapter combination."""

    model_config = ConfigDict(frozen=True)

    generative: bool
    chat_template: bool
    native_tools: bool
    structured_fallback: bool
    reasoning_channels: bool
    streaming_abort: bool
    context_limit: int | None = Field(default=None, ge=1)


class ModelProfile(BaseModel):
    """A named model plus model-specific, non-placement defaults."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    context_limit: int | None = Field(default=None, ge=1)
    supports_thinking: bool = False
    supports_native_tools: bool = False
    is_custom: bool = False
    estimated_parameter_count: int | None = Field(default=None, ge=1)


class TokenUsage(BaseModel):
    """Token counts for one turn or an aggregate of multiple turns."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    generated_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """All input and output tokens counted against a total budget."""

        return self.input_tokens + self.generated_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Aggregate token accounting without mutating either operand."""

        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            generated_tokens=self.generated_tokens + other.generated_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @classmethod
    def aggregate(cls, usages: list[TokenUsage]) -> TokenUsage:
        """Aggregate a sequence of per-turn usage records."""

        total = cls()
        for usage in usages:
            total += usage
        return total


class ModelRequest(BaseModel):
    """A backend-independent raw chat generation request."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    max_generated_tokens: int = Field(default=512, ge=1)
    thinking_enabled: bool = False
    tools: tuple[ToolDefinition, ...] = ()
    seed: int = 0


class ModelDelta(BaseModel):
    """One streamed text fragment or the final usage record."""

    model_config = ConfigDict(frozen=True)

    text: str = ""
    thought: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None
    finish_reason: FinishReason | None = None

    @model_validator(mode="after")
    def has_content_or_terminal_data(self) -> Self:
        """Reject empty deltas that carry no observable state."""

        if (
            not self.text
            and not self.thought
            and not self.tool_calls
            and self.usage is None
            and self.finish_reason is None
        ):
            raise ValueError("a model delta must contain text, thought, usage, or a finish reason")
        return self


class ModelTurn(BaseModel):
    """Collected output from one completed model request."""

    model_config = ConfigDict(frozen=True)

    message: ChatMessage
    usage: TokenUsage
    finish_reason: FinishReason
