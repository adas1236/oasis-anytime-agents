"""Basic registry-driven model/tool loop without controller or domain policy."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from oasis.errors import ToolCallParseError
from oasis.llm.protocols import ModelBackend
from oasis.llm.schemas import ChatMessage, ChatRole, ModelRequest, ModelTurn, TokenUsage, ToolCall
from oasis.schemas.tools import ToolError, ToolErrorCode, ToolResult, ToolResultStatus
from oasis.tools.execution import invoke_tool
from oasis.tools.protocols import ToolContext
from oasis.tools.registry import ToolRegistry, ToolRegistryError


class ToolInvocationRecord(BaseModel):
    """One model-proposed call and its normalized execution result."""

    model_config = ConfigDict(frozen=True)

    call: ToolCall
    result: ToolResult


class ToolLoopResult(BaseModel):
    """Completed assistant response plus the compact reproducible conversation."""

    model_config = ConfigDict(frozen=True)

    final_turn: ModelTurn
    messages: tuple[ChatMessage, ...]
    invocations: tuple[ToolInvocationRecord, ...]
    usage: TokenUsage
    repair_attempts: int = Field(ge=0)


def _unknown_tool_result(name: str) -> ToolResult:
    message = f"model requested unknown tool {name!r}"
    return ToolResult(
        status=ToolResultStatus.FAILED,
        summary=message,
        error=ToolError(code=ToolErrorCode.NOT_FOUND, message=message),
    )


async def run_tool_loop(
    backend: ModelBackend,
    registry: ToolRegistry,
    context: ToolContext,
    messages: tuple[ChatMessage, ...],
    *,
    request_id: str = "tool-loop",
    max_generated_tokens: int = 512,
    thinking_enabled: bool = False,
    max_tool_rounds: int = 8,
    max_repair_attempts: int = 1,
) -> ToolLoopResult:
    """Let a model invoke registry tools until it returns a final text response.

    This phase-2 loop establishes model/adapter/tool interoperability only. Admission, budgets,
    incumbent validation, and fallback policy remain controller responsibilities in later phases.
    """

    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds must be non-negative")
    if max_repair_attempts not in {0, 1}:
        raise ValueError("tool-call repair is bounded to zero or one attempt")
    history = list(messages)
    definitions = registry.model_definitions()
    invocations: list[ToolInvocationRecord] = []
    usages: list[TokenUsage] = []
    repairs = 0
    generation_number = 0

    while True:
        generation_number += 1
        request = ModelRequest(
            request_id=f"{request_id}-{generation_number}",
            messages=tuple(history),
            max_generated_tokens=max_generated_tokens,
            thinking_enabled=thinking_enabled,
            tools=definitions,
            seed=context.seed,
        )
        try:
            turn = await backend.generate(request)
        except ToolCallParseError:
            if repairs >= max_repair_attempts:
                raise
            repairs += 1
            history.append(
                ChatMessage(
                    role=ChatRole.USER,
                    content=(
                        "The previous tool call was malformed. Return exactly one corrected tool "
                        "call in the required format, with a known name and object arguments."
                    ),
                )
            )
            continue

        usages.append(turn.usage)
        history.append(turn.message)
        calls = turn.message.tool_calls
        if not calls:
            return ToolLoopResult(
                final_turn=turn,
                messages=tuple(history),
                invocations=tuple(invocations),
                usage=TokenUsage.aggregate(usages),
                repair_attempts=repairs,
            )
        if len(invocations) + len(calls) > max_tool_rounds:
            raise RuntimeError("model exceeded the configured tool-call round limit")

        for call in calls:
            try:
                tool = registry.get(call.name)
            except ToolRegistryError:
                result = _unknown_tool_result(call.name)
            else:
                result = await invoke_tool(tool, call.arguments, context)
            invocations.append(ToolInvocationRecord(call=call, result=result))
            history.append(
                ChatMessage(
                    role=ChatRole.TOOL,
                    name=call.name,
                    tool_call_id=call.id,
                    content=result.model_summary(),
                )
            )
