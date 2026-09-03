from __future__ import annotations

import time
from pathlib import Path

import pytest

from oasis.artifacts import LocalArtifactStore
from oasis.llm.fake import FakeModelBackend
from oasis.llm.schemas import ChatMessage, ChatRole, ToolCall
from oasis.llm.tool_loop import run_tool_loop
from oasis.tools import CancellationToken, ToolContext, create_tool_registry


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        run_id="tool-loop-test",
        artifact_store=LocalArtifactStore(tmp_path),
        deadline_monotonic=time.monotonic() + 2,
        cancellation=CancellationToken(),
        seed=11,
    )


@pytest.mark.asyncio
async def test_fake_model_calls_calculator_receives_result_and_finishes(tmp_path: Path) -> None:
    call = ToolCall(
        id="calculation-1",
        name="calculator",
        arguments={"operation": "multiply", "operands": [6, 7]},
    )
    backend = FakeModelBackend([call, "The verified result is 42."])
    registry = create_tool_registry(discover_entry_points=False)

    result = await run_tool_loop(
        backend,
        registry,
        context(tmp_path),
        (ChatMessage(role=ChatRole.USER, content="What is six times seven?"),),
    )

    assert result.final_turn.message.content == "The verified result is 42."
    assert result.invocations[0].call == call
    assert result.invocations[0].result.metrics == {"value": 42.0}
    tool_messages = [message for message in result.messages if message.role is ChatRole.TOOL]
    assert len(tool_messages) == 1
    assert '"value":42.0' in tool_messages[0].content


@pytest.mark.asyncio
async def test_fallback_tool_call_gets_one_bounded_repair(tmp_path: Path) -> None:
    malformed = "<tool_call>{not-json}</tool_call>"
    corrected = (
        '<tool_call>{"name":"calculator","arguments":{"operation":"add",'
        '"operands":[8,9]}}</tool_call>'
    )
    backend = FakeModelBackend([malformed, corrected, "The result is 17."])

    result = await run_tool_loop(
        backend,
        create_tool_registry(discover_entry_points=False),
        context(tmp_path),
        (ChatMessage(role="user", content="Add eight and nine."),),
    )

    assert result.repair_attempts == 1
    assert result.invocations[0].result.metrics == {"value": 17.0}
    assert result.final_turn.message.content == "The result is 17."
