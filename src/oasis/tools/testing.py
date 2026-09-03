"""Reusable public checks for third-party OASIS tool authors."""

from __future__ import annotations

from oasis.schemas.tools import DeterminismClassification, ToolResult
from oasis.tools.execution import invoke_tool, stream_tool
from oasis.tools.protocols import StreamingTool, Tool, ToolContext
from oasis.tools.registry import ToolRegistry


async def assert_tool_contract(
    tool: Tool | StreamingTool,
    *,
    context: ToolContext,
) -> ToolResult:
    """Validate registration, smoke execution, serialization, and determinism."""

    async def execute() -> ToolResult:
        if tool.spec.streams_progress or tool.spec.streams_candidates or tool.spec.streams_bounds:
            events = [event async for event in stream_tool(tool, tool.spec.smoke_input, context)]
            if not events or events[-1].result is None:
                raise AssertionError("streaming tool contract produced no terminal result")
            return events[-1].result
        return await invoke_tool(tool, tool.spec.smoke_input, context)

    registry = ToolRegistry([tool])
    result = await execute()
    ToolResult.model_validate_json(result.model_dump_json())
    if tool.spec.determinism is DeterminismClassification.DETERMINISTIC:
        second = await execute()
        if result != second:
            raise AssertionError(f"deterministic tool {tool.spec.name!r} changed its result")
    assert registry.get(tool.spec.name).spec == tool.spec
    return result
