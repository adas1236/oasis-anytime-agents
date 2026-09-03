from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from oasis.artifacts import LocalArtifactStore
from oasis.schemas.tools import (
    ToolEvent,
    ToolEventKind,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools import (
    CancellationToken,
    ToolContext,
    ToolRegistry,
    ToolRegistryError,
    invoke_tool,
    stream_tool,
)
from oasis.tools.calculator import CalculatorTool
from oasis.tools.testing import assert_tool_contract


def context(tmp_path: Path, *, deadline: float | None = None) -> ToolContext:
    return ToolContext(
        run_id="test-run",
        artifact_store=LocalArtifactStore(tmp_path),
        deadline_monotonic=deadline if deadline is not None else time.monotonic() + 2,
        cancellation=CancellationToken(),
        seed=7,
    )


class AlternateCalculator(CalculatorTool):
    pass


def test_registry_rejects_duplicate_names_and_bad_versions() -> None:
    registry = ToolRegistry([CalculatorTool()])

    with pytest.raises(ToolRegistryError, match="duplicate"):
        registry.register(AlternateCalculator())
    with pytest.raises(ValidationError, match="semantic"):
        CalculatorTool.spec.model_copy(update={"version": "v1"}, deep=True).__class__(
            **{**CalculatorTool.spec.model_dump(), "version": "v1"}
        )

    class InvalidVersionTool(CalculatorTool):
        spec = CalculatorTool.spec.model_copy(update={"name": "bad_version", "version": "v1"})

    with pytest.raises(ToolRegistryError, match="invalid spec"):
        registry.register(InvalidVersionTool())


def test_registry_rejects_invalid_json_schema() -> None:
    invalid_spec = CalculatorTool.spec.model_copy(
        update={"name": "invalid_schema", "input_schema": {"type": "nonsense"}}
    )

    class InvalidSchemaTool(CalculatorTool):
        spec = invalid_spec

    with pytest.raises(ToolRegistryError, match=r"invalid .* JSON Schema"):
        ToolRegistry([InvalidSchemaTool()])


def test_registry_rejects_wrong_handler_signature() -> None:
    class InvalidHandlerTool(CalculatorTool):
        spec = CalculatorTool.spec.model_copy(update={"name": "invalid_handler"})

        async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
            del arguments
            raise AssertionError("must not run")

    with pytest.raises(ToolRegistryError, match="exactly"):
        ToolRegistry([InvalidHandlerTool()])


def test_tool_spec_rejects_unstable_capability_tags() -> None:
    values = CalculatorTool.spec.model_dump()
    values["capability_tags"] = {" arithmetic"}

    with pytest.raises(ValidationError, match="tags"):
        ToolSpec(**values)


def test_registry_filters_by_capability_and_task() -> None:
    registry = ToolRegistry([CalculatorTool()])

    assert [spec.name for spec in registry.select(capabilities=frozenset({"arithmetic"}))] == [
        "calculator"
    ]
    assert registry.select(capabilities=frozenset({"network"})) == ()
    assert registry.select(problem_tags=frozenset({"routing"})) == ()


class FakeEntryPoint:
    name = "test_plugin"

    def load(self) -> object:
        return lambda: TestPluginTool()


class TestPluginTool(CalculatorTool):
    spec = CalculatorTool.spec.model_copy(
        update={
            "name": "test_plugin_calculator",
            "version": "2.0.0",
            "description": "Calculator supplied by a test-only entry point.",
        }
    )


def test_entry_point_plugin_is_discoverable_without_registry_edits() -> None:
    registry = ToolRegistry()

    registry.discover([FakeEntryPoint()])

    assert registry.get("test_plugin_calculator").spec.version == "2.0.0"


@pytest.mark.asyncio
async def test_calculator_and_reusable_contract_suite(tmp_path: Path) -> None:
    tool = CalculatorTool()

    result = await assert_tool_contract(tool, context=context(tmp_path))

    assert result.status is ToolResultStatus.COMPLETE
    assert result.metrics == {"value": 5.0}


class WaitingTool:
    spec = ToolSpec(
        name="waiting",
        version="1.0.0",
        description="Wait until cancelled.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object"},
        runtime=ToolRuntimeEstimate(p50_ms=1, p95_ms=2),
        smoke_input={},
    )

    async def run(self, arguments: Mapping[str, Any], tool_context: ToolContext) -> ToolResult:
        del arguments
        await tool_context.cancellation.wait()
        tool_context.cancellation.raise_if_cancelled()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_tool_cancellation_is_cooperative_and_structured(tmp_path: Path) -> None:
    tool_context = context(tmp_path)
    pending = asyncio.create_task(invoke_tool(WaitingTool(), {}, tool_context))
    await asyncio.sleep(0)
    tool_context.cancellation.cancel("test requested cancellation")

    result = await pending

    assert result.status is ToolResultStatus.EXPIRED
    assert result.error is not None
    assert result.error.code.value == "cancelled"


@pytest.mark.asyncio
async def test_expired_deadline_does_not_start_handler(tmp_path: Path) -> None:
    result = await invoke_tool(WaitingTool(), {}, context(tmp_path, deadline=0))

    assert result.status is ToolResultStatus.EXPIRED
    assert result.error is not None
    assert result.error.code.value == "deadline_exceeded"


@pytest.mark.asyncio
async def test_running_tool_is_stopped_at_deadline(tmp_path: Path) -> None:
    result = await invoke_tool(
        WaitingTool(), {}, context(tmp_path, deadline=time.monotonic() + 0.01)
    )

    assert result.status is ToolResultStatus.EXPIRED
    assert result.error is not None
    assert result.error.code.value == "deadline_exceeded"


@pytest.mark.asyncio
async def test_invalid_arguments_never_reach_handler(tmp_path: Path) -> None:
    result = await invoke_tool(
        CalculatorTool(), {"operation": "add", "operands": []}, context(tmp_path)
    )

    assert result.status is ToolResultStatus.FAILED
    assert result.error is not None
    assert result.error.code.value == "invalid_arguments"


class ProgressTool:
    spec = ToolSpec(
        name="progress_example",
        version="1.0.0",
        description="Emit deterministic progress before a result.",
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object", "additionalProperties": False},
        runtime=ToolRuntimeEstimate(p50_ms=1, p95_ms=20),
        streams_progress=True,
        smoke_input={},
    )

    async def stream(
        self, arguments: Mapping[str, Any], tool_context: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        del arguments
        tool_context.cancellation.raise_if_cancelled()
        yield ToolEvent(sequence=0, kind=ToolEventKind.PROGRESS, progress=0.5)
        yield ToolEvent(
            sequence=1,
            kind=ToolEventKind.RESULT,
            result=ToolResult(status=ToolResultStatus.COMPLETE, summary="done"),
        )


@pytest.mark.asyncio
async def test_streaming_tool_events_are_ordered_and_contract_checked(tmp_path: Path) -> None:
    tool = ProgressTool()
    events = [event async for event in stream_tool(tool, {}, context(tmp_path))]

    assert [event.sequence for event in events] == [0, 1]
    assert events[-1].result is not None
    assert events[-1].result.status is ToolResultStatus.COMPLETE
    assert (await assert_tool_contract(tool, context=context(tmp_path))).status is (
        ToolResultStatus.COMPLETE
    )
