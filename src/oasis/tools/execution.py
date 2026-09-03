"""Deadline-, cancellation-, and schema-aware invocation of registry tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

from oasis.schemas.tools import (
    ToolError,
    ToolErrorCode,
    ToolEvent,
    ToolEventKind,
    ToolResult,
    ToolResultStatus,
)
from oasis.tools.protocols import (
    StreamingTool,
    Tool,
    ToolCancelledError,
    ToolContext,
    ToolExecutionError,
)
from oasis.tools.registry import ToolRegistryError, validate_arguments, validate_output


def _failure(code: ToolErrorCode, message: str, *, expired: bool = False) -> ToolResult:
    return ToolResult(
        status=ToolResultStatus.EXPIRED if expired else ToolResultStatus.FAILED,
        summary=message,
        error=ToolError(code=code, message=message),
    )


def _preflight(
    tool: Tool | StreamingTool,
    arguments: Mapping[str, Any],
    context: ToolContext,
) -> ToolResult | None:
    spec = tool.spec
    try:
        validate_arguments(spec, arguments)
    except ToolRegistryError as error:
        return _failure(ToolErrorCode.INVALID_ARGUMENTS, str(error))
    missing_providers = spec.required_providers - context.providers.keys()
    missing_resources = spec.required_resources - context.resources.keys()
    if missing_providers or missing_resources or spec.privacy not in context.allowed_privacy:
        return _failure(
            ToolErrorCode.CAPABILITY_DENIED,
            "tool prerequisites or privacy permission are unavailable",
        )
    if context.cancellation.cancelled:
        return _failure(
            ToolErrorCode.CANCELLED,
            context.cancellation.reason or "tool invocation was cancelled",
            expired=True,
        )
    if context.remaining_seconds <= 0:
        context.cancellation.cancel("deadline exceeded")
        return _failure(ToolErrorCode.DEADLINE_EXCEEDED, "tool deadline exceeded", expired=True)
    return None


async def invoke_tool(
    tool: Tool | StreamingTool,
    arguments: Mapping[str, Any],
    context: ToolContext,
) -> ToolResult:
    """Invoke a non-streaming tool under its absolute deadline and cancellation signal."""

    spec = tool.spec
    preflight = _preflight(tool, arguments, context)
    if preflight is not None:
        return preflight
    remaining = context.remaining_seconds
    if not isinstance(tool, Tool):
        return _failure(ToolErrorCode.INTERNAL_ERROR, "streaming tool requires stream execution")

    task = asyncio.create_task(tool.run(arguments, context))
    cancellation_wait = asyncio.create_task(context.cancellation.wait())
    try:
        done, _ = await asyncio.wait(
            {task, cancellation_wait}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            cancellation_wait.cancel()
            try:
                result = task.result()
                validate_output(spec, result)
                return result
            except ToolCancelledError as error:
                return _failure(
                    ToolErrorCode.CANCELLED, str(error) or "tool was cancelled", expired=True
                )
            except ToolExecutionError as error:
                return ToolResult(
                    status=ToolResultStatus.FAILED,
                    summary=error.detail.message,
                    error=error.detail,
                )
            except ToolRegistryError as error:
                return _failure(ToolErrorCode.INTERNAL_ERROR, str(error))
            except Exception as error:
                return _failure(ToolErrorCode.INTERNAL_ERROR, f"tool failed: {error}")

        if cancellation_wait in done:
            message = cancellation_wait.result()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return _failure(ToolErrorCode.CANCELLED, message, expired=True)

        context.cancellation.cancel("deadline exceeded")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return _failure(ToolErrorCode.DEADLINE_EXCEEDED, "tool deadline exceeded", expired=True)
    finally:
        if not cancellation_wait.done():
            cancellation_wait.cancel()
        await asyncio.gather(cancellation_wait, return_exceptions=True)


async def stream_tool(
    tool: Tool | StreamingTool,
    arguments: Mapping[str, Any],
    context: ToolContext,
) -> AsyncIterator[ToolEvent]:
    """Execute a streaming handler while enforcing ordering, declarations, and deadline."""

    preflight = _preflight(tool, arguments, context)
    if preflight is not None:
        yield ToolEvent(sequence=0, kind=ToolEventKind.RESULT, result=preflight)
        return
    if not isinstance(tool, StreamingTool):
        result = _failure(ToolErrorCode.INTERNAL_ERROR, "tool has no streaming handler")
        yield ToolEvent(sequence=0, kind=ToolEventKind.RESULT, result=result)
        return
    generator = tool.stream(arguments, context)
    expected_sequence = 0
    cancellation_wait = asyncio.create_task(context.cancellation.wait())
    try:
        while True:
            next_event: asyncio.Future[ToolEvent] = asyncio.ensure_future(anext(generator))
            waiters = {
                cast(asyncio.Future[object], next_event),
                cast(asyncio.Future[object], cancellation_wait),
            }
            done, _ = await asyncio.wait(
                waiters,
                timeout=context.remaining_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if next_event in done:
                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    result = _failure(
                        ToolErrorCode.INTERNAL_ERROR,
                        "streaming tool ended without a terminal result",
                    )
                    yield ToolEvent(
                        sequence=expected_sequence,
                        kind=ToolEventKind.RESULT,
                        result=result,
                    )
                    return
                if event.sequence != expected_sequence:
                    result = _failure(
                        ToolErrorCode.INTERNAL_ERROR,
                        "streaming tool emitted an out-of-order event",
                    )
                    yield ToolEvent(
                        sequence=expected_sequence,
                        kind=ToolEventKind.RESULT,
                        result=result,
                    )
                    return
                if event.kind is ToolEventKind.PROGRESS and not tool.spec.streams_progress:
                    raise ToolRegistryError("tool emitted undeclared progress")
                if event.kind is ToolEventKind.CANDIDATE and not tool.spec.streams_candidates:
                    raise ToolRegistryError("tool emitted undeclared candidate")
                if event.kind is ToolEventKind.BOUND and not tool.spec.streams_bounds:
                    raise ToolRegistryError("tool emitted undeclared bound")
                if event.kind is ToolEventKind.RESULT:
                    assert event.result is not None
                    validate_output(tool.spec, event.result)
                    yield event
                    return
                yield event
                expected_sequence += 1
                continue

            next_event.cancel()
            await asyncio.gather(next_event, return_exceptions=True)
            if cancellation_wait in done:
                code = ToolErrorCode.CANCELLED
                message = cancellation_wait.result()
            else:
                context.cancellation.cancel("deadline exceeded")
                code = ToolErrorCode.DEADLINE_EXCEEDED
                message = "tool deadline exceeded"
            result = _failure(code, message, expired=True)
            yield ToolEvent(sequence=expected_sequence, kind=ToolEventKind.RESULT, result=result)
            return
    except ToolCancelledError as error:
        result = _failure(ToolErrorCode.CANCELLED, str(error) or "tool was cancelled", expired=True)
        yield ToolEvent(sequence=expected_sequence, kind=ToolEventKind.RESULT, result=result)
    except (ToolRegistryError, ToolExecutionError) as error:
        detail = (
            error.detail
            if isinstance(error, ToolExecutionError)
            else ToolError(code=ToolErrorCode.INTERNAL_ERROR, message=str(error))
        )
        result = ToolResult(status=ToolResultStatus.FAILED, summary=detail.message, error=detail)
        yield ToolEvent(sequence=expected_sequence, kind=ToolEventKind.RESULT, result=result)
    except Exception as error:
        result = _failure(ToolErrorCode.INTERNAL_ERROR, f"streaming tool failed: {error}")
        yield ToolEvent(sequence=expected_sequence, kind=ToolEventKind.RESULT, result=result)
    finally:
        if not cancellation_wait.done():
            cancellation_wait.cancel()
        await asyncio.gather(cancellation_wait, return_exceptions=True)
        close = getattr(generator, "aclose", None)
        if close is not None:
            await close()
