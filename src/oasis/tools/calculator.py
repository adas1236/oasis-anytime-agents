"""Small deterministic calculator used to demonstrate the tool contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from oasis.schemas.tools import (
    DeterminismClassification,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.protocols import ToolContext, ToolExecutionError


class CalculatorOperation(StrEnum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class CalculatorInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: CalculatorOperation
    operands: tuple[float, ...] = Field(min_length=1, max_length=100)


class CalculatorOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float


class CalculatorTool:
    """Perform bounded basic arithmetic without expression evaluation."""

    spec = ToolSpec(
        name="calculator",
        version="1.0.0",
        description="Perform addition, subtraction, multiplication, or division on numbers.",
        input_schema=CalculatorInput.model_json_schema(),
        output_schema=CalculatorOutput.model_json_schema(),
        capability_tags=frozenset({"arithmetic", "demonstration"}),
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=1, p95_ms=10),
        smoke_input={"operation": "add", "operands": [2, 3]},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        context.cancellation.raise_if_cancelled()
        request = CalculatorInput.model_validate(arguments)
        first, *rest = request.operands
        if request.operation is CalculatorOperation.ADD:
            value = sum(request.operands)
        elif request.operation is CalculatorOperation.SUBTRACT:
            value = first - sum(rest)
        elif request.operation is CalculatorOperation.MULTIPLY:
            value = math.prod(request.operands)
        else:
            value = first
            for operand in rest:
                if operand == 0:
                    raise ToolExecutionError(
                        ToolError(
                            code=ToolErrorCode.INVALID_ARGUMENTS,
                            message="division by zero is undefined",
                        )
                    )
                value /= operand
        output = CalculatorOutput(value=value)
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={"value": value},
            metrics=output.model_dump(mode="json"),
        )
