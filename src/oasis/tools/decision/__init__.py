"""Stable Phase 5 decision tools."""

from oasis.tools.decision.compile import CompileProblemTool
from oasis.tools.decision.improve import ImproveTool
from oasis.tools.decision.output import RenderMapTool, SummarizePlanTool
from oasis.tools.protocols import StreamingTool, Tool


def decision_tools() -> tuple[Tool | StreamingTool, ...]:
    """Create decision tools without import-time registration or global state."""

    return (CompileProblemTool(), ImproveTool(), RenderMapTool(), SummarizePlanTool())


__all__ = [
    "CompileProblemTool",
    "ImproveTool",
    "RenderMapTool",
    "SummarizePlanTool",
    "decision_tools",
]
