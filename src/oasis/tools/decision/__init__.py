"""Stable location, routing, and scenario decision tools."""

from oasis.tools.decision.compile import CompileProblemTool
from oasis.tools.decision.improve import ImproveTool
from oasis.tools.decision.output import RenderMapTool, SummarizePlanTool
from oasis.tools.decision.scenario import ScenarioSweepTool
from oasis.tools.protocols import StreamingTool, Tool


def decision_tools() -> tuple[Tool | StreamingTool, ...]:
    """Create decision tools without import-time registration or global state."""

    return (
        CompileProblemTool(),
        ImproveTool(),
        ScenarioSweepTool(),
        RenderMapTool(),
        SummarizePlanTool(),
    )


__all__ = [
    "CompileProblemTool",
    "ImproveTool",
    "RenderMapTool",
    "ScenarioSweepTool",
    "SummarizePlanTool",
    "decision_tools",
]
