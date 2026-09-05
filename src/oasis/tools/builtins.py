"""Explicit factory for tools shipped in the base OASIS package."""

from oasis.tools.calculator import CalculatorTool
from oasis.tools.decision import decision_tools
from oasis.tools.evidence import evidence_tools
from oasis.tools.protocols import StreamingTool, Tool
from oasis.tools.providers import provider_tools
from oasis.tools.registry import ToolRegistry


def builtin_tools() -> tuple[Tool | StreamingTool, ...]:
    """Create built-ins without import-time registration or shared mutable state."""

    return (CalculatorTool(), *decision_tools(), *evidence_tools(), *provider_tools())


def create_tool_registry(*, discover_entry_points: bool = True) -> ToolRegistry:
    """Create the advanced SDK registry (legacy/internal, not the default agent surface)."""

    registry = ToolRegistry(builtin_tools())
    if discover_entry_points:
        registry.discover()
    return registry


def create_public_tool_registry(*, discover_entry_points: bool = True) -> ToolRegistry:
    """Create the shared compact live-agent and evaluation tool surface."""
    from oasis.tools.public import public_tools

    registry = ToolRegistry(public_tools(builtin_tools()))
    if discover_entry_points:
        registry.discover()
    return registry
