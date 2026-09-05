"""Deterministic geospatial and public-health evidence tools."""

from oasis.tools.evidence.access import IsochronesTool, ServiceMatrixTool, TravelMatrixTool
from oasis.tools.evidence.construction import BuildCandidatesTool, BuildDemandTool
from oasis.tools.evidence.health import DeriveHealthMeasureTool
from oasis.tools.evidence.locations import InspectArtifactTool, MaterializeLocationsTool
from oasis.tools.evidence.normalize import NormalizeArtifactTool
from oasis.tools.evidence.overlay import OverlayReduceTool
from oasis.tools.evidence.profile import ProfileArtifactTool
from oasis.tools.protocols import Tool


def evidence_tools() -> tuple[Tool, ...]:
    """Create the complete deterministic Phase 3 evidence-tool catalog."""

    return (
        BuildCandidatesTool(),
        BuildDemandTool(),
        DeriveHealthMeasureTool(),
        IsochronesTool(),
        InspectArtifactTool(),
        MaterializeLocationsTool(),
        NormalizeArtifactTool(),
        OverlayReduceTool(),
        ProfileArtifactTool(),
        ServiceMatrixTool(),
        TravelMatrixTool(),
    )


__all__ = [
    "BuildCandidatesTool",
    "BuildDemandTool",
    "DeriveHealthMeasureTool",
    "InspectArtifactTool",
    "IsochronesTool",
    "MaterializeLocationsTool",
    "NormalizeArtifactTool",
    "OverlayReduceTool",
    "ProfileArtifactTool",
    "ServiceMatrixTool",
    "TravelMatrixTool",
    "evidence_tools",
]
