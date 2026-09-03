from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from oasis.artifacts import LocalArtifactStore
from oasis.schemas import ArtifactKind, ArtifactMetadata
from oasis.schemas.tools import (
    MAX_TOOL_SUMMARY_BYTES,
    ToolCostModel,
    ToolResult,
    ToolResultStatus,
)
from oasis.tools.calculator import CalculatorTool


def test_tool_spec_and_result_round_trip() -> None:
    spec = CalculatorTool.spec
    result = ToolResult(
        status=ToolResultStatus.COMPLETE,
        summary={"value": 5},
        metrics={"value": 5},
    )

    assert type(spec).model_validate_json(spec.model_dump_json()) == spec
    assert ToolResult.model_validate_json(result.model_dump_json()) == result


def test_cost_model_estimates_from_instance_features() -> None:
    model = ToolCostModel(base_units=2, feature_weights={"rows": 0.25})

    estimate = model.estimate({"rows": 12, "unused": 4})

    assert estimate.units == 5
    assert estimate.features == {"rows": 12, "unused": 4}


def test_large_tool_summary_must_be_an_artifact(tmp_path: Path) -> None:
    payload = b"x" * (MAX_TOOL_SUMMARY_BYTES + 1)

    with pytest.raises(ValidationError, match="store large output as an artifact"):
        ToolResult(status="complete", summary=payload.decode())

    store = LocalArtifactStore(tmp_path)
    artifact = store.put_bytes(
        payload,
        ArtifactMetadata(kind=ArtifactKind.TRACE_ATTACHMENT, media_type="application/octet-stream"),
    )
    result = ToolResult(
        status="complete",
        summary="Large output stored as an artifact.",
        artifacts=(artifact,),
    )

    assert len(result.model_summary()) < MAX_TOOL_SUMMARY_BYTES
    assert artifact.id in result.model_summary()


def test_large_metrics_cannot_bypass_model_payload_limit() -> None:
    with pytest.raises(ValidationError, match="store large output as an artifact"):
        ToolResult(
            status="complete",
            summary="small",
            metrics={"payload": "x" * MAX_TOOL_SUMMARY_BYTES},
        )
