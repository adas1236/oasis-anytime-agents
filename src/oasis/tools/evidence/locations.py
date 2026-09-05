"""Explicit geocoder-candidate selection and bounded artifact inspection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Point

from oasis.artifacts import put_vector, read_json, read_matrix
from oasis.providers.models import PlaceCandidate
from oasis.schemas import (
    ArtifactKind,
    DeterminismClassification,
    SideEffectClassification,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.evidence.common import (
    MISSING_ARTIFACT_ID,
    artifact_ref,
    child_provenance,
    invalid,
    read_frame,
    require_kind,
)
from oasis.tools.protocols import ToolContext


class MaterializeLocationsInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resolution_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    provider_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    metadata_fields: tuple[str, ...] = ()
    unit_opening_cost: float | None = Field(default=None, ge=0)


class MaterializeLocationsOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    row_count: int
    fields: tuple[str, ...]
    location_ids: tuple[str, ...]


class MaterializeLocationsTool:
    """Select explicitly identified candidates, preserving the caller's order."""

    spec = ToolSpec(
        name="materialize_locations",
        version="1.0.0",
        description=(
            "Create a WGS84 point artifact from explicit provider_ids returned by resolve_area "
            "or resolve_locations. Order is preserved; ambiguity is never auto-selected. "
            "Fields are id, name, latitude, longitude plus requested scalar metadata_fields "
            "(e.g. population). Optional unit_opening_cost adds a uniform opening_cost column."
        ),
        input_schema=MaterializeLocationsInput.model_json_schema(),
        output_schema=MaterializeLocationsOutput.model_json_schema(),
        runtime=ToolRuntimeEstimate(p50_ms=20, p95_ms=1_000),
        capability_tags=frozenset({"evidence", "geocoding", "offline"}),
        artifact_tags=frozenset({ArtifactKind.JSON_SPECIFICATION, ArtifactKind.VECTOR}),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        smoke_input={"resolution_artifact_id": MISSING_ARTIFACT_ID, "provider_ids": ["example"]},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = MaterializeLocationsInput.model_validate(arguments)
        reference = artifact_ref(context, request.resolution_artifact_id)
        require_kind(reference, {ArtifactKind.JSON_SPECIFICATION})
        payload = read_json(context.artifact_store, reference)
        if not isinstance(payload, dict):
            invalid("expected a place-resolution artifact")
        groups = payload.get("results", [payload])
        candidates: dict[str, PlaceCandidate] = {}
        for group in groups:
            for raw in group.get("candidates", []):
                candidate = PlaceCandidate.model_validate(raw)
                previous = candidates.get(candidate.provider_id)
                if previous is not None and (
                    previous.latitude != candidate.latitude
                    or previous.longitude != candidate.longitude
                    or previous.provider_metadata != candidate.provider_metadata
                ):
                    invalid("the same provider ID has conflicting source attributes")
                candidates[candidate.provider_id] = candidate
        if len(set(request.provider_ids)) != len(request.provider_ids):
            invalid("provider_ids must not contain duplicate locations")
        reserved = {"id", "name", "latitude", "longitude", "geometry", "opening_cost"}
        if set(request.metadata_fields) & reserved:
            invalid("metadata_fields cannot replace canonical location columns")
        rows: list[dict[str, Any]] = []
        points = []
        for identifier in request.provider_ids:
            context.cancellation.raise_if_cancelled()
            if identifier not in candidates:
                invalid(f"provider ID {identifier!r} is not in the supplied resolution artifact")
            candidate = candidates[identifier]
            row: dict[str, Any] = {
                "id": identifier,
                "name": candidate.display_name,
                "latitude": candidate.latitude,
                "longitude": candidate.longitude,
            }
            for field in request.metadata_fields:
                if field not in candidate.provider_metadata:
                    invalid(f"candidate {identifier!r} has no metadata field {field!r}")
                value = candidate.provider_metadata[field]
                if isinstance(value, (list, dict)):
                    invalid("only scalar candidate metadata can become a point attribute")
                row[field] = value
            if request.unit_opening_cost is not None:
                row["opening_cost"] = request.unit_opening_cost
            rows.append(row)
            points.append(Point(candidate.longitude, candidate.latitude))
        frame = gpd.GeoDataFrame(rows, geometry=points, crs="EPSG:4326")
        result_ref = put_vector(
            context.artifact_store,
            frame,
            units="location_attributes",
            provenance=child_provenance(self.spec.name, self.spec.version, [reference], request),
        )
        output = MaterializeLocationsOutput(
            artifact_id=result_ref.id,
            row_count=len(rows),
            fields=tuple(str(column) for column in frame.columns),
            location_ids=request.provider_ids,
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary="Selected geocoder candidates are available as a point artifact.",
            artifacts=(result_ref,),
            metrics=output.model_dump(mode="json"),
        )


class InspectArtifactInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    path: tuple[str | int, ...] = ()
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)


class InspectArtifactTool:
    spec = ToolSpec(
        name="inspect_artifact",
        version="1.0.0",
        description=(
            "Read bounded JSON, point/table records, or matrix rows from a tool-produced artifact. "
            "Use path to select nested JSON keys/indices and offset/limit to page lists. "
            "Matrix rows contain row_id and values; column_ids gives their labels."
        ),
        input_schema=InspectArtifactInput.model_json_schema(),
        output_schema={"type": "object"},
        runtime=ToolRuntimeEstimate(p50_ms=5, p95_ms=1_000),
        capability_tags=frozenset({"evidence", "inspection", "offline"}),
        determinism=DeterminismClassification.DETERMINISTIC,
        smoke_input={"artifact_id": MISSING_ARTIFACT_ID},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = InspectArtifactInput.model_validate(arguments)
        reference = artifact_ref(context, request.artifact_id)
        labels: dict[str, Any] = {}
        if reference.kind in {ArtifactKind.TABLE, ArtifactKind.VECTOR}:
            frame = read_frame(context, reference)
            if isinstance(frame, gpd.GeoDataFrame):
                frame = frame.drop(columns=frame.geometry.name)
            value = json.loads(frame.to_json(orient="records"))
        elif reference.kind is ArtifactKind.MATRIX:
            matrix = read_matrix(context.artifact_store, reference)
            labels["column_ids"] = list(matrix.column_ids)
            value = [
                {
                    "row_id": identifier,
                    "values": [float(v) if np.isfinite(v) else None for v in row],
                }
                for identifier, row in zip(matrix.row_ids, matrix.values, strict=True)
            ]
        elif reference.media_type == "application/json":
            value = read_json(context.artifact_store, reference)
        else:
            invalid("inspection supports JSON, table, vector, and matrix artifacts")
        for key in request.path:
            try:
                value = value[key]
            except (KeyError, IndexError, TypeError):
                invalid(f"artifact path does not contain {key!r}")
        total = len(value) if isinstance(value, list) else None
        if total is not None:
            value = value[request.offset : request.offset + request.limit]
        output = {"value": value, "total_items": total, "next_offset": None, **labels}
        while len(json.dumps(output, ensure_ascii=False).encode()) > 5_500:
            if isinstance(value, list) and len(value) > 1:
                value = value[:-1]
                output["value"] = value
            else:
                invalid("value is too large for a tool result; select a narrower JSON path")
        if total is not None and request.offset + len(value) < total:
            output["next_offset"] = request.offset + len(value)
        return ToolResult(
            status=ToolResultStatus.COMPLETE, summary="Artifact contents.", metrics=output
        )
