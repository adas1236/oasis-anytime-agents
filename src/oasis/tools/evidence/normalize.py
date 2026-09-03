"""Deterministic vector/table normalization and transformation lineage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from shapely import make_valid

from oasis.schemas import (
    ArtifactKind,
    DeterminismClassification,
    QualitySummary,
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
    put_frame,
    read_frame,
    require_fields,
    require_kind,
)
from oasis.tools.protocols import ToolContext


class NormalizeArtifactInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    target_crs: str | None = None
    clip_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    repair_geometries: bool = True
    drop_empty_geometries: bool = True
    source_id_field: str | None = None
    output_id_field: str = "id"
    id_prefix: str = "feature"
    value_fields: tuple[str, ...] = ()
    unit_scale: float = Field(default=1.0, gt=0.0)
    output_units: str | None = None

    @model_validator(mode="after")
    def declared_unit_conversion(self) -> NormalizeArtifactInput:
        if self.unit_scale != 1.0 and self.output_units is None:
            raise ValueError("a non-identity unit scale requires output_units")
        return self


class NormalizeArtifactOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    row_count: int
    repaired_geometry_count: int
    dropped_empty_geometry_count: int
    duplicate_id_count: int
    crs: str | None
    units: str


def _normalized_ids(values: pd.Series, *, prefix: str) -> tuple[list[str], int]:
    base = [
        f"{prefix}-{index + 1:06d}"
        if pd.isna(value) or not str(value).strip()
        else str(value).strip()
        for index, value in enumerate(values)
    ]
    counts = Counter(base)
    seen: Counter[str] = Counter()
    result: list[str] = []
    for value in base:
        seen[value] += 1
        result.append(value if seen[value] == 1 else f"{value}-{seen[value]}")
    return result, sum(count - 1 for count in counts.values() if count > 1)


class NormalizeArtifactTool:
    """Normalize CRS, geometry, identifiers, clipping, and declared numeric units."""

    version = "1.0.0"
    spec = ToolSpec(
        name="normalize_artifact",
        version=version,
        description=(
            "Normalize a vector or table artifact with explicit CRS transformation, clipping, "
            "geometry repair, stable IDs, and declared unit scaling."
        ),
        input_schema=NormalizeArtifactInput.model_json_schema(),
        output_schema=NormalizeArtifactOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "normalization", "offline"}),
        artifact_tags=frozenset({ArtifactKind.VECTOR, ArtifactKind.TABLE}),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=20, p95_ms=2_000),
        smoke_input={"artifact_id": MISSING_ARTIFACT_ID},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = NormalizeArtifactInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        reference = artifact_ref(context, request.artifact_id)
        require_kind(reference, {ArtifactKind.VECTOR, ArtifactKind.TABLE})
        frame = read_frame(context, reference).copy()
        require_fields(frame, request.value_fields)

        repaired = 0
        dropped_empty = 0
        parents = [reference]
        if reference.kind is ArtifactKind.TABLE:
            if request.target_crs is not None or request.clip_artifact_id is not None:
                invalid("CRS transformation and clipping require a vector artifact")
        else:
            assert isinstance(frame, gpd.GeoDataFrame)
            if frame.crs is None:
                invalid("vector artifact has no CRS")
            if request.target_crs is not None:
                frame = frame.to_crs(request.target_crs)
            if request.repair_geometries:
                invalid_mask = ~frame.geometry.is_valid & ~frame.geometry.isna()
                repaired = int(invalid_mask.sum())
                if repaired:
                    frame.loc[invalid_mask, frame.geometry.name] = frame.loc[
                        invalid_mask, frame.geometry.name
                    ].map(make_valid)
            if request.clip_artifact_id is not None:
                clip_reference = artifact_ref(context, request.clip_artifact_id)
                require_kind(clip_reference, {ArtifactKind.VECTOR})
                clip = read_frame(context, clip_reference)
                assert isinstance(clip, gpd.GeoDataFrame)
                if clip.empty:
                    invalid("clip artifact has no geometries")
                if clip.crs != frame.crs:
                    clip = clip.to_crs(frame.crs)
                frame = frame.clip(clip.geometry.union_all(), keep_geom_type=False)
                parents.append(clip_reference)
            empty_mask = frame.geometry.isna() | frame.geometry.is_empty
            dropped_empty = int(empty_mask.sum()) if request.drop_empty_geometries else 0
            if request.drop_empty_geometries:
                frame = frame.loc[~empty_mask].copy()

        if request.source_id_field is not None:
            require_fields(frame, [request.source_id_field])
            source_ids = frame[request.source_id_field]
        elif request.output_id_field in frame:
            source_ids = frame[request.output_id_field]
        else:
            source_ids = pd.Series([None] * len(frame), index=frame.index, dtype="object")
        normalized_ids, duplicate_ids = _normalized_ids(source_ids, prefix=request.id_prefix)
        frame[request.output_id_field] = normalized_ids
        for field in request.value_fields:
            try:
                frame[field] = pd.to_numeric(frame[field], errors="raise") * request.unit_scale
            except (TypeError, ValueError) as error:
                invalid(f"field {field!r} cannot be converted to numeric units: {error}")

        frame = frame.reset_index(drop=True)
        invalid_count = 0
        if isinstance(frame, gpd.GeoDataFrame):
            invalid_count = int((~frame.geometry.is_valid & ~frame.geometry.isna()).sum())
        warnings: list[str] = []
        if repaired:
            warnings.append(f"repaired {repaired} invalid geometries")
        if dropped_empty:
            warnings.append(f"dropped {dropped_empty} empty geometries")
        if duplicate_ids:
            warnings.append(f"disambiguated {duplicate_ids} duplicate IDs")
        quality = QualitySummary(
            missing_fraction=(
                0.0
                if frame.empty
                else float(
                    frame.drop(columns=frame.geometry.name).isna().to_numpy().mean()
                    if isinstance(frame, gpd.GeoDataFrame)
                    else frame.isna().to_numpy().mean()
                )
            ),
            invalid_geometry_count=invalid_count,
            duplicate_count=duplicate_ids,
            warnings=tuple(warnings),
        )
        output_units = request.output_units or reference.units or "unitless"
        output_ref = put_frame(
            context.artifact_store,
            frame,
            source_kind=reference.kind,
            crs=str(frame.crs) if isinstance(frame, gpd.GeoDataFrame) else reference.crs,
            units=output_units,
            provenance=child_provenance(self.spec.name, self.version, parents, request),
            quality=quality,
        )
        output = NormalizeArtifactOutput(
            artifact_id=output_ref.id,
            row_count=len(frame),
            repaired_geometry_count=repaired,
            dropped_empty_geometry_count=dropped_empty,
            duplicate_id_count=duplicate_ids,
            crs=output_ref.crs,
            units=output_units,
        )
        warning_values: list[JsonValue] = [*warnings]
        summary: dict[str, JsonValue] = {
            "artifact_id": output_ref.id,
            "rows": len(frame),
            "warnings": warning_values,
        }
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary=summary,
            artifacts=(output_ref,),
            metrics=output.model_dump(mode="json"),
        )
