"""Artifact profiling with explicit data-quality measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from oasis.artifacts import put_json, read_graph, read_matrix, read_raster
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
    read_frame,
)
from oasis.tools.protocols import ToolContext


class ProfileArtifactInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    suppression_fields: tuple[str, ...] = ()
    temporal_fields: tuple[str, ...] = ()


class ProfileArtifactOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_artifact_id: str
    row_count: int | None
    cell_count: int | None
    edge_count: int | None
    missing_fraction: float
    duplicate_count: int
    invalid_geometry_count: int
    empty_geometry_count: int
    suppressed_count: int
    warning_count: int


def _suppression_count(frame: pd.DataFrame, fields: tuple[str, ...]) -> int:
    present = [field for field in fields if field in frame]
    if not present:
        return 0
    markers = frame[present].apply(
        lambda column: column.map(
            lambda value: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else str(value).strip().lower() in {"s", "suppressed", "*", "true"}
            )
        )
    )
    return int(markers.any(axis=1).sum())


def _frame_profile(
    frame: pd.DataFrame, suppression_fields: tuple[str, ...]
) -> tuple[dict[str, object], QualitySummary]:
    geometry_name = frame.geometry.name if isinstance(frame, gpd.GeoDataFrame) else None
    values = frame.drop(columns=geometry_name) if geometry_name is not None else frame
    missing_fraction = (
        0.0 if values.empty or not len(values.columns) else float(values.isna().to_numpy().mean())
    )
    duplicate_source = values.copy()
    if isinstance(frame, gpd.GeoDataFrame):
        duplicate_source["__geometry_wkb__"] = frame.geometry.map(
            lambda geometry: None if geometry is None else geometry.wkb_hex
        )
    duplicates = int(duplicate_source.duplicated().sum())
    invalid = 0
    empty = 0
    duplicate_geometries = 0
    geometry_types: list[str] = []
    if isinstance(frame, gpd.GeoDataFrame):
        non_null = ~frame.geometry.isna()
        invalid = int((~frame.geometry.is_valid & non_null).sum())
        empty = int((frame.geometry.isna() | frame.geometry.is_empty).sum())
        geometry_keys = frame.geometry.loc[non_null].map(lambda geometry: geometry.wkb_hex)
        duplicate_geometries = int(geometry_keys.duplicated().sum())
        geometry_types = sorted(set(str(value) for value in frame.geom_type.dropna()))
    suppressed = _suppression_count(values, suppression_fields)
    warnings: list[str] = []
    if missing_fraction:
        warnings.append("artifact contains missing attribute values")
    if duplicates:
        warnings.append("artifact contains duplicate records")
    if invalid:
        warnings.append("artifact contains invalid geometries")
    if empty:
        warnings.append("artifact contains empty geometries")
    if suppressed:
        warnings.append("artifact contains suppressed records")
    quality = QualitySummary(
        missing_fraction=missing_fraction,
        invalid_geometry_count=invalid,
        duplicate_count=duplicates,
        suppressed_count=suppressed,
        warnings=tuple(warnings),
    )
    return (
        {
            "columns": [
                {"name": str(column), "dtype": str(dtype)}
                for column, dtype in zip(values.columns, values.dtypes, strict=True)
            ],
            "geometry_types": geometry_types,
            "missing_fraction": missing_fraction,
            "duplicate_count": duplicates,
            "duplicate_geometry_count": duplicate_geometries,
            "invalid_geometry_count": invalid,
            "empty_geometry_count": empty,
            "suppressed_count": suppressed,
            "warnings": warnings,
        },
        quality,
    )


class ProfileArtifactTool:
    """Measure structural, spatial, and disclosure quality without mutating evidence."""

    version = "1.0.0"
    spec = ToolSpec(
        name="profile_artifact",
        version=version,
        description=(
            "Profile an immutable vector, raster, table, graph, or matrix artifact for schema, "
            "extent, missingness, duplicates, invalid geometry, and suppression markers."
        ),
        input_schema=ProfileArtifactInput.model_json_schema(),
        output_schema=ProfileArtifactOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "profiling", "offline"}),
        artifact_tags=frozenset(
            {
                ArtifactKind.VECTOR,
                ArtifactKind.RASTER,
                ArtifactKind.TABLE,
                ArtifactKind.GRAPH,
                ArtifactKind.MATRIX,
            }
        ),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=10, p95_ms=1_000),
        smoke_input={"artifact_id": MISSING_ARTIFACT_ID},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = ProfileArtifactInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        reference = artifact_ref(context, request.artifact_id)
        empty_geometry_count = 0
        if reference.kind in {ArtifactKind.VECTOR, ArtifactKind.TABLE}:
            frame = read_frame(context, reference)
            profile, quality = _frame_profile(frame, request.suppression_fields)
            empty_count = profile["empty_geometry_count"]
            if not isinstance(empty_count, int):
                raise AssertionError("frame profile returned a non-integer empty count")
            empty_geometry_count = empty_count
            if request.temporal_fields:
                missing_temporal = sorted(set(request.temporal_fields) - set(frame.columns))
                if missing_temporal:
                    invalid(f"missing temporal fields: {', '.join(missing_temporal)}")
                try:
                    temporal_values = pd.concat(
                        [
                            pd.to_datetime(frame[field], utc=True, errors="raise")
                            for field in request.temporal_fields
                        ]
                    ).dropna()
                except (TypeError, ValueError) as error:
                    invalid(f"temporal fields could not be parsed: {error}")
                profile["observed_temporal_extent"] = (
                    None
                    if temporal_values.empty
                    else {
                        "start": temporal_values.min().isoformat(),
                        "end": temporal_values.max().isoformat(),
                    }
                )
        elif reference.kind is ArtifactKind.RASTER:
            raster = read_raster(context.artifact_store, reference)
            values = raster.values.astype(np.float64, copy=False)
            missing = np.isnan(values)
            if raster.nodata is not None:
                missing |= values == raster.nodata
            missing_fraction = float(missing.mean())
            warnings = ["raster contains nodata cells"] if missing_fraction else []
            quality = QualitySummary(missing_fraction=missing_fraction, warnings=tuple(warnings))
            profile = {
                "shape": list(values.shape),
                "dtype": str(raster.values.dtype),
                "band_names": list(raster.band_names),
                "missing_fraction": missing_fraction,
                "duplicate_count": 0,
                "invalid_geometry_count": 0,
                "suppressed_count": 0,
                "warnings": warnings,
            }
        elif reference.kind is ArtifactKind.GRAPH:
            graph = read_graph(context.artifact_store, reference)
            attributes = [value for _, data in graph.nodes(data=True) for value in data.values()]
            missing_fraction = (
                0.0
                if not attributes
                else sum(value is None for value in attributes) / len(attributes)
            )
            quality = QualitySummary(missing_fraction=missing_fraction)
            profile = {
                "directed": graph.is_directed(),
                "multigraph": graph.is_multigraph(),
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "missing_fraction": missing_fraction,
                "duplicate_count": 0,
                "invalid_geometry_count": 0,
                "suppressed_count": 0,
                "warnings": [],
            }
        elif reference.kind is ArtifactKind.MATRIX:
            matrix = read_matrix(context.artifact_store, reference)
            missing_fraction = float(np.isnan(matrix.values).mean())
            unreachable = int(np.isinf(matrix.values).sum())
            warnings = ["matrix contains unreachable entries"] if unreachable else []
            quality = QualitySummary(missing_fraction=missing_fraction, warnings=tuple(warnings))
            profile = {
                "shape": list(matrix.values.shape),
                "dtype": str(matrix.values.dtype),
                "missing_fraction": missing_fraction,
                "unreachable_count": unreachable,
                "duplicate_count": 0,
                "invalid_geometry_count": 0,
                "suppressed_count": 0,
                "warnings": warnings,
            }
        else:
            profile = {
                "missing_fraction": reference.quality.missing_fraction or 0.0,
                "duplicate_count": reference.quality.duplicate_count,
                "invalid_geometry_count": reference.quality.invalid_geometry_count,
                "suppressed_count": reference.quality.suppressed_count,
                "warnings": list(reference.quality.warnings),
            }
            quality = reference.quality
        report = {
            "artifact_id": reference.id,
            "kind": reference.kind.value,
            "schema": reference.data_schema,
            "spatial_extent": (
                reference.spatial_extent.model_dump(mode="json")
                if reference.spatial_extent is not None
                else None
            ),
            "temporal_extent": (
                reference.temporal_extent.model_dump(mode="json")
                if reference.temporal_extent is not None
                else None
            ),
            **profile,
        }
        report_ref = put_json(
            context.artifact_store,
            report,
            kind=ArtifactKind.JSON_SPECIFICATION,
            units="unitless",
            provenance=child_provenance(self.spec.name, self.version, [reference], request),
            data_schema={"type": "artifact_profile", "version": self.version},
            quality=quality,
        )
        output = ProfileArtifactOutput(
            profile_artifact_id=report_ref.id,
            row_count=reference.row_count,
            cell_count=reference.cell_count,
            edge_count=reference.edge_count,
            missing_fraction=quality.missing_fraction or 0.0,
            duplicate_count=quality.duplicate_count,
            invalid_geometry_count=quality.invalid_geometry_count,
            empty_geometry_count=empty_geometry_count,
            suppressed_count=quality.suppressed_count,
            warning_count=len(quality.warnings),
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "artifact_id": reference.id,
                "kind": reference.kind.value,
                "warnings": list(quality.warnings),
            },
            artifacts=(report_ref,),
            metrics=output.model_dump(mode="json"),
        )
