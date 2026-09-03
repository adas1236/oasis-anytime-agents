"""Closed spatial overlay, sampling, nearest, and zonal reduction operations."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rasterio.features import geometry_mask
from rasterio.transform import rowcol

from oasis.artifacts import put_vector, read_raster, read_vector
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
    require_fields,
    require_kind,
)
from oasis.tools.protocols import ToolContext


class OverlayOperation(StrEnum):
    SPATIAL_JOIN = "spatial_join"
    ZONAL_AGGREGATION = "zonal_aggregation"
    NEAREST_FEATURE = "nearest_feature"
    RASTER_SAMPLING = "raster_sampling"


class SpatialPredicate(StrEnum):
    INTERSECTS = "intersects"
    WITHIN = "within"
    CONTAINS = "contains"
    TOUCHES = "touches"
    COVERS = "covers"


class Reducer(StrEnum):
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    FIRST = "first"


class OverlayReduceInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    left_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    right_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    operation: OverlayOperation
    predicate: SpatialPredicate = SpatialPredicate.INTERSECTS
    reducer: Reducer = Reducer.SUM
    value_fields: tuple[str, ...] = ()
    left_id_field: str = "id"
    right_id_field: str = "id"
    output_prefix: str = "overlay"
    max_distance: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def operation_arguments_match(self) -> OverlayReduceInput:
        if self.operation is OverlayOperation.ZONAL_AGGREGATION:
            if self.reducer is not Reducer.COUNT and not self.value_fields:
                raise ValueError("zonal aggregation requires value_fields unless reducer=count")
        elif self.operation is OverlayOperation.RASTER_SAMPLING:
            if self.value_fields:
                raise ValueError("raster sampling names output fields from raster bands")
        return self


class OverlayReduceOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    row_count: int
    matched_count: int
    unmatched_count: int
    operation: OverlayOperation
    reducer: Reducer


def _predicate(zone: object, feature: object, predicate: SpatialPredicate) -> bool:
    if predicate is SpatialPredicate.INTERSECTS:
        return bool(feature.intersects(zone))  # type: ignore[attr-defined]
    if predicate is SpatialPredicate.WITHIN:
        return bool(feature.within(zone))  # type: ignore[attr-defined]
    if predicate is SpatialPredicate.CONTAINS:
        return bool(zone.contains(feature))  # type: ignore[attr-defined]
    if predicate is SpatialPredicate.TOUCHES:
        return bool(feature.touches(zone))  # type: ignore[attr-defined]
    return bool(zone.covers(feature))  # type: ignore[attr-defined]


def _left_predicate(left: object, right: object, predicate: SpatialPredicate) -> bool:
    """Apply GeoPandas-style predicates from the left geometry to the right geometry."""

    return bool(getattr(left, predicate.value)(right))


def _reduce(values: pd.Series, reducer: Reducer) -> float | int | str | None:
    available = values.dropna()
    if reducer is Reducer.COUNT:
        return len(available)
    if available.empty:
        return None
    if reducer is Reducer.FIRST:
        value = available.iloc[0]
        value = value.item() if isinstance(value, np.generic) else value
        return value if isinstance(value, (str, int, float)) else str(value)
    numeric = pd.to_numeric(available, errors="raise")
    if reducer is Reducer.SUM:
        return float(numeric.sum())
    if reducer is Reducer.MEAN:
        return float(numeric.mean())
    if reducer is Reducer.MIN:
        return float(numeric.min())
    return float(numeric.max())


def _spatial_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    request: OverlayReduceInput,
) -> tuple[gpd.GeoDataFrame, int]:
    require_fields(left, [request.left_id_field])
    require_fields(right, [request.right_id_field, *request.value_fields])
    rows: list[dict[str, object]] = []
    matched_left: set[int] = set()
    right_columns = [
        str(column)
        for column in right.columns
        if column != right.geometry.name and column != request.right_id_field
    ]
    for left_position, (_, left_row) in enumerate(left.iterrows()):
        for _, right_row in right.iterrows():
            if _left_predicate(left_row.geometry, right_row.geometry, request.predicate):
                record = {str(column): left_row[column] for column in left.columns}
                record[f"{request.output_prefix}_{request.right_id_field}"] = right_row[
                    request.right_id_field
                ]
                record.update(
                    {
                        f"{request.output_prefix}_{column}": right_row[column]
                        for column in right_columns
                    }
                )
                rows.append(record)
                matched_left.add(left_position)
    if not rows:
        return left.iloc[0:0].copy(), 0
    return gpd.GeoDataFrame(rows, geometry=left.geometry.name, crs=left.crs), len(matched_left)


def _zonal_vector(
    zones: gpd.GeoDataFrame,
    features: gpd.GeoDataFrame,
    request: OverlayReduceInput,
) -> tuple[gpd.GeoDataFrame, int]:
    require_fields(zones, [request.left_id_field])
    require_fields(features, request.value_fields)
    output = zones.copy()
    matched = 0
    aggregated: dict[str, list[object]] = {
        f"{request.output_prefix}_count"
        if request.reducer is Reducer.COUNT
        else (f"{request.output_prefix}_{field}_{request.reducer.value}"): []
        for field in (request.value_fields or ("features",))
    }
    for zone in zones.geometry:
        mask = features.geometry.map(
            lambda feature, current_zone=zone: (
                False
                if feature is None
                or feature.is_empty
                or current_zone is None
                or current_zone.is_empty
                else _predicate(current_zone, feature, request.predicate)
            )
        )
        selected = features.loc[mask]
        if not selected.empty:
            matched += 1
        if request.reducer is Reducer.COUNT:
            aggregated[f"{request.output_prefix}_count"].append(len(selected))
        else:
            for field in request.value_fields:
                try:
                    value = _reduce(selected[field], request.reducer)
                except (TypeError, ValueError) as error:
                    invalid(f"cannot apply {request.reducer.value} to field {field!r}: {error}")
                aggregated[f"{request.output_prefix}_{field}_{request.reducer.value}"].append(value)
    for field, values in aggregated.items():
        output[field] = values
    return output, matched


def _zonal_raster(
    zones: gpd.GeoDataFrame,
    values: np.ndarray,
    transform: object,
    nodata: float | int | None,
    request: OverlayReduceInput,
) -> tuple[gpd.GeoDataFrame, int]:
    output = zones.copy()
    bands = values[np.newaxis, ...] if values.ndim == 2 else values
    aggregated: dict[str, list[object]] = {}
    matched = 0
    for band_index in range(bands.shape[0]):
        aggregated[f"{request.output_prefix}_band_{band_index + 1}_{request.reducer.value}"] = []
    for zone in zones.geometry:
        inside = geometry_mask(
            [zone.__geo_interface__],
            out_shape=bands.shape[-2:],
            transform=transform,
            invert=True,
        )
        if inside.any():
            matched += 1
        for band_index, band in enumerate(bands):
            selected = band[inside].astype(np.float64, copy=False)
            selected = selected[~np.isnan(selected)]
            if nodata is not None:
                selected = selected[selected != nodata]
            series = pd.Series(selected)
            value = (
                len(series)
                if request.reducer is Reducer.COUNT
                else _reduce(series, request.reducer)
            )
            aggregated[
                f"{request.output_prefix}_band_{band_index + 1}_{request.reducer.value}"
            ].append(value)
    for field, aggregated_values in aggregated.items():
        output[field] = aggregated_values
    return output, matched


def _nearest(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    request: OverlayReduceInput,
) -> tuple[gpd.GeoDataFrame, int]:
    require_fields(right, [request.right_id_field, *request.value_fields])
    if left.crs is not None and left.crs.is_geographic:
        invalid("nearest_feature requires a projected CRS so distance units are explicit")
    output = left.copy()
    nearest_ids: list[object] = []
    distances: list[float] = []
    copied: dict[str, list[object]] = {field: [] for field in request.value_fields}
    matched = 0
    for geometry in left.geometry:
        distance_series = right.geometry.distance(geometry)
        if distance_series.empty:
            nearest_ids.append(None)
            distances.append(float("nan"))
            for field in copied:
                copied[field].append(None)
            continue
        position = int(np.argmin(distance_series.to_numpy()))
        distance = float(distance_series.iloc[position])
        if request.max_distance is not None and distance > request.max_distance:
            nearest_ids.append(None)
            distances.append(float("nan"))
            for field in copied:
                copied[field].append(None)
            continue
        row = right.iloc[position]
        nearest_ids.append(row[request.right_id_field])
        distances.append(distance)
        for field in copied:
            copied[field].append(row[field])
        matched += 1
    output[f"{request.output_prefix}_{request.right_id_field}"] = nearest_ids
    output[f"{request.output_prefix}_distance"] = distances
    for field, copied_values in copied.items():
        output[f"{request.output_prefix}_{field}"] = copied_values
    return output, matched


def _raster_sample(
    points: gpd.GeoDataFrame,
    values: np.ndarray,
    transform: object,
    nodata: float | int | None,
    request: OverlayReduceInput,
) -> tuple[gpd.GeoDataFrame, int]:
    output = points.copy()
    bands = values[np.newaxis, ...] if values.ndim == 2 else values
    sampled: list[list[float | None]] = [[] for _ in range(bands.shape[0])]
    matched = 0
    for point in points.geometry:
        if point is None or point.is_empty or point.geom_type != "Point":
            for band_values in sampled:
                band_values.append(None)
            continue
        row, column = rowcol(transform, point.x, point.y)
        in_bounds = 0 <= row < bands.shape[1] and 0 <= column < bands.shape[2]
        current: list[float | None] = []
        for band in bands:
            value = float(band[row, column]) if in_bounds else None
            current.append(None if value is None or np.isnan(value) or value == nodata else value)
        if any(value is not None for value in current):
            matched += 1
        for band_values, value in zip(sampled, current, strict=True):
            band_values.append(value)
    for band_index, band_values in enumerate(sampled, start=1):
        output[f"{request.output_prefix}_band_{band_index}"] = band_values
    return output, matched


class OverlayReduceTool:
    """Execute a validated member of the closed spatial overlay operation set."""

    version = "1.0.0"
    spec = ToolSpec(
        name="overlay_reduce",
        version=version,
        description=(
            "Spatially join, aggregate by zone, find nearest features, or sample rasters using "
            "closed operation, predicate, and reducer enums."
        ),
        input_schema=OverlayReduceInput.model_json_schema(),
        output_schema=OverlayReduceOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "overlay", "offline"}),
        artifact_tags=frozenset({ArtifactKind.VECTOR, ArtifactKind.RASTER}),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=30, p95_ms=3_000),
        smoke_input={
            "left_artifact_id": MISSING_ARTIFACT_ID,
            "right_artifact_id": MISSING_ARTIFACT_ID,
            "operation": "spatial_join",
        },
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = OverlayReduceInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        left_ref = artifact_ref(context, request.left_artifact_id)
        right_ref = artifact_ref(context, request.right_artifact_id)
        require_kind(left_ref, {ArtifactKind.VECTOR})
        left = read_vector(context.artifact_store, left_ref)
        if left.crs is None:
            invalid("left vector has no CRS")

        if request.operation is OverlayOperation.RASTER_SAMPLING:
            require_kind(right_ref, {ArtifactKind.RASTER})
            raster = read_raster(context.artifact_store, right_ref)
            if str(left.crs) != right_ref.crs:
                left = left.to_crs(right_ref.crs)
            output, matched = _raster_sample(
                left, raster.values, raster.transform, raster.nodata, request
            )
        elif request.operation is OverlayOperation.ZONAL_AGGREGATION and right_ref.kind is (
            ArtifactKind.RASTER
        ):
            raster = read_raster(context.artifact_store, right_ref)
            if str(left.crs) != right_ref.crs:
                left = left.to_crs(right_ref.crs)
            output, matched = _zonal_raster(
                left, raster.values, raster.transform, raster.nodata, request
            )
        else:
            require_kind(right_ref, {ArtifactKind.VECTOR})
            right = read_vector(context.artifact_store, right_ref)
            if right.crs != left.crs:
                invalid("overlay inputs must have matching CRS; normalize them first")
            if request.operation is OverlayOperation.SPATIAL_JOIN:
                output, matched = _spatial_join(left, right, request)
            elif request.operation is OverlayOperation.ZONAL_AGGREGATION:
                output, matched = _zonal_vector(left, right, request)
            elif request.operation is OverlayOperation.NEAREST_FEATURE:
                output, matched = _nearest(left, right, request)
            else:
                invalid("raster_sampling requires a raster right artifact")

        output_units = (
            right_ref.units or "unitless"
            if request.operation
            in {OverlayOperation.ZONAL_AGGREGATION, OverlayOperation.RASTER_SAMPLING}
            else "mixed"
        )
        output_ref = put_vector(
            context.artifact_store,
            output.reset_index(drop=True),
            units=output_units,
            provenance=child_provenance(
                self.spec.name, self.version, [left_ref, right_ref], request
            ),
        )
        result = OverlayReduceOutput(
            artifact_id=output_ref.id,
            row_count=len(output),
            matched_count=matched,
            unmatched_count=max(0, len(left) - matched),
            operation=request.operation,
            reducer=request.reducer,
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "artifact_id": output_ref.id,
                "operation": request.operation.value,
                "matched": matched,
            },
            artifacts=(output_ref,),
            metrics=result.model_dump(mode="json"),
        )
