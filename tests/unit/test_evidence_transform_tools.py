from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import Point, Polygon, box

from oasis.artifacts import (
    ArtifactProvenance,
    LocalArtifactStore,
    RasterData,
    put_raster,
    put_table,
    put_vector,
    read_json,
    read_table,
    read_vector,
)
from oasis.schemas import ToolResult, ToolResultStatus
from oasis.tools import CancellationToken, ToolContext, invoke_tool
from oasis.tools.evidence.health import DeriveHealthMeasureTool
from oasis.tools.evidence.normalize import NormalizeArtifactTool
from oasis.tools.evidence.overlay import OverlayReduceTool
from oasis.tools.evidence.profile import ProfileArtifactTool


def provenance(name: str) -> ArtifactProvenance:
    return ArtifactProvenance(source_uri=f"fixture://{name}", license="CC0-1.0")


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        run_id="evidence-transform-test",
        artifact_store=LocalArtifactStore(tmp_path),
        deadline_monotonic=time.monotonic() + 10,
        cancellation=CancellationToken(),
        seed=0,
    )


async def complete(tool: object, arguments: dict[str, object], ctx: ToolContext) -> ToolResult:
    result = await invoke_tool(tool, arguments, ctx)  # type: ignore[arg-type]
    assert result.status is ToolResultStatus.COMPLETE, result
    return result


@pytest.mark.asyncio
async def test_profile_reports_invalid_empty_duplicate_and_suppressed_features(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    frame = gpd.GeoDataFrame(
        {
            "value": [1.0, 1.0, 2.0, 3.0],
            "suppression": ["suppressed"] * 4,
            "date": ["2025-01-01"] * 4,
        },
        geometry=[Point(2, 2), Point(2, 2), invalid, Point()],
        crs="EPSG:4326",
    )
    reference = put_vector(
        ctx.artifact_store, frame, units="count", provenance=provenance("profile")
    )

    result = await complete(
        ProfileArtifactTool(),
        {
            "artifact_id": reference.id,
            "suppression_fields": ["suppression"],
            "temporal_fields": ["date"],
        },
        ctx,
    )

    assert result.metrics["invalid_geometry_count"] == 1
    assert result.metrics["empty_geometry_count"] == 1
    assert result.metrics["duplicate_count"] == 1
    assert result.metrics["suppressed_count"] == 4
    assert result.artifacts[0].lineage.parent_ids == (reference.id,)
    report = read_json(ctx.artifact_store, result.artifacts[0])
    assert isinstance(report, dict)
    assert report["duplicate_geometry_count"] == 1
    assert report["observed_temporal_extent"]["start"] == "2025-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_normalization_reprojects_clips_repairs_ids_units_and_retains_dimensions(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    frame = gpd.GeoDataFrame(
        {
            "raw_id": ["same", "same"],
            "need": [1.0, 2.0],
            "group": ["a", "b"],
            "suppressed": [False, True],
        },
        geometry=[Point(-73.99, 40.75), Point(-73.98, 40.76)],
        crs="EPSG:4326",
    )
    clip = gpd.GeoDataFrame(
        {"id": ["clip"]}, geometry=[box(-74.1, 40.7, -73.9, 40.8)], crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    source_ref = put_vector(
        ctx.artifact_store, frame, units="persons", provenance=provenance("normalize")
    )
    clip_ref = put_vector(ctx.artifact_store, clip, units="meters", provenance=provenance("clip"))

    result = await complete(
        NormalizeArtifactTool(),
        {
            "artifact_id": source_ref.id,
            "target_crs": "EPSG:3857",
            "clip_artifact_id": clip_ref.id,
            "source_id_field": "raw_id",
            "output_id_field": "id",
            "value_fields": ["need"],
            "unit_scale": 1000,
            "output_units": "people_per_1000",
        },
        ctx,
    )
    normalized = read_vector(ctx.artifact_store, str(result.metrics["artifact_id"]))

    assert list(normalized["id"]) == ["same", "same-2"]
    assert list(normalized["need"]) == [1_000.0, 2_000.0]
    assert list(normalized["group"]) == ["a", "b"]
    assert list(normalized["suppressed"]) == [False, True]
    assert str(normalized.crs) == "EPSG:3857"
    assert result.metrics["duplicate_id_count"] == 1
    assert set(result.artifacts[0].lineage.parent_ids) == {source_ref.id, clip_ref.id}


@pytest.mark.asyncio
async def test_normalization_repairs_invalid_geometry(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    invalid = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    source = gpd.GeoDataFrame({"id": ["bad"]}, geometry=[invalid], crs="EPSG:3857")
    source_ref = put_vector(
        ctx.artifact_store, source, units="meters", provenance=provenance("invalid")
    )

    result = await complete(
        NormalizeArtifactTool(), {"artifact_id": source_ref.id, "output_id_field": "id"}, ctx
    )
    repaired = read_vector(ctx.artifact_store, str(result.metrics["artifact_id"]))

    assert repaired.geometry.is_valid.all()
    assert result.metrics["repaired_geometry_count"] == 1


@pytest.mark.asyncio
async def test_overlay_operations_cover_join_zonal_nearest_and_raster_sampling(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    zones = gpd.GeoDataFrame(
        {"zone_id": ["z1", "z2"]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"point_id": ["p1", "p2", "p3"], "need": [2.0, 3.0, 5.0]},
        geometry=[Point(2, 2), Point(10, 5), Point(18, 5)],
        crs="EPSG:3857",
    )
    zone_ref = put_vector(ctx.artifact_store, zones, units="meters", provenance=provenance("zones"))
    point_ref = put_vector(
        ctx.artifact_store, points, units="persons", provenance=provenance("points")
    )

    zonal_result = await complete(
        OverlayReduceTool(),
        {
            "left_artifact_id": zone_ref.id,
            "right_artifact_id": point_ref.id,
            "operation": "zonal_aggregation",
            "predicate": "intersects",
            "reducer": "sum",
            "value_fields": ["need"],
            "left_id_field": "zone_id",
            "output_prefix": "demand",
        },
        ctx,
    )
    zonal = read_vector(ctx.artifact_store, str(zonal_result.metrics["artifact_id"]))
    assert list(zonal["demand_need_sum"]) == [5.0, 8.0]

    join_result = await complete(
        OverlayReduceTool(),
        {
            "left_artifact_id": point_ref.id,
            "right_artifact_id": zone_ref.id,
            "operation": "spatial_join",
            "predicate": "within",
            "left_id_field": "point_id",
            "right_id_field": "zone_id",
        },
        ctx,
    )
    joined = read_vector(ctx.artifact_store, str(join_result.metrics["artifact_id"]))
    assert set(joined["overlay_zone_id"]) == {"z1", "z2"}
    assert "p2" not in set(joined["point_id"])

    nearest_result = await complete(
        OverlayReduceTool(),
        {
            "left_artifact_id": point_ref.id,
            "right_artifact_id": point_ref.id,
            "operation": "nearest_feature",
            "right_id_field": "point_id",
            "value_fields": ["need"],
        },
        ctx,
    )
    nearest = read_vector(ctx.artifact_store, str(nearest_result.metrics["artifact_id"]))
    assert list(nearest["overlay_distance"]) == [0.0, 0.0, 0.0]

    raster = RasterData(
        values=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        transform=Affine.translation(0, 20) @ Affine.scale(10, -10),
        crs="EPSG:3857",
        band_names=("exposure",),
    )
    raster_ref = put_raster(
        ctx.artifact_store, raster, units="index", provenance=provenance("raster")
    )
    sample_result = await complete(
        OverlayReduceTool(),
        {
            "left_artifact_id": point_ref.id,
            "right_artifact_id": raster_ref.id,
            "operation": "raster_sampling",
            "left_id_field": "point_id",
        },
        ctx,
    )
    sampled = read_vector(ctx.artifact_store, str(sample_result.metrics["artifact_id"]))
    assert list(sampled["overlay_band_1"]) == [3.0, 4.0, 4.0]


@pytest.mark.asyncio
async def test_health_rates_uncertainty_age_standardization_and_suppression(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    table = pd.DataFrame(
        {
            "id": ["a", "b"],
            "cases": [10, 2],
            "population": [100, 50],
            "young_cases": [6, 1],
            "young_population": [60, 25],
            "old_cases": [4, 1],
            "old_population": [40, 25],
            "group": ["urban", "rural"],
            "suppressed": [False, True],
        }
    )
    source_ref = put_table(
        ctx.artifact_store,
        table,
        crs=None,
        units="count",
        provenance=provenance("health"),
    )

    rate_result = await complete(
        DeriveHealthMeasureTool(),
        {
            "artifact_id": source_ref.id,
            "kind": "rate",
            "numerator_field": "cases",
            "denominator_field": "population",
            "output_field": "case_rate",
            "scale": 1_000,
            "suppression_fields": ["suppressed"],
        },
        ctx,
    )
    rates = read_table(ctx.artifact_store, str(rate_result.metrics["artifact_id"]))
    assert rates.loc[0, "case_rate"] == 100.0
    assert np.isnan(rates.loc[1, "case_rate"])
    assert rates.loc[1, "case_rate_suppressed"]
    assert list(rates["group"]) == ["urban", "rural"]
    assert "case_rate_standard_error" in rates

    standardized_result = await complete(
        DeriveHealthMeasureTool(),
        {
            "artifact_id": source_ref.id,
            "kind": "direct_age_standardized_rate",
            "output_field": "standardized_rate",
            "scale": 1_000,
            "age_strata": [
                {
                    "numerator_field": "young_cases",
                    "denominator_field": "young_population",
                    "standard_weight": 0.5,
                },
                {
                    "numerator_field": "old_cases",
                    "denominator_field": "old_population",
                    "standard_weight": 0.5,
                },
            ],
            "confidence_level": None,
        },
        ctx,
    )
    standardized = read_table(ctx.artifact_store, str(standardized_result.metrics["artifact_id"]))
    assert standardized.loc[0, "standardized_rate"] == pytest.approx(100.0)
    assert standardized.loc[1, "standardized_rate"] == pytest.approx(40.0)
