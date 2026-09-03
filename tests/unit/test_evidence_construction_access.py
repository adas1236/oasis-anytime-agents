from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, box

from oasis.artifacts import (
    ArtifactProvenance,
    LocalArtifactStore,
    put_graph,
    put_vector,
    read_json,
    read_matrix,
    read_vector,
)
from oasis.schemas import CandidateSpec, DemandSpec, ToolResult, ToolResultStatus
from oasis.tools import CancellationToken, ToolContext, invoke_tool
from oasis.tools.evidence.access import IsochronesTool, ServiceMatrixTool, TravelMatrixTool
from oasis.tools.evidence.construction import BuildCandidatesTool, BuildDemandTool


def provenance(name: str) -> ArtifactProvenance:
    return ArtifactProvenance(source_uri=f"fixture://{name}", license="CC0-1.0")


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        run_id="evidence-access-test",
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
async def test_build_demand_preserves_group_need_and_suppression_dimensions(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    source = gpd.GeoDataFrame(
        {
            "demand_id": ["a", "b"],
            "need": [100.0, None],
            "group": ["urban", "rural"],
            "period": [2025, 2025],
            "suppressed": [False, True],
            "unselected": [999, 999],
        },
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:3857",
    )
    source_ref = put_vector(
        ctx.artifact_store, source, units="persons", provenance=provenance("demand")
    )

    first = await complete(
        BuildDemandTool(),
        {
            "artifact_id": source_ref.id,
            "location_id_field": "demand_id",
            "need_fields": ["need"],
            "group_fields": ["group"],
            "time_fields": ["period"],
            "suppression_fields": ["suppressed"],
            "missing_data_policy": "error",
        },
        ctx,
    )
    second = await complete(
        BuildDemandTool(),
        {
            "artifact_id": source_ref.id,
            "location_id_field": "demand_id",
            "need_fields": ["need"],
            "group_fields": ["group"],
            "time_fields": ["period"],
            "suppression_fields": ["suppressed"],
            "missing_data_policy": "error",
        },
        ctx,
    )
    demand_ref_id = str(first.metrics["demand_artifact_id"])
    demand = read_vector(ctx.artifact_store, demand_ref_id)
    raw_spec = read_json(ctx.artifact_store, str(first.metrics["demand_spec_artifact_id"]))
    spec = DemandSpec.model_validate(raw_spec)

    assert list(demand["group"]) == ["urban", "rural"]
    assert list(demand["suppressed"]) == [False, True]
    assert "unselected" not in demand
    assert spec.need_fields == ("need",)
    assert spec.group_fields == ("group",)
    assert spec.suppression_fields == ("suppressed",)
    assert first.metrics["demand_artifact_id"] == second.metrics["demand_artifact_id"]
    assert first.metrics["demand_spec_artifact_id"] == second.metrics["demand_spec_artifact_id"]


@pytest.mark.asyncio
async def test_build_candidates_supports_supplied_sites_and_suitability_grid(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    supplied = gpd.GeoDataFrame(
        {
            "site_id": ["a", "b"],
            "cost": [1.0, 2.0],
            "capacity": [10.0, 20.0],
            "eligible": [True, False],
        },
        geometry=[Point(0, 0), Point(10, 0)],
        crs="EPSG:3857",
    )
    supplied_ref = put_vector(
        ctx.artifact_store, supplied, units="meters", provenance=provenance("sites")
    )
    supplied_result = await complete(
        BuildCandidatesTool(),
        {
            "mode": "supplied",
            "artifact_id": supplied_ref.id,
            "candidate_id_field": "site_id",
            "opening_cost_field": "cost",
            "capacity_field": "capacity",
            "eligibility_field": "eligible",
        },
        ctx,
    )
    supplied_spec = CandidateSpec.model_validate(
        read_json(ctx.artifact_store, str(supplied_result.metrics["candidate_spec_artifact_id"]))
    )
    assert supplied_spec.capacity_field == "capacity"
    assert supplied_result.metrics["row_count"] == 2

    suitability = gpd.GeoDataFrame(
        {"allowed": [True]}, geometry=[box(-0.1, -0.1, 1.1, 1.1)], crs="EPSG:3857"
    )
    suitability_ref = put_vector(
        ctx.artifact_store,
        suitability,
        units="meters",
        provenance=provenance("suitability"),
    )
    grid_result = await complete(
        BuildCandidatesTool(),
        {
            "mode": "grid",
            "grid_bounds": [0, 0, 2, 2],
            "grid_spacing": 1,
            "grid_crs": "EPSG:3857",
            "suitability_artifact_id": suitability_ref.id,
            "suitability_predicate": "within",
        },
        ctx,
    )
    grid = read_vector(ctx.artifact_store, str(grid_result.metrics["candidate_artifact_id"]))
    assert list(grid["id"]) == ["grid-000001", "grid-000002", "grid-000004", "grid-000005"]
    assert grid_result.metrics["filtered_count"] == 5


@pytest.mark.asyncio
async def test_euclidean_geodesic_axis_validation_and_binary_threshold_boundary(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    origins = gpd.GeoDataFrame(
        {"id": ["o1", "o2"]},
        geometry=[Point(0, 0), Point(1_000, 0)],
        crs="EPSG:3857",
    )
    destinations = gpd.GeoDataFrame(
        {"id": ["d1", "d2"]},
        geometry=[Point(1_000, 0), Point(2_000, 0)],
        crs="EPSG:3857",
    )
    origin_ref = put_vector(
        ctx.artifact_store, origins, units="meters", provenance=provenance("origins")
    )
    destination_ref = put_vector(
        ctx.artifact_store,
        destinations,
        units="meters",
        provenance=provenance("destinations"),
    )
    travel = await complete(
        TravelMatrixTool(),
        {
            "origins_artifact_id": origin_ref.id,
            "destinations_artifact_id": destination_ref.id,
            "strategy": "euclidean",
            "output_units": "meters",
        },
        ctx,
    )
    access = read_matrix(ctx.artifact_store, str(travel.metrics["artifact_id"]))
    np.testing.assert_allclose(access.values, [[1_000, 2_000], [0, 1_000]])

    service = await complete(
        ServiceMatrixTool(),
        {
            "access_matrix_artifact_id": travel.metrics["artifact_id"],
            "strategy": "binary_threshold",
            "threshold": 1_000,
        },
        ctx,
    )
    benefit = read_matrix(ctx.artifact_store, str(service.metrics["artifact_id"]))
    np.testing.assert_array_equal(benefit.values, [[1, 0], [1, 1]])
    assert set(np.unique(benefit.values)) <= {0.0, 1.0}

    geodesic_origins = gpd.GeoDataFrame({"id": ["g1"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    geodesic_destinations = gpd.GeoDataFrame(
        {"id": ["g2"]}, geometry=[Point(1, 0)], crs="EPSG:4326"
    )
    geodesic_origin_ref = put_vector(
        ctx.artifact_store,
        geodesic_origins,
        units="degrees",
        provenance=provenance("geodesic-origin"),
    )
    geodesic_destination_ref = put_vector(
        ctx.artifact_store,
        geodesic_destinations,
        units="degrees",
        provenance=provenance("geodesic-destination"),
    )
    crs_mismatch = await invoke_tool(
        TravelMatrixTool(),
        {
            "origins_artifact_id": origin_ref.id,
            "destinations_artifact_id": geodesic_destination_ref.id,
            "strategy": "euclidean",
            "output_units": "meters",
        },
        ctx,
    )
    assert crs_mismatch.status is ToolResultStatus.FAILED
    assert crs_mismatch.error is not None
    assert "matching CRS" in crs_mismatch.error.message

    geodesic = await complete(
        TravelMatrixTool(),
        {
            "origins_artifact_id": geodesic_origin_ref.id,
            "destinations_artifact_id": geodesic_destination_ref.id,
            "strategy": "geodesic",
            "output_units": "kilometers",
        },
        ctx,
    )
    assert read_matrix(ctx.artifact_store, str(geodesic.metrics["artifact_id"])).values[0, 0] == (
        pytest.approx(111.319, rel=1e-4)
    )

    reversed_axis = gpd.GeoDataFrame({"id": ["bad"]}, geometry=[Point(40, -120)], crs="EPSG:4326")
    reversed_ref = put_vector(
        ctx.artifact_store,
        reversed_axis,
        units="degrees",
        provenance=provenance("reversed"),
    )
    failed = await invoke_tool(
        TravelMatrixTool(),
        {
            "origins_artifact_id": reversed_ref.id,
            "destinations_artifact_id": geodesic_destination_ref.id,
            "strategy": "geodesic",
            "output_units": "kilometers",
        },
        ctx,
    )
    assert failed.status is ToolResultStatus.FAILED
    assert failed.error is not None
    assert failed.error.code.value == "invalid_arguments"
    assert "axis reversal" in failed.error.message


@pytest.mark.asyncio
async def test_directed_graph_travel_unreachable_isochrones_piecewise_and_decay(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path)
    graph = nx.DiGraph()
    graph.add_edge("a", "b", minutes=5.0)
    graph.add_edge("b", "c", minutes=7.0)
    graph_ref = put_graph(
        ctx.artifact_store,
        graph,
        crs="EPSG:3857",
        units="minutes",
        provenance=provenance("graph"),
    )
    points = gpd.GeoDataFrame(
        {"id": ["a", "c"], "node": ["a", "c"]},
        geometry=[Point(0, 0), Point(2, 0)],
        crs="EPSG:3857",
    )
    points_ref = put_vector(
        ctx.artifact_store, points, units="meters", provenance=provenance("graph-points")
    )
    travel = await complete(
        TravelMatrixTool(),
        {
            "origins_artifact_id": points_ref.id,
            "destinations_artifact_id": points_ref.id,
            "strategy": "graph_shortest_path",
            "output_units": "minutes",
            "graph_artifact_id": graph_ref.id,
            "origin_graph_node_field": "node",
            "destination_graph_node_field": "node",
            "graph_weight_field": "minutes",
        },
        ctx,
    )
    matrix = read_matrix(ctx.artifact_store, str(travel.metrics["artifact_id"]))
    assert matrix.values[0, 1] == 12
    assert np.isinf(matrix.values[1, 0])
    assert travel.metrics["directed"] is True
    assert travel.metrics["unreachable_count"] == 1

    isochrones = await complete(
        IsochronesTool(),
        {
            "graph_artifact_id": graph_ref.id,
            "origin_node_ids": ["a"],
            "cutoffs": [5, 12],
            "weight_field": "minutes",
        },
        ctx,
    )
    data = read_json(ctx.artifact_store, str(isochrones.metrics["artifact_id"]))
    assert isinstance(data, dict)
    assert data["polygonal_isochrones"] == "deferred"
    assert data["sets"][0]["reachable_node_ids"] == ["a", "b"]
    assert data["sets"][1]["reachable_node_ids"] == ["a", "b", "c"]

    piecewise = await complete(
        ServiceMatrixTool(),
        {
            "access_matrix_artifact_id": travel.metrics["artifact_id"],
            "strategy": "piecewise",
            "piecewise_points": [
                {"access": 0, "benefit": 1},
                {"access": 12, "benefit": 0},
            ],
        },
        ctx,
    )
    piecewise_values = read_matrix(ctx.artifact_store, str(piecewise.metrics["artifact_id"])).values
    assert piecewise_values[0, 0] == 1
    assert piecewise_values[0, 1] == 0
    assert piecewise_values[1, 0] == 0

    decay = await complete(
        ServiceMatrixTool(),
        {
            "access_matrix_artifact_id": travel.metrics["artifact_id"],
            "strategy": "exponential_decay",
            "decay_scale": 12,
        },
        ctx,
    )
    decay_values = read_matrix(ctx.artifact_store, str(decay.metrics["artifact_id"])).values
    assert decay_values[0, 1] == pytest.approx(np.exp(-1))
    assert decay_values[1, 0] == 0
