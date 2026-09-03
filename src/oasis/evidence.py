"""Public offline evidence-plane example built entirely from deterministic local artifacts."""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
from pydantic import BaseModel, ConfigDict
from shapely.geometry import Point

from oasis.artifacts import ArtifactProvenance, LocalArtifactStore, put_vector
from oasis.schemas import ToolResult, ToolResultStatus
from oasis.tools import CancellationToken, ToolContext, create_tool_registry, invoke_tool


class EvidenceDemoResult(BaseModel):
    """Artifact identities produced by the frozen Phase 3 example pipeline."""

    model_config = ConfigDict(frozen=True)

    demand_source_artifact_id: str
    candidate_source_artifact_id: str
    demand_artifact_id: str
    demand_spec_artifact_id: str
    candidate_artifact_id: str
    candidate_spec_artifact_id: str
    access_matrix_artifact_id: str
    service_matrix_artifact_id: str


def _fixture_provenance(name: str) -> ArtifactProvenance:
    return ArtifactProvenance(
        source_uri=f"fixture://phase3/{name}",
        source_provider="oasis-synthetic",
        source_version="1.0.0",
        license="CC0-1.0",
    )


def create_demo_sources(store: LocalArtifactStore) -> tuple[str, str]:
    """Publish tiny fully synthetic demand and facility point layers."""

    demand = gpd.GeoDataFrame(
        {
            "demand_id": ["d1", "d2", "d3"],
            "population": [120.0, 80.0, 100.0],
            "older_adult": [0, 1, 1],
            "suppressed": [False, False, False],
        },
        geometry=[Point(0, 0), Point(1_000, 0), Point(2_000, 0)],
        crs="EPSG:3857",
    )
    candidates = gpd.GeoDataFrame(
        {
            "site_id": ["s1", "s2"],
            "opening_cost": [1.0, 1.5],
            "capacity": [200.0, 250.0],
        },
        geometry=[Point(0, 0), Point(2_000, 0)],
        crs="EPSG:3857",
    )
    demand_ref = put_vector(
        store,
        demand,
        units="persons",
        provenance=_fixture_provenance("demand"),
    )
    candidate_ref = put_vector(
        store,
        candidates,
        units="meters",
        provenance=_fixture_provenance("candidates"),
    )
    return demand_ref.id, candidate_ref.id


async def _invoke(
    name: str,
    arguments: dict[str, object],
    *,
    store: LocalArtifactStore,
    context: ToolContext,
) -> ToolResult:
    registry = create_tool_registry(discover_entry_points=False)
    result = await invoke_tool(registry.get(name), arguments, context)
    if result.status is not ToolResultStatus.COMPLETE:
        detail = result.error.message if result.error is not None else str(result.summary)
        raise RuntimeError(f"evidence demo tool {name!r} failed: {detail}")
    return result


async def run_evidence_demo(artifact_root: str | Path) -> EvidenceDemoResult:
    """Turn frozen synthetic layers into canonical demand, candidate, and service evidence."""

    store = LocalArtifactStore(artifact_root)
    demand_source_id, candidate_source_id = create_demo_sources(store)
    context = ToolContext(
        run_id="phase3-evidence-demo",
        artifact_store=store,
        deadline_monotonic=time.monotonic() + 30,
        cancellation=CancellationToken(),
        seed=0,
    )
    demand = await _invoke(
        "build_demand",
        {
            "artifact_id": demand_source_id,
            "location_id_field": "demand_id",
            "need_fields": ["population"],
            "group_fields": ["older_adult"],
            "suppression_fields": ["suppressed"],
            "missing_data_policy": "error",
            "spatial_resolution": "synthetic points",
        },
        store=store,
        context=context,
    )
    candidates = await _invoke(
        "build_candidates",
        {
            "mode": "supplied",
            "artifact_id": candidate_source_id,
            "candidate_id_field": "site_id",
            "opening_cost_field": "opening_cost",
            "capacity_field": "capacity",
        },
        store=store,
        context=context,
    )
    access = await _invoke(
        "travel_matrix",
        {
            "origins_artifact_id": demand.metrics["demand_artifact_id"],
            "destinations_artifact_id": candidates.metrics["candidate_artifact_id"],
            "origin_id_field": "demand_id",
            "destination_id_field": "site_id",
            "strategy": "euclidean",
            "output_units": "meters",
        },
        store=store,
        context=context,
    )
    service = await _invoke(
        "service_matrix",
        {
            "access_matrix_artifact_id": access.metrics["artifact_id"],
            "strategy": "binary_threshold",
            "threshold": 1_000,
        },
        store=store,
        context=context,
    )
    return EvidenceDemoResult(
        demand_source_artifact_id=demand_source_id,
        candidate_source_artifact_id=candidate_source_id,
        demand_artifact_id=str(demand.metrics["demand_artifact_id"]),
        demand_spec_artifact_id=str(demand.metrics["demand_spec_artifact_id"]),
        candidate_artifact_id=str(candidates.metrics["candidate_artifact_id"]),
        candidate_spec_artifact_id=str(candidates.metrics["candidate_spec_artifact_id"]),
        access_matrix_artifact_id=str(access.metrics["artifact_id"]),
        service_matrix_artifact_id=str(service.metrics["artifact_id"]),
    )
