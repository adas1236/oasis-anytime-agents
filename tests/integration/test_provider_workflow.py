from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import httpx
import pytest
from shapely.geometry import Point

from oasis.artifacts import (
    ArtifactProvenance,
    LocalArtifactStore,
    put_vector,
    read_json,
    read_matrix,
    read_table,
)
from oasis.providers import (
    HttpPolicy,
    HttpSourceSnapshotProvider,
    MemorySnapshotCache,
    NominatimPlaceResolver,
    OsrmRoutingMatrixProvider,
    ResilientHttpClient,
)
from oasis.schemas import ArtifactKind, ToolResultStatus
from oasis.tools import CancellationToken, ToolContext, create_tool_registry, invoke_tool
from oasis.tools.providers import (
    PLACE_PROVIDER,
    ROUTING_PROVIDER,
    SNAPSHOT_CACHE,
    SOURCE_PROVIDER,
)


@pytest.mark.asyncio
async def test_frozen_http_fixture_runs_the_live_provider_tool_workflow(tmp_path: Path) -> None:
    """The opt-in live command's tool chain remains reproducible with frozen HTTP responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoder.test":
            return httpx.Response(
                200,
                request=request,
                json=[
                    {
                        "osm_type": "relation",
                        "osm_id": 1,
                        "display_name": "Fixture City",
                        "lon": "-71.1",
                        "lat": "42.1",
                        "boundingbox": ["42", "43", "-72", "-71"],
                        "importance": 0.9,
                    }
                ],
            )
        if request.url.host == "data.test":
            return httpx.Response(
                200,
                request=request,
                content=b"id,need,group\na,10,A\nb,20,B\n",
                headers={"Content-Type": "text/csv", "ETag": '"fixture"'},
            )
        if request.url.host == "router.test":
            return httpx.Response(
                200,
                request=request,
                json={"code": "Ok", "durations": [[0, 60], [65, 0]]},
            )
        raise AssertionError(f"unexpected fixture request: {request.url}")

    store = LocalArtifactStore(tmp_path / "artifacts")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        http = ResilientHttpClient(
            client,
            policy=HttpPolicy(
                user_agent="oasis-frozen-workflow/1.0",
                timeout_seconds=1,
                max_attempts=1,
            ),
        )
        tool_context = ToolContext(
            run_id="frozen-provider-workflow",
            artifact_store=store,
            deadline_monotonic=time.monotonic() + 5,
            cancellation=CancellationToken(),
            seed=0,
            providers={
                PLACE_PROVIDER: NominatimPlaceResolver(http, endpoint="https://geocoder.test"),
                SOURCE_PROVIDER: HttpSourceSnapshotProvider(http),
                ROUTING_PROVIDER: OsrmRoutingMatrixProvider(http, endpoint="https://router.test"),
            },
            resources={SNAPSHOT_CACHE: MemorySnapshotCache()},
        )
        registry = create_tool_registry(discover_entry_points=False)
        place = await invoke_tool(
            registry.get("resolve_area"), {"query": "Fixture City"}, tool_context
        )
        snapshot = await invoke_tool(
            registry.get("snapshot_source"),
            {
                "url": "https://data.test/health.csv",
                "format": "csv",
                "license": "CC0",
                "units": "people",
            },
            tool_context,
        )
        points = put_vector(
            store,
            gpd.GeoDataFrame(
                {"id": ["one", "two"]},
                geometry=[Point(-71.1, 42.1), Point(-71.2, 42.2)],
                crs="EPSG:4326",
            ),
            units="degrees",
            provenance=ArtifactProvenance(source_uri="fixture://route-points", license="CC0"),
        )
        route = await invoke_tool(
            registry.get("travel_matrix"),
            {
                "origins_artifact_id": points.id,
                "destinations_artifact_id": points.id,
                "strategy": "routed_provider",
                "output_units": "seconds",
                "routing_profile": "driving",
                "route_annotation": "duration",
            },
            tool_context,
        )

    assert place.status is ToolResultStatus.COMPLETE
    resolution = read_json(store, place.artifacts[0])
    assert resolution["candidates"][0]["display_name"] == "Fixture City"
    assert snapshot.status is ToolResultStatus.COMPLETE
    assert snapshot.artifacts[0].kind is ArtifactKind.TABLE
    assert list(read_table(store, snapshot.artifacts[0]).columns) == ["id", "need", "group"]
    assert route.status is ToolResultStatus.COMPLETE
    matrix = read_matrix(store, route.artifacts[0])
    assert matrix.row_ids == ("one", "two")
    assert matrix.column_ids == ("one", "two")
    assert matrix.values.tolist() == [[0, 60], [65, 0]]
