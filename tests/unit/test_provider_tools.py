from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
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
    CatalogItem,
    CatalogSearchResult,
    LocalSnapshotCache,
    MemorySnapshotCache,
    PlaceCandidate,
    PlaceResolution,
    ProviderError,
    ProviderErrorCode,
    ProviderProvenance,
    RetrievedSource,
    RouteMatrixResult,
)
from oasis.schemas import ToolResultStatus
from oasis.tools import CancellationToken, ToolContext, invoke_tool
from oasis.tools.evidence.access import TravelMatrixTool
from oasis.tools.providers import (
    CATALOG_PROVIDER,
    PLACE_PROVIDER,
    ROUTING_PROVIDER,
    SNAPSHOT_CACHE,
    SOURCE_PROVIDER,
    ResolveAreaTool,
    ResolveLocationsTool,
    SearchSourcesTool,
    SnapshotSourceTool,
)


def provenance(provider: str = "fixture") -> ProviderProvenance:
    return ProviderProvenance(
        provider=provider,
        source_uri=f"https://{provider}.test/source",
        retrieved_at=datetime.now(UTC),
        source_version="fixture-v1",
        provider_metadata={"fixture": True},
    )


def context(
    tmp_path: Path,
    *,
    providers: dict[str, object] | None = None,
    resources: dict[str, object] | None = None,
    cancellation: CancellationToken | None = None,
) -> ToolContext:
    return ToolContext(
        run_id="provider-test",
        artifact_store=LocalArtifactStore(tmp_path),
        deadline_monotonic=time.monotonic() + 5,
        cancellation=cancellation or CancellationToken(),
        seed=0,
        providers=providers or {},
        resources=resources or {},
    )


class FakePlaces:
    def __init__(self, candidates: tuple[PlaceCandidate, ...]) -> None:
        self.candidates = candidates
        self.queries: list[str] = []

    async def resolve(self, request: object, provider_context: object) -> PlaceResolution:
        del provider_context
        self.queries.append(request.query)  # type: ignore[attr-defined]
        return PlaceResolution(candidates=self.candidates, provenance=provenance("geocoder"))


@pytest.mark.asyncio
async def test_resolve_tools_preserve_ranked_ambiguity_and_no_result(tmp_path: Path) -> None:
    candidates = (
        PlaceCandidate(
            provider_id="one",
            display_name="First",
            longitude=-71,
            latitude=42,
            rank=1,
        ),
        PlaceCandidate(
            provider_id="two",
            display_name="Second",
            longitude=-72,
            latitude=41,
            rank=2,
        ),
    )
    provider = FakePlaces(candidates)
    tool_context = context(tmp_path, providers={PLACE_PROVIDER: provider})

    area = await invoke_tool(ResolveAreaTool(), {"query": "ambiguous"}, tool_context)
    locations = await invoke_tool(ResolveLocationsTool(), {"queries": ["one", "two"]}, tool_context)

    assert area.status is ToolResultStatus.AMBIGUOUS
    resolution = read_json(tool_context.artifact_store, area.artifacts[0])
    assert [item["rank"] for item in resolution["candidates"]] == [1, 2]
    assert locations.status is ToolResultStatus.AMBIGUOUS
    assert locations.metrics["ambiguous_query_count"] == 2
    assert provider.queries == ["ambiguous", "one", "two"]

    empty = await invoke_tool(
        ResolveAreaTool(),
        {"query": "missing"},
        context(tmp_path / "empty", providers={PLACE_PROVIDER: FakePlaces(())}),
    )
    assert empty.status is ToolResultStatus.FAILED
    assert empty.error is not None
    assert empty.error.code.value == "not_found"
    assert empty.artifacts[0].row_count == 0


class FakeCatalog:
    async def search(self, request: object, provider_context: object) -> CatalogSearchResult:
        del request, provider_context
        return CatalogSearchResult(
            items=(CatalogItem(id="one"), CatalogItem(id="two")),
            page_count=2,
            provenance=provenance("stac"),
        )


@pytest.mark.asyncio
async def test_catalog_tool_stores_normalized_result_as_immutable_artifact(tmp_path: Path) -> None:
    tool_context = context(tmp_path, providers={CATALOG_PROVIDER: FakeCatalog()})

    result = await invoke_tool(SearchSourcesTool(), {"limit": 10}, tool_context)

    assert result.status is ToolResultStatus.COMPLETE
    assert len(result.artifacts) == 1
    reference = result.artifacts[0]
    assert reference.source_provider == "stac"
    assert reference.provider_metadata == {"fixture": True}
    assert reference.row_count == 2


class SequenceSource:
    def __init__(self, outcomes: list[RetrievedSource | ProviderError]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def fetch(self, request: object, provider_context: object) -> RetrievedSource:
        del request, provider_context
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


def csv_source(
    content: bytes = b"id,value\na,1\nb,2\n", *, retrieved_at: datetime | None = None
) -> RetrievedSource:
    return RetrievedSource(
        content=content,
        media_type="text/csv",
        provenance=ProviderProvenance(
            provider="http",
            source_uri="https://http.test/source",
            retrieved_at=retrieved_at or datetime.now(UTC),
            source_version="fixture-v1",
            provider_metadata={"fixture": True},
        ),
    )


def snapshot_arguments(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "url": "https://data.test/health.csv",
        "format": "csv",
        "license": "CC0-1.0",
        "units": "people",
    }
    arguments.update(overrides)
    return arguments


@pytest.mark.asyncio
async def test_snapshot_cache_hit_avoids_live_request_and_preserves_schema(tmp_path: Path) -> None:
    provider = SequenceSource([csv_source()])
    cache = MemorySnapshotCache()
    tool_context = context(
        tmp_path,
        providers={SOURCE_PROVIDER: provider},
        resources={SNAPSHOT_CACHE: cache},
    )

    first = await invoke_tool(SnapshotSourceTool(), snapshot_arguments(), tool_context)
    second = await invoke_tool(SnapshotSourceTool(), snapshot_arguments(), tool_context)

    assert first.status is ToolResultStatus.COMPLETE
    assert second.status is ToolResultStatus.COMPLETE
    assert provider.calls == 1
    assert first.artifacts[0].id == second.artifacts[0].id
    assert second.metrics["cache_status"] == "hit"
    frame = read_table(tool_context.artifact_store, first.artifacts[0])
    assert frame.to_dict(orient="records") == [{"id": "a", "value": 1}, {"id": "b", "value": 2}]


@pytest.mark.asyncio
async def test_stale_snapshot_fallback_is_labeled_in_result_and_artifact(tmp_path: Path) -> None:
    provider = SequenceSource(
        [
            csv_source(),
            ProviderError(ProviderErrorCode.UNAVAILABLE, "provider unavailable", retryable=True),
        ]
    )
    cache = MemorySnapshotCache()
    tool_context = context(
        tmp_path,
        providers={SOURCE_PROVIDER: provider},
        resources={SNAPSHOT_CACHE: cache},
    )
    arguments = snapshot_arguments(fresh_for_seconds=0, max_stale_seconds=60)
    first = await invoke_tool(SnapshotSourceTool(), arguments, tool_context)
    await asyncio.sleep(0.002)

    fallback = await invoke_tool(SnapshotSourceTool(), arguments, tool_context)

    assert first.status is ToolResultStatus.COMPLETE
    assert fallback.status is ToolResultStatus.PARTIAL
    assert fallback.metrics["cache_status"] == "stale_fallback"
    assert fallback.metrics["stale"] is True
    assert fallback.artifacts[0].id != first.artifacts[0].id
    assert "stale" in " ".join(fallback.artifacts[0].quality.warnings)
    assert fallback.artifacts[0].lineage.parent_ids == (first.artifacts[0].id,)


@pytest.mark.asyncio
async def test_network_failure_without_acceptable_snapshot_is_never_fabricated(
    tmp_path: Path,
) -> None:
    provider = SequenceSource(
        [ProviderError(ProviderErrorCode.UNAVAILABLE, "provider unavailable", retryable=True)]
    )
    result = await invoke_tool(
        SnapshotSourceTool(),
        snapshot_arguments(),
        context(
            tmp_path,
            providers={SOURCE_PROVIDER: provider},
            resources={SNAPSHOT_CACHE: MemorySnapshotCache()},
        ),
    )

    assert result.status is ToolResultStatus.FAILED
    assert result.error is not None
    assert result.error.code.value == "provider_failure"
    assert result.artifacts == ()


@pytest.mark.asyncio
async def test_over_age_cached_snapshot_is_rejected_when_refresh_fails(tmp_path: Path) -> None:
    provider = SequenceSource(
        [
            csv_source(retrieved_at=datetime.now(UTC) - timedelta(hours=1)),
            ProviderError(ProviderErrorCode.UNAVAILABLE, "provider unavailable", retryable=True),
        ]
    )
    cache = MemorySnapshotCache()
    tool_context = context(
        tmp_path,
        providers={SOURCE_PROVIDER: provider},
        resources={SNAPSHOT_CACHE: cache},
    )
    arguments = snapshot_arguments(fresh_for_seconds=0, max_stale_seconds=10)
    initial = await invoke_tool(SnapshotSourceTool(), arguments, tool_context)

    rejected = await invoke_tool(SnapshotSourceTool(), arguments, tool_context)

    assert initial.status is ToolResultStatus.COMPLETE
    assert rejected.status is ToolResultStatus.FAILED
    assert rejected.artifacts == ()


@pytest.mark.asyncio
async def test_csv_field_and_spatial_filters_are_applied_before_snapshot(tmp_path: Path) -> None:
    provider = SequenceSource([csv_source(b"id,lon,lat,private\na,-71.5,41.5,x\nb,-70,44,y\n")])
    tool_context = context(
        tmp_path,
        providers={SOURCE_PROVIDER: provider},
        resources={SNAPSHOT_CACHE: MemorySnapshotCache()},
    )

    result = await invoke_tool(
        SnapshotSourceTool(),
        snapshot_arguments(
            fields=["id", "lon", "lat"],
            bounding_box={"west": -72, "south": 41, "east": -71, "north": 42},
            longitude_field="lon",
            latitude_field="lat",
        ),
        tool_context,
    )

    assert result.status is ToolResultStatus.COMPLETE
    frame = read_table(tool_context.artifact_store, result.artifacts[0])
    assert frame.to_dict(orient="records") == [{"id": "a", "lon": -71.5, "lat": 41.5}]


@pytest.mark.asyncio
async def test_live_and_frozen_provider_bytes_produce_identical_downstream_schema(
    tmp_path: Path,
) -> None:
    geojson = b"""{
      "type": "FeatureCollection",
      "features": [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-71.1, 42.1]},
        "properties": {"id": "a", "need": 10}
      }]
    }"""
    references = []
    for label in ("live", "frozen"):
        tool_context = context(
            tmp_path / label,
            providers={SOURCE_PROVIDER: SequenceSource([csv_source(geojson)])},
            resources={SNAPSHOT_CACHE: MemorySnapshotCache()},
        )
        result = await invoke_tool(
            SnapshotSourceTool(),
            snapshot_arguments(format="geojson", crs="EPSG:4326"),
            tool_context,
        )
        assert result.status is ToolResultStatus.COMPLETE
        references.append(result.artifacts[0])

    assert references[0].kind == references[1].kind
    assert references[0].crs == references[1].crs
    assert references[0].data_schema == references[1].data_schema


@pytest.mark.asyncio
async def test_malformed_snapshot_payload_is_a_typed_failure(tmp_path: Path) -> None:
    result = await invoke_tool(
        SnapshotSourceTool(),
        snapshot_arguments(),
        context(
            tmp_path,
            providers={SOURCE_PROVIDER: SequenceSource([csv_source(b'"unterminated')])},
            resources={SNAPSHOT_CACHE: MemorySnapshotCache()},
        ),
    )

    assert result.status is ToolResultStatus.FAILED
    assert result.error is not None
    assert result.error.context["provider_code"] == "malformed_response"


@pytest.mark.asyncio
async def test_snapshot_tool_rejects_credentials_without_echoing_them(tmp_path: Path) -> None:
    secret = "do-not-log-this"
    result = await invoke_tool(
        SnapshotSourceTool(),
        snapshot_arguments(url=f"https://data.test/health.csv?api_key={secret}"),
        context(
            tmp_path,
            providers={SOURCE_PROVIDER: SequenceSource([csv_source()])},
            resources={SNAPSHOT_CACHE: MemorySnapshotCache()},
        ),
    )

    assert result.status is ToolResultStatus.FAILED
    assert result.error is not None
    assert result.error.code.value == "invalid_arguments"
    assert secret not in result.model_dump_json()


class LateSource:
    async def fetch(self, request: object, provider_context: object) -> RetrievedSource:
        del request
        provider_context.cancellation.cancel("snapshot closed")  # type: ignore[attr-defined]
        return csv_source()


@pytest.mark.asyncio
async def test_late_provider_response_cannot_publish_or_advance_closed_snapshot(
    tmp_path: Path,
) -> None:
    cancellation = CancellationToken()
    cache = MemorySnapshotCache()
    tool_context = context(
        tmp_path,
        providers={SOURCE_PROVIDER: LateSource()},
        resources={SNAPSHOT_CACHE: cache},
        cancellation=cancellation,
    )

    with pytest.raises(asyncio.CancelledError):
        await SnapshotSourceTool().run(snapshot_arguments(), tool_context)

    assert not list(tmp_path.rglob("content"))  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_local_snapshot_cache_round_trips_across_instances(tmp_path: Path) -> None:
    provider = SequenceSource([csv_source()])
    cache_root = tmp_path / "cache"
    first_context = context(
        tmp_path / "artifacts",
        providers={SOURCE_PROVIDER: provider},
        resources={SNAPSHOT_CACHE: LocalSnapshotCache(cache_root)},
    )
    first = await invoke_tool(SnapshotSourceTool(), snapshot_arguments(), first_context)
    second_context = context(
        tmp_path / "artifacts",
        providers={SOURCE_PROVIDER: provider},
        resources={SNAPSHOT_CACHE: LocalSnapshotCache(cache_root)},
    )

    second = await invoke_tool(SnapshotSourceTool(), snapshot_arguments(), second_context)

    assert first.artifacts[0].id == second.artifacts[0].id
    assert provider.calls == 1


class FakeRouting:
    async def matrix(self, request: object, provider_context: object) -> RouteMatrixResult:
        del provider_context
        return RouteMatrixResult(
            values=((10.0, None), (20.0, 30.0)),
            source_ids=request.source_ids,  # type: ignore[attr-defined]
            destination_ids=request.destination_ids,  # type: ignore[attr-defined]
            units="seconds",
            provenance=provenance("osrm"),
        )


@pytest.mark.asyncio
async def test_travel_matrix_routed_strategy_uses_provider_and_records_provenance(
    tmp_path: Path,
) -> None:
    tool_context = context(tmp_path, providers={ROUTING_PROVIDER: FakeRouting()})
    origins = put_vector(
        tool_context.artifact_store,
        gpd.GeoDataFrame(
            {"id": ["o1", "o2"]},
            geometry=[Point(-71.1, 42.1), Point(-71.2, 42.2)],
            crs="EPSG:4326",
        ),
        units="degrees",
        provenance=ArtifactProvenance(source_uri="fixture://origins", license="CC0"),
    )
    destinations = put_vector(
        tool_context.artifact_store,
        gpd.GeoDataFrame(
            {"id": ["d1", "d2"]},
            geometry=[Point(-71.3, 42.3), Point(-71.4, 42.4)],
            crs="EPSG:4326",
        ),
        units="degrees",
        provenance=ArtifactProvenance(source_uri="fixture://destinations", license="CC0"),
    )

    result = await invoke_tool(
        TravelMatrixTool(),
        {
            "origins_artifact_id": origins.id,
            "destinations_artifact_id": destinations.id,
            "strategy": "routed_provider",
            "output_units": "seconds",
            "routing_profile": "driving",
            "route_annotation": "duration",
        },
        tool_context,
    )

    assert result.status is ToolResultStatus.COMPLETE
    matrix = read_matrix(tool_context.artifact_store, result.artifacts[0])
    assert np.array_equal(matrix.values, np.array([[10, np.inf], [20, 30]]))
    assert result.artifacts[0].source_provider == "osrm"
    assert result.artifacts[0].provider_metadata == {"fixture": True}
    assert result.metrics["unreachable_count"] == 1


def test_provider_models_and_extended_artifact_metadata_round_trip() -> None:
    value = provenance("fixture")
    assert ProviderProvenance.model_validate_json(value.model_dump_json()) == value
