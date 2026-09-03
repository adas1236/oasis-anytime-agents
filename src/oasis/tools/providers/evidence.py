"""Stable evidence tools backed by replaceable live or frozen provider adapters."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import parse_qsl, urlsplit

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator
from shapely.geometry import box, shape

from oasis.artifacts import (
    ArtifactProvenance,
    canonical_json_bytes,
    put_json,
    put_table,
    put_vector,
    read_table,
    read_vector,
)
from oasis.providers.models import (
    CatalogSearchRequest,
    FreshnessPolicy,
    PlaceResolveRequest,
    ProviderError,
    ProviderErrorCode,
    ProviderRequestContext,
    SnapshotCacheEntry,
    SourceFormat,
    SourceSnapshotRequest,
)
from oasis.providers.protocols import (
    CatalogSearcher,
    PlaceResolver,
    SnapshotCache,
    SourceSnapshotProvider,
)
from oasis.providers.redaction import is_sensitive_key
from oasis.schemas import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactRef,
    ArtifactTransformation,
    DeterminismClassification,
    QualitySummary,
    SideEffectClassification,
    SpatialExtent,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.protocols import ToolContext, ToolExecutionError

PLACE_PROVIDER = "place_resolution"
CATALOG_PROVIDER = "catalog_search"
SOURCE_PROVIDER = "source_snapshot"
SNAPSHOT_CACHE = "snapshot_cache"
ROUTING_PROVIDER = "routing_matrix"


class ResolveAreaInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)
    country_codes: tuple[str, ...] = ()
    viewbox: SpatialExtent | None = None


class ResolveLocationsInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    queries: tuple[str, ...] = Field(min_length=1, max_length=25)
    limit_per_query: int = Field(default=3, ge=1, le=10)
    country_codes: tuple[str, ...] = ()


class ResolveOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    resolution_artifact_id: str
    result_count: int = Field(ge=0)
    ambiguous: bool
    provider: str


class ResolveLocationsOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    resolution_artifact_id: str
    query_count: int = Field(ge=1)
    result_count: int = Field(ge=0)
    ambiguous_query_count: int = Field(ge=0)
    provider: str


class SearchSourcesInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    collections: tuple[str, ...] = ()
    bounding_box: SpatialExtent | None = None
    datetime_range: str | None = None
    query: dict[str, JsonValue] = Field(default_factory=dict)
    limit: int = Field(default=100, ge=1, le=10_000)


class SearchSourcesOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    catalog_artifact_id: str
    item_count: int = Field(ge=0)
    page_count: int = Field(ge=1)
    truncated: bool
    provider: str


class SnapshotSourceInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str = Field(min_length=1)
    format: SourceFormat
    license: str = Field(min_length=1)
    units: str = Field(min_length=1)
    crs: str | None = None
    fields: tuple[str, ...] = ()
    bounding_box: SpatialExtent | None = None
    longitude_field: str | None = None
    latitude_field: str | None = None
    fresh_for_seconds: float = Field(default=86_400, ge=0)
    max_stale_seconds: float = Field(default=604_800, ge=0)
    allow_stale_on_error: bool = True
    cache_partition: str = Field(default="public", pattern=r"^[a-zA-Z0-9_.:-]{1,64}$")

    @model_validator(mode="after")
    def safe_url_and_freshness(self) -> SnapshotSourceInput:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("snapshot source URL must use HTTP or HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("snapshot credentials must use an authentication hook")
        if any(is_sensitive_key(key) for key, _ in parse_qsl(parsed.query)):
            raise ValueError("snapshot credentials must use an authentication hook")
        if self.max_stale_seconds < self.fresh_for_seconds:
            raise ValueError("maximum stale age must not be shorter than the freshness window")
        if (self.longitude_field is None) != (self.latitude_field is None):
            raise ValueError("longitude and latitude fields must be configured together")
        return self


class SnapshotSourceOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    kind: ArtifactKind
    row_count: int = Field(ge=0)
    cache_status: str
    stale: bool
    age_seconds: float = Field(ge=0)
    warning: str | None = None


def _provider_context(context: ToolContext) -> ProviderRequestContext:
    return ProviderRequestContext(
        deadline_monotonic=context.deadline_monotonic,
        cancellation=context.cancellation,
        monotonic=context.monotonic,
    )


def _provider_error(error: ProviderError) -> ToolResult:
    status = (
        ToolResultStatus.RATE_LIMITED
        if error.code is ProviderErrorCode.RATE_LIMITED
        else ToolResultStatus.FAILED
    )
    return ToolResult(
        status=status,
        summary=str(error),
        error=ToolError(
            code=ToolErrorCode.PROVIDER_FAILURE,
            message=str(error),
            retryable=error.retryable,
            context={"provider_code": error.code.value},
        ),
    )


def _place_provider(context: ToolContext) -> PlaceResolver:
    provider = context.providers.get(PLACE_PROVIDER)
    if not isinstance(provider, PlaceResolver):
        raise TypeError("configured place provider does not implement PlaceResolver")
    return provider


def _resolution_artifact(
    context: ToolContext,
    *,
    value: object,
    source_uri: str,
    provider: str,
    provider_metadata: dict[str, JsonValue],
    source_version: str | None,
    retrieved_at: datetime,
    license: str,
    row_count: int,
    schema_type: str,
) -> ArtifactRef:
    return put_json(
        context.artifact_store,
        value,
        kind=ArtifactKind.JSON_SPECIFICATION,
        units="unitless",
        provenance=ArtifactProvenance(
            source_uri=source_uri,
            source_provider=provider,
            provider_metadata=provider_metadata,
            source_version=source_version,
            retrieved_at=retrieved_at,
            license=license,
        ),
        data_schema={"type": schema_type, "version": "1.0.0"},
        row_count=row_count,
    )


class ResolveAreaTool:
    """Return ranked place/area candidates without silently selecting one."""

    version = "1.0.0"
    spec = ToolSpec(
        name="resolve_area",
        version=version,
        description=(
            "Resolve a place or area to explicitly ranked candidates; never auto-select ambiguity."
        ),
        input_schema=ResolveAreaInput.model_json_schema(),
        output_schema=ResolveOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "geocoding", "online"}),
        artifact_tags=frozenset({ArtifactKind.JSON_SPECIFICATION}),
        side_effects=SideEffectClassification.EXTERNAL_READ,
        required_providers=frozenset({PLACE_PROVIDER}),
        determinism=DeterminismClassification.EXTERNAL,
        seed_description="seed is ignored; the external provider snapshot is authoritative",
        runtime=ToolRuntimeEstimate(p50_ms=300, p95_ms=5_000),
        smoke_input={"query": "Springfield", "limit": 3},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = ResolveAreaInput.model_validate(arguments)
        try:
            result = await _place_provider(context).resolve(
                PlaceResolveRequest(**request.model_dump()), _provider_context(context)
            )
        except ProviderError as error:
            return _provider_error(error)
        candidates = tuple(candidate.model_dump(mode="json") for candidate in result.candidates)
        reference = _resolution_artifact(
            context,
            value=result.model_dump(mode="json"),
            source_uri=result.provenance.source_uri,
            provider=result.provenance.provider,
            provider_metadata=result.provenance.provider_metadata,
            source_version=result.provenance.source_version,
            retrieved_at=result.provenance.retrieved_at,
            license=result.provenance.license,
            row_count=len(candidates),
            schema_type="ranked_place_resolution",
        )
        output = ResolveOutput(
            resolution_artifact_id=reference.id,
            result_count=len(candidates),
            ambiguous=result.ambiguous,
            provider=result.provenance.provider,
        )
        if not candidates:
            return ToolResult(
                status=ToolResultStatus.FAILED,
                summary="No matching place candidates were returned.",
                metrics=output.model_dump(mode="json"),
                artifacts=(reference,),
                error=ToolError(code=ToolErrorCode.NOT_FOUND, message="place was not found"),
            )
        return ToolResult(
            status=ToolResultStatus.AMBIGUOUS if result.ambiguous else ToolResultStatus.COMPLETE,
            summary={
                "candidate_count": len(candidates),
                "ambiguous": result.ambiguous,
                "resolution": reference.id,
            },
            artifacts=(reference,),
            metrics=output.model_dump(mode="json"),
        )


class ResolveLocationsTool:
    """Resolve several independent locations while retaining per-query rankings."""

    version = "1.0.0"
    spec = ToolSpec(
        name="resolve_locations",
        version=version,
        description="Resolve multiple locations to ranked candidates without choosing ambiguities.",
        input_schema=ResolveLocationsInput.model_json_schema(),
        output_schema=ResolveLocationsOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "geocoding", "online"}),
        artifact_tags=frozenset({ArtifactKind.JSON_SPECIFICATION}),
        side_effects=SideEffectClassification.EXTERNAL_READ,
        required_providers=frozenset({PLACE_PROVIDER}),
        determinism=DeterminismClassification.EXTERNAL,
        seed_description="seed is ignored; external provider snapshots are authoritative",
        runtime=ToolRuntimeEstimate(p50_ms=500, p95_ms=10_000),
        smoke_input={"queries": ["Springfield"], "limit_per_query": 3},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = ResolveLocationsInput.model_validate(arguments)
        provider = _place_provider(context)
        raw_results: list[dict[str, JsonValue]] = []
        result_count = 0
        ambiguous_count = 0
        provider_name = "unknown"
        source_uri = "oasis://provider/resolve-locations"
        source_version: str | None = None
        retrieved_at = datetime.now(UTC)
        license = "provider terms apply"
        provider_metadata: dict[str, JsonValue] = {}
        try:
            for index, query in enumerate(request.queries):
                context.cancellation.raise_if_cancelled()
                result = await provider.resolve(
                    PlaceResolveRequest(
                        query=query,
                        limit=request.limit_per_query,
                        country_codes=request.country_codes,
                    ),
                    _provider_context(context),
                )
                provider_name = result.provenance.provider
                source_uri = result.provenance.source_uri
                source_version = result.provenance.source_version
                retrieved_at = result.provenance.retrieved_at
                license = result.provenance.license
                provider_metadata = result.provenance.provider_metadata
                candidates = [candidate.model_dump(mode="json") for candidate in result.candidates]
                result_count += len(candidates)
                ambiguous_count += int(result.ambiguous)
                raw_results.append(
                    {
                        "query_index": index,
                        "candidate_count": len(candidates),
                        "ambiguous": result.ambiguous,
                        "candidates": cast(list[JsonValue], candidates),
                        "provenance": cast(
                            dict[str, JsonValue], result.provenance.model_dump(mode="json")
                        ),
                    }
                )
        except ProviderError as error:
            return _provider_error(error)
        reference = _resolution_artifact(
            context,
            value={"results": raw_results},
            source_uri=source_uri,
            provider=provider_name,
            provider_metadata={**provider_metadata, "query_count": len(request.queries)},
            source_version=source_version,
            retrieved_at=retrieved_at,
            license=license,
            row_count=result_count,
            schema_type="ranked_location_resolutions",
        )
        output = ResolveLocationsOutput(
            resolution_artifact_id=reference.id,
            query_count=len(request.queries),
            result_count=result_count,
            ambiguous_query_count=ambiguous_count,
            provider=provider_name,
        )
        return ToolResult(
            status=ToolResultStatus.AMBIGUOUS if ambiguous_count else ToolResultStatus.COMPLETE,
            summary={
                "query_count": len(request.queries),
                "result_count": result_count,
                "ambiguous_query_count": ambiguous_count,
                "resolution": reference.id,
            },
            artifacts=(reference,),
            metrics=output.model_dump(mode="json"),
        )


class SearchSourcesTool:
    """Search a catalog and snapshot the bounded normalized result as an artifact."""

    version = "1.0.0"
    spec = ToolSpec(
        name="search_sources",
        version=version,
        description="Search a bounded STAC-compatible catalog and store normalized results.",
        input_schema=SearchSourcesInput.model_json_schema(),
        output_schema=SearchSourcesOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "catalog", "online"}),
        artifact_tags=frozenset({ArtifactKind.JSON_SPECIFICATION}),
        side_effects=SideEffectClassification.EXTERNAL_READ,
        required_providers=frozenset({CATALOG_PROVIDER}),
        determinism=DeterminismClassification.EXTERNAL,
        seed_description="seed is ignored; the immutable result records external retrieval time",
        runtime=ToolRuntimeEstimate(p50_ms=500, p95_ms=15_000),
        smoke_input={"limit": 2},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = SearchSourcesInput.model_validate(arguments)
        provider = context.providers.get(CATALOG_PROVIDER)
        if not isinstance(provider, CatalogSearcher):
            raise TypeError("configured catalog provider does not implement CatalogSearcher")
        try:
            result = await provider.search(
                CatalogSearchRequest(**request.model_dump()), _provider_context(context)
            )
        except ProviderError as error:
            return _provider_error(error)
        provenance = ArtifactProvenance(
            source_uri=result.provenance.source_uri,
            source_provider=result.provenance.provider,
            provider_metadata=result.provenance.provider_metadata,
            source_version=result.provenance.source_version,
            retrieved_at=result.provenance.retrieved_at,
            license=result.provenance.license,
        )
        reference = put_json(
            context.artifact_store,
            result.model_dump(mode="json"),
            kind=ArtifactKind.JSON_SPECIFICATION,
            units="unitless",
            provenance=provenance,
            data_schema={"type": "catalog_search", "version": self.version},
            row_count=len(result.items),
        )
        output = SearchSourcesOutput(
            catalog_artifact_id=reference.id,
            item_count=len(result.items),
            page_count=result.page_count,
            truncated=result.truncated,
            provider=result.provenance.provider,
        )
        return ToolResult(
            status=ToolResultStatus.PARTIAL if result.truncated else ToolResultStatus.COMPLETE,
            summary={
                "catalog": reference.id,
                "item_count": len(result.items),
                "truncated": result.truncated,
            },
            artifacts=(reference,),
            metrics=output.model_dump(mode="json"),
        )


def _request_key(request: SnapshotSourceInput) -> str:
    relevant = request.model_dump(
        mode="json",
        exclude={
            "fresh_for_seconds",
            "max_stale_seconds",
            "allow_stale_on_error",
            "license",
            "units",
            "crs",
        },
    )
    relevant.update({"license": request.license, "units": request.units, "crs": request.crs})
    return hashlib.sha256(canonical_json_bytes(relevant)).hexdigest()


def _read_geojson(content: bytes, request: SnapshotSourceInput) -> gpd.GeoDataFrame:
    try:
        document = json.loads(content)
        if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
            raise ValueError("not a feature collection")
        features = document.get("features")
        if not isinstance(features, list):
            raise ValueError("features must be an array")
        records: list[dict[str, object]] = []
        for raw_feature in features:
            if not isinstance(raw_feature, dict):
                raise ValueError("feature must be an object")
            properties = raw_feature.get("properties") or {}
            if not isinstance(properties, dict):
                raise ValueError("feature properties must be an object")
            row = {str(key): value for key, value in properties.items()}
            raw_geometry = raw_feature.get("geometry")
            row["geometry"] = None if raw_geometry is None else shape(raw_geometry)
            records.append(row)
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs=request.crs or "EPSG:4326")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            "source is not valid GeoJSON",
        ) from error
    if request.fields:
        missing = sorted(set(request.fields) - set(str(column) for column in frame.columns))
        if missing:
            raise ProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "source is missing requested fields",
            )
        frame = frame[[*request.fields, frame.geometry.name]]
    if request.bounding_box is not None:
        bounds = request.bounding_box
        filter_geometry = (
            gpd.GeoSeries(
                [box(bounds.west, bounds.south, bounds.east, bounds.north)], crs="EPSG:4326"
            )
            .to_crs(frame.crs)
            .iloc[0]
        )
        frame = frame.loc[frame.geometry.intersects(filter_geometry)].reset_index(drop=True)
    return frame


def _read_csv(content: bytes, request: SnapshotSourceInput) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE, "source is not valid CSV"
        ) from error
    if request.bounding_box is not None:
        if request.longitude_field is None or request.latitude_field is None:
            raise ProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "CSV bounding-box filtering requires longitude and latitude fields",
            )
        required = {request.longitude_field, request.latitude_field}
        if not required <= set(str(column) for column in frame.columns):
            raise ProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "CSV source is missing coordinate fields",
            )
        bounds = request.bounding_box
        longitude = pd.to_numeric(frame[request.longitude_field], errors="coerce")
        latitude = pd.to_numeric(frame[request.latitude_field], errors="coerce")
        frame = frame.loc[
            longitude.between(bounds.west, bounds.east)
            & latitude.between(bounds.south, bounds.north)
        ]
    if request.fields:
        missing = sorted(set(request.fields) - set(str(column) for column in frame.columns))
        if missing:
            raise ProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "source is missing requested fields",
            )
        frame = frame[list(request.fields)]
    return frame.reset_index(drop=True)


def _publish(
    context: ToolContext,
    request: SnapshotSourceInput,
    content: bytes,
    provenance: ArtifactProvenance,
    *,
    quality: QualitySummary | None = None,
) -> ArtifactRef:
    if request.format is SourceFormat.GEOJSON:
        return put_vector(
            context.artifact_store,
            _read_geojson(content, request),
            units=request.units,
            provenance=provenance,
            quality=quality,
        )
    return put_table(
        context.artifact_store,
        _read_csv(content, request),
        crs=request.crs,
        units=request.units,
        provenance=provenance,
        quality=quality,
    )


def _republish_stale(
    context: ToolContext,
    request: SnapshotSourceInput,
    cached: ArtifactRef,
    *,
    age_seconds: float,
) -> ArtifactRef:
    warning = f"cached snapshot is stale ({age_seconds:.0f} seconds old)"
    quality = cached.quality.model_copy(update={"warnings": (*cached.quality.warnings, warning)})
    provenance = ArtifactProvenance(
        source_uri=cached.source_uri or "unknown",
        source_provider=cached.source_provider,
        provider_metadata={**cached.provider_metadata, "cache_status": "stale_fallback"},
        source_version=cached.source_version,
        retrieved_at=cached.retrieved_at,
        license=cached.license or request.license,
        lineage=ArtifactLineage(
            parent_ids=(cached.id,),
            transformations=(
                ArtifactTransformation(
                    name="cached_snapshot_fallback",
                    version="1.0.0",
                    parameters={"age_seconds": age_seconds},
                ),
            ),
        ),
        privacy=cached.privacy,
    )
    if cached.kind is ArtifactKind.VECTOR:
        return put_vector(
            context.artifact_store,
            read_vector(context.artifact_store, cached),
            units=request.units,
            provenance=provenance,
            quality=quality,
        )
    if cached.kind is ArtifactKind.TABLE:
        return put_table(
            context.artifact_store,
            read_table(context.artifact_store, cached),
            crs=request.crs,
            units=request.units,
            provenance=provenance,
            quality=quality,
        )
    raise TypeError("snapshot cache entry must reference a vector or table")


class SnapshotSourceTool:
    """Turn mutable HTTP bytes into freshness-aware immutable canonical artifacts."""

    version = "1.0.0"
    spec = ToolSpec(
        name="snapshot_source",
        version=version,
        description="Fetch and immutably snapshot bounded CSV or GeoJSON with explicit freshness.",
        input_schema=SnapshotSourceInput.model_json_schema(),
        output_schema=SnapshotSourceOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "snapshot", "online", "cache"}),
        artifact_tags=frozenset({ArtifactKind.TABLE, ArtifactKind.VECTOR}),
        side_effects=SideEffectClassification.EXTERNAL_READ,
        required_providers=frozenset({SOURCE_PROVIDER}),
        required_resources=frozenset({SNAPSHOT_CACHE}),
        determinism=DeterminismClassification.EXTERNAL,
        seed_description="seed is ignored; immutable content and retrieval metadata are recorded",
        runtime=ToolRuntimeEstimate(p50_ms=500, p95_ms=20_000),
        smoke_input={
            "url": "https://example.invalid/data.csv",
            "format": "csv",
            "license": "example",
            "units": "unitless",
        },
    )

    @staticmethod
    def _output(
        reference: ArtifactRef,
        *,
        status: str,
        stale: bool,
        age_seconds: float,
        warning: str | None = None,
    ) -> ToolResult:
        output = SnapshotSourceOutput(
            artifact_id=reference.id,
            kind=reference.kind,
            row_count=reference.row_count or 0,
            cache_status=status,
            stale=stale,
            age_seconds=age_seconds,
            warning=warning,
        )
        return ToolResult(
            status=ToolResultStatus.PARTIAL if stale else ToolResultStatus.COMPLETE,
            summary={
                "snapshot": reference.id,
                "kind": reference.kind.value,
                "cache_status": status,
                "stale": stale,
                "warning": warning,
            },
            artifacts=(reference,),
            metrics=output.model_dump(mode="json"),
        )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        try:
            request = SnapshotSourceInput.model_validate(arguments)
        except ValidationError as error:
            raise ToolExecutionError(
                ToolError(
                    code=ToolErrorCode.INVALID_ARGUMENTS,
                    message="invalid source snapshot request",
                )
            ) from error
        policy = FreshnessPolicy(
            fresh_for_seconds=request.fresh_for_seconds,
            max_stale_seconds=request.max_stale_seconds,
            allow_stale_on_error=request.allow_stale_on_error,
        )
        cache = context.resources.get(SNAPSHOT_CACHE)
        provider = context.providers.get(SOURCE_PROVIDER)
        if not isinstance(cache, SnapshotCache):
            raise TypeError("configured snapshot cache does not implement SnapshotCache")
        if not isinstance(provider, SourceSnapshotProvider):
            raise TypeError("configured source provider does not implement SourceSnapshotProvider")
        key = _request_key(request)
        entry = cache.get(key)
        now = datetime.now(UTC)
        cached: ArtifactRef | None = None
        age = 0.0
        if entry is not None and context.artifact_store.exists(entry.artifact_id):
            cached = context.artifact_store.get_metadata(entry.artifact_id)
            age = max(0.0, (now - entry.retrieved_at).total_seconds())
            if age <= policy.fresh_for_seconds:
                return self._output(cached, status="hit", stale=False, age_seconds=age)
        try:
            source = await provider.fetch(
                SourceSnapshotRequest(
                    url=request.url,
                    format=request.format,
                    fields=request.fields,
                    bounding_box=request.bounding_box,
                ),
                _provider_context(context),
            )
            context.cancellation.raise_if_cancelled()
            provenance = ArtifactProvenance(
                source_uri=source.provenance.source_uri,
                source_provider=source.provenance.provider,
                provider_metadata=source.provenance.provider_metadata,
                source_version=source.provenance.source_version,
                retrieved_at=source.provenance.retrieved_at,
                license=request.license,
            )
            reference = _publish(context, request, source.content, provenance)
            context.cancellation.raise_if_cancelled()
            cache.put(
                SnapshotCacheEntry(
                    request_key=key,
                    artifact_id=reference.id,
                    retrieved_at=source.provenance.retrieved_at,
                )
            )
            source_age = max(0.0, (now - source.provenance.retrieved_at).total_seconds())
            return self._output(
                reference,
                status="miss",
                stale=False,
                age_seconds=source_age,
            )
        except ProviderError as error:
            if (
                cached is not None
                and policy.allow_stale_on_error
                and age <= policy.max_stale_seconds
            ):
                warning = f"live retrieval failed; using a cached snapshot {age:.0f} seconds old"
                stale_reference = _republish_stale(
                    context,
                    request,
                    cached,
                    age_seconds=age,
                )
                return self._output(
                    stale_reference,
                    status="stale_fallback",
                    stale=True,
                    age_seconds=age,
                    warning=warning,
                )
            return _provider_error(error)


def provider_tools() -> tuple[
    ResolveAreaTool, ResolveLocationsTool, SearchSourcesTool, SnapshotSourceTool
]:
    """Create the provider-backed evidence tools without provider instances or network calls."""

    return (
        ResolveAreaTool(),
        ResolveLocationsTool(),
        SearchSourcesTool(),
        SnapshotSourceTool(),
    )
