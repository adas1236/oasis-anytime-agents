"""Typed provider requests and normalized responses kept below domain schemas."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from oasis.schemas.artifacts import SpatialExtent


class ProviderErrorCode(StrEnum):
    """Stable failure categories shared by provider adapters."""

    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    RESPONSE_TOO_LARGE = "response_too_large"
    NO_RESULT = "no_result"


class ProviderError(RuntimeError):
    """Safe provider failure that never embeds request headers or credentials."""

    def __init__(
        self,
        code: ProviderErrorCode,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class ProviderProvenance(BaseModel):
    """Normalized source facts plus opaque, provider-owned metadata."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(pattern=r"^[a-z][a-z0-9_.:-]*$")
    source_uri: str = Field(min_length=1)
    retrieved_at: datetime
    license: str = Field(default="provider terms apply", min_length=1)
    source_version: str | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def retrieval_time_is_utc(self) -> Self:
        offset = self.retrieved_at.utcoffset()
        if self.retrieved_at.tzinfo is None or offset is None:
            raise ValueError("provider retrieval time must include a timezone")
        if offset.total_seconds() != 0:
            raise ValueError("provider retrieval time must use UTC")
        return self


class CancellationSignal(Protocol):
    """Structural cancellation signal accepted from controller or tool layers."""

    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ProviderRequestContext:
    """Request deadline and cancellation-independent public context."""

    deadline_monotonic: float
    cancellation: CancellationSignal
    monotonic: Callable[[], float]


class PlaceResolveRequest(BaseModel):
    """Provider-neutral ranked place lookup."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)
    country_codes: tuple[str, ...] = ()
    viewbox: SpatialExtent | None = None


class PlaceCandidate(BaseModel):
    """One normalized, explicitly ranked geocoder candidate."""

    model_config = ConfigDict(frozen=True)

    provider_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    bounding_box: SpatialExtent | None = None
    category: str | None = None
    importance: float | None = None
    rank: int = Field(ge=1)
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PlaceResolution(BaseModel):
    """Ranked place results; callers must resolve any ambiguity explicitly."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[PlaceCandidate, ...]
    provenance: ProviderProvenance

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


class CatalogSearchRequest(BaseModel):
    """Portable subset of STAC search fields."""

    model_config = ConfigDict(frozen=True)

    collections: tuple[str, ...] = ()
    bounding_box: SpatialExtent | None = None
    datetime_range: str | None = None
    query: dict[str, JsonValue] = Field(default_factory=dict)
    limit: int = Field(default=100, ge=1, le=10_000)


class CatalogAsset(BaseModel):
    """One catalog asset without leaking provider fields into artifact schemas."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    href: str = Field(min_length=1)
    media_type: str | None = None
    roles: tuple[str, ...] = ()
    title: str | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CatalogItem(BaseModel):
    """Normalized catalog item returned across provider implementations."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    collection: str | None = None
    title: str | None = None
    bounding_box: SpatialExtent | None = None
    observed_at: datetime | None = None
    assets: tuple[CatalogAsset, ...] = ()
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CatalogSearchResult(BaseModel):
    """All normalized catalog pages up to the caller's declared limit."""

    model_config = ConfigDict(frozen=True)

    items: tuple[CatalogItem, ...]
    page_count: int = Field(ge=1)
    truncated: bool = False
    provenance: ProviderProvenance


class SourceFormat(StrEnum):
    """Live source formats canonicalized by the snapshot evidence tool."""

    CSV = "csv"
    GEOJSON = "geojson"


class SourceSnapshotRequest(BaseModel):
    """HTTP source request with optional server-side projection and filtering."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(min_length=1)
    format: SourceFormat
    fields: tuple[str, ...] = ()
    bounding_box: SpatialExtent | None = None
    query_parameters: dict[str, str] = Field(default_factory=dict)


class RetrievedSource(BaseModel):
    """Bounded bytes returned by a source provider before canonical publication."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    media_type: str
    provenance: ProviderProvenance


class RouteAnnotation(StrEnum):
    """OSRM table quantities with fixed canonical units."""

    DURATION = "duration"
    DISTANCE = "distance"

    @property
    def units(self) -> str:
        return "seconds" if self is RouteAnnotation.DURATION else "meters"


class RouteMatrixRequest(BaseModel):
    """Coordinates and labels for a routed many-to-many matrix."""

    model_config = ConfigDict(frozen=True)

    coordinates: tuple[tuple[float, float], ...] = Field(min_length=1, max_length=100)
    source_indices: tuple[int, ...] = Field(min_length=1)
    destination_indices: tuple[int, ...] = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    destination_ids: tuple[str, ...] = Field(min_length=1)
    profile: str = Field(default="driving", pattern=r"^[a-z][a-z0-9_-]*$")
    annotation: RouteAnnotation = RouteAnnotation.DURATION

    @model_validator(mode="after")
    def dimensions_and_coordinates_match(self) -> Self:
        if len(self.source_indices) != len(self.source_ids):
            raise ValueError("route source indices and IDs must have equal lengths")
        if len(self.destination_indices) != len(self.destination_ids):
            raise ValueError("route destination indices and IDs must have equal lengths")
        upper = len(self.coordinates)
        if any(index < 0 or index >= upper for index in self.source_indices):
            raise ValueError("route source index is outside the coordinate array")
        if any(index < 0 or index >= upper for index in self.destination_indices):
            raise ValueError("route destination index is outside the coordinate array")
        for longitude, latitude in self.coordinates:
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError("route coordinates must be longitude/latitude pairs")
        return self


class RouteMatrixResult(BaseModel):
    """Normalized route values; unreachable pairs use null until artifact encoding."""

    model_config = ConfigDict(frozen=True)

    values: tuple[tuple[Annotated[float, Field(ge=0)] | None, ...], ...]
    source_ids: tuple[str, ...]
    destination_ids: tuple[str, ...]
    units: str
    provenance: ProviderProvenance

    @model_validator(mode="after")
    def shape_matches_labels(self) -> Self:
        if len(self.values) != len(self.source_ids):
            raise ValueError("route matrix row count does not match source labels")
        if any(len(row) != len(self.destination_ids) for row in self.values):
            raise ValueError("route matrix column count does not match destination labels")
        return self


class FreshnessPolicy(BaseModel):
    """Cache reuse and bounded stale-fallback policy."""

    model_config = ConfigDict(frozen=True)

    fresh_for_seconds: float = Field(default=86_400, ge=0)
    max_stale_seconds: float = Field(default=604_800, ge=0)
    allow_stale_on_error: bool = True

    @model_validator(mode="after")
    def stale_window_contains_fresh_window(self) -> Self:
        if self.max_stale_seconds < self.fresh_for_seconds:
            raise ValueError("maximum stale age must not be shorter than the freshness window")
        return self


class SnapshotCacheEntry(BaseModel):
    """Mutable cache index entry pointing only at an immutable artifact."""

    model_config = ConfigDict(frozen=True)

    request_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    retrieved_at: datetime

    @model_validator(mode="after")
    def retrieval_time_is_utc(self) -> Self:
        offset = self.retrieved_at.utcoffset()
        if self.retrieved_at.tzinfo is None or offset is None:
            raise ValueError("snapshot retrieval time must include a timezone")
        if offset.total_seconds() != 0:
            raise ValueError("snapshot retrieval time must use UTC")
        return self
