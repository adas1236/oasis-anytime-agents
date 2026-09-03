"""Provider and snapshot-cache protocols used by evidence tools."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from oasis.providers.models import (
    CatalogSearchRequest,
    CatalogSearchResult,
    PlaceResolution,
    PlaceResolveRequest,
    ProviderRequestContext,
    RetrievedSource,
    RouteMatrixRequest,
    RouteMatrixResult,
    SnapshotCacheEntry,
    SourceSnapshotRequest,
)


@runtime_checkable
class PlaceResolver(Protocol):
    """Resolve a text query to zero or more explicitly ranked candidates."""

    async def resolve(
        self, request: PlaceResolveRequest, context: ProviderRequestContext
    ) -> PlaceResolution: ...


@runtime_checkable
class CatalogSearcher(Protocol):
    """Search a possibly paginated source catalog."""

    async def search(
        self, request: CatalogSearchRequest, context: ProviderRequestContext
    ) -> CatalogSearchResult: ...


@runtime_checkable
class SourceSnapshotProvider(Protocol):
    """Fetch bounded source bytes without publishing mutable live data."""

    async def fetch(
        self, request: SourceSnapshotRequest, context: ProviderRequestContext
    ) -> RetrievedSource: ...


@runtime_checkable
class RoutingMatrixProvider(Protocol):
    """Return a normalized routed impedance matrix."""

    async def matrix(
        self, request: RouteMatrixRequest, context: ProviderRequestContext
    ) -> RouteMatrixResult: ...


@runtime_checkable
class SnapshotCache(Protocol):
    """Look up and advance cache pointers to immutable evidence artifacts."""

    def get(self, request_key: str) -> SnapshotCacheEntry | None: ...

    def put(self, entry: SnapshotCacheEntry) -> None: ...
