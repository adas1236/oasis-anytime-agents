"""Server-owned provider configuration for message-driven runs."""

import httpx

from oasis.config import OasisSettings
from oasis.providers.adapters import (
    HttpSourceSnapshotProvider,
    NominatimPlaceResolver,
    OsrmRoutingMatrixProvider,
    StacCatalogSearcher,
)
from oasis.providers.cache import LocalSnapshotCache
from oasis.providers.http import HttpPolicy, ResilientHttpClient


class ServiceProviders:
    """Own the HTTP client; all network activity remains lazy until a tool requests it."""

    def __init__(self, settings: OasisSettings) -> None:
        self.client = httpx.AsyncClient()
        http = ResilientHttpClient(
            self.client,
            policy=HttpPolicy(
                user_agent=settings.provider_user_agent,
                timeout_seconds=settings.provider_timeout_seconds,
                max_attempts=settings.provider_max_attempts,
                backoff_base_seconds=settings.provider_backoff_base_seconds,
                max_response_bytes=settings.provider_max_response_bytes,
                max_pages=settings.provider_max_pages,
            ),
        )
        self.providers: dict[str, object] = {
            "place_resolution": NominatimPlaceResolver(http, endpoint=settings.place_endpoint),
            "source_snapshot": HttpSourceSnapshotProvider(http),
            "routing_matrix": OsrmRoutingMatrixProvider(http, endpoint=settings.routing_endpoint),
        }
        if settings.catalog_endpoint:
            self.providers["catalog_search"] = StacCatalogSearcher(
                http,
                endpoint=settings.catalog_endpoint,
            )
        self.resources: dict[str, object] = {
            "snapshot_cache": LocalSnapshotCache(settings.provider_cache_root),
        }

    async def close(self) -> None:
        await self.client.aclose()
