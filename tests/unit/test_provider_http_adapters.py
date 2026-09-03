from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from oasis.providers import (
    ApiKeyAuthentication,
    CatalogSearchRequest,
    HttpPolicy,
    HttpSourceSnapshotProvider,
    NominatimPlaceResolver,
    OsrmRoutingMatrixProvider,
    PlaceResolveRequest,
    ProviderError,
    ProviderErrorCode,
    ProviderRequestContext,
    ResilientHttpClient,
    RouteAnnotation,
    RouteMatrixRequest,
    SourceFormat,
    SourceSnapshotRequest,
    StacCatalogSearcher,
)
from oasis.schemas import SpatialExtent
from oasis.tools import CancellationToken


async def no_sleep(delay: float) -> None:
    del delay


def provider_context() -> ProviderRequestContext:
    return ProviderRequestContext(
        deadline_monotonic=time.monotonic() + 5,
        cancellation=CancellationToken(),
        monotonic=time.monotonic,
    )


def policy(**overrides: Any) -> HttpPolicy:
    return HttpPolicy(
        user_agent="oasis-tests/1.0 contact@example.test",
        timeout_seconds=1,
        max_attempts=overrides.pop("max_attempts", 3),
        backoff_base_seconds=0,
        max_response_bytes=overrides.pop("max_response_bytes", 10_000),
        max_pages=overrides.pop("max_pages", 3),
        **overrides,
    )


@pytest.mark.asyncio
async def test_http_retries_transient_status_then_returns_bounded_payload() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, request=request, text="retry")
        return httpx.Response(200, request=request, content=b'{"ok":true}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = ResilientHttpClient(client, policy=policy(), sleep=no_sleep)
        result = await transport.request(
            "GET",
            "https://provider.test/data",
            deadline_monotonic=time.monotonic() + 5,
            cancellation=CancellationToken(),
        )

    assert attempts == 3
    assert result.json() == {"ok": True}


@pytest.mark.asyncio
async def test_http_timeout_retries_then_returns_typed_failure() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("secret upstream detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = ResilientHttpClient(client, policy=policy(max_attempts=2), sleep=no_sleep)
        with pytest.raises(ProviderError) as raised:
            await transport.request(
                "GET",
                "https://provider.test/data",
                deadline_monotonic=time.monotonic() + 5,
                cancellation=CancellationToken(),
            )

    assert attempts == 2
    assert raised.value.code is ProviderErrorCode.TIMEOUT
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_http_rate_limit_and_oversized_response_are_typed() -> None:
    def rate_limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request, headers={"Retry-After": "0"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(rate_limited)) as client:
        transport = ResilientHttpClient(client, policy=policy(max_attempts=1), sleep=no_sleep)
        with pytest.raises(ProviderError) as raised:
            await transport.request(
                "GET",
                "https://provider.test/data",
                deadline_monotonic=time.monotonic() + 5,
                cancellation=CancellationToken(),
            )
    assert raised.value.code is ProviderErrorCode.RATE_LIMITED
    assert raised.value.retry_after_seconds == 0

    def too_large(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"12345")

    async with httpx.AsyncClient(transport=httpx.MockTransport(too_large)) as client:
        transport = ResilientHttpClient(client, policy=policy(max_attempts=1, max_response_bytes=4))
        with pytest.raises(ProviderError) as raised:
            await transport.request(
                "GET",
                "https://provider.test/data",
                deadline_monotonic=time.monotonic() + 5,
                cancellation=CancellationToken(),
            )
    assert raised.value.code is ProviderErrorCode.RESPONSE_TOO_LARGE


class CapturingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, message: str, **context: object) -> None:
        self.events.append((message, context))

    def warning(self, message: str, **context: object) -> None:
        self.events.append((message, context))


@pytest.mark.asyncio
async def test_authentication_and_sensitive_url_values_are_redacted_from_logs() -> None:
    secret = "super-secret-value"
    seen_url = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = str(request.url)
        return httpx.Response(200, request=request, json={})

    logger = CapturingLogger()
    authentication = ApiKeyAuthentication(name="api_key", value=secret, in_query=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = ResilientHttpClient(
            client,
            policy=policy(),
            authentication=authentication,
            logger=logger,
        )
        result = await transport.request(
            "GET",
            "https://provider.test/data?token=source-secret",
            deadline_monotonic=time.monotonic() + 5,
            cancellation=CancellationToken(),
        )

    assert secret in seen_url
    assert secret not in repr(authentication)
    assert "source-secret" not in result.url
    assert secret not in repr(logger.events)
    assert "source-secret" not in repr(logger.events)


@pytest.mark.asyncio
async def test_nominatim_returns_ranked_ambiguity_and_no_result() -> None:
    records = [
        {
            "osm_type": "relation",
            "osm_id": 1,
            "display_name": "Lower-ranked Springfield",
            "lon": "-89.64",
            "lat": "39.78",
            "boundingbox": ["39", "40", "-90", "-89"],
            "importance": 0.4,
            "place_rank": 16,
            "category": "place",
            "type": "city",
        },
        {
            "osm_type": "relation",
            "osm_id": 2,
            "display_name": "Higher-ranked Springfield",
            "lon": "-72.59",
            "lat": "42.10",
            "boundingbox": ["42", "43", "-73", "-72"],
            "importance": 0.8,
            "place_rank": 16,
            "category": "place",
            "type": "city",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"].startswith("oasis-tests")
        return httpx.Response(200, request=request, json=records)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolver = NominatimPlaceResolver(ResilientHttpClient(client, policy=policy()))
        result = await resolver.resolve(
            PlaceResolveRequest(query="Springfield", limit=5), provider_context()
        )

    assert result.ambiguous
    assert [candidate.rank for candidate in result.candidates] == [1, 2]
    assert result.candidates[0].display_name == "Higher-ranked Springfield"
    assert result.provenance.provider == "nominatim"
    assert "Springfield" not in result.provenance.source_uri

    def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(empty_handler)) as client:
        resolver = NominatimPlaceResolver(ResilientHttpClient(client, policy=policy()))
        empty = await resolver.resolve(
            PlaceResolveRequest(query="No Such Place"), provider_context()
        )
    assert empty.candidates == ()


@pytest.mark.asyncio
async def test_stac_search_follows_pagination_and_normalizes_assets() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/search":
            payload = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "stac_version": "1.0.0",
                        "id": "one",
                        "collection": "health",
                        "bbox": [-72, 41, -71, 42],
                        "properties": {"datetime": "2026-01-01T00:00:00Z"},
                        "assets": {
                            "data": {
                                "href": "https://assets.test/one.tif?signature=secret",
                                "type": "image/tiff",
                                "roles": ["data"],
                            }
                        },
                    }
                ],
                "links": [{"rel": "next", "href": "/page-2"}],
            }
        else:
            payload = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "two",
                        "properties": {},
                        "assets": {},
                    }
                ],
                "links": [],
            }
        return httpx.Response(200, request=request, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        searcher = StacCatalogSearcher(
            ResilientHttpClient(client, policy=policy()), endpoint="https://stac.test"
        )
        result = await searcher.search(
            CatalogSearchRequest(
                collections=("health",),
                bounding_box=SpatialExtent(west=-72, south=41, east=-71, north=42),
                limit=10,
            ),
            provider_context(),
        )

    assert calls == ["/search", "/page-2"]
    assert [item.id for item in result.items] == ["one", "two"]
    assert result.page_count == 2
    assert "secret" not in result.items[0].assets[0].href


@pytest.mark.asyncio
async def test_stac_page_limit_is_labeled_truncated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"features": [], "links": [{"rel": "next", "href": "/again"}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        searcher = StacCatalogSearcher(
            ResilientHttpClient(client, policy=policy(max_pages=1)), endpoint="https://stac.test"
        )
        result = await searcher.search(CatalogSearchRequest(), provider_context())

    assert result.page_count == 1
    assert result.truncated


@pytest.mark.asyncio
async def test_http_source_and_osrm_normalize_provenance_and_unreachable_pairs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "source.test":
            assert request.url.params["fields"] == "id,value"
            assert request.url.params["bbox"] == "-72.0,41.0,-71.0,42.0"
            return httpx.Response(
                200,
                request=request,
                content=b"id,value\na,1\n",
                headers={"Content-Type": "text/csv", "ETag": '"v1"'},
            )
        return httpx.Response(
            200,
            request=request,
            json={"code": "Ok", "durations": [[0, None], [12.5, 0]], "data_version": "x"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = ResilientHttpClient(client, policy=policy())
        source = await HttpSourceSnapshotProvider(transport).fetch(
            SourceSnapshotRequest(
                url="https://source.test/data.csv",
                format=SourceFormat.CSV,
                fields=("id", "value"),
                bounding_box=SpatialExtent(west=-72, south=41, east=-71, north=42),
            ),
            provider_context(),
        )
        matrix = await OsrmRoutingMatrixProvider(transport, endpoint="https://router.test").matrix(
            RouteMatrixRequest(
                coordinates=((-71.1, 42.1), (-71.2, 42.2)),
                source_indices=(0, 1),
                destination_indices=(0, 1),
                source_ids=("a", "b"),
                destination_ids=("a", "b"),
                annotation=RouteAnnotation.DURATION,
            ),
            provider_context(),
        )

    assert source.provenance.source_version == '"v1"'
    assert source.media_type == "text/csv"
    assert matrix.values == ((0.0, None), (12.5, 0.0))
    assert matrix.units == "seconds"
    assert matrix.provenance.provider_metadata["profile"] == "driving"
    assert "-71.1" not in matrix.provenance.source_uri


@pytest.mark.asyncio
async def test_malformed_provider_json_is_a_typed_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"not-json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolver = NominatimPlaceResolver(ResilientHttpClient(client, policy=policy()))
        with pytest.raises(ProviderError) as raised:
            await resolver.resolve(PlaceResolveRequest(query="Boston"), provider_context())

    assert raised.value.code is ProviderErrorCode.MALFORMED_RESPONSE
