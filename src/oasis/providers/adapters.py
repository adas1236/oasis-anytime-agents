"""Nominatim, STAC, generic HTTP, and OSRM-compatible provider adapters."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urljoin, urlsplit

from pydantic import ValidationError

from oasis.providers.http import ResilientHttpClient
from oasis.providers.models import (
    CatalogAsset,
    CatalogItem,
    CatalogSearchRequest,
    CatalogSearchResult,
    PlaceCandidate,
    PlaceResolution,
    PlaceResolveRequest,
    ProviderError,
    ProviderErrorCode,
    ProviderProvenance,
    ProviderRequestContext,
    RetrievedSource,
    RouteMatrixRequest,
    RouteMatrixResult,
    SourceSnapshotRequest,
)
from oasis.providers.redaction import redact_url
from oasis.schemas.artifacts import SpatialExtent


def _now() -> datetime:
    return datetime.now(UTC)


def _object_payload(raw: object, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            f"{label} response must be a JSON object",
        )
    return cast(dict[str, Any], raw)


def _array_payload(raw: object, *, label: str) -> list[Any]:
    if not isinstance(raw, list):
        raise ProviderError(
            ProviderErrorCode.MALFORMED_RESPONSE,
            f"{label} response must be a JSON array",
        )
    return raw


def _extent_from_bbox(raw: object) -> SpatialExtent | None:
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    try:
        return SpatialExtent(
            west=float(raw[0]), south=float(raw[1]), east=float(raw[2]), north=float(raw[3])
        )
    except (TypeError, ValueError, ValidationError):
        return None


class NominatimPlaceResolver:
    """Ranked Nominatim-compatible place and area resolution."""

    name = "nominatim"

    def __init__(
        self,
        http: ResilientHttpClient,
        *,
        endpoint: str = "https://nominatim.openstreetmap.org",
        license: str = "ODbL-1.0",
    ) -> None:
        self.http = http
        self.endpoint = endpoint.rstrip("/")
        self.license = license

    async def resolve(
        self, request: PlaceResolveRequest, context: ProviderRequestContext
    ) -> PlaceResolution:
        parameters = {
            "q": request.query,
            "format": "jsonv2",
            "addressdetails": "1",
            "limit": str(request.limit),
        }
        if request.country_codes:
            parameters["countrycodes"] = ",".join(request.country_codes)
        if request.viewbox is not None:
            box = request.viewbox
            parameters["viewbox"] = f"{box.west},{box.north},{box.east},{box.south}"
        payload = await self.http.request(
            "GET",
            f"{self.endpoint}/search",
            parameters=parameters,
            redact_parameter_names=frozenset({"q"}),
            deadline_monotonic=context.deadline_monotonic,
            cancellation=context.cancellation,
        )
        records = _array_payload(payload.json(), label="Nominatim")
        candidates: list[PlaceCandidate] = []
        for raw in records:
            record = _object_payload(raw, label="Nominatim candidate")
            try:
                osm_type = str(record.get("osm_type", "object"))
                osm_id = str(record["osm_id"])
                south, north, west, east = (
                    float(cast(Any, value)) for value in cast(list[object], record["boundingbox"])
                )
                candidates.append(
                    PlaceCandidate(
                        provider_id=f"{osm_type}:{osm_id}",
                        display_name=str(record["display_name"]),
                        longitude=float(record["lon"]),
                        latitude=float(record["lat"]),
                        bounding_box=SpatialExtent(
                            west=west,
                            south=south,
                            east=east,
                            north=north,
                        ),
                        category=str(record["category"]) if record.get("category") else None,
                        importance=(
                            float(record["importance"])
                            if record.get("importance") is not None
                            else None
                        ),
                        rank=1,
                        provider_metadata={
                            "place_rank": int(record["place_rank"])
                            if record.get("place_rank") is not None
                            else None,
                            "type": str(record["type"]) if record.get("type") else None,
                        },
                    )
                )
            except (KeyError, TypeError, ValueError, ValidationError) as error:
                raise ProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "Nominatim returned an invalid candidate",
                ) from error
        candidates.sort(
            key=lambda candidate: (
                candidate.importance is not None,
                candidate.importance or -math.inf,
            ),
            reverse=True,
        )
        ranked = tuple(
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(candidates[: request.limit], start=1)
        )
        return PlaceResolution(
            candidates=ranked,
            provenance=ProviderProvenance(
                provider=self.name,
                source_uri=payload.url,
                retrieved_at=_now(),
                license=self.license,
                provider_metadata={"result_count": len(ranked)},
            ),
        )


class HttpSourceSnapshotProvider:
    """Bounded generic HTTP CSV/GeoJSON retrieval with optional common filters."""

    name = "http"

    def __init__(self, http: ResilientHttpClient) -> None:
        self.http = http

    async def fetch(
        self, request: SourceSnapshotRequest, context: ProviderRequestContext
    ) -> RetrievedSource:
        parameters = dict(request.query_parameters)
        if request.fields:
            parameters.setdefault("fields", ",".join(request.fields))
        if request.bounding_box is not None:
            box = request.bounding_box
            parameters.setdefault("bbox", f"{box.west},{box.south},{box.east},{box.north}")
        payload = await self.http.request(
            "GET",
            request.url,
            parameters=parameters,
            headers={"Accept": "text/csv, application/geo+json, application/json"},
            redact_parameter_names=frozenset({"bbox", *request.query_parameters}),
            deadline_monotonic=context.deadline_monotonic,
            cancellation=context.cancellation,
        )
        media_type = payload.headers.get("content-type", "application/octet-stream").split(";", 1)[
            0
        ]
        version = payload.headers.get("etag") or payload.headers.get("last-modified")
        return RetrievedSource(
            content=payload.content,
            media_type=media_type,
            provenance=ProviderProvenance(
                provider=self.name,
                source_uri=payload.url,
                retrieved_at=_now(),
                source_version=version,
                provider_metadata={
                    "status_code": payload.status_code,
                    "content_type": media_type,
                },
            ),
        )


class StacCatalogSearcher:
    """STAC API item search with bounded link pagination."""

    name = "stac"

    def __init__(
        self,
        http: ResilientHttpClient,
        *,
        endpoint: str,
        license: str = "catalog metadata; item licenses are asset-specific",
    ) -> None:
        self.http = http
        self.endpoint = endpoint.rstrip("/")
        self.license = license

    @staticmethod
    def _body(request: CatalogSearchRequest) -> dict[str, object]:
        body: dict[str, object] = {"limit": min(request.limit, 1_000)}
        if request.collections:
            body["collections"] = list(request.collections)
        if request.bounding_box is not None:
            box = request.bounding_box
            body["bbox"] = [box.west, box.south, box.east, box.north]
        if request.datetime_range is not None:
            body["datetime"] = request.datetime_range
        if request.query:
            body["query"] = request.query
        return body

    @staticmethod
    def _item(raw: object) -> CatalogItem:
        record = _object_payload(raw, label="STAC item")
        properties = record.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        raw_assets = record.get("assets")
        raw_assets = raw_assets if isinstance(raw_assets, dict) else {}
        assets: list[CatalogAsset] = []
        for key in sorted(raw_assets):
            value = raw_assets[key]
            if not isinstance(value, dict) or not isinstance(value.get("href"), str):
                raise ProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "STAC item contains an invalid asset",
                )
            roles = value.get("roles")
            assets.append(
                CatalogAsset(
                    key=str(key),
                    href=redact_url(value["href"]),
                    media_type=str(value["type"]) if value.get("type") else None,
                    roles=tuple(str(role) for role in roles) if isinstance(roles, list) else (),
                    title=str(value["title"]) if value.get("title") else None,
                    provider_metadata={
                        str(name): cast(Any, item)
                        for name, item in value.items()
                        if name not in {"href", "type", "roles", "title"}
                    },
                )
            )
        date_value = properties.get("datetime")
        try:
            date = (
                datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
                if date_value
                else None
            )
            return CatalogItem(
                id=str(record["id"]),
                collection=str(record["collection"]) if record.get("collection") else None,
                title=str(properties["title"]) if properties.get("title") else None,
                bounding_box=_extent_from_bbox(record.get("bbox")),
                observed_at=date,
                assets=tuple(assets),
                provider_metadata={
                    "stac_version": str(record["stac_version"])
                    if record.get("stac_version")
                    else None
                },
            )
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise ProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "STAC returned an invalid item",
            ) from error

    async def search(
        self, request: CatalogSearchRequest, context: ProviderRequestContext
    ) -> CatalogSearchResult:
        url = f"{self.endpoint}/search"
        method = "POST"
        body: object | None = self._body(request)
        items: list[CatalogItem] = []
        seen_pages: set[tuple[str, str]] = set()
        page_count = 0
        truncated = False
        last_url = url
        while url:
            if page_count >= self.http.policy.max_pages:
                truncated = True
                break
            marker = (method, url)
            if marker in seen_pages:
                raise ProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "STAC pagination contains a cycle",
                )
            seen_pages.add(marker)
            page_count += 1
            payload = await self.http.request(
                method,
                url,
                json_body=body,
                deadline_monotonic=context.deadline_monotonic,
                cancellation=context.cancellation,
            )
            last_url = payload.url
            page = _object_payload(payload.json(), label="STAC")
            features = page.get("features")
            if not isinstance(features, list):
                raise ProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "STAC response has no feature array",
                )
            for raw_item in features:
                if len(items) >= request.limit:
                    truncated = True
                    break
                items.append(self._item(raw_item))
            if len(items) >= request.limit:
                break
            next_link = next(
                (
                    link
                    for link in page.get("links", [])
                    if isinstance(link, dict) and link.get("rel") == "next"
                ),
                None,
            )
            if next_link is None:
                break
            href = next_link.get("href")
            if not isinstance(href, str):
                raise ProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "STAC next link has no href",
                )
            url = urljoin(url, href)
            expected_origin = urlsplit(self.endpoint)
            actual_origin = urlsplit(url)
            if (actual_origin.scheme, actual_origin.netloc) != (
                expected_origin.scheme,
                expected_origin.netloc,
            ):
                raise ProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "STAC next link changes origin",
                )
            method = str(next_link.get("method", "GET")).upper()
            if method not in {"GET", "POST"}:
                raise ProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "STAC next link uses an unsupported method",
                )
            body = next_link.get("body") if method == "POST" else None
        return CatalogSearchResult(
            items=tuple(items),
            page_count=page_count,
            truncated=truncated,
            provenance=ProviderProvenance(
                provider=self.name,
                source_uri=last_url,
                retrieved_at=_now(),
                license=self.license,
                provider_metadata={"pages": page_count},
            ),
        )


class OsrmRoutingMatrixProvider:
    """OSRM-compatible table service returning routed duration or distance."""

    name = "osrm"

    def __init__(
        self,
        http: ResilientHttpClient,
        *,
        endpoint: str = "https://router.project-osrm.org",
        license: str = "ODbL-1.0",
    ) -> None:
        self.http = http
        self.endpoint = endpoint.rstrip("/")
        self.license = license

    async def matrix(
        self, request: RouteMatrixRequest, context: ProviderRequestContext
    ) -> RouteMatrixResult:
        coordinates = ";".join(f"{lon:.8f},{lat:.8f}" for lon, lat in request.coordinates)
        url = f"{self.endpoint}/table/v1/{request.profile}/{coordinates}"
        parameters = {
            "sources": ";".join(str(index) for index in request.source_indices),
            "destinations": ";".join(str(index) for index in request.destination_indices),
            "annotations": request.annotation.value,
        }
        payload = await self.http.request(
            "GET",
            url,
            parameters=parameters,
            safe_url_override=(
                f"{self.endpoint}/table/v1/{request.profile}/[REDACTED_COORDINATES]"
            ),
            deadline_monotonic=context.deadline_monotonic,
            cancellation=context.cancellation,
        )
        document = _object_payload(payload.json(), label="OSRM")
        if document.get("code") != "Ok":
            raise ProviderError(
                ProviderErrorCode.UNAVAILABLE,
                "OSRM could not compute the requested matrix",
            )
        key = "durations" if request.annotation.value == "duration" else "distances"
        raw_values = document.get(key)
        if not isinstance(raw_values, list):
            raise ProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                f"OSRM response has no {key} matrix",
            )
        try:
            values = tuple(
                tuple(
                    None if value is None else float(cast(Any, value))
                    for value in cast(list[object], row)
                )
                for row in raw_values
                if isinstance(row, list)
            )
            if len(values) != len(raw_values):
                raise ValueError("non-array matrix row")
            return RouteMatrixResult(
                values=values,
                source_ids=request.source_ids,
                destination_ids=request.destination_ids,
                units=request.annotation.units,
                provenance=ProviderProvenance(
                    provider=self.name,
                    source_uri=payload.url,
                    retrieved_at=_now(),
                    license=self.license,
                    source_version=(
                        str(document["data_version"]) if document.get("data_version") else None
                    ),
                    provider_metadata={
                        "engine_code": str(document["code"]),
                        "profile": request.profile,
                        "annotation": request.annotation.value,
                    },
                ),
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise ProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "OSRM returned an invalid matrix",
            ) from error


__all__ = [
    "HttpSourceSnapshotProvider",
    "NominatimPlaceResolver",
    "OsrmRoutingMatrixProvider",
    "StacCatalogSearcher",
]
