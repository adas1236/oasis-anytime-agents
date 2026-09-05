"""Dataset-backed implementations of the live provider contracts.

Only source locations enter these adapters: prompts, answers, and solver parameters
are deliberately absent. Driving weights come from the cached OSRM table service.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlencode, urlsplit

from pydantic import JsonValue

from oasis.mock_experiments import Location, LocationIndex, OsrmMatrixStore
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
    RouteAnnotation,
    RouteMatrixRequest,
    RouteMatrixResult,
    SourceFormat,
    SourceSnapshotRequest,
)

SOURCE_BASE = "https://mock-dataset.invalid/locations"


class DatasetEvidenceProvider:
    """Replace geocoder/catalog/source I/O while keeping their public tool contracts."""

    def __init__(self, locations: Sequence[Location], region: str) -> None:
        self.locations = tuple(locations)
        self.region = region
        self.index = LocationIndex(locations)
        self.retrieved_at = datetime.now(UTC)

    def provenance(self, source_uri: str = SOURCE_BASE) -> ProviderProvenance:
        return ProviderProvenance(
            provider="mock_dataset",
            source_uri=source_uri,
            retrieved_at=self.retrieved_at,
            license="mock example; no license supplied",
            provider_metadata={
                "region": self.region,
                "coordinates": "dataset coordinates; reported source OSM/Nominatim",
                "population": "synthetic planning counts when present",
            },
        )

    async def resolve(
        self, request: PlaceResolveRequest, context: ProviderRequestContext
    ) -> PlaceResolution:
        context.cancellation.raise_if_cancelled()
        matches = self.index.search(request.query, limit=request.limit)
        matches = [(score, location) for score, location in matches if score >= 0.6]
        if matches and matches[0][0] == 1:
            matches = [match for match in matches if match[0] == 1]
        countries = {
            "United States": "us",
            "United Kingdom": "gb",
            "India": "in",
            "Germany": "de",
            "Japan": "jp",
            "South Africa": "za",
            "Kenya": "ke",
        }
        country = next(
            (code for name, code in countries.items() if self.region.endswith(name)), None
        )
        if request.country_codes and country not in {c.lower() for c in request.country_codes}:
            matches = []
        candidates: list[PlaceCandidate] = []
        for score, location in matches:
            box = request.viewbox
            if box is not None and not (
                box.west <= location.longitude <= box.east
                and box.south <= location.latitude <= box.north
            ):
                continue
            metadata: dict[str, JsonValue] = {}
            if location.population is not None:
                metadata["population"] = location.population
            candidates.append(
                PlaceCandidate(
                    provider_id=location.location_id,
                    display_name=location.name,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    rank=len(candidates) + 1,
                    importance=score,
                    provider_metadata=metadata,
                )
            )
        return PlaceResolution(candidates=tuple(candidates), provenance=self.provenance())

    def _resolve_names(self, names: list[str]) -> tuple[Location, ...]:
        if not names:
            raise ProviderError(
                ProviderErrorCode.INVALID_REQUEST, "Supply plaintext location names."
            )
        try:
            return self.index.resolve_many(names)
        except ValueError as exc:
            raise ProviderError(ProviderErrorCode.INVALID_REQUEST, str(exc)) from exc

    async def search(
        self, request: CatalogSearchRequest, context: ProviderRequestContext
    ) -> CatalogSearchResult:
        context.cancellation.raise_if_cancelled()
        if (
            request.bounding_box is not None
            or request.datetime_range is not None
            or (request.collections and request.collections != ("mock_locations",))
            or set(request.query) != {"names"}
        ):
            raise ProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "This toy catalog supports only plaintext names and the "
                "mock_locations collection, not spatial/temporal filters.",
            )
        names = request.query.get("names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                'The dataset catalog accepts query={"names": ["plaintext place", ...]}.',
            )
        locations = self._resolve_names([str(name) for name in names])
        url = SOURCE_BASE + "?" + urlencode([("name", location.name) for location in locations])
        return CatalogSearchResult(
            items=(
                CatalogItem(
                    id=hashlib.sha256(url.encode()).hexdigest(),
                    collection="mock_locations",
                    title="Location attributes selected by plaintext names",
                    assets=(
                        CatalogAsset(key="locations", href=url, media_type="application/geo+json"),
                    ),
                ),
            ),
            page_count=1,
            provenance=self.provenance(),
        )

    async def fetch(
        self, request: SourceSnapshotRequest, context: ProviderRequestContext
    ) -> RetrievedSource:
        context.cancellation.raise_if_cancelled()
        parsed = urlsplit(request.url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "mock-dataset.invalid"
            or parsed.path != "/locations"
            or request.format is not SourceFormat.GEOJSON
        ):
            raise ProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "Only the GeoJSON location URLs returned by the dataset catalog are available.",
            )
        locations = self._resolve_names(parse_qs(parsed.query).get("name", []))
        features = []
        for location in locations:
            properties = {
                "id": location.location_id,
                "name": location.name,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "opening_cost": 1,
            }
            if location.population is not None:
                properties["population"] = location.population
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [location.longitude, location.latitude],
                    },
                }
            )
        return RetrievedSource(
            content=json.dumps({"type": "FeatureCollection", "features": features}).encode(),
            media_type="application/geo+json",
            provenance=self.provenance(request.url),
        )


class DatasetRoutingProvider:
    """Serve directed OSM/OSRM road distances through the real routing contract."""

    def __init__(self, store: OsrmMatrixStore, region: str) -> None:
        self.store = store
        self.region = region
        self.retrieved_at = datetime.now(UTC)

    async def matrix(
        self, request: RouteMatrixRequest, context: ProviderRequestContext
    ) -> RouteMatrixResult:
        context.cancellation.raise_if_cancelled()
        if request.profile != "driving" or request.annotation is not RouteAnnotation.DISTANCE:
            raise ProviderError(
                ProviderErrorCode.INVALID_REQUEST,
                "The frozen OSRM evidence supplies driving distances in meters, not durations.",
            )
        master = self.store.region_locations[self.region]
        indices = []
        for longitude, latitude in request.coordinates:
            index = next(
                (
                    i
                    for i, location in enumerate(master)
                    if math.isclose(location.longitude, longitude, rel_tol=0, abs_tol=1e-8)
                    and math.isclose(location.latitude, latitude, rel_tol=0, abs_tol=1e-8)
                ),
                None,
            )
            if index is None:
                raise ProviderError(
                    ProviderErrorCode.INVALID_REQUEST,
                    "Coordinates are absent from the regional OSRM evidence; "
                    "resolve dataset locations first.",
                )
            indices.append(index)
        matrix = await self.store.region_matrix(self.region)
        context.cancellation.raise_if_cancelled()
        matrix_hash = hashlib.sha256(json.dumps(matrix, separators=(",", ":")).encode()).hexdigest()
        return RouteMatrixResult(
            values=tuple(
                tuple(matrix[indices[i]][indices[j]] for j in request.destination_indices)
                for i in request.source_indices
            ),
            source_ids=request.source_ids,
            destination_ids=request.destination_ids,
            units="meters",
            provenance=ProviderProvenance(
                provider="osrm_snapshot",
                source_uri=self.store.endpoint + "/table/v1/driving",
                retrieved_at=self.retrieved_at,
                provider_metadata={
                    "matrix_sha256": matrix_hash,
                    "region": self.region,
                    "source": "OSRM driving table over OpenStreetMap roads",
                },
            ),
        )
