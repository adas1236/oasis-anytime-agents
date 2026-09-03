"""Deterministic travel, reachable-node isochrone, and service-response matrices."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from itertools import pairwise
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from pyproj import CRS, Geod

from oasis.artifacts import (
    ArtifactProvenance,
    MatrixData,
    put_json,
    put_matrix,
    read_graph,
    read_matrix,
    read_vector,
)
from oasis.providers.models import (
    ProviderError,
    ProviderErrorCode,
    ProviderRequestContext,
    RouteAnnotation,
    RouteMatrixRequest,
)
from oasis.providers.protocols import RoutingMatrixProvider
from oasis.schemas import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactRef,
    ArtifactTransformation,
    DeterminismClassification,
    QualitySummary,
    SideEffectClassification,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.evidence.common import (
    MISSING_ARTIFACT_ID,
    artifact_ref,
    child_provenance,
    invalid,
    require_fields,
    require_kind,
)
from oasis.tools.protocols import ToolContext


class TravelStrategy(StrEnum):
    EUCLIDEAN = "euclidean"
    GEODESIC = "geodesic"
    GRAPH_SHORTEST_PATH = "graph_shortest_path"
    ROUTED_PROVIDER = "routed_provider"


class DistanceUnits(StrEnum):
    METERS = "meters"
    KILOMETERS = "kilometers"


class TravelMatrixInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    origins_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    destinations_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    origin_id_field: str = "id"
    destination_id_field: str = "id"
    strategy: TravelStrategy
    output_units: str = "meters"
    graph_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    origin_graph_node_field: str | None = None
    destination_graph_node_field: str | None = None
    graph_weight_field: str = "weight"
    routing_profile: str | None = None
    route_annotation: RouteAnnotation | None = None

    @model_validator(mode="after")
    def graph_fields_match_strategy(self) -> TravelMatrixInput:
        graph_values = (
            self.graph_artifact_id,
            self.origin_graph_node_field,
            self.destination_graph_node_field,
        )
        route_values = (self.routing_profile, self.route_annotation)
        if self.strategy is TravelStrategy.GRAPH_SHORTEST_PATH:
            if any(value is None for value in graph_values):
                raise ValueError(
                    "graph travel requires graph artifact and origin/destination nodes"
                )
            if any(value is not None for value in route_values):
                raise ValueError("route fields are only valid for routed_provider")
        elif self.strategy is TravelStrategy.ROUTED_PROVIDER:
            if any(value is not None for value in graph_values):
                raise ValueError("graph fields are only valid for graph_shortest_path")
            if self.routing_profile is None or self.route_annotation is None:
                raise ValueError("routed travel requires a routing profile and annotation")
            if self.output_units != self.route_annotation.units:
                raise ValueError("routed output_units must match the annotation's canonical units")
        elif any(value is not None for value in (*graph_values, *route_values)):
            raise ValueError("graph/route fields do not apply to this travel strategy")
        return self


class TravelMatrixOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    origin_count: int
    destination_count: int
    strategy: TravelStrategy
    units: str
    directed: bool
    unreachable_count: int


class IsochroneRepresentation(StrEnum):
    REACHABLE_NODES = "reachable_nodes"


class IsochronesInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    graph_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    origin_node_ids: tuple[str, ...] = Field(min_length=1)
    cutoffs: tuple[float, ...] = Field(min_length=1)
    weight_field: str = "weight"
    representation: IsochroneRepresentation = IsochroneRepresentation.REACHABLE_NODES

    @model_validator(mode="after")
    def valid_cutoffs(self) -> IsochronesInput:
        if any(value < 0 or not math.isfinite(value) for value in self.cutoffs):
            raise ValueError("isochrone cutoffs must be finite and non-negative")
        if tuple(sorted(set(self.cutoffs))) != self.cutoffs:
            raise ValueError("isochrone cutoffs must be unique and increasing")
        return self


class IsochronesOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    origin_count: int
    cutoff_count: int
    reachable_set_count: int
    representation: IsochroneRepresentation


class ServiceStrategy(StrEnum):
    BINARY_THRESHOLD = "binary_threshold"
    PIECEWISE = "piecewise"
    EXPONENTIAL_DECAY = "exponential_decay"


class PiecewisePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    access: float = Field(ge=0.0)
    benefit: float = Field(ge=0.0, le=1.0)


class ServiceMatrixInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    access_matrix_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    strategy: ServiceStrategy
    threshold: float | None = Field(default=None, ge=0.0)
    piecewise_points: tuple[PiecewisePoint, ...] = ()
    decay_scale: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def parameters_match_strategy(self) -> ServiceMatrixInput:
        if self.strategy is ServiceStrategy.BINARY_THRESHOLD:
            if self.threshold is None or self.piecewise_points or self.decay_scale is not None:
                raise ValueError("binary threshold requires only threshold")
        elif self.strategy is ServiceStrategy.PIECEWISE:
            if (
                len(self.piecewise_points) < 2
                or self.threshold is not None
                or self.decay_scale is not None
            ):
                raise ValueError(
                    "piecewise service requires at least two points and no other parameter"
                )
            accesses = tuple(point.access for point in self.piecewise_points)
            benefits = tuple(point.benefit for point in self.piecewise_points)
            if tuple(sorted(set(accesses))) != accesses:
                raise ValueError("piecewise access values must be unique and increasing")
            if any(left < right for left, right in pairwise(benefits)):
                raise ValueError("piecewise benefit must be non-increasing with access")
        elif self.decay_scale is None or self.threshold is not None or self.piecewise_points:
            raise ValueError("exponential decay requires only decay_scale")
        return self


class ServiceMatrixOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    row_count: int
    column_count: int
    strategy: ServiceStrategy
    minimum_benefit: float
    maximum_benefit: float


def _point_frame(
    context: ToolContext, artifact_id: str, id_field: str
) -> tuple[ArtifactRef, gpd.GeoDataFrame, tuple[str, ...]]:
    reference = artifact_ref(context, artifact_id)
    require_kind(reference, {ArtifactKind.VECTOR})
    frame = read_vector(context.artifact_store, reference)
    require_fields(frame, [id_field])
    if frame.crs is None:
        invalid("travel matrix point artifacts require an explicit CRS")
    if frame[id_field].isna().any() or frame[id_field].astype(str).duplicated().any():
        invalid(f"travel matrix field {id_field!r} must contain unique non-missing IDs")
    if not frame.geometry.map(
        lambda geometry: (
            geometry is not None and not geometry.is_empty and geometry.geom_type == "Point"
        )
    ).all():
        invalid("travel matrix origins and destinations must be non-empty points")
    return reference, frame, tuple(frame[id_field].astype(str))


def _convert_meters(values: np.ndarray, output_units: str) -> np.ndarray:
    if output_units == DistanceUnits.METERS.value:
        return values
    if output_units == DistanceUnits.KILOMETERS.value:
        return values / 1_000
    invalid("Euclidean/geodesic output_units must be meters or kilometers")


def _euclidean(
    origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame, output_units: str
) -> np.ndarray:
    if origins.crs != destinations.crs:
        invalid("Euclidean travel requires matching CRS")
    crs = CRS.from_user_input(origins.crs)
    if crs.is_geographic:
        invalid("Euclidean travel requires a projected CRS; use geodesic for longitude/latitude")
    conversion = crs.axis_info[0].unit_conversion_factor if crs.axis_info else None
    if conversion is None:
        invalid("projected CRS does not declare a conversion to meters")
    origin_coordinates = np.array([(geometry.x, geometry.y) for geometry in origins.geometry])
    destination_coordinates = np.array(
        [(geometry.x, geometry.y) for geometry in destinations.geometry]
    )
    differences = origin_coordinates[:, np.newaxis, :] - destination_coordinates[np.newaxis, :, :]
    meters = np.sqrt(np.square(differences).sum(axis=2)) * conversion
    return _convert_meters(meters, output_units)


def _validate_lon_lat(frame: gpd.GeoDataFrame) -> None:
    for geometry in frame.geometry:
        if not (-180 <= geometry.x <= 180 and -90 <= geometry.y <= 90):
            invalid(
                "longitude/latitude coordinates are outside valid x/y ranges; check axis reversal"
            )


def _geodesic(
    origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame, output_units: str
) -> np.ndarray:
    origin_wgs84 = origins.to_crs("EPSG:4326")
    destination_wgs84 = destinations.to_crs("EPSG:4326")
    _validate_lon_lat(origin_wgs84)
    _validate_lon_lat(destination_wgs84)
    geod = Geod(ellps="WGS84")
    values = np.empty((len(origins), len(destinations)), dtype=np.float64)
    for row, origin in enumerate(origin_wgs84.geometry):
        for column, destination in enumerate(destination_wgs84.geometry):
            _, _, distance = geod.inv(origin.x, origin.y, destination.x, destination.y)
            values[row, column] = distance
    return _convert_meters(values, output_units)


def _graph_distances(
    graph: nx.Graph[str],
    origins: pd.Series,
    destinations: pd.Series,
    weight_field: str,
) -> np.ndarray:
    origin_nodes = tuple(str(value) for value in origins)
    destination_nodes = tuple(str(value) for value in destinations)
    unknown = sorted((set(origin_nodes) | set(destination_nodes)) - set(graph.nodes))
    if unknown:
        invalid("origin/destination references unknown graph nodes", context={"nodes": unknown})
    _validate_graph_weights(graph, weight_field)
    result = np.full((len(origin_nodes), len(destination_nodes)), np.inf, dtype=np.float64)
    for row, node in enumerate(origin_nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, node, weight=weight_field)
        except (TypeError, ValueError) as error:
            invalid(f"invalid graph weight field {weight_field!r}: {error}")
        for column, destination in enumerate(destination_nodes):
            if destination in lengths:
                result[row, column] = float(lengths[destination])
    return result


def _validate_graph_weights(graph: nx.Graph[str], weight_field: str) -> None:
    for _, _, attributes in graph.edges(data=True):
        raw_weight = attributes.get(weight_field)
        if not isinstance(raw_weight, (int, float)) or not math.isfinite(raw_weight):
            invalid(f"graph edge weights in {weight_field!r} must be finite numbers")
        if raw_weight < 0:
            invalid(f"graph edge weights in {weight_field!r} cannot be negative")


class TravelMatrixTool:
    """Build local or provider-routed labeled travel matrices."""

    version = "1.1.0"
    spec = ToolSpec(
        name="travel_matrix",
        version=version,
        description=(
            "Build a labeled travel matrix using projected Euclidean, WGS84 geodesic, directed "
            "graph shortest paths, or an explicitly configured routing provider."
        ),
        input_schema=TravelMatrixInput.model_json_schema(),
        output_schema=TravelMatrixOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "access", "travel", "offline", "online"}),
        problem_tags=frozenset({"location_allocation", "routing"}),
        artifact_tags=frozenset({ArtifactKind.VECTOR, ArtifactKind.GRAPH, ArtifactKind.MATRIX}),
        side_effects=SideEffectClassification.EXTERNAL_READ,
        determinism=DeterminismClassification.EXTERNAL,
        seed_description="local strategies ignore seed; routed results record provider provenance",
        runtime=ToolRuntimeEstimate(p50_ms=20, p95_ms=3_000),
        smoke_input={
            "origins_artifact_id": MISSING_ARTIFACT_ID,
            "destinations_artifact_id": MISSING_ARTIFACT_ID,
            "strategy": "geodesic",
        },
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = TravelMatrixInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        origin_ref, origins, origin_ids = _point_frame(
            context, request.origins_artifact_id, request.origin_id_field
        )
        destination_ref, destinations, destination_ids = _point_frame(
            context, request.destinations_artifact_id, request.destination_id_field
        )
        parents = [origin_ref, destination_ref]
        directed = False
        if request.strategy is TravelStrategy.EUCLIDEAN:
            values = _euclidean(origins, destinations, request.output_units)
        elif request.strategy is TravelStrategy.GEODESIC:
            values = _geodesic(origins, destinations, request.output_units)
        elif request.strategy is TravelStrategy.GRAPH_SHORTEST_PATH:
            assert request.graph_artifact_id is not None
            assert request.origin_graph_node_field is not None
            assert request.destination_graph_node_field is not None
            graph_ref = artifact_ref(context, request.graph_artifact_id)
            require_kind(graph_ref, {ArtifactKind.GRAPH})
            graph = read_graph(context.artifact_store, graph_ref)
            require_fields(origins, [request.origin_graph_node_field])
            require_fields(destinations, [request.destination_graph_node_field])
            values = _graph_distances(
                graph,
                origins[request.origin_graph_node_field],
                destinations[request.destination_graph_node_field],
                request.graph_weight_field,
            )
            if graph_ref.units != request.output_units:
                invalid(
                    "graph output_units must exactly match the graph edge-weight units",
                    context={"graph_units": graph_ref.units},
                )
            parents.append(graph_ref)
            directed = graph.is_directed()
        else:
            provider = context.providers.get("routing_matrix")
            if not isinstance(provider, RoutingMatrixProvider):
                invalid("routed_provider requires the routing_matrix provider handle")
            origin_wgs84 = origins.to_crs("EPSG:4326")
            destination_wgs84 = destinations.to_crs("EPSG:4326")
            _validate_lon_lat(origin_wgs84)
            _validate_lon_lat(destination_wgs84)
            coordinates = tuple(
                (float(point.x), float(point.y))
                for point in (*tuple(origin_wgs84.geometry), *tuple(destination_wgs84.geometry))
            )
            assert request.routing_profile is not None
            assert request.route_annotation is not None
            route_request = RouteMatrixRequest(
                coordinates=coordinates,
                source_indices=tuple(range(len(origin_ids))),
                destination_indices=tuple(
                    range(len(origin_ids), len(origin_ids) + len(destination_ids))
                ),
                source_ids=origin_ids,
                destination_ids=destination_ids,
                profile=request.routing_profile,
                annotation=request.route_annotation,
            )
            try:
                route_result = await provider.matrix(
                    route_request,
                    ProviderRequestContext(
                        deadline_monotonic=context.deadline_monotonic,
                        cancellation=context.cancellation,
                        monotonic=context.monotonic,
                    ),
                )
            except ProviderError as error:
                return ToolResult(
                    status=(
                        ToolResultStatus.RATE_LIMITED
                        if error.code is ProviderErrorCode.RATE_LIMITED
                        else ToolResultStatus.FAILED
                    ),
                    summary=str(error),
                    error=ToolError(
                        code=ToolErrorCode.PROVIDER_FAILURE,
                        message=str(error),
                        retryable=error.retryable,
                        context={"provider_code": error.code.value},
                    ),
                )
            context.cancellation.raise_if_cancelled()
            values = np.array(
                [
                    [np.inf if value is None else value for value in row]
                    for row in route_result.values
                ],
                dtype=np.float64,
            )
            if values.shape != (len(origin_ids), len(destination_ids)):
                invalid("routing provider returned a matrix with the wrong shape")
            directed = True
        unreachable = int(np.isinf(values).sum())
        quality = QualitySummary(
            warnings=(f"{unreachable} origin-destination pairs are unreachable",)
            if unreachable
            else ()
        )
        if request.strategy is TravelStrategy.ROUTED_PROVIDER:
            provenance = ArtifactProvenance(
                source_uri=route_result.provenance.source_uri,
                source_provider=route_result.provenance.provider,
                provider_metadata=route_result.provenance.provider_metadata,
                source_version=route_result.provenance.source_version,
                retrieved_at=route_result.provenance.retrieved_at,
                license=route_result.provenance.license,
                lineage=ArtifactLineage(
                    parent_ids=tuple(parent.id for parent in parents),
                    transformations=(
                        ArtifactTransformation(
                            name=self.spec.name,
                            version=self.version,
                            parameters=request.model_dump(mode="json", exclude_none=True),
                        ),
                    ),
                ),
            )
        else:
            provenance = child_provenance(self.spec.name, self.version, parents, request)
        matrix_ref = put_matrix(
            context.artifact_store,
            MatrixData(values=values, row_ids=origin_ids, column_ids=destination_ids),
            crs=None,
            units=request.output_units,
            provenance=provenance,
            quality=quality,
        )
        output = TravelMatrixOutput(
            artifact_id=matrix_ref.id,
            origin_count=len(origin_ids),
            destination_count=len(destination_ids),
            strategy=request.strategy,
            units=request.output_units,
            directed=directed,
            unreachable_count=unreachable,
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "matrix": matrix_ref.id,
                "shape": [len(origin_ids), len(destination_ids)],
                "units": request.output_units,
                "unreachable": unreachable,
            },
            artifacts=(matrix_ref,),
            metrics=output.model_dump(mode="json"),
        )


class IsochronesTool:
    """Return deterministic reachable node sets for graph-based local isochrones."""

    version = "1.0.0"
    spec = ToolSpec(
        name="isochrones",
        version=version,
        description=(
            "Compute graph-local isochrones as explicit reachable node sets; polygonal output is "
            "deferred until a topology-safe polygonization contract is available."
        ),
        input_schema=IsochronesInput.model_json_schema(),
        output_schema=IsochronesOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "access", "isochrone", "offline"}),
        problem_tags=frozenset({"location_allocation", "routing"}),
        artifact_tags=frozenset({ArtifactKind.GRAPH, ArtifactKind.JSON_SPECIFICATION}),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=20, p95_ms=3_000),
        smoke_input={
            "graph_artifact_id": MISSING_ARTIFACT_ID,
            "origin_node_ids": ["origin"],
            "cutoffs": [10],
        },
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = IsochronesInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        graph_ref = artifact_ref(context, request.graph_artifact_id)
        require_kind(graph_ref, {ArtifactKind.GRAPH})
        graph = read_graph(context.artifact_store, graph_ref)
        unknown = sorted(set(request.origin_node_ids) - set(graph.nodes))
        if unknown:
            invalid("isochrone origins reference unknown graph nodes", context={"nodes": unknown})
        _validate_graph_weights(graph, request.weight_field)
        sets: list[dict[str, JsonValue]] = []
        for origin in request.origin_node_ids:
            distances = nx.single_source_dijkstra_path_length(
                graph, origin, cutoff=max(request.cutoffs), weight=request.weight_field
            )
            for cutoff in request.cutoffs:
                reachable = sorted(
                    node for node, distance in distances.items() if distance <= cutoff
                )
                sets.append(
                    {
                        "origin_node_id": origin,
                        "cutoff": cutoff,
                        "reachable_node_ids": reachable,
                    }
                )
        data = {
            "representation": request.representation.value,
            "units": graph_ref.units,
            "sets": sets,
            "polygonal_isochrones": "deferred",
        }
        output_ref = put_json(
            context.artifact_store,
            data,
            kind=ArtifactKind.JSON_SPECIFICATION,
            units=graph_ref.units or "unitless",
            provenance=child_provenance(self.spec.name, self.version, [graph_ref], request),
            data_schema={"type": "reachable_node_isochrones", "version": self.version},
            row_count=len(sets),
        )
        output = IsochronesOutput(
            artifact_id=output_ref.id,
            origin_count=len(request.origin_node_ids),
            cutoff_count=len(request.cutoffs),
            reachable_set_count=len(sets),
            representation=request.representation,
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "artifact_id": output_ref.id,
                "representation": request.representation.value,
                "reachable_sets": len(sets),
            },
            artifacts=(output_ref,),
            metrics=output.model_dump(mode="json"),
        )


class ServiceMatrixTool:
    """Convert access impedance into bounded benefit without collapsing demand dimensions."""

    version = "1.0.0"
    spec = ToolSpec(
        name="service_matrix",
        version=version,
        description=(
            "Convert a labeled access matrix into binary-threshold, piecewise-linear, or "
            "exponential-decay service benefit in the closed interval zero to one."
        ),
        input_schema=ServiceMatrixInput.model_json_schema(),
        output_schema=ServiceMatrixOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "service", "offline"}),
        problem_tags=frozenset({"location_allocation", "routing"}),
        artifact_tags=frozenset({ArtifactKind.MATRIX}),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=5, p95_ms=1_000),
        smoke_input={
            "access_matrix_artifact_id": MISSING_ARTIFACT_ID,
            "strategy": "binary_threshold",
            "threshold": 10,
        },
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = ServiceMatrixInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        access_ref = artifact_ref(context, request.access_matrix_artifact_id)
        require_kind(access_ref, {ArtifactKind.MATRIX})
        access = read_matrix(context.artifact_store, access_ref)
        if (access.values < 0).any():
            invalid("access matrix cannot contain negative impedance")
        if request.strategy is ServiceStrategy.BINARY_THRESHOLD:
            assert request.threshold is not None
            values = (access.values <= request.threshold).astype(np.float64)
        elif request.strategy is ServiceStrategy.PIECEWISE:
            x = np.array([point.access for point in request.piecewise_points])
            y = np.array([point.benefit for point in request.piecewise_points])
            values = np.interp(access.values, x, y, left=y[0], right=y[-1])
            values[np.isinf(access.values)] = 0
        else:
            assert request.decay_scale is not None
            values = np.exp(-access.values / request.decay_scale)
        values[np.isnan(access.values)] = np.nan
        finite = values[np.isfinite(values)]
        minimum = float(finite.min()) if finite.size else 0.0
        maximum = float(finite.max()) if finite.size else 0.0
        service_ref = put_matrix(
            context.artifact_store,
            MatrixData(values=values, row_ids=access.row_ids, column_ids=access.column_ids),
            crs=None,
            units="unitless",
            provenance=child_provenance(self.spec.name, self.version, [access_ref], request),
        )
        output = ServiceMatrixOutput(
            artifact_id=service_ref.id,
            row_count=values.shape[0],
            column_count=values.shape[1],
            strategy=request.strategy,
            minimum_benefit=minimum,
            maximum_benefit=maximum,
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "matrix": service_ref.id,
                "strategy": request.strategy.value,
                "range": [minimum, maximum],
            },
            artifacts=(service_ref,),
            metrics=output.model_dump(mode="json"),
        )
