"""Deterministic codecs for interoperable geospatial and numerical artifacts."""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from affine import Affine
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from shapely.geometry import mapping, shape

from oasis.artifacts.protocols import ArtifactStore
from oasis.schemas.artifacts import (
    ARTIFACT_METADATA_SCHEMA_VERSION,
    ArtifactKind,
    ArtifactLineage,
    ArtifactMetadata,
    ArtifactRef,
    PrivacyClassification,
    QualitySummary,
    SpatialExtent,
    TemporalExtent,
)


class ArtifactCodecError(ValueError):
    """Raised when artifact bytes do not satisfy their declared codec contract."""


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Source and lineage facts required whenever a codec publishes an artifact."""

    source_uri: str
    license: str
    lineage: ArtifactLineage = field(default_factory=ArtifactLineage)
    source_provider: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    source_version: str | None = None
    retrieved_at: datetime | None = None
    privacy: PrivacyClassification = PrivacyClassification.PUBLIC

    def __post_init__(self) -> None:
        if not self.source_uri or not self.license:
            raise ArtifactCodecError("artifact source URI and license must be explicit")


@dataclass(frozen=True, slots=True)
class RasterData:
    """In-memory raster values and georeferencing independent of rasterio datasets."""

    values: NDArray[np.generic]
    transform: Affine
    crs: str
    nodata: float | int | None = None
    band_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.values.ndim not in {2, 3}:
            raise ArtifactCodecError(
                "raster values must have shape (rows, cols) or (bands, rows, cols)"
            )
        band_count = 1 if self.values.ndim == 2 else self.values.shape[0]
        if self.band_names and len(self.band_names) != band_count:
            raise ArtifactCodecError("raster band names must match the number of bands")
        CRS.from_user_input(self.crs)


@dataclass(frozen=True, slots=True)
class MatrixData:
    """Labeled two-dimensional numerical matrix."""

    values: NDArray[np.float64]
    row_ids: tuple[str, ...]
    column_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ArtifactCodecError("matrix values must be two-dimensional")
        if self.values.shape != (len(self.row_ids), len(self.column_ids)):
            raise ArtifactCodecError("matrix labels must match its shape")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ArtifactCodecError("matrix row IDs must be unique")
        if len(set(self.column_ids)) != len(self.column_ids):
            raise ArtifactCodecError("matrix column IDs must be unique")


def _json_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON-compatible value with stable ordering and no non-finite numbers."""

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _embedded_descriptor(
    *, crs: str | None, units: str, provenance: ArtifactProvenance
) -> dict[str, object]:
    return {
        "metadata_schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "crs": crs,
        "units": units,
        "lineage": provenance.lineage.model_dump(mode="json"),
        "source_uri": provenance.source_uri,
        "source_provider": provenance.source_provider,
        "provider_metadata": provenance.provider_metadata,
        "source_version": provenance.source_version,
        "retrieved_at": (
            provenance.retrieved_at.isoformat() if provenance.retrieved_at is not None else None
        ),
    }


def _metadata(
    *,
    kind: ArtifactKind,
    media_type: str,
    crs: str | None,
    units: str,
    data_schema: dict[str, Any],
    provenance: ArtifactProvenance,
    quality: QualitySummary,
    spatial_extent: SpatialExtent | None = None,
    temporal_extent: TemporalExtent | None = None,
    row_count: int | None = None,
    cell_count: int | None = None,
    edge_count: int | None = None,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        kind=kind,
        media_type=media_type,
        crs=crs,
        units=units,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        data_schema=data_schema,
        row_count=row_count,
        cell_count=cell_count,
        edge_count=edge_count,
        source_uri=provenance.source_uri,
        source_provider=provenance.source_provider,
        provider_metadata=provenance.provider_metadata,
        license=provenance.license,
        retrieved_at=provenance.retrieved_at,
        source_version=provenance.source_version,
        lineage=provenance.lineage,
        quality=quality,
        privacy=provenance.privacy,
    )


def _frame_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "type": "table",
        "fields": [
            {"name": str(name), "dtype": str(dtype)}
            for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
        ],
    }


def _frame_quality(frame: pd.DataFrame) -> QualitySummary:
    missing_fraction = (
        0.0 if frame.empty or not len(frame.columns) else float(frame.isna().to_numpy().mean())
    )
    return QualitySummary(
        missing_fraction=missing_fraction,
        duplicate_count=int(frame.duplicated().sum()),
    )


def put_vector(
    store: ArtifactStore,
    frame: gpd.GeoDataFrame,
    *,
    units: str,
    provenance: ArtifactProvenance,
    quality: QualitySummary | None = None,
    temporal_extent: TemporalExtent | None = None,
) -> ArtifactRef:
    """Publish a deterministic GeoJSON feature collection."""

    if frame.crs is None:
        raise ArtifactCodecError("vector artifacts require an explicit CRS")
    crs = CRS.from_user_input(frame.crs).to_string()
    geometry_name = frame.geometry.name
    property_columns = sorted(str(column) for column in frame.columns if column != geometry_name)
    features: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        geometry = row[geometry_name]
        features.append(
            {
                "type": "Feature",
                "geometry": None if geometry is None else mapping(geometry),
                "properties": {column: _json_value(row[column]) for column in property_columns},
            }
        )
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
        "oasis:metadata": _embedded_descriptor(crs=crs, units=units, provenance=provenance),
    }
    encoded = canonical_json_bytes(payload)
    bounds = None if frame.empty else frame.total_bounds
    extent = (
        None
        if bounds is None or not np.isfinite(bounds).all()
        else SpatialExtent(west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3])
    )
    vector_quality = quality or QualitySummary(
        missing_fraction=_frame_quality(frame.drop(columns=geometry_name)).missing_fraction,
        invalid_geometry_count=int((~frame.geometry.is_valid & ~frame.geometry.isna()).sum()),
        duplicate_count=int(frame.drop(columns=geometry_name).duplicated().sum()),
    )
    schema = _frame_schema(frame.drop(columns=geometry_name))
    schema.update(
        {
            "geometry_column": geometry_name,
            "geometry_types": sorted(set(str(value) for value in frame.geom_type.dropna())),
        }
    )
    return store.put_bytes(
        encoded,
        _metadata(
            kind=ArtifactKind.VECTOR,
            media_type="application/geo+json",
            crs=crs,
            units=units,
            data_schema=schema,
            provenance=provenance,
            quality=vector_quality,
            spatial_extent=extent,
            temporal_extent=temporal_extent,
            row_count=len(frame),
        ),
    )


def read_vector(store: ArtifactStore, artifact: ArtifactRef | str) -> gpd.GeoDataFrame:
    """Decode a vector artifact and restore its declared CRS."""

    reference = store.get_metadata(artifact) if isinstance(artifact, str) else artifact
    if reference.kind is not ArtifactKind.VECTOR:
        raise ArtifactCodecError(f"expected vector artifact, received {reference.kind.value}")
    payload = json.loads(store.read_bytes(reference.id))
    if payload.get("type") != "FeatureCollection":
        raise ArtifactCodecError("vector artifact is not a GeoJSON feature collection")
    records: list[dict[str, object]] = []
    for feature in payload.get("features", []):
        properties = dict(feature.get("properties") or {})
        geometry = feature.get("geometry")
        properties["geometry"] = None if geometry is None else shape(geometry)
        records.append(properties)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=reference.crs)


def put_table(
    store: ArtifactStore,
    frame: pd.DataFrame,
    *,
    crs: str | None,
    units: str,
    provenance: ArtifactProvenance,
    quality: QualitySummary | None = None,
    temporal_extent: TemporalExtent | None = None,
) -> ArtifactRef:
    """Publish a deterministic Zstandard-compressed Parquet table."""

    normalized = frame.reset_index(drop=True).copy()
    table = pa.Table.from_pandas(normalized, preserve_index=False)
    descriptor = canonical_json_bytes(
        _embedded_descriptor(crs=crs, units=units, provenance=provenance)
    )
    table = table.replace_schema_metadata({b"oasis": descriptor})
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        write_statistics=False,
        version="2.6",
    )
    encoded = sink.getvalue().to_pybytes()
    return store.put_bytes(
        encoded,
        _metadata(
            kind=ArtifactKind.TABLE,
            media_type="application/vnd.apache.parquet",
            crs=crs,
            units=units,
            data_schema=_frame_schema(normalized),
            provenance=provenance,
            quality=quality or _frame_quality(normalized),
            temporal_extent=temporal_extent,
            row_count=len(normalized),
        ),
    )


def read_table(store: ArtifactStore, artifact: ArtifactRef | str) -> pd.DataFrame:
    """Decode a Parquet table without preserving implementation-specific indexes."""

    reference = store.get_metadata(artifact) if isinstance(artifact, str) else artifact
    if reference.kind is not ArtifactKind.TABLE:
        raise ArtifactCodecError(f"expected table artifact, received {reference.kind.value}")
    table = pq.read_table(pa.BufferReader(store.read_bytes(reference.id)))
    return cast(pd.DataFrame, table.to_pandas())


def put_raster(
    store: ArtifactStore,
    raster: RasterData,
    *,
    units: str,
    provenance: ArtifactProvenance,
    quality: QualitySummary | None = None,
) -> ArtifactRef:
    """Publish a deterministic compressed GeoTIFF."""

    values = raster.values[np.newaxis, ...] if raster.values.ndim == 2 else raster.values
    count, height, width = values.shape
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=str(values.dtype),
            crs=raster.crs,
            transform=raster.transform,
            nodata=raster.nodata,
            compress="deflate",
        ) as dataset:
            dataset.write(values)
            dataset.update_tags(
                oasis=canonical_json_bytes(
                    {
                        **_embedded_descriptor(
                            crs=CRS.from_user_input(raster.crs).to_string(),
                            units=units,
                            provenance=provenance,
                        ),
                        "band_names": list(raster.band_names),
                    }
                ).decode("utf-8")
            )
        encoded = memory.read()
    west, south, east, north = array_bounds(height, width, raster.transform)
    west, east = sorted((west, east))
    south, north = sorted((south, north))
    nodata_mask = np.isnan(values) if raster.nodata is None else values == raster.nodata
    raster_quality = quality or QualitySummary(missing_fraction=float(nodata_mask.mean()))
    return store.put_bytes(
        encoded,
        _metadata(
            kind=ArtifactKind.RASTER,
            media_type="image/tiff; application=geotiff",
            crs=CRS.from_user_input(raster.crs).to_string(),
            units=units,
            data_schema={
                "type": "raster",
                "dtype": str(values.dtype),
                "shape": [count, height, width],
                "band_names": list(raster.band_names),
                "nodata": raster.nodata,
            },
            provenance=provenance,
            quality=raster_quality,
            spatial_extent=SpatialExtent(west=west, south=south, east=east, north=north),
            cell_count=count * height * width,
        ),
    )


def read_raster(store: ArtifactStore, artifact: ArtifactRef | str) -> RasterData:
    """Decode GeoTIFF values and georeferencing into a closed in-memory object."""

    reference = store.get_metadata(artifact) if isinstance(artifact, str) else artifact
    if reference.kind is not ArtifactKind.RASTER:
        raise ArtifactCodecError(f"expected raster artifact, received {reference.kind.value}")
    with MemoryFile(store.read_bytes(reference.id)) as memory, memory.open() as dataset:
        values = dataset.read()
        descriptor = json.loads(dataset.tags().get("oasis", "{}"))
        band_names = tuple(str(value) for value in descriptor.get("band_names", []))
        data = values[0] if values.shape[0] == 1 else values
        return RasterData(
            values=data,
            transform=dataset.transform,
            crs=dataset.crs.to_string() if dataset.crs is not None else reference.crs or "",
            nodata=dataset.nodata,
            band_names=band_names,
        )


def put_graph(
    store: ArtifactStore,
    graph: nx.Graph[Any],
    *,
    crs: str | None,
    units: str,
    provenance: ArtifactProvenance,
    quality: QualitySummary | None = None,
) -> ArtifactRef:
    """Publish a directedness-preserving node-link JSON graph."""

    nodes = [
        {"id": str(node), "attributes": _json_value(attributes)}
        for node, attributes in sorted(graph.nodes(data=True), key=lambda item: str(item[0]))
    ]
    edges: list[dict[str, object]] = []
    if graph.is_multigraph():
        for source, target, key, attributes in cast(nx.MultiGraph[Any], graph).edges(
            data=True, keys=True
        ):
            edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "key": str(key),
                    "attributes": _json_value(attributes),
                }
            )
    else:
        for source, target, attributes in graph.edges(data=True):
            edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "attributes": _json_value(attributes),
                }
            )
    edges.sort(key=canonical_json_bytes)
    payload = {
        "format": "oasis-node-link-v1",
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "nodes": nodes,
        "edges": edges,
        "metadata": _embedded_descriptor(crs=crs, units=units, provenance=provenance),
    }
    encoded = canonical_json_bytes(payload)
    return store.put_bytes(
        encoded,
        _metadata(
            kind=ArtifactKind.GRAPH,
            media_type="application/vnd.oasis.graph+json",
            crs=crs,
            units=units,
            data_schema={
                "type": "graph",
                "directed": graph.is_directed(),
                "multigraph": graph.is_multigraph(),
                "node_attribute_fields": sorted(
                    {str(key) for _, attributes in graph.nodes(data=True) for key in attributes}
                ),
                "edge_attribute_fields": sorted(
                    {str(key) for _, _, attributes in graph.edges(data=True) for key in attributes}
                ),
            },
            provenance=provenance,
            quality=quality or QualitySummary(),
            row_count=graph.number_of_nodes(),
            edge_count=graph.number_of_edges(),
        ),
    )


def read_graph(store: ArtifactStore, artifact: ArtifactRef | str) -> nx.Graph[str]:
    """Decode the canonical graph representation with string node identities."""

    reference = store.get_metadata(artifact) if isinstance(artifact, str) else artifact
    if reference.kind is not ArtifactKind.GRAPH:
        raise ArtifactCodecError(f"expected graph artifact, received {reference.kind.value}")
    payload = json.loads(store.read_bytes(reference.id))
    if payload.get("format") != "oasis-node-link-v1":
        raise ArtifactCodecError("unsupported graph artifact format")
    directed = bool(payload["directed"])
    multigraph = bool(payload["multigraph"])
    if directed and multigraph:
        graph: nx.Graph[str] = nx.MultiDiGraph()
    elif directed:
        graph = nx.DiGraph()
    elif multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    for node in payload["nodes"]:
        graph.add_node(str(node["id"]), **dict(node["attributes"]))
    for edge in payload["edges"]:
        if multigraph:
            cast(nx.MultiGraph[str], graph).add_edge(
                str(edge["source"]),
                str(edge["target"]),
                key=str(edge["key"]),
                **dict(edge["attributes"]),
            )
        else:
            graph.add_edge(str(edge["source"]), str(edge["target"]), **dict(edge["attributes"]))
    return graph


def _zip_entry(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    info.create_system = 3
    return info


def put_matrix(
    store: ArtifactStore,
    matrix: MatrixData,
    *,
    crs: str | None,
    units: str,
    provenance: ArtifactProvenance,
    quality: QualitySummary | None = None,
) -> ArtifactRef:
    """Publish a deterministic compressed NumPy matrix with stable labels."""

    array_buffer = io.BytesIO()
    np.lib.format.write_array(array_buffer, matrix.values, allow_pickle=False)
    descriptor = canonical_json_bytes(
        {
            "row_ids": matrix.row_ids,
            "column_ids": matrix.column_ids,
            **_embedded_descriptor(crs=crs, units=units, provenance=provenance),
        }
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as output:
        output.writestr(_zip_entry("values.npy"), array_buffer.getvalue())
        output.writestr(_zip_entry("metadata.json"), descriptor)
    encoded = archive.getvalue()
    return store.put_bytes(
        encoded,
        _metadata(
            kind=ArtifactKind.MATRIX,
            media_type="application/vnd.oasis.matrix+npz",
            crs=crs,
            units=units,
            data_schema={
                "type": "matrix",
                "dtype": str(matrix.values.dtype),
                "shape": list(matrix.values.shape),
                "row_id_count": len(matrix.row_ids),
                "column_id_count": len(matrix.column_ids),
            },
            provenance=provenance,
            quality=quality
            or QualitySummary(
                missing_fraction=(
                    float(np.isnan(matrix.values).mean()) if matrix.values.size else 0.0
                )
            ),
            row_count=matrix.values.shape[0],
            cell_count=matrix.values.size,
        ),
    )


def read_matrix(store: ArtifactStore, artifact: ArtifactRef | str) -> MatrixData:
    """Decode a deterministic compressed matrix and its stable row/column labels."""

    reference = store.get_metadata(artifact) if isinstance(artifact, str) else artifact
    if reference.kind is not ArtifactKind.MATRIX:
        raise ArtifactCodecError(f"expected matrix artifact, received {reference.kind.value}")
    with zipfile.ZipFile(io.BytesIO(store.read_bytes(reference.id))) as archive:
        with archive.open("values.npy") as values_file:
            values = np.load(values_file, allow_pickle=False).astype(np.float64, copy=False)
        descriptor = json.loads(archive.read("metadata.json"))
    return MatrixData(
        values=values,
        row_ids=tuple(str(value) for value in descriptor["row_ids"]),
        column_ids=tuple(str(value) for value in descriptor["column_ids"]),
    )


def put_json(
    store: ArtifactStore,
    value: object,
    *,
    kind: ArtifactKind,
    units: str,
    provenance: ArtifactProvenance,
    data_schema: dict[str, Any],
    row_count: int | None = None,
    quality: QualitySummary | None = None,
) -> ArtifactRef:
    """Publish a small canonical JSON specification, profile, or reachable-node result."""

    payload = {
        "data": _json_value(value),
        "oasis:metadata": _embedded_descriptor(crs=None, units=units, provenance=provenance),
    }
    return store.put_bytes(
        canonical_json_bytes(payload),
        _metadata(
            kind=kind,
            media_type="application/json",
            crs=None,
            units=units,
            data_schema=data_schema,
            provenance=provenance,
            quality=quality or QualitySummary(),
            row_count=row_count,
        ),
    )


def read_json(store: ArtifactStore, artifact: ArtifactRef | str) -> object:
    """Read the data component of a canonical JSON artifact."""

    reference = store.get_metadata(artifact) if isinstance(artifact, str) else artifact
    payload = json.loads(store.read_bytes(reference.id))
    if not isinstance(payload, dict) or "data" not in payload:
        raise ArtifactCodecError("JSON artifact is missing its canonical data envelope")
    return payload["data"]
