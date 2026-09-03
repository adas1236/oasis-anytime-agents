from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import Point

from oasis.artifacts import (
    ArtifactProvenance,
    LocalArtifactStore,
    MatrixData,
    RasterData,
    put_graph,
    put_matrix,
    put_raster,
    put_table,
    put_vector,
    read_graph,
    read_matrix,
    read_raster,
    read_table,
    read_vector,
)
from oasis.schemas import ArtifactKind


def provenance() -> ArtifactProvenance:
    return ArtifactProvenance(source_uri="fixture://codecs", license="CC0-1.0")


def test_all_evidence_codecs_round_trip_with_explicit_metadata(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    vector = gpd.GeoDataFrame(
        {"id": ["a", "b"], "need": [1.0, 2.0]},
        geometry=[Point(0, 0), Point(1, 2)],
        crs="EPSG:4326",
    )
    table = pd.DataFrame({"id": ["a", "b"], "group": [0, 1]})
    raster = RasterData(
        values=np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32),
        transform=Affine.translation(0, 2) @ Affine.scale(1, -1),
        crs="EPSG:3857",
        nodata=-9999.0,
        band_names=("need",),
    )
    graph = nx.DiGraph()
    graph.add_node("a", x=0.0, y=0.0)
    graph.add_node("b", x=1.0, y=0.0)
    graph.add_edge("a", "b", minutes=3.0)
    matrix = MatrixData(
        values=np.array([[0.0, np.inf], [np.nan, 1.0]]),
        row_ids=("a", "b"),
        column_ids=("x", "y"),
    )

    vector_ref = put_vector(store, vector, units="persons", provenance=provenance())
    table_ref = put_table(store, table, crs=None, units="unitless", provenance=provenance())
    raster_ref = put_raster(store, raster, units="persons_per_cell", provenance=provenance())
    graph_ref = put_graph(store, graph, crs="EPSG:3857", units="minutes", provenance=provenance())
    matrix_ref = put_matrix(store, matrix, crs=None, units="minutes", provenance=provenance())

    assert vector_ref.kind is ArtifactKind.VECTOR
    assert vector_ref.crs == "EPSG:4326"
    assert vector_ref.units == "persons"
    assert list(read_vector(store, vector_ref)["id"]) == ["a", "b"]
    pd.testing.assert_frame_equal(read_table(store, table_ref), table)
    decoded_raster = read_raster(store, raster_ref)
    np.testing.assert_array_equal(decoded_raster.values, raster.values)
    assert decoded_raster.transform == raster.transform
    assert decoded_raster.band_names == ("need",)
    decoded_graph = read_graph(store, graph_ref)
    assert decoded_graph.is_directed()
    assert decoded_graph["a"]["b"]["minutes"] == 3.0
    decoded_matrix = read_matrix(store, matrix_ref)
    np.testing.assert_equal(decoded_matrix.values, matrix.values)
    assert decoded_matrix.row_ids == matrix.row_ids
    assert decoded_matrix.column_ids == matrix.column_ids


def test_codec_bytes_and_hashes_are_deterministic_across_stores(tmp_path: Path) -> None:
    vector = gpd.GeoDataFrame(
        {"id": ["one"], "value": [2.0]}, geometry=[Point(1, 2)], crs="EPSG:4326"
    )
    matrix = MatrixData(
        values=np.array([[42.0]], dtype=np.float64),
        row_ids=("one",),
        column_ids=("answer",),
    )
    table = pd.DataFrame({"id": ["one"], "value": [2.0]})
    raster = RasterData(
        values=np.array([[1.0]], dtype=np.float32),
        transform=Affine.translation(0, 1) @ Affine.scale(1, -1),
        crs="EPSG:3857",
        band_names=("value",),
    )
    graph = nx.DiGraph()
    graph.add_edge("one", "two", distance=2.0)
    first = LocalArtifactStore(tmp_path / "first")
    second = LocalArtifactStore(tmp_path / "second")

    first_vector = put_vector(first, vector, units="count", provenance=provenance())
    second_vector = put_vector(second, vector, units="count", provenance=provenance())
    first_matrix = put_matrix(first, matrix, crs=None, units="count", provenance=provenance())
    second_matrix = put_matrix(second, matrix, crs=None, units="count", provenance=provenance())
    first_table = put_table(first, table, crs=None, units="count", provenance=provenance())
    second_table = put_table(second, table, crs=None, units="count", provenance=provenance())
    first_raster = put_raster(first, raster, units="count", provenance=provenance())
    second_raster = put_raster(second, raster, units="count", provenance=provenance())
    first_graph = put_graph(first, graph, crs="EPSG:3857", units="meters", provenance=provenance())
    second_graph = put_graph(
        second, graph, crs="EPSG:3857", units="meters", provenance=provenance()
    )

    assert first_vector.id == second_vector.id
    assert first.read_bytes(first_vector.id) == second.read_bytes(second_vector.id)
    assert first_matrix.id == second_matrix.id
    assert first.read_bytes(first_matrix.id) == second.read_bytes(second_matrix.id)
    assert first_table.id == second_table.id
    assert first.read_bytes(first_table.id) == second.read_bytes(second_table.id)
    assert first_raster.id == second_raster.id
    assert first.read_bytes(first_raster.id) == second.read_bytes(second_raster.id)
    assert first_graph.id == second_graph.id
    assert first.read_bytes(first_graph.id) == second.read_bytes(second_graph.id)
