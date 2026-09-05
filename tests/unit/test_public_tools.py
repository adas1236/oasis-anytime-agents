"""CPU-only public-schema, translation, and resumable-search regressions."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from jinja2 import Environment
from shapely.geometry import Point

from oasis.artifacts import MatrixData, put_matrix, put_vector, read_json, read_matrix, read_vector
from oasis.llm.gemma_schema import gemma_tool_schema
from oasis.schemas import ToolErrorCode
from oasis.tools import create_public_tool_registry, create_tool_registry, invoke_tool
from unit.test_evidence_construction_access import context, provenance


def test_public_schema_is_small_and_native_gemma_renderable():
    public = create_public_tool_registry(discover_entry_points=False)
    advanced = create_tool_registry(discover_entry_points=False)
    expected = {
        "compile_max_coverage": 5,
        "compile_min_facilities": 5,
        "compile_tsp": 3,
        "improve": 4,
        "travel_matrix": 3,
        "service_matrix": 2,
        "build_demand": 3,
        "build_candidates": 2,
        "materialize_locations": 3,
    }
    for name, count in expected.items():
        schema = public.get(name).spec.input_schema
        assert len(schema["properties"]) == count
        assert schema["additionalProperties"] is False
    names = {s.name for s in public.list()}
    assert "compile_problem" not in names and "scenario_sweep" not in names
    assert {"resolve_locations", "derive_health_measure", "overlay_reduce"} <= names
    # The advanced interfaces still exist, but are not co-advertised to the model.
    assert len(advanced.get("compile_problem").spec.input_schema["properties"]) == 22

    def size(registry):
        return len(json.dumps([d.model_dump() for d in registry.model_definitions()]))

    assert size(public) < 0.75 * size(advanced)
    native = (
        Environment()
        .from_string(
            (Path(__file__).parents[1] / "fixtures/gemma4_schema_template.jinja").read_text()
        )
        .module
    )
    for definition in public.model_definitions():
        rendered = native.format_function_declaration(gemma_tool_schema(definition))
        assert 'type:<|"|><|"|>' not in rendered
        assert "$ref" not in rendered
    improve = public.get("improve").spec.input_schema
    assert "SearchResumeToken" not in json.dumps(improve)
    assert "Plan" not in json.dumps(improve)


@pytest.fixture
def setup(tmp_path):
    ctx = context(tmp_path)
    registry = create_public_tool_registry(discover_entry_points=False)
    frame = gpd.GeoDataFrame(
        {"id": ["a", "b", "c", "d"], "population": [100, 70, 50, 20]},
        geometry=[Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)],
        crs="EPSG:4326",
    )
    points = put_vector(ctx.artifact_store, frame, units="persons", provenance=provenance("points"))
    return ctx, registry, points


async def call(setup, name, **arguments):
    ctx, registry, _ = setup
    result = await invoke_tool(registry.get(name), arguments, ctx)
    assert result.error is None, result
    return result


async def facility_args(setup):
    _, _, points = setup
    demand = await call(setup, "build_demand", artifact_id=points.id, need_field="population")
    candidates = await call(setup, "build_candidates", artifact_id=points.id)
    access = await call(setup, "travel_matrix", origins_artifact_id=points.id, metric="haversine")
    service = await call(
        setup,
        "service_matrix",
        access_matrix_artifact_id=access.metrics["artifact_id"],
        threshold=120,
    )
    return {
        "demand": demand.metrics["demand_spec_artifact_id"],
        "candidates": candidates.metrics["candidate_spec_artifact_id"],
        "access_matrix": access.metrics["artifact_id"],
        "service_matrix": service.metrics["artifact_id"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("minimum", [False, True])
async def test_facility_compilation_auto_search_and_count_objective(setup, minimum):
    ctx, _, _ = setup
    args = await facility_args(setup)
    compiled = await call(
        setup,
        "compile_min_facilities" if minimum else "compile_max_coverage",
        **args,
        **({"coverage_target": 0.9} if minimum else {"site_limit": 1}),
    )
    assert compiled.metrics["feasible"]
    problem = read_json(ctx.artifact_store, compiled.metrics["problem_artifact_id"])
    assert problem["need_field"] == "population"
    if minimum:
        candidate = problem["candidates"]
        frame = read_vector(ctx.artifact_store, candidate["artifact"]["id"])
        assert list(frame[candidate["opening_cost_field"]]) == [1, 1, 1, 1]
        assert problem["policy"]["site_limit"] == 4
    improved = await call(
        setup, "improve", problem_artifact_id=compiled.metrics["problem_artifact_id"]
    )
    assert improved.metrics["strategy"] == "add_swap"
    assert improved.metrics["best_plan_artifact_id"]


@pytest.mark.asyncio
async def test_tsp_defaults_and_opaque_resume_preserve_strategy_and_problem(setup):
    ctx, registry, points = setup
    # Directed, non-metric weights: no symmetrization or straight-line substitution.
    values = np.array([[0, 3, 8, 2], [6, 0, 1, 7], [2, 6, 0, 1], [1, 3, 5, 0]], dtype=float)
    matrix = put_matrix(
        ctx.artifact_store,
        MatrixData(values=values, row_ids=("a", "b", "c", "d"), column_ids=("a", "b", "c", "d")),
        crs=None,
        units="kilometers",
        provenance=provenance("roads"),
    )
    compiled = await call(setup, "compile_tsp", nodes=points.id, travel_matrix=matrix.id, depot="a")
    problem_id = compiled.metrics["problem_artifact_id"]
    problem = read_json(ctx.artifact_store, problem_id)
    assert problem["policy"]["time_units"] == "kilometers"
    assert problem["policy"]["shift_length"] >= 4 * values.max()
    assert problem["policy"]["vehicle_count"] == 1
    assert problem["policy"]["require_return"]
    assert np.array_equal(read_matrix(ctx.artifact_store, matrix).values, values)
    automatic = await call(setup, "improve", problem_artifact_id=problem_id)
    assert automatic.metrics["strategy"] == "two_opt"
    partial = await call(
        setup,
        "improve",
        problem_artifact_id=problem_id,
        strategy="exact_enumeration",
        max_candidates=1,
    )
    token = partial.metrics["resume_token_artifact_id"]
    assert token
    resumed = await call(setup, "improve", problem_artifact_id=problem_id, resume_from=token)
    assert resumed.metrics["strategy"] == "exact_enumeration"
    assert resumed.metrics["complete"]
    other = await call(setup, "compile_tsp", nodes=points.id, travel_matrix=matrix.id, depot="b")
    for arguments in (
        {"problem_artifact_id": other.metrics["problem_artifact_id"], "resume_from": token},
        {"problem_artifact_id": problem_id, "strategy": "add_swap"},
        {"problem_artifact_id": problem_id, "resume_token": {}},
    ):
        result = await invoke_tool(registry.get("improve"), arguments, ctx)
        assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS


@pytest.mark.asyncio
async def test_compact_inputs_reject_ignored_constraints_and_wrong_artifact_ids(setup):
    ctx, registry, points = setup
    for arguments in (
        {"nodes": points.id, "travel_matrix": points.id, "depot": "a"},
        {"nodes": points.id, "travel_matrix": points.id, "depot": "a", "capacity": 10},
    ):
        result = await invoke_tool(registry.get("compile_tsp"), arguments, ctx)
        assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS
    args = await facility_args(setup)
    result = await invoke_tool(
        registry.get("compile_max_coverage"), {**args, "demand": points.id, "site_limit": 1}, ctx
    )
    assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS
