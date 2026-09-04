"""Public frozen mobile-vaccination routing workflow for Phase 6."""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from pydantic import BaseModel, ConfigDict
from shapely.geometry import Point

from oasis.artifacts import (
    ArtifactProvenance,
    LocalArtifactStore,
    MatrixData,
    put_matrix,
    put_vector,
    read_json,
)
from oasis.problems import Scorecard
from oasis.schemas import ToolEventKind, ToolResult, ToolResultStatus
from oasis.tools import (
    CancellationToken,
    ToolContext,
    create_tool_registry,
    invoke_tool,
    stream_tool,
)


class RoutingDemoResult(BaseModel):
    """Artifacts and authoritative metrics from the mobile-vaccination fixture."""

    model_config = ConfigDict(frozen=True)

    problem_artifact_id: str
    baseline_plan_artifact_id: str
    baseline_scorecard_artifact_id: str
    best_plan_artifact_id: str
    best_scorecard_artifact_id: str
    summary_artifact_id: str
    overall_metrics: dict[str, float]
    scenario_metrics: dict[str, dict[str, float]]


def _require(result: ToolResult, name: str) -> ToolResult:
    if result.status not in {ToolResultStatus.COMPLETE, ToolResultStatus.PARTIAL}:
        detail = result.error.message if result.error is not None else str(result.summary)
        raise RuntimeError(f"routing demo tool {name!r} failed: {detail}")
    return result


async def run_routing_demo(artifact_root: str | Path) -> RoutingDemoResult:
    """Compile, seed, improve, and independently score a frozen vaccination route."""

    store = LocalArtifactStore(artifact_root)
    provenance = ArtifactProvenance(
        source_uri="oasis://phase6/mobile-vaccination-fixture",
        source_provider="oasis",
        source_version="1.0.0",
        license="CC0-1.0",
    )
    node_ids = ("depot", "neighborhood-a", "neighborhood-b", "neighborhood-c")
    nodes = gpd.GeoDataFrame(
        {
            "node_id": node_ids,
            "priority": [0.0, 4.0, 3.0, 8.0],
            "demand": [0.0, 4.0, 3.0, 8.0],
            "service_minutes": [0.0, 5.0, 5.0, 5.0],
            "window_start": [0.0, 0.0, 0.0, 0.0],
            "window_end": [70.0, 50.0, 50.0, 60.0],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(8, 0)],
        crs="EPSG:3857",
    )
    node_ref = put_vector(store, nodes, units="people", provenance=provenance)
    travel = np.array(
        [
            [0.0, 2.0, 3.0, 15.0],
            [2.0, 0.0, 2.0, 14.0],
            [3.0, 2.0, 0.0, 13.0],
            [15.0, 14.0, 13.0, 0.0],
        ],
        dtype=np.float64,
    )
    travel_ref = put_matrix(
        store,
        MatrixData(values=travel, row_ids=node_ids, column_ids=node_ids),
        crs=None,
        units="minutes",
        provenance=provenance,
    )
    registry = create_tool_registry(discover_entry_points=False)
    context = ToolContext(
        run_id="phase6-routing-demo",
        artifact_store=store,
        deadline_monotonic=time.monotonic() + 30,
        cancellation=CancellationToken(),
        seed=0,
    )
    compiled = _require(
        await invoke_tool(
            registry.get("compile_problem"),
            {
                "type_id": "mobile_service_route",
                "nodes_artifact_id": node_ref.id,
                "node_id_field": "node_id",
                "prize_field": "priority",
                "demand_field": "demand",
                "service_time_field": "service_minutes",
                "window_start_field": "window_start",
                "window_end_field": "window_end",
                "travel_matrix_artifact_ids": {"normal": travel_ref.id},
                "policy": {
                    "depot_ids": ["depot"],
                    "vehicle_count": 1,
                    "shift_length": 70.0,
                    "time_units": "minutes",
                    "vehicle_capacity": 8.0,
                    "capacity_units": "people",
                },
            },
            context,
        ),
        "compile_problem",
    )
    problem_id = str(compiled.metrics["problem_artifact_id"])
    baseline_id = str(compiled.metrics["baseline_plan_artifact_id"])
    terminal = None
    async for event in stream_tool(
        registry.get("improve"),
        {
            "problem_artifact_id": problem_id,
            "starting_plan_artifact_id": baseline_id,
            "strategy": "exact_enumeration",
            "max_candidates": 100,
        },
        context,
    ):
        if event.kind is ToolEventKind.RESULT:
            terminal = event.result
    if terminal is None:
        raise RuntimeError("routing demo improvement ended without a result")
    improved = _require(terminal, "improve")
    best_plan_id = str(improved.metrics["best_plan_artifact_id"])
    best_score_id = str(improved.metrics["best_scorecard_artifact_id"])
    summary = _require(
        await invoke_tool(
            registry.get("summarize_plan"),
            {"problem_artifact_id": problem_id, "plan_artifact_id": best_plan_id},
            context,
        ),
        "summarize_plan",
    )
    score = Scorecard.model_validate(read_json(store, best_score_id))
    return RoutingDemoResult(
        problem_artifact_id=problem_id,
        baseline_plan_artifact_id=baseline_id,
        baseline_scorecard_artifact_id=str(compiled.metrics["baseline_scorecard_artifact_id"]),
        best_plan_artifact_id=best_plan_id,
        best_scorecard_artifact_id=best_score_id,
        summary_artifact_id=str(summary.metrics["summary_artifact_id"]),
        overall_metrics=score.overall_metrics,
        scenario_metrics=score.scenario_metrics,
    )
