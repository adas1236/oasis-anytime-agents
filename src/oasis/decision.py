"""Public frozen cooling-center decision workflow for Phase 5."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from oasis.artifacts import LocalArtifactStore, read_json
from oasis.evidence import run_evidence_demo
from oasis.problems import Scorecard
from oasis.schemas import ToolEventKind, ToolResult, ToolResultStatus
from oasis.tools import (
    CancellationToken,
    ToolContext,
    create_tool_registry,
    invoke_tool,
    stream_tool,
)


class DecisionDemoResult(BaseModel):
    """Artifacts and authoritative metrics from the frozen cooling-center example."""

    model_config = ConfigDict(frozen=True)

    problem_artifact_id: str
    baseline_plan_artifact_id: str
    best_plan_artifact_id: str
    best_scorecard_artifact_id: str
    summary_artifact_id: str
    geojson_map_artifact_id: str
    svg_map_artifact_id: str
    overall_metrics: dict[str, float]
    group_metrics: dict[str, dict[str, float]]


def _require(result: ToolResult, name: str) -> ToolResult:
    if result.status not in {ToolResultStatus.COMPLETE, ToolResultStatus.PARTIAL}:
        detail = result.error.message if result.error is not None else str(result.summary)
        raise RuntimeError(f"decision demo tool {name!r} failed: {detail}")
    return result


async def run_decision_demo(artifact_root: str | Path) -> DecisionDemoResult:
    """Compile, seed, exactly search, score, summarize, and map a frozen instance."""

    evidence = await run_evidence_demo(artifact_root)
    store = LocalArtifactStore(artifact_root)
    registry = create_tool_registry(discover_entry_points=False)
    context = ToolContext(
        run_id="phase5-decision-demo",
        artifact_store=store,
        deadline_monotonic=time.monotonic() + 30,
        cancellation=CancellationToken(),
        seed=0,
    )
    compiled = _require(
        await invoke_tool(
            registry.get("compile_problem"),
            {
                "type_id": "max_weighted_coverage",
                "demand_spec_artifact_id": evidence.demand_spec_artifact_id,
                "candidate_spec_artifact_id": evidence.candidate_spec_artifact_id,
                "access_matrix_artifact_id": evidence.access_matrix_artifact_id,
                "service_matrix_artifact_ids": {"normal": evidence.service_matrix_artifact_id},
                "need_field": "population",
                "groups": [{"name": "older_adults", "field": "older_adult"}],
                "policy": {
                    "site_limit": 1,
                    "equity_objective": "floors",
                    "group_floors": {"older_adults": 0.4},
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
        raise RuntimeError("decision demo improvement ended without a result")
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
    maps: dict[str, str] = {}
    for map_format in ("geojson", "svg"):
        rendered = _require(
            await invoke_tool(
                registry.get("render_map"),
                {
                    "problem_artifact_id": problem_id,
                    "plan_artifact_id": best_plan_id,
                    "format": map_format,
                },
                context,
            ),
            "render_map",
        )
        maps[map_format] = str(rendered.metrics["map_artifact_id"])
    score = Scorecard.model_validate(read_json(store, best_score_id))
    return DecisionDemoResult(
        problem_artifact_id=problem_id,
        baseline_plan_artifact_id=baseline_id,
        best_plan_artifact_id=best_plan_id,
        best_scorecard_artifact_id=best_score_id,
        summary_artifact_id=str(summary.metrics["summary_artifact_id"]),
        geojson_map_artifact_id=maps["geojson"],
        svg_map_artifact_id=maps["svg"],
        overall_metrics=score.overall_metrics,
        group_metrics=score.group_metrics,
    )
