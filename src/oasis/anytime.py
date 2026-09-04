"""Public offline anytime-controller demonstration."""

from __future__ import annotations

import time
from pathlib import Path

from oasis.artifacts import LocalArtifactStore
from oasis.controller import AnytimeController, BudgetSpec, LocalRunStore, RunRequest, RunResult
from oasis.evidence import run_evidence_demo
from oasis.llm import FakeModelBackend, ToolCall
from oasis.schemas import ToolResult, ToolResultStatus
from oasis.tools import CancellationToken, ToolContext, create_tool_registry, invoke_tool


def _require_complete(result: ToolResult, name: str) -> ToolResult:
    if result.status is not ToolResultStatus.COMPLETE:
        detail = result.error.message if result.error is not None else str(result.summary)
        raise RuntimeError(f"anytime demo tool {name!r} failed: {detail}")
    return result


async def run_anytime_demo(
    artifact_root: str | Path,
    *,
    run_id: str | None = None,
    budget: BudgetSpec | None = None,
) -> RunResult:
    """Compile and exactly search a synthetic coverage problem through the controller."""

    root = Path(artifact_root)
    evidence = await run_evidence_demo(root)
    artifacts = LocalArtifactStore(root)
    tools = create_tool_registry(discover_entry_points=False)
    compile_context = ToolContext(
        run_id="phase7-anytime-compile",
        artifact_store=artifacts,
        deadline_monotonic=time.monotonic() + 30,
        cancellation=CancellationToken(),
        seed=0,
    )
    compiled = _require_complete(
        await invoke_tool(
            tools.get("compile_problem"),
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
            compile_context,
        ),
        "compile_problem",
    )
    problem_id = str(compiled.metrics["problem_artifact_id"])
    baseline_id = str(compiled.metrics["baseline_plan_artifact_id"])
    effective_run_id = run_id or f"phase7-anytime-{time.time_ns()}"
    effective_budget = budget or BudgetSpec(
        wall_time_ms=30_000,
        max_total_model_tokens=2_000,
        max_generated_tokens=256,
        max_tool_calls=2,
    )
    backend = FakeModelBackend(
        [
            ToolCall(
                id="exact-search",
                name="improve",
                arguments={"strategy": "exact_enumeration", "max_candidates": 1_000},
            )
        ]
    )
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=LocalRunStore(root / "runs"),
        backend=backend,
        tool_registry=tools,
    )
    return await controller.run(
        RunRequest(
            run_id=effective_run_id,
            problem_artifact_id=problem_id,
            baseline_plan_artifact_id=baseline_id,
            budget=effective_budget,
            seed=0,
        )
    )
