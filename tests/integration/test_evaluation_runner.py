from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from oasis.cli import _run
from oasis.controller import BudgetSpec
from oasis.evaluation import (
    BenchmarkBudget,
    BenchmarkInstanceSpec,
    BenchmarkManifest,
    BenchmarkRunRecord,
    ComparisonKind,
    EquityStructure,
    FixtureName,
    ProblemFamily,
    SpatialDistribution,
    SyntheticInstanceSpec,
    read_run_records,
    run_benchmark,
    summarize_records,
    summarize_results,
)
from oasis.runtimes import DiscoveryMode, RuntimeKind


def _manifest() -> BenchmarkManifest:
    location = SyntheticInstanceSpec(
        family=ProblemFamily.LOCATION_ALLOCATION,
        problem_type="max_weighted_coverage",
        seed=101,
        distribution=SpatialDistribution.GRID,
        demand_count=5,
        candidate_count=4,
        site_limit=2,
        equity_structure=EquityStructure.BALANCED,
    )
    return BenchmarkManifest(
        benchmark_id="phase10-test",
        instances=(
            BenchmarkInstanceSpec(id="location", generator=location),
            BenchmarkInstanceSpec(id="routing", fixture=FixtureName.MOBILE_ROUTING),
        ),
        comparisons=tuple(ComparisonKind),
        budgets=(
            BenchmarkBudget(
                id="small",
                resources=BudgetSpec(
                    wall_time_ms=20_000,
                    max_total_model_tokens=4_096,
                    max_generated_tokens=1_024,
                    max_tool_calls=8,
                ),
            ),
        ),
        run_seeds=(3, 7),
        expected_evaluator_versions={"*": "1.0.0"},
        max_candidates_per_action=1_000,
    )


@pytest.mark.asyncio
async def test_runner_resumes_and_writes_complete_two_family_reports(tmp_path: Path) -> None:
    manifest = _manifest()
    partial = await run_benchmark(manifest, tmp_path, stop_after_runs=3)
    assert partial.run_count == 3
    first_record_path = sorted((tmp_path / "runs").glob("*.json"))[0]
    first_record = first_record_path.read_bytes()

    summary = await run_benchmark(manifest, tmp_path)
    assert summary.run_count == 20
    assert len(summary.groups) == 10
    assert {group.problem_type for group in summary.groups} == {
        "max_weighted_coverage",
        "mobile_service_route",
    }
    assert summary.paired_deltas
    assert first_record_path.read_bytes() == first_record

    records = read_run_records(tmp_path)
    assert len(records) == len({record.run_key for record in records}) == 20
    assert all(record.feasible for record in records)
    assert all(record.checkpoints for record in records)
    assert all(record.schema_version == "1.1.0" for record in records)
    assert all(
        record.problem_hash and record.evidence_hash and record.policy_hash for record in records
    )
    assert all(record.artifact_count > 0 for record in records)
    assert all(record.artifact_bytes >= record.largest_artifact_bytes for record in records)
    assert all(record.runtime_group_key for record in records)
    assert all(record.runtime_plan.runtime is RuntimeKind.FAKE for record in records)
    assert all(
        record.compute_inventory.discovery_mode is DiscoveryMode.SAFE_CPU for record in records
    )
    assert all(record.compute_inventory.accelerator_count == 0 for record in records)
    assert any(len(record.checkpoints) > 1 for record in records)
    assert {record.comparison for record in records} == set(ComparisonKind)
    location_portfolio = [
        record
        for record in records
        if record.instance_id == "location"
        and record.comparison is ComparisonKind.DETERMINISTIC_PORTFOLIO
    ]
    assert all(
        record.comparator_key == pytest.approx((0.6422413793103449, 0.8192771084337349, -12.125537))
        for record in location_portfolio
    )
    routing_reference = (0.48148148148148145, 13.0, -122.40157700795915)
    assert all(
        record.comparator_key == pytest.approx(routing_reference)
        for record in records
        if record.instance_id == "routing"
        and record.comparison
        in {
            ComparisonKind.DETERMINISTIC_PORTFOLIO,
            ComparisonKind.ONE_SHOT_MODEL,
            ComparisonKind.ITERATIVE_MODEL,
        }
    )
    assert (tmp_path / "runs.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "aggregate.csv").is_file()
    assert (tmp_path / "aggregate.parquet").is_file()
    assert len(pd.read_csv(tmp_path / "aggregate.csv")) == 10
    assert len(pd.read_parquet(tmp_path / "aggregate.parquet")) == 10
    assert summarize_results(tmp_path) == summary
    assert _run(["summarize", str(tmp_path)]) == 0

    model_record = next(
        record for record in records if record.comparison is ComparisonKind.ONE_SHOT_MODEL
    )
    missing_metadata = model_record.model_dump(mode="json")
    missing_metadata["runtime_plan"]["runtime"] = "cpu_transformers"
    missing_metadata["model_profile"] = None
    missing_metadata["model_id"] = None
    missing_metadata["compute_inventory"]["library_versions"] = {}
    with pytest.raises(ValidationError, match="model identity metadata"):
        BenchmarkRunRecord.model_validate(missing_metadata)

    legacy = model_record.model_dump(mode="json")
    legacy["schema_version"] = "1.0.0"
    for field in (
        "problem_hash",
        "evidence_hash",
        "policy_hash",
        "model_action_input_tokens",
        "model_action_generated_tokens",
        "compact_context_bytes",
        "tool_latency_observations_ms",
        "tool_p50_estimates_ms",
        "tool_p95_estimates_ms",
        "artifact_count",
        "artifact_bytes",
        "largest_artifact_bytes",
    ):
        legacy.pop(field)
    assert BenchmarkRunRecord.model_validate(legacy).schema_version == "1.0.0"

    alternate_runtime = records[0].model_copy(
        update={
            "run_key": "f" * 64,
            "runtime_group_key": (*records[0].runtime_group_key, "different-hardware"),
        }
    )
    stratified = summarize_records((records[0], alternate_runtime))
    assert len(stratified.groups) == 2


@pytest.mark.asyncio
async def test_real_model_run_requires_separate_execution_confirmation(tmp_path: Path) -> None:
    raw = _manifest().model_dump(mode="json")
    raw["model"] = {
        "backend": "transformers",
        "profile": "gemma4_e2b_it",
        "allow_real_model": True,
    }
    manifest = BenchmarkManifest.model_validate(raw)
    with pytest.raises(ValueError, match="explicit approval"):
        await run_benchmark(manifest, tmp_path)
