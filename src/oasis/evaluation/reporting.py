"""Resume-safe raw output and fully stratified aggregate benchmark reports."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from oasis.evaluation.metrics import descriptive_summary
from oasis.evaluation.models import (
    AggregateGroup,
    BenchmarkRunRecord,
    BenchmarkSummary,
    ComparisonKind,
    PairedDeltaSummary,
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_run_record(root: Path, record: BenchmarkRunRecord) -> None:
    """Atomically publish one complete run; partial process output is never a record."""

    _atomic_text(root / "runs" / f"{record.run_key}.json", record.model_dump_json(indent=2))


def write_manifest_copy(root: Path, content: str) -> None:
    """Atomically publish the normalized manifest that owns a result directory."""

    _atomic_text(root / "manifest.json", content)


def read_run_records(path: str | Path) -> tuple[BenchmarkRunRecord, ...]:
    """Read raw records from a results directory, JSONL stream, or one JSON document."""

    source = Path(path)
    if source.is_dir():
        files = sorted((source / "runs").glob("*.json"))
        return tuple(
            BenchmarkRunRecord.model_validate_json(item.read_text(encoding="utf-8"))
            for item in files
        )
    if source.suffix == ".jsonl":
        return tuple(
            BenchmarkRunRecord.model_validate_json(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    raw = json.loads(source.read_text(encoding="utf-8"))
    values = raw if isinstance(raw, list) else [raw]
    return tuple(BenchmarkRunRecord.model_validate(value) for value in values)


def _mean(values: Iterable[float]) -> float | None:
    mean, _, _ = descriptive_summary(values)
    return mean


def _flat_metric_means(values: Iterable[Mapping[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for mapping in values:
        for name, value in mapping.items():
            grouped[name].append(value)
    return {name: mean for name in sorted(grouped) if (mean := _mean(grouped[name])) is not None}


def _nested_metric_means(
    values: Iterable[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for mapping in values:
        for outer, inner in mapping.items():
            grouped[outer].append(inner)
    return {name: _flat_metric_means(grouped[name]) for name in sorted(grouped)}


def summarize_records(records: Sequence[BenchmarkRunRecord]) -> BenchmarkSummary:
    """Aggregate repetitions without pooling unlike recorded runtimes or hardware."""

    if not records:
        raise ValueError("at least one benchmark record is required")
    manifest_hashes = {record.manifest_hash for record in records}
    benchmark_ids = {record.benchmark_id for record in records}
    if len(manifest_hashes) != 1 or len(benchmark_ids) != 1:
        raise ValueError("summaries cannot mix benchmark manifests")
    grouped: dict[tuple[str, str, ComparisonKind, tuple[str, ...]], list[BenchmarkRunRecord]] = (
        defaultdict(list)
    )
    for record in records:
        grouped[
            (
                record.instance_id,
                record.budget_id,
                record.comparison,
                record.runtime_group_key,
            )
        ].append(record)
    aggregates: list[AggregateGroup] = []
    for (instance_id, budget_id, comparison, runtime_key), rows in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        feasible = [row for row in rows if row.feasible]
        quality_mean, quality_variance, quality_ci = descriptive_summary(
            row.comparator_key[0] for row in feasible
        )
        aggregates.append(
            AggregateGroup(
                instance_id=instance_id,
                problem_type=rows[0].problem_type,
                comparison=comparison,
                budget_id=budget_id,
                runtime_group_key=runtime_key,
                runs=len(rows),
                feasibility_rate=len(feasible) / len(rows),
                primary_quality_mean=quality_mean,
                primary_quality_sample_variance=quality_variance,
                primary_quality_ci95_half_width=quality_ci,
                raw_objective_means=_flat_metric_means(row.raw_objective for row in feasible),
                overall_metric_means=_flat_metric_means(row.overall_metrics for row in feasible),
                group_metric_means=_nested_metric_means(row.group_metrics for row in feasible),
                scenario_metric_means=_nested_metric_means(
                    row.scenario_metrics for row in feasible
                ),
                baseline_gain_mean=_mean(
                    row.baseline_gain for row in feasible if row.baseline_gain is not None
                ),
                reference_gap_mean=_mean(
                    row.reference_gap for row in feasible if row.reference_gap is not None
                ),
                time_to_first_feasible_ms_mean=_mean(
                    float(row.time_to_first_feasible_ms)
                    for row in feasible
                    if row.time_to_first_feasible_ms is not None
                ),
                deadline_violation_rate=sum(row.deadline_overshoot_ms > 0 for row in rows)
                / len(rows),
                deadline_overshoot_ms_mean=_mean(float(row.deadline_overshoot_ms) for row in rows)
                or 0.0,
                input_tokens_mean=_mean(float(row.input_tokens) for row in rows) or 0.0,
                generated_tokens_mean=_mean(float(row.generated_tokens) for row in rows) or 0.0,
                reasoning_tokens_mean=_mean(float(row.reasoning_tokens) for row in rows) or 0.0,
                total_model_tokens_mean=_mean(float(row.total_model_tokens) for row in rows) or 0.0,
                tool_calls_mean=_mean(float(row.tool_calls) for row in rows) or 0.0,
                tool_latency_ms_mean=_mean(float(row.tool_latency_ms) for row in rows) or 0.0,
                tool_failure_rate=sum(row.tool_failure_count > 0 for row in rows) / len(rows),
                model_failure_rate=sum(row.model_failure_count > 0 for row in rows) / len(rows),
                parse_repair_rate=sum(row.parse_repair_count > 0 for row in rows) / len(rows),
                auc_quality_log_time_mean=_mean(
                    row.auc_quality_log_time
                    for row in feasible
                    if row.auc_quality_log_time is not None
                ),
                auc_quality_log_tokens_mean=_mean(
                    row.auc_quality_log_tokens
                    for row in feasible
                    if row.auc_quality_log_tokens is not None
                ),
            )
        )

    baseline = {
        (row.pair_id, row.runtime_group_key): row
        for row in records
        if row.comparison is ComparisonKind.DETERMINISTIC_PORTFOLIO and row.feasible
    }
    deltas: dict[tuple[str, str, tuple[str, ...], ComparisonKind], list[float]] = defaultdict(list)
    for row in records:
        if (
            row.comparison
            in {
                ComparisonKind.DETERMINISTIC_BASELINE,
                ComparisonKind.DETERMINISTIC_PORTFOLIO,
            }
            or not row.feasible
        ):
            continue
        reference = baseline.get((row.pair_id, row.runtime_group_key))
        if reference is None:
            continue
        key = (row.instance_id, row.budget_id, row.runtime_group_key, row.comparison)
        deltas[key].append(row.comparator_key[0] - reference.comparator_key[0])
    paired: list[PairedDeltaSummary] = []
    for (instance_id, budget_id, runtime_key, contender), values in sorted(
        deltas.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        mean, variance, ci = descriptive_summary(values)
        assert mean is not None
        paired.append(
            PairedDeltaSummary(
                instance_id=instance_id,
                budget_id=budget_id,
                runtime_group_key=runtime_key,
                contender=contender,
                pairs=len(values),
                mean_primary_quality_delta=mean,
                sample_variance=variance,
                ci95_half_width=ci,
            )
        )
    return BenchmarkSummary(
        manifest_hash=next(iter(manifest_hashes)),
        benchmark_id=next(iter(benchmark_ids)),
        run_count=len(records),
        groups=tuple(aggregates),
        paired_deltas=tuple(paired),
    )


def _tabular(summary: BenchmarkSummary) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in summary.groups:
        row = group.model_dump(mode="json")
        row["runtime_group_key"] = "|".join(group.runtime_group_key)
        for field in (
            "raw_objective_means",
            "overall_metric_means",
            "group_metric_means",
            "scenario_metric_means",
        ):
            row[field] = json.dumps(row[field], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    return pd.DataFrame(rows)


def write_aggregate_outputs(root: Path, records: Sequence[BenchmarkRunRecord]) -> BenchmarkSummary:
    """Rebuild JSONL, summary JSON, CSV, and Parquet from atomic raw run documents."""

    summary = summarize_records(records)
    ordered = sorted(records, key=lambda item: item.run_key)
    jsonl = "".join(record.model_dump_json() + "\n" for record in ordered)
    _atomic_text(root / "runs.jsonl", jsonl)
    _atomic_text(root / "summary.json", summary.model_dump_json(indent=2))
    table = _tabular(summary)
    csv_text = table.to_csv(index=False)
    _atomic_text(root / "aggregate.csv", csv_text)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".aggregate.", suffix=".parquet", dir=root)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_parquet(temporary, index=False)
        temporary.replace(root / "aggregate.parquet")
    finally:
        if temporary.exists():
            temporary.unlink()
    return summary


def summarize_results(path: str | Path) -> BenchmarkSummary:
    """Read raw output and recompute its aggregate summary."""

    return summarize_records(read_run_records(path))
