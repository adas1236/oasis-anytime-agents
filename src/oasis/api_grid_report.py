"""Aggregate sharded hosted-model evaluation rows into one dataset/budget report.

The runner is serial per process, so a grid is executed as disjoint deterministic
slices of the same seeded shuffle. This module merges those shard files back into a
single report without re-deriving any score: it only reads the per-cell rows the
runner already wrote and independently graded.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# Rows are keyed by record and budget, so re-running an overlapping shard cannot
# double-count a cell.
CellKey = tuple[str, str]


def load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read every JSONL shard, keeping the last row written for each cell."""

    seen: dict[CellKey, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen[(str(row["record_id"]), str(row["budget_id"]))] = row
    return list(seen.values())


def _rate(rows: Sequence[dict[str, Any]], field: str) -> float:
    return sum(bool(row.get(field)) for row in rows) / len(rows) if rows else 0.0


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def _consumed(row: dict[str, Any]) -> dict[str, Any]:
    """Per-cell accounting lives under consumed_budget, not at the row top level."""

    consumed = row.get("consumed_budget")
    return consumed if isinstance(consumed, dict) else {}


def _group_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wall = [float(_consumed(row).get("wall_time_seconds") or 0.0) for row in rows]
    input_tokens = [int(_consumed(row).get("input_tokens") or 0) for row in rows]
    generated = [int(_consumed(row).get("generated_tokens") or 0) for row in rows]
    total_tokens = [int(_consumed(row).get("total_model_tokens") or 0) for row in rows]
    turns = [int(_consumed(row).get("model_generations") or 0) for row in rows]
    tool_calls = [int(_consumed(row).get("tool_calls") or 0) for row in rows]
    terminal: dict[str, int] = defaultdict(int)
    for row in rows:
        terminal[str(row.get("terminal_reason") or "unknown")] += 1
    return {
        "cells": len(rows),
        # `correct` is exact reference matching; `objective_correct` also accepts a
        # plan that reaches the reference objective with fewer sites than allowed.
        "accuracy": _rate(rows, "correct"),
        "objective_accuracy": _rate(rows, "objective_correct"),
        # Distinguishes a real model-produced feasible plan from the evaluator's
        # cheap baseline, which every cell starts from.
        "agent_plan_rate": _rate(rows, "agent_plan_found"),
        "error_rate": sum(row.get("error") is not None for row in rows) / len(rows)
        if rows
        else 0.0,
        "terminal_reasons": dict(sorted(terminal.items())),
        "wall_time_seconds": {
            "mean": round(statistics.fmean(wall), 3) if wall else 0.0,
            "p50": round(_quantile(wall, 0.5), 3),
            "p95": round(_quantile(wall, 0.95), 3),
        },
        "tokens": {
            "mean_input": round(statistics.fmean(input_tokens), 1) if input_tokens else 0.0,
            "mean_generated": round(statistics.fmean(generated), 1) if generated else 0.0,
            "mean_total": round(statistics.fmean(total_tokens), 1) if total_tokens else 0.0,
            "p95_total": round(_quantile([float(v) for v in total_tokens], 0.95), 1),
            "total_input": sum(input_tokens),
            "total_generated": sum(generated),
        },
        "mean_model_turns": round(statistics.fmean(turns), 2) if turns else 0.0,
        "mean_tool_calls": round(statistics.fmean(tool_calls), 2) if tool_calls else 0.0,
    }


def _by(rows: Sequence[dict[str, Any]], *fields: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["/".join(str(row.get(field)) for field in fields)].append(row)
    return {key: _group_metrics(value) for key, value in sorted(grouped.items())}


def build_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize overall, per-dataset, per-budget, and per-cell-type outcomes."""

    return {
        "protocol": sorted({str(row.get("evaluation_protocol")) for row in rows}),
        "overall": _group_metrics(rows),
        "by_dataset": _by(rows, "dataset"),
        "by_budget": _by(rows, "budget_id"),
        "by_dataset_and_budget": _by(rows, "dataset", "budget_id"),
    }


def format_table(report: dict[str, Any]) -> str:
    """Render the dataset x budget grid the experiment is designed to compare."""

    header = (
        f"{'dataset / budget':<44}{'cells':>6}{'exact':>8}{'objectv':>9}"
        f"{'plans':>7}{'p50 s':>8}{'p95 s':>8}{'tokens':>9}{'turns':>7}"
    )
    lines = [header, "-" * len(header)]

    def row(label: str, metrics: dict[str, Any]) -> str:
        return (
            f"{label:<44}{metrics['cells']:>6}{metrics['accuracy']:>8.3f}"
            f"{metrics['objective_accuracy']:>9.3f}{metrics['agent_plan_rate']:>7.3f}"
            f"{metrics['wall_time_seconds']['p50']:>8.1f}"
            f"{metrics['wall_time_seconds']['p95']:>8.1f}"
            f"{metrics['tokens']['mean_total']:>9.0f}{metrics['mean_model_turns']:>7.1f}"
        )

    for key, metrics in report["by_dataset_and_budget"].items():
        lines.append(row(key, metrics))
    lines.append("-" * len(header))
    for key, metrics in report["by_budget"].items():
        lines.append(row(f"ALL/{key}", metrics))
    lines.append(row("ALL", report["overall"]))
    return "\n".join(lines)


def _resolve(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for entry in inputs:
        if entry.is_dir():
            paths.extend(sorted(p for p in entry.glob("*.jsonl") if p.stat().st_size))
        elif entry.exists() and entry.stat().st_size:
            paths.append(entry)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Shard JSONL files or directories.")
    parser.add_argument("--output", type=Path, help="Write the report JSON here.")
    arguments = parser.parse_args(argv)

    paths = _resolve(arguments.inputs)
    if not paths:
        parser.error("no non-empty JSONL shard files were found")
    rows = load_rows(paths)
    if not rows:
        parser.error("the selected shard files contained no rows")
    report = build_report(rows)
    report["shard_files"] = [str(path) for path in paths]
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
