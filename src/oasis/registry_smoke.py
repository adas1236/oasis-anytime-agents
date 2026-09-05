"""Offline, CPU-only acceptance check for the live-registry evaluation plumbing."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from oasis.mock_experiments import DATASET_FILES, _validated_config, build_parser, run_experiment
from oasis.tools import create_tool_registry


async def check(output: Path, data_root: Path, osrm_cache: Path, rows: int) -> None:
    expected_tools = {
        definition.name
        for definition in create_tool_registry(discover_entry_points=False).model_definitions()
    }
    for dataset in DATASET_FILES:
        parser = build_parser()
        config = _validated_config(
            parser.parse_args(
                [
                    "--dataset",
                    dataset,
                    "--data-root",
                    str(data_root),
                    "--tool-mode",
                    "registry",
                    "--model-type",
                    "fake",
                    "--gpus",
                    "none",
                    "--limit",
                    str(rows),
                    "--shuffle",
                    "--osrm-cache",
                    str(osrm_cache),
                    "--osrm-cache-only",
                    "--output",
                    str(output / f"{dataset}.jsonl"),
                    "--quiet",
                ]
            ),
            parser,
        )
        summary = await run_experiment(config)
        assert config.output is not None
        records = [json.loads(line) for line in config.output.read_text().splitlines()]
        if len(records) != rows or summary["errors"]:
            raise RuntimeError(f"{dataset}: incomplete or errored smoke run: {summary}")
        for record in records:
            if (
                not record["agent_plan_found"]
                or not record["objective_correct"]
                or record["terminal_reason"] != "completed"
                or set(record["tool_names"]) != expected_tools
            ):
                raise RuntimeError(
                    f"{dataset}: registry smoke failed for {record['record_id']}; "
                    f"see {config.output} for the trace and failures"
                )
        print(f"{dataset}: {rows} prompt-only scripted cases passed, {len(expected_tools)} tools")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation-output/registry-smoke"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--osrm-cache", type=Path, default=Path("infra/runpod/osrm-cache"))
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    asyncio.run(check(args.output, args.data_root, args.osrm_cache, args.rows))


if __name__ == "__main__":
    main()
