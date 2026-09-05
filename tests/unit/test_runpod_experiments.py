from __future__ import annotations

import json
from pathlib import Path

import pytest

import oasis.runpod_experiments as runpod_module
from oasis.runpod_experiments import (
    PlanOverrides,
    _runner_arguments,
    build_plan,
    launch_plan,
    load_plan,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
EXAMPLE_CONFIG = DATA_ROOT.parent / "infra" / "runpod" / "experiment.toml"


def _write_config(tmp_path: Path, *, environment: str = "") -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(
        f"""
[experiment]
name = "test-grid"
data_root = {json.dumps(str(DATA_ROOT))}
datasets = ["max_coverage", "minimum_facility", "tsp"]
rows_per_dataset = 12
seed = 42
shards_per_condition = 1
time_budgets = ["30s", "60s", "unlimited"]
token_budgets = ["2k", "8k", "unlimited"]
model_type = "transformers"
profile = "gemma4_e2b_it"

[runpod]
image = "ghcr.io/example/oasis:test"
gpu_types = ["NVIDIA GeForce RTX 5090"]
gpu_count = 1
{environment}

[artifacts]
local_root = "/workspace/oasis-results"
s3_uri = "s3://test-results/oasis"
upload_interval_seconds = 60
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_plan_reuses_exact_selection_for_every_budget(tmp_path: Path) -> None:
    plan = build_plan(_write_config(tmp_path))

    assert len(plan["jobs"]) == 27
    for dataset, selection in plan["selections"].items():
        jobs = [job for job in plan["jobs"] if job["dataset"] == dataset]
        assert len(jobs) == 9
        assert {job["selection_digest"] for job in jobs} == {selection["selection_digest"]}
        assert all(job["record_ids"] == selection["record_ids"] for job in jobs)
        assert all(len(job["record_ids"]) == 12 for job in jobs)


def test_tracked_full_plan_is_27_jobs_with_500_rows_each() -> None:
    plan = build_plan(EXAMPLE_CONFIG)

    assert len(plan["jobs"]) == 27
    assert {selection["rows"] for selection in plan["selections"].values()} == {500}


def test_plan_shards_without_changing_selected_rows_and_overrides_gpus(
    tmp_path: Path,
) -> None:
    plan = build_plan(
        _write_config(tmp_path),
        PlanOverrides(
            rows=11,
            shards=2,
            time_budgets="unlimited",
            token_budgets="unlimited",
            gpu_types=("NVIDIA RTX A6000",),
            gpu_count=4,
        ),
    )

    assert len(plan["jobs"]) == 6
    assert plan["runpod"]["gpu_count"] == 4
    assert plan["runpod"]["gpu_types"] == ["NVIDIA RTX A6000"]
    for selection in plan["selections"].values():
        shard_ids = [
            record_id for shard in selection["shards"] for record_id in shard["record_ids"]
        ]
        assert shard_ids == selection["record_ids"]
        assert [shard["limit"] for shard in selection["shards"]] == [6, 5]
    assert all(job["pod_payload"]["gpuCount"] == 4 for job in plan["jobs"])
    assert all(job["pod_payload"]["env"]["OASIS_EXPECTED_GPU_COUNT"] == "4" for job in plan["jobs"])


def test_plan_refuses_literal_secrets(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        environment='\n[runpod.env]\nHF_TOKEN = "not-a-secret-reference"',
    )

    with pytest.raises(ValueError, match="literal secret"):
        build_plan(config)


def test_plan_refuses_worker_environment_overrides(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        environment='\n[runpod.env]\nOASIS_LIMIT = "999"',
    )

    with pytest.raises(ValueError, match="reserved worker variable"):
        build_plan(config)


def test_plan_fingerprint_detects_edits(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    plan = build_plan(_write_config(tmp_path))
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert load_plan(path)["plan_id"] == plan["plan_id"]

    plan["jobs"][0]["limit"] = 999
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_plan(path)


def test_launch_is_dry_by_default_and_needs_no_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    plan = build_plan(_write_config(tmp_path))
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    assert launch_plan(plan_path=plan_path, state_path=None, execute=False) == 0
    assert not plan_path.with_suffix(".state.json").exists()


def test_executed_launch_refuses_unfilled_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = (
        _write_config(tmp_path)
        .read_text(encoding="utf-8")
        .replace(
            's3_uri = "s3://test-results/oasis"',
            's3_uri = "s3://REPLACE_WITH_BUCKET/oasis"',
        )
    )
    config_path = tmp_path / "placeholder.toml"
    config_path.write_text(config, encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(build_plan(config_path)), encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "unused")

    with pytest.raises(ValueError, match=r"artifacts\.s3_uri"):
        launch_plan(plan_path=plan_path, state_path=None, execute=True)


def test_executed_launch_checkpoints_each_pod_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    state_path = tmp_path / "state.json"
    plan = build_plan(
        _write_config(tmp_path),
        PlanOverrides(time_budgets="unlimited", token_budgets="unlimited"),
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls: list[str] = []

    class FakeRunpodApi:
        def __init__(self, api_key: str, base_url: str) -> None:
            assert api_key == "test-api-key"

        def request(self, method: str, path: str, payload: object = None) -> object:
            assert method == "POST"
            assert path == "/pods"
            calls.append(path)
            return {"id": f"pod-{len(calls)}", "desiredStatus": "RUNNING"}

    monkeypatch.setattr(runpod_module, "RunpodApi", FakeRunpodApi)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-api-key")

    assert (
        launch_plan(
            plan_path=plan_path,
            state_path=state_path,
            execute=True,
            maximum=2,
        )
        == 0
    )
    assert len(calls) == 2
    assert len(json.loads(state_path.read_text(encoding="utf-8"))["pods"]) == 2

    assert (
        launch_plan(
            plan_path=plan_path,
            state_path=state_path,
            execute=True,
            maximum=2,
        )
        == 0
    )
    assert len(calls) == 2


def test_worker_arguments_include_selection_guard_and_auto_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {
        "OASIS_MODEL_TYPE": "transformers",
        "OASIS_MODEL_PROFILE": "gemma4_e2b_it",
        "OASIS_DATASET": "tsp",
        "OASIS_DTYPE": "bfloat16",
        "OASIS_QUANTIZATION": "none",
        "OASIS_ATTENTION_BACKEND": "sdpa",
        "OASIS_TIME_BUDGET": "60s",
        "OASIS_TOKEN_BUDGET": "8000",
        "OASIS_MAX_GENERATED_TOKENS": "768",
        "OASIS_MAX_TOOL_ROUNDS": "4",
        "OASIS_MAX_TOOL_CALLS": "unlimited",
        "OASIS_MODEL_CALL_TIMEOUT": "unlimited",
        "OASIS_START": "0",
        "OASIS_LIMIT": "20",
        "OASIS_SEED": "42",
        "OASIS_SELECTION_DIGEST": "a" * 64,
        "OASIS_OSRM_ENDPOINT": "https://router.project-osrm.org",
        "OASIS_TSP_TOLERANCE_KM": "1.0",
        "OASIS_THINKING": "1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    arguments = _runner_arguments(tmp_path / "results.jsonl")

    assert arguments[arguments.index("--gpus") + 1] == "auto"
    assert arguments[arguments.index("--expected-selection-digest") + 1] == "a" * 64
    assert arguments[arguments.index("--limit") + 1] == "20"
