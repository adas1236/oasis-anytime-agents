from __future__ import annotations

import json
from pathlib import Path

import pytest

import oasis.runpod_experiments as runpod_module
from oasis.runpod_experiments import (
    PlanOverrides,
    _runner_arguments,
    build_plan,
    cost_report,
    launch_plan,
    load_plan,
    main,
    stop_pods,
    worker_main,
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


def test_tracked_full_plan_is_27_registry_jobs_with_200_rows_each() -> None:
    plan = build_plan(EXAMPLE_CONFIG)

    assert len(plan["jobs"]) == 27
    assert {selection["rows"] for selection in plan["selections"].values()} == {200}
    assert plan["experiment"]["tool_mode"] == "registry"
    assert plan["experiment"]["max_tool_rounds"] == 20
    assert all(job["pod_payload"]["env"]["OASIS_TOOL_MODE"] == "registry" for job in plan["jobs"])


def test_tracked_plan_settings_reach_evaluator_for_all_27_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oasis.mock_experiments import _validated_config, build_parser

    plan = build_plan(EXAMPLE_CONFIG)
    observed = set()
    for job in plan["jobs"]:
        assert job["pod_payload"]["image"].endswith(":runpod-eval-v5")
        with monkeypatch.context() as environment:
            for name, value in job["pod_payload"]["env"].items():
                environment.setenv(name, value)
            arguments = _runner_arguments(tmp_path / "results.jsonl")
            parser = build_parser()
            config = _validated_config(parser.parse_args(arguments[3:]), parser)
        assert config.tool_mode == "registry"
        assert config.max_tool_rounds == 20
        assert config.max_generated_tokens == 1536
        assert config.osrm_cache_only
        assert config.seed == 42 and config.limit == 200
        assert config.gpus == "auto"
        assert config.expected_selection_digest == job["selection_digest"]
        observed.add((config.dataset, config.time_budgets_seconds[0], config.token_budgets[0]))
    assert observed == {
        (dataset, seconds, tokens)
        for dataset in ("max_coverage", "minimum_facility", "tsp")
        for seconds in (30, 60, None)
        for tokens in (32_000, 256_000, None)
    }


def test_s3_sync_uploads_nested_traces_but_skips_unchanged_and_temporary_files(tmp_path: Path):
    sync = runpod_module.ArtifactSync(tmp_path, "")
    uploaded = []

    class Client:
        def upload_file(self, filename, bucket, key):
            uploaded.append(key)

    sync.client = Client()
    sync.prefix = "experiment"
    nested = tmp_path / "results.artifacts/case/budget"
    nested.mkdir(parents=True)
    trace = nested / "trace.jsonl"
    trace.write_text("{}\n")
    staging = nested / ".artifact-write"
    staging.mkdir()
    (staging / "metadata.json").write_text("{}")
    (nested / "metadata.json.tmp").write_text("{}")
    (nested / "alias").symlink_to(trace)
    sync.upload()
    assert uploaded == ["experiment/results.artifacts/case/budget/trace.jsonl"]
    sync.upload()
    assert len(uploaded) == 1
    trace.write_text("{}\n{}\n")
    sync.upload()
    assert len(uploaded) == 2


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
    assert all(job["pod_payload"]["gpu"]["count"] == 4 for job in plan["jobs"])
    assert all(job["pod_payload"]["gpu"]["id"] == "NVIDIA RTX A6000" for job in plan["jobs"])
    assert all(job["pod_payload"]["env"]["OASIS_EXPECTED_GPU_COUNT"] == "4" for job in plan["jobs"])
    assert all(
        job["pod_payload"]["env"]["OASIS_HOLD_AFTER_COMPLETION"] == "1" for job in plan["jobs"]
    )


def test_plan_uses_runpod_v2_payload_and_requires_one_gpu_type(tmp_path: Path) -> None:
    plan = build_plan(
        _write_config(tmp_path),
        PlanOverrides(time_budgets="unlimited", token_budgets="unlimited"),
    )

    payload = plan["jobs"][0]["pod_payload"]
    assert payload["image"] == "ghcr.io/example/oasis:test"
    assert payload["cloud"] == "SECURE"
    assert payload["disk"] == 60
    assert payload["gpu"] == {
        "id": "NVIDIA GeForce RTX 5090",
        "count": 1,
        "minVcpuCountPerGpu": 4,
        "minRamPerGpu": 16,
    }
    assert payload["mounts"] == {"persistent": {"size": 40, "path": "/workspace"}}
    assert "imageName" not in payload
    assert "gpuTypeIds" not in payload

    with pytest.raises(ValueError, match="exactly one gpu_type"):
        build_plan(
            _write_config(tmp_path),
            PlanOverrides(gpu_types=("NVIDIA RTX A5000", "NVIDIA RTX A6000")),
        )


def test_probe_plan_has_distinct_identity_and_worker_environment(tmp_path: Path) -> None:
    normal = build_plan(
        _write_config(tmp_path),
        PlanOverrides(time_budgets="unlimited", token_budgets="unlimited"),
    )
    probe = build_plan(
        _write_config(tmp_path),
        PlanOverrides(
            rows=1,
            time_budgets="unlimited",
            token_budgets="unlimited",
            probe_only=True,
        ),
    )

    assert probe["plan_id"] != normal["plan_id"]
    assert probe["experiment"]["probe_only"] is True
    assert all(job["pod_payload"]["env"]["OASIS_PROBE_ONLY"] == "1" for job in probe["jobs"])
    assert all(
        job["pod_payload"]["env"]["OASIS_IMAGE"] == "ghcr.io/example/oasis:test"
        for job in probe["jobs"]
    )


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
            return {
                "id": f"pod-{len(calls)}",
                "status": "PROVISIONING",
                "cost": 0.74,
                "createdAt": "2026-09-05T01:00:00Z",
                "startedAt": None,
            }

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


def test_stop_checkpoints_cost_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "plan_id": "plan-1",
                "plan_fingerprint": "fingerprint",
                "pods": {
                    "job-1": {
                        "pod_id": "pod-1",
                        "status": "RUNNING",
                        "cost_per_hour": 0.72,
                        "created_at": "2026-09-05T01:00:00+00:00",
                        "stopped_at": None,
                        "terminated_at": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str, object]] = []

    class FakeRunpodApi:
        def __init__(self, api_key: str, base_url: str) -> None:
            assert api_key == "test-api-key"

        def request(self, method: str, path: str, payload: object = None) -> object:
            calls.append((method, path, payload))
            if method == "GET":
                return {"id": "pod-1", "status": "RUNNING"}
            return {"id": "pod-1", "status": "EXITED", "cost": 0.0}

    monkeypatch.setattr(runpod_module, "RunpodApi", FakeRunpodApi)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-api-key")

    assert stop_pods(state_path=state_path, execute=True) == 0
    saved = json.loads(state_path.read_text(encoding="utf-8"))["pods"]["job-1"]
    assert saved["status"] == "EXITED"
    assert saved["stopped_at"] is not None
    assert isinstance(saved["estimated_compute_cost_usd"], float)
    assert calls == [
        ("GET", "/pods/pod-1", None),
        ("POST", "/pods/pod-1/action", {"action": "stop"}),
    ]

    assert stop_pods(state_path=state_path, execute=True) == 0
    assert len(calls) == 2


def test_cost_report_combines_state_and_downloaded_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "plan_id": "plan-1",
                "pods": {
                    "job-1": {
                        "pod_id": "pod-1",
                        "status": "EXITED",
                        "cost_per_hour": 0.6,
                        "created_at": "2026-09-05T01:00:00Z",
                        "stopped_at": "2026-09-05T01:02:00Z",
                        "terminated_at": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    status_dir = tmp_path / "results" / "plan-1" / "job-1"
    status_dir.mkdir(parents=True)
    (status_dir / "job-status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "mode": "probe",
                "image": "ghcr.io/example/oasis:test",
                "container_started_at": "2026-09-05T01:01:30Z",
                "started_at": "2026-09-05T01:01:35Z",
                "finished_at": "2026-09-05T01:01:40Z",
                "elapsed_seconds": 5.0,
            }
        ),
        encoding="utf-8",
    )

    assert cost_report(state_path=state_path, results_root=tmp_path / "results") == 0
    report = json.loads(capsys.readouterr().out)
    row = report["jobs"][0]
    assert row["image_pull_and_container_start_seconds"] == 90.0
    assert row["worker_setup_seconds"] == 5.0
    assert row["completion_to_stop_seconds"] == 20.0
    assert row["billed_seconds_estimate"] == 120.0
    assert row["estimated_compute_cost_usd"] == 0.02
    assert report["summary"]["complete"] is True


def test_probe_worker_records_hardware_without_starting_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    environment = {
        "OASIS_PLAN_ID": "probe-plan",
        "OASIS_JOB_ID": "probe-job",
        "OASIS_PROBE_ONLY": "1",
        "OASIS_EXPECTED_GPU_COUNT": "1",
        "OASIS_RESULTS_ROOT": str(tmp_path),
        "OASIS_RESULTS_S3_URI": "",
        "OASIS_SELECTION_DIGEST": "a" * 64,
        "OASIS_IMAGE": "ghcr.io/example/oasis:test",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        runpod_module,
        "_gpu_inventory",
        lambda: {
            "pod_id": "pod-1",
            "torch_version": "2.8.0+cu128",
            "cuda_runtime": "12.8",
            "cuda_visible_devices": "0",
            "devices": [{"index": 0, "name": "test GPU"}],
        },
    )

    assert worker_main() == 0
    status = json.loads(
        (tmp_path / "probe-plan" / "probe-job" / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "complete"
    assert status["mode"] == "probe"
    assert status["visible_gpu_count"] == 1
    assert not (tmp_path / "probe-plan" / "probe-job" / "results.jsonl").exists()
    assert "oasis_worker_finished" in capsys.readouterr().out


def test_worker_entrypoint_surfaces_setup_crashes_without_restart_loop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_worker() -> int:
        raise RuntimeError("test setup failure")

    monkeypatch.setattr(runpod_module, "worker_main", fail_worker)
    monkeypatch.setenv("OASIS_HOLD_AFTER_COMPLETION", "0")

    assert main(["worker"]) == 1
    event = json.loads(capsys.readouterr().err)
    assert event["event"] == "oasis_worker_crashed"
    assert event["error_type"] == "RuntimeError"
    assert event["error"] == "test setup failure"
    assert "fail_worker" in event["traceback_tail"]


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
        "OASIS_OSRM_CACHE": "/opt/oasis/osrm-cache",
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


def test_evaluation_worker_runs_subprocess_and_reports_durable_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = build_plan(
        _write_config(tmp_path),
        PlanOverrides(
            rows=1,
            time_budgets="unlimited",
            token_budgets="unlimited",
            gpu_count=1,
            model_type="fake",
        ),
    )
    environment = dict(plan["jobs"][0]["pod_payload"]["env"])
    environment.update(
        {
            "OASIS_EXPECTED_GPU_COUNT": "0",
            "OASIS_RESULTS_ROOT": str(tmp_path / "results"),
            "OASIS_RESULTS_S3_URI": "",
            "OASIS_OSRM_CACHE": str(DATA_ROOT.parent / "infra" / "runpod" / "osrm-cache"),
            "OASIS_OSRM_CACHE_ONLY": "1",
            "OASIS_HOLD_AFTER_COMPLETION": "0",
        }
    )
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        runpod_module,
        "_gpu_inventory",
        lambda: {
            "pod_id": None,
            "torch_version": "local-test",
            "cuda_runtime": None,
            "cuda_visible_devices": None,
            "devices": [],
        },
    )

    assert worker_main() == 0

    local_dir = tmp_path / "results" / plan["plan_id"] / plan["jobs"][0]["job_id"]
    status = json.loads((local_dir / "job-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["runner_exit_code"] == 0
    assert status["evaluation_summary"]["selected_records"] == 1
    assert status["evaluation_summary"]["completed_cells"] == 1
    assert status["evaluation_summary"]["errors"] == 0
    record = json.loads((local_dir / "results.jsonl").read_text().splitlines()[0])
    from oasis.tools import create_tool_registry

    expected_tools = [
        definition.name
        for definition in create_tool_registry(discover_entry_points=False).model_definitions()
    ]
    assert record["evaluation_protocol"] == "live_registry_v1"
    assert record["tool_names"] == expected_tools
    assert len(expected_tools) == 21
    assert "solve_current_problem" not in record["tool_names"]
    assert record["agent_plan_found"]
    events = capsys.readouterr().out
    assert "oasis_worker_finished" in events
    assert '"completed_cells": 1' in events
