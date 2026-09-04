from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from oasis.config import OasisSettings, RuntimeEngine


def test_fake_cli_completes_multi_turn_exchange() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oasis.cli",
            "chat",
            "--backend",
            "fake",
            "--prompt",
            "Hello",
            "--prompt",
            "Second turn",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Assistant: [fake] Hello" in completed.stdout
    assert "Assistant: [fake] Second turn" in completed.stdout
    assert "Usage: input=" in completed.stdout


def test_tools_cli_lists_describes_and_smokes_calculator(tmp_path: Path) -> None:
    base = [sys.executable, "-m", "oasis.cli", "tools", "--no-plugins"]

    listed = subprocess.run([*base, "list"], check=True, capture_output=True, text=True)
    described = subprocess.run(
        [*base, "describe", "calculator"], check=True, capture_output=True, text=True
    )
    smoked = subprocess.run(
        [
            *base,
            "smoke",
            "calculator",
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "calculator\t1.0.0" in listed.stdout
    assert '"name": "calculator"' in described.stdout
    assert '"status": "complete"' in smoked.stdout
    assert '"value": 5.0' in smoked.stdout


def test_evidence_cli_builds_frozen_demand_access_and_service_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "evidence"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oasis.cli",
            "evidence",
            "demo",
            "--artifact-root",
            str(artifact_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["demand_spec_artifact_id"].startswith("sha256-")
    assert result["candidate_spec_artifact_id"].startswith("sha256-")
    assert result["access_matrix_artifact_id"].startswith("sha256-")
    assert result["service_matrix_artifact_id"].startswith("sha256-")
    assert len(list(artifact_root.rglob("content"))) == 8


def test_decision_cli_runs_frozen_cooling_center_workflow(tmp_path: Path) -> None:
    artifact_root = tmp_path / "decision"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oasis.cli",
            "decision",
            "demo",
            "--artifact-root",
            str(artifact_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["problem_artifact_id"].startswith("sha256-")
    assert result["baseline_plan_artifact_id"].startswith("sha256-")
    assert result["best_scorecard_artifact_id"].startswith("sha256-")
    assert result["overall_metrics"]["coverage"] == 2 / 3
    assert result["group_metrics"]["older_adults"]["coverage"] == 4 / 9


def test_decision_cli_runs_frozen_mobile_vaccination_workflow(tmp_path: Path) -> None:
    artifact_root = tmp_path / "routing"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oasis.cli",
            "decision",
            "mobile-demo",
            "--artifact-root",
            str(artifact_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["problem_artifact_id"].startswith("sha256-")
    assert result["baseline_plan_artifact_id"].startswith("sha256-")
    assert result["best_scorecard_artifact_id"].startswith("sha256-")
    assert result["overall_metrics"]["served_value"] == 8.0
    assert result["scenario_metrics"]["normal"]["coverage"] == 8 / 15


def test_decision_cli_runs_fake_model_anytime_workflow(tmp_path: Path) -> None:
    artifact_root = tmp_path / "anytime"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oasis.cli",
            "decision",
            "anytime-demo",
            "--artifact-root",
            str(artifact_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["status"] == "complete"
    assert result["terminal_reason"] == "proven_optimal"
    assert result["best_plan_artifact_id"].startswith("sha256-")
    assert result["best_scorecard"]["feasible"] is True
    assert result["consumed_budget"]["model_usage"]["total_tokens"] > 0
    assert result["deadline_overshoot_ms"] == 0
    assert (artifact_root / "runs").is_dir()


def test_live_provider_smoke_requires_explicit_network_confirmation(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "oasis.cli",
            "evidence",
            "live-smoke",
            "--place",
            "Cambridge",
            "--source-url",
            "https://example.invalid/data.csv",
            "--source-format",
            "csv",
            "--source-license",
            "CC0",
            "--route-coordinate=-71.1,42.1",
            "--route-coordinate=-71.2,42.2",
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "requires --confirm-live-network" in completed.stderr
    assert not (tmp_path / "artifacts").exists()


def test_service_cli_exposes_phase8_configuration() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "oasis.cli", "serve", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--max-concurrent-runs" in completed.stdout
    assert "--max-request-bytes" in completed.stdout
    assert "--serve-ui" in completed.stdout


def test_track_b_showcase_cli_exposes_resumable_output() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "oasis.cli", "demo", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "complete offline Track B showcase release" in completed.stdout
    assert "--output" in completed.stdout


def test_service_cli_forwards_runtime_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    import oasis.api
    from oasis.cli import _parser, _serve

    observed: dict[str, object] = {}

    def fake_create_app(
        settings: OasisSettings,
        *,
        compute_inventory: object = None,
    ) -> object:
        observed["settings"] = settings
        observed["inventory"] = compute_inventory
        return object()

    monkeypatch.setattr(oasis.api, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)
    args = _parser().parse_args(
        [
            "serve",
            "--engine",
            "remote",
            "--remote-endpoint",
            "https://worker.example.test",
            "--quantization",
            "int8",
            "--attention-backend",
            "sdpa",
        ]
    )

    assert _serve(args) == 0
    settings = observed["settings"]
    assert isinstance(settings, OasisSettings)
    assert settings.runtime_engine is RuntimeEngine.REMOTE
    assert str(settings.remote_endpoint) == "https://worker.example.test/"
    assert settings.quantization == "int8"
    assert settings.attention_backend == "sdpa"
    assert observed["inventory"] is None


def test_hardware_and_runtime_cli_use_fake_inventories_without_cuda() -> None:
    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "oasis.cli",
            "hardware",
            "inspect",
            "--fixture",
            "local-5060ti-16gb",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    planned = subprocess.run(
        [
            sys.executable,
            "-m",
            "oasis.cli",
            "runtime",
            "plan",
            "--profile",
            "gemma4_e2b_it",
            "--device",
            "cuda",
            "--fixture",
            "local-5060ti-16gb",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    inventory = json.loads(inspected.stdout)
    result = json.loads(planned.stdout)
    assert inventory["discovery_mode"] == "fake"
    assert inventory["accelerator_count"] == 1
    assert result["dry_run"] is True
    assert result["plan"]["runtime"] == "cuda_transformers"
    assert result["plan"]["requested_model_id"] == "google/gemma-4-E2B-it"


def test_runtime_cli_dry_runs_single_machine_multi_gpu_fixtures() -> None:
    fixtures = ("2x24gb", "4x80gb")
    for fixture in fixtures:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "oasis.cli",
                "runtime",
                "plan",
                "--profile",
                "gemma4_e4b_it",
                "--device",
                "cuda",
                "--fixture",
                fixture,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        assert result["dry_run"] is True
        assert result["inventory"]["discovery_mode"] == "fake"
        assert result["plan"]["requested_model_id"] == "google/gemma-4-E4B-it"


def test_model_worker_cli_exposes_serve_health_and_capabilities() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "oasis.cli", "model-worker", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "serve" in completed.stdout
    assert "health" in completed.stdout
    assert "capabilities" in completed.stdout
