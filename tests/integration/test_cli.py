from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
