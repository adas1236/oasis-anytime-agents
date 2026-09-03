"""Default test policy: CPU-only, offline, and deterministic."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def cpu_only_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CPU policy explicit without importing or probing PyTorch."""

    monkeypatch.setenv("OASIS_DEVICE", "cpu")
