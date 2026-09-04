from __future__ import annotations

from pathlib import Path

import pytest

from oasis.controller import LocalRunStore, RunMetadata, RunStoreError


def test_local_run_store_rejects_unsafe_and_duplicate_ids(tmp_path: Path) -> None:
    store = LocalRunStore(tmp_path)
    with pytest.raises(RunStoreError, match="unsafe"):
        store.create(RunMetadata(run_id="../escape", problem_artifact_id="problem", seed=0))

    metadata = RunMetadata(run_id="safe-run", problem_artifact_id="problem", seed=3)
    store.create(metadata)
    assert store.read_metadata("safe-run") == metadata
    assert store.read_metadata("missing-run") is None
    with pytest.raises(RunStoreError, match="already exists"):
        store.create(metadata)


def test_local_run_store_has_no_result_before_finalization(tmp_path: Path) -> None:
    store = LocalRunStore(tmp_path)
    store.create(RunMetadata(run_id="pending", problem_artifact_id="problem", seed=0))

    assert store.read_metadata("pending") is not None
    assert store.read_events("pending") == ()
    assert store.read_result("pending") is None
