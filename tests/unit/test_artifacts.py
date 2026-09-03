from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from oasis.artifacts import ArtifactIntegrityError, LocalArtifactStore
from oasis.schemas import ArtifactKind, ArtifactMetadata, ArtifactRef


def metadata(*, media_type: str = "application/json") -> ArtifactMetadata:
    return ArtifactMetadata(
        kind=ArtifactKind.JSON_SPECIFICATION,
        media_type=media_type,
        units="unitless",
        data_schema={"type": "object"},
        source_uri="fixture://artifact",
        license="CC0-1.0",
    )


def test_artifact_store_deduplicates_and_round_trips_metadata(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    content = b'{"answer":42}'

    first = store.put_bytes(content, metadata())
    second = store.put_bytes(content, metadata())

    assert first == second
    assert first.id == f"sha256-{hashlib.sha256(content).hexdigest()}"
    assert store.get_metadata(first.id) == first
    assert store.read_bytes(first.id) == content
    assert ArtifactRef.model_validate_json(first.model_dump_json()) == first


def test_artifact_store_rejects_expected_hash_mismatch(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ArtifactIntegrityError, match="content hash mismatch"):
        store.put_bytes(b"actual", metadata(), expected_hash="0" * 64)


@pytest.mark.parametrize(
    "artifact_id",
    ["../metadata.json", "sha256-../../etc/passwd", "/tmp/file", "sha256-not-a-hash"],
)
def test_artifact_store_rejects_path_traversal(tmp_path: Path, artifact_id: str) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ArtifactIntegrityError, match="unsafe or invalid"):
        store.get_metadata(artifact_id)


def test_artifact_publication_renames_only_after_both_files_are_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    original_rename = Path.rename
    observed = False

    def checked_rename(source: Path, target: Path) -> Path:
        nonlocal observed
        if source.name.startswith(".artifact-"):
            observed = True
            assert (source / "content").is_file()
            assert (source / "metadata.json").is_file()
            assert not target.exists()
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", checked_rename)

    reference = store.put_bytes(b"atomic", metadata())

    assert observed
    assert store.exists(reference.id)
    assert not list(tmp_path.rglob(".artifact-*"))


def test_artifact_store_detects_tampered_content(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    reference = store.put_bytes(b"trusted", metadata())
    content_path = next(tmp_path.rglob("content"))
    content_path.write_bytes(b"altered")

    with pytest.raises(ArtifactIntegrityError):
        store.read_bytes(reference.id)


def test_same_content_cannot_mutate_immutable_metadata(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    store.put_bytes(b"same", metadata())

    with pytest.raises(ArtifactIntegrityError, match="different immutable metadata"):
        store.put_bytes(b"same", metadata(media_type="text/plain"))
