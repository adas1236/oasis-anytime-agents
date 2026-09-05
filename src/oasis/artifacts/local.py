"""Atomic content-addressed artifact storage on a local filesystem."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from oasis.artifacts.protocols import ArtifactIntegrityError, ArtifactNotFoundError
from oasis.schemas.artifacts import ArtifactMetadata, ArtifactRef

_ID_PATTERN: Final = re.compile(r"^sha256-([0-9a-f]{64})$")
_CONTENT_NAME: Final = "content"
_METADATA_NAME: Final = "metadata.json"


class LocalArtifactStore:
    """Store each complete artifact in one atomically published directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def _parse_id(artifact_id: str) -> str:
        match = _ID_PATTERN.fullmatch(artifact_id)
        if match is None:
            raise ArtifactIntegrityError(f"unsafe or invalid artifact id: {artifact_id!r}")
        return match.group(1)

    def _artifact_dir(self, artifact_id: str) -> Path:
        content_hash = self._parse_id(artifact_id)
        path = (self._root / "objects" / content_hash[:2] / artifact_id).resolve()
        objects_root = (self._root / "objects").resolve()
        if not path.is_relative_to(objects_root):
            raise ArtifactIntegrityError("artifact path escapes the configured store")
        return path

    @staticmethod
    def _hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _sync_file(path: Path) -> None:
        with path.open("rb") as file_handle:
            os.fsync(file_handle.fileno())

    def put_bytes(
        self,
        content: bytes,
        metadata: ArtifactMetadata,
        *,
        expected_hash: str | None = None,
    ) -> ArtifactRef:
        """Verify, deduplicate, and atomically publish content plus metadata."""

        content_hash = self._hash(content)
        if expected_hash is not None and expected_hash != content_hash:
            raise ArtifactIntegrityError(
                f"content hash mismatch: expected {expected_hash}, computed {content_hash}"
            )
        artifact_id = f"sha256-{content_hash}"
        target = self._artifact_dir(artifact_id)
        if target.exists():
            existing = self.get_metadata(artifact_id)
            if existing.byte_size != len(content):
                raise ArtifactIntegrityError("existing artifact size does not match its content")
            if self.read_bytes(artifact_id) != content:
                raise ArtifactIntegrityError("existing artifact bytes do not match their identity")
            supplied = metadata.model_dump(mode="json", by_alias=True)
            stored = existing.model_dump(
                mode="json",
                by_alias=True,
                exclude={"id", "content_hash", "byte_size", "created_at"},
            )
            if supplied != stored:
                raise ArtifactIntegrityError(
                    "content already exists with different immutable metadata"
                )
            return existing

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".artifact-", dir=target.parent))
        try:
            content_path = temporary / _CONTENT_NAME
            metadata_path = temporary / _METADATA_NAME
            content_path.write_bytes(content)
            reference = ArtifactRef(
                **metadata.model_dump(),
                id=artifact_id,
                content_hash=content_hash,
                byte_size=len(content),
                created_at=datetime.now(UTC),
            )
            metadata_path.write_text(
                reference.model_dump_json(by_alias=True, indent=2), encoding="utf-8"
            )
            self._sync_file(content_path)
            self._sync_file(metadata_path)
            try:
                temporary.rename(target)
            except FileExistsError:
                return self.put_bytes(content, metadata, expected_hash=expected_hash)
            return reference
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def get_metadata(self, artifact_id: str) -> ArtifactRef:
        """Read and validate metadata only after its containing directory is committed."""

        artifact_dir = self._artifact_dir(artifact_id)
        metadata_path = artifact_dir / _METADATA_NAME
        content_path = artifact_dir / _CONTENT_NAME
        if not artifact_dir.is_dir() or not metadata_path.is_file() or not content_path.is_file():
            raise ArtifactNotFoundError(f"artifact {artifact_id!r} was not found")
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            reference = ArtifactRef.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise ArtifactIntegrityError(
                f"artifact {artifact_id!r} has invalid metadata"
            ) from error
        expected_hash = self._parse_id(artifact_id)
        if reference.id != artifact_id or reference.content_hash != expected_hash:
            raise ArtifactIntegrityError("stored metadata identity does not match its path")
        if reference.byte_size != content_path.stat().st_size:
            raise ArtifactIntegrityError("stored metadata byte size does not match content")
        return reference

    def read_bytes(self, artifact_id: str, *, verify: bool = True) -> bytes:
        """Read content and optionally verify it against its content-addressed identity."""

        reference = self.get_metadata(artifact_id)
        content = (self._artifact_dir(artifact_id) / _CONTENT_NAME).read_bytes()
        if verify and self._hash(content) != reference.content_hash:
            raise ArtifactIntegrityError(f"artifact {artifact_id!r} failed hash verification")
        return content

    def exists(self, artifact_id: str) -> bool:
        """Return true only for a complete, valid artifact identity path."""

        artifact_dir = self._artifact_dir(artifact_id)
        return (artifact_dir / _CONTENT_NAME).is_file() and (
            artifact_dir / _METADATA_NAME
        ).is_file()
