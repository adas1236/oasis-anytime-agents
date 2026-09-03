"""Local and in-memory indices for immutable provider snapshots."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from oasis.providers.models import SnapshotCacheEntry


class MemorySnapshotCache:
    """Deterministic cache index for embedding and tests."""

    def __init__(self) -> None:
        self._entries: dict[str, SnapshotCacheEntry] = {}

    def get(self, request_key: str) -> SnapshotCacheEntry | None:
        return self._entries.get(request_key)

    def put(self, entry: SnapshotCacheEntry) -> None:
        self._entries[entry.request_key] = entry


class LocalSnapshotCache:
    """Atomically persisted cache pointers; referenced artifact content stays immutable."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    def _path(self, request_key: str) -> Path:
        if len(request_key) != 64 or any(
            character not in "0123456789abcdef" for character in request_key
        ):
            raise ValueError("snapshot request key must be a lowercase SHA-256 digest")
        return self._root / request_key[:2] / f"{request_key}.json"

    def get(self, request_key: str) -> SnapshotCacheEntry | None:
        path = self._path(request_key)
        if not path.is_file():
            return None
        try:
            return SnapshotCacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            raise ValueError("snapshot cache index is invalid") from error

    def put(self, entry: SnapshotCacheEntry) -> None:
        path = self._path(entry.request_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = entry.model_dump_json(indent=2)
        handle, temporary_name = tempfile.mkstemp(prefix=".snapshot-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(descriptor)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
