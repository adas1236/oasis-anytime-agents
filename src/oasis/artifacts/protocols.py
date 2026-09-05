"""Artifact-store protocol independent of local or object storage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from oasis.schemas.artifacts import ArtifactMetadata, ArtifactRef


class ArtifactStoreError(RuntimeError):
    """Base exception for artifact storage failures."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact ID has no committed object."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when identity, metadata, or content verification fails."""


@runtime_checkable
class ArtifactStore(Protocol):
    """Content-addressed byte store suitable for later object-store implementations."""

    def put_bytes(
        self,
        content: bytes,
        metadata: ArtifactMetadata,
        *,
        expected_hash: str | None = None,
    ) -> ArtifactRef: ...

    def get_metadata(self, artifact_id: str) -> ArtifactRef: ...

    def read_bytes(self, artifact_id: str, *, verify: bool = True) -> bytes: ...

    def exists(self, artifact_id: str) -> bool: ...
