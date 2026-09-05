"""Framework-neutral in-memory and local JSON/JSONL run trace stores."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from oasis.controller.schemas import ControllerEvent, RunResult


class RunStoreError(RuntimeError):
    """Raised for unsafe identities, invalid ordering, or corrupt persisted run data."""


class RunMetadata(BaseModel):
    """Small immutable run index document; large values remain artifact references."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    run_generation: int = Field(default=1, ge=1)
    problem_artifact_id: str | None = None
    seed: int
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


@runtime_checkable
class RunStore(Protocol):
    """Persistence boundary used by the controller and the later service layer."""

    def create(self, metadata: RunMetadata) -> None: ...

    def read_metadata(self, run_id: str) -> RunMetadata | None: ...

    def append_event(self, event: ControllerEvent) -> None: ...

    def read_events(
        self, run_id: str, *, after_sequence: int = -1
    ) -> tuple[ControllerEvent, ...]: ...

    def write_result(self, result: RunResult) -> None: ...

    def read_result(self, run_id: str) -> RunResult | None: ...


class InMemoryRunStore:
    """Deterministic store for embeddings and timing-focused tests."""

    def __init__(self) -> None:
        self.metadata: dict[str, RunMetadata] = {}
        self.events: dict[str, list[ControllerEvent]] = {}
        self.results: dict[str, RunResult] = {}

    def create(self, metadata: RunMetadata) -> None:
        if metadata.run_id in self.metadata:
            raise RunStoreError(f"run {metadata.run_id!r} already exists")
        self.metadata[metadata.run_id] = metadata
        self.events[metadata.run_id] = []

    def append_event(self, event: ControllerEvent) -> None:
        events = self.events.get(event.run_id)
        if events is None:
            raise RunStoreError(f"unknown run {event.run_id!r}")
        if event.sequence != len(events):
            raise RunStoreError("run events must be appended in sequence order")
        events.append(event)

    def read_metadata(self, run_id: str) -> RunMetadata | None:
        return self.metadata.get(run_id)

    def read_events(self, run_id: str, *, after_sequence: int = -1) -> tuple[ControllerEvent, ...]:
        return tuple(
            event for event in self.events.get(run_id, []) if event.sequence > after_sequence
        )

    def write_result(self, result: RunResult) -> None:
        if result.run_id not in self.metadata:
            raise RunStoreError(f"unknown run {result.run_id!r}")
        self.results[result.run_id] = result

    def read_result(self, run_id: str) -> RunResult | None:
        return self.results.get(run_id)


class LocalRunStore:
    """Persist run metadata/results as JSON and ordered events as durable JSONL."""

    _ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()
        self._lock = RLock()

    def _directory(self, run_id: str) -> Path:
        if self._ID.fullmatch(run_id) is None:
            raise RunStoreError(f"unsafe or invalid run ID: {run_id!r}")
        directory = (self._root / run_id).resolve()
        if not directory.is_relative_to(self._root):
            raise RunStoreError("run path escapes the configured store")
        return directory

    @staticmethod
    def _atomic_json(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, metadata: RunMetadata) -> None:
        directory = self._directory(metadata.run_id)
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RunStoreError(f"run {metadata.run_id!r} already exists") from error
        self._atomic_json(directory / "run.json", metadata.model_dump_json(indent=2))
        (directory / "events.jsonl").touch(exist_ok=False)

    def read_metadata(self, run_id: str) -> RunMetadata | None:
        path = self._directory(run_id) / "run.json"
        if not path.exists():
            return None
        try:
            return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RunStoreError(f"run {run_id!r} has invalid metadata") from error

    def append_event(self, event: ControllerEvent) -> None:
        path = self._directory(event.run_id) / "events.jsonl"
        if not path.is_file():
            raise RunStoreError(f"unknown run {event.run_id!r}")
        with self._lock:
            events = self.read_events(event.run_id)
            expected = len(events)
            if event.sequence != expected:
                raise RunStoreError(
                    f"event sequence {event.sequence} does not follow persisted {expected - 1}"
                )
            with path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def read_events(self, run_id: str, *, after_sequence: int = -1) -> tuple[ControllerEvent, ...]:
        path = self._directory(run_id) / "events.jsonl"
        if not path.is_file():
            raise RunStoreError(f"unknown run {run_id!r}")
        events: list[ControllerEvent] = []
        try:
            with self._lock:
                lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                event = ControllerEvent.model_validate_json(line)
                if event.sequence != len(events):
                    raise RunStoreError("persisted events are not contiguous and ordered")
                events.append(event)
        except (OSError, ValueError) as error:
            if isinstance(error, RunStoreError):
                raise
            raise RunStoreError(f"run {run_id!r} has an invalid event trace") from error
        return tuple(event for event in events if event.sequence > after_sequence)

    def write_result(self, result: RunResult) -> None:
        directory = self._directory(result.run_id)
        if not (directory / "run.json").is_file():
            raise RunStoreError(f"unknown run {result.run_id!r}")
        self._atomic_json(directory / "result.json", result.model_dump_json(indent=2))

    def read_result(self, run_id: str) -> RunResult | None:
        path = self._directory(run_id) / "result.json"
        if not path.exists():
            return None
        try:
            return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RunStoreError(f"run {run_id!r} has an invalid result") from error
