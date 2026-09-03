"""Controller-independent problem plugin protocol."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from oasis.artifacts import ArtifactStore
from oasis.problems.schemas import (
    Comparison,
    ResultView,
    Scorecard,
    SearchStrategy,
    ValidationReport,
)
from oasis.schemas import Plan


@dataclass(frozen=True, slots=True)
class Deadline:
    """Absolute monotonic deadline passed to baseline/search implementations."""

    at_monotonic: float
    monotonic: Callable[[], float] = time.monotonic

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.at_monotonic - self.monotonic())

    @property
    def expired(self) -> bool:
        return self.remaining_seconds <= 0.0


@runtime_checkable
class ProblemPlugin(Protocol):
    """Shared contract used by all problem families and the future controller."""

    type_id: str
    version: str

    def validate_spec(self, spec: object, store: ArtifactStore) -> ValidationReport: ...

    def make_baseline(self, spec: object, store: ArtifactStore, deadline: Deadline) -> Plan: ...

    def validate_plan(self, spec: object, plan: Plan, store: ArtifactStore) -> ValidationReport: ...

    def measure(self, spec: object, plan: Plan, store: ArtifactStore) -> Scorecard: ...

    def compare(self, left: Scorecard, right: Scorecard) -> Comparison: ...

    def fallback_actions(self) -> tuple[SearchStrategy, ...]: ...

    def render_result(self, spec: object, plan: Plan, scorecard: Scorecard) -> ResultView: ...
