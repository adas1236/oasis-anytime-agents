"""Atomic, monotone incumbent and verified-bound storage."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from oasis.controller.schemas import IncumbentRecord
from oasis.problems import Comparison, Scorecard
from oasis.schemas import Plan


class IncumbentStore:
    """Keep one feasible incumbent and an immutable commit timeline."""

    def __init__(self, compare: Callable[[Scorecard, Scorecard], Comparison]) -> None:
        self._compare = compare
        self._current: IncumbentRecord | None = None
        self._timeline: list[IncumbentRecord] = []
        self._lock = asyncio.Lock()

    @property
    def current(self) -> IncumbentRecord | None:
        return self._current

    @property
    def timeline(self) -> tuple[IncumbentRecord, ...]:
        return tuple(self._timeline)

    async def try_commit(
        self,
        *,
        plan: Plan,
        scorecard: Scorecard,
        plan_artifact_id: str,
        scorecard_artifact_id: str,
        source_action_id: str,
        committed_at_ms: int,
        seed: int,
        committed_at: datetime | None = None,
    ) -> IncumbentRecord | None:
        """Atomically commit only a feasible strict improvement (or the first baseline)."""

        if not scorecard.feasible:
            return None
        async with self._lock:
            if self._current is not None:
                comparison = self._compare(scorecard, self._current.scorecard)
                if comparison is not Comparison.BETTER:
                    return None
            record = IncumbentRecord(
                plan=plan,
                scorecard=scorecard,
                plan_artifact_id=plan_artifact_id,
                scorecard_artifact_id=scorecard_artifact_id,
                problem_hash=scorecard.problem_hash,
                evidence_hash=scorecard.evidence_hash,
                policy_hash=scorecard.policy_hash,
                comparator_key=scorecard.comparator_key,
                source_action_id=source_action_id,
                committed_at=committed_at or datetime.now(UTC),
                committed_at_ms=committed_at_ms,
                seed=seed,
            )
            self._current = record
            self._timeline.append(record)
            return record

    async def refresh_artifact_ids(
        self,
        *,
        plan_artifact_id: str,
        scorecard_artifact_id: str,
    ) -> None:
        """Attach materialized IDs without changing the plan, score, or comparator ordering."""

        async with self._lock:
            if self._current is None:
                return
            updated = self._current.model_copy(
                update={
                    "plan_artifact_id": plan_artifact_id,
                    "scorecard_artifact_id": scorecard_artifact_id,
                }
            )
            self._current = updated
            self._timeline[-1] = updated
