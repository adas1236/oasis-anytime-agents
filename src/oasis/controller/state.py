"""Validated state transitions, action generations, and trace construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import JsonValue

from oasis.controller.budget import BudgetAccount, Deadline
from oasis.controller.schemas import (
    ActionRecord,
    ActionStatus,
    ControllerEvent,
    ControllerState,
    EventActor,
    EventKind,
)


class InvalidStateTransitionError(RuntimeError):
    """Raised when orchestration attempts to skip or reverse a lifecycle state."""


LEGAL_TRANSITIONS: dict[ControllerState, frozenset[ControllerState]] = {
    ControllerState.RECEIVED: frozenset({ControllerState.GROUNDING, ControllerState.QUIESCING}),
    ControllerState.GROUNDING: frozenset(
        {ControllerState.PROBLEM_LOCKED, ControllerState.REASONING, ControllerState.QUIESCING}
    ),
    ControllerState.PROBLEM_LOCKED: frozenset(
        {ControllerState.ADMITTED, ControllerState.QUIESCING}
    ),
    ControllerState.ADMITTED: frozenset(
        {ControllerState.BASELINE_COMMITTED, ControllerState.QUIESCING}
    ),
    ControllerState.BASELINE_COMMITTED: frozenset(
        {ControllerState.SEARCHING, ControllerState.QUIESCING}
    ),
    ControllerState.SEARCHING: frozenset({ControllerState.QUIESCING}),
    ControllerState.REASONING: frozenset({ControllerState.QUIESCING}),
    ControllerState.QUIESCING: frozenset({ControllerState.FINALIZED}),
    ControllerState.FINALIZED: frozenset(),
}


class StateMachine:
    """Small state holder which permits only declared lifecycle edges."""

    def __init__(self, *, initial: ControllerState = ControllerState.RECEIVED) -> None:
        self._state = initial

    @property
    def state(self) -> ControllerState:
        return self._state

    def transition(self, target: ControllerState) -> None:
        if target not in LEGAL_TRANSITIONS[self._state]:
            raise InvalidStateTransitionError(
                f"illegal controller transition {self._state.value} -> {target.value}"
            )
        self._state = target


def action_fingerprint(payload: Mapping[str, Any]) -> str:
    """Create a stable duplicate-detection key for one validated action."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class ActionLedger:
    """Track unique action generations and reject events after cancellation/completion."""

    def __init__(self) -> None:
        self._records: dict[str, ActionRecord] = {}
        self._fingerprints: set[str] = set()
        self._generation = 0

    def is_duplicate(self, fingerprint: str) -> bool:
        return fingerprint in self._fingerprints

    def admit(
        self,
        *,
        fingerprint: str,
        admitted_at_ms: int,
        tool_name: str | None = None,
        subdeadline_monotonic: float | None = None,
    ) -> ActionRecord:
        if fingerprint in self._fingerprints:
            raise ValueError("duplicate action fingerprint cannot be admitted twice")
        self._generation += 1
        record = ActionRecord(
            action_id=f"action-{self._generation:06d}",
            generation=self._generation,
            tool_name=tool_name,
            fingerprint=fingerprint,
            status=ActionStatus.ADMITTED,
            admitted_at_ms=admitted_at_ms,
            subdeadline_monotonic=subdeadline_monotonic,
        )
        self._records[record.action_id] = record
        self._fingerprints.add(fingerprint)
        return record

    def mark(self, record: ActionRecord, status: ActionStatus) -> ActionRecord:
        current = self._records.get(record.action_id)
        if current is None or current.generation != record.generation:
            raise KeyError("unknown or stale action generation")
        updated = current.model_copy(update={"status": status})
        self._records[record.action_id] = updated
        return updated

    def accepts(self, action_id: str, generation: int) -> bool:
        current = self._records.get(action_id)
        return (
            current is not None
            and current.generation == generation
            and current.status
            in {
                ActionStatus.ADMITTED,
                ActionStatus.RUNNING,
            }
        )


EventCallback = Callable[[ControllerEvent], Awaitable[None] | None]


class EventJournal:
    """Build ordered redacted events, persist first, then notify an optional observer."""

    def __init__(
        self,
        *,
        run_id: str,
        run_generation: int = 1,
        deadline: Deadline,
        budget: BudgetAccount,
        append: Callable[[ControllerEvent], None],
        state: StateMachine,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
        callback: EventCallback | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_generation = run_generation
        self._deadline = deadline
        self._budget = budget
        self._append = append
        self._state = state
        self._utcnow = utcnow
        self._callback = callback
        self._sequence = 0

    @property
    def count(self) -> int:
        return self._sequence

    async def emit(
        self,
        kind: EventKind,
        *,
        actor: EventActor = EventActor.CONTROLLER,
        action: ActionRecord | None = None,
        artifact_ids: tuple[str, ...] = (),
        payload: Mapping[str, JsonValue] | None = None,
        budget_before: Any | None = None,
    ) -> ControllerEvent:
        before = budget_before or self._budget.snapshot()
        event = ControllerEvent(
            sequence=self._sequence,
            kind=kind,
            state=self._state.state,
            relative_monotonic_ms=self._deadline.elapsed_ms,
            timestamp=self._utcnow(),
            run_id=self.run_id,
            run_generation=self.run_generation,
            action_id=action.action_id if action is not None else None,
            action_generation=action.generation if action is not None else None,
            actor=actor,
            budget_before=before,
            budget_after=self._budget.snapshot(),
            artifact_ids=artifact_ids,
            payload=cast(dict[str, JsonValue], _redact(dict(payload or {}))),
        )
        self._append(event)
        self._sequence += 1
        if self._callback is not None:
            observed = self._callback(event)
            if observed is not None:
                await observed
        return event


_SENSITIVE_FRAGMENTS = ("token", "secret", "password", "authorization", "api_key", "cookie")


def _redact(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        redacted: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if any(fragment in key.lower() for fragment in _SENSITIVE_FRAGMENTS):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact(nested)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
