from __future__ import annotations

from datetime import UTC, datetime

import pytest

from oasis.controller import (
    LEGAL_TRANSITIONS,
    ActionLedger,
    ActionStatus,
    BudgetAccount,
    BudgetExceededError,
    BudgetSpec,
    ControllerPolicy,
    ControllerState,
    Deadline,
    EventJournal,
    EventKind,
    InMemoryRunStore,
    InvalidStateTransitionError,
    RunMetadata,
    StateMachine,
    TokenLedger,
    ToolCallLedger,
    action_fingerprint,
)
from oasis.llm import TokenUsage


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance_ms(self, milliseconds: int) -> None:
        self.value += milliseconds / 1_000


def test_deadline_protects_configured_finalization_reserve() -> None:
    clock = FakeClock()
    spec = BudgetSpec(
        wall_time_ms=1_000,
        max_total_model_tokens=10,
        max_generated_tokens=5,
        max_tool_calls=2,
        finalization_reserve_ms=100,
    )
    deadline = Deadline(spec, ControllerPolicy(), monotonic=clock)

    assert deadline.remaining_ms == 1_000
    assert deadline.search_remaining_ms == 900
    assert deadline.admits(850, 50)
    assert not deadline.admits(851, 50)
    assert deadline.action_subdeadline(2_000) == pytest.approx(100.9)

    clock.advance_ms(901)
    assert deadline.search_expired
    assert not deadline.expired
    clock.advance_ms(100)
    assert deadline.expired
    assert deadline.overshoot_ms == 1


def test_default_reserve_is_bounded_for_small_and_large_runs() -> None:
    policy = ControllerPolicy(
        minimum_finalization_reserve_ms=10,
        maximum_finalization_reserve_ms=2_000,
    )
    small = Deadline(BudgetSpec(wall_time_ms=20), policy, monotonic=FakeClock())
    large = Deadline(BudgetSpec(wall_time_ms=100_000), policy, monotonic=FakeClock())

    assert small.finalization_reserve_ms == 10
    assert large.finalization_reserve_ms == 2_000


def test_token_ledger_enforces_exact_aggregate_limits() -> None:
    ledger = TokenLedger(
        BudgetSpec(
            wall_time_ms=100,
            max_total_model_tokens=12,
            max_generated_tokens=5,
        )
    )
    ledger.record(TokenUsage(input_tokens=3, generated_tokens=2))
    ledger.record(TokenUsage(input_tokens=2, generated_tokens=1))

    assert ledger.usage == TokenUsage(input_tokens=5, generated_tokens=3)
    assert ledger.remaining_total == 4
    assert ledger.remaining_generated == 2
    assert ledger.generation_allowance(estimated_input_tokens=2, requested=10) == 2
    with pytest.raises(BudgetExceededError, match="generated-token"):
        ledger.record(TokenUsage(input_tokens=0, generated_tokens=3))
    assert ledger.usage == TokenUsage(input_tokens=5, generated_tokens=3)


def test_tool_call_ledger_never_exceeds_limit() -> None:
    ledger = ToolCallLedger(2)
    assert ledger.admit()
    assert ledger.admit()
    assert not ledger.admit()
    assert ledger.used == 2
    assert ledger.remaining == 0


@pytest.mark.parametrize("source", list(ControllerState))
def test_state_machine_accepts_exactly_the_legal_transitions(source: ControllerState) -> None:
    for target in ControllerState:
        machine = StateMachine(initial=source)
        if target in LEGAL_TRANSITIONS[source]:
            machine.transition(target)
            assert machine.state is target
        else:
            with pytest.raises(InvalidStateTransitionError):
                machine.transition(target)


def test_action_ledger_rejects_duplicates_and_late_generations() -> None:
    ledger = ActionLedger()
    fingerprint = action_fingerprint({"tool": "improve", "strategy": "add_swap"})
    record = ledger.admit(fingerprint=fingerprint, admitted_at_ms=0, tool_name="improve")

    assert ledger.is_duplicate(fingerprint)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.admit(fingerprint=fingerprint, admitted_at_ms=1, tool_name="improve")
    assert ledger.accepts(record.action_id, record.generation)
    ledger.mark(record, ActionStatus.COMPLETED)
    assert not ledger.accepts(record.action_id, record.generation)
    assert not ledger.accepts(record.action_id, record.generation + 1)


def test_budget_snapshot_combines_each_ledger() -> None:
    clock = FakeClock()
    spec = BudgetSpec(
        wall_time_ms=100,
        max_total_model_tokens=10,
        max_generated_tokens=4,
        max_tool_calls=2,
        finalization_reserve_ms=10,
    )
    account = BudgetAccount(spec, Deadline(spec, ControllerPolicy(), monotonic=clock))
    account.tokens.record(TokenUsage(input_tokens=2, generated_tokens=1))
    assert account.tools.admit()
    clock.advance_ms(25)

    snapshot = account.snapshot()
    assert snapshot.wall_elapsed_ms == 25
    assert snapshot.wall_remaining_ms == 75
    assert snapshot.search_remaining_ms == 65
    assert snapshot.remaining_total_model_tokens == 7
    assert snapshot.remaining_generated_tokens == 3
    assert snapshot.tool_calls == 1
    assert datetime.now(UTC).tzinfo is not None


@pytest.mark.asyncio
async def test_event_journal_redacts_sensitive_payloads_before_persistence() -> None:
    clock = FakeClock()
    spec = BudgetSpec(wall_time_ms=1_000)
    deadline = Deadline(spec, ControllerPolicy(), monotonic=clock)
    account = BudgetAccount(spec, deadline)
    state = StateMachine()
    store = InMemoryRunStore()
    store.create(RunMetadata(run_id="redacted", problem_artifact_id="problem", seed=0))
    journal = EventJournal(
        run_id="redacted",
        deadline=deadline,
        budget=account,
        append=store.append_event,
        state=state,
        utcnow=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    await journal.emit(
        EventKind.RUN_CREATED,
        payload={
            "api_key": "do-not-store",
            "nested": {"authorization": "Bearer secret", "safe": "visible"},
        },
    )

    payload = store.read_events("redacted")[0].payload
    assert payload == {
        "api_key": "[redacted]",
        "nested": {"authorization": "[redacted]", "safe": "visible"},
    }
