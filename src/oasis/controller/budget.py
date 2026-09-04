"""Injectable deadline and exact aggregate token/tool-call accounting."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from threading import Lock

from oasis.controller.schemas import BudgetSnapshot, BudgetSpec, ControllerPolicy
from oasis.llm.schemas import TokenUsage


class BudgetExceededError(RuntimeError):
    """Raised before a ledger mutation would exceed a declared hard limit."""


class Deadline:
    """Absolute monotonic run deadline with a protected finalization interval."""

    def __init__(
        self,
        budget: BudgetSpec,
        policy: ControllerPolicy,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        started_at: float | None = None,
    ) -> None:
        self._monotonic = monotonic
        self.started_at = monotonic() if started_at is None else started_at
        self.at_monotonic = self.started_at + budget.wall_time_ms / 1_000
        computed = round(budget.wall_time_ms * policy.reserve_fraction)
        bounded = min(
            policy.maximum_finalization_reserve_ms,
            max(policy.minimum_finalization_reserve_ms, computed),
        )
        reserve = budget.finalization_reserve_ms
        self.finalization_reserve_ms = min(
            budget.wall_time_ms - 1,
            reserve if reserve is not None else bounded,
        )
        self.search_deadline_monotonic = self.at_monotonic - self.finalization_reserve_ms / 1_000

    @property
    def now(self) -> float:
        return self._monotonic()

    @property
    def elapsed_ms(self) -> int:
        return self._whole_milliseconds(self.now - self.started_at)

    @property
    def remaining_ms(self) -> int:
        return self._whole_milliseconds(self.at_monotonic - self.now)

    @property
    def search_remaining_ms(self) -> int:
        return self._whole_milliseconds(self.search_deadline_monotonic - self.now)

    @property
    def expired(self) -> bool:
        return self.now >= self.at_monotonic

    @property
    def search_expired(self) -> bool:
        return self.now >= self.search_deadline_monotonic

    @property
    def overshoot_ms(self) -> int:
        return self._whole_milliseconds(self.now - self.at_monotonic)

    @staticmethod
    def _whole_milliseconds(seconds: float) -> int:
        """Floor durations while neutralizing binary noise at exact millisecond boundaries."""

        return max(0, math.floor(seconds * 1_000 + 1e-7))

    def admits(self, estimated_ms: int, validation_reserve_ms: int = 0) -> bool:
        """Return whether estimated work and validation fit before quiescence."""

        return estimated_ms + validation_reserve_ms <= self.search_remaining_ms

    def action_subdeadline(self, estimated_p95_ms: int) -> float:
        """Bound one action by both its p95 estimate and the search deadline."""

        return min(self.search_deadline_monotonic, self.now + estimated_p95_ms / 1_000)


class TokenLedger:
    """Thread-safe exact accounting for repeated model calls."""

    def __init__(self, budget: BudgetSpec) -> None:
        self._budget = budget
        self._usage = TokenUsage()
        self._lock = Lock()

    @property
    def usage(self) -> TokenUsage:
        with self._lock:
            return self._usage

    @property
    def remaining_total(self) -> int:
        return max(0, self._budget.max_total_model_tokens - self.usage.total_tokens)

    @property
    def remaining_generated(self) -> int:
        return max(0, self._budget.max_generated_tokens - self.usage.generated_tokens)

    def generation_allowance(self, *, estimated_input_tokens: int, requested: int) -> int:
        """Return a safe output cap after reserving this call's estimated input."""

        if estimated_input_tokens < 0 or requested < 0:
            raise ValueError("token estimates and requests must be non-negative")
        return max(
            0,
            min(
                requested,
                self.remaining_generated,
                self.remaining_total - estimated_input_tokens,
            ),
        )

    def record(self, usage: TokenUsage) -> None:
        """Atomically add observed usage, rejecting any aggregate overrun."""

        with self._lock:
            updated = self._usage + usage
            if updated.total_tokens > self._budget.max_total_model_tokens:
                raise BudgetExceededError("model reported usage beyond the total-token budget")
            if updated.generated_tokens > self._budget.max_generated_tokens:
                raise BudgetExceededError("model reported usage beyond the generated-token budget")
            self._usage = updated


class ToolCallLedger:
    """Thread-safe aggregate admission counter for tool invocations."""

    def __init__(self, maximum: int) -> None:
        self._maximum = maximum
        self._used = 0
        self._lock = Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        return max(0, self._maximum - self.used)

    def admit(self) -> bool:
        with self._lock:
            if self._used >= self._maximum:
                return False
            self._used += 1
            return True


class BudgetAccount:
    """Single source for budget snapshots attached to run events and results."""

    def __init__(self, spec: BudgetSpec, deadline: Deadline) -> None:
        self.spec = spec
        self.deadline = deadline
        self.tokens = TokenLedger(spec)
        self.tools = ToolCallLedger(spec.max_tool_calls)

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            wall_elapsed_ms=self.deadline.elapsed_ms,
            wall_remaining_ms=self.deadline.remaining_ms,
            search_remaining_ms=self.deadline.search_remaining_ms,
            model_usage=self.tokens.usage,
            remaining_total_model_tokens=self.tokens.remaining_total,
            remaining_generated_tokens=self.tokens.remaining_generated,
            tool_calls=self.tools.used,
            remaining_tool_calls=self.tools.remaining,
        )
