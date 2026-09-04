"""Protocols for runtime planning and inference execution."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from oasis.config import RuntimePolicy
from oasis.llm.schemas import ModelDelta, ModelProfile, ModelRequest
from oasis.runtimes.schemas import ComputeInventory, RuntimePlan


@runtime_checkable
class RuntimePlanner(Protocol):
    """Resolve an explicit policy against only the supplied visible inventory."""

    def plan(
        self,
        model: ModelProfile,
        inventory: ComputeInventory,
        policy: RuntimePolicy,
        *,
        revision: str | None = None,
    ) -> RuntimePlan: ...


@runtime_checkable
class InferenceRuntime(Protocol):
    """Placement-neutral streaming inference lifecycle."""

    @property
    def plan(self) -> RuntimePlan | None: ...

    @property
    def inventory(self) -> ComputeInventory: ...

    async def load(self, model: ModelProfile, plan: RuntimePlan) -> None: ...

    async def count_input_tokens(self, request: ModelRequest) -> int: ...

    def generate(self, request: ModelRequest) -> AsyncIterator[ModelDelta]: ...

    async def abort(self, request_id: str) -> None: ...

    async def close(self) -> None: ...
