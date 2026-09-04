"""ModelBackend facade for remote and plugin-provided inference runtimes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from oasis.llm.protocols import collect_turn
from oasis.llm.schemas import (
    ModelCapabilities,
    ModelDelta,
    ModelProfile,
    ModelRequest,
    ModelTurn,
)
from oasis.runtimes.protocols import InferenceRuntime
from oasis.runtimes.schemas import ComputeInventory, RuntimePlan


class RuntimeModelBackend:
    """Preserve the public model contract while delegating execution to a runtime."""

    def __init__(
        self,
        *,
        profile: ModelProfile,
        capabilities: ModelCapabilities,
        runtime: InferenceRuntime,
        plan: RuntimePlan,
    ) -> None:
        if plan.requested_model_id != profile.model_id:
            raise ValueError("runtime plans may not substitute the requested model")
        self._profile = profile
        self._declared_capabilities = capabilities
        self._runtime = runtime
        self._plan = plan
        self._loaded = False
        self._load_lock = asyncio.Lock()

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def capabilities(self) -> ModelCapabilities:
        runtime_capabilities = getattr(self._runtime, "capabilities", None)
        return runtime_capabilities or self._declared_capabilities

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def runtime_plan(self) -> RuntimePlan:
        return self._runtime.plan or self._plan

    @property
    def compute_inventory(self) -> ComputeInventory:
        return self._runtime.inventory

    async def load(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if not self._loaded:
                await self._runtime.load(self._profile, self._plan)
                self._loaded = True

    async def count_input_tokens(self, request: ModelRequest) -> int:
        if not self._loaded:
            await self.load()
        return await self._runtime.count_input_tokens(request)

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        async def lazy_stream() -> AsyncIterator[ModelDelta]:
            if not self._loaded:
                await self.load()
            async for delta in self._runtime.generate(request):
                yield delta

        return lazy_stream()

    async def generate(self, request: ModelRequest) -> ModelTurn:
        return await collect_turn(self, request)

    async def abort(self, request_id: str) -> None:
        await self._runtime.abort(request_id)

    async def close(self) -> None:
        await self._runtime.close()
