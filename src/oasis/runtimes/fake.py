"""InferenceRuntime adapter for deterministic backend-based tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from oasis.llm.fake import FakeModelBackend
from oasis.llm.schemas import ModelDelta, ModelProfile, ModelRequest
from oasis.runtimes.inventory import fake_inventory
from oasis.runtimes.schemas import (
    ComputeInventory,
    HardwareValidationStatus,
    RuntimeKind,
    RuntimePlan,
)


class FakeInferenceRuntime:
    """Delegate runtime calls to a deterministic fake backend."""

    def __init__(
        self,
        backend: FakeModelBackend | None = None,
        *,
        inventory: ComputeInventory | None = None,
    ) -> None:
        self._inventory = inventory or fake_inventory()
        self._backend = backend
        self._plan: RuntimePlan | None = None

    @property
    def plan(self) -> RuntimePlan | None:
        return self._plan

    @property
    def inventory(self) -> ComputeInventory:
        return self._inventory

    async def load(self, model: ModelProfile, plan: RuntimePlan) -> None:
        if plan.requested_model_id != model.model_id:
            raise ValueError("fake runtime plan changed the requested model")
        if self._backend is None:
            fake_plan = plan.model_copy(
                update={
                    "runtime": RuntimeKind.FAKE,
                    "hardware_validation": HardwareValidationStatus.NOT_APPLICABLE,
                }
            )
            self._backend = FakeModelBackend(
                profile=model,
                inventory=self._inventory,
                runtime_plan=fake_plan,
            )
        self._plan = self._backend.runtime_plan
        await self._backend.load()

    def _loaded_backend(self) -> FakeModelBackend:
        if self._backend is None:
            raise RuntimeError("fake runtime is not loaded")
        return self._backend

    async def count_input_tokens(self, request: ModelRequest) -> int:
        return await self._loaded_backend().count_input_tokens(request)

    def generate(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        return self._loaded_backend().stream(request)

    async def abort(self, request_id: str) -> None:
        await self._loaded_backend().abort(request_id)

    async def close(self) -> None:
        if self._backend is not None:
            await self._backend.close()
