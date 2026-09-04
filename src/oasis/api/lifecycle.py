"""Service-owned, non-probing model lifecycle management."""

from __future__ import annotations

import asyncio
import time
import uuid

from pydantic import JsonValue

from oasis.api.schemas import (
    ChatRequest,
    ChatResponse,
    RuntimeCapabilities,
    RuntimeOptions,
    RuntimeResponse,
)
from oasis.config import DevicePolicy, OasisSettings, RuntimeConfig, RuntimeEngine
from oasis.llm import MODEL_PROFILES, ModelBackend, ModelCapabilities, ModelRequest
from oasis.llm.factory import create_model_backend
from oasis.llm.fake import FakeModelBackend
from oasis.llm.profiles import DEFAULT_PROFILE_NAME, resolve_model_profile
from oasis.runtimes import (
    ComputeInventory,
    DiscoveryMode,
    RuntimeKind,
    RuntimePlan,
    installed_runtime_capabilities,
)


class ModelService:
    """Own and reuse lazy model backends selected through the versioned API."""

    def __init__(
        self,
        settings: OasisSettings,
        *,
        backend: ModelBackend | None = None,
        compute_inventory: ComputeInventory | None = None,
    ) -> None:
        started = time.monotonic()
        self.settings = settings
        self.backend = backend or create_model_backend(settings, inventory=compute_inventory)
        self.startup_ms = max(0, round((time.monotonic() - started) * 1_000))
        self._closed = False
        self._ready: bool = False
        self._load_lock = asyncio.Lock()
        self._generation_slots = asyncio.Semaphore(settings.api_max_concurrent_runs)
        self._backend_was_injected = backend is not None
        self._alternate_backends: dict[str, ModelBackend] = {}
        self._alternate_ready: set[int] = set()
        self._alternate_load_locks: dict[int, asyncio.Lock] = {}
        self._alternate_startup_ms: dict[int, int] = {}
        self._backend_settings: dict[int, OasisSettings] = {id(self.backend): settings}

    @property
    def model_loaded(self) -> bool:
        """Report lazy materialization without asking any runtime to probe hardware."""

        return bool(getattr(self.backend, "is_loaded", isinstance(self.backend, FakeModelBackend)))

    def _is_ready(self) -> bool:
        return self._ready

    async def ensure_ready(self, backend: ModelBackend | None = None) -> None:
        """Load at most once, accounting service startup outside any run deadline."""

        selected = backend or self.backend
        if selected is not self.backend:
            identity = id(selected)
            if identity in self._alternate_ready:
                return
            lock = self._alternate_load_locks.setdefault(identity, asyncio.Lock())
            async with lock:
                if identity in self._alternate_ready:
                    return
                started = time.monotonic()
                await selected.load()
                elapsed = max(0, round((time.monotonic() - started) * 1_000))
                self._alternate_startup_ms[identity] = elapsed
                self._alternate_ready.add(identity)
            return
        if self._is_ready():
            return
        async with self._load_lock:
            if self._is_ready():
                return
            started = time.monotonic()
            await self.backend.load()
            self.startup_ms += max(0, round((time.monotonic() - started) * 1_000))
            self._ready = True

    def runtime_plan(self) -> dict[str, JsonValue]:
        """Return the typed placement record without triggering discovery or loading."""

        return self.runtime_plan_model().model_dump(mode="json")

    def runtime_plan_model(self, backend: ModelBackend | None = None) -> RuntimePlan:
        """Return the current typed plan with service-lifecycle startup accounting."""

        selected = backend or self.backend
        selected_settings = self._backend_settings.get(id(selected), self.settings)
        plan = getattr(selected, "runtime_plan", None)
        if not isinstance(plan, RuntimePlan):
            runtime = (
                RuntimeKind.FAKE
                if isinstance(selected, FakeModelBackend)
                else RuntimeKind.CPU_TRANSFORMERS
            )
            plan = RuntimePlan(
                requested_profile=selected.profile.name,
                requested_model_id=selected.profile.model_id,
                runtime=runtime,
                device_placement=("cpu",),
                dtype=selected_settings.dtype,
                quantization=selected_settings.quantization,
                attention_backend=selected_settings.attention_backend,
                rationale=("Preloaded backend supplied without runtime metadata.",),
            )
        startup_ms = (
            self.startup_ms
            if selected is self.backend
            else self._alternate_startup_ms.get(id(selected), 0)
        )
        metrics = plan.metrics.model_copy(
            update={"startup_ms": max(plan.metrics.startup_ms, startup_ms)}
        )
        return plan.model_copy(update={"metrics": metrics})

    def inventory_model(self, backend: ModelBackend | None = None) -> ComputeInventory:
        selected = backend or self.backend
        inventory = getattr(selected, "compute_inventory", None)
        if isinstance(inventory, ComputeInventory):
            return inventory
        from oasis.runtimes import safe_cpu_inventory

        return safe_cpu_inventory()

    def inventory(self) -> dict[str, JsonValue]:
        """Return a sanitized typed inventory already known to the service."""

        return self.inventory_model().sanitized().model_dump(mode="json")

    def runtime_response(self) -> RuntimeResponse:
        capabilities = self.runtime_capabilities()
        inventory = self.inventory_model()
        devices = [DevicePolicy.CPU, DevicePolicy.AUTO]
        if inventory.accelerators or self.settings.device is DevicePolicy.CUDA:
            devices.append(DevicePolicy.CUDA)
        engines = [RuntimeEngine.AUTO]
        if capabilities.transformers:
            engines.append(RuntimeEngine.TRANSFORMERS)
        if capabilities.accelerate:
            engines.append(RuntimeEngine.ACCELERATE)
        if self.settings.remote_endpoint is not None:
            engines.append(RuntimeEngine.REMOTE)
        return RuntimeResponse(
            requested_policy=self.settings.runtime_config(),
            resolved_plan=self.runtime_plan(),
            capabilities=capabilities,
            options=RuntimeOptions(
                devices=tuple(devices),
                engines=tuple(engines),
                dtypes=("auto", "float32", "float16", "bfloat16"),
                quantizations=("int8", "int4"),
                attention_backends=("auto", "eager", "sdpa", "flash_attention_2"),
            ),
            inventory=self.inventory(),
            inventory_probed=(self.inventory_model().discovery_mode is DiscoveryMode.CUDA_PROBE),
            model_loaded=self.model_loaded,
            model_startup_ms=self.startup_ms,
        )

    def backend_for(
        self,
        *,
        model_profile: str | None,
        model_id: str | None,
        runtime_policy: RuntimeConfig | None,
    ) -> ModelBackend:
        """Resolve and cache one request-selected backend without loading model weights."""

        if model_profile is None and model_id is None and runtime_policy is None:
            return self.backend

        profile_name = model_profile or self.settings.model_profile
        explicit_model_id = (
            model_id
            if model_id is not None
            else (self.settings.model_id if model_profile is None else None)
        )
        profile = resolve_model_profile(profile_name, explicit_model_id)
        policy = runtime_policy or self.settings.runtime_config()
        active_policy = self.settings.runtime_config()
        if policy.offload_directory != active_policy.offload_directory:
            raise ValueError("per-run runtime policy must use the server offload directory")
        if policy.remote_endpoint != active_policy.remote_endpoint:
            raise ValueError("per-run runtime policy cannot replace the server remote endpoint")
        if profile.model_id == self.backend.profile.model_id and policy == active_policy:
            return self.backend

        key = f"{profile.name}:{profile.model_id}:{policy.model_dump_json()}"
        existing = self._alternate_backends.get(key)
        if existing is not None:
            return existing
        if self._backend_was_injected and not isinstance(self.backend, FakeModelBackend):
            raise ValueError("an injected service backend cannot be replaced per run")

        selected: ModelBackend
        if isinstance(self.backend, FakeModelBackend):
            selected = FakeModelBackend(profile=profile, inventory=self.inventory_model())
            selected_settings = self.settings
        else:
            selected_settings = self.settings.model_copy(
                update={
                    "model_profile": profile_name,
                    "model_id": explicit_model_id,
                    "device": policy.device,
                    "runtime_engine": policy.engine,
                    "dtype": policy.dtype,
                    "quantization": policy.quantization,
                    "attention_backend": policy.attention_backend,
                    "memory_headroom_fraction": policy.memory_headroom_fraction,
                    "allow_cpu_offload": policy.allow_cpu_offload,
                    "allow_disk_offload": policy.allow_disk_offload,
                    "offload_root": policy.offload_directory,
                    "remote_endpoint": policy.remote_endpoint,
                    "model_memory_bytes": policy.model_memory_bytes,
                }
            )
            selected = create_model_backend(
                selected_settings,
                inventory=self.inventory_model(),
            )
        self._alternate_backends[key] = selected
        self._backend_settings[id(selected)] = selected_settings
        return selected

    @staticmethod
    def runtime_capabilities() -> RuntimeCapabilities:
        capabilities = installed_runtime_capabilities()
        return RuntimeCapabilities(
            transformers=capabilities[RuntimeKind.CPU_TRANSFORMERS].installed,
            accelerate=capabilities[RuntimeKind.ACCELERATE_DISPATCH].installed,
            remote=capabilities[RuntimeKind.REMOTE].installed,
        )

    def profile_capabilities(self, profile_name: str) -> ModelCapabilities:
        profile = MODEL_PROFILES[profile_name]
        active = self.backend.capabilities
        return active.model_copy(
            update={
                "native_tools": profile.supports_native_tools,
                "reasoning_channels": profile.supports_thinking,
                "context_limit": profile.context_limit,
            }
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        await self.ensure_ready()
        model_request = ModelRequest(
            request_id=f"api-chat-{uuid.uuid4().hex}",
            messages=request.messages,
            max_generated_tokens=request.max_generated_tokens,
            thinking_enabled=request.thinking_enabled,
            seed=request.seed,
        )
        async with self._generation_slots:
            turn = await self.backend.generate(model_request)
        return ChatResponse(
            model_profile=self.backend.profile.name,
            model_id=self.backend.profile.model_id,
            content=turn.message.content,
            tool_calls=turn.message.tool_calls,
            usage=turn.usage,
            finish_reason=turn.finish_reason,
            model_startup_ms=self.startup_ms,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(
            self.backend.close(),
            *(backend.close() for backend in self._alternate_backends.values()),
        )


__all__ = ["DEFAULT_PROFILE_NAME", "ModelService"]
