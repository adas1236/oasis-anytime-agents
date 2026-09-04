"""Conservative placement planning over already visible resources."""

from __future__ import annotations

import importlib.util
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from oasis.config import DevicePolicy, RuntimeEngine, RuntimePolicy
from oasis.llm.schemas import ModelProfile
from oasis.runtimes.schemas import (
    ComputeInventory,
    HardwareValidationStatus,
    RuntimeCapability,
    RuntimeKind,
    RuntimePlan,
)


class RuntimeRejectionCode(StrEnum):
    """Machine-readable reasons why no allowed placement is credible."""

    NO_VISIBLE_ACCELERATOR = "no_visible_accelerator"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    HETEROGENEOUS_DEVICES = "heterogeneous_devices"
    UNKNOWN_MODEL_SIZE = "unknown_model_size"
    INVALID_DTYPE = "invalid_dtype"
    INVALID_QUANTIZATION = "invalid_quantization"
    INVALID_ATTENTION_BACKEND = "invalid_attention_backend"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"
    MISSING_REMOTE_ENDPOINT = "missing_remote_endpoint"


class RuntimeRejection(BaseModel):
    """Safe typed planning error detail."""

    model_config = ConfigDict(frozen=True)

    code: RuntimeRejectionCode
    message: str
    requested_model_id: str
    requested_device: DevicePolicy
    requested_engine: RuntimeEngine
    context: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RuntimePlanningError(RuntimeError):
    """Raised when a policy cannot be placed without violating its constraints."""

    def __init__(self, rejection: RuntimeRejection) -> None:
        super().__init__(rejection.message)
        self.rejection = rejection


_DTYPE_BYTES = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0}
_QUANTIZATION_BYTES = {"int8": 1.0, "int4": 0.5}


def installed_runtime_capabilities() -> dict[RuntimeKind, RuntimeCapability]:
    """Probe package presence without importing optional accelerator runtimes."""

    packages = {
        RuntimeKind.CPU_TRANSFORMERS: "transformers",
        RuntimeKind.CUDA_TRANSFORMERS: "transformers",
        RuntimeKind.ACCELERATE_DISPATCH: "accelerate",
    }
    capabilities = {
        kind: RuntimeCapability(
            runtime=kind, installed=importlib.util.find_spec(package) is not None
        )
        for kind, package in packages.items()
    }
    capabilities[RuntimeKind.FAKE] = RuntimeCapability(runtime=RuntimeKind.FAKE, installed=True)
    capabilities[RuntimeKind.REMOTE] = RuntimeCapability(runtime=RuntimeKind.REMOTE, installed=True)
    return capabilities


class ConservativeRuntimePlanner:
    """Choose CPU, one fitting GPU, or single-process multi-GPU dispatch."""

    def __init__(
        self,
        capabilities: dict[RuntimeKind, RuntimeCapability] | None = None,
        *,
        quantization_support: frozenset[str] | None = None,
    ) -> None:
        self.capabilities = capabilities or installed_runtime_capabilities()
        self.quantization_support = (
            (
                frozenset({"int8", "int4"})
                if importlib.util.find_spec("bitsandbytes") is not None
                else frozenset()
            )
            if quantization_support is None
            else quantization_support
        )

    def _reject(
        self,
        code: RuntimeRejectionCode,
        message: str,
        model: ModelProfile,
        policy: RuntimePolicy,
        **context: str | int | float | bool | None,
    ) -> RuntimePlanningError:
        return RuntimePlanningError(
            RuntimeRejection(
                code=code,
                message=message,
                requested_model_id=model.model_id,
                requested_device=policy.device,
                requested_engine=policy.engine,
                context=context,
            )
        )

    @staticmethod
    def _dtype(model: ModelProfile, inventory: ComputeInventory, policy: RuntimePolicy) -> str:
        if policy.dtype != "auto":
            if policy.dtype not in _DTYPE_BYTES:
                raise ValueError("dtype")
            return policy.dtype
        if policy.device is DevicePolicy.CPU or not inventory.accelerators:
            return "auto"
        capabilities = [device.compute_capability for device in inventory.accelerators]
        return (
            "bfloat16"
            if all(value and int(value.split(".", 1)[0]) >= 8 for value in capabilities)
            else "float16"
        )

    @staticmethod
    def _attention(dtype: str, inventory: ComputeInventory, policy: RuntimePolicy) -> str:
        del dtype
        allowed = {"auto", "eager", "sdpa", "flash_attention_2"}
        if policy.attention_backend not in allowed:
            raise ValueError("attention")
        if policy.attention_backend == "auto":
            return (
                "sdpa"
                if inventory.accelerators and policy.device is not DevicePolicy.CPU
                else "eager"
            )
        if policy.attention_backend == "flash_attention_2":
            if policy.device is DevicePolicy.CPU or not inventory.accelerators:
                raise ValueError("attention")
            if importlib.util.find_spec("flash_attn") is None:
                raise ValueError("attention")
        return policy.attention_backend

    @staticmethod
    def _model_memory(model: ModelProfile, policy: RuntimePolicy, dtype: str) -> int | None:
        if policy.model_memory_bytes is not None:
            return policy.model_memory_bytes
        parameters = model.estimated_parameter_count
        if parameters is None:
            return None
        bytes_per_parameter = (
            _QUANTIZATION_BYTES[policy.quantization]
            if policy.quantization is not None
            else (
                2.0
                if dtype == "auto" and model.family == "gemma4"
                else _DTYPE_BYTES.get(dtype, 4.0)
            )
        )
        return max(1, int(parameters * bytes_per_parameter * 1.15))

    @staticmethod
    def _homogeneous(inventory: ComputeInventory) -> bool:
        signatures = {
            (device.kind, device.name, device.total_memory_bytes, device.compute_capability)
            for device in inventory.accelerators
        }
        return len(signatures) <= 1

    def _require_capability(
        self,
        runtime: RuntimeKind,
        model: ModelProfile,
        policy: RuntimePolicy,
    ) -> None:
        capability = self.capabilities.get(runtime)
        if capability is None or not capability.installed:
            raise self._reject(
                RuntimeRejectionCode.UNSUPPORTED_RUNTIME,
                f"Runtime {runtime.value!r} is not installed in this environment.",
                model,
                policy,
                runtime=runtime.value,
            )

    def plan(
        self,
        model: ModelProfile,
        inventory: ComputeInventory,
        policy: RuntimePolicy,
        *,
        revision: str | None = None,
    ) -> RuntimePlan:
        """Return one plan or a typed rejection, considering no undiscovered devices."""

        if policy.engine is RuntimeEngine.REMOTE:
            if policy.dtype not in {"auto", *_DTYPE_BYTES}:
                raise self._reject(
                    RuntimeRejectionCode.INVALID_DTYPE,
                    "Dtype must be auto, float32, float16, or bfloat16.",
                    model,
                    policy,
                )
            if policy.quantization not in {None, "int8", "int4"}:
                raise self._reject(
                    RuntimeRejectionCode.INVALID_QUANTIZATION,
                    "Quantization must be null, int8, or int4.",
                    model,
                    policy,
                )
            if policy.attention_backend not in {
                "auto",
                "eager",
                "sdpa",
                "flash_attention_2",
            }:
                raise self._reject(
                    RuntimeRejectionCode.INVALID_ATTENTION_BACKEND,
                    "The requested attention backend is invalid.",
                    model,
                    policy,
                )
            if policy.remote_endpoint is None:
                raise self._reject(
                    RuntimeRejectionCode.MISSING_REMOTE_ENDPOINT,
                    "The remote runtime requires a configured endpoint.",
                    model,
                    policy,
                )
            return RuntimePlan(
                requested_profile=model.name,
                requested_model_id=model.model_id,
                model_revision=revision,
                runtime=RuntimeKind.REMOTE,
                device_placement=("remote",),
                dtype=policy.dtype,
                quantization=policy.quantization,
                attention_backend=policy.attention_backend,
                allow_cpu_offload=policy.allow_cpu_offload,
                allow_disk_offload=policy.allow_disk_offload,
                offload_directory=(
                    str(policy.offload_directory) if policy.allow_disk_offload else None
                ),
                remote_endpoint=str(policy.remote_endpoint).rstrip("/"),
                rationale=("Explicit remote runtime policy; placement is verified by the worker.",),
                hardware_validation=HardwareValidationStatus.PENDING,
            )

        if policy.quantization not in {None, "int8", "int4"}:
            raise self._reject(
                RuntimeRejectionCode.INVALID_QUANTIZATION,
                "Quantization must be null, int8, or int4.",
                model,
                policy,
            )
        if policy.quantization is not None and policy.quantization not in self.quantization_support:
            raise self._reject(
                RuntimeRejectionCode.INVALID_QUANTIZATION,
                "The requested quantization adapter is not installed or capability-probed.",
                model,
                policy,
                quantization=policy.quantization,
            )
        try:
            dtype = self._dtype(model, inventory, policy)
        except ValueError as error:
            raise self._reject(
                RuntimeRejectionCode.INVALID_DTYPE,
                "Dtype must be auto, float32, float16, or bfloat16.",
                model,
                policy,
            ) from error
        if dtype == "bfloat16" and inventory.accelerators:
            capabilities = [device.compute_capability for device in inventory.accelerators]
            if any(value is not None and int(value.split(".", 1)[0]) < 8 for value in capabilities):
                raise self._reject(
                    RuntimeRejectionCode.INVALID_DTYPE,
                    "bfloat16 requires a compatible visible accelerator.",
                    model,
                    policy,
                )
        try:
            attention = self._attention(dtype, inventory, policy)
        except ValueError as error:
            raise self._reject(
                RuntimeRejectionCode.INVALID_ATTENTION_BACKEND,
                "The requested attention backend is invalid or unavailable for this placement.",
                model,
                policy,
            ) from error

        memory_estimate = self._model_memory(model, policy, dtype)

        if policy.device is DevicePolicy.CPU:
            if policy.engine not in {RuntimeEngine.AUTO, RuntimeEngine.TRANSFORMERS}:
                raise self._reject(
                    RuntimeRejectionCode.UNSUPPORTED_RUNTIME,
                    "Explicit CPU placement supports the Transformers runtime only.",
                    model,
                    policy,
                )
            if policy.quantization is not None:
                raise self._reject(
                    RuntimeRejectionCode.INVALID_QUANTIZATION,
                    "The base CPU runtime does not enable accelerator quantization adapters.",
                    model,
                    policy,
                )
            self._require_capability(RuntimeKind.CPU_TRANSFORMERS, model, policy)
            usable_cpu_memory = int(
                min(inventory.total_ram_bytes, inventory.available_ram_bytes)
                * (1 - policy.memory_headroom_fraction)
            )
            if memory_estimate is not None and memory_estimate > usable_cpu_memory:
                if not policy.allow_disk_offload:
                    raise self._reject(
                        RuntimeRejectionCode.INSUFFICIENT_MEMORY,
                        "The requested model does not credibly fit available CPU memory.",
                        model,
                        policy,
                        required_bytes=memory_estimate,
                        available_bytes=usable_cpu_memory,
                    )
                if policy.engine is not RuntimeEngine.AUTO:
                    raise self._reject(
                        RuntimeRejectionCode.UNSUPPORTED_RUNTIME,
                        "CPU disk offload requires the Accelerate runtime.",
                        model,
                        policy,
                    )
                self._require_capability(RuntimeKind.ACCELERATE_DISPATCH, model, policy)
                return RuntimePlan(
                    requested_profile=model.name,
                    requested_model_id=model.model_id,
                    model_revision=revision,
                    runtime=RuntimeKind.ACCELERATE_DISPATCH,
                    device_placement=("cpu",),
                    dtype=dtype,
                    quantization=None,
                    attention_backend=attention,
                    allow_disk_offload=True,
                    offload_directory=str(policy.offload_directory),
                    model_memory_estimate_bytes=memory_estimate,
                    reserved_headroom_bytes=(inventory.available_ram_bytes - usable_cpu_memory),
                    device_map={"model": "auto"},
                    memory_limits={"cpu": usable_cpu_memory},
                    rationale=(
                        "Explicit disk offload permits memory-oriented CPU Accelerate dispatch.",
                    ),
                    warnings=("Disk offload is enabled and may be very slow.",),
                    hardware_validation=HardwareValidationStatus.NOT_APPLICABLE,
                )
            return RuntimePlan(
                requested_profile=model.name,
                requested_model_id=model.model_id,
                model_revision=revision,
                runtime=RuntimeKind.CPU_TRANSFORMERS,
                device_placement=("cpu",),
                dtype=dtype,
                quantization=None,
                attention_backend=attention,
                allow_cpu_offload=False,
                allow_disk_offload=policy.allow_disk_offload,
                offload_directory=(
                    str(policy.offload_directory) if policy.allow_disk_offload else None
                ),
                model_memory_estimate_bytes=memory_estimate,
                rationale=("Explicit CPU policy takes precedence over every visible accelerator.",),
                warnings=(
                    ("Disk offload is enabled and may be very slow.",)
                    if policy.allow_disk_offload
                    else ()
                ),
                hardware_validation=HardwareValidationStatus.NOT_APPLICABLE,
            )

        devices = inventory.accelerators
        if not devices:
            if policy.device is DevicePolicy.AUTO and policy.engine in {
                RuntimeEngine.AUTO,
                RuntimeEngine.TRANSFORMERS,
            }:
                cpu_policy = policy.model_copy(update={"device": DevicePolicy.CPU})
                return self.plan(model, inventory, cpu_policy, revision=revision)
            raise self._reject(
                RuntimeRejectionCode.NO_VISIBLE_ACCELERATOR,
                "No accelerator was discovered in the supplied inventory.",
                model,
                policy,
            )
        if memory_estimate is None:
            raise self._reject(
                RuntimeRejectionCode.UNKNOWN_MODEL_SIZE,
                "GPU planning for a custom model requires model_memory_bytes.",
                model,
                policy,
            )

        usable = tuple(
            int(
                min(device.total_memory_bytes, device.free_memory_bytes)
                * (1 - policy.memory_headroom_fraction)
            )
            for device in devices
        )
        fitting = [index for index, amount in enumerate(usable) if amount >= memory_estimate]
        requested_engine = policy.engine
        if requested_engine in {RuntimeEngine.AUTO, RuntimeEngine.TRANSFORMERS} and fitting:
            self._require_capability(RuntimeKind.CUDA_TRANSFORMERS, model, policy)
            selected = fitting[0]
            device = devices[selected]
            return RuntimePlan(
                requested_profile=model.name,
                requested_model_id=model.model_id,
                model_revision=revision,
                runtime=RuntimeKind.CUDA_TRANSFORMERS,
                device_placement=(f"cuda:{device.visible_index}",),
                dtype=dtype,
                quantization=policy.quantization,
                attention_backend=attention,
                allow_cpu_offload=policy.allow_cpu_offload,
                allow_disk_offload=policy.allow_disk_offload,
                offload_directory=(
                    str(policy.offload_directory) if policy.allow_disk_offload else None
                ),
                model_memory_estimate_bytes=memory_estimate,
                reserved_headroom_bytes=device.free_memory_bytes - usable[selected],
                rationale=(
                    "The unchanged requested model fits one visible GPU with configured headroom.",
                ),
                warnings=(
                    ("Disk offload is enabled but is not required by this placement.",)
                    if policy.allow_disk_offload
                    else ()
                ),
                hardware_validation=HardwareValidationStatus.PENDING,
            )
        if requested_engine is RuntimeEngine.TRANSFORMERS:
            raise self._reject(
                RuntimeRejectionCode.INSUFFICIENT_MEMORY,
                "The requested model does not fit any single visible GPU with headroom.",
                model,
                policy,
                required_bytes=memory_estimate,
                largest_usable_bytes=max(usable),
            )

        if not self._homogeneous(inventory):
            raise self._reject(
                RuntimeRejectionCode.HETEROGENEOUS_DEVICES,
                "Multi-device placement requires a homogeneous validated inventory.",
                model,
                policy,
            )
        usable_cpu_memory = int(
            inventory.available_ram_bytes * (1 - policy.memory_headroom_fraction)
        )
        fallback_memory = sum(usable) + (
            usable_cpu_memory if policy.allow_cpu_offload or policy.allow_disk_offload else 0
        )
        if sum(usable) < memory_estimate:
            if fallback_memory < memory_estimate and not policy.allow_disk_offload:
                raise self._reject(
                    RuntimeRejectionCode.INSUFFICIENT_MEMORY,
                    "The requested model does not fit visible GPU and allowed CPU memory.",
                    model,
                    policy,
                    required_bytes=memory_estimate,
                    aggregate_usable_bytes=sum(usable),
                    usable_cpu_bytes=(usable_cpu_memory if policy.allow_cpu_offload else 0),
                )
            if not policy.allow_cpu_offload and not policy.allow_disk_offload:
                raise self._reject(
                    RuntimeRejectionCode.INSUFFICIENT_MEMORY,
                    "The requested model does not fit aggregate visible GPU memory with headroom.",
                    model,
                    policy,
                    required_bytes=memory_estimate,
                    aggregate_usable_bytes=sum(usable),
                )

        if requested_engine in {RuntimeEngine.AUTO, RuntimeEngine.ACCELERATE}:
            self._require_capability(RuntimeKind.ACCELERATE_DISPATCH, model, policy)
            memory_limits = {
                f"cuda:{device.visible_index}": amount
                for device, amount in zip(devices, usable, strict=True)
            }
            if policy.allow_cpu_offload or policy.allow_disk_offload:
                memory_limits["cpu"] = usable_cpu_memory
            return RuntimePlan(
                requested_profile=model.name,
                requested_model_id=model.model_id,
                model_revision=revision,
                runtime=RuntimeKind.ACCELERATE_DISPATCH,
                device_placement=tuple(memory_limits),
                dtype=dtype,
                quantization=policy.quantization,
                attention_backend=attention,
                allow_cpu_offload=policy.allow_cpu_offload,
                allow_disk_offload=policy.allow_disk_offload,
                offload_directory=(
                    str(policy.offload_directory) if policy.allow_disk_offload else None
                ),
                model_memory_estimate_bytes=memory_estimate,
                reserved_headroom_bytes=sum(
                    device.free_memory_bytes - amount
                    for device, amount in zip(devices, usable, strict=True)
                ),
                device_map={"model": "auto"},
                memory_limits=memory_limits,
                rationale=(
                    "The model needs memory-oriented Accelerate dispatch across visible devices.",
                    "This plan makes no claim of tensor-parallel latency or throughput gains.",
                ),
                warnings=(
                    ("Disk offload is enabled and may be very slow.",)
                    if policy.allow_disk_offload
                    else ()
                ),
                hardware_validation=HardwareValidationStatus.PENDING,
            )

        raise self._reject(
            RuntimeRejectionCode.UNSUPPORTED_RUNTIME,
            f"Runtime {requested_engine.value!r} is not supported.",
            model,
            policy,
        )
