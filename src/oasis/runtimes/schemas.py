"""Typed, serializable hardware inventory and resolved runtime contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class DiscoveryMode(StrEnum):
    """How inventory facts were obtained."""

    SAFE_CPU = "safe_cpu"
    FAKE = "fake"
    CUDA_PROBE = "cuda_probe"


class RuntimeKind(StrEnum):
    """Concrete execution adapter selected by the planner."""

    FAKE = "fake"
    CPU_TRANSFORMERS = "cpu_transformers"
    CUDA_TRANSFORMERS = "cuda_transformers"
    ACCELERATE_DISPATCH = "accelerate_dispatch"
    REMOTE = "remote"


class HardwareValidationStatus(StrEnum):
    """Whether a plan has been exercised on the hardware it describes."""

    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class AcceleratorDevice(BaseModel):
    """One accelerator visible inside the current process allocation."""

    model_config = ConfigDict(frozen=True)

    visible_index: int = Field(ge=0)
    kind: str = Field(default="cuda", min_length=1)
    name: str = Field(min_length=1)
    total_memory_bytes: int = Field(ge=1)
    free_memory_bytes: int = Field(ge=0)
    compute_capability: str | None = None
    uuid: str | None = None

    @model_validator(mode="after")
    def free_memory_fits_total(self) -> Self:
        if self.free_memory_bytes > self.total_memory_bytes:
            raise ValueError("accelerator free memory cannot exceed total memory")
        return self


class ComputeInventory(BaseModel):
    """CPU/RAM and explicitly discovered devices visible on one machine."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    cpu_count: int = Field(ge=1)
    total_ram_bytes: int = Field(ge=1)
    available_ram_bytes: int = Field(ge=0)
    accelerators: tuple[AcceleratorDevice, ...] = ()
    discovery_mode: DiscoveryMode = DiscoveryMode.SAFE_CPU
    platform: str | None = None
    python_version: str | None = None
    driver_version: str | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accelerator_count(self) -> int:
        return len(self.accelerators)

    @model_validator(mode="after")
    def memory_and_devices_are_consistent(self) -> Self:
        if self.available_ram_bytes > self.total_ram_bytes:
            raise ValueError("available RAM cannot exceed total RAM")
        indices = tuple(device.visible_index for device in self.accelerators)
        if len(indices) != len(set(indices)):
            raise ValueError("accelerator visible indices must be unique")
        return self

    def sanitized(self) -> ComputeInventory:
        """Remove device identifiers that are unnecessary for clients."""

        devices = tuple(device.model_copy(update={"uuid": None}) for device in self.accelerators)
        return self.model_copy(update={"accelerators": devices})


class RuntimeMetrics(BaseModel):
    """Observed load/generation measurements for reproducible comparisons."""

    model_config = ConfigDict(frozen=True)

    startup_ms: int = Field(default=0, ge=0)
    request_count: int = Field(default=0, ge=0)
    generated_tokens: int = Field(default=0, ge=0)
    generation_ms: int = Field(default=0, ge=0)
    peak_device_memory_bytes: int | None = Field(default=None, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tokens_per_second(self) -> float | None:
        if self.generation_ms == 0:
            return None
        return self.generated_tokens / (self.generation_ms / 1_000)


class RuntimePlan(BaseModel):
    """Resolved, immutable placement plan for one unchanged requested model."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    requested_profile: str
    requested_model_id: str
    model_revision: str | None = None
    runtime: RuntimeKind
    device_placement: tuple[str, ...]
    dtype: str
    quantization: str | None = None
    attention_backend: str
    allow_cpu_offload: bool = False
    allow_disk_offload: bool = False
    offload_directory: str | None = None
    model_memory_estimate_bytes: int | None = Field(default=None, ge=1)
    reserved_headroom_bytes: int = Field(default=0, ge=0)
    device_map: dict[str, str | int] = Field(default_factory=dict)
    memory_limits: dict[str, int] = Field(default_factory=dict)
    remote_endpoint: str | None = None
    rationale: tuple[str, ...] = Field(min_length=1)
    warnings: tuple[str, ...] = ()
    hardware_validation: HardwareValidationStatus = HardwareValidationStatus.PENDING
    metrics: RuntimeMetrics = Field(default_factory=RuntimeMetrics)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_startup_ms(self) -> int:
        """Compatibility projection for the Phase 8 opaque plan dictionary."""

        return self.metrics.startup_ms

    def __getitem__(self, key: str) -> object:
        """Retain read compatibility with Phase 8 dictionary-valued plan fields."""

        if key == "model_startup_ms":
            return self.model_startup_ms
        try:
            return self.model_dump(mode="json")[key]
        except KeyError as error:
            raise KeyError(key) from error

    @model_validator(mode="after")
    def placement_is_consistent(self) -> Self:
        if self.runtime is RuntimeKind.REMOTE and self.remote_endpoint is None:
            raise ValueError("remote runtime plans require an endpoint")
        if self.runtime is not RuntimeKind.REMOTE and self.remote_endpoint is not None:
            raise ValueError("only remote runtime plans may contain an endpoint")
        if self.allow_disk_offload and self.offload_directory is None:
            raise ValueError("disk-offload plans require an offload directory")
        if not self.allow_disk_offload and self.offload_directory is not None:
            raise ValueError("only disk-offload plans may contain an offload directory")
        return self

    def evaluation_group_key(self) -> tuple[str, ...]:
        """Stable grouping key that prevents unlike runtime/hardware results being pooled."""

        return (
            self.requested_model_id,
            self.model_revision or "default",
            self.runtime.value,
            self.dtype,
            self.quantization or "none",
            ",".join(self.device_placement),
        )


class RuntimeCapability(BaseModel):
    """Safe package-availability result for one execution adapter."""

    model_config = ConfigDict(frozen=True)

    runtime: RuntimeKind
    installed: bool
    version: str | None = None
    reason: str | None = None


def evaluation_group_key(
    plan: RuntimePlan,
    inventory: ComputeInventory,
) -> tuple[str, ...]:
    """Group only runs sharing model, runtime, visible devices, and library stack."""

    devices = (
        ";".join(
            ":".join(
                (
                    device.kind,
                    device.name,
                    str(device.total_memory_bytes),
                    device.compute_capability or "-",
                )
            )
            for device in inventory.accelerators
        )
        or "cpu"
    )
    libraries = ";".join(
        f"{name}={version}" for name, version in sorted(inventory.library_versions.items())
    )
    return (
        *plan.evaluation_group_key(),
        devices,
        inventory.driver_version or "none",
        libraries,
    )
