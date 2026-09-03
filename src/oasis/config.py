"""Side-effect-free application configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

from oasis.errors import ModelBackendError, ModelErrorCode, ModelErrorDetail


class DevicePolicy(StrEnum):
    """User-requested device policy; probing happens only during lazy model load."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class RuntimeEngine(StrEnum):
    """Supported configuration values, including engines implemented in later phases."""

    AUTO = "auto"
    TRANSFORMERS = "transformers"
    ACCELERATE = "accelerate"
    DEEPSPEED = "deepspeed"
    VLLM = "vllm"
    REMOTE = "remote"


class BackendKind(StrEnum):
    """Available model backend choices."""

    TRANSFORMERS = "transformers"
    FAKE = "fake"


class RuntimeConfig(BaseModel):
    """Serializable requested runtime policy without discovered hardware state."""

    model_config = ConfigDict(frozen=True)

    device: DevicePolicy = DevicePolicy.CPU
    engine: RuntimeEngine = RuntimeEngine.AUTO
    dtype: str = "auto"
    quantization: str | None = None
    parallelism: str = "auto"
    memory_headroom_fraction: float = Field(default=0.10, ge=0.0, lt=1.0)
    allow_cpu_offload: bool = False
    allow_disk_offload: bool = False
    remote_endpoint: HttpUrl | None = None


class OasisSettings(BaseSettings):
    """Application settings with init/CLI values taking precedence over the environment."""

    model_config = SettingsConfigDict(
        env_prefix="OASIS_",
        case_sensitive=False,
        extra="forbid",
        validate_default=True,
    )

    backend: BackendKind = BackendKind.TRANSFORMERS
    model_profile: str = "gemma4_e4b_it"
    model_id: str | None = None
    model_revision: str | None = None
    max_generated_tokens: int = Field(default=512, ge=1)
    thinking: bool = False
    trust_remote_code: bool = False

    device: DevicePolicy = DevicePolicy.CPU
    runtime_engine: RuntimeEngine = RuntimeEngine.AUTO
    dtype: str = "auto"
    quantization: str | None = None
    parallelism: str = "auto"
    memory_headroom_fraction: float = Field(default=0.10, ge=0.0, lt=1.0)
    allow_cpu_offload: bool = False
    allow_disk_offload: bool = False
    remote_endpoint: HttpUrl | None = None
    artifact_root: Path = Path(".oasis/artifacts")
    provider_cache_root: Path = Path(".oasis/provider-cache")
    provider_user_agent: str = "oasis-anytime-agents/0.1 (configure OASIS_PROVIDER_USER_AGENT)"
    provider_timeout_seconds: float = Field(default=10.0, gt=0)
    provider_max_attempts: int = Field(default=3, ge=1, le=10)
    provider_backoff_base_seconds: float = Field(default=0.25, ge=0)
    provider_max_response_bytes: int = Field(default=10_000_000, ge=1)
    provider_max_pages: int = Field(default=20, ge=1)

    @classmethod
    def resolve(
        cls,
        *,
        cli_overrides: Mapping[str, Any] | None = None,
        explicit_overrides: Mapping[str, Any] | None = None,
    ) -> OasisSettings:
        """Resolve defaults < environment < CLI < explicit object overrides."""

        values: dict[str, Any] = {}
        for source in (cli_overrides, explicit_overrides):
            if source:
                values.update({key: value for key, value in source.items() if value is not None})
        return cls(**values)

    def runtime_config(self) -> RuntimeConfig:
        """Return the requested runtime policy without probing hardware."""

        return RuntimeConfig(
            device=self.device,
            engine=self.runtime_engine,
            dtype=self.dtype,
            quantization=self.quantization,
            parallelism=self.parallelism,
            memory_headroom_fraction=self.memory_headroom_fraction,
            allow_cpu_offload=self.allow_cpu_offload,
            allow_disk_offload=self.allow_disk_offload,
            remote_endpoint=self.remote_endpoint,
        )


def resolve_device(
    policy: DevicePolicy,
    cuda_is_available: Callable[[], bool],
) -> DevicePolicy:
    """Resolve an explicit policy; CPU never invokes the CUDA availability probe."""

    if policy is DevicePolicy.CPU:
        return DevicePolicy.CPU
    available = cuda_is_available()
    if policy is DevicePolicy.CUDA and not available:
        raise ModelBackendError(
            ModelErrorDetail(
                code=ModelErrorCode.MODEL_UNAVAILABLE,
                message="CUDA was explicitly requested but is not available to this process.",
                capability="cuda",
            )
        )
    return DevicePolicy.CUDA if available else DevicePolicy.CPU
