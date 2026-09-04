"""Side-effect-free backend construction from application settings."""

from __future__ import annotations

import httpx

from oasis.config import BackendKind, OasisSettings, RuntimeEngine
from oasis.llm.adapters import Gemma4ChatAdapter, PlainChatAdapter
from oasis.llm.fake import FakeModelBackend
from oasis.llm.profiles import resolve_model_profile
from oasis.llm.protocols import ModelBackend
from oasis.llm.runtime_backend import RuntimeModelBackend
from oasis.llm.transformers_backend import TransformersModelBackend
from oasis.runtimes import (
    ComputeInventory,
    ConservativeRuntimePlanner,
    RemoteModelRuntime,
    safe_cpu_inventory,
)


def create_model_backend(
    settings: OasisSettings,
    *,
    inventory: ComputeInventory | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> ModelBackend:
    """Construct a lazy backend without probing CUDA, loading weights, or contacting a worker."""

    profile = resolve_model_profile(settings.model_profile, settings.model_id)
    resolved_inventory = inventory or safe_cpu_inventory()
    if settings.backend is BackendKind.FAKE:
        return FakeModelBackend(profile=profile, inventory=resolved_inventory)
    if settings.runtime_engine is RuntimeEngine.REMOTE:
        if settings.remote_endpoint is None or settings.remote_auth_token is None:
            raise ValueError(
                "remote runtime requires OASIS_REMOTE_ENDPOINT and OASIS_REMOTE_AUTH_TOKEN"
            )
        plan = ConservativeRuntimePlanner().plan(
            profile,
            resolved_inventory,
            settings.runtime_config(),
            revision=settings.model_revision,
        )
        runtime = RemoteModelRuntime(
            str(settings.remote_endpoint),
            auth_token=settings.remote_auth_token.get_secret_value(),
            client=http_client,
            inventory=resolved_inventory,
        )
        adapter = (
            Gemma4ChatAdapter(profile.model_id, profile.context_limit)
            if profile.family == "gemma4"
            else PlainChatAdapter(profile.model_id, profile.context_limit)
        )
        return RuntimeModelBackend(
            profile=profile,
            capabilities=adapter.capabilities,
            runtime=runtime,
            plan=plan,
        )
    return TransformersModelBackend(
        profile_name=settings.model_profile,
        model_id=settings.model_id,
        revision=settings.model_revision,
        device=settings.device,
        engine=settings.runtime_engine,
        dtype=settings.dtype,
        quantization=settings.quantization,
        attention_backend=settings.attention_backend,
        memory_headroom_fraction=settings.memory_headroom_fraction,
        allow_cpu_offload=settings.allow_cpu_offload,
        allow_disk_offload=settings.allow_disk_offload,
        offload_directory=settings.offload_root,
        model_memory_bytes=settings.model_memory_bytes,
        trust_remote_code=settings.trust_remote_code,
        inventory=resolved_inventory,
    )
