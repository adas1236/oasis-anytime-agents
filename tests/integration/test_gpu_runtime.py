from __future__ import annotations

import gc
import importlib
import os
from pathlib import Path
from typing import Any

import pytest

from oasis.artifacts import LocalArtifactStore
from oasis.config import DevicePolicy, RuntimePolicy
from oasis.controller import AnytimeController, BudgetSpec, InMemoryRunStore, RunRequest
from oasis.evaluation import FixtureName, generate_instance, load_fixture
from oasis.llm import ChatMessage, FinishReason, ModelRequest
from oasis.llm.profiles import resolve_model_profile
from oasis.llm.transformers_backend import TransformersModelBackend
from oasis.runtimes import (
    ConservativeRuntimePlanner,
    DiscoveryMode,
    HardwareValidationStatus,
    RuntimeKind,
    inspect_cuda_inventory,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_explicit_gpu_opt_in() -> None:
    if os.environ.get("OASIS_RUN_GPU_TESTS") != "1":
        pytest.skip("set OASIS_RUN_GPU_TESTS=1 after selecting the GPU dependency group")


def _torch() -> Any:
    torch: Any = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible to the test process")
    return torch


def _model_memory_override(model_id: str) -> int | None:
    configured = os.environ.get("OASIS_GPU_TEST_MODEL_MEMORY_BYTES")
    if configured:
        return int(configured)
    if resolve_model_profile(explicit_model_id=model_id).estimated_parameter_count is None:
        pytest.skip("set OASIS_GPU_TEST_MODEL_MEMORY_BYTES for an unregistered model")
    return None


def test_real_cuda_inventory_is_complete_and_sanitizable() -> None:
    inventory = inspect_cuda_inventory()

    assert inventory.discovery_mode is DiscoveryMode.CUDA_PROBE
    assert inventory.accelerator_count >= 1
    assert inventory.driver_version is not None
    assert inventory.library_versions["cuda_runtime"]
    assert all(
        device.free_memory_bytes <= device.total_memory_bytes for device in inventory.accelerators
    )
    assert all(device.compute_capability for device in inventory.accelerators)
    assert all(device.uuid is None for device in inventory.sanitized().accelerators)


def test_real_cuda_bfloat16_matmul_and_sdpa_kernels() -> None:
    torch = _torch()
    torch.manual_seed(2026)
    left = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    product = left @ left.transpose(0, 1)
    query = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.bfloat16)
    attended = torch.nn.functional.scaled_dot_product_attention(query, query, query)
    torch.cuda.synchronize()

    assert product.shape == (256, 256)
    assert attended.shape == query.shape
    assert bool(torch.isfinite(product).all())
    assert bool(torch.isfinite(attended).all())


def test_real_cuda_inventory_plans_single_gpu_gemma_e2b() -> None:
    inventory = inspect_cuda_inventory()
    plan = ConservativeRuntimePlanner().plan(
        resolve_model_profile("gemma4_e2b_it"),
        inventory,
        RuntimePolicy(
            device=DevicePolicy.CUDA,
            dtype="bfloat16",
            attention_backend="sdpa",
        ),
    )

    assert plan.runtime is RuntimeKind.CUDA_TRANSFORMERS
    assert plan.device_placement == ("cuda:0",)
    assert plan.hardware_validation is HardwareValidationStatus.PENDING


@pytest.mark.asyncio
async def test_cached_compatible_model_generates_and_aborts_on_real_cuda() -> None:
    model_id = os.environ.get("OASIS_GPU_TEST_MODEL")
    if not model_id:
        pytest.skip("set OASIS_GPU_TEST_MODEL to a compatible local or cached chat model")
    memory_bytes = _model_memory_override(model_id)
    torch = _torch()
    backend = TransformersModelBackend(
        model_id=model_id,
        device=DevicePolicy.CUDA,
        dtype="bfloat16",
        attention_backend="sdpa",
        model_memory_bytes=memory_bytes,
        inventory=inspect_cuda_inventory(),
    )
    try:
        completed = await backend.generate(
            ModelRequest(
                request_id="gpu-complete",
                messages=(ChatMessage(role="user", content="Reply with one short word."),),
                max_generated_tokens=8,
            )
        )

        assert completed.message.content
        assert completed.usage.generated_tokens > 0
        assert completed.finish_reason in {FinishReason.STOP, FinishReason.LENGTH}
        assert backend.runtime_plan.hardware_validation is HardwareValidationStatus.PASSED
        assert backend.runtime_plan.metrics.peak_device_memory_bytes

        cancelled = []
        abort_sent = False
        request = ModelRequest(
            request_id="gpu-abort",
            messages=(ChatMessage(role="user", content="Count upward for a long time."),),
            max_generated_tokens=128,
        )
        async for delta in backend.stream(request):
            cancelled.append(delta)
            if delta.text and not abort_sent:
                await backend.abort(request.request_id)
                abort_sent = True

        assert abort_sent
        assert cancelled[-1].finish_reason is FinishReason.CANCELLED
        assert cancelled[-1].usage is not None
        assert cancelled[-1].usage.generated_tokens < request.max_generated_tokens
    finally:
        await backend.close()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.asyncio
async def test_cached_model_controller_run_retains_a_verified_plan_on_real_cuda(
    tmp_path: Path,
) -> None:
    model_id = os.environ.get("OASIS_GPU_TEST_MODEL")
    if not model_id:
        pytest.skip("set OASIS_GPU_TEST_MODEL to a compatible local or cached chat model")
    memory_bytes = _model_memory_override(model_id)
    torch = _torch()
    inventory = inspect_cuda_inventory()
    backend = TransformersModelBackend(
        model_id=model_id,
        device=DevicePolicy.CUDA,
        dtype="bfloat16",
        attention_backend="sdpa",
        model_memory_bytes=memory_bytes,
        inventory=inventory,
    )
    store = LocalArtifactStore(tmp_path / "artifacts")
    instance = await generate_instance(
        "gpu-controller",
        load_fixture(FixtureName.CLINIC_ACCESS),
        store,
    )
    assert instance.problem_artifact_id and instance.baseline_plan_artifact_id
    try:
        await backend.load()
        assert backend.runtime_plan.hardware_validation is HardwareValidationStatus.PENDING
        result = await AnytimeController(
            artifact_store=store,
            run_store=InMemoryRunStore(),
            backend=backend,
        ).run(
            RunRequest(
                run_id="gpu-controller",
                problem_artifact_id=instance.problem_artifact_id,
                baseline_plan_artifact_id=instance.baseline_plan_artifact_id,
                budget=BudgetSpec(
                    wall_time_ms=30_000,
                    max_total_model_tokens=4_096,
                    max_generated_tokens=8,
                    max_tool_calls=2,
                ),
                runtime_plan=backend.runtime_plan,
                compute_inventory=inventory,
            )
        )

        assert result.best_plan is not None
        assert result.best_scorecard is not None and result.best_scorecard.feasible
        assert result.hardware_validation == "passed", (
            result.terminal_reason,
            result.failures,
            result.consumed_budget,
            result.runtime_plan.metrics,
        )
        assert result.runtime_plan.runtime is RuntimeKind.CUDA_TRANSFORMERS
        assert result.runtime_plan.metrics.generated_tokens > 0
        assert result.compute_inventory.accelerators[0].uuid is None
    finally:
        await backend.close()
        gc.collect()
        torch.cuda.empty_cache()
