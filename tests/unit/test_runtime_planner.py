from __future__ import annotations

import json
import subprocess
import sys

import pytest

from oasis.config import RuntimeEngine, RuntimePolicy
from oasis.llm.profiles import resolve_model_profile
from oasis.runtimes import (
    ComputeInventory,
    ConservativeRuntimePlanner,
    RuntimeKind,
    RuntimePlan,
    RuntimePlanningError,
    RuntimeRejectionCode,
    evaluation_group_key,
    fake_inventory,
    named_fake_inventory,
    safe_cpu_inventory,
)

GIB = 1024**3


def test_explicit_cpu_wins_over_visible_cuda() -> None:
    inventory = fake_inventory(accelerator_memory_bytes=(80 * GIB,) * 4)

    plan = ConservativeRuntimePlanner().plan(
        resolve_model_profile(),
        inventory,
        RuntimePolicy(device="cpu"),
    )

    assert plan.runtime is RuntimeKind.CPU_TRANSFORMERS
    assert plan.device_placement == ("cpu",)
    assert plan.requested_model_id == "google/gemma-4-E4B-it"


def test_scheduler_environment_does_not_create_a_separate_runtime_mode() -> None:
    inventory = safe_cpu_inventory(
        {
            "SLURM_JOB_ID": "ignored",
            "SLURM_NNODES": "8",
            "WORLD_SIZE": "64",
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
        library_versions=False,
    )

    assert inventory.discovery_mode.value == "safe_cpu"
    assert "topology" not in inventory.model_dump()
    assert inventory.accelerator_count == 0
    assert inventory.warnings
    assert {engine.value for engine in RuntimeEngine} == {
        "auto",
        "transformers",
        "accelerate",
        "remote",
    }


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_planner_accepts_arbitrary_homogeneous_fake_gpu_counts(count: int) -> None:
    inventory = fake_inventory(accelerator_memory_bytes=(24 * GIB,) * count)
    model = resolve_model_profile("gemma4_31b_it")
    if count < 4:
        with pytest.raises(RuntimePlanningError) as caught:
            ConservativeRuntimePlanner().plan(
                model,
                inventory,
                RuntimePolicy(device="cuda", model_memory_bytes=70 * GIB),
            )
        assert caught.value.rejection.code is RuntimeRejectionCode.INSUFFICIENT_MEMORY
    else:
        plan = ConservativeRuntimePlanner().plan(
            model,
            inventory,
            RuntimePolicy(device="cuda", model_memory_bytes=70 * GIB),
        )
        assert plan.runtime is RuntimeKind.ACCELERATE_DISPATCH
        assert len(plan.memory_limits) == count
        assert "no claim" in plan.rationale[1].lower()


def test_zero_gpu_and_heterogeneous_gpu_rejections_are_typed() -> None:
    model = resolve_model_profile("gemma4_12b_it")
    planner = ConservativeRuntimePlanner()
    with pytest.raises(RuntimePlanningError) as missing:
        planner.plan(model, fake_inventory(), RuntimePolicy(device="cuda"))
    assert missing.value.rejection.code is RuntimeRejectionCode.NO_VISIBLE_ACCELERATOR

    heterogeneous = fake_inventory(
        accelerator_memory_bytes=(16 * GIB, 24 * GIB),
        accelerator_names=("A", "B"),
    )
    with pytest.raises(RuntimePlanningError) as rejected:
        planner.plan(
            model,
            heterogeneous,
            RuntimePolicy(device="cuda", model_memory_bytes=25 * GIB),
        )
    assert rejected.value.rejection.code is RuntimeRejectionCode.HETEROGENEOUS_DEVICES


def test_explicit_disk_offload_is_a_typed_accelerate_plan() -> None:
    inventory = fake_inventory(total_ram_bytes=8 * GIB)
    plan = ConservativeRuntimePlanner().plan(
        resolve_model_profile("gemma4_12b_it"),
        inventory,
        RuntimePolicy(device="cpu", allow_disk_offload=True),
    )

    assert plan.runtime is RuntimeKind.ACCELERATE_DISPATCH
    assert plan.allow_disk_offload
    assert plan.offload_directory == ".oasis/offload"
    assert plan.memory_limits["cpu"] < plan.model_memory_estimate_bytes


def test_headroom_dtype_quantization_and_attention_validation() -> None:
    planner = ConservativeRuntimePlanner(quantization_support=frozenset({"int8", "int4"}))
    inventory = fake_inventory(accelerator_memory_bytes=(16 * GIB,))
    model = resolve_model_profile("gemma4_e2b_it")
    with pytest.raises(RuntimePlanningError) as headroom:
        planner.plan(
            model,
            inventory,
            RuntimePolicy(
                device="cuda",
                model_memory_bytes=15 * GIB,
                memory_headroom_fraction=0.10,
            ),
        )
    assert headroom.value.rejection.code is RuntimeRejectionCode.INSUFFICIENT_MEMORY

    plan = planner.plan(
        model,
        inventory,
        RuntimePolicy(
            device="cuda",
            dtype="float16",
            quantization="int8",
            attention_backend="sdpa",
            model_memory_bytes=8 * GIB,
        ),
    )
    assert plan.dtype == "float16"
    assert plan.quantization == "int8"
    assert plan.attention_backend == "sdpa"

    with pytest.raises(RuntimePlanningError) as unavailable_quantization:
        ConservativeRuntimePlanner(quantization_support=frozenset()).plan(
            model,
            inventory,
            RuntimePolicy(device="cuda", quantization="int4", model_memory_bytes=4 * GIB),
        )
    assert (
        unavailable_quantization.value.rejection.code is RuntimeRejectionCode.INVALID_QUANTIZATION
    )

    old_gpu = fake_inventory(accelerator_memory_bytes=(16 * GIB,), compute_capabilities=("7.5",))
    with pytest.raises(RuntimePlanningError) as dtype:
        planner.plan(
            model,
            old_gpu,
            RuntimePolicy(device="cuda", dtype="bfloat16", model_memory_bytes=4 * GIB),
        )
    assert dtype.value.rejection.code is RuntimeRejectionCode.INVALID_DTYPE


def test_gemma_e2b_memory_estimate_uses_full_checkpoint_not_effective_parameters() -> None:
    planner = ConservativeRuntimePlanner()
    model = resolve_model_profile("gemma4_e2b_it")

    with pytest.raises(RuntimePlanningError) as too_small:
        planner.plan(
            model,
            named_fake_inventory("local-5060ti-8gb"),
            RuntimePolicy(device="cuda"),
        )

    assert too_small.value.rejection.code is RuntimeRejectionCode.INSUFFICIENT_MEMORY
    plan = planner.plan(
        model,
        named_fake_inventory("local-5060ti-16gb"),
        RuntimePolicy(device="cuda"),
    )
    assert plan.runtime is RuntimeKind.CUDA_TRANSFORMERS
    assert plan.model_memory_estimate_bytes is not None
    assert plan.model_memory_estimate_bytes > 11_500_000_000


def test_runtime_metadata_round_trip_and_grouping_include_hardware_plan() -> None:
    inventory = named_fake_inventory("4x80gb")
    plan = ConservativeRuntimePlanner().plan(
        resolve_model_profile("gemma4_31b_it"),
        inventory,
        RuntimePolicy(device="cuda"),
    )
    restored_inventory = ComputeInventory.model_validate_json(inventory.model_dump_json())
    restored_plan = RuntimePlan.model_validate_json(plan.model_dump_json())
    assert restored_inventory == inventory
    assert restored_plan == plan
    assert evaluation_group_key(restored_plan, restored_inventory) == evaluation_group_key(
        plan, inventory
    )
    other_hardware = named_fake_inventory("2x24gb")
    assert evaluation_group_key(plan, inventory) != evaluation_group_key(plan, other_hardware)
    assert plan.requested_model_id == "google/gemma-4-31B-it"


def test_importing_runtime_modules_does_not_import_torch() -> None:
    code = """
import json
import sys
import oasis.runtimes
print(json.dumps({"torch": "torch" in sys.modules, "cuda": "torch.cuda" in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {"torch": False, "cuda": False}
