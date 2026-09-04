"""Hardware-neutral runtime planning and inference adapters."""

from oasis.config import RuntimePolicy
from oasis.runtimes.inventory import (
    fake_inventory,
    inspect_cuda_inventory,
    named_fake_inventory,
    safe_cpu_inventory,
)
from oasis.runtimes.planner import (
    ConservativeRuntimePlanner,
    RuntimePlanningError,
    RuntimeRejection,
    RuntimeRejectionCode,
    installed_runtime_capabilities,
)
from oasis.runtimes.protocols import InferenceRuntime, RuntimePlanner
from oasis.runtimes.remote import RemoteModelRuntime, RemoteRuntimeError
from oasis.runtimes.schemas import (
    AcceleratorDevice,
    ComputeInventory,
    DiscoveryMode,
    HardwareValidationStatus,
    RuntimeCapability,
    RuntimeKind,
    RuntimeMetrics,
    RuntimePlan,
    evaluation_group_key,
)

__all__ = [
    "AcceleratorDevice",
    "ComputeInventory",
    "ConservativeRuntimePlanner",
    "DiscoveryMode",
    "HardwareValidationStatus",
    "InferenceRuntime",
    "RemoteModelRuntime",
    "RemoteRuntimeError",
    "RuntimeCapability",
    "RuntimeKind",
    "RuntimeMetrics",
    "RuntimePlan",
    "RuntimePlanner",
    "RuntimePlanningError",
    "RuntimePolicy",
    "RuntimeRejection",
    "RuntimeRejectionCode",
    "evaluation_group_key",
    "fake_inventory",
    "inspect_cuda_inventory",
    "installed_runtime_capabilities",
    "named_fake_inventory",
    "safe_cpu_inventory",
]
