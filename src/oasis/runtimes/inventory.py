"""Lazy CPU/environment inventory and explicitly invoked CUDA discovery."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from collections.abc import Mapping
from importlib import metadata
from typing import Any

from oasis.runtimes.schemas import AcceleratorDevice, ComputeInventory, DiscoveryMode


def _memory() -> tuple[int, int]:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
    available_pages_name = "SC_AVPHYS_PAGES"
    available = (
        page_size * int(os.sysconf(available_pages_name))
        if available_pages_name in os.sysconf_names
        else total
    )
    return total, min(total, available)


def safe_cpu_inventory(
    environment: Mapping[str, str] | None = None,
    *,
    library_versions: bool = True,
) -> ComputeInventory:
    """Inspect CPU and RAM without importing Torch or probing CUDA."""

    resolved_environment = os.environ if environment is None else environment
    total_ram, available_ram = _memory()
    versions: dict[str, str] = {}
    if library_versions:
        for distribution in ("transformers", "accelerate"):
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
    warnings: list[str] = []
    if resolved_environment.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}:
        warnings.append(
            "CUDA visibility is configured but devices were not inspected; use the explicit "
            "hardware probe command to discover them."
        )
    return ComputeInventory(
        cpu_count=os.cpu_count() or 1,
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        discovery_mode=DiscoveryMode.SAFE_CPU,
        platform=platform.platform(),
        python_version=platform.python_version(),
        library_versions=versions,
        warnings=tuple(warnings),
    )


def inspect_cuda_inventory(
    environment: Mapping[str, str] | None = None,
) -> ComputeInventory:
    """Explicitly import Torch and query only CUDA devices visible to this process."""

    base = safe_cpu_inventory(environment)
    torch: Any = importlib.import_module("torch")
    if not torch.cuda.is_available():
        return base.model_copy(
            update={
                "discovery_mode": DiscoveryMode.CUDA_PROBE,
                "warnings": (*base.warnings, "CUDA probe completed but no device is available."),
                "library_versions": {**base.library_versions, "torch": str(torch.__version__)},
            }
        )
    accelerators: list[AcceleratorDevice] = []
    for index in range(int(torch.cuda.device_count())):
        properties = torch.cuda.get_device_properties(index)
        free_memory, total_memory = torch.cuda.mem_get_info(index)
        major = getattr(properties, "major", None)
        minor = getattr(properties, "minor", None)
        capability = f"{major}.{minor}" if major is not None and minor is not None else None
        accelerators.append(
            AcceleratorDevice(
                visible_index=index,
                kind="cuda",
                name=str(properties.name),
                total_memory_bytes=int(total_memory),
                free_memory_bytes=int(free_memory),
                compute_capability=capability,
                uuid=str(getattr(properties, "uuid", "")) or None,
            )
        )
    driver_version = None
    driver_getter = getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None)
    if callable(driver_getter):
        raw_driver = int(driver_getter())
        driver_version = f"{raw_driver // 1000}.{(raw_driver % 1000) // 10}"
    else:
        try:
            cuda_driver: Any = importlib.import_module("cuda.bindings.driver")
            result, raw_driver = cuda_driver.cuDriverGetVersion()
            if int(result) == 0:
                raw_driver = int(raw_driver)
                driver_version = f"{raw_driver // 1000}.{(raw_driver % 1000) // 10}"
        except (ImportError, OSError, TypeError, ValueError):
            pass
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    versions = {**base.library_versions, "torch": str(torch.__version__)}
    if cuda_version is not None:
        versions["cuda_runtime"] = str(cuda_version)
    return base.model_copy(
        update={
            "accelerators": tuple(accelerators),
            "discovery_mode": DiscoveryMode.CUDA_PROBE,
            "driver_version": driver_version,
            "library_versions": versions,
        }
    )


def fake_inventory(
    *,
    accelerator_memory_bytes: tuple[int, ...] = (),
    accelerator_names: tuple[str, ...] | None = None,
    free_memory_bytes: tuple[int, ...] | None = None,
    compute_capabilities: tuple[str, ...] | None = None,
    total_ram_bytes: int = 64 * 1024**3,
) -> ComputeInventory:
    """Create deterministic CPU/GPU fixtures without importing accelerator libraries."""

    count = len(accelerator_memory_bytes)
    names = accelerator_names or tuple("Fake CUDA" for _ in range(count))
    free = free_memory_bytes or accelerator_memory_bytes
    capabilities = compute_capabilities or tuple("8.9" for _ in range(count))
    if not (len(names) == len(free) == len(capabilities) == count):
        raise ValueError("fake accelerator fields must have equal lengths")
    return ComputeInventory(
        cpu_count=16,
        total_ram_bytes=total_ram_bytes,
        available_ram_bytes=int(total_ram_bytes * 0.9),
        accelerators=tuple(
            AcceleratorDevice(
                visible_index=index,
                name=names[index],
                total_memory_bytes=memory,
                free_memory_bytes=free[index],
                compute_capability=capabilities[index],
                uuid=f"fake-{index}",
            )
            for index, memory in enumerate(accelerator_memory_bytes)
        ),
        discovery_mode=DiscoveryMode.FAKE,
        platform="fake",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        library_versions={"torch": "fake", "transformers": "fake"},
    )


def named_fake_inventory(name: str) -> ComputeInventory:
    """Return documented single-machine dry-run inventories."""

    gib = 1024**3
    local_8 = fake_inventory(
        accelerator_memory_bytes=(8 * gib,),
        accelerator_names=("NVIDIA GeForce RTX 5060 Ti (fake)",),
    ).model_copy(
        update={
            "warnings": (
                "Unverified 8 GiB local-device scenario; inspect real hardware before use.",
            )
        }
    )
    local_16 = fake_inventory(
        accelerator_memory_bytes=(16 * gib,),
        accelerator_names=("NVIDIA GeForce RTX 5060 Ti (fake)",),
    ).model_copy(
        update={
            "warnings": (
                "Unverified 16 GiB local-device scenario; inspect real hardware before use.",
            )
        }
    )
    fixtures = {
        "cpu": fake_inventory(),
        "local-5060ti-8gb": local_8,
        "local-5060ti-16gb": local_16,
        "2x24gb": fake_inventory(accelerator_memory_bytes=(24 * gib, 24 * gib)),
        "4x80gb": fake_inventory(accelerator_memory_bytes=(80 * gib,) * 4),
    }
    try:
        return fixtures[name]
    except KeyError as error:
        raise ValueError(
            f"unknown fake inventory {name!r}; choose {', '.join(fixtures)}"
        ) from error
