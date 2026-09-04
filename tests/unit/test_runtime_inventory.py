from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import oasis.runtimes.inventory as inventory_module


def test_cuda_inventory_uses_cuda_bindings_driver_fallback(monkeypatch: Any) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu128",
        _C=SimpleNamespace(),
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda _index: SimpleNamespace(
                name="Fake CUDA",
                major=12,
                minor=0,
                uuid="fake-uuid",
            ),
            mem_get_info=lambda _index: (12_000, 16_000),
        ),
    )
    fake_driver = SimpleNamespace(cuDriverGetVersion=lambda: (0, 13_030))

    def import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "cuda.bindings.driver":
            return fake_driver
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(inventory_module.importlib, "import_module", import_module)

    inventory = inventory_module.inspect_cuda_inventory()

    assert inventory.driver_version == "13.3"
    assert inventory.library_versions["cuda_runtime"] == "12.8"
    assert inventory.accelerators[0].compute_capability == "12.0"
