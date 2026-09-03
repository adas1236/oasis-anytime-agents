from __future__ import annotations

import json
import subprocess
import sys
from queue import Queue
from types import SimpleNamespace
from typing import Any

import pytest

from oasis.config import DevicePolicy
from oasis.llm import transformers_backend
from oasis.llm.schemas import ChatMessage, ModelRequest
from oasis.llm.transformers_backend import TransformersModelBackend, _LoadedComponents


def test_constructing_transformers_backend_does_not_load_model() -> None:
    backend = TransformersModelBackend(device=DevicePolicy.CPU)

    assert not backend.is_loaded
    assert backend.profile.model_id == "google/gemma-4-E4B-it"


def test_cpu_loader_uses_gemma_processor_without_probing_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class Factory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Any:
            calls[cls.__name__] = (model_id, kwargs)
            if cls.__name__ == "AutoConfig":
                return SimpleNamespace(
                    is_decoder=False,
                    model_type="gemma4",
                    architectures=["Gemma4ForConditionalGeneration"],
                )
            if cls.__name__ == "AutoProcessor":
                return StubProcessor()
            model = SimpleNamespace()
            model.to = lambda device: calls.update(device=device)
            model.eval = lambda: calls.update(evaluated=True)
            return model

    auto_config = type("AutoConfig", (Factory,), {})
    auto_processor = type("AutoProcessor", (Factory,), {})
    auto_model = type("AutoModelForCausalLM", (Factory,), {})

    def unexpected_cuda_probe() -> bool:
        raise AssertionError("CPU policy must not call torch.cuda.is_available")

    fake_modules = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=unexpected_cuda_probe)),
        "transformers": SimpleNamespace(
            AutoConfig=auto_config,
            AutoProcessor=auto_processor,
            AutoTokenizer=object(),
            AutoModelForCausalLM=auto_model,
        ),
    }
    monkeypatch.setattr(
        transformers_backend.importlib,
        "import_module",
        lambda name: fake_modules[name],
    )
    backend = TransformersModelBackend(device=DevicePolicy.CPU)

    loaded = backend._load_sync()

    assert loaded.device is DevicePolicy.CPU
    assert calls["AutoProcessor"][0] == "google/gemma-4-E4B-it"
    assert calls["AutoModelForCausalLM"][1]["trust_remote_code"] is False
    assert calls["AutoModelForCausalLM"][1]["dtype"] == "auto"
    assert calls["device"] == "cpu"
    assert calls["evaluated"] is True


def test_import_does_not_import_torch_or_transformers() -> None:
    code = """
import json
import sys
import oasis.llm.transformers_backend
print(json.dumps({name: name in sys.modules for name in (\"torch\", \"transformers\")}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"torch": False, "transformers": False}


class StubTensor:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)
        self.device: str | None = None

    def to(self, device: str) -> StubTensor:
        self.device = device
        return self


class StubProcessor:
    chat_template = "{{ messages }}"
    tokenizer: StubProcessor

    def __init__(self) -> None:
        self.tokenizer = self

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> dict[str, StubTensor]:
        return {"input_ids": StubTensor(3)}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        assert not add_special_tokens
        return text.split()


class StubStreamer:
    _END = object()

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.queue: Queue[str | object] = Queue()

    def __iter__(self) -> StubStreamer:
        return self

    def __next__(self) -> str:
        value = self.queue.get(timeout=1)
        if value is self._END:
            raise StopIteration
        assert isinstance(value, str)
        return value

    def push(self, text: str) -> None:
        self.queue.put(text)

    def end(self) -> None:
        self.queue.put(self._END)


class StubModel:
    def generate(self, **kwargs: Any) -> StubTensor:
        streamer: StubStreamer = kwargs["streamer"]
        streamer.push("raw ")
        streamer.push("reply")
        streamer.end()
        return StubTensor(5)


@pytest.mark.asyncio
async def test_transformers_backend_streams_and_accounts_with_loaded_stubs() -> None:
    backend = TransformersModelBackend(device=DevicePolicy.CPU)
    transformers = SimpleNamespace(
        TextIteratorStreamer=StubStreamer,
        StoppingCriteriaList=list,
    )
    backend._components = _LoadedComponents(
        torch=SimpleNamespace(),
        transformers=transformers,
        model=StubModel(),
        processor=StubProcessor(),
        device=DevicePolicy.CPU,
    )
    request = ModelRequest(
        request_id="stub-stream",
        messages=(ChatMessage(role="user", content="hello"),),
        max_generated_tokens=8,
    )

    deltas = [delta async for delta in backend.stream(request)]

    assert "".join(delta.text for delta in deltas) == "raw reply"
    assert deltas[-1].usage is not None
    assert deltas[-1].usage.input_tokens == 3
    assert deltas[-1].usage.generated_tokens == 2
