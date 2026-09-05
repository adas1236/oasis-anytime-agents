"""Lazy Hugging Face Transformers backend for raw streaming text chat."""

from __future__ import annotations

import asyncio
import functools
import importlib
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oasis.config import DevicePolicy, RuntimeConfig, RuntimeEngine
from oasis.errors import ModelBackendError, ModelErrorCode, ModelErrorDetail, ToolCallParseError
from oasis.llm.adapters import (
    Gemma4ChatAdapter,
    ParsedModelChunk,
    PlainChatAdapter,
    StreamParser,
    has_chat_template,
)
from oasis.llm.profiles import resolve_model_profile
from oasis.llm.protocols import collect_turn
from oasis.llm.schemas import (
    FinishReason,
    ModelCapabilities,
    ModelDelta,
    ModelProfile,
    ModelRequest,
    ModelTurn,
    TokenUsage,
)
from oasis.runtimes import (
    ComputeInventory,
    ConservativeRuntimePlanner,
    HardwareValidationStatus,
    RuntimeKind,
    RuntimeMetrics,
    RuntimePlan,
    RuntimePlanner,
    safe_cpu_inventory,
)


async def _run_in_daemon_thread[ResultT](operation: Callable[[], ResultT]) -> ResultT:
    """Run blocking setup without retaining asyncio's process-wide default executor."""

    loop = asyncio.get_running_loop()
    future: asyncio.Future[ResultT] = loop.create_future()

    def set_result(result: ResultT) -> None:
        if not future.done():
            future.set_result(result)

    def set_exception(error: BaseException) -> None:
        if not future.done():
            future.set_exception(error)

    def run() -> None:
        try:
            result = operation()
        except BaseException as error:
            loop.call_soon_threadsafe(set_exception, error)
        else:
            loop.call_soon_threadsafe(set_result, result)

    threading.Thread(target=run, name="oasis-blocking-operation", daemon=True).start()
    return await future


def _join_threads(workers: tuple[threading.Thread, ...]) -> None:
    for worker in workers:
        worker.join()


@dataclass
class _LoadedComponents:
    torch: Any
    transformers: Any
    model: Any
    processor: Any
    device: DevicePolicy | str


class _AbortCriteria:
    """Duck-typed stopping criterion for cancellation and Gemma's tool handoff token."""

    def __init__(
        self, event: threading.Event, stop_token_ids: frozenset[int] = frozenset()
    ) -> None:
        self._event = event
        self._stop_token_ids = stop_token_ids

    def __call__(self, input_ids: Any, *_args: Any, **_kwargs: Any) -> bool:
        if self._event.is_set():
            return True
        if not self._stop_token_ids:
            return False
        try:
            last_token = int(input_ids[0, -1])
        except (IndexError, TypeError, ValueError):
            return False
        return last_token in self._stop_token_ids


class TransformersInferenceRuntime:
    """Transformers execution adapter for one already resolved placement plan."""

    def __init__(
        self,
        *,
        profile_name: str = "gemma4_e4b_it",
        model_id: str | None = None,
        revision: str | None = None,
        device: DevicePolicy = DevicePolicy.CPU,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        inventory: ComputeInventory | None = None,
        plan: RuntimePlan | None = None,
    ) -> None:
        self._profile = resolve_model_profile(profile_name, model_id)
        self._revision = revision
        self._device_policy = device
        self._dtype = dtype
        self._trust_remote_code = trust_remote_code
        self._inventory = inventory or safe_cpu_inventory()
        self._plan = plan
        self._metrics = RuntimeMetrics()
        self._adapter = (
            Gemma4ChatAdapter(self._profile.model_id, self._profile.context_limit)
            if self._profile.family == "gemma4"
            else PlainChatAdapter(self._profile.model_id, self._profile.context_limit)
        )
        self._components: _LoadedComponents | None = None
        self._load_lock = asyncio.Lock()
        self._abort_events: dict[str, threading.Event] = {}
        self._generation_workers: dict[str, threading.Thread] = {}
        self._closed = False

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._adapter.capabilities

    @property
    def is_loaded(self) -> bool:
        """Whether weights and processor have been materialized."""

        return self._components is not None

    @property
    def plan(self) -> RuntimePlan | None:
        """Return the resolved plan plus current measurements."""

        if self._plan is None:
            return None
        return self._plan.model_copy(update={"metrics": self._metrics})

    @property
    def inventory(self) -> ComputeInventory:
        return self._inventory

    def _load_sync(self) -> _LoadedComponents:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise ModelBackendError(
                ModelErrorDetail(
                    code=ModelErrorCode.MODEL_LOAD_FAILED,
                    message="The Transformers backend dependencies are not installed.",
                    model_id=self.profile.model_id,
                )
            ) from error

        selected_device: DevicePolicy | str = self._device_policy
        if self._plan is not None:
            placement = self._plan.device_placement[0]
            selected_device = DevicePolicy.CPU if placement == "cpu" else placement
        common_kwargs: dict[str, Any] = {
            "revision": self._revision,
            "trust_remote_code": self._trust_remote_code,
        }
        common_kwargs = {key: value for key, value in common_kwargs.items() if value is not None}

        try:
            config = transformers.AutoConfig.from_pretrained(self.profile.model_id, **common_kwargs)
            if not getattr(config, "is_decoder", False) and getattr(
                config, "model_type", None
            ) not in {"gemma4", "gemma4_text"}:
                architectures = " ".join(getattr(config, "architectures", ()) or ())
                if not any(term in architectures for term in ("CausalLM", "ConditionalGeneration")):
                    raise ModelBackendError(
                        ModelErrorDetail(
                            code=ModelErrorCode.INVALID_MODEL,
                            message=(
                                f"Model {self.profile.model_id!r} is not a supported generative "
                                "causal architecture."
                            ),
                            model_id=self.profile.model_id,
                        )
                    )

            if self.profile.family == "gemma4":
                processor = transformers.AutoProcessor.from_pretrained(
                    self.profile.model_id,
                    padding_side="left",
                    **common_kwargs,
                )
            else:
                processor = transformers.AutoTokenizer.from_pretrained(
                    self.profile.model_id,
                    padding_side="left",
                    **common_kwargs,
                )

            if not has_chat_template(processor):
                raise ModelBackendError(
                    ModelErrorDetail(
                        code=ModelErrorCode.MISSING_CHAT_TEMPLATE,
                        message=(
                            f"Model {self.profile.model_id!r} does not provide a chat template."
                        ),
                        model_id=self.profile.model_id,
                        capability="chat_template",
                    )
                )

            model_kwargs = dict(common_kwargs)
            model_kwargs["dtype"] = self._resolve_dtype(torch)
            if self._plan is not None:
                model_kwargs["attn_implementation"] = self._plan.attention_backend
                if self._plan.quantization == "int8":
                    model_kwargs["load_in_8bit"] = True
                elif self._plan.quantization == "int4":
                    model_kwargs["load_in_4bit"] = True
                if self._plan.runtime is RuntimeKind.ACCELERATE_DISPATCH:
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["max_memory"] = {
                        (int(key.split(":", 1)[1]) if key.startswith("cuda:") else key): value
                        for key, value in self._plan.memory_limits.items()
                    }
                    if self._plan.offload_directory is not None:
                        Path(self._plan.offload_directory).mkdir(parents=True, exist_ok=True)
                        model_kwargs["offload_folder"] = self._plan.offload_directory
                        model_kwargs["offload_state_dict"] = True
            # Gemma 4 checkpoints are multimodal conditional-generation models.
            # Loading them through AutoModelForCausalLM fails before weights are
            # downloaded because Gemma4Config is not registered for that auto
            # class. Text-only/custom profiles continue to use the causal-LM
            # factory.
            model_factory = (
                transformers.AutoModelForMultimodalLM
                if self.profile.family == "gemma4"
                else transformers.AutoModelForCausalLM
            )
            model = model_factory.from_pretrained(
                self.profile.model_id,
                **model_kwargs,
            )
            if self._plan is None or self._plan.runtime is not RuntimeKind.ACCELERATE_DISPATCH:
                target = (
                    selected_device.value
                    if isinstance(selected_device, DevicePolicy)
                    else selected_device
                )
                model.to(target)
            model.eval()
            actual_dtype = str(getattr(model, "dtype", "")).removeprefix("torch.")
            if (
                self._plan is not None
                and self._plan.dtype == "auto"
                and actual_dtype in {"float16", "bfloat16", "float32"}
            ):
                self._plan = self._plan.model_copy(update={"dtype": actual_dtype})
        except ModelBackendError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ModelBackendError(
                ModelErrorDetail(
                    code=ModelErrorCode.MODEL_LOAD_FAILED,
                    message=f"Could not load model {self.profile.model_id!r}.",
                    model_id=self.profile.model_id,
                    context={
                        "reason": str(error),
                        "trust_remote_code": self._trust_remote_code,
                    },
                )
            ) from error
        return _LoadedComponents(torch, transformers, model, processor, selected_device)

    def _resolve_dtype(self, torch: Any) -> Any:
        if self._dtype == "auto":
            return "auto"
        allowed = {"float16", "bfloat16", "float32"}
        if self._dtype not in allowed:
            valid_values = ", ".join(sorted(allowed))
            raise ModelBackendError(
                ModelErrorDetail(
                    code=ModelErrorCode.INVALID_MODEL,
                    message=f"Unsupported dtype {self._dtype!r}; choose {valid_values} or auto.",
                    model_id=self.profile.model_id,
                )
            )
        return getattr(torch, self._dtype, self._dtype)

    async def _ensure_loaded(self) -> _LoadedComponents:
        if self._closed:
            raise RuntimeError("Transformers backend is closed")
        if self._components is None:
            async with self._load_lock:
                if self._components is None:
                    self._components = await _run_in_daemon_thread(
                        functools.partial(self._load_sync)
                    )
        return self._components

    async def load(
        self,
        model: ModelProfile | None = None,
        plan: RuntimePlan | None = None,
    ) -> None:
        """Materialize the processor and weights once under the backend's load lock."""

        if model is not None and model.model_id != self.profile.model_id:
            raise ValueError("runtime model must match its configured backend profile")
        if plan is not None:
            if plan.requested_model_id != self.profile.model_id:
                raise ValueError("runtime plan must preserve the configured model ID")
            if self._components is not None and plan != self._plan:
                raise RuntimeError("a loaded runtime cannot change placement plans")
            self._plan = plan
            self._dtype = plan.dtype
        started = asyncio.get_running_loop().time()
        await self._ensure_loaded()
        startup_ms = max(0, round((asyncio.get_running_loop().time() - started) * 1_000))
        self._metrics = self._metrics.model_copy(update={"startup_ms": startup_ms})

    @staticmethod
    def _move_inputs(inputs: Mapping[str, Any], device: DevicePolicy | str) -> dict[str, Any]:
        target = device.value if isinstance(device, DevicePolicy) else device
        if target == "auto":
            return dict(inputs)
        return {
            key: value.to(target) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _input_token_count(inputs: Mapping[str, Any]) -> int:
        input_ids = inputs.get("input_ids")
        if input_ids is None or not hasattr(input_ids, "shape"):
            raise ModelBackendError(
                ModelErrorDetail(
                    code=ModelErrorCode.INVALID_MODEL,
                    message="The processor did not return input_ids for token accounting.",
                )
            )
        return int(input_ids.shape[-1])

    @staticmethod
    def _count_text_tokens(processor: Any, text: str) -> int:
        if not text:
            return 0
        tokenizer = getattr(processor, "tokenizer", processor)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)

    def _tool_stop_token_ids(self, processor: Any, request: ModelRequest) -> frozenset[int]:
        if not request.tools or self.profile.family != "gemma4":
            return frozenset()
        tokenizer = getattr(processor, "tokenizer", processor)
        converter = getattr(tokenizer, "convert_tokens_to_ids", None)
        if converter is None:
            return frozenset()
        token_id = converter("<|tool_response>")
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if not isinstance(token_id, int) or token_id < 0 or token_id == unknown_id:
            return frozenset()
        return frozenset({token_id})

    async def count_input_tokens(self, request: ModelRequest) -> int:
        """Render and tokenize the exact request without starting generation."""

        components = await self._ensure_loaded()
        prepared = await _run_in_daemon_thread(
            functools.partial(
                self._adapter.prepare_inputs,
                components.processor,
                request.messages,
                tools=request.tools,
                thinking_enabled=request.thinking_enabled,
            )
        )
        return self._input_token_count(prepared)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        started = asyncio.get_running_loop().time()
        components = await self._ensure_loaded()
        prepared = await _run_in_daemon_thread(
            functools.partial(
                self._adapter.prepare_inputs,
                components.processor,
                request.messages,
                tools=request.tools,
                thinking_enabled=request.thinking_enabled,
            )
        )
        inputs = self._move_inputs(prepared, components.device)
        input_tokens = self._input_token_count(inputs)
        preserve_special = self._adapter.preserve_special_tokens(
            thinking_enabled=request.thinking_enabled,
            tools_enabled=bool(request.tools),
        )
        streamer = components.transformers.TextIteratorStreamer(
            components.processor,
            skip_prompt=True,
            skip_special_tokens=not preserve_special,
            clean_up_tokenization_spaces=False,
        )
        abort_event = threading.Event()
        self._abort_events[request.request_id] = abort_event
        result: dict[str, Any] = {}
        parser: StreamParser = self._adapter.stream_parser(
            thinking_enabled=request.thinking_enabled,
            tools_enabled=bool(request.tools),
        )
        thought_text = ""

        def generate() -> None:
            try:
                stopping = components.transformers.StoppingCriteriaList(
                    [
                        _AbortCriteria(
                            abort_event,
                            self._tool_stop_token_ids(components.processor, request),
                        )
                    ]
                )
                result["output"] = components.model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=request.max_generated_tokens,
                    do_sample=False,
                    stopping_criteria=stopping,
                )
            except BaseException as error:
                result["error"] = error
                streamer.end()

        worker = threading.Thread(target=generate, name=f"oasis-{request.request_id}", daemon=True)
        self._generation_workers[request.request_id] = worker
        worker.start()
        loop = asyncio.get_running_loop()
        stream_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def read_stream() -> None:
            try:
                for raw_chunk in streamer:
                    loop.call_soon_threadsafe(stream_queue.put_nowait, ("chunk", raw_chunk))
            except BaseException as error:
                loop.call_soon_threadsafe(stream_queue.put_nowait, ("error", error))
            finally:
                loop.call_soon_threadsafe(stream_queue.put_nowait, ("end", None))

        reader = threading.Thread(
            target=read_stream,
            name=f"oasis-stream-{request.request_id}",
            daemon=True,
        )
        reader.start()
        try:
            while True:
                item_kind, item = await stream_queue.get()
                if item_kind == "end":
                    break
                if item_kind == "error":
                    raise ModelBackendError(
                        ModelErrorDetail(
                            code=ModelErrorCode.GENERATION_FAILED,
                            message=f"Streaming failed for model {self.profile.model_id!r}.",
                            model_id=self.profile.model_id,
                            context={"reason": str(item)},
                        )
                    ) from item
                raw_chunk = str(item)
                if not raw_chunk:
                    continue
                parsed = parser.feed(raw_chunk)
                thought_text += parsed.thought
                if parsed.text or parsed.thought or parsed.tool_calls:
                    yield ModelDelta(
                        text=parsed.text,
                        thought=parsed.thought,
                        tool_calls=parsed.tool_calls,
                    )
            reader.join(timeout=1.0)
            parse_error: ToolCallParseError | None = None
            try:
                parsed = parser.finish()
            except ToolCallParseError as error:
                parse_error = error
                parsed = ParsedModelChunk()
            worker.join(timeout=1.0)
            if worker.is_alive():
                raise ModelBackendError(
                    ModelErrorDetail(
                        code=ModelErrorCode.GENERATION_FAILED,
                        message=(
                            f"Generation worker did not stop for model {self.profile.model_id!r}."
                        ),
                        model_id=self.profile.model_id,
                    )
                )
            if "error" in result:
                generation_error = result["error"]
                raise ModelBackendError(
                    ModelErrorDetail(
                        code=ModelErrorCode.GENERATION_FAILED,
                        message=f"Generation failed for model {self.profile.model_id!r}.",
                        model_id=self.profile.model_id,
                        context={"reason": str(generation_error)},
                    )
                ) from generation_error
            output: Any = result.get("output")
            output_length = int(output.shape[-1]) if hasattr(output, "shape") else input_tokens
            generated_tokens = max(0, output_length - input_tokens)
            reason = FinishReason.CANCELLED if abort_event.is_set() else FinishReason.STOP
            if parsed.tool_calls and not abort_event.is_set():
                reason = FinishReason.TOOL_CALL
            if not abort_event.is_set() and generated_tokens >= request.max_generated_tokens:
                reason = FinishReason.LENGTH
            elapsed_ms = max(0, round((asyncio.get_running_loop().time() - started) * 1_000))
            peak_memory: int | None = self._metrics.peak_device_memory_bytes
            target = (
                components.device.value
                if isinstance(components.device, DevicePolicy)
                else components.device
            )
            if target.startswith("cuda") and hasattr(components.torch.cuda, "max_memory_allocated"):
                peak_memory = int(components.torch.cuda.max_memory_allocated())
            self._metrics = self._metrics.model_copy(
                update={
                    "request_count": self._metrics.request_count + 1,
                    "generated_tokens": self._metrics.generated_tokens + generated_tokens,
                    "generation_ms": self._metrics.generation_ms + elapsed_ms,
                    "peak_device_memory_bytes": peak_memory,
                }
            )
            if self._plan is not None and self._plan.runtime in {
                RuntimeKind.CUDA_TRANSFORMERS,
                RuntimeKind.ACCELERATE_DISPATCH,
            }:
                self._plan = self._plan.model_copy(
                    update={"hardware_validation": HardwareValidationStatus.PASSED}
                )
            thought_text += parsed.thought
            usage = TokenUsage(
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
                reasoning_tokens=self._count_text_tokens(components.processor, thought_text),
            )
            if parse_error is not None:
                parse_error.detail = parse_error.detail.model_copy(
                    update={
                        "context": {
                            **parse_error.detail.context,
                            "token_usage": {
                                "input_tokens": usage.input_tokens,
                                "generated_tokens": usage.generated_tokens,
                                "reasoning_tokens": usage.reasoning_tokens,
                            },
                        }
                    }
                )
                raise parse_error
            if parsed.text or parsed.thought or parsed.tool_calls:
                yield ModelDelta(
                    text=parsed.text,
                    thought=parsed.thought,
                    tool_calls=parsed.tool_calls,
                )
            yield ModelDelta(
                usage=usage,
                finish_reason=reason,
            )
        finally:
            abort_event.set()
            self._abort_events.pop(request.request_id, None)
            if not worker.is_alive():
                self._generation_workers.pop(request.request_id, None)

    def generate(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        """Expose the runtime protocol name while the backend retains ``stream``."""

        return self.stream(request)

    async def abort(self, request_id: str) -> None:
        event = self._abort_events.get(request_id)
        if event is not None:
            event.set()

    async def close(self) -> None:
        self._closed = True
        for event in self._abort_events.values():
            event.set()
        workers = tuple(self._generation_workers.values())
        if workers:
            await _run_in_daemon_thread(functools.partial(_join_threads, workers))
        self._generation_workers.clear()
        self._components = None


class CpuTransformersRuntime(TransformersInferenceRuntime):
    """Explicit CPU Transformers runtime."""


class CudaTransformersRuntime(TransformersInferenceRuntime):
    """Explicit single-visible-GPU Transformers runtime."""


class AccelerateDispatchRuntime(TransformersInferenceRuntime):
    """Memory-oriented device-map dispatch; it is not parallel-serving acceleration."""


class TransformersModelBackend:
    """Public chat backend delegating placement and execution to an inference runtime."""

    def __init__(
        self,
        *,
        profile_name: str = "gemma4_e4b_it",
        model_id: str | None = None,
        revision: str | None = None,
        device: DevicePolicy = DevicePolicy.CPU,
        engine: RuntimeEngine = RuntimeEngine.AUTO,
        dtype: str = "auto",
        quantization: str | None = None,
        attention_backend: str = "auto",
        memory_headroom_fraction: float = 0.10,
        allow_cpu_offload: bool = False,
        allow_disk_offload: bool = False,
        offload_directory: Path = Path(".oasis/offload"),
        model_memory_bytes: int | None = None,
        trust_remote_code: bool = False,
        inventory: ComputeInventory | None = None,
        planner: RuntimePlanner | None = None,
        runtime: TransformersInferenceRuntime | None = None,
    ) -> None:
        self._profile = resolve_model_profile(profile_name, model_id)
        self._revision = revision
        self._inventory = inventory or safe_cpu_inventory()
        self._policy = RuntimeConfig(
            device=device,
            engine=engine,
            dtype=dtype,
            quantization=quantization,
            attention_backend=attention_backend,
            memory_headroom_fraction=memory_headroom_fraction,
            allow_cpu_offload=allow_cpu_offload,
            allow_disk_offload=allow_disk_offload,
            offload_directory=offload_directory,
            model_memory_bytes=model_memory_bytes,
        )
        self._planner = planner or ConservativeRuntimePlanner()
        self._resolved_plan = self._planner.plan(
            self._profile,
            self._inventory,
            self._policy,
            revision=revision,
        )
        runtime_type = {
            RuntimeKind.CPU_TRANSFORMERS: CpuTransformersRuntime,
            RuntimeKind.CUDA_TRANSFORMERS: CudaTransformersRuntime,
            RuntimeKind.ACCELERATE_DISPATCH: AccelerateDispatchRuntime,
        }.get(self._resolved_plan.runtime)
        if runtime is None and runtime_type is None:
            raise ValueError(
                f"TransformersModelBackend cannot execute {self._resolved_plan.runtime.value}"
            )
        if runtime is not None:
            self._runtime = runtime
        else:
            assert runtime_type is not None
            self._runtime = runtime_type(
                profile_name=profile_name,
                model_id=model_id,
                revision=revision,
                device=device,
                dtype=self._resolved_plan.dtype,
                trust_remote_code=trust_remote_code,
                inventory=self._inventory,
                plan=self._resolved_plan,
            )

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._runtime.capabilities

    @property
    def is_loaded(self) -> bool:
        return self._runtime.is_loaded

    @property
    def runtime_plan(self) -> RuntimePlan:
        return self._runtime.plan or self._resolved_plan

    @property
    def compute_inventory(self) -> ComputeInventory:
        return self._inventory

    @property
    def _components(self) -> _LoadedComponents | None:
        """Compatibility access for deterministic low-level backend tests."""

        return self._runtime._components

    @_components.setter
    def _components(self, value: _LoadedComponents | None) -> None:
        self._runtime._components = value

    def _load_sync(self) -> _LoadedComponents:
        """Compatibility hook delegated to the concrete runtime."""

        return self._runtime._load_sync()

    async def load(self) -> None:
        await self._runtime.load(self._profile, self._resolved_plan)

    async def count_input_tokens(self, request: ModelRequest) -> int:
        return await self._runtime.count_input_tokens(request)

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        return self._runtime.stream(request)

    async def generate(self, request: ModelRequest) -> ModelTurn:
        return await collect_turn(self, request)

    async def abort(self, request_id: str) -> None:
        await self._runtime.abort(request_id)

    async def close(self) -> None:
        await self._runtime.close()
