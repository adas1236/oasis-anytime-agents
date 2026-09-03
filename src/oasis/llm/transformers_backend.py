"""Lazy Hugging Face Transformers backend for raw streaming text chat."""

from __future__ import annotations

import asyncio
import functools
import importlib
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from oasis.config import DevicePolicy, resolve_device
from oasis.errors import ModelBackendError, ModelErrorCode, ModelErrorDetail
from oasis.llm.adapters import (
    Gemma4ChatAdapter,
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


@dataclass
class _LoadedComponents:
    torch: Any
    transformers: Any
    model: Any
    processor: Any
    device: DevicePolicy


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


class TransformersModelBackend:
    """Load a causal language model only when the first generation starts."""

    def __init__(
        self,
        *,
        profile_name: str = "gemma4_e4b_it",
        model_id: str | None = None,
        revision: str | None = None,
        device: DevicePolicy = DevicePolicy.CPU,
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self._profile = resolve_model_profile(profile_name, model_id)
        self._revision = revision
        self._device_policy = device
        self._dtype = dtype
        self._trust_remote_code = trust_remote_code
        self._adapter = (
            Gemma4ChatAdapter(self._profile.model_id, self._profile.context_limit)
            if self._profile.family == "gemma4"
            else PlainChatAdapter(self._profile.model_id, self._profile.context_limit)
        )
        self._components: _LoadedComponents | None = None
        self._load_lock = asyncio.Lock()
        self._abort_events: dict[str, threading.Event] = {}
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

        selected_device = resolve_device(self._device_policy, torch.cuda.is_available)
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
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.profile.model_id,
                **model_kwargs,
            )
            model.to(selected_device.value)
            model.eval()
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
        return getattr(torch, self._dtype)

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

    @staticmethod
    def _move_inputs(inputs: Mapping[str, Any], device: DevicePolicy) -> dict[str, Any]:
        return {
            key: value.to(device.value) if hasattr(value, "to") else value
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

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
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
            parsed = parser.finish()
            thought_text += parsed.thought
            if parsed.text or parsed.thought or parsed.tool_calls:
                yield ModelDelta(
                    text=parsed.text,
                    thought=parsed.thought,
                    tool_calls=parsed.tool_calls,
                )
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
                error = result["error"]
                raise ModelBackendError(
                    ModelErrorDetail(
                        code=ModelErrorCode.GENERATION_FAILED,
                        message=f"Generation failed for model {self.profile.model_id!r}.",
                        model_id=self.profile.model_id,
                        context={"reason": str(error)},
                    )
                ) from error
            output: Any = result.get("output")
            output_length = int(output.shape[-1]) if hasattr(output, "shape") else input_tokens
            generated_tokens = max(0, output_length - input_tokens)
            reason = FinishReason.CANCELLED if abort_event.is_set() else FinishReason.STOP
            if parsed.tool_calls and not abort_event.is_set():
                reason = FinishReason.TOOL_CALL
            if not abort_event.is_set() and generated_tokens >= request.max_generated_tokens:
                reason = FinishReason.LENGTH
            yield ModelDelta(
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    reasoning_tokens=self._count_text_tokens(components.processor, thought_text),
                ),
                finish_reason=reason,
            )
        finally:
            abort_event.set()
            self._abort_events.pop(request.request_id, None)

    async def generate(self, request: ModelRequest) -> ModelTurn:
        return await collect_turn(self, request)

    async def abort(self, request_id: str) -> None:
        event = self._abort_events.get(request_id)
        if event is not None:
            event.set()

    async def close(self) -> None:
        self._closed = True
        for event in self._abort_events.values():
            event.set()
        self._components = None
