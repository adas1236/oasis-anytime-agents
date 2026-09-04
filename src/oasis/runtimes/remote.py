"""Authenticated remote OASIS model-worker client runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from pydantic import ValidationError

from oasis.llm.schemas import ModelCapabilities, ModelDelta, ModelProfile, ModelRequest
from oasis.model_worker.schemas import (
    MODEL_WORKER_SCHEMA_VERSION,
    WorkerAbortResponse,
    WorkerCapabilities,
    WorkerCountRequest,
    WorkerCountResponse,
    WorkerHealth,
    WorkerStreamEnvelope,
    WorkerUsageResponse,
)
from oasis.runtimes.inventory import fake_inventory
from oasis.runtimes.schemas import ComputeInventory, RuntimePlan


class RemoteRuntimeError(RuntimeError):
    """Safe remote transport, authentication, protocol, or generation failure."""


class RemoteModelRuntime:
    """Stream generation through the versioned OASIS model-worker protocol."""

    def __init__(
        self,
        endpoint: str,
        *,
        auth_token: str,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
        inventory: ComputeInventory | None = None,
    ) -> None:
        if not auth_token:
            raise ValueError("remote model worker token must not be empty")
        self._endpoint = endpoint.rstrip("/")
        self._headers = {"Authorization": f"Bearer {auth_token}"}
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._plan: RuntimePlan | None = None
        self._inventory = inventory or fake_inventory()
        self._capabilities: ModelCapabilities | None = None

    @property
    def plan(self) -> RuntimePlan | None:
        return self._plan

    @property
    def inventory(self) -> ComputeInventory:
        return self._inventory

    @property
    def capabilities(self) -> ModelCapabilities | None:
        return self._capabilities

    def _url(self, path: str) -> str:
        return f"{self._endpoint}/api/v1/{path.lstrip('/')}"

    @staticmethod
    def _check_version(version: str) -> None:
        if version != MODEL_WORKER_SCHEMA_VERSION:
            raise RemoteRuntimeError(
                f"model-worker protocol version mismatch: expected {MODEL_WORKER_SCHEMA_VERSION}"
            )

    @staticmethod
    def _raise_status(response: httpx.Response) -> None:
        if response.status_code == 401:
            raise RemoteRuntimeError("model-worker authentication failed")
        if response.is_error:
            raise RemoteRuntimeError(f"model-worker returned HTTP {response.status_code}")

    async def health(self) -> WorkerHealth:
        try:
            response = await self._client.get(self._url("health"), headers=self._headers)
            self._raise_status(response)
            health = WorkerHealth.model_validate(response.json())
            self._check_version(health.schema_version)
            return health
        except (httpx.RequestError, ValueError, ValidationError) as error:
            if isinstance(error, RemoteRuntimeError):
                raise
            raise RemoteRuntimeError("could not read model-worker health") from error

    async def capability_report(self) -> WorkerCapabilities:
        try:
            response = await self._client.get(self._url("capabilities"), headers=self._headers)
            self._raise_status(response)
            report = WorkerCapabilities.model_validate(response.json())
            self._check_version(report.schema_version)
            return report
        except (httpx.RequestError, ValueError, ValidationError) as error:
            if isinstance(error, RemoteRuntimeError):
                raise
            raise RemoteRuntimeError("could not read model-worker capabilities") from error

    async def load(self, model: ModelProfile, plan: RuntimePlan) -> None:
        await self.health()
        report = await self.capability_report()
        worker_plan = report.runtime_plan
        if (
            report.model_id != model.model_id
            or report.model_profile != model.name
            or plan.requested_model_id != model.model_id
            or worker_plan.requested_model_id != model.model_id
        ):
            raise RemoteRuntimeError("model-worker serves a different model than requested")
        if worker_plan.model_revision != plan.model_revision:
            raise RemoteRuntimeError(
                "model-worker serves a different model revision than requested"
            )
        if plan.dtype != "auto" and worker_plan.dtype != plan.dtype:
            raise RemoteRuntimeError("model-worker cannot honor the requested dtype")
        if worker_plan.quantization != plan.quantization:
            raise RemoteRuntimeError("model-worker cannot honor the requested quantization")
        if (
            plan.attention_backend != "auto"
            and worker_plan.attention_backend != plan.attention_backend
        ):
            raise RemoteRuntimeError("model-worker cannot honor the requested attention backend")
        if (
            worker_plan.allow_cpu_offload != plan.allow_cpu_offload
            or worker_plan.allow_disk_offload != plan.allow_disk_offload
        ):
            raise RemoteRuntimeError("model-worker cannot honor the requested offload policy")
        self._capabilities = report.capabilities
        self._inventory = report.inventory
        self._plan = plan.model_copy(
            update={
                "dtype": worker_plan.dtype,
                "quantization": worker_plan.quantization,
                "attention_backend": worker_plan.attention_backend,
                "allow_cpu_offload": worker_plan.allow_cpu_offload,
                "allow_disk_offload": worker_plan.allow_disk_offload,
                "offload_directory": worker_plan.offload_directory,
                "model_memory_estimate_bytes": worker_plan.model_memory_estimate_bytes,
                "reserved_headroom_bytes": worker_plan.reserved_headroom_bytes,
                "device_placement": worker_plan.device_placement,
                "device_map": worker_plan.device_map,
                "memory_limits": worker_plan.memory_limits,
                "warnings": worker_plan.warnings,
                "hardware_validation": worker_plan.hardware_validation,
                "metrics": worker_plan.metrics,
                "rationale": (*plan.rationale, "Worker capability report verified at load."),
            }
        )

    async def count_input_tokens(self, request: ModelRequest) -> int:
        try:
            response = await self._client.post(
                self._url("tokens/count"),
                headers=self._headers,
                json=WorkerCountRequest(request=request).model_dump(mode="json"),
            )
            self._raise_status(response)
            result = WorkerCountResponse.model_validate(response.json())
            self._check_version(result.schema_version)
            return result.input_tokens
        except (httpx.RequestError, ValueError, ValidationError) as error:
            if isinstance(error, RemoteRuntimeError):
                raise
            raise RemoteRuntimeError("remote token counting failed") from error

    async def generate(self, request: ModelRequest) -> AsyncIterator[ModelDelta]:
        terminal = False
        try:
            async with self._client.stream(
                "POST",
                self._url("generate"),
                headers=self._headers,
                json={
                    "schema_version": MODEL_WORKER_SCHEMA_VERSION,
                    "request": request.model_dump(mode="json"),
                },
            ) as response:
                self._raise_status(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    envelope = WorkerStreamEnvelope.model_validate_json(line)
                    self._check_version(envelope.schema_version)
                    if envelope.type == "error" or envelope.delta is None:
                        message = (
                            envelope.error.message if envelope.error is not None else "unknown"
                        )
                        raise RemoteRuntimeError(f"remote generation failed: {message}")
                    delta = envelope.delta
                    terminal = terminal or (
                        delta.usage is not None and delta.finish_reason is not None
                    )
                    yield delta
        except (httpx.RequestError, httpx.StreamError, ValidationError) as error:
            raise RemoteRuntimeError(
                "remote generation disconnected or returned invalid data"
            ) from error
        if not terminal:
            raise RemoteRuntimeError("remote generation stream ended without terminal usage")

    async def abort(self, request_id: str) -> None:
        try:
            response = await self._client.post(
                self._url(f"abort/{request_id}"), headers=self._headers
            )
            self._raise_status(response)
            result = WorkerAbortResponse.model_validate(response.json())
            self._check_version(result.schema_version)
            if result.request_id != request_id:
                raise RemoteRuntimeError("model-worker acknowledged a different abort request")
        except (httpx.RequestError, ValueError, ValidationError) as error:
            if isinstance(error, RemoteRuntimeError):
                raise
            raise RemoteRuntimeError("remote abort failed") from error

    async def usage(self, request_id: str) -> WorkerUsageResponse:
        try:
            response = await self._client.get(
                self._url(f"usage/{request_id}"), headers=self._headers
            )
            self._raise_status(response)
            result = WorkerUsageResponse.model_validate(response.json())
            self._check_version(result.schema_version)
            return result
        except (httpx.RequestError, ValueError, ValidationError) as error:
            if isinstance(error, RemoteRuntimeError):
                raise
            raise RemoteRuntimeError("remote usage lookup failed") from error

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
