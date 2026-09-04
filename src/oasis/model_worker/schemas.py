"""Versioned OASIS model-worker wire contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from oasis.llm.schemas import FinishReason, ModelCapabilities, ModelDelta, ModelRequest, TokenUsage
from oasis.runtimes.schemas import ComputeInventory, RuntimePlan

MODEL_WORKER_SCHEMA_VERSION = "1.0.0"


class WorkerHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    status: Literal["ok"] = "ok"


class WorkerCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    model_profile: str
    model_id: str
    capabilities: ModelCapabilities
    runtime_plan: RuntimePlan
    inventory: ComputeInventory


class WorkerGenerateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    request: ModelRequest


class WorkerCountRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    request: ModelRequest


class WorkerCountResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    input_tokens: int


class WorkerAbortResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    request_id: str
    abort_requested: bool = True


class WorkerUsageResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    request_id: str
    found: bool
    usage: TokenUsage | None = None
    finish_reason: FinishReason | None = None


class WorkerError(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    code: str
    message: str


class WorkerStreamEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = MODEL_WORKER_SCHEMA_VERSION
    type: Literal["delta", "error"]
    delta: ModelDelta | None = None
    error: WorkerError | None = None
