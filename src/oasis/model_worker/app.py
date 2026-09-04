"""Minimal authenticated streaming OASIS model-worker service."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from oasis.llm import FinishReason, ModelBackend, TokenUsage
from oasis.model_worker.schemas import (
    MODEL_WORKER_SCHEMA_VERSION,
    WorkerAbortResponse,
    WorkerCapabilities,
    WorkerCountRequest,
    WorkerCountResponse,
    WorkerError,
    WorkerGenerateRequest,
    WorkerHealth,
    WorkerStreamEnvelope,
    WorkerUsageResponse,
)
from oasis.runtimes.schemas import ComputeInventory, RuntimePlan


def create_model_worker_app(
    backend: ModelBackend,
    *,
    auth_token: str,
    runtime_plan: RuntimePlan | None = None,
    inventory: ComputeInventory | None = None,
) -> FastAPI:
    """Create a worker whose only mutable state is active generation and usage summaries."""

    if not auth_token:
        raise ValueError("model worker authentication token must not be empty")
    resolved_plan = runtime_plan or getattr(backend, "runtime_plan", None)
    resolved_inventory = inventory or getattr(backend, "compute_inventory", None)
    if not isinstance(resolved_plan, RuntimePlan) or not isinstance(
        resolved_inventory, ComputeInventory
    ):
        raise ValueError("model worker requires typed runtime plan and inventory metadata")
    usage: dict[str, tuple[TokenUsage, FinishReason]] = {}
    app = FastAPI(title="OASIS Model Worker", version=MODEL_WORKER_SCHEMA_VERSION)

    def authorized(authorization: str | None) -> bool:
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            return False
        return hmac.compare_digest(authorization[len(prefix) :], auth_token)

    def unauthorized() -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content=WorkerError(
                code="authentication_failed", message="Authentication failed."
            ).model_dump(mode="json"),
            headers={"WWW-Authenticate": "Bearer"},
        )

    def incompatible_version() -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=WorkerError(
                code="version_mismatch",
                message=f"This worker requires schema version {MODEL_WORKER_SCHEMA_VERSION}.",
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=WorkerError(
                code="internal_error",
                message="The model worker could not complete the request.",
            ).model_dump(mode="json"),
        )

    @app.get("/api/v1/health", response_model=WorkerHealth)
    async def health(
        authorization: str | None = Header(default=None),
    ) -> WorkerHealth | JSONResponse:
        if not authorized(authorization):
            return unauthorized()
        return WorkerHealth()

    @app.get("/api/v1/capabilities", response_model=WorkerCapabilities)
    async def capabilities(
        authorization: str | None = Header(default=None),
    ) -> WorkerCapabilities | JSONResponse:
        if not authorized(authorization):
            return unauthorized()
        current_plan = getattr(backend, "runtime_plan", resolved_plan)
        current_inventory = getattr(backend, "compute_inventory", resolved_inventory)
        return WorkerCapabilities(
            model_profile=backend.profile.name,
            model_id=backend.profile.model_id,
            capabilities=backend.capabilities,
            runtime_plan=current_plan,
            inventory=current_inventory.sanitized(),
        )

    @app.post("/api/v1/tokens/count", response_model=WorkerCountResponse)
    async def count_tokens(
        body: WorkerCountRequest,
        authorization: str | None = Header(default=None),
    ) -> WorkerCountResponse | JSONResponse:
        if not authorized(authorization):
            return unauthorized()
        if body.schema_version != MODEL_WORKER_SCHEMA_VERSION:
            return incompatible_version()
        return WorkerCountResponse(input_tokens=await backend.count_input_tokens(body.request))

    @app.post("/api/v1/generate", response_model=None)
    async def generate(
        body: WorkerGenerateRequest,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse | JSONResponse:
        if not authorized(authorization):
            return unauthorized()
        if body.schema_version != MODEL_WORKER_SCHEMA_VERSION:
            return incompatible_version()

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for delta in backend.stream(body.request):
                    if delta.usage is not None and delta.finish_reason is not None:
                        usage[body.request.request_id] = (delta.usage, delta.finish_reason)
                    envelope = WorkerStreamEnvelope(type="delta", delta=delta)
                    yield (envelope.model_dump_json() + "\n").encode()
            except Exception:
                envelope = WorkerStreamEnvelope(
                    type="error",
                    error=WorkerError(
                        code="generation_failed",
                        message="The worker could not complete generation.",
                    ),
                )
                yield (envelope.model_dump_json() + "\n").encode()

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/api/v1/abort/{request_id}", response_model=WorkerAbortResponse)
    async def abort(
        request_id: str,
        authorization: str | None = Header(default=None),
    ) -> WorkerAbortResponse | JSONResponse:
        if not authorized(authorization):
            return unauthorized()
        await backend.abort(request_id)
        return WorkerAbortResponse(request_id=request_id)

    @app.get("/api/v1/usage/{request_id}", response_model=WorkerUsageResponse)
    async def request_usage(
        request_id: str,
        authorization: str | None = Header(default=None),
    ) -> WorkerUsageResponse | JSONResponse:
        if not authorized(authorization):
            return unauthorized()
        record = usage.get(request_id)
        if record is None:
            return WorkerUsageResponse(request_id=request_id, found=False)
        token_usage, finish_reason = record
        return WorkerUsageResponse(
            request_id=request_id,
            found=True,
            usage=token_usage,
            finish_reason=finish_reason,
        )

    return app
