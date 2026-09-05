"""FastAPI application exposing the versioned OASIS HTTP and SSE boundary."""

from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHttpException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from oasis.api.examples import example_catalog
from oasis.api.lifecycle import ModelService
from oasis.api.manager import RunManager, RunManagerError
from oasis.api.schemas import (
    API_SCHEMA_VERSION,
    ApiErrorDetail,
    ApiErrorResponse,
    CancelRunResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MessageRequest,
    ModelCatalogEntry,
    ModelCatalogResponse,
    ProblemCatalogEntry,
    ProblemCatalogResponse,
    RunCreatedResponse,
    RunCreateRequest,
    RunInspectionResponse,
    RuntimeResponse,
    ToolCatalogEntry,
    ToolCatalogResponse,
)
from oasis.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactStore,
    LocalArtifactStore,
)
from oasis.config import OasisSettings
from oasis.controller import LocalRunStore, RunResult, RunStore, RunStoreError
from oasis.errors import ModelBackendError
from oasis.llm import DEFAULT_PROFILE_NAME, MODEL_PROFILES, ModelBackend
from oasis.problems import ProblemRegistry, create_builtin_problem_registry
from oasis.providers.service import ServiceProviders
from oasis.runtimes import ComputeInventory
from oasis.schemas import ArtifactKind, ArtifactRef, PrivacyClassification
from oasis.tools import ToolRegistry, create_tool_registry

_ARTIFACT_KINDS = frozenset(
    {
        ArtifactKind.VECTOR,
        ArtifactKind.RASTER,
        ArtifactKind.TABLE,
        ArtifactKind.GRAPH,
        ArtifactKind.MATRIX,
        ArtifactKind.JSON_SPECIFICATION,
        ArtifactKind.PLAN,
        ArtifactKind.SCORECARD,
        ArtifactKind.MAP,
    }
)
_ARTIFACT_MEDIA_TYPES = frozenset(
    {
        "application/geo+json",
        "application/json",
        "application/vnd.apache.parquet",
        "application/vnd.oasis.graph+json",
        "application/vnd.oasis.matrix+npz",
        "image/svg+xml",
        "image/tiff; application=geotiff",
    }
)


def _error(status_code: int, code: str, message: str, fields: tuple[str, ...] = ()) -> JSONResponse:
    payload = ApiErrorResponse(error=ApiErrorDetail(code=code, message=message, fields=fields))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


class RequestSizeLimitMiddleware:
    """Reject oversized fixed-length or chunked request bodies before route handling."""

    def __init__(self, app: ASGIApp, *, maximum_bytes: int) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > self.maximum_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return

        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.maximum_bytes:
                    await self._reject(send)
                    return
                if not message.get("more_body", False):
                    break

        iterator = iter(buffered)

        async def replay() -> Message:
            return next(iterator, {"type": "http.request", "body": b"", "more_body": False})

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = (
            ApiErrorResponse(
                error=ApiErrorDetail(
                    code="request_too_large",
                    message="The request body exceeds the configured service limit.",
                )
            )
            .model_dump_json()
            .encode()
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _sse_event(event_id: int, event_name: str, data: str) -> bytes:
    lines = data.splitlines() or [""]
    encoded = [f"id: {event_id}", f"event: {event_name}"]
    encoded.extend(f"data: {line}" for line in lines)
    return ("\n".join(encoded) + "\n\n").encode()


def create_app(
    settings: OasisSettings | None = None,
    *,
    backend: ModelBackend | None = None,
    artifact_store: ArtifactStore | None = None,
    run_store: RunStore | None = None,
    tool_registry: ToolRegistry | None = None,
    problem_registry: ProblemRegistry | None = None,
    compute_inventory: ComputeInventory | None = None,
    providers: Mapping[str, object] | None = None,
    resources: Mapping[str, object] | None = None,
) -> FastAPI:
    """Build a side-effect-bounded service with injectable stores and fake backends."""

    resolved = settings or OasisSettings()
    artifacts = artifact_store or LocalArtifactStore(resolved.artifact_root)
    runs = run_store or LocalRunStore(resolved.run_root)
    tools = tool_registry or create_tool_registry(discover_entry_points=False)
    problems = problem_registry or create_builtin_problem_registry()
    models = ModelService(resolved, backend=backend, compute_inventory=compute_inventory)
    service_providers = ServiceProviders(resolved) if providers is None else None
    manager = RunManager(
        artifact_store=artifacts,
        run_store=runs,
        model_service=models,
        max_concurrent_runs=resolved.api_max_concurrent_runs,
        cancel_wait_seconds=resolved.api_cancel_wait_seconds,
        tool_registry=tools,
        problem_registry=problems,
        providers=service_providers.providers if service_providers else providers,
        resources=(
            resources
            if resources is not None
            else service_providers.resources
            if service_providers
            else {}
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await manager.close()
            await models.close()
            if service_providers is not None:
                await service_providers.close()

    app = FastAPI(
        title="OASIS Anytime GeoAI API",
        version=API_SCHEMA_VERSION,
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url=None,
        responses={
            status: {"model": ApiErrorResponse, "description": "Structured API error"}
            for status in (400, 403, 404, 409, 413, 415, 422, 500, 503)
        },
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.artifact_store = artifacts
    app.state.run_store = runs
    app.state.model_service = models
    app.state.run_manager = manager
    app.add_middleware(RequestSizeLimitMiddleware, maximum_bytes=resolved.api_max_request_bytes)

    @app.exception_handler(RunManagerError)
    async def handle_run_error(_request: Request, error: RunManagerError) -> JSONResponse:
        return _error(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        fields = tuple(
            ".".join(str(value) for value in item["loc"] if value not in {"body", "path"})
            for item in error.errors()
        )
        return _error(422, "validation_error", "The request did not match the API schema.", fields)

    @app.exception_handler(ArtifactNotFoundError)
    async def handle_missing_artifact(
        _request: Request, _error_value: ArtifactNotFoundError
    ) -> JSONResponse:
        return _error(404, "artifact_not_found", "The requested artifact does not exist.")

    @app.exception_handler(ArtifactIntegrityError)
    async def handle_invalid_artifact(
        _request: Request, _error_value: ArtifactIntegrityError
    ) -> JSONResponse:
        return _error(400, "invalid_artifact_id", "The artifact ID is invalid.")

    @app.exception_handler(RunStoreError)
    async def handle_run_store_error(
        _request: Request, _error_value: RunStoreError
    ) -> JSONResponse:
        return _error(404, "run_not_found", "The requested run does not exist.")

    @app.exception_handler(ModelBackendError)
    async def handle_model_error(_request: Request, error: ModelBackendError) -> JSONResponse:
        return _error(503, error.detail.code.value, error.detail.message)

    @app.exception_handler(StarletteHttpException)
    async def handle_http_error(_request: Request, error: StarletteHttpException) -> JSONResponse:
        message = "The requested endpoint or resource is unavailable."
        return _error(error.status_code, "http_error", message)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _error_value: Exception) -> JSONResponse:
        return _error(500, "internal_error", "The service could not complete the request.")

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/api/v1/models", response_model=ModelCatalogResponse)
    async def list_models() -> ModelCatalogResponse:
        return ModelCatalogResponse(
            active_profile=models.backend.profile.name,
            active_model_id=models.backend.profile.model_id,
            models=tuple(
                ModelCatalogEntry(
                    name=name,
                    model_id=profile.model_id,
                    family=profile.family,
                    context_limit=profile.context_limit,
                    is_default=name == DEFAULT_PROFILE_NAME,
                    capabilities=models.profile_capabilities(name),
                )
                for name, profile in MODEL_PROFILES.items()
            ),
        )

    @app.get("/api/v1/runtime", response_model=RuntimeResponse)
    async def runtime() -> RuntimeResponse:
        return models.runtime_response()

    @app.get("/api/v1/tools", response_model=ToolCatalogResponse)
    async def list_tools() -> ToolCatalogResponse:
        return ToolCatalogResponse(
            tools=tuple(ToolCatalogEntry.from_spec(spec) for spec in tools.list())
        )

    @app.get("/api/v1/problems", response_model=ProblemCatalogResponse)
    async def list_problems() -> ProblemCatalogResponse:
        return ProblemCatalogResponse(
            problems=tuple(
                ProblemCatalogEntry(type_id=plugin.type_id, version=plugin.version)
                for plugin in problems.list()
            ),
            examples=example_catalog(),
        )

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        return await models.chat(request)

    @app.post("/api/v1/runs", status_code=202, response_model=RunCreatedResponse)
    async def create_run(request: RunCreateRequest) -> RunCreatedResponse:
        return await manager.start(request)

    @app.post("/api/v1/ask", response_model=RunResult)
    async def ask(request: MessageRequest) -> RunResult:
        """Submit only a message and wait for an answer using server defaults."""
        created = await manager.start(RunCreateRequest(message=request.message))
        return await manager.wait(created.run_id)

    @app.get("/api/v1/runs/{run_id}", response_model=RunInspectionResponse)
    async def inspect_run(run_id: str) -> RunInspectionResponse:
        return await manager.inspect(run_id)

    @app.get("/api/v1/runs/{run_id}/events")
    async def stream_events(
        run_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        if last_event_id is None:
            after_sequence = -1
        else:
            try:
                after_sequence = int(last_event_id)
            except ValueError as error:
                raise RunManagerError(
                    400, "invalid_event_id", "Last-Event-ID must be a non-negative integer."
                ) from error
            if after_sequence < 0:
                raise RunManagerError(
                    400, "invalid_event_id", "Last-Event-ID must be a non-negative integer."
                )
        queue = await manager.subscribe(run_id)

        async def generate() -> AsyncIterator[bytes]:
            cursor = after_sequence
            try:
                while True:
                    events = manager.read_events(run_id, after_sequence=cursor)
                    for event in events:
                        cursor = event.sequence
                        yield _sse_event(event.sequence, event.kind.value, event.model_dump_json())
                    inspection = await manager.inspect(run_id)
                    if inspection.result is not None:
                        return
                    if not await manager.is_active(run_id) and inspection.result is None:
                        return
                    if await request.is_disconnected():
                        return
                    try:
                        await asyncio.wait_for(
                            queue.get(), timeout=resolved.api_sse_heartbeat_seconds
                        )
                    except TimeoutError:
                        yield b": keep-alive\n\n"
            finally:
                await manager.unsubscribe(run_id, queue)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/api/v1/runs/{run_id}/cancel", response_model=CancelRunResponse)
    async def cancel_run(run_id: str) -> CancelRunResponse:
        return await manager.cancel(run_id)

    def artifact_response(reference: ArtifactRef) -> Response:
        if reference.privacy is not PrivacyClassification.PUBLIC:
            raise RunManagerError(
                403,
                "artifact_access_denied",
                "The artifact is not available through the local public API policy.",
            )
        if (
            reference.kind not in _ARTIFACT_KINDS
            or reference.media_type not in _ARTIFACT_MEDIA_TYPES
        ):
            raise RunManagerError(
                415,
                "artifact_type_not_allowed",
                "The artifact content type is not allowlisted for API delivery.",
            )
        if reference.byte_size > resolved.api_max_artifact_response_bytes:
            raise RunManagerError(
                413,
                "artifact_too_large",
                "The artifact exceeds the configured response-size limit.",
            )
        content = artifacts.read_bytes(reference.id)
        headers = {
            "Content-Length": str(len(content)),
            "ETag": f'"{reference.content_hash}"',
            "X-Content-Type-Options": "nosniff",
        }
        if reference.media_type == "image/svg+xml":
            headers["Content-Disposition"] = f'attachment; filename="{reference.id}.svg"'
            headers["Content-Security-Policy"] = "sandbox; default-src 'none'"
        return Response(content, media_type=reference.media_type, headers=headers)

    @app.get("/api/v1/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str) -> Response:
        return artifact_response(artifacts.get_metadata(artifact_id))

    @app.get("/api/v1/runs/{run_id}/map")
    async def get_run_map(
        run_id: str,
        format_name: Literal["geojson", "svg"] = Query(default="geojson", alias="format"),
    ) -> Response:
        reference = await manager.render_map(run_id, format_name=format_name)
        return artifact_response(reference)

    if resolved.serve_ui:
        ui_root = Path(resolved.ui_root).expanduser().resolve()
        if not ui_root.is_dir():
            raise ValueError("serve_ui requires an existing configured UI directory")

        @app.get("/{ui_path:path}", include_in_schema=False)
        async def static_ui(ui_path: str) -> Response:
            relative = ui_path or "index.html"
            candidate = (ui_root / relative).resolve()
            if not candidate.is_relative_to(ui_root) or not candidate.is_file():
                raise RunManagerError(404, "ui_file_not_found", "The UI file does not exist.")
            content = candidate.read_bytes()
            if len(content) > resolved.api_max_artifact_response_bytes:
                raise RunManagerError(413, "ui_file_too_large", "The UI file is too large.")
            media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            return Response(
                content,
                media_type=media_type,
                headers={"X-Content-Type-Options": "nosniff"},
            )

    return app


__all__ = ["create_app"]
