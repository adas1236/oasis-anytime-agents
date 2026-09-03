"""Validated explicit and entry-point-backed tool registration."""

from __future__ import annotations

import importlib.metadata
import inspect
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, TypeGuard

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from oasis.llm.schemas import ToolDefinition
from oasis.schemas.artifacts import ArtifactKind
from oasis.schemas.tools import ToolResult, ToolResultStatus, ToolSpec
from oasis.tools.protocols import StreamingTool, Tool


class ToolRegistryError(ValueError):
    """Raised when a tool cannot safely enter a registry."""


class EntryPointLike(Protocol):
    """Narrow entry-point interface used to make discovery independently testable."""

    @property
    def name(self) -> str: ...

    def load(self) -> object: ...


def _check_schema(schema: Mapping[str, Any], *, label: str, require_object: bool = True) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ToolRegistryError(f"invalid {label} JSON Schema: {error.message}") from error
    if require_object and schema.get("type") != "object":
        raise ToolRegistryError(f"{label} JSON Schema must describe an object")


def _is_tool(value: object) -> TypeGuard[Tool | StreamingTool]:
    return hasattr(value, "spec") and (
        callable(getattr(value, "run", None)) or callable(getattr(value, "stream", None))
    )


def _check_handler_signature(handler: object, *, label: str) -> None:
    try:
        parameters = tuple(inspect.signature(handler).parameters.values())  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ToolRegistryError(f"{label} has no inspectable signature") from error
    if len(parameters) != 2 or any(
        parameter.kind
        not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        for parameter in parameters
    ):
        raise ToolRegistryError(f"{label} must accept exactly (arguments, context)")


class ToolRegistry:
    """Read-mostly registry validated before tools are exposed to a model."""

    def __init__(self, tools: Iterable[Tool | StreamingTool] = ()) -> None:
        self._tools: dict[str, Tool | StreamingTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool | StreamingTool) -> None:
        """Validate and add one tool; registration is never an import side effect."""

        if not _is_tool(tool):
            raise ToolRegistryError("tool handler must expose spec and async run or stream")
        spec = tool.spec
        if not isinstance(spec, ToolSpec):
            raise ToolRegistryError("tool spec must be a ToolSpec instance")
        try:
            ToolSpec.model_validate(spec.model_dump())
        except PydanticValidationError as error:
            raise ToolRegistryError(f"tool {spec.name!r} has an invalid spec: {error}") from error
        if spec.name in self._tools:
            existing = self._tools[spec.name].spec.version
            raise ToolRegistryError(
                f"duplicate tool name {spec.name!r}: versions {existing} and {spec.version}"
            )
        _check_schema(spec.input_schema, label=f"{spec.name} input")
        _check_schema(spec.output_schema, label=f"{spec.name} output")
        if spec.resume_token_schema is not None:
            _check_schema(
                spec.resume_token_schema,
                label=f"{spec.name} resume token",
                require_object=False,
            )
        try:
            Draft202012Validator(spec.input_schema).validate(spec.smoke_input)
        except JsonSchemaValidationError as error:
            raise ToolRegistryError(
                f"smoke input for {spec.name!r} does not satisfy its schema: {error.message}"
            ) from error

        run = getattr(tool, "run", None)
        stream = getattr(tool, "stream", None)
        if run is not None and not inspect.iscoroutinefunction(run):
            raise ToolRegistryError(f"tool {spec.name!r} run handler must be async")
        if run is not None:
            _check_handler_signature(run, label=f"tool {spec.name!r} run handler")
        if stream is not None and not inspect.isasyncgenfunction(stream):
            raise ToolRegistryError(f"tool {spec.name!r} stream handler must be an async generator")
        if stream is not None:
            _check_handler_signature(stream, label=f"tool {spec.name!r} stream handler")
        declares_streaming = spec.streams_progress or spec.streams_candidates or spec.streams_bounds
        if declares_streaming and stream is None:
            raise ToolRegistryError(
                f"tool {spec.name!r} declares streaming without a stream handler"
            )
        if not declares_streaming and run is None:
            raise ToolRegistryError(f"tool {spec.name!r} needs an async run handler")
        self._tools[spec.name] = tool

    def get(self, name: str) -> Tool | StreamingTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolRegistryError(f"unknown tool {name!r}") from error

    def list(self) -> tuple[ToolSpec, ...]:
        """Return specs in stable name order."""

        return tuple(self._tools[name].spec for name in sorted(self._tools))

    def select(
        self,
        *,
        capabilities: frozenset[str] = frozenset(),
        problem_tags: frozenset[str] = frozenset(),
        artifact_tags: frozenset[ArtifactKind] = frozenset(),
    ) -> tuple[ToolSpec, ...]:
        """Return only tools satisfying every requested task/capability dimension."""

        return tuple(
            spec
            for spec in self.list()
            if capabilities <= spec.capability_tags
            and (not problem_tags or bool(problem_tags & spec.problem_tags))
            and (not artifact_tags or bool(artifact_tags & spec.artifact_tags))
        )

    def model_definitions(
        self, specs: Iterable[ToolSpec] | None = None
    ) -> tuple[ToolDefinition, ...]:
        """Project selected registry specs into the model-facing schema."""

        selected = self.list() if specs is None else tuple(specs)
        return tuple(
            ToolDefinition(
                name=spec.name,
                description=spec.description,
                input_schema=spec.input_schema,
            )
            for spec in selected
        )

    def discover(self, entry_points: Iterable[EntryPointLike] | None = None) -> None:
        """Load third-party tools from the ``oasis.tools`` entry-point group."""

        discovered: Iterable[EntryPointLike]
        if entry_points is None:
            discovered = importlib.metadata.entry_points(group="oasis.tools")
        else:
            discovered = entry_points
        for entry_point in discovered:
            loaded = entry_point.load()
            provided = (
                loaded()
                if callable(loaded) and (inspect.isclass(loaded) or not _is_tool(loaded))
                else loaded
            )
            if _is_tool(provided):
                self.register(provided)
                continue
            if isinstance(provided, Iterable) and not isinstance(provided, (str, bytes, Mapping)):
                for tool in provided:
                    if not _is_tool(tool):
                        raise ToolRegistryError(
                            f"entry point {entry_point.name!r} returned a non-tool value"
                        )
                    self.register(tool)
                continue
            raise ToolRegistryError(f"entry point {entry_point.name!r} did not provide a tool")


def validate_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> None:
    """Validate invocation arguments with the same schema accepted at registration."""

    try:
        Draft202012Validator(spec.input_schema).validate(dict(arguments))
    except JsonSchemaValidationError as error:
        raise ToolRegistryError(f"invalid arguments for {spec.name!r}: {error.message}") from error


def validate_output(spec: ToolSpec, result: ToolResult) -> None:
    """Validate tool-specific metrics and any declared opaque resume token."""

    if result.status in {
        ToolResultStatus.EXPIRED,
        ToolResultStatus.FAILED,
        ToolResultStatus.RATE_LIMITED,
    }:
        return
    try:
        Draft202012Validator(spec.output_schema).validate(dict(result.metrics))
    except JsonSchemaValidationError as error:
        raise ToolRegistryError(f"invalid output from {spec.name!r}: {error.message}") from error
    if result.resume_token is not None:
        if spec.resume_token_schema is None:
            raise ToolRegistryError(f"non-resumable tool {spec.name!r} returned a resume token")
        try:
            Draft202012Validator(spec.resume_token_schema).validate(result.resume_token)
        except JsonSchemaValidationError as error:
            raise ToolRegistryError(
                f"invalid resume token from {spec.name!r}: {error.message}"
            ) from error
