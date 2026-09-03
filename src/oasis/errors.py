"""Typed public errors for model configuration and execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelErrorCode(StrEnum):
    """Stable machine-readable model error categories."""

    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    INVALID_MODEL = "invalid_model"
    MISSING_CHAT_TEMPLATE = "missing_chat_template"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_LOAD_FAILED = "model_load_failed"
    GENERATION_FAILED = "generation_failed"
    MALFORMED_TOOL_CALL = "malformed_tool_call"


class ModelErrorDetail(BaseModel):
    """Serializable details safe to expose at a public boundary."""

    model_config = ConfigDict(frozen=True)

    code: ModelErrorCode
    message: str
    model_id: str | None = None
    capability: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ModelBackendError(RuntimeError):
    """Base exception carrying structured, serializable model error details."""

    def __init__(self, detail: ModelErrorDetail) -> None:
        super().__init__(detail.message)
        self.detail = detail


class UnsupportedCapabilityError(ModelBackendError):
    """Raised when a request needs a capability the selected backend lacks."""

    def __init__(self, capability: str, model_id: str, message: str | None = None) -> None:
        super().__init__(
            ModelErrorDetail(
                code=ModelErrorCode.UNSUPPORTED_CAPABILITY,
                message=message or f"Model {model_id!r} does not support {capability}.",
                model_id=model_id,
                capability=capability,
            )
        )


class ToolCallParseError(ModelBackendError):
    """Raised when model output resembles a tool call but violates its adapter grammar."""

    def __init__(self, model_id: str, message: str, *, output: str = "") -> None:
        super().__init__(
            ModelErrorDetail(
                code=ModelErrorCode.MALFORMED_TOOL_CALL,
                message=message,
                model_id=model_id,
                capability="tool_call_parsing",
                context={"output_excerpt": output[:500]},
            )
        )
