"""Model-independent chat, adapter, and tool-loop contracts."""

from oasis.llm.adapters import (
    Gemma4ChatAdapter,
    PlainChatAdapter,
    parse_gemma_tool_calls,
    parse_tagged_tool_calls,
)
from oasis.llm.fake import FakeModelBackend
from oasis.llm.profiles import DEFAULT_PROFILE_NAME, MODEL_PROFILES, resolve_model_profile
from oasis.llm.protocols import ConversationAdapter, ModelBackend
from oasis.llm.runtime_backend import RuntimeModelBackend
from oasis.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    ModelCapabilities,
    ModelDelta,
    ModelProfile,
    ModelRequest,
    ModelTurn,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    "DEFAULT_PROFILE_NAME",
    "MODEL_PROFILES",
    "ChatMessage",
    "ChatRole",
    "ConversationAdapter",
    "FakeModelBackend",
    "FinishReason",
    "Gemma4ChatAdapter",
    "ModelBackend",
    "ModelCapabilities",
    "ModelDelta",
    "ModelProfile",
    "ModelRequest",
    "ModelTurn",
    "PlainChatAdapter",
    "RuntimeModelBackend",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "parse_gemma_tool_calls",
    "parse_tagged_tool_calls",
    "resolve_model_profile",
]
