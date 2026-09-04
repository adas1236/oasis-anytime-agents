"""Central Gemma 4 model profile registry."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from oasis.errors import ModelBackendError, ModelErrorCode, ModelErrorDetail
from oasis.llm.schemas import ModelProfile

DEFAULT_PROFILE_NAME = "gemma4_e4b_it"

_MODEL_PROFILES = {
    "gemma4_e2b_it": ModelProfile(
        name="gemma4_e2b_it",
        model_id="google/gemma-4-E2B-it",
        family="gemma4",
        context_limit=131_072,
        supports_thinking=True,
        supports_native_tools=True,
        # E2B describes active/effective parameters. Placement must reserve for
        # the complete multimodal checkpoint, measured at 5.104B parameters.
        estimated_parameter_count=5_200_000_000,
    ),
    "gemma4_e4b_it": ModelProfile(
        name="gemma4_e4b_it",
        model_id="google/gemma-4-E4B-it",
        family="gemma4",
        context_limit=131_072,
        supports_thinking=True,
        supports_native_tools=True,
        estimated_parameter_count=4_000_000_000,
    ),
    "gemma4_12b_it": ModelProfile(
        name="gemma4_12b_it",
        model_id="google/gemma-4-12B-it",
        family="gemma4",
        context_limit=262_144,
        supports_thinking=True,
        supports_native_tools=True,
        estimated_parameter_count=12_000_000_000,
    ),
    "gemma4_26b_a4b_it": ModelProfile(
        name="gemma4_26b_a4b_it",
        model_id="google/gemma-4-26B-A4B-it",
        family="gemma4",
        context_limit=262_144,
        supports_thinking=True,
        supports_native_tools=True,
        estimated_parameter_count=26_000_000_000,
    ),
    "gemma4_31b_it": ModelProfile(
        name="gemma4_31b_it",
        model_id="google/gemma-4-31B-it",
        family="gemma4",
        context_limit=262_144,
        supports_thinking=True,
        supports_native_tools=True,
        estimated_parameter_count=31_000_000_000,
    ),
}

MODEL_PROFILES: Mapping[str, ModelProfile] = MappingProxyType(_MODEL_PROFILES)


def resolve_model_profile(
    profile_name: str = DEFAULT_PROFILE_NAME,
    explicit_model_id: str | None = None,
) -> ModelProfile:
    """Resolve a profile, with an explicit Hugging Face ID taking precedence."""

    if explicit_model_id:
        registered = next(
            (
                candidate
                for candidate in _MODEL_PROFILES.values()
                if candidate.model_id == explicit_model_id
            ),
            None,
        )
        family = "gemma4" if explicit_model_id.lower().startswith("google/gemma-4-") else "custom"
        return ModelProfile(
            name="custom",
            model_id=explicit_model_id,
            family=family,
            context_limit=registered.context_limit if registered is not None else None,
            supports_thinking=family == "gemma4",
            supports_native_tools=family == "gemma4",
            is_custom=True,
            estimated_parameter_count=(
                registered.estimated_parameter_count if registered is not None else None
            ),
        )
    try:
        return _MODEL_PROFILES[profile_name]
    except KeyError as error:
        raise ModelBackendError(
            ModelErrorDetail(
                code=ModelErrorCode.INVALID_MODEL,
                message=f"Unknown model profile {profile_name!r}.",
                context={"available_profiles": sorted(_MODEL_PROFILES)},
            )
        ) from error
