from __future__ import annotations

import pytest

from oasis.errors import ModelBackendError, ModelErrorCode
from oasis.llm.profiles import DEFAULT_PROFILE_NAME, MODEL_PROFILES, resolve_model_profile

EXPECTED_PROFILE_IDS = {
    "gemma4_e2b_it": "google/gemma-4-E2B-it",
    "gemma4_e4b_it": "google/gemma-4-E4B-it",
    "gemma4_12b_it": "google/gemma-4-12B-it",
    "gemma4_26b_a4b_it": "google/gemma-4-26B-A4B-it",
    "gemma4_31b_it": "google/gemma-4-31B-it",
}


def test_profile_names_resolve_to_exact_model_ids() -> None:
    assert {name: profile.model_id for name, profile in MODEL_PROFILES.items()} == (
        EXPECTED_PROFILE_IDS
    )


def test_default_profile_is_gemma_4_e4b_it() -> None:
    assert DEFAULT_PROFILE_NAME == "gemma4_e4b_it"
    assert resolve_model_profile().model_id == "google/gemma-4-E4B-it"


def test_explicit_model_id_overrides_profile() -> None:
    profile = resolve_model_profile("gemma4_e2b_it", "organization/custom-chat-model")

    assert profile.name == "custom"
    assert profile.model_id == "organization/custom-chat-model"
    assert profile.is_custom
    assert profile.family == "custom"


def test_unknown_profile_is_a_typed_error() -> None:
    with pytest.raises(ModelBackendError) as caught:
        resolve_model_profile("not-a-profile")

    assert caught.value.detail.code is ModelErrorCode.INVALID_MODEL
    assert "gemma4_e4b_it" in caught.value.detail.context["available_profiles"]
