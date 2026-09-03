from __future__ import annotations

import pytest

from oasis.config import DevicePolicy, OasisSettings, RuntimeConfig, resolve_device


def test_configuration_precedence_is_defaults_env_cli_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OASIS_MODEL_PROFILE", "gemma4_e2b_it")
    monkeypatch.setenv("OASIS_MAX_GENERATED_TOKENS", "11")

    settings = OasisSettings.resolve(
        cli_overrides={"model_profile": "gemma4_12b_it", "max_generated_tokens": 22},
        explicit_overrides={"model_profile": "gemma4_31b_it"},
    )

    assert settings.model_profile == "gemma4_31b_it"
    assert settings.max_generated_tokens == 22


def test_environment_wins_over_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OASIS_BACKEND", "fake")
    monkeypatch.setenv("OASIS_THINKING", "true")

    settings = OasisSettings.resolve()

    assert settings.backend.value == "fake"
    assert settings.thinking is True


def test_none_cli_values_do_not_mask_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OASIS_MODEL_PROFILE", "gemma4_e2b_it")

    settings = OasisSettings.resolve(cli_overrides={"model_profile": None})

    assert settings.model_profile == "gemma4_e2b_it"


def test_cpu_default_wins_without_calling_mocked_cuda_probe() -> None:
    probed = False

    def visible_cuda() -> bool:
        nonlocal probed
        probed = True
        return True

    assert resolve_device(DevicePolicy.CPU, visible_cuda) is DevicePolicy.CPU
    assert not probed


def test_runtime_configuration_round_trips_without_a_probe() -> None:
    runtime = RuntimeConfig(
        device="auto",
        engine="transformers",
        dtype="bfloat16",
        parallelism="auto",
        allow_cpu_offload=True,
    )

    restored = RuntimeConfig.model_validate_json(runtime.model_dump_json())

    assert restored == runtime


def test_settings_runtime_projection_is_side_effect_free() -> None:
    settings = OasisSettings(device="cpu", runtime_engine="auto")

    assert settings.runtime_config().device is DevicePolicy.CPU


def test_provider_limits_are_environment_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OASIS_PROVIDER_USER_AGENT", "example/1.0 contact@example.test")
    monkeypatch.setenv("OASIS_PROVIDER_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("OASIS_PROVIDER_MAX_RESPONSE_BYTES", "4096")
    monkeypatch.setenv("OASIS_PROVIDER_MAX_PAGES", "4")

    settings = OasisSettings()

    assert settings.provider_user_agent == "example/1.0 contact@example.test"
    assert settings.provider_timeout_seconds == 2.5
    assert settings.provider_max_response_bytes == 4096
    assert settings.provider_max_pages == 4
