"""Load frozen public-health-inspired synthetic fixture definitions."""

from __future__ import annotations

import json
from importlib.resources import files

from oasis.evaluation.models import FixtureName, SyntheticInstanceSpec


def load_fixture(name: FixtureName | str) -> SyntheticInstanceSpec:
    """Load and validate one package-owned immutable fixture definition."""

    resolved = FixtureName(name)
    resource = files("oasis.evaluation").joinpath("fixtures", f"{resolved.value}.json")
    return SyntheticInstanceSpec.model_validate(json.loads(resource.read_text(encoding="utf-8")))


def fixture_catalog() -> dict[str, SyntheticInstanceSpec]:
    """Return every frozen fixture through the same public validation path."""

    return {name.value: load_fixture(name) for name in FixtureName}
