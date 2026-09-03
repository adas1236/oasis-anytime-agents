"""Public schemas for canonical demand, candidate, access, and service evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from oasis.schemas.artifacts import ArtifactRef


class MissingDataPolicy(StrEnum):
    """Declared treatment of missing values while building decision evidence."""

    ERROR = "error"
    DROP = "drop"
    ZERO = "zero"


class DemandSpec(BaseModel):
    """Canonical demand dimensions backed by one immutable vector or table artifact."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    artifact: ArtifactRef
    location_id_field: str = Field(min_length=1)
    need_fields: tuple[str, ...] = Field(min_length=1)
    group_fields: tuple[str, ...] = ()
    time_fields: tuple[str, ...] = ()
    suppression_fields: tuple[str, ...] = ()
    weighting_rules: dict[str, JsonValue] = Field(default_factory=dict)
    spatial_resolution: str | None = None
    missing_data_policy: MissingDataPolicy

    @model_validator(mode="after")
    def dimensions_are_unique(self) -> Self:
        dimensions = (
            self.need_fields + self.group_fields + self.time_fields + self.suppression_fields
        )
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("demand dimension fields must be unique across roles")
        return self


class CandidateSpec(BaseModel):
    """Canonical facility candidates backed by an immutable vector artifact."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    artifact: ArtifactRef
    candidate_id_field: str = Field(min_length=1)
    opening_cost_field: str | None = None
    capacity_field: str | None = None
    eligibility_field: str | None = None
    existing_site_field: str | None = None
    minimum_spacing: float | None = Field(default=None, ge=0.0)
    spacing_units: str | None = None
    generation_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def spacing_has_units(self) -> Self:
        if (self.minimum_spacing is None) != (self.spacing_units is None):
            raise ValueError("candidate spacing and spacing units must be declared together")
        return self


class AccessMatrixSpec(BaseModel):
    """Description of a demand-by-candidate impedance matrix."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    artifact: ArtifactRef
    origin_ids: tuple[str, ...]
    destination_ids: tuple[str, ...]
    strategy: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    units: str = Field(min_length=1)
    directed: bool
    unreachable_count: int = Field(ge=0)


class ServiceMatrixSpec(BaseModel):
    """Description of an access-to-benefit response matrix."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    artifact: ArtifactRef
    response: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    units: str = "unitless"
