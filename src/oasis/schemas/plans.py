"""Problem-neutral plan envelope used at tool boundaries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Plan(BaseModel):
    """A versioned plan whose permitted fields are constrained by each problem plugin."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default="1.0.0", min_length=1)
    problem_type: str = Field(min_length=1)
    selected_site_ids: tuple[str, ...] = ()
    assignments: tuple[dict[str, JsonValue], ...] = ()
    allocations: tuple[dict[str, JsonValue], ...] = ()
    routes: tuple[dict[str, JsonValue], ...] = ()
    schedules: tuple[dict[str, JsonValue], ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
