"""Portable metadata for immutable evidence and result artifacts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, model_validator

ARTIFACT_METADATA_SCHEMA_VERSION = "1.1.0"


class ArtifactKind(StrEnum):
    """Interoperable artifact categories understood across OASIS layers."""

    VECTOR = "vector"
    RASTER = "raster"
    TABLE = "table"
    GRAPH = "graph"
    MATRIX = "matrix"
    JSON_SPECIFICATION = "json_specification"
    PLAN = "plan"
    SCORECARD = "scorecard"
    MAP = "map"
    TRACE_ATTACHMENT = "trace_attachment"


class PrivacyClassification(StrEnum):
    """Coarse data-handling classification used by tools and artifacts."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class SpatialExtent(BaseModel):
    """Axis-aligned extent in the artifact CRS."""

    model_config = ConfigDict(frozen=True)

    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.west > self.east or self.south > self.north:
            raise ValueError("spatial extent bounds must be ordered")
        return self


class TemporalExtent(BaseModel):
    """Optional inclusive time range represented by timezone-aware timestamps."""

    model_config = ConfigDict(frozen=True)

    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        for value in (self.start, self.end):
            if value is not None and value.tzinfo is None:
                raise ValueError("temporal extent timestamps must include a timezone")
            if value is not None and value.utcoffset() != timedelta(0):
                raise ValueError("temporal extent timestamps must use UTC")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("temporal extent start must not follow its end")
        return self


class ArtifactTransformation(BaseModel):
    """One deterministic or externally versioned lineage operation."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ArtifactLineage(BaseModel):
    """Parents and transformations used to produce an artifact."""

    model_config = ConfigDict(frozen=True)

    parent_ids: tuple[str, ...] = ()
    transformations: tuple[ArtifactTransformation, ...] = ()


class QualitySummary(BaseModel):
    """Compact quality facts that travel with an artifact reference."""

    model_config = ConfigDict(frozen=True)

    missing_fraction: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    invalid_geometry_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    suppressed_count: int = Field(default=0, ge=0)
    warnings: tuple[str, ...] = ()


class ArtifactMetadata(BaseModel):
    """Caller-supplied metadata independent of storage identity and byte size."""

    model_config = ConfigDict(frozen=True)

    metadata_schema_version: str = Field(
        default=ARTIFACT_METADATA_SCHEMA_VERSION,
        pattern=r"^[1-9]\d*\.(0|[1-9]\d*)\.(0|[1-9]\d*)$",
    )
    kind: ArtifactKind
    media_type: str = Field(min_length=1)
    crs: str | None = None
    units: str | None = None
    spatial_extent: SpatialExtent | None = None
    temporal_extent: TemporalExtent | None = None
    data_schema: dict[str, JsonValue] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("data_schema", "schema"),
        serialization_alias="schema",
    )
    row_count: int | None = Field(default=None, ge=0)
    cell_count: int | None = Field(default=None, ge=0)
    edge_count: int | None = Field(default=None, ge=0)
    source_uri: str | None = None
    source_provider: str | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    license: str | None = None
    retrieved_at: datetime | None = None
    source_version: str | None = None
    lineage: ArtifactLineage = Field(default_factory=ArtifactLineage)
    quality: QualitySummary = Field(default_factory=QualitySummary)
    privacy: PrivacyClassification = PrivacyClassification.PUBLIC

    @model_validator(mode="after")
    def aware_retrieval_time(self) -> Self:
        if self.retrieved_at is not None and self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must include a timezone")
        if self.retrieved_at is not None and self.retrieved_at.utcoffset() != timedelta(0):
            raise ValueError("retrieved_at must use UTC")
        return self


class ArtifactRef(ArtifactMetadata):
    """Immutable content-addressed artifact reference with complete compact metadata."""

    id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def identity_matches_hash(self) -> Self:
        if self.id != f"sha256-{self.content_hash}":
            raise ValueError("artifact id must contain the content hash")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must use UTC")
        return self
