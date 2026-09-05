"""Canonical demand and candidate construction without hidden composite scoring."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from shapely.geometry import Point

from oasis.artifacts import ArtifactProvenance, put_json, put_table, put_vector
from oasis.schemas import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactTransformation,
    CandidateSpec,
    DemandSpec,
    DeterminismClassification,
    MissingDataPolicy,
    SideEffectClassification,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.evidence.common import (
    MISSING_ARTIFACT_ID,
    artifact_ref,
    child_provenance,
    invalid,
    read_frame,
    require_fields,
    require_kind,
)
from oasis.tools.protocols import ToolContext


class BuildDemandInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    location_id_field: str = "id"
    need_fields: tuple[str, ...] = Field(min_length=1)
    group_fields: tuple[str, ...] = ()
    time_fields: tuple[str, ...] = ()
    suppression_fields: tuple[str, ...] = ()
    weighting_rules: dict[str, JsonValue] = Field(default_factory=dict)
    spatial_resolution: str | None = None
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.ERROR


class BuildDemandOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    demand_artifact_id: str
    demand_spec_artifact_id: str
    row_count: int
    need_fields: tuple[str, ...]
    group_fields: tuple[str, ...]
    suppressed_count: int


class CandidateGenerationMode(StrEnum):
    SUPPLIED = "supplied"
    GRID = "grid"


class SuitabilityPredicate(StrEnum):
    WITHIN = "within"
    INTERSECTS = "intersects"


class BuildCandidatesInput(BaseModel):
    """For supplied mode, require artifact_id and omit all grid inputs.

    For grid mode, require grid_bounds, grid_spacing, and grid_crs and omit
    artifact_id. Supply minimum_spacing and spacing_units together; nonempty
    allowed_suitability_values requires suitability_field.
    """

    model_config = ConfigDict(frozen=True)

    mode: CandidateGenerationMode
    artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    grid_bounds: tuple[float, float, float, float] | None = None
    grid_spacing: float | None = Field(default=None, gt=0.0)
    grid_crs: str | None = None
    suitability_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    suitability_predicate: SuitabilityPredicate = SuitabilityPredicate.WITHIN
    suitability_field: str | None = None
    allowed_suitability_values: tuple[str | int | float | bool | None, ...] = ()
    candidate_id_field: str = "id"
    opening_cost_field: str | None = None
    capacity_field: str | None = None
    eligibility_field: str | None = None
    existing_site_field: str | None = None
    minimum_spacing: float | None = Field(default=None, ge=0.0)
    spacing_units: str | None = None

    @model_validator(mode="after")
    def generation_inputs_match(self) -> BuildCandidatesInput:
        if self.mode is CandidateGenerationMode.SUPPLIED:
            if self.artifact_id is None:
                raise ValueError("supplied candidate generation requires artifact_id")
            if (
                self.grid_bounds is not None
                or self.grid_spacing is not None
                or self.grid_crs is not None
            ):
                raise ValueError("supplied candidate generation cannot include grid inputs")
        else:
            if self.artifact_id is not None:
                raise ValueError("grid candidate generation cannot include artifact_id")
            if self.grid_bounds is None or self.grid_spacing is None or self.grid_crs is None:
                raise ValueError("grid generation requires bounds, spacing, and CRS")
            west, south, east, north = self.grid_bounds
            if west > east or south > north:
                raise ValueError("grid bounds must be ordered")
        if (self.minimum_spacing is None) != (self.spacing_units is None):
            raise ValueError("minimum spacing and spacing units must be supplied together")
        if self.allowed_suitability_values and self.suitability_field is None:
            raise ValueError("allowed suitability values require suitability_field")
        return self


class BuildCandidatesOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_artifact_id: str
    candidate_spec_artifact_id: str
    row_count: int
    generation_method: CandidateGenerationMode
    filtered_count: int


def _selected_columns(
    frame: pd.DataFrame, fields: tuple[str, ...], *, keep_geometry: bool
) -> pd.DataFrame:
    selected = list(dict.fromkeys(fields))
    if keep_geometry:
        assert isinstance(frame, gpd.GeoDataFrame)
        selected.append(frame.geometry.name)
        return gpd.GeoDataFrame(frame[selected].copy(), geometry=frame.geometry.name, crs=frame.crs)
    return frame[selected].copy()


def _suppressed(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.Series:
    result = pd.Series(False, index=frame.index, dtype=bool)
    for field in fields:
        result |= frame[field].map(
            lambda value: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else str(value).strip().lower() in {"s", "suppressed", "*", "true"}
            )
        )
    return result


class BuildDemandTool:
    """Build a dimension-preserving demand artifact and immutable typed specification."""

    version = "1.0.0"
    spec = ToolSpec(
        name="build_demand",
        version=version,
        description=(
            "Build canonical demand evidence while retaining separate need, group, time, and "
            "suppression dimensions and an explicit missing-data policy."
        ),
        input_schema=BuildDemandInput.model_json_schema(),
        output_schema=BuildDemandOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "demand", "public_health", "offline"}),
        problem_tags=frozenset({"location_allocation", "routing"}),
        artifact_tags=frozenset({ArtifactKind.VECTOR, ArtifactKind.TABLE}),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=10, p95_ms=1_000),
        smoke_input={"artifact_id": MISSING_ARTIFACT_ID, "need_fields": ["need"]},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = BuildDemandInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        source = artifact_ref(context, request.artifact_id)
        require_kind(source, {ArtifactKind.VECTOR, ArtifactKind.TABLE})
        frame = read_frame(context, source)
        roles = (
            request.need_fields
            + request.group_fields
            + request.time_fields
            + request.suppression_fields
        )
        if len(roles) != len(set(roles)):
            invalid("demand fields cannot have more than one semantic role")
        require_fields(frame, [request.location_id_field, *roles])
        if frame[request.location_id_field].isna().any():
            invalid("demand location IDs cannot be missing")
        if frame[request.location_id_field].astype(str).duplicated().any():
            invalid("demand location IDs must be unique")
        canonical = _selected_columns(
            frame,
            (request.location_id_field, *roles),
            keep_geometry=source.kind is ArtifactKind.VECTOR,
        )
        canonical[request.location_id_field] = canonical[request.location_id_field].astype(str)
        for field in request.need_fields:
            try:
                canonical[field] = pd.to_numeric(canonical[field], errors="raise")
            except (TypeError, ValueError) as error:
                invalid(f"need field {field!r} must be numeric: {error}")
            if (canonical[field].dropna() < 0).any():
                invalid(f"need field {field!r} cannot contain negative values")

        required = [*request.need_fields, *request.group_fields, *request.time_fields]
        missing = canonical[required].isna().any(axis=1)
        suppressed = _suppressed(canonical, request.suppression_fields)
        actionable_missing = missing & ~suppressed
        if actionable_missing.any() and request.missing_data_policy is MissingDataPolicy.ERROR:
            invalid(
                "demand contains missing unsuppressed dimensions",
                context={"row_count": int(actionable_missing.sum())},
            )
        if request.missing_data_policy is MissingDataPolicy.DROP:
            canonical = canonical.loc[~actionable_missing].copy()
            suppressed = suppressed.loc[canonical.index]
        elif request.missing_data_policy is MissingDataPolicy.ZERO:
            if canonical[[*request.group_fields, *request.time_fields]].isna().any().any():
                invalid(
                    "zero missing-data policy applies only to need fields, not group/time fields"
                )
            canonical[list(request.need_fields)] = canonical[list(request.need_fields)].fillna(0)
        canonical = canonical.reset_index(drop=True)
        suppressed_count = int(suppressed.sum())
        provenance = child_provenance(self.spec.name, self.version, [source], request)
        if isinstance(canonical, gpd.GeoDataFrame):
            demand_ref = put_vector(
                context.artifact_store,
                canonical,
                units=source.units or "count",
                provenance=provenance,
            )
        else:
            demand_ref = put_table(
                context.artifact_store,
                canonical,
                crs=source.crs,
                units=source.units or "count",
                provenance=provenance,
            )
        demand_spec = DemandSpec(
            artifact=demand_ref,
            location_id_field=request.location_id_field,
            need_fields=request.need_fields,
            group_fields=request.group_fields,
            time_fields=request.time_fields,
            suppression_fields=request.suppression_fields,
            weighting_rules=request.weighting_rules,
            spatial_resolution=request.spatial_resolution,
            missing_data_policy=request.missing_data_policy,
        )
        spec_ref = put_json(
            context.artifact_store,
            demand_spec.model_dump(mode="json"),
            kind=ArtifactKind.JSON_SPECIFICATION,
            units="unitless",
            provenance=child_provenance(
                self.spec.name,
                self.version,
                [demand_ref],
                {"schema": "DemandSpec", "schema_version": demand_spec.schema_version},
            ),
            data_schema={"type": "DemandSpec", "version": demand_spec.schema_version},
        )
        output = BuildDemandOutput(
            demand_artifact_id=demand_ref.id,
            demand_spec_artifact_id=spec_ref.id,
            row_count=len(canonical),
            need_fields=request.need_fields,
            group_fields=request.group_fields,
            suppressed_count=suppressed_count,
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "demand_spec": spec_ref.id,
                "rows": len(canonical),
                "need_fields": list(request.need_fields),
                "group_fields": list(request.group_fields),
            },
            artifacts=(demand_ref, spec_ref),
            metrics=output.model_dump(mode="json"),
        )


def _grid(request: BuildCandidatesInput) -> gpd.GeoDataFrame:
    assert request.grid_bounds is not None
    assert request.grid_spacing is not None
    assert request.grid_crs is not None
    west, south, east, north = request.grid_bounds
    xs = np.arange(west, east + request.grid_spacing * 0.5, request.grid_spacing)
    ys = np.arange(south, north + request.grid_spacing * 0.5, request.grid_spacing)
    points = [Point(float(x), float(y)) for y in ys for x in xs if x <= east and y <= north]
    return gpd.GeoDataFrame(
        {request.candidate_id_field: [f"grid-{index + 1:06d}" for index in range(len(points))]},
        geometry=points,
        crs=request.grid_crs,
    )


def _grid_provenance(request: BuildCandidatesInput) -> ArtifactProvenance:
    return ArtifactProvenance(
        source_uri="oasis://generator/deterministic-grid/1.0.0",
        source_provider="oasis",
        source_version="1.0.0",
        license="CC0-1.0",
        lineage=ArtifactLineage(
            transformations=(
                ArtifactTransformation(
                    name="deterministic_grid",
                    version="1.0.0",
                    parameters=request.model_dump(mode="json", exclude_none=True),
                ),
            )
        ),
    )


class BuildCandidatesTool:
    """Build candidates from supplied point facilities or a deterministic suitability grid."""

    version = "1.0.0"
    spec = ToolSpec(
        name="build_candidates",
        version=version,
        description=(
            "Build canonical facility candidates from supplied sites or a deterministic grid, "
            "with explicit suitability, cost, capacity, eligibility, and spacing semantics."
        ),
        input_schema=BuildCandidatesInput.model_json_schema(),
        output_schema=BuildCandidatesOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "candidates", "offline"}),
        problem_tags=frozenset({"location_allocation"}),
        artifact_tags=frozenset({ArtifactKind.VECTOR}),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=10, p95_ms=1_500),
        smoke_input={"mode": "supplied", "artifact_id": MISSING_ARTIFACT_ID},
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = BuildCandidatesInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        source_refs = []
        if request.mode is CandidateGenerationMode.SUPPLIED:
            assert request.artifact_id is not None
            source = artifact_ref(context, request.artifact_id)
            require_kind(source, {ArtifactKind.VECTOR})
            source_refs.append(source)
            frame = read_frame(context, source)
            assert isinstance(frame, gpd.GeoDataFrame)
            provenance = child_provenance(self.spec.name, self.version, source_refs, request)
        else:
            frame = _grid(request)
            provenance = _grid_provenance(request)
        if frame.crs is None:
            invalid("candidate vector must have an explicit CRS")
        if not frame.geometry.map(
            lambda geometry: geometry is not None and geometry.geom_type == "Point"
        ).all():
            invalid("candidate locations must be non-empty point geometries")

        fields = [
            field
            for field in (
                request.candidate_id_field,
                request.opening_cost_field,
                request.capacity_field,
                request.eligibility_field,
                request.existing_site_field,
                request.suitability_field if request.suitability_artifact_id is None else None,
            )
            if field is not None
        ]
        require_fields(frame, fields)
        if frame[request.candidate_id_field].isna().any():
            invalid("candidate IDs cannot be missing")
        frame[request.candidate_id_field] = frame[request.candidate_id_field].astype(str)
        if frame[request.candidate_id_field].duplicated().any():
            invalid("candidate IDs must be unique")
        original_count = len(frame)
        if request.suitability_field is not None and request.suitability_artifact_id is None:
            if request.allowed_suitability_values:
                frame = frame.loc[
                    frame[request.suitability_field].isin(request.allowed_suitability_values)
                ].copy()
            else:
                frame = frame.loc[frame[request.suitability_field].astype(bool)].copy()
        if request.suitability_artifact_id is not None:
            suitability_ref = artifact_ref(context, request.suitability_artifact_id)
            require_kind(suitability_ref, {ArtifactKind.VECTOR})
            source_refs.append(suitability_ref)
            suitability = read_frame(context, suitability_ref)
            assert isinstance(suitability, gpd.GeoDataFrame)
            if request.suitability_field is not None:
                require_fields(suitability, [request.suitability_field])
                if request.allowed_suitability_values:
                    suitability = suitability.loc[
                        suitability[request.suitability_field].isin(
                            request.allowed_suitability_values
                        )
                    ].copy()
                else:
                    suitability = suitability.loc[
                        suitability[request.suitability_field].astype(bool)
                    ].copy()
            if suitability.crs != frame.crs:
                suitability = suitability.to_crs(frame.crs)
            allowed_area = suitability.geometry.union_all()
            predicate = (
                frame.geometry.within(allowed_area)
                if request.suitability_predicate is SuitabilityPredicate.WITHIN
                else frame.geometry.intersects(allowed_area)
            )
            frame = frame.loc[predicate].copy()
            provenance = child_provenance(self.spec.name, self.version, source_refs, request)
        for field in (request.opening_cost_field, request.capacity_field):
            if field is not None:
                try:
                    frame[field] = pd.to_numeric(frame[field], errors="raise")
                except (TypeError, ValueError) as error:
                    invalid(f"candidate field {field!r} must be numeric: {error}")
                if (frame[field].dropna() < 0).any():
                    invalid(f"candidate field {field!r} cannot be negative")
        keep = [*dict.fromkeys(fields), frame.geometry.name]
        canonical = gpd.GeoDataFrame(
            frame[keep].reset_index(drop=True), geometry=frame.geometry.name, crs=frame.crs
        )
        candidate_ref = put_vector(
            context.artifact_store,
            canonical,
            units=(source_refs[0].units if source_refs else request.spacing_units) or "unitless",
            provenance=provenance,
        )
        candidate_spec = CandidateSpec(
            artifact=candidate_ref,
            candidate_id_field=request.candidate_id_field,
            opening_cost_field=request.opening_cost_field,
            capacity_field=request.capacity_field,
            eligibility_field=request.eligibility_field,
            existing_site_field=request.existing_site_field,
            minimum_spacing=request.minimum_spacing,
            spacing_units=request.spacing_units,
            generation_method=request.mode.value,
        )
        spec_ref = put_json(
            context.artifact_store,
            candidate_spec.model_dump(mode="json"),
            kind=ArtifactKind.JSON_SPECIFICATION,
            units="unitless",
            provenance=child_provenance(
                self.spec.name,
                self.version,
                [candidate_ref],
                {"schema": "CandidateSpec", "schema_version": candidate_spec.schema_version},
            ),
            data_schema={"type": "CandidateSpec", "version": candidate_spec.schema_version},
        )
        output = BuildCandidatesOutput(
            candidate_artifact_id=candidate_ref.id,
            candidate_spec_artifact_id=spec_ref.id,
            row_count=len(canonical),
            generation_method=request.mode,
            filtered_count=original_count - len(canonical),
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "candidate_spec": spec_ref.id,
                "rows": len(canonical),
                "generation": request.mode.value,
            },
            artifacts=(candidate_ref, spec_ref),
            metrics=output.model_dump(mode="json"),
        )
