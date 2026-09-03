"""Deterministic plan summaries and simple GeoJSON/SVG map rendering."""

from __future__ import annotations

import html
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from shapely.geometry import mapping

from oasis.artifacts import canonical_json_bytes, put_json, read_vector
from oasis.problems.location_allocation import create_problem_registry
from oasis.problems.registry import ProblemRegistry
from oasis.schemas import (
    ArtifactKind,
    ArtifactMetadata,
    DeterminismClassification,
    QualitySummary,
    SideEffectClassification,
    SpatialExtent,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.decision.common import decision_provenance, read_plan, read_problem
from oasis.tools.evidence.common import MISSING_ARTIFACT_ID, invalid
from oasis.tools.protocols import ToolContext


class SummarizePlanInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    plan_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")


class SummarizePlanOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary_artifact_id: str
    scorecard_artifact_id: str
    feasible: bool
    selected_site_count: int = Field(ge=0)
    comparator_key: tuple[float, ...]


class SummarizePlanTool:
    """Independently score a plan and publish a deterministic compact summary."""

    version = "1.0.0"
    spec = ToolSpec(
        name="summarize_plan",
        version=version,
        description=(
            "Validate and independently score a location plan, then publish a deterministic "
            "summary containing raw overall, group, and scenario metrics."
        ),
        input_schema=SummarizePlanInput.model_json_schema(),
        output_schema=SummarizePlanOutput.model_json_schema(),
        capability_tags=frozenset({"decision", "summary", "location_allocation", "offline"}),
        problem_tags=frozenset({"location_allocation"}),
        artifact_tags=frozenset(
            {ArtifactKind.JSON_SPECIFICATION, ArtifactKind.PLAN, ArtifactKind.SCORECARD}
        ),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=5, p95_ms=1_000),
        smoke_input={
            "problem_artifact_id": MISSING_ARTIFACT_ID,
            "plan_artifact_id": MISSING_ARTIFACT_ID,
        },
    )

    def __init__(self, registry: ProblemRegistry | None = None) -> None:
        self._registry = registry or create_problem_registry()

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = SummarizePlanInput.model_validate(arguments)
        problem_ref, problem = read_problem(context, request.problem_artifact_id)
        plan_ref, plan = read_plan(context, request.plan_artifact_id)
        plugin = self._registry.get(problem.type_id.value)
        problem_report = plugin.validate_spec(problem, context.artifact_store)
        if not problem_report.valid:
            invalid(f"invalid problem: {problem_report.issues[0].message}")
        score = plugin.measure(problem, plan, context.artifact_store)
        view = plugin.render_result(problem, plan, score)
        score_ref = put_json(
            context.artifact_store,
            score.model_dump(mode="json"),
            kind=ArtifactKind.SCORECARD,
            units="unitless",
            provenance=decision_provenance(
                self.spec.name,
                self.version,
                (problem_ref, plan_ref),
                {"evaluator_version": problem.evaluator_version},
            ),
            data_schema={"type": "Scorecard", "version": score.schema_version},
        )
        summary = {
            **view.model_dump(mode="json"),
            "problem_hash": problem.problem_hash,
            "plan_hash": score.plan_hash,
            "raw_objective": score.raw_objective,
            "comparator_key": list(score.comparator_key),
            "assumptions": list(score.assumptions),
        }
        summary_ref = put_json(
            context.artifact_store,
            summary,
            kind=ArtifactKind.JSON_SPECIFICATION,
            units="unitless",
            provenance=decision_provenance(
                self.spec.name,
                self.version,
                (problem_ref, plan_ref, score_ref),
                {"role": "plan_summary"},
            ),
            data_schema={"type": "PlanSummary", "version": self.version},
        )
        output = SummarizePlanOutput(
            summary_artifact_id=summary_ref.id,
            scorecard_artifact_id=score_ref.id,
            feasible=score.feasible,
            selected_site_count=len(plan.selected_site_ids),
            comparator_key=score.comparator_key,
        )
        compact_metrics: dict[str, JsonValue] = {
            name: value for name, value in score.overall_metrics.items()
        }
        compact_summary: dict[str, JsonValue] = {
            "summary": summary_ref.id,
            "scorecard": score_ref.id,
            "feasible": score.feasible,
            "selected_sites": len(plan.selected_site_ids),
            "overall_metrics": compact_metrics,
        }
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary=compact_summary,
            artifacts=(summary_ref, score_ref),
            metrics=output.model_dump(mode="json"),
        )


class MapFormat(StrEnum):
    GEOJSON = "geojson"
    SVG = "svg"


class RenderMapInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    plan_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    format: MapFormat = MapFormat.GEOJSON


class RenderMapOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    map_artifact_id: str
    format: MapFormat
    selected_site_count: int = Field(ge=0)


def _geojson(frame: Any, id_field: str, selected: set[str]) -> bytes:
    features = []
    for _, row in frame.sort_values(id_field, kind="stable").iterrows():
        identifier = str(row[id_field])
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {"site_id": identifier, "selected": identifier in selected},
            }
        )
    return canonical_json_bytes({"type": "FeatureCollection", "features": features})


def _svg(frame: Any, id_field: str, selected: set[str]) -> bytes:
    width, height, padding = 640.0, 480.0, 30.0
    min_x, min_y, max_x, max_y = (float(value) for value in frame.total_bounds)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    circles = []
    for _, row in frame.sort_values(id_field, kind="stable").iterrows():
        identifier = str(row[id_field])
        x = padding + (float(row.geometry.x) - min_x) / span_x * (width - 2 * padding)
        y = height - padding - (float(row.geometry.y) - min_y) / span_y * (height - 2 * padding)
        is_selected = identifier in selected
        circles.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{8 if is_selected else 4}" '
            f'fill="{"#b42318" if is_selected else "#667085"}">'
            f"<title>{html.escape(identifier)}</title></circle>"
        )
    payload = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        'role="img" aria-label="Location-allocation candidate map">'
        '<rect width="100%" height="100%" fill="#f8fafc"/>' + "".join(circles) + "</svg>"
    )
    return payload.encode("utf-8")


class RenderMapTool:
    """Render all candidate points and selected sites without a frontend dependency."""

    version = "1.0.0"
    spec = ToolSpec(
        name="render_map",
        version=version,
        description="Render a validated location plan as deterministic GeoJSON or standalone SVG.",
        input_schema=RenderMapInput.model_json_schema(),
        output_schema=RenderMapOutput.model_json_schema(),
        capability_tags=frozenset({"decision", "map", "location_allocation", "offline"}),
        problem_tags=frozenset({"location_allocation"}),
        artifact_tags=frozenset(
            {ArtifactKind.JSON_SPECIFICATION, ArtifactKind.PLAN, ArtifactKind.MAP}
        ),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=5, p95_ms=1_000),
        smoke_input={
            "problem_artifact_id": MISSING_ARTIFACT_ID,
            "plan_artifact_id": MISSING_ARTIFACT_ID,
            "format": "geojson",
        },
    )

    def __init__(self, registry: ProblemRegistry | None = None) -> None:
        self._registry = registry or create_problem_registry()

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = RenderMapInput.model_validate(arguments)
        problem_ref, problem = read_problem(context, request.problem_artifact_id)
        plan_ref, plan = read_plan(context, request.plan_artifact_id)
        plugin = self._registry.get(problem.type_id.value)
        problem_report = plugin.validate_spec(problem, context.artifact_store)
        if not problem_report.valid:
            invalid(f"invalid problem: {problem_report.issues[0].message}")
        report = plugin.validate_plan(problem, plan, context.artifact_store)
        if not report.valid:
            invalid(f"cannot render an invalid plan: {report.issues[0].message}")
        frame = read_vector(context.artifact_store, problem.candidates.artifact)
        selected = set(plan.selected_site_ids)
        content = (
            _geojson(frame, problem.candidates.candidate_id_field, selected)
            if request.format is MapFormat.GEOJSON
            else _svg(frame, problem.candidates.candidate_id_field, selected)
        )
        media_type = (
            "application/geo+json" if request.format is MapFormat.GEOJSON else "image/svg+xml"
        )
        min_x, min_y, max_x, max_y = (float(value) for value in frame.total_bounds)
        provenance = decision_provenance(
            self.spec.name,
            self.version,
            (problem_ref, plan_ref),
            {"format": request.format.value},
        )
        map_ref = context.artifact_store.put_bytes(
            content,
            ArtifactMetadata(
                kind=ArtifactKind.MAP,
                media_type=media_type,
                crs=str(frame.crs),
                units=problem.candidates.artifact.units,
                spatial_extent=SpatialExtent(west=min_x, south=min_y, east=max_x, north=max_y),
                data_schema={"type": request.format.value, "version": self.version},
                row_count=len(frame),
                source_uri=provenance.source_uri,
                source_provider=provenance.source_provider,
                license=provenance.license,
                source_version=provenance.source_version,
                lineage=provenance.lineage,
                quality=QualitySummary(),
                privacy=provenance.privacy,
            ),
        )
        output = RenderMapOutput(
            map_artifact_id=map_ref.id,
            format=request.format,
            selected_site_count=len(selected),
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "map": map_ref.id,
                "format": request.format.value,
                "selected_sites": len(selected),
            },
            artifacts=(map_ref,),
            metrics=output.model_dump(mode="json"),
        )
