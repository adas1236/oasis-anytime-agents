"""Small model-facing interfaces over the full evidence and decision implementations.

These tools only read caller-selected artifacts. They have no dataset, prompt parser,
case metadata, or oracle access. The advanced registry remains available separately.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Mapping
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from oasis.artifacts import MatrixData, put_json, put_matrix, put_vector, read_json, read_matrix
from oasis.problems.schemas import LocationAllocationProblem, SearchResumeToken
from oasis.schemas import (
    ArtifactKind,
    CandidateSpec,
    DemandSpec,
    ToolEvent,
    ToolEventKind,
    ToolResult,
)
from oasis.tools.decision.common import read_problem
from oasis.tools.decision.compile import CompileProblemInput, CompileProblemTool
from oasis.tools.decision.improve import ImproveTool
from oasis.tools.evidence.access import ServiceMatrixTool, TravelMatrixTool
from oasis.tools.evidence.common import (
    MISSING_ARTIFACT_ID,
    artifact_ref,
    child_provenance,
    invalid,
    read_frame,
    require_kind,
)
from oasis.tools.evidence.construction import BuildCandidatesTool, BuildDemandTool
from oasis.tools.evidence.locations import MaterializeLocationsTool
from oasis.tools.protocols import StreamingTool, Tool, ToolContext

ArtifactID = Annotated[str, Field(pattern=r"^sha256-[0-9a-f]{64}$")]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DemandInput(Arguments):
    artifact_id: ArtifactID
    need_field: str = Field(description="Numeric demand field from the evidence, e.g. population.")
    location_id_field: str = "id"


class CandidatesInput(Arguments):
    artifact_id: ArtifactID
    candidate_id_field: str = "id"


class LocationsInput(Arguments):
    resolution_artifact_id: ArtifactID
    provider_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    metadata_fields: tuple[str, ...] = Field(
        default=(), description="Scalar fields to copy from resolved candidates, e.g. population."
    )


class TravelInput(Arguments):
    origins_artifact_id: ArtifactID
    metric: Literal["haversine", "driving_distance", "driving_time"]
    destinations_artifact_id: ArtifactID | None = Field(
        default=None,
        description="Omit to travel between all origins. Points must have an id field.",
    )


class ServiceInput(Arguments):
    access_matrix_artifact_id: ArtifactID
    threshold: float = Field(ge=0, description="Inclusive coverage cutoff in the matrix's units.")


class FacilityInput(Arguments):
    demand: ArtifactID = Field(description="demand_spec_artifact_id returned by build_demand.")
    candidates: ArtifactID = Field(description="candidate_spec_artifact_id from build_candidates.")
    access_matrix: ArtifactID
    service_matrix: ArtifactID


class MaxCoverageInput(FacilityInput):
    site_limit: int = Field(ge=0)


class MinFacilitiesInput(FacilityInput):
    coverage_target: float = Field(ge=0, le=1, description="Fraction of demand, e.g. 0.9 for 90%.")


class TspInput(Arguments):
    nodes: ArtifactID = Field(description="Point artifact with an id field; every node is visited.")
    travel_matrix: ArtifactID
    depot: str = Field(
        min_length=1, description="Node id to start and finish at, from tool results."
    )


class SearchInput(Arguments):
    problem_artifact_id: ArtifactID
    strategy: Literal[
        "auto",
        "add_swap",
        "multi_swap",
        "local_assignment",
        "scenario_aware",
        "exact_enumeration",
        "ortools_cp_sat",
        "two_opt",
        "relocate",
        "swap",
        "ortools_routing",
    ] = "auto"
    max_candidates: int = Field(default=1_000, ge=1, le=1_000_000)
    resume_from: ArtifactID | None = Field(
        default=None,
        description="A returned resume_token_artifact_id to continue, or plan artifact to refine."
        " Auto preserves a resume token's strategy. Candidate limit does not bound OR-Tools;"
        " OR-Tools uses the remaining run deadline.",
    )


def _validate(model: type[BaseModel], arguments: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return model.model_validate(arguments).model_dump(exclude_none=True)
    except ValidationError as error:
        issues = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]
        invalid("; ".join(issues)[:1000])


class CompactTool:
    """Validate a small public schema and translate into the existing implementation."""

    def __init__(
        self,
        delegate: Tool,
        model: type[BaseModel],
        description: str,
        smoke_input: dict[str, Any],
        *,
        name: str | None = None,
    ) -> None:
        self.delegate = delegate
        self.model = model
        self.spec = delegate.spec.model_copy(
            update={
                "name": name or delegate.spec.name,
                "version": "2.0.0",
                "description": description,
                "input_schema": model.model_json_schema(),
                "smoke_input": smoke_input,
            }
        )

    async def translate(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        if self.spec.name == "build_demand":
            args["need_fields"] = [args.pop("need_field")]
        elif self.spec.name == "build_candidates":
            args["mode"] = "supplied"
        elif self.spec.name == "service_matrix":
            args["strategy"] = "binary_threshold"
        elif self.spec.name == "travel_matrix":
            metric = args.pop("metric")
            args.setdefault("destinations_artifact_id", args["origins_artifact_id"])
            if metric == "haversine":
                args.update(strategy="haversine", output_units="kilometers")
            else:
                args.update(
                    strategy="routed_provider",
                    routing_profile="driving",
                    route_annotation="distance" if metric == "driving_distance" else "duration",
                    output_units="meters" if metric == "driving_distance" else "seconds",
                )
        return args

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        args = await self.translate(_validate(self.model, arguments), context)
        result = await self.delegate.run(args, context)
        if self.spec.name != "travel_matrix" or result.metrics.get("units") != "meters":
            return result
        # Provider contracts use meters. Public distance tools consistently return km.
        parent = artifact_ref(context, str(result.metrics["artifact_id"]))
        matrix = read_matrix(context.artifact_store, parent)
        converted = put_matrix(
            context.artifact_store,
            MatrixData(
                values=matrix.values / 1000, row_ids=matrix.row_ids, column_ids=matrix.column_ids
            ),
            crs=None,
            units="kilometers",
            provenance=child_provenance(self.spec.name, self.spec.version, [parent], arguments),
        )
        return result.model_copy(
            update={
                "artifacts": (converted,),
                "summary": "Directed driving distances in kilometers.",
                "metrics": {**result.metrics, "artifact_id": converted.id, "units": "kilometers"},
            }
        )


class CompileTool(CompactTool):
    async def translate(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        if self.spec.name == "compile_tsp":
            reference = artifact_ref(context, args["travel_matrix"])
            require_kind(reference, {ArtifactKind.MATRIX})
            matrix = read_matrix(context.artifact_store, reference)
            if not reference.units:
                invalid("travel matrix must declare its units")
            finite = matrix.values[np.isfinite(matrix.values)]
            if finite.size == 0 or (finite < 0).any():
                invalid("travel matrix must contain nonnegative finite travel costs")
            # Any simple tour uses at most N edges. This bound adds no route constraint.
            cap = max(1.0, (len(matrix.row_ids) + 1) * float(finite.max()))
            if not math.isfinite(cap):
                invalid("travel costs are too large to construct a finite tour bound")
            return {
                "type_id": "tsp",
                "nodes_artifact_id": args["nodes"],
                "node_id_field": "id",
                "travel_matrix_artifact_ids": {"base": reference.id},
                "policy": {
                    "depot_ids": [args["depot"]],
                    "vehicle_count": 1,
                    "require_return": True,
                    "shift_length": cap,
                    "time_units": reference.units,
                },
            }
        demand_ref = artifact_ref(context, args["demand"])
        candidate_ref = artifact_ref(context, args["candidates"])
        require_kind(demand_ref, {ArtifactKind.JSON_SPECIFICATION})
        require_kind(candidate_ref, {ArtifactKind.JSON_SPECIFICATION})
        try:
            demand = DemandSpec.model_validate(read_json(context.artifact_store, demand_ref))
            candidates = CandidateSpec.model_validate(
                read_json(context.artifact_store, candidate_ref)
            )
        except ValidationError:
            invalid(
                "demand/candidates must be specification IDs from build_demand/build_candidates"
            )
        if len(demand.need_fields) != 1:
            invalid("demand must have one need field; use build_demand to select it explicitly")
        minimum = self.spec.name == "compile_min_facilities"
        if minimum:
            # Counting facilities explicitly means unit cost, not inferred economic cost.
            frame = read_frame(context, candidates.artifact).copy()
            frame["_facility_count_cost"] = 1.0
            points = put_vector(
                context.artifact_store,
                frame,
                units=candidates.artifact.units or "unitless",
                provenance=child_provenance(
                    self.spec.name, self.spec.version, [candidate_ref, candidates.artifact], args
                ),
            )
            candidates = candidates.model_copy(
                update={
                    "artifact": points,
                    "opening_cost_field": "_facility_count_cost",
                }
            )
            candidate_ref = put_json(
                context.artifact_store,
                candidates.model_dump(mode="json"),
                kind=ArtifactKind.JSON_SPECIFICATION,
                units="unitless",
                data_schema={"type": "CandidateSpec", "version": candidates.schema_version},
                provenance=child_provenance(
                    self.spec.name, self.spec.version, [candidate_ref, points], args
                ),
            )
            policy = {"coverage_target": args["coverage_target"], "site_limit": len(frame)}
        else:
            policy = {"site_limit": args["site_limit"]}
        return {
            "type_id": "min_cost_target_coverage" if minimum else "max_weighted_coverage",
            "demand_spec_artifact_id": demand_ref.id,
            "candidate_spec_artifact_id": candidate_ref.id,
            "access_matrix_artifact_id": args["access_matrix"],
            "service_matrix_artifact_ids": {"base": args["service_matrix"]},
            "need_field": demand.need_fields[0],
            "policy": policy,
        }

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        args = await self.translate(_validate(self.model, arguments), context)
        # Conditional validation errors are argument errors, not adapter/internal failures.
        _validate(CompileProblemInput, args)
        return await self.delegate.run(args, context)


class CompactImproveTool:
    def __init__(self) -> None:
        self.delegate = ImproveTool()
        self.spec = self.delegate.spec.model_copy(
            update={
                "version": "2.0.0",
                "input_schema": SearchInput.model_json_schema(),
                "description": "Improve a compiled problem, streaming verified plans. Auto chooses "
                "add_swap for facilities or two_opt for routes. Resume with an artifact ID; "
                "never copy a plan or search-state object into arguments.",
                "smoke_input": {"problem_artifact_id": MISSING_ARTIFACT_ID},
            }
        )

    async def stream(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> AsyncIterator[ToolEvent]:
        args = _validate(SearchInput, arguments)
        _, problem = read_problem(context, args["problem_artifact_id"])
        resume = args.pop("resume_from", None)
        token = None
        if resume is not None:
            reference = artifact_ref(context, resume)
            if reference.kind is ArtifactKind.PLAN:
                args["starting_plan_artifact_id"] = resume
            else:
                try:
                    token = SearchResumeToken.model_validate(
                        read_json(context.artifact_store, reference)
                    )
                except (ValueError, TypeError):
                    invalid("resume_from must be a plan or returned resume_token_artifact_id")
                args["resume_token_artifact_id"] = resume
        facility = isinstance(problem, LocationAllocationProblem)
        if args["strategy"] == "auto":
            args["strategy"] = (
                token.strategy.value if token else ("add_swap" if facility else "two_opt")
            )
        allowed = (
            {
                "add_swap",
                "multi_swap",
                "local_assignment",
                "scenario_aware",
                "exact_enumeration",
                "ortools_cp_sat",
            }
            if facility
            else {"two_opt", "relocate", "swap", "exact_enumeration", "ortools_routing"}
        )
        if args["strategy"] not in allowed:
            invalid(
                "strategy is incompatible with this problem; use auto or "
                + ", ".join(sorted(allowed))
            )
        async for event in self.delegate.stream(args, context):
            yield event

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        async for event in self.stream(arguments, context):
            if event.kind is ToolEventKind.RESULT and event.result is not None:
                return event.result
        raise RuntimeError("improvement stream ended without a result")


def public_tools(advanced: tuple[Tool | StreamingTool, ...]) -> tuple[Tool | StreamingTool, ...]:
    """Retain general evidence tools; replace verbose decision/access interfaces."""
    missing = MISSING_ARTIFACT_ID
    facility = {
        "demand": missing,
        "candidates": missing,
        "access_matrix": missing,
        "service_matrix": missing,
    }
    replacements: tuple[Tool | StreamingTool, ...] = (
        CompactTool(
            BuildDemandTool(),
            DemandInput,
            "Select one numeric demand field from evidence; preserve its values without "
            "hidden weighting.",
            {"artifact_id": missing, "need_field": "population"},
        ),
        CompactTool(
            BuildCandidatesTool(),
            CandidatesInput,
            "Use supplied points as candidate facilities. No locations are added or inferred.",
            {"artifact_id": missing},
        ),
        CompactTool(
            MaterializeLocationsTool(),
            LocationsInput,
            "Select explicit provider_ids from a resolution artifact. Copy requested scalar "
            "metadata (e.g. population). Returns points with id and name; never auto-selects "
            "ambiguous matches.",
            {"resolution_artifact_id": missing, "provider_ids": ["example"]},
        ),
        CompactTool(
            TravelMatrixTool(),
            TravelInput,
            "Travel between selected points with id fields. Haversine is spherical straight-line "
            "distance; driving uses the provider's directed roads. Distances are kilometers; "
            "driving_time is seconds. Omit destinations for an all-pairs matrix.",
            {"origins_artifact_id": missing, "metric": "haversine"},
        ),
        CompactTool(
            ServiceMatrixTool(),
            ServiceInput,
            "Mark demand reachable when travel cost is at most threshold. Use the matrix's units "
            "(kilometers for distance, seconds for driving_time).",
            {"access_matrix_artifact_id": missing, "threshold": 10},
        ),
        CompileTool(
            CompileProblemTool(),
            MaxCoverageInput,
            "Maximize covered demand using at most site_limit facilities. Supply demand/candidate "
            "specification IDs and one access/service matrix. Returns a verified initial plan "
            "and problem ID.",
            {**facility, "site_limit": 1},
            name="compile_max_coverage",
        ),
        CompileTool(
            CompileProblemTool(),
            MinFacilitiesInput,
            "Minimize the NUMBER of facilities covering at least coverage_target fraction of "
            "demand. All sites count equally, irrespective of financial opening costs. Supply "
            "specification IDs and one access/service matrix.",
            {**facility, "coverage_target": 0.9},
            name="compile_min_facilities",
        ),
        CompileTool(
            CompileProblemTool(),
            TspInput,
            "Shortest single-vehicle closed tour visiting every supplied node once, starting "
            "and ending at depot. Uses the matrix's costs and directionality. No capacity, "
            "time windows, or shift constraint.",
            {"nodes": missing, "travel_matrix": missing, "depot": "example"},
            name="compile_tsp",
        ),
        CompactImproveTool(),
    )
    replaced = {tool.spec.name for tool in replacements} | {"compile_problem", "scenario_sweep"}
    return (*[tool for tool in advanced if tool.spec.name not in replaced], *replacements)
