"""Compile immutable evidence and policy into an admitted location problem."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from oasis.artifacts import put_json, read_json
from oasis.problems.location_allocation import create_problem_registry, problem_hashes
from oasis.problems.protocols import Deadline
from oasis.problems.registry import ProblemRegistry
from oasis.problems.schemas import (
    EquityGroup,
    LocationAllocationPolicy,
    LocationAllocationProblem,
    LocationProblemType,
    ServiceScenario,
)
from oasis.schemas import (
    ArtifactKind,
    CandidateSpec,
    DemandSpec,
    DeterminismClassification,
    SideEffectClassification,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.decision.common import decision_provenance, put_plan_and_scorecard
from oasis.tools.evidence.common import MISSING_ARTIFACT_ID, artifact_ref, invalid, require_kind
from oasis.tools.protocols import ToolContext


class CompileProblemInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    type_id: LocationProblemType
    demand_spec_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    candidate_spec_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    access_matrix_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    service_matrix_artifact_ids: dict[str, str] = Field(min_length=1)
    service_scenario_weights: dict[str, float] = Field(default_factory=dict)
    need_field: str = Field(min_length=1)
    groups: tuple[EquityGroup, ...] = ()
    policy: LocationAllocationPolicy


class CompileProblemOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_artifact_id: str | None = None
    baseline_plan_artifact_id: str | None = None
    baseline_scorecard_artifact_id: str | None = None
    problem_hash: str | None = None
    feasible: bool
    issue_codes: tuple[str, ...] = ()


class CompileProblemTool:
    """Validate dimensions/units/policy and commit a problem plus feasible baseline."""

    version = "1.0.0"
    spec = ToolSpec(
        name="compile_problem",
        version=version,
        description=(
            "Compile immutable demand, candidate, access, service, and policy evidence into a "
            "validated location-allocation problem and independently scored feasible baseline."
        ),
        input_schema=CompileProblemInput.model_json_schema(),
        output_schema=CompileProblemOutput.model_json_schema(),
        capability_tags=frozenset({"decision", "compile", "location_allocation", "offline"}),
        problem_tags=frozenset({"location_allocation"}),
        artifact_tags=frozenset(
            {ArtifactKind.JSON_SPECIFICATION, ArtifactKind.MATRIX, ArtifactKind.PLAN}
        ),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=20, p95_ms=2_000),
        smoke_input={
            "type_id": "max_weighted_coverage",
            "demand_spec_artifact_id": MISSING_ARTIFACT_ID,
            "candidate_spec_artifact_id": MISSING_ARTIFACT_ID,
            "access_matrix_artifact_id": MISSING_ARTIFACT_ID,
            "service_matrix_artifact_ids": {"base": MISSING_ARTIFACT_ID},
            "need_field": "need",
            "policy": {"site_limit": 1},
        },
    )

    def __init__(self, registry: ProblemRegistry | None = None) -> None:
        self._registry = registry or create_problem_registry()

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = CompileProblemInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        demand_spec_ref = artifact_ref(context, request.demand_spec_artifact_id)
        candidate_spec_ref = artifact_ref(context, request.candidate_spec_artifact_id)
        access_ref = artifact_ref(context, request.access_matrix_artifact_id)
        require_kind(demand_spec_ref, {ArtifactKind.JSON_SPECIFICATION})
        require_kind(candidate_spec_ref, {ArtifactKind.JSON_SPECIFICATION})
        require_kind(access_ref, {ArtifactKind.MATRIX})
        try:
            demand = DemandSpec.model_validate(read_json(context.artifact_store, demand_spec_ref))
            candidates = CandidateSpec.model_validate(
                read_json(context.artifact_store, candidate_spec_ref)
            )
        except ValueError as error:
            invalid(f"invalid evidence specification: {error}")
        scenarios: list[ServiceScenario] = []
        service_refs = []
        if request.service_scenario_weights and set(request.service_scenario_weights) != set(
            request.service_matrix_artifact_ids
        ):
            invalid("service_scenario_weights must name every service matrix exactly")
        if (
            request.service_scenario_weights
            and request.policy.scenario_weights
            and request.service_scenario_weights != request.policy.scenario_weights
        ):
            invalid("scenario weights declared in two places must agree exactly")
        for name, artifact_id in sorted(request.service_matrix_artifact_ids.items()):
            reference = artifact_ref(context, artifact_id)
            require_kind(reference, {ArtifactKind.MATRIX})
            service_refs.append(reference)
            scenarios.append(
                ServiceScenario(
                    name=name,
                    service_matrix=reference,
                    weight=request.service_scenario_weights.get(name, 1.0),
                )
            )
        placeholder = "0" * 64
        problem = LocationAllocationProblem(
            type_id=request.type_id,
            demand=demand,
            candidates=candidates,
            access_matrix=access_ref,
            service_scenarios=tuple(scenarios),
            need_field=request.need_field,
            groups=request.groups,
            policy=request.policy,
            evidence_hash=placeholder,
            policy_hash=placeholder,
            problem_hash=placeholder,
        )
        evidence_hash, policy_hash, complete_hash = problem_hashes(problem)
        problem = problem.model_copy(
            update={
                "evidence_hash": evidence_hash,
                "policy_hash": policy_hash,
                "problem_hash": complete_hash,
            }
        )
        plugin = self._registry.get(problem.type_id.value)
        report = plugin.validate_spec(problem, context.artifact_store)
        if not report.valid:
            output = CompileProblemOutput(
                feasible=False, issue_codes=tuple(issue.code for issue in report.issues)
            )
            return ToolResult(
                status=ToolResultStatus.INFEASIBLE,
                summary={
                    "feasible": False,
                    "issues": [issue.model_dump(mode="json") for issue in report.issues],
                },
                metrics=output.model_dump(mode="json"),
            )
        parents = (demand_spec_ref, candidate_spec_ref, access_ref, *service_refs)
        problem_ref = put_json(
            context.artifact_store,
            problem.model_dump(mode="json"),
            kind=ArtifactKind.JSON_SPECIFICATION,
            units="unitless",
            provenance=decision_provenance(
                self.spec.name,
                self.version,
                parents,
                {"type_id": problem.type_id.value, "problem_hash": problem.problem_hash},
            ),
            data_schema={
                "type": "LocationAllocationProblem",
                "version": problem.schema_version,
            },
        )
        try:
            baseline = plugin.make_baseline(
                problem,
                context.artifact_store,
                Deadline(context.deadline_monotonic, context.monotonic),
            )
        except ValueError as error:
            output = CompileProblemOutput(
                problem_artifact_id=problem_ref.id,
                problem_hash=problem.problem_hash,
                feasible=False,
                issue_codes=("no_feasible_baseline",),
            )
            return ToolResult(
                status=ToolResultStatus.INFEASIBLE,
                summary={"feasible": False, "issue": str(error)},
                artifacts=(problem_ref,),
                metrics=output.model_dump(mode="json"),
            )
        baseline_score = plugin.measure(problem, baseline, context.artifact_store)
        baseline_ref, score_ref = put_plan_and_scorecard(
            context,
            baseline,
            baseline_score,
            name=self.spec.name,
            version=self.version,
            parents=(problem_ref,),
            parameters={"role": "baseline", "problem_hash": problem.problem_hash},
        )
        output = CompileProblemOutput(
            problem_artifact_id=problem_ref.id,
            baseline_plan_artifact_id=baseline_ref.id,
            baseline_scorecard_artifact_id=score_ref.id,
            problem_hash=problem.problem_hash,
            feasible=True,
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "problem": problem_ref.id,
                "problem_hash": problem.problem_hash,
                "baseline_plan": baseline_ref.id,
                "baseline_scorecard": score_ref.id,
                "comparator_key": list(baseline_score.comparator_key),
            },
            artifacts=(problem_ref, baseline_ref, score_ref),
            metrics=output.model_dump(mode="json"),
            candidate=baseline,
        )
