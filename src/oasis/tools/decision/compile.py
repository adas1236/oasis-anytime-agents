"""Compile immutable evidence and policy into an admitted decision problem."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oasis.artifacts import put_json, read_json
from oasis.problems.location_allocation import problem_hashes
from oasis.problems.protocols import Deadline
from oasis.problems.registry import ProblemRegistry, create_builtin_problem_registry
from oasis.problems.routing import route_problem_hashes
from oasis.problems.schemas import (
    EquityGroup,
    LocationAllocationPolicy,
    LocationAllocationProblem,
    LocationProblemType,
    RouteProblemType,
    RouteScenario,
    RouteServicePolicy,
    RouteServiceProblem,
    ServiceScenario,
)
from oasis.schemas import (
    ArtifactKind,
    ArtifactRef,
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
    """Select the compilation inputs and policy using type_id.

    Location types require demand_spec_artifact_id, candidate_spec_artifact_id,
    access_matrix_artifact_id, nonempty service_matrix_artifact_ids, need_field,
    and a LocationAllocationPolicy. Route types (tsp, orienteering,
    mobile_service_route) require nodes_artifact_id, node_id_field, nonempty
    travel_matrix_artifact_ids, and a RouteServicePolicy including depot_ids,
    shift_length, and time_units. Supply artifact IDs returned by prior tools.
    """

    model_config = ConfigDict(frozen=True)

    type_id: LocationProblemType | RouteProblemType

    demand_spec_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    candidate_spec_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    access_matrix_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    service_matrix_artifact_ids: dict[str, str] = Field(default_factory=dict)
    access_scenario_artifact_ids: dict[str, str] = Field(default_factory=dict)
    demand_multiplier_artifact_ids: dict[str, str] = Field(default_factory=dict)
    failed_site_ids_by_scenario: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    service_scenario_weights: dict[str, float] = Field(default_factory=dict)
    need_field: str | None = None
    groups: tuple[EquityGroup, ...] = ()

    nodes_artifact_id: str | None = Field(default=None, pattern=r"^sha256-[0-9a-f]{64}$")
    node_id_field: str | None = None
    prize_field: str | None = None
    demand_field: str | None = None
    service_time_field: str | None = None
    window_start_field: str | None = None
    window_end_field: str | None = None
    travel_matrix_artifact_ids: dict[str, str] = Field(default_factory=dict)
    travel_demand_multiplier_artifact_ids: dict[str, str] = Field(default_factory=dict)
    travel_scenario_weights: dict[str, float] = Field(default_factory=dict)

    policy: LocationAllocationPolicy | RouteServicePolicy

    @model_validator(mode="after")
    def family_fields_are_complete(self) -> Self:
        if isinstance(self.type_id, LocationProblemType):
            if not isinstance(self.policy, LocationAllocationPolicy):
                raise ValueError("location problem types require a location-allocation policy")
            required = {
                "demand_spec_artifact_id": self.demand_spec_artifact_id,
                "candidate_spec_artifact_id": self.candidate_spec_artifact_id,
                "access_matrix_artifact_id": self.access_matrix_artifact_id,
                "need_field": self.need_field,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if not self.service_matrix_artifact_ids:
                missing.append("service_matrix_artifact_ids")
            if missing:
                raise ValueError("location compilation is missing: " + ", ".join(missing))
        else:
            if not isinstance(self.policy, RouteServicePolicy):
                raise ValueError("route problem types require a route-service policy")
            required = {
                "nodes_artifact_id": self.nodes_artifact_id,
                "node_id_field": self.node_id_field,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if not self.travel_matrix_artifact_ids:
                missing.append("travel_matrix_artifact_ids")
            if missing:
                raise ValueError("route compilation is missing: " + ", ".join(missing))
        return self


class CompileProblemOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_artifact_id: str | None = None
    baseline_plan_artifact_id: str | None = None
    baseline_scorecard_artifact_id: str | None = None
    problem_hash: str | None = None
    feasible: bool
    issue_codes: tuple[str, ...] = ()


def _scenario_keys_are_valid(request: CompileProblemInput, names: set[str]) -> None:
    mappings = (
        ("access_scenario_artifact_ids", set(request.access_scenario_artifact_ids)),
        ("demand_multiplier_artifact_ids", set(request.demand_multiplier_artifact_ids)),
        ("failed_site_ids_by_scenario", set(request.failed_site_ids_by_scenario)),
    )
    for label, values in mappings:
        if not set(values) <= names:
            invalid(f"{label} may name only declared service scenarios")
    if request.service_scenario_weights and set(request.service_scenario_weights) != names:
        invalid("service_scenario_weights must name every service matrix exactly")


def _compile_location(
    request: CompileProblemInput, context: ToolContext
) -> tuple[LocationAllocationProblem, tuple[ArtifactRef, ...]]:
    assert request.demand_spec_artifact_id is not None
    assert request.candidate_spec_artifact_id is not None
    assert request.access_matrix_artifact_id is not None
    assert request.need_field is not None
    assert isinstance(request.type_id, LocationProblemType)
    assert isinstance(request.policy, LocationAllocationPolicy)
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
    names = set(request.service_matrix_artifact_ids)
    _scenario_keys_are_valid(request, names)
    if (
        request.service_scenario_weights
        and request.policy.scenario_weights
        and request.service_scenario_weights != request.policy.scenario_weights
    ):
        invalid("scenario weights declared in two places must agree exactly")
    parents: list[ArtifactRef] = [demand_spec_ref, candidate_spec_ref, access_ref]
    scenarios: list[ServiceScenario] = []
    for name, service_id in sorted(request.service_matrix_artifact_ids.items()):
        service_ref = artifact_ref(context, service_id)
        require_kind(service_ref, {ArtifactKind.MATRIX})
        parents.append(service_ref)
        scenario_access = None
        if name in request.access_scenario_artifact_ids:
            scenario_access = artifact_ref(context, request.access_scenario_artifact_ids[name])
            require_kind(scenario_access, {ArtifactKind.MATRIX})
            parents.append(scenario_access)
        multiplier = None
        if name in request.demand_multiplier_artifact_ids:
            multiplier = artifact_ref(context, request.demand_multiplier_artifact_ids[name])
            require_kind(multiplier, {ArtifactKind.MATRIX})
            parents.append(multiplier)
        scenarios.append(
            ServiceScenario(
                name=name,
                service_matrix=service_ref,
                access_matrix=scenario_access,
                demand_multiplier=multiplier,
                failed_site_ids=request.failed_site_ids_by_scenario.get(name, ()),
                weight=request.service_scenario_weights.get(name, 1.0),
            )
        )
    blank = "0" * 64
    problem = LocationAllocationProblem(
        type_id=request.type_id,
        demand=demand,
        candidates=candidates,
        access_matrix=access_ref,
        service_scenarios=tuple(scenarios),
        need_field=request.need_field,
        groups=request.groups,
        policy=request.policy,
        evidence_hash=blank,
        policy_hash=blank,
        problem_hash=blank,
    )
    evidence_hash, policy_hash, complete_hash = problem_hashes(problem)
    return (
        problem.model_copy(
            update={
                "evidence_hash": evidence_hash,
                "policy_hash": policy_hash,
                "problem_hash": complete_hash,
            }
        ),
        tuple(parents),
    )


def _compile_route(
    request: CompileProblemInput, context: ToolContext
) -> tuple[RouteServiceProblem, tuple[ArtifactRef, ...]]:
    assert request.nodes_artifact_id is not None
    assert request.node_id_field is not None
    assert isinstance(request.type_id, RouteProblemType)
    assert isinstance(request.policy, RouteServicePolicy)
    nodes_ref = artifact_ref(context, request.nodes_artifact_id)
    require_kind(nodes_ref, {ArtifactKind.VECTOR, ArtifactKind.TABLE})
    names = set(request.travel_matrix_artifact_ids)
    if request.travel_scenario_weights and set(request.travel_scenario_weights) != names:
        invalid("travel_scenario_weights must name every travel matrix exactly")
    if not set(request.travel_demand_multiplier_artifact_ids) <= names:
        invalid("travel demand multipliers may name only declared travel scenarios")
    if (
        request.travel_scenario_weights
        and request.policy.scenario_weights
        and request.travel_scenario_weights != request.policy.scenario_weights
    ):
        invalid("scenario weights declared in two places must agree exactly")
    parents: list[ArtifactRef] = [nodes_ref]
    scenarios: list[RouteScenario] = []
    for name, travel_id in sorted(request.travel_matrix_artifact_ids.items()):
        travel_ref = artifact_ref(context, travel_id)
        require_kind(travel_ref, {ArtifactKind.MATRIX})
        parents.append(travel_ref)
        multiplier = None
        if name in request.travel_demand_multiplier_artifact_ids:
            multiplier = artifact_ref(context, request.travel_demand_multiplier_artifact_ids[name])
            require_kind(multiplier, {ArtifactKind.MATRIX})
            parents.append(multiplier)
        scenarios.append(
            RouteScenario(
                name=name,
                travel_matrix=travel_ref,
                demand_multiplier=multiplier,
                weight=request.travel_scenario_weights.get(name, 1.0),
            )
        )
    blank = "0" * 64
    problem = RouteServiceProblem(
        type_id=request.type_id,
        nodes=nodes_ref,
        node_id_field=request.node_id_field,
        prize_field=request.prize_field,
        demand_field=request.demand_field,
        service_time_field=request.service_time_field,
        window_start_field=request.window_start_field,
        window_end_field=request.window_end_field,
        travel_scenarios=tuple(scenarios),
        policy=request.policy,
        evidence_hash=blank,
        policy_hash=blank,
        problem_hash=blank,
    )
    evidence_hash, policy_hash, complete_hash = route_problem_hashes(problem)
    return (
        problem.model_copy(
            update={
                "evidence_hash": evidence_hash,
                "policy_hash": policy_hash,
                "problem_hash": complete_hash,
            }
        ),
        tuple(parents),
    )


class CompileProblemTool:
    """Validate evidence/policy and commit a problem plus independently scored baseline."""

    version = "1.1.0"
    spec = ToolSpec(
        name="compile_problem",
        version=version,
        description=(
            "Compile immutable location-allocation or route-service evidence and policy into a "
            "validated problem with an independently scored feasible baseline."
        ),
        input_schema=CompileProblemInput.model_json_schema(),
        output_schema=CompileProblemOutput.model_json_schema(),
        capability_tags=frozenset(
            {"decision", "compile", "location_allocation", "routing", "offline"}
        ),
        problem_tags=frozenset({"location_allocation", "routing"}),
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
        self._registry = registry or create_builtin_problem_registry()

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = CompileProblemInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        problem, parents = (
            _compile_location(request, context)
            if isinstance(request.type_id, LocationProblemType)
            else _compile_route(request, context)
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
            data_schema={"type": type(problem).__name__, "version": problem.schema_version},
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
