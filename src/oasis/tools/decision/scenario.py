"""Immutable policy-variant sweeps over registered decision problems."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oasis.artifacts import put_json
from oasis.problems.location_allocation import problem_hashes
from oasis.problems.protocols import Deadline
from oasis.problems.registry import ProblemRegistry, create_builtin_problem_registry
from oasis.problems.routing import route_problem_hashes
from oasis.problems.schemas import (
    LocationAllocationPolicy,
    LocationAllocationProblem,
    RouteServicePolicy,
    RouteServiceProblem,
)
from oasis.schemas import (
    ArtifactKind,
    DeterminismClassification,
    SideEffectClassification,
    ToolResult,
    ToolResultStatus,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools.decision.common import (
    decision_provenance,
    put_plan_and_scorecard,
    read_problem,
)
from oasis.tools.evidence.common import MISSING_ARTIFACT_ID, invalid
from oasis.tools.protocols import ToolContext


class PolicyVariant(BaseModel):
    """One named complete policy; policy fields are never merged implicitly."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$")
    policy: LocationAllocationPolicy | RouteServicePolicy


class ScenarioSweepInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    variants: tuple[PolicyVariant, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def names_are_unique(self) -> Self:
        names = tuple(variant.name for variant in self.variants)
        if len(names) != len(set(names)):
            raise ValueError("policy variant names must be unique")
        return self


class ScenarioSweepRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    problem_hash: str
    problem_artifact_id: str
    feasible: bool
    baseline_plan_artifact_id: str | None = None
    baseline_scorecard_artifact_id: str | None = None
    issue_codes: tuple[str, ...] = ()


class ScenarioSweepOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_problem_hash: str
    variants: tuple[ScenarioSweepRecord, ...]


def _variant_problem(
    problem: LocationAllocationProblem | RouteServiceProblem,
    variant: PolicyVariant,
) -> LocationAllocationProblem | RouteServiceProblem:
    if isinstance(problem, LocationAllocationProblem):
        if not isinstance(variant.policy, LocationAllocationPolicy):
            invalid("location problems require location-allocation policy variants")
        changed_location = problem.model_copy(update={"policy": variant.policy})
        evidence_hash, policy_hash, complete_hash = problem_hashes(changed_location)
        return changed_location.model_copy(
            update={
                "evidence_hash": evidence_hash,
                "policy_hash": policy_hash,
                "problem_hash": complete_hash,
            }
        )
    else:
        if not isinstance(variant.policy, RouteServicePolicy):
            invalid("route problems require route-service policy variants")
        changed_route = problem.model_copy(update={"policy": variant.policy})
        evidence_hash, policy_hash, complete_hash = route_problem_hashes(changed_route)
    return changed_route.model_copy(
        update={
            "evidence_hash": evidence_hash,
            "policy_hash": policy_hash,
            "problem_hash": complete_hash,
        }
    )


class ScenarioSweepTool:
    """Compile and baseline explicit policies as isolated immutable problem variants."""

    version = "1.0.0"
    spec = ToolSpec(
        name="scenario_sweep",
        version=version,
        description=(
            "Evaluate explicit complete policy variants as separate immutable problems and "
            "independently scored feasible baselines."
        ),
        input_schema=ScenarioSweepInput.model_json_schema(),
        output_schema=ScenarioSweepOutput.model_json_schema(),
        capability_tags=frozenset(
            {"decision", "scenario", "policy", "location_allocation", "routing", "offline"}
        ),
        problem_tags=frozenset({"location_allocation", "routing"}),
        artifact_tags=frozenset(
            {ArtifactKind.JSON_SPECIFICATION, ArtifactKind.PLAN, ArtifactKind.SCORECARD}
        ),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=20, p95_ms=5_000),
        smoke_input={
            "problem_artifact_id": MISSING_ARTIFACT_ID,
            "variants": [{"name": "alternative", "policy": {"site_limit": 1}}],
        },
    )

    def __init__(self, registry: ProblemRegistry | None = None) -> None:
        self._registry = registry or create_builtin_problem_registry()

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = ScenarioSweepInput.model_validate(arguments)
        source_ref, source = read_problem(context, request.problem_artifact_id)
        plugin = self._registry.get(source.type_id.value)
        records: list[ScenarioSweepRecord] = []
        artifacts = []
        seen_hashes = {source.problem_hash}
        for variant in request.variants:
            context.cancellation.raise_if_cancelled()
            problem = _variant_problem(source, variant)
            if problem.problem_hash in seen_hashes:
                invalid("every policy variant must create a distinct immutable problem hash")
            seen_hashes.add(problem.problem_hash)
            problem_ref = put_json(
                context.artifact_store,
                problem.model_dump(mode="json"),
                kind=ArtifactKind.JSON_SPECIFICATION,
                units="unitless",
                provenance=decision_provenance(
                    self.spec.name,
                    self.version,
                    (source_ref,),
                    {"variant": variant.name, "problem_hash": problem.problem_hash},
                ),
                data_schema={"type": type(problem).__name__, "version": problem.schema_version},
            )
            artifacts.append(problem_ref)
            report = plugin.validate_spec(problem, context.artifact_store)
            if not report.valid:
                records.append(
                    ScenarioSweepRecord(
                        name=variant.name,
                        problem_hash=problem.problem_hash,
                        problem_artifact_id=problem_ref.id,
                        feasible=False,
                        issue_codes=tuple(issue.code for issue in report.issues),
                    )
                )
                continue
            try:
                baseline = plugin.make_baseline(
                    problem,
                    context.artifact_store,
                    Deadline(context.deadline_monotonic, context.monotonic),
                )
            except ValueError:
                records.append(
                    ScenarioSweepRecord(
                        name=variant.name,
                        problem_hash=problem.problem_hash,
                        problem_artifact_id=problem_ref.id,
                        feasible=False,
                        issue_codes=("no_feasible_baseline",),
                    )
                )
                continue
            score = plugin.measure(problem, baseline, context.artifact_store)
            plan_ref, score_ref = put_plan_and_scorecard(
                context,
                baseline,
                score,
                name=self.spec.name,
                version=self.version,
                parents=(problem_ref,),
                parameters={"variant": variant.name, "problem_hash": problem.problem_hash},
            )
            artifacts.extend((plan_ref, score_ref))
            records.append(
                ScenarioSweepRecord(
                    name=variant.name,
                    problem_hash=problem.problem_hash,
                    problem_artifact_id=problem_ref.id,
                    feasible=True,
                    baseline_plan_artifact_id=plan_ref.id,
                    baseline_scorecard_artifact_id=score_ref.id,
                )
            )
        output = ScenarioSweepOutput(
            source_problem_hash=source.problem_hash,
            variants=tuple(records),
        )
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary={
                "source_problem_hash": source.problem_hash,
                "variants": [
                    {
                        "name": record.name,
                        "problem_hash": record.problem_hash,
                        "feasible": record.feasible,
                    }
                    for record in records
                ],
            },
            artifacts=tuple(artifacts),
            metrics=output.model_dump(mode="json"),
        )
