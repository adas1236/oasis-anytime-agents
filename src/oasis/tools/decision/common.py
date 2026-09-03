"""Shared artifact helpers for deterministic decision tools."""

from __future__ import annotations

from pydantic import JsonValue

from oasis.artifacts import ArtifactProvenance, put_json, read_json
from oasis.problems.schemas import LocationAllocationProblem, Scorecard
from oasis.schemas import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactRef,
    ArtifactTransformation,
    Plan,
    PrivacyClassification,
)
from oasis.tools.evidence.common import artifact_ref, invalid, require_kind
from oasis.tools.protocols import ToolContext


def read_problem(
    context: ToolContext, artifact_id: str
) -> tuple[ArtifactRef, LocationAllocationProblem]:
    reference = artifact_ref(context, artifact_id)
    require_kind(reference, {ArtifactKind.JSON_SPECIFICATION})
    try:
        problem = LocationAllocationProblem.model_validate(
            read_json(context.artifact_store, reference)
        )
    except ValueError as error:
        invalid(f"invalid LocationAllocationProblem artifact: {error}")
    return reference, problem


def read_plan(context: ToolContext, artifact_id: str) -> tuple[ArtifactRef, Plan]:
    reference = artifact_ref(context, artifact_id)
    require_kind(reference, {ArtifactKind.PLAN})
    try:
        plan = Plan.model_validate(read_json(context.artifact_store, reference))
    except ValueError as error:
        invalid(f"invalid Plan artifact: {error}")
    return reference, plan


def decision_provenance(
    name: str,
    version: str,
    parents: tuple[ArtifactRef, ...],
    parameters: dict[str, JsonValue],
) -> ArtifactProvenance:
    privacy_order = {
        PrivacyClassification.PUBLIC: 0,
        PrivacyClassification.INTERNAL: 1,
        PrivacyClassification.SENSITIVE: 2,
        PrivacyClassification.RESTRICTED: 3,
    }
    privacy = max((parent.privacy for parent in parents), key=privacy_order.__getitem__)
    licenses = " AND ".join(sorted({parent.license or "unknown" for parent in parents}))
    return ArtifactProvenance(
        source_uri=f"oasis://{name}/{version}",
        source_provider="oasis-decision",
        source_version=version,
        license=licenses,
        privacy=privacy,
        lineage=ArtifactLineage(
            parent_ids=tuple(parent.id for parent in parents),
            transformations=(
                ArtifactTransformation(name=name, version=version, parameters=parameters),
            ),
        ),
    )


def put_plan_and_scorecard(
    context: ToolContext,
    plan: Plan,
    scorecard: Scorecard,
    *,
    name: str,
    version: str,
    parents: tuple[ArtifactRef, ...],
    parameters: dict[str, JsonValue],
) -> tuple[ArtifactRef, ArtifactRef]:
    plan_ref = put_json(
        context.artifact_store,
        plan.model_dump(mode="json"),
        kind=ArtifactKind.PLAN,
        units="unitless",
        provenance=decision_provenance(name, version, parents, parameters),
        data_schema={"type": "Plan", "version": plan.schema_version},
    )
    score_ref = put_json(
        context.artifact_store,
        scorecard.model_dump(mode="json"),
        kind=ArtifactKind.SCORECARD,
        units="unitless",
        provenance=decision_provenance(
            name,
            version,
            (*parents, plan_ref),
            {**parameters, "evaluator_version": scorecard.evaluator_version},
        ),
        data_schema={"type": "Scorecard", "version": scorecard.schema_version},
    )
    return plan_ref, score_ref
