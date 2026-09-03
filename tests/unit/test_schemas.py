from __future__ import annotations

from datetime import UTC, datetime

from oasis.llm.schemas import ChatMessage, ModelRequest, TokenUsage, ToolDefinition
from oasis.schemas import (
    ARTIFACT_METADATA_SCHEMA_VERSION,
    AccessMatrixSpec,
    ArtifactKind,
    ArtifactRef,
    CandidateSpec,
    DemandSpec,
    MissingDataPolicy,
    ServiceMatrixSpec,
)


def test_public_schema_serialization_round_trip() -> None:
    request = ModelRequest(
        request_id="round-trip",
        messages=(ChatMessage(role="user", content="hello"),),
        max_generated_tokens=7,
        tools=(
            ToolDefinition(
                name="future_tool",
                description="A future model-facing tool.",
                input_schema={"type": "object", "properties": {}},
            ),
        ),
    )

    assert ModelRequest.model_validate_json(request.model_dump_json()) == request


def test_token_usage_aggregates_across_raw_chat_turns() -> None:
    usage = TokenUsage.aggregate(
        [
            TokenUsage(input_tokens=4, generated_tokens=3, reasoning_tokens=1),
            TokenUsage(input_tokens=9, generated_tokens=5, reasoning_tokens=2),
        ]
    )

    assert usage.input_tokens == 13
    assert usage.generated_tokens == 8
    assert usage.reasoning_tokens == 3
    assert usage.total_tokens == 21


def artifact(kind: ArtifactKind) -> ArtifactRef:
    content_hash = "0" * 64
    return ArtifactRef(
        id=f"sha256-{content_hash}",
        content_hash=content_hash,
        byte_size=0,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        kind=kind,
        media_type="application/octet-stream",
        units="unitless",
    )


def test_public_evidence_schemas_serialize_round_trip() -> None:
    demand = DemandSpec(
        artifact=artifact(ArtifactKind.VECTOR),
        location_id_field="demand_id",
        need_fields=("population",),
        group_fields=("older_adult",),
        suppression_fields=("suppressed",),
        missing_data_policy=MissingDataPolicy.ERROR,
    )
    candidates = CandidateSpec(
        artifact=artifact(ArtifactKind.VECTOR),
        candidate_id_field="site_id",
        capacity_field="capacity",
        generation_method="supplied",
    )
    access = AccessMatrixSpec(
        artifact=artifact(ArtifactKind.MATRIX),
        origin_ids=("d1",),
        destination_ids=("s1",),
        strategy="geodesic",
        metric="distance",
        units="kilometers",
        directed=False,
        unreachable_count=0,
    )
    service = ServiceMatrixSpec(
        artifact=artifact(ArtifactKind.MATRIX),
        response="binary_threshold",
        parameters={"threshold": 5.0},
    )

    for schema in (demand, candidates, access, service):
        assert type(schema).model_validate_json(schema.model_dump_json()) == schema


def test_artifact_metadata_provider_extension_accepts_legacy_payloads() -> None:
    legacy = artifact(ArtifactKind.TABLE).model_dump(mode="json")
    legacy.pop("metadata_schema_version")
    legacy.pop("provider_metadata")

    restored = ArtifactRef.model_validate(legacy)

    assert restored.metadata_schema_version == ARTIFACT_METADATA_SCHEMA_VERSION
    assert restored.provider_metadata == {}
