"""Shared validation and lineage helpers for deterministic evidence tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

import geopandas as gpd
import pandas as pd
from pydantic import BaseModel

from oasis.artifacts import ArtifactProvenance, put_table, put_vector, read_table, read_vector
from oasis.artifacts.protocols import ArtifactNotFoundError, ArtifactStore
from oasis.schemas import (
    ArtifactKind,
    ArtifactLineage,
    ArtifactRef,
    ArtifactTransformation,
    PrivacyClassification,
    QualitySummary,
    ToolError,
    ToolErrorCode,
)
from oasis.tools.protocols import ToolContext, ToolExecutionError

MISSING_ARTIFACT_ID = "sha256-" + "0" * 64


def invalid(message: str, *, context: dict[str, Any] | None = None) -> NoReturn:
    """Raise a model-safe semantic input error."""

    raise ToolExecutionError(
        ToolError(
            code=ToolErrorCode.INVALID_ARGUMENTS,
            message=message,
            context=context or {},
        )
    )


def artifact_ref(context: ToolContext, artifact_id: str) -> ArtifactRef:
    """Resolve an artifact ID while preserving the public not-found error category."""

    try:
        return context.artifact_store.get_metadata(artifact_id)
    except ArtifactNotFoundError as error:
        raise ToolExecutionError(
            ToolError(
                code=ToolErrorCode.NOT_FOUND, message=f"artifact {artifact_id!r} was not found"
            )
        ) from error


def require_kind(reference: ArtifactRef, allowed: Iterable[ArtifactKind]) -> None:
    allowed_set = frozenset(allowed)
    if reference.kind not in allowed_set:
        invalid(
            f"artifact {reference.id!r} has kind {reference.kind.value}; expected one of "
            + ", ".join(sorted(kind.value for kind in allowed_set))
        )


def require_fields(frame: pd.DataFrame, fields: Iterable[str]) -> None:
    missing = sorted(set(fields) - set(str(column) for column in frame.columns))
    if missing:
        invalid("artifact is missing required fields", context={"fields": missing})


def read_frame(context: ToolContext, reference: ArtifactRef) -> pd.DataFrame:
    """Read a vector or table artifact through its canonical codec."""

    if reference.kind is ArtifactKind.VECTOR:
        return read_vector(context.artifact_store, reference)
    if reference.kind is ArtifactKind.TABLE:
        return read_table(context.artifact_store, reference)
    invalid("operation requires a vector or table artifact")


def _privacy_rank(value: PrivacyClassification) -> int:
    return {
        PrivacyClassification.PUBLIC: 0,
        PrivacyClassification.INTERNAL: 1,
        PrivacyClassification.SENSITIVE: 2,
        PrivacyClassification.RESTRICTED: 3,
    }[value]


def child_provenance(
    tool_name: str,
    version: str,
    parents: Iterable[ArtifactRef],
    parameters: Mapping[str, Any] | BaseModel,
) -> ArtifactProvenance:
    """Create complete direct lineage and conservatively inherited handling metadata."""

    parent_tuple = tuple(parents)
    if not parent_tuple:
        invalid("derived artifacts require at least one parent")
    raw_parameters = (
        parameters.model_dump(mode="json", exclude_none=True)
        if isinstance(parameters, BaseModel)
        else dict(parameters)
    )
    licenses = sorted({reference.license or "unknown" for reference in parent_tuple})
    privacy = max((reference.privacy for reference in parent_tuple), key=_privacy_rank)
    return ArtifactProvenance(
        source_uri=f"oasis://tool/{tool_name}/{version}",
        source_provider="oasis",
        source_version=version,
        license=" AND ".join(licenses),
        privacy=privacy,
        lineage=ArtifactLineage(
            parent_ids=tuple(reference.id for reference in parent_tuple),
            transformations=(
                ArtifactTransformation(
                    name=tool_name,
                    version=version,
                    parameters=raw_parameters,
                ),
            ),
        ),
    )


def put_frame(
    store: ArtifactStore,
    frame: pd.DataFrame,
    *,
    source_kind: ArtifactKind,
    crs: str | None,
    units: str,
    provenance: ArtifactProvenance,
    quality: QualitySummary | None = None,
) -> ArtifactRef:
    """Publish a transformed frame without changing vector/table representation unexpectedly."""

    if source_kind is ArtifactKind.VECTOR:
        if not isinstance(frame, gpd.GeoDataFrame):
            invalid("vector transformation did not retain geometry")
        return put_vector(store, frame, units=units, provenance=provenance, quality=quality)
    if source_kind is ArtifactKind.TABLE:
        return put_table(
            store,
            frame,
            crs=crs,
            units=units,
            provenance=provenance,
            quality=quality,
        )
    invalid("only vector and table frames can be published")
