"""Transparent count, rate, and direct age-standardized health measures."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from oasis.schemas import (
    ArtifactKind,
    DeterminismClassification,
    QualitySummary,
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
    put_frame,
    read_frame,
    require_fields,
    require_kind,
)
from oasis.tools.protocols import ToolContext


class HealthMeasureKind(StrEnum):
    COUNT = "count"
    RATE = "rate"
    DIRECT_AGE_STANDARDIZED_RATE = "direct_age_standardized_rate"


class AgeStratum(BaseModel):
    model_config = ConfigDict(frozen=True)

    numerator_field: str
    denominator_field: str
    standard_weight: float = Field(gt=0.0)


class DeriveHealthMeasureInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    kind: HealthMeasureKind
    output_field: str = "health_measure"
    numerator_field: str | None = None
    denominator_field: str | None = None
    scale: float = Field(default=100_000.0, gt=0.0)
    age_strata: tuple[AgeStratum, ...] = ()
    suppression_fields: tuple[str, ...] = ()
    small_cell_threshold: int | None = Field(default=None, ge=1)
    confidence_level: float | None = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def fields_match_kind(self) -> DeriveHealthMeasureInput:
        if self.kind is HealthMeasureKind.DIRECT_AGE_STANDARDIZED_RATE:
            if not self.age_strata:
                raise ValueError("direct age standardization requires age_strata")
            if self.numerator_field is not None or self.denominator_field is not None:
                raise ValueError("age-standardized input uses age_strata rather than scalar fields")
        else:
            if self.numerator_field is None:
                raise ValueError("count and rate measures require numerator_field")
            if self.kind is HealthMeasureKind.RATE and self.denominator_field is None:
                raise ValueError("rate measures require denominator_field")
            if self.age_strata:
                raise ValueError("age_strata is only valid for direct age standardization")
        return self


class DeriveHealthMeasureOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    row_count: int
    valid_count: int
    suppressed_count: int
    output_field: str
    units: str
    uncertainty_fields: tuple[str, ...]


def _marker_mask(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.Series:
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for field in fields:
        values = frame[field]
        mask |= values.map(
            lambda value: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else str(value).strip().lower() in {"s", "suppressed", "*", "true"}
            )
        )
    return mask


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    try:
        return pd.to_numeric(frame[field], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        invalid(f"health field {field!r} must be numeric: {error}")


class DeriveHealthMeasureTool:
    """Derive interpretable public-health measures while propagating suppression."""

    version = "1.0.0"
    spec = ToolSpec(
        name="derive_health_measure",
        version=version,
        description=(
            "Derive counts, denominator-explicit rates, or direct age-standardized rates with "
            "optional uncertainty and suppression propagation."
        ),
        input_schema=DeriveHealthMeasureInput.model_json_schema(),
        output_schema=DeriveHealthMeasureOutput.model_json_schema(),
        capability_tags=frozenset({"evidence", "public_health", "offline"}),
        artifact_tags=frozenset({ArtifactKind.VECTOR, ArtifactKind.TABLE}),
        side_effects=SideEffectClassification.LOCAL_WRITE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=10, p95_ms=1_000),
        smoke_input={
            "artifact_id": MISSING_ARTIFACT_ID,
            "kind": "rate",
            "numerator_field": "cases",
            "denominator_field": "population",
        },
    )

    async def run(self, arguments: Mapping[str, Any], context: ToolContext) -> ToolResult:
        request = DeriveHealthMeasureInput.model_validate(arguments)
        context.cancellation.raise_if_cancelled()
        reference = artifact_ref(context, request.artifact_id)
        require_kind(reference, {ArtifactKind.VECTOR, ArtifactKind.TABLE})
        frame = read_frame(context, reference).copy()
        fields = list(request.suppression_fields)
        if request.numerator_field is not None:
            fields.append(request.numerator_field)
        if request.denominator_field is not None:
            fields.append(request.denominator_field)
        for stratum in request.age_strata:
            fields.extend([stratum.numerator_field, stratum.denominator_field])
        require_fields(frame, fields)
        suppressed = _marker_mask(frame, request.suppression_fields)

        standard_error = pd.Series(0.0, index=frame.index)
        if request.kind is HealthMeasureKind.COUNT:
            assert request.numerator_field is not None
            numerator = _numeric(frame, request.numerator_field)
            if (numerator.dropna() < 0).any():
                invalid("health counts cannot be negative")
            measure = numerator.copy()
            standard_error = np.sqrt(numerator.clip(lower=0))
            if request.small_cell_threshold is not None:
                suppressed |= numerator < request.small_cell_threshold
            output_units = reference.units or "count"
        elif request.kind is HealthMeasureKind.RATE:
            assert request.numerator_field is not None
            assert request.denominator_field is not None
            numerator = _numeric(frame, request.numerator_field)
            denominator = _numeric(frame, request.denominator_field)
            if (numerator.dropna() < 0).any() or (denominator.dropna() <= 0).any():
                invalid("rate numerators must be non-negative and denominators must be positive")
            proportion = numerator / denominator
            measure = proportion * request.scale
            standard_error = np.sqrt(
                proportion.clip(0, 1) * (1 - proportion.clip(0, 1)) / denominator
            )
            standard_error *= request.scale
            if request.small_cell_threshold is not None:
                suppressed |= numerator < request.small_cell_threshold
            output_units = f"per_{request.scale:g}"
        else:
            weighted_rates = pd.Series(0.0, index=frame.index)
            variance = pd.Series(0.0, index=frame.index)
            total_weight = sum(stratum.standard_weight for stratum in request.age_strata)
            for stratum in request.age_strata:
                numerator = _numeric(frame, stratum.numerator_field)
                denominator = _numeric(frame, stratum.denominator_field)
                if (numerator.dropna() < 0).any() or (denominator.dropna() <= 0).any():
                    invalid("age-stratum numerators must be non-negative and denominators positive")
                if request.small_cell_threshold is not None:
                    suppressed |= numerator < request.small_cell_threshold
                weight = stratum.standard_weight / total_weight
                proportion = numerator / denominator
                weighted_rates += weight * proportion
                variance += (
                    weight**2 * proportion.clip(0, 1) * (1 - proportion.clip(0, 1)) / denominator
                )
            measure = weighted_rates * request.scale
            standard_error = np.sqrt(variance) * request.scale
            output_units = f"age_standardized_per_{request.scale:g}"

        suppressed |= measure.isna()
        frame[request.output_field] = measure.mask(suppressed)
        suppression_output = f"{request.output_field}_suppressed"
        frame[suppression_output] = suppressed
        uncertainty_fields: tuple[str, ...] = ()
        if request.confidence_level is not None:
            z_score = NormalDist().inv_cdf((1 + request.confidence_level) / 2)
            standard_error_field = f"{request.output_field}_standard_error"
            lower_field = f"{request.output_field}_lower"
            upper_field = f"{request.output_field}_upper"
            frame[standard_error_field] = standard_error.mask(suppressed)
            frame[lower_field] = (measure - z_score * standard_error).clip(lower=0).mask(suppressed)
            frame[upper_field] = (measure + z_score * standard_error).mask(suppressed)
            uncertainty_fields = (standard_error_field, lower_field, upper_field)

        suppressed_count = int(suppressed.sum())
        warnings = (
            (f"suppressed {suppressed_count} rows in derived measure",) if suppressed_count else ()
        )
        quality = QualitySummary(
            missing_fraction=float(frame[request.output_field].isna().mean())
            if len(frame)
            else 0.0,
            suppressed_count=suppressed_count,
            warnings=warnings,
        )
        output_ref = put_frame(
            context.artifact_store,
            frame,
            source_kind=reference.kind,
            crs=reference.crs,
            units=output_units,
            provenance=child_provenance(self.spec.name, self.version, [reference], request),
            quality=quality,
        )
        output = DeriveHealthMeasureOutput(
            artifact_id=output_ref.id,
            row_count=len(frame),
            valid_count=len(frame) - suppressed_count,
            suppressed_count=suppressed_count,
            output_field=request.output_field,
            units=output_units,
            uncertainty_fields=uncertainty_fields,
        )
        summary: dict[str, JsonValue] = {
            "artifact_id": output_ref.id,
            "measure": request.kind.value,
            "valid": output.valid_count,
            "suppressed": suppressed_count,
        }
        return ToolResult(
            status=ToolResultStatus.COMPLETE,
            summary=summary,
            artifacts=(output_ref,),
            metrics=output.model_dump(mode="json"),
        )
