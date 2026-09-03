"""Public schemas for immutable optimization problems and authoritative evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from oasis.schemas import ArtifactRef, CandidateSpec, DemandSpec, Plan


class LocationProblemType(StrEnum):
    """Location-allocation families supported by the first decision plugin set."""

    MAX_WEIGHTED_COVERAGE = "max_weighted_coverage"
    MIN_COST_TARGET_COVERAGE = "min_cost_target_coverage"
    WEIGHTED_P_MEDIAN = "weighted_p_median"
    P_CENTER = "p_center"
    QUANTILE_ACCESS = "quantile_access"
    CAPACITATED_ALLOCATION = "capacitated_allocation"
    EQUITY_COVERAGE = "equity_coverage"
    INCREMENTAL_COVERAGE = "incremental_coverage"
    RESILIENT_COVERAGE = "resilient_coverage"


class EquityObjective(StrEnum):
    """How named-group coverage participates in the fixed comparator."""

    NONE = "none"
    FLOORS = "floors"
    MAX_MIN = "max_min"


class ScenarioAggregation(StrEnum):
    """How scenario-specific service outcomes become an overall metric."""

    EXPECTED = "expected"
    WORST_CASE = "worst_case"


class EquityGroup(BaseModel):
    """A named group defined by a categorical match or numeric membership field."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$")
    field: str = Field(min_length=1)
    match_value: str | int | float | bool | None = None


class ServiceScenario(BaseModel):
    """One immutable demand-by-candidate service matrix and its expected-value weight."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$")
    service_matrix: ArtifactRef
    weight: float = Field(default=1.0, gt=0.0)


class LocationAllocationPolicy(BaseModel):
    """Locked constraints and comparator settings for a location-allocation problem."""

    model_config = ConfigDict(frozen=True)

    site_limit: int | None = Field(default=None, ge=0)
    new_site_limit: int | None = Field(default=None, ge=0)
    financial_budget: float | None = Field(default=None, ge=0.0)
    coverage_target: float | None = Field(default=None, ge=0.0, le=1.0)
    equity_objective: EquityObjective = EquityObjective.NONE
    group_floors: dict[str, float] = Field(default_factory=dict)
    quantile: float = Field(default=0.9, gt=0.0, le=1.0)
    redundancy: int = Field(default=2, ge=2)
    scenario_aggregation: ScenarioAggregation = ScenarioAggregation.EXPECTED
    scenario_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_floors_and_weights(self) -> Self:
        if any(value < 0.0 or value > 1.0 for value in self.group_floors.values()):
            raise ValueError("group coverage floors must lie in the closed interval zero to one")
        if any(value <= 0.0 for value in self.scenario_weights.values()):
            raise ValueError("scenario weights must be positive")
        return self


class LocationAllocationProblem(BaseModel):
    """Complete immutable decision representation consumed by location plugins."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    type_id: LocationProblemType
    plugin_version: str = "1.0.0"
    evaluator_version: str = "1.0.0"
    demand: DemandSpec
    candidates: CandidateSpec
    access_matrix: ArtifactRef
    service_scenarios: tuple[ServiceScenario, ...] = Field(min_length=1)
    need_field: str = Field(min_length=1)
    groups: tuple[EquityGroup, ...] = ()
    policy: LocationAllocationPolicy
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    problem_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def names_are_unique(self) -> Self:
        scenario_names = tuple(scenario.name for scenario in self.service_scenarios)
        group_names = tuple(group.name for group in self.groups)
        if len(scenario_names) != len(set(scenario_names)):
            raise ValueError("service scenario names must be unique")
        if len(group_names) != len(set(group_names)):
            raise ValueError("equity group names must be unique")
        return self


class ValidationIssue(BaseModel):
    """One stable, structured validation finding."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1)
    context: dict[str, JsonValue] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Pure validation result; evaluators never silently repair a plan."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    issues: tuple[ValidationIssue, ...] = ()

    @model_validator(mode="after")
    def validity_matches_issues(self) -> Self:
        if self.valid == bool(self.issues):
            raise ValueError("validation is valid exactly when it has no issues")
        return self


class Scorecard(BaseModel):
    """Authoritative problem-plugin measurement under one frozen comparator."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    feasible: bool
    violations: tuple[ValidationIssue, ...] = ()
    raw_objective: dict[str, float] = Field(default_factory=dict)
    comparator_key: tuple[float, ...]
    overall_metrics: dict[str, float] = Field(default_factory=dict)
    group_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    scenario_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_relative_improvement: float | None = None
    verified_lower_bound: tuple[float, ...] | None = None
    verified_upper_bound: tuple[float, ...] | None = None
    optimality_gap: float | None = Field(default=None, ge=0.0)
    problem_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_version: str = Field(min_length=1)
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def feasibility_matches_violations(self) -> Self:
        if self.feasible == bool(self.violations):
            raise ValueError("a scorecard is feasible exactly when it has no violations")
        return self


class Comparison(StrEnum):
    """Total ordering result under a plugin's immutable comparator."""

    BETTER = "better"
    EQUAL = "equal"
    WORSE = "worse"


class SearchStrategy(StrEnum):
    """Stable strategy registry values exposed through the improve tool."""

    ADD_SWAP = "add_swap"
    MULTI_SWAP = "multi_swap"
    LOCAL_ASSIGNMENT = "local_assignment"
    SCENARIO_AWARE = "scenario_aware"
    EXACT_ENUMERATION = "exact_enumeration"
    ORTOOLS_CP_SAT = "ortools_cp_sat"


class SearchResumeToken(BaseModel):
    """Opaque-to-callers deterministic cursor for resumable combination search."""

    model_config = ConfigDict(frozen=True)

    problem_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy: SearchStrategy
    next_index: int = Field(ge=0)
    incumbent: Plan


class VerifiedBound(BaseModel):
    """Independently checkable exact-search or solver bound."""

    model_config = ConfigDict(frozen=True)

    problem_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy: SearchStrategy
    complete: bool
    explored_candidates: int = Field(ge=0)
    total_candidates: int | None = Field(default=None, ge=0)
    best_comparator_key: tuple[float, ...]
    certificate: dict[str, JsonValue] = Field(default_factory=dict)


class ResultView(BaseModel):
    """Deterministic problem-rendered view independent of HTTP or UI frameworks."""

    model_config = ConfigDict(frozen=True)

    problem_type: str
    feasible: bool
    selected_site_ids: tuple[str, ...]
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    overall_metrics: dict[str, float] = Field(default_factory=dict)
    group_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    scenario_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    violations: tuple[ValidationIssue, ...] = ()
    warnings: tuple[str, ...] = ()
