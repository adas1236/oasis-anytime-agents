"""Versioned schemas for manifests, generated instances, runs, and reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oasis.config import RuntimePolicy
from oasis.controller import BudgetSpec, RunStatus, TerminalReason
from oasis.problems import LocationProblemType, RouteProblemType, Scorecard, SearchStrategy
from oasis.runtimes import ComputeInventory, RuntimePlan, evaluation_group_key
from oasis.schemas import Plan

EVALUATION_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
BENCHMARK_RUN_SCHEMA_VERSION: Literal["1.1.0"] = "1.1.0"
GENERATOR_VERSION: Literal["1.0.0"] = "1.0.0"


class BenchmarkTrack(StrEnum):
    """Evaluation tracks defined by the project plan."""

    FORMAL_OPTIMIZATION = "formal_optimization"
    PROBLEM_COMPILATION = "problem_compilation"
    GEOGRAPHIC_GROUNDING = "geographic_grounding"
    END_TO_END = "end_to_end"
    PRIMITIVE_TOOLS = "primitive_tools"
    EXPERT_SOLVERS = "expert_solvers_allowed"


class ComparisonKind(StrEnum):
    """Policies compared at equal declared budgets."""

    DETERMINISTIC_BASELINE = "deterministic_baseline"
    DETERMINISTIC_PORTFOLIO = "deterministic_portfolio"
    ONE_SHOT_MODEL = "one_shot_model"
    ITERATIVE_MODEL = "iterative_model"
    DIRECT_MODEL_CANDIDATE = "direct_model_candidate"


class ProblemFamily(StrEnum):
    """Problem families supported by the Phase 10 offline runner."""

    LOCATION_ALLOCATION = "location_allocation"
    ROUTING = "routing"


class SpatialDistribution(StrEnum):
    """Versioned spatial distributions available to synthetic generators."""

    UNIFORM = "uniform"
    CLUSTERED = "clustered"
    GRID = "grid"
    RING = "ring"
    CORRIDOR = "corridor"
    ISLANDS = "islands"
    OUTLIERS = "outliers"


class InstanceScale(StrEnum):
    """Reference-solution policy for an instance size."""

    SMALL = "small"
    MEDIUM = "medium"
    STRESS = "stress"


class DatasetSplit(StrEnum):
    """Seed namespaces used to keep development and held-out cases separate."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class ConstraintRegime(StrEnum):
    """Constraint tightness represented by a generated instance."""

    FEASIBLE = "feasible"
    TIGHT = "tight"
    INFEASIBLE = "infeasible"


class EquityStructure(StrEnum):
    """How a named underserved group is distributed spatially."""

    NONE = "none"
    BALANCED = "balanced"
    ISOLATED = "isolated"
    EMPTY = "empty"


class FixtureName(StrEnum):
    """Frozen public-health-inspired benchmark definitions shipped with OASIS."""

    COOLING_CENTERS = "cooling_centers"
    CLINIC_ACCESS = "clinic_access"
    EMERGENCY_TAIL_ACCESS = "emergency_tail_access"
    CAPACITY_ALLOCATION = "capacity_allocation"
    EQUITY_COVERAGE = "equity_coverage"
    RESILIENT_COVERAGE = "resilient_coverage"
    MOBILE_ROUTING = "mobile_routing"
    MONITORS = "monitors"


class SyntheticInstanceSpec(BaseModel):
    """Complete, reproducible input to a versioned synthetic instance generator."""

    model_config = ConfigDict(frozen=True)

    generator_version: Literal["1.0.0"] = GENERATOR_VERSION
    family: ProblemFamily
    problem_type: str = Field(min_length=1)
    seed: int
    split: DatasetSplit = DatasetSplit.DEVELOPMENT
    scale: InstanceScale = InstanceScale.SMALL
    distribution: SpatialDistribution = SpatialDistribution.UNIFORM
    demand_count: int = Field(default=8, ge=2, le=1_000)
    candidate_count: int = Field(default=5, ge=2, le=1_000)
    site_limit: int = Field(default=2, ge=1)
    constraint_regime: ConstraintRegime = ConstraintRegime.FEASIBLE
    equity_structure: EquityStructure = EquityStructure.BALANCED
    directed_travel: bool = False
    unreachable_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    duplicate_fraction: float = Field(default=0.0, ge=0.0, le=0.5)
    force_distance_ties: bool = False
    service_threshold_boundary: bool = False
    scenario_count: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def dimensions_match_family(self) -> Self:
        if self.site_limit > self.candidate_count:
            raise ValueError("site_limit cannot exceed candidate_count")
        route_types = {value.value for value in RouteProblemType}
        location_types = {value.value for value in LocationProblemType}
        if self.family is ProblemFamily.ROUTING and self.problem_type not in route_types:
            raise ValueError("routing generators require a route problem type")
        if (
            self.family is ProblemFamily.LOCATION_ALLOCATION
            and self.problem_type not in location_types
        ):
            raise ValueError("location generators require a location-allocation problem type")
        return self


class BenchmarkInstanceSpec(BaseModel):
    """Manifest entry selecting either a frozen fixture or an inline generator spec."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    fixture: FixtureName | None = None
    generator: SyntheticInstanceSpec | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> Self:
        if (self.fixture is None) == (self.generator is None):
            raise ValueError("benchmark instances require exactly one fixture or generator")
        return self


class BenchmarkBudget(BaseModel):
    """Named resource budget used as one paired comparison stratum."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    resources: BudgetSpec


class EvaluationModelSpec(BaseModel):
    """Model selection for fake or explicitly authorized real-model comparisons."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["fake", "transformers"] = "fake"
    profile: str = "gemma4_e4b_it"
    model_id: str | None = None
    revision: str | None = None
    thinking: bool = False
    trust_remote_code: bool = False
    allow_real_model: bool = False

    @model_validator(mode="after")
    def real_backend_is_explicit(self) -> Self:
        if self.backend == "transformers" and not self.allow_real_model:
            raise ValueError("real-model manifests must set model.allow_real_model=true")
        if self.backend == "fake" and self.allow_real_model:
            raise ValueError("allow_real_model applies only to the transformers backend")
        return self


class BenchmarkManifest(BaseModel):
    """Versioned, self-contained benchmark execution manifest."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0.0"] = EVALUATION_SCHEMA_VERSION
    benchmark_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    track: BenchmarkTrack = BenchmarkTrack.FORMAL_OPTIMIZATION
    instances: tuple[BenchmarkInstanceSpec, ...] = Field(min_length=1)
    comparisons: tuple[ComparisonKind, ...] = Field(
        default=(
            ComparisonKind.DETERMINISTIC_BASELINE,
            ComparisonKind.DETERMINISTIC_PORTFOLIO,
            ComparisonKind.ONE_SHOT_MODEL,
            ComparisonKind.ITERATIVE_MODEL,
            ComparisonKind.DIRECT_MODEL_CANDIDATE,
        ),
        min_length=1,
    )
    budgets: tuple[BenchmarkBudget, ...] = Field(min_length=1)
    run_seeds: tuple[int, ...] = Field(default=(0,), min_length=1)
    model: EvaluationModelSpec = Field(default_factory=EvaluationModelSpec)
    runtime_policy: RuntimePolicy = Field(default_factory=RuntimePolicy)
    tool_suite: tuple[SearchStrategy, ...] = Field(
        default=(
            SearchStrategy.ADD_SWAP,
            SearchStrategy.MULTI_SWAP,
            SearchStrategy.LOCAL_ASSIGNMENT,
            SearchStrategy.SCENARIO_AWARE,
            SearchStrategy.TWO_OPT,
            SearchStrategy.RELOCATE,
            SearchStrategy.SWAP,
            SearchStrategy.EXACT_ENUMERATION,
        ),
        min_length=1,
    )
    expected_evaluator_versions: dict[str, str] = Field(min_length=1)
    max_exact_candidates: int = Field(default=50_000, ge=1)
    max_reference_candidates: int = Field(default=2_000, ge=1)
    max_candidates_per_action: int = Field(default=5_000, ge=1)

    @model_validator(mode="after")
    def entries_are_unique_and_supported(self) -> Self:
        collections = {
            "instance IDs": tuple(item.id for item in self.instances),
            "comparison kinds": self.comparisons,
            "budget IDs": tuple(item.id for item in self.budgets),
            "run seeds": self.run_seeds,
            "tool strategies": self.tool_suite,
        }
        for label, values in collections.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        expert_strategies = {
            SearchStrategy.EXACT_ENUMERATION,
            SearchStrategy.ORTOOLS_CP_SAT,
            SearchStrategy.ORTOOLS_ROUTING,
        }
        if self.track is BenchmarkTrack.PRIMITIVE_TOOLS and expert_strategies.intersection(
            self.tool_suite
        ):
            raise ValueError("primitive_tools manifests cannot expose exact or expert solvers")
        return self


class ReferenceKind(StrEnum):
    """Strength of the independently evaluated comparison reference."""

    EXACT_OPTIMUM = "exact_optimum"
    BEST_KNOWN = "best_known"
    INFEASIBLE = "infeasible"


class ReferenceSolution(BaseModel):
    """Exact small-instance oracle or named medium/stress best-known solution."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0.0"] = EVALUATION_SCHEMA_VERSION
    kind: ReferenceKind
    plan: Plan | None = None
    scorecard: Scorecard | None = None
    plan_artifact_id: str | None = None
    scorecard_artifact_id: str | None = None
    bound_artifact_id: str | None = None
    evaluated_candidates: int = Field(ge=0)
    total_candidates: int | None = Field(default=None, ge=0)
    method: str = Field(min_length=1)

    @model_validator(mode="after")
    def solution_fields_are_complete(self) -> Self:
        values = (
            self.plan,
            self.scorecard,
            self.plan_artifact_id,
            self.scorecard_artifact_id,
        )
        if self.kind is ReferenceKind.INFEASIBLE:
            if any(value is not None for value in values):
                raise ValueError("infeasible references cannot contain a solution")
        elif any(value is None for value in values):
            raise ValueError("solution references require a plan, scorecard, and artifact IDs")
        if self.kind is ReferenceKind.EXACT_OPTIMUM and self.bound_artifact_id is None:
            raise ValueError("exact references require a verified bound artifact")
        return self


class GeneratedInstance(BaseModel):
    """Immutable artifacts and independent reference produced from one instance spec."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0.0"] = EVALUATION_SCHEMA_VERSION
    id: str
    generator_spec: SyntheticInstanceSpec
    effective_seed: int = Field(ge=0)
    problem_artifact_id: str | None = None
    baseline_plan_artifact_id: str | None = None
    baseline_scorecard_artifact_id: str | None = None
    problem_hash: str | None = None
    problem_type: str
    evaluator_version: str
    admitted: bool
    issue_codes: tuple[str, ...] = ()
    reference: ReferenceSolution | None = None


class QualityCheckpoint(BaseModel):
    """One independently measured incumbent from a single continuing controller run."""

    model_config = ConfigDict(frozen=True)

    elapsed_ms: int = Field(ge=0)
    total_model_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    comparator_key: tuple[float, ...]
    primary_quality: float
    baseline_gain: float
    normalized_gap_closed: float | None = None
    plan_artifact_id: str
    scorecard_artifact_id: str


class BenchmarkRunRecord(BaseModel):
    """Raw, independently derived measurements for one paired benchmark run."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0.0", "1.1.0"] = BENCHMARK_RUN_SCHEMA_VERSION
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_id: str
    run_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str
    pair_id: str
    instance_id: str
    problem_family: ProblemFamily
    problem_type: str
    track: BenchmarkTrack
    comparison: ComparisonKind
    budget_id: str
    run_seed: int
    generator_seed: int
    problem_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: RunStatus
    terminal_reason: TerminalReason
    feasible: bool
    invalid_candidate_count: int = Field(ge=0)
    rejected_candidate_count: int = Field(ge=0)
    raw_objective: dict[str, float] = Field(default_factory=dict)
    comparator_key: tuple[float, ...] = ()
    overall_metrics: dict[str, float] = Field(default_factory=dict)
    group_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    scenario_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_gain: float | None = None
    reference_kind: ReferenceKind | None = None
    reference_gap: float | None = None
    normalized_gap_closed: float | None = None
    time_to_first_feasible_ms: int | None = Field(default=None, ge=0)
    deadline_overshoot_ms: int = Field(ge=0)
    wall_elapsed_ms: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    generated_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    total_model_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    tool_latency_ms: int = Field(ge=0)
    tool_failure_count: int = Field(ge=0)
    model_failure_count: int = Field(ge=0)
    parse_repair_count: int = Field(ge=0)
    model_action_input_tokens: tuple[int, ...] = ()
    model_action_generated_tokens: tuple[int, ...] = ()
    compact_context_bytes: tuple[int, ...] = ()
    tool_latency_observations_ms: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    tool_p50_estimates_ms: dict[str, int] = Field(default_factory=dict)
    tool_p95_estimates_ms: dict[str, int] = Field(default_factory=dict)
    artifact_count: int = Field(default=0, ge=0)
    artifact_bytes: int = Field(default=0, ge=0)
    largest_artifact_bytes: int = Field(default=0, ge=0)
    checkpoints: tuple[QualityCheckpoint, ...] = ()
    quality_at_wall_time_ms: dict[str, float | None] = Field(default_factory=dict)
    quality_at_total_tokens: dict[str, float | None] = Field(default_factory=dict)
    auc_quality_log_time: float | None = None
    auc_quality_log_tokens: float | None = None
    runtime_plan: RuntimePlan
    compute_inventory: ComputeInventory
    runtime_group_key: tuple[str, ...] = Field(min_length=1)
    model_profile: str | None = None
    model_id: str | None = None
    evaluator_version: str
    problem_plugin_version: str
    controller_version: str
    tool_versions: dict[str, str]
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def objective_exists_only_for_feasible_results(self) -> Self:
        if self.schema_version == "1.1.0" and None in {
            self.problem_hash,
            self.evidence_hash,
            self.policy_hash,
        }:
            raise ValueError("version 1.1 benchmark records require immutable run hashes")
        if not self.feasible and (self.raw_objective or self.comparator_key):
            raise ValueError("invalid plans must be separated from objective quality")
        if self.model_id is not None and self.model_profile is None:
            raise ValueError("model-backed records require both model profile and model ID")
        if self.total_model_tokens != self.input_tokens + self.generated_tokens:
            raise ValueError("total model tokens must equal input plus generated tokens")
        model_comparisons = {
            ComparisonKind.ONE_SHOT_MODEL,
            ComparisonKind.ITERATIVE_MODEL,
            ComparisonKind.DIRECT_MODEL_CANDIDATE,
        }
        if self.comparison in model_comparisons and self.runtime_plan.runtime.value != "fake":
            if self.model_id is None or self.model_profile is None:
                raise ValueError("real-model comparisons require model identity metadata")
            if not self.compute_inventory.library_versions:
                raise ValueError("real-model comparisons require runtime library metadata")
        expected_group = (
            *evaluation_group_key(self.runtime_plan, self.compute_inventory),
            f"platform={self.compute_inventory.platform or 'unknown'}",
            f"cpu_count={self.compute_inventory.cpu_count}",
            f"total_ram_bytes={self.compute_inventory.total_ram_bytes}",
        )
        if self.runtime_group_key != expected_group:
            raise ValueError("runtime group key does not match the recorded runtime and hardware")
        return self


class AggregateGroup(BaseModel):
    """Descriptive repeated-run summary for one fully stratified result group."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    problem_type: str
    comparison: ComparisonKind
    budget_id: str
    runtime_group_key: tuple[str, ...]
    runs: int = Field(ge=1)
    feasibility_rate: float = Field(ge=0.0, le=1.0)
    primary_quality_mean: float | None = None
    primary_quality_sample_variance: float | None = Field(default=None, ge=0.0)
    primary_quality_ci95_half_width: float | None = Field(default=None, ge=0.0)
    raw_objective_means: dict[str, float] = Field(default_factory=dict)
    overall_metric_means: dict[str, float] = Field(default_factory=dict)
    group_metric_means: dict[str, dict[str, float]] = Field(default_factory=dict)
    scenario_metric_means: dict[str, dict[str, float]] = Field(default_factory=dict)
    baseline_gain_mean: float | None = None
    reference_gap_mean: float | None = None
    time_to_first_feasible_ms_mean: float | None = Field(default=None, ge=0.0)
    deadline_violation_rate: float = Field(ge=0.0, le=1.0)
    deadline_overshoot_ms_mean: float = Field(ge=0.0)
    input_tokens_mean: float = Field(ge=0.0)
    generated_tokens_mean: float = Field(ge=0.0)
    reasoning_tokens_mean: float = Field(ge=0.0)
    total_model_tokens_mean: float = Field(ge=0.0)
    tool_calls_mean: float = Field(ge=0.0)
    tool_latency_ms_mean: float = Field(ge=0.0)
    tool_failure_rate: float = Field(ge=0.0, le=1.0)
    model_failure_rate: float = Field(ge=0.0, le=1.0)
    parse_repair_rate: float = Field(ge=0.0, le=1.0)
    auc_quality_log_time_mean: float | None = None
    auc_quality_log_tokens_mean: float | None = None


class PairedDeltaSummary(BaseModel):
    """Descriptive paired quality delta against the deterministic portfolio."""

    model_config = ConfigDict(frozen=True)

    instance_id: str
    budget_id: str
    runtime_group_key: tuple[str, ...]
    contender: ComparisonKind
    pairs: int = Field(ge=1)
    mean_primary_quality_delta: float
    sample_variance: float | None = Field(default=None, ge=0.0)
    ci95_half_width: float | None = Field(default=None, ge=0.0)


class BenchmarkSummary(BaseModel):
    """Aggregate report with explicit strata and modest descriptive uncertainty."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0.0"] = EVALUATION_SCHEMA_VERSION
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_id: str
    run_count: int = Field(ge=0)
    groups: tuple[AggregateGroup, ...]
    paired_deltas: tuple[PairedDeltaSummary, ...]
    notes: tuple[str, ...] = (
        "Intervals are descriptive normal-approximation summaries, not claims of significance.",
        "Results are stratified by recorded model, runtime, devices, driver, and libraries.",
    )
