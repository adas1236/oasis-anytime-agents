"""Public optimization problem schemas, plugins, registry, and search contracts."""

from oasis.problems.location_allocation import (
    LocationAllocationPlugin,
    ProblemDataError,
    create_problem_registry,
    load_problem_data,
    plan_hash,
    problem_hashes,
)
from oasis.problems.protocols import Deadline, ProblemPlugin
from oasis.problems.registry import ProblemRegistry, ProblemRegistryError
from oasis.problems.schemas import (
    Comparison,
    EquityGroup,
    EquityObjective,
    LocationAllocationPolicy,
    LocationAllocationProblem,
    LocationProblemType,
    ResultView,
    ScenarioAggregation,
    Scorecard,
    SearchResumeToken,
    SearchStrategy,
    ServiceScenario,
    ValidationIssue,
    ValidationReport,
    VerifiedBound,
)

__all__ = [
    "Comparison",
    "Deadline",
    "EquityGroup",
    "EquityObjective",
    "LocationAllocationPlugin",
    "LocationAllocationPolicy",
    "LocationAllocationProblem",
    "LocationProblemType",
    "ProblemDataError",
    "ProblemPlugin",
    "ProblemRegistry",
    "ProblemRegistryError",
    "ResultView",
    "ScenarioAggregation",
    "Scorecard",
    "SearchResumeToken",
    "SearchStrategy",
    "ServiceScenario",
    "ValidationIssue",
    "ValidationReport",
    "VerifiedBound",
    "create_problem_registry",
    "load_problem_data",
    "plan_hash",
    "problem_hashes",
]
