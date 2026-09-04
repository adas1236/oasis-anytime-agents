from __future__ import annotations

import math

import pytest
from pydantic import TypeAdapter, ValidationError

from oasis.artifacts import LocalArtifactStore, read_json
from oasis.evaluation import (
    BenchmarkBudget,
    BenchmarkInstanceSpec,
    BenchmarkManifest,
    ComparisonKind,
    ConstraintRegime,
    DatasetSplit,
    EquityStructure,
    EvaluationModelSpec,
    FixtureName,
    InstanceScale,
    ProblemFamily,
    ReferenceKind,
    SpatialDistribution,
    SyntheticInstanceSpec,
    area_under_log_curve,
    attach_reference,
    descriptive_summary,
    effective_seed,
    fixture_catalog,
    generate_instance,
    incumbent_quality_at,
    independently_evaluate_plan,
    load_fixture,
    split_seeds_are_disjoint,
)
from oasis.problems import (
    LocationAllocationProblem,
    LocationProblemType,
    RouteProblemType,
    RouteServiceProblem,
)
from oasis.schemas import Plan

_PROBLEM = TypeAdapter(LocationAllocationProblem | RouteServiceProblem)


def _location_spec(**updates: object) -> SyntheticInstanceSpec:
    base = SyntheticInstanceSpec(
        family=ProblemFamily.LOCATION_ALLOCATION,
        problem_type="max_weighted_coverage",
        seed=41,
        distribution=SpatialDistribution.GRID,
        demand_count=5,
        candidate_count=4,
        site_limit=2,
        equity_structure=EquityStructure.BALANCED,
    )
    return base.model_copy(update=updates)


def test_auc_and_descriptive_statistics_match_hand_calculation() -> None:
    area = area_under_log_curve(((0, 1.0), (9, 2.0)), 99)
    assert area == pytest.approx(3.0 * math.log(10.0))
    assert area_under_log_curve(((0, 1.0), (0, 3.0)), 9) == pytest.approx(3.0 * math.log(10.0))
    mean, variance, half_width = descriptive_summary((1.0, 3.0))
    assert mean == 2.0
    assert variance == 2.0
    assert half_width == pytest.approx(1.96)
    assert incumbent_quality_at(((2, 1.0), (5, 3.0)), (0, 2, 4, 5, 9)) == {
        0: None,
        2: 1.0,
        4: 1.0,
        5: 3.0,
        9: 3.0,
    }
    with pytest.raises(ValueError, match="ordered"):
        area_under_log_curve(((2, 1.0), (1, 2.0)), 3)


def test_fixture_catalog_and_held_out_seed_namespace() -> None:
    assert set(fixture_catalog()) == {name.value for name in FixtureName}
    development = _location_spec(split=DatasetSplit.DEVELOPMENT)
    held_out = development.model_copy(update={"split": DatasetSplit.HELD_OUT})
    assert effective_seed(development) != effective_seed(held_out)
    assert split_seeds_are_disjoint(41)


@pytest.mark.parametrize("fixture", tuple(FixtureName))
@pytest.mark.asyncio
async def test_frozen_fixture_compiles_and_has_an_independent_reference(
    fixture: FixtureName, tmp_path
) -> None:
    store = LocalArtifactStore(tmp_path / fixture.value)
    generated = await generate_instance(fixture.value, load_fixture(fixture), store)
    assert generated.admitted
    referenced = attach_reference(
        generated,
        store,
        max_exact_candidates=50_000,
        max_reference_candidates=2_000,
    )
    assert referenced.reference is not None
    assert referenced.reference.scorecard is not None


@pytest.mark.parametrize("problem_type", tuple(LocationProblemType))
@pytest.mark.asyncio
async def test_location_generator_supports_every_registered_family(
    problem_type: LocationProblemType, tmp_path
) -> None:
    generated = await generate_instance(
        problem_type.value,
        _location_spec(problem_type=problem_type.value),
        LocalArtifactStore(tmp_path / problem_type.value),
    )
    assert generated.admitted


@pytest.mark.parametrize("problem_type", tuple(RouteProblemType))
@pytest.mark.asyncio
async def test_route_generator_supports_every_registered_family(
    problem_type: RouteProblemType, tmp_path
) -> None:
    generated = await generate_instance(
        problem_type.value,
        SyntheticInstanceSpec(
            family=ProblemFamily.ROUTING,
            problem_type=problem_type.value,
            seed=73,
            demand_count=3,
            candidate_count=3,
            site_limit=1,
            equity_structure=EquityStructure.NONE,
            directed_travel=True,
        ),
        LocalArtifactStore(tmp_path / problem_type.value),
    )
    assert generated.admitted


@pytest.mark.asyncio
async def test_generator_is_deterministic_and_oracle_uses_independent_evaluator(tmp_path) -> None:
    spec = _location_spec()
    first_store = LocalArtifactStore(tmp_path / "first")
    second_store = LocalArtifactStore(tmp_path / "second")
    first = await generate_instance("grid", spec, first_store)
    second = await generate_instance("grid", spec, second_store)
    assert first.problem_hash == second.problem_hash
    assert first.problem_artifact_id is not None
    assert second.problem_artifact_id is not None

    with_reference = attach_reference(
        first,
        first_store,
        max_exact_candidates=10_000,
        max_reference_candidates=100,
    )
    assert with_reference.reference is not None
    assert with_reference.reference.kind is ReferenceKind.EXACT_OPTIMUM
    assert with_reference.reference.plan is not None
    assert with_reference.reference.scorecard is not None
    problem = _PROBLEM.validate_python(read_json(first_store, first.problem_artifact_id))
    rescored = independently_evaluate_plan(problem, with_reference.reference.plan, first_store)
    assert rescored == with_reference.reference.scorecard

    invalid = Plan(problem_type=problem.type_id.value, selected_site_ids=("not-a-site",))
    invalid_score = independently_evaluate_plan(problem, invalid, first_store)
    assert not invalid_score.feasible
    assert invalid_score.violations

    claimed = with_reference.reference.plan.model_copy(
        update={"metadata": {"claimed_objective": 1_000_000.0}}
    )
    claimed_score = independently_evaluate_plan(problem, claimed, first_store)
    assert claimed_score.comparator_key == with_reference.reference.scorecard.comparator_key


@pytest.mark.asyncio
async def test_medium_reference_is_labeled_best_known(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    generated = await generate_instance(
        "medium",
        _location_spec(scale=InstanceScale.MEDIUM, candidate_count=8, site_limit=3),
        store,
    )
    referenced = attach_reference(
        generated,
        store,
        max_exact_candidates=1,
        max_reference_candidates=5,
    )
    assert referenced.reference is not None
    assert referenced.reference.kind is ReferenceKind.BEST_KNOWN
    assert referenced.reference.evaluated_candidates == 5


def test_manifest_validation_requires_explicit_real_model_opt_in() -> None:
    with pytest.raises(ValidationError, match="allow_real_model"):
        EvaluationModelSpec(backend="transformers")
    model = EvaluationModelSpec(backend="transformers", allow_real_model=True)
    manifest = BenchmarkManifest(
        benchmark_id="real-model",
        instances=(BenchmarkInstanceSpec(id="fixture", fixture=FixtureName.COOLING_CENTERS),),
        comparisons=(ComparisonKind.ONE_SHOT_MODEL,),
        budgets=(
            BenchmarkBudget(
                id="short",
                resources={
                    "wall_time_ms": 1_000,
                    "max_total_model_tokens": 100,
                    "max_generated_tokens": 50,
                    "max_tool_calls": 1,
                },
            ),
        ),
        model=model,
        expected_evaluator_versions={"*": "1.0.0"},
    )
    assert manifest.model.model_id is None
    duplicate_seeds = manifest.model_dump(mode="json")
    duplicate_seeds["run_seeds"] = [4, 4]
    with pytest.raises(ValidationError, match="run seeds must be unique"):
        BenchmarkManifest.model_validate(duplicate_seeds)
    primitive = manifest.model_dump(mode="json")
    primitive["track"] = "primitive_tools"
    primitive["tool_suite"] = ["exact_enumeration"]
    with pytest.raises(ValidationError, match="cannot expose exact"):
        BenchmarkManifest.model_validate(primitive)


@pytest.mark.asyncio
async def test_infeasible_equity_generator_preserves_failure_label(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    generated = await generate_instance(
        "infeasible",
        _location_spec(
            constraint_regime=ConstraintRegime.INFEASIBLE,
            equity_structure=EquityStructure.ISOLATED,
        ),
        store,
    )
    assert not generated.admitted
    assert generated.problem_artifact_id is not None
    assert "no_feasible_baseline" in generated.issue_codes
    referenced = attach_reference(
        generated,
        store,
        max_exact_candidates=10_000,
        max_reference_candidates=100,
    )
    assert referenced.reference is not None
    assert referenced.reference.kind is ReferenceKind.INFEASIBLE
