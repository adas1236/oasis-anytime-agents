from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from unit.test_location_allocation import problem_fixture, provenance

from oasis.artifacts import LocalArtifactStore, put_json
from oasis.controller import (
    AnytimeController,
    BudgetSpec,
    BudgetTier,
    ControllerEvent,
    ControllerPolicy,
    EventKind,
    LocalRunStore,
    RunRequest,
    RunStatus,
    TerminalReason,
)
from oasis.llm import FakeModelBackend, ModelRequest, ModelTurn, TokenUsage, ToolCall
from oasis.problems import LocationAllocationPolicy, LocationProblemType
from oasis.routing import run_routing_demo
from oasis.schemas import (
    ArtifactKind,
    DeterminismClassification,
    Plan,
    PrivacyClassification,
    SideEffectClassification,
    ToolEvent,
    ToolRuntimeEstimate,
    ToolSpec,
)
from oasis.tools import CancellationToken, ToolContext, ToolRegistry


class FakeClock:
    def __init__(self, value: float = 200.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance_ms(self, milliseconds: int) -> None:
        self.value += milliseconds / 1_000


class RecordingFakeModelBackend(FakeModelBackend):
    def __init__(self, responses: list[str | ToolCall]) -> None:
        super().__init__(responses)
        self.turn_usages: list[TokenUsage] = []

    async def generate(self, request: ModelRequest) -> ModelTurn:
        turn = await super().generate(request)
        self.turn_usages.append(turn.usage)
        return turn


def published_problem(
    tmp_path: Path,
    *,
    policy: LocationAllocationPolicy | None = None,
) -> tuple[LocalArtifactStore, str, str, Any]:
    artifact_store, problem = problem_fixture(
        tmp_path / "artifacts",
        LocationProblemType.MAX_WEIGHTED_COVERAGE,
        policy or LocationAllocationPolicy(site_limit=1),
    )
    problem_ref = put_json(
        artifact_store,
        problem.model_dump(mode="json"),
        kind=ArtifactKind.JSON_SPECIFICATION,
        units="unitless",
        provenance=provenance("phase7-problem"),
        data_schema={"type": "LocationAllocationProblem", "version": problem.schema_version},
    )
    deliberately_weak = Plan(
        problem_type=problem.type_id.value,
        selected_site_ids=("s1",),
    )
    plan_ref = put_json(
        artifact_store,
        deliberately_weak.model_dump(mode="json"),
        kind=ArtifactKind.PLAN,
        units="unitless",
        provenance=provenance("phase7-baseline"),
        data_schema={"type": "Plan", "version": deliberately_weak.schema_version},
    )
    return artifact_store, problem_ref.id, plan_ref.id, problem


def request(
    *,
    run_id: str,
    problem_id: str,
    baseline_id: str,
    budget: BudgetSpec,
    enable_model: bool = True,
    enable_fallback: bool = True,
    tier: BudgetTier | None = None,
) -> RunRequest:
    return RunRequest(
        run_id=run_id,
        problem_artifact_id=problem_id,
        baseline_plan_artifact_id=baseline_id,
        budget=budget,
        enable_model=enable_model,
        enable_deterministic_fallback=enable_fallback,
        requested_tier=tier,
        seed=19,
    )


@pytest.mark.asyncio
async def test_baseline_only_run_is_finalized_and_persisted(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, problem = published_problem(tmp_path)
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(artifact_store=artifacts, run_store=runs)

    result = await controller.run(
        request(
            run_id="baseline-only",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(wall_time_ms=1_000),
            enable_model=False,
        )
    )

    assert result.status is RunStatus.COMPLETE
    assert result.terminal_reason is TerminalReason.BASELINE_ONLY
    assert result.budget_tier is BudgetTier.BASELINE_ONLY
    assert result.best_plan is not None
    assert result.best_plan.selected_site_ids == ("s1",)
    assert result.best_scorecard is not None and result.best_scorecard.feasible
    assert result.problem_hash == problem.problem_hash
    assert result.consumed_budget.model_usage.total_tokens == 0
    assert result.result_view is not None and result.result_view.feasible
    assert runs.read_result(result.run_id) == result
    events = runs.read_events(result.run_id)
    assert [event.sequence for event in events] == list(range(len(events)))
    assert events[-1].kind is EventKind.RUN_FINALIZED


@pytest.mark.asyncio
async def test_same_controller_contract_finalizes_a_route_problem(tmp_path: Path) -> None:
    artifact_root = tmp_path / "route-artifacts"
    route = await run_routing_demo(artifact_root)
    controller = AnytimeController(
        artifact_store=LocalArtifactStore(artifact_root),
        run_store=LocalRunStore(tmp_path / "runs"),
    )

    result = await controller.run(
        request(
            run_id="route-baseline",
            problem_id=route.problem_artifact_id,
            baseline_id=route.baseline_plan_artifact_id,
            budget=BudgetSpec(wall_time_ms=1_000),
            enable_model=False,
        )
    )

    assert result.status is RunStatus.COMPLETE
    assert result.best_plan is not None and result.best_plan.routes
    assert result.result_view is not None and result.result_view.route_count == 1


@pytest.mark.asyncio
async def test_fake_model_real_tool_run_streams_and_commits_verified_optimum(
    tmp_path: Path,
) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    clock = FakeClock()
    backend = FakeModelBackend(
        [
            ToolCall(
                id="exact-1",
                name="improve",
                arguments={"strategy": "exact_enumeration", "max_candidates": 100},
            )
        ]
    )
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        backend=backend,
        monotonic=clock,
    )

    result = await controller.run(
        request(
            run_id="fake-exact",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=30_000,
                max_total_model_tokens=5_000,
                max_generated_tokens=1_000,
                max_tool_calls=1,
            ),
        )
    )

    assert result.terminal_reason is TerminalReason.PROVEN_OPTIMAL
    assert result.budget_tier is BudgetTier.ITERATIVE_MODEL
    assert result.best_plan is not None
    assert result.best_plan.selected_site_ids in {("s2",), ("s3",)}
    assert result.baseline_comparison is not None
    assert result.baseline_comparison.value == "better"
    assert result.verified_bound_artifact_id is not None
    assert result.consumed_budget.tool_calls == 1
    assert result.consumed_budget.model_usage.total_tokens > 0
    kinds = [event.kind for event in runs.read_events(result.run_id)]
    assert EventKind.INCUMBENT_COMMITTED in kinds
    assert EventKind.BOUND_VERIFIED in kinds


@pytest.mark.asyncio
async def test_malformed_model_action_gets_one_bounded_repair(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    backend = RecordingFakeModelBackend(
        ["not-json", '{"type":"stop","rationale":"baseline is sufficient"}']
    )
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        backend=backend,
    )

    result = await controller.run(
        request(
            run_id="repair",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=2_000,
                max_total_model_tokens=500,
                max_generated_tokens=50,
                max_tool_calls=1,
            ),
            enable_fallback=False,
        )
    )

    assert result.terminal_reason is TerminalReason.MODEL_STOPPED
    assert result.budget_tier is BudgetTier.ONE_SHOT_MODEL
    assert len([failure for failure in result.failures if "malformed model action" in failure]) == 1
    rejected = [
        event
        for event in runs.read_events(result.run_id)
        if event.kind is EventKind.ACTION_REJECTED
    ]
    assert len(rejected) == 1
    assert result.consumed_budget.model_usage.generated_tokens <= 50
    assert result.consumed_budget.model_usage.total_tokens <= 500
    assert result.consumed_budget.model_usage == TokenUsage.aggregate(backend.turn_usages)


@pytest.mark.asyncio
async def test_invalid_direct_candidate_never_replaces_baseline(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    invalid_action = (
        '{"type":"submit_candidate","candidate":{"problem_type":"max_weighted_coverage",'
        '"selected_site_ids":["missing"]},"rationale":"try an unknown site"}'
    )
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        backend=FakeModelBackend([invalid_action]),
    )

    result = await controller.run(
        request(
            run_id="invalid-candidate",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=2_000,
                max_total_model_tokens=500,
                max_generated_tokens=50,
                max_tool_calls=1,
            ),
            enable_fallback=False,
        )
    )

    assert result.best_plan is not None
    assert result.best_plan.selected_site_ids == ("s1",)
    assert len(result.incumbent_timeline) == 1
    assert EventKind.CANDIDATE_REJECTED in {event.kind for event in runs.read_events(result.run_id)}


@pytest.mark.asyncio
async def test_token_exhaustion_activates_deterministic_fallback(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    backend = FakeModelBackend(
        [
            ToolCall(
                id="local-1",
                name="improve",
                arguments={"strategy": "add_swap", "max_candidates": 1},
            )
        ]
    )
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        backend=backend,
        policy=ControllerPolicy(one_shot_total_token_threshold=1),
    )

    result = await controller.run(
        request(
            run_id="token-fallback",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=30_000,
                max_total_model_tokens=35,
                max_generated_tokens=3,
                max_tool_calls=4,
            ),
        )
    )

    assert result.best_plan is not None and result.best_scorecard is not None
    assert result.consumed_budget.model_usage.total_tokens <= 35
    assert result.consumed_budget.model_usage.generated_tokens <= 3
    assert EventKind.FALLBACK_INVOKED in {event.kind for event in runs.read_events(result.run_id)}


@pytest.mark.asyncio
async def test_reserve_prevents_action_admission_after_search_deadline(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    clock = FakeClock()
    cancellation = CancellationToken()

    async def advance_at_search(event: ControllerEvent) -> None:
        if (
            event.kind is EventKind.BUDGET_CHECKPOINT
            and event.payload.get("checkpoint") == "search_started"
        ):
            clock.advance_ms(91)

    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        monotonic=clock,
        event_callback=advance_at_search,
        policy=ControllerPolicy(
            minimum_finalization_reserve_ms=10,
            maximum_finalization_reserve_ms=10,
            minimum_baseline_budget_ms=1,
        ),
    )

    result = await controller.run(
        request(
            run_id="reserve",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(wall_time_ms=100, max_tool_calls=1),
            enable_model=False,
        ),
        cancellation=cancellation,
    )

    assert result.terminal_reason is TerminalReason.TIME_EXHAUSTED
    assert result.best_plan is not None
    assert result.deadline_overshoot_ms == 0
    assert result.consumed_budget.search_remaining_ms == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_point", ["baseline", "search", "tool"])
async def test_interruption_after_baseline_always_returns_a_verified_incumbent(
    tmp_path: Path, cancel_point: str
) -> None:
    artifacts, problem_id, baseline_id, problem = published_problem(tmp_path)
    cancellation = CancellationToken()

    async def cancel_on_event(event: ControllerEvent) -> None:
        matches = (
            (cancel_point == "baseline" and event.kind is EventKind.BASELINE_COMMITTED)
            or (
                cancel_point == "search"
                and event.kind is EventKind.BUDGET_CHECKPOINT
                and event.payload.get("checkpoint") == "search_started"
            )
            or (cancel_point == "tool" and event.kind is EventKind.TOOL_STARTED)
        )
        if matches:
            cancellation.cancel(f"interrupt at {cancel_point}")

    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
        backend=FakeModelBackend(
            [
                ToolCall(
                    id="interruptible",
                    name="improve",
                    arguments={"strategy": "exact_enumeration", "max_candidates": 100},
                )
            ]
        ),
        event_callback=cancel_on_event,
    )
    result = await controller.run(
        request(
            run_id=f"interrupt-{cancel_point}",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=30_000,
                max_total_model_tokens=1_000,
                max_generated_tokens=100,
                max_tool_calls=2,
            ),
        ),
        cancellation=cancellation,
    )

    assert result.status is RunStatus.CANCELLED
    assert result.terminal_reason is TerminalReason.USER_CANCELLED
    assert result.best_plan is not None and result.best_scorecard is not None
    plugin = controller._problems.get(problem.type_id.value)
    assert plugin.validate_plan(problem, result.best_plan, artifacts).valid
    assert result.best_scorecard.feasible


class HangingImproveTool:
    spec = ToolSpec(
        name="improve",
        version="1.0.0",
        description="Hang until the controller-enforced action subdeadline.",
        input_schema={"type": "object", "additionalProperties": True},
        output_schema={"type": "object"},
        capability_tags=frozenset({"decision", "search", "offline"}),
        problem_tags=frozenset({"location_allocation"}),
        side_effects=SideEffectClassification.NONE,
        determinism=DeterminismClassification.DETERMINISTIC,
        runtime=ToolRuntimeEstimate(p50_ms=1, p95_ms=10, time_to_first_candidate_ms=1),
        streams_candidates=True,
        smoke_input={},
    )

    async def stream(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        del arguments, context
        await CancellationToken().wait()
        if False:
            yield ToolEvent.model_construct()


class CrashingImproveTool(HangingImproveTool):
    spec = HangingImproveTool.spec.model_copy(
        update={
            "description": "Crash inside a streamed tool invocation.",
            "runtime": ToolRuntimeEstimate(
                p50_ms=10,
                p95_ms=100,
                time_to_first_candidate_ms=10,
            ),
        }
    )

    async def stream(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> AsyncIterator[ToolEvent]:
        del arguments, context
        if False:
            yield ToolEvent.model_construct()
        raise RuntimeError("scripted tool crash")


class RestrictedImproveTool(HangingImproveTool):
    spec = HangingImproveTool.spec.model_copy(
        update={
            "description": "Require an unavailable private resource.",
            "privacy": PrivacyClassification.INTERNAL,
            "required_resources": frozenset({"private_solver"}),
        }
    )


class HangingModelBackend(FakeModelBackend):
    def __init__(self) -> None:
        super().__init__()
        self.abort_called = False

    async def generate(self, request: ModelRequest) -> ModelTurn:
        del request
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def abort(self, request_id: str) -> None:
        del request_id
        self.abort_called = True


class CrashingModelBackend(FakeModelBackend):
    async def generate(self, request: ModelRequest) -> ModelTurn:
        del request
        raise RuntimeError("scripted model crash")


@pytest.mark.asyncio
async def test_hanging_tool_is_cancelled_at_its_subdeadline(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    runs = LocalRunStore(tmp_path / "runs")
    tools = ToolRegistry((HangingImproveTool(),))
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        backend=FakeModelBackend(
            [ToolCall(id="hang", name="improve", arguments={"strategy": "add_swap"})]
        ),
        tool_registry=tools,
        policy=ControllerPolicy(
            minimum_finalization_reserve_ms=20,
            maximum_finalization_reserve_ms=20,
            minimum_baseline_budget_ms=1,
            cancellation_grace_ms=2,
        ),
    )

    result = await controller.run(
        request(
            run_id="tool-hang",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=1_000,
                max_total_model_tokens=500,
                max_generated_tokens=50,
                max_tool_calls=1,
            ),
            enable_fallback=False,
        )
    )

    assert result.best_plan is not None
    assert EventKind.TOOL_CANCELLED in {event.kind for event in runs.read_events(result.run_id)}
    assert result.deadline_overshoot_ms == 0


@pytest.mark.asyncio
async def test_crashing_tool_cannot_erase_the_committed_baseline(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        backend=FakeModelBackend(
            [ToolCall(id="crash", name="improve", arguments={"strategy": "add_swap"})]
        ),
        tool_registry=ToolRegistry((CrashingImproveTool(),)),
    )

    result = await controller.run(
        request(
            run_id="tool-crash",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=1_000,
                max_total_model_tokens=500,
                max_generated_tokens=50,
                max_tool_calls=1,
            ),
            enable_fallback=False,
        )
    )

    assert result.best_plan is not None
    assert result.best_plan.selected_site_ids == ("s1",)
    assert EventKind.TOOL_FAILED in {event.kind for event in runs.read_events(result.run_id)}
    assert any("scripted tool crash" in failure for failure in result.failures)


@pytest.mark.asyncio
async def test_prerequisite_and_privacy_failure_is_rejected_before_tool_admission(
    tmp_path: Path,
) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        backend=FakeModelBackend(
            [ToolCall(id="restricted", name="improve", arguments={"strategy": "add_swap"})]
        ),
        tool_registry=ToolRegistry((RestrictedImproveTool(),)),
    )

    result = await controller.run(
        request(
            run_id="restricted-tool",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=1_000,
                max_total_model_tokens=500,
                max_generated_tokens=50,
                max_tool_calls=1,
            ),
            enable_fallback=False,
        )
    )

    assert result.best_plan is not None
    assert result.consumed_budget.tool_calls == 0
    assert any("privacy permission" in failure for failure in result.failures)
    events = runs.read_events(result.run_id)
    assert EventKind.ACTION_ADMITTED not in {event.kind for event in events}
    assert any(
        event.kind is EventKind.ACTION_REJECTED
        and event.payload.get("reason") == "invalid_tool_action"
        for event in events
    )


@pytest.mark.asyncio
async def test_hanging_model_is_aborted_without_consuming_finalization_reserve(
    tmp_path: Path,
) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    backend = HangingModelBackend()
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
        backend=backend,
        policy=ControllerPolicy(
            minimum_finalization_reserve_ms=80,
            maximum_finalization_reserve_ms=80,
            minimum_baseline_budget_ms=1,
            cancellation_grace_ms=2,
        ),
    )

    result = await controller.run(
        request(
            run_id="model-hang",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=200,
                max_total_model_tokens=500,
                max_generated_tokens=50,
                max_tool_calls=1,
            ),
            enable_fallback=False,
        )
    )

    assert backend.abort_called
    assert result.best_plan is not None
    assert result.deadline_overshoot_ms == 0
    assert result.consumed_budget.model_usage.total_tokens == 0


@pytest.mark.asyncio
async def test_crashing_model_returns_the_committed_baseline(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    result = await AnytimeController(
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
        backend=CrashingModelBackend(),
    ).run(
        request(
            run_id="model-crash",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=1_000,
                max_total_model_tokens=500,
                max_generated_tokens=50,
                max_tool_calls=1,
            ),
            enable_fallback=False,
        )
    )

    assert result.best_plan is not None
    assert result.best_plan.selected_site_ids == ("s1",)
    assert any("scripted model crash" in failure for failure in result.failures)
    assert result.consumed_budget.model_usage.total_tokens == 0


@pytest.mark.asyncio
async def test_zero_model_budget_still_allows_deterministic_improvement(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    clock = FakeClock()
    result = await AnytimeController(
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
        monotonic=clock,
    ).run(
        request(
            run_id="zero-model-budget",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(wall_time_ms=30_000, max_tool_calls=4),
        )
    )

    assert result.budget_tier is BudgetTier.DETERMINISTIC_IMPROVEMENT
    assert result.consumed_budget.model_usage.total_tokens == 0
    assert result.best_plan is not None
    assert result.best_plan.selected_site_ids in {("s2",), ("s3",)}
    assert result.baseline_comparison is not None
    assert result.baseline_comparison.value == "better"


@pytest.mark.asyncio
async def test_too_small_and_missing_evidence_requests_return_structured_failures(
    tmp_path: Path,
) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=LocalRunStore(tmp_path / "runs"),
        policy=ControllerPolicy(
            minimum_finalization_reserve_ms=9,
            maximum_finalization_reserve_ms=9,
            minimum_baseline_budget_ms=5,
        ),
    )
    too_small = await controller.run(
        request(
            run_id="too-small",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(wall_time_ms=10),
            enable_model=False,
        )
    )
    missing = await controller.run(
        request(
            run_id="missing-evidence",
            problem_id="sha256-" + "0" * 64,
            baseline_id=baseline_id,
            budget=BudgetSpec(wall_time_ms=1_000),
            enable_model=False,
        )
    )

    assert too_small.status is RunStatus.REJECTED
    assert too_small.terminal_reason is TerminalReason.BUDGET_TOO_SMALL
    assert too_small.best_plan is None
    assert missing.status is RunStatus.REJECTED
    assert missing.terminal_reason is TerminalReason.MISSING_EVIDENCE
    assert missing.best_plan is None


@pytest.mark.asyncio
async def test_distinct_policy_hashes_keep_separate_incumbent_artifacts(tmp_path: Path) -> None:
    first_store, first_problem_id, first_baseline_id, first_problem = published_problem(tmp_path)
    second_store, second_problem_id, second_baseline_id, second_problem = published_problem(
        tmp_path,
        policy=LocationAllocationPolicy(site_limit=2),
    )
    assert first_store.root == second_store.root
    assert first_problem.policy_hash != second_problem.policy_hash
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(artifact_store=first_store, run_store=runs)

    first = await controller.run(
        request(
            run_id="policy-one",
            problem_id=first_problem_id,
            baseline_id=first_baseline_id,
            budget=BudgetSpec(wall_time_ms=1_000),
            enable_model=False,
        )
    )
    second = await controller.run(
        request(
            run_id="policy-two",
            problem_id=second_problem_id,
            baseline_id=second_baseline_id,
            budget=BudgetSpec(wall_time_ms=1_000),
            enable_model=False,
        )
    )

    assert first.problem_hash != second.problem_hash
    assert first.best_plan_artifact_id != second.best_plan_artifact_id
    assert first.incumbent_timeline[0].policy_hash != second.incumbent_timeline[0].policy_hash


@pytest.mark.asyncio
async def test_duplicate_model_actions_trigger_fallback_circuit_breaker(tmp_path: Path) -> None:
    artifacts, problem_id, baseline_id, _ = published_problem(tmp_path)
    duplicate = ToolCall(
        id="same",
        name="improve",
        arguments={"strategy": "multi_swap", "max_candidates": 1},
    )
    runs = LocalRunStore(tmp_path / "runs")
    controller = AnytimeController(
        artifact_store=artifacts,
        run_store=runs,
        backend=FakeModelBackend([duplicate, duplicate, duplicate]),
        policy=ControllerPolicy(
            one_shot_total_token_threshold=1,
            max_no_progress_actions=2,
        ),
    )

    result = await controller.run(
        request(
            run_id="duplicates",
            problem_id=problem_id,
            baseline_id=baseline_id,
            budget=BudgetSpec(
                wall_time_ms=30_000,
                max_total_model_tokens=2_000,
                max_generated_tokens=100,
                max_tool_calls=4,
            ),
        )
    )

    events = runs.read_events(result.run_id)
    assert any(
        event.kind is EventKind.ACTION_REJECTED
        and event.payload.get("reason") == "duplicate_action"
        for event in events
    )
    assert EventKind.FALLBACK_INVOKED in {event.kind for event in events}
