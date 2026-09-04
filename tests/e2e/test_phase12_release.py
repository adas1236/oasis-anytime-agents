from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from oasis.artifacts import LocalArtifactStore, read_json
from oasis.controller import (
    AnytimeController,
    BudgetSpec,
    ControllerEvent,
    ControllerPolicy,
    EventKind,
    InMemoryRunStore,
    RunRequest,
    RunStatus,
    TerminalReason,
)
from oasis.evaluation import (
    FixtureName,
    GeneratedInstance,
    generate_instance,
    load_fixture,
    read_run_records,
)
from oasis.failure_injection import (
    FailureInjectingImproveTool,
    FailureInjectingModelBackend,
    FailureInjection,
    FailureMode,
    ScriptedSourceProvider,
    cancel_on_tool_start,
    provider_outage,
)
from oasis.llm import FakeModelBackend, ToolCall
from oasis.problems import (
    LocationAllocationProblem,
    RouteServiceProblem,
    create_builtin_problem_registry,
)
from oasis.providers import MemorySnapshotCache, ProviderProvenance, RetrievedSource
from oasis.schemas import ToolResultStatus
from oasis.showcase import ShowcaseReport, load_showcase_manifest, run_showcase
from oasis.tools import CancellationToken, ToolContext, ToolRegistry, invoke_tool
from oasis.tools.providers import SNAPSHOT_CACHE, SOURCE_PROVIDER, SnapshotSourceTool


@pytest.mark.asyncio
async def test_offline_showcase_is_complete_profiled_and_warm_resumable(tmp_path: Path) -> None:
    report = await run_showcase(tmp_path)
    run_files = sorted((tmp_path / "runs").glob("*.json"))
    before = {path.name: path.read_bytes() for path in run_files}

    resumed = await run_showcase(tmp_path)

    assert resumed == report
    assert before == {path.name: path.read_bytes() for path in run_files}
    assert (
        ShowcaseReport.model_validate_json((tmp_path / "release-report.json").read_text()) == report
    )
    assert report.benchmark_summary.run_count == 21
    assert {instance.problem_type for instance in report.instances} == set(
        report.dataset.included_problem_types
    )
    assert all(instance.raw_objective and instance.overall_metrics for instance in report.instances)
    assert sum(len(instance.comparator_history) > 1 for instance in report.instances) >= 3
    assert report.profile.model_action_count > 0
    assert report.profile.compact_context_within_limit
    assert report.profile.max_deadline_overshoot_ms == 0
    assert report.profile.max_artifact_bytes_per_run > 0
    assert report.profile.tool_latency["improve"].declared_p50_ms == 125
    assert report.profile.tool_latency["improve"].within_declared_p95
    assert report.metadata_audit.all_public
    assert report.metadata_audit.all_sources_attributed
    assert report.metadata_audit.all_licenses_declared
    assert report.metadata_audit.no_sensitive_metadata_keys
    assert report.metadata_audit.reproducibility_complete
    assert report.hardware_validation["cpu_fake"] == "passed"
    assert report.hardware_validation["local_cuda"].startswith("passed: RTX 5060 Ti 16 GB")
    assert "monitor_placement" in report.dataset.deferred_scenarios

    records = read_run_records(tmp_path)
    assert all(record.feasible for record in records)
    assert all(record.deadline_overshoot_ms == 0 for record in records)


def test_showcase_manifest_locks_documented_release_budget_and_problem_suite() -> None:
    manifest = load_showcase_manifest()
    budget = manifest.budgets[0].resources
    injection = FailureInjection(mode=FailureMode.MODEL_UNAVAILABLE)

    assert manifest.benchmark_id == "track-b-showcase-v1"
    assert len(manifest.instances) == 7
    assert budget == BudgetSpec(
        wall_time_ms=30_000,
        max_total_model_tokens=4_096,
        max_generated_tokens=1_024,
        max_tool_calls=8,
    )
    assert FailureInjection.model_validate_json(injection.model_dump_json()) == injection


async def _generated_location(tmp_path: Path) -> tuple[LocalArtifactStore, GeneratedInstance]:
    store = LocalArtifactStore(tmp_path / "artifacts")
    instance = await generate_instance(
        "failure-matrix",
        load_fixture(FixtureName.CLINIC_ACCESS),
        store,
    )
    assert instance.problem_artifact_id is not None
    assert instance.baseline_plan_artifact_id is not None
    return store, instance


def _run_request(instance: GeneratedInstance, run_id: str) -> RunRequest:
    problem_id = instance.problem_artifact_id
    baseline_id = instance.baseline_plan_artifact_id
    assert isinstance(problem_id, str) and isinstance(baseline_id, str)
    return RunRequest(
        run_id=run_id,
        problem_artifact_id=problem_id,
        baseline_plan_artifact_id=baseline_id,
        budget=BudgetSpec(
            wall_time_ms=1_000,
            max_total_model_tokens=500,
            max_generated_tokens=50,
            max_tool_calls=3,
        ),
        enable_model=True,
        enable_deterministic_fallback=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        FailureMode.MALFORMED_MODEL_CALL,
        FailureMode.MODEL_UNAVAILABLE,
        FailureMode.MODEL_OOM,
        FailureMode.MODEL_ERROR,
    ],
)
async def test_model_failure_injection_retains_baseline_and_uses_deterministic_fallback(
    tmp_path: Path,
    mode: FailureMode,
) -> None:
    store, instance = await _generated_location(tmp_path)
    backend = FailureInjectingModelBackend(FailureInjection(mode=mode))
    result = await AnytimeController(
        artifact_store=store,
        run_store=InMemoryRunStore(),
        backend=backend,
        policy=ControllerPolicy(max_no_progress_actions=1),
    ).run(_run_request(instance, f"injected-{mode.value}"))

    assert result.best_plan is not None and result.best_scorecard is not None
    assert result.best_scorecard.feasible
    assert result.deadline_overshoot_ms == 0
    assert result.failures
    if mode is FailureMode.MALFORMED_MODEL_CALL:
        assert any("malformed model action" in failure for failure in result.failures)
    elif mode is FailureMode.MODEL_OOM:
        assert any("out-of-memory" in failure for failure in result.failures)
    elif mode is FailureMode.MODEL_UNAVAILABLE:
        assert any("unavailable" in failure for failure in result.failures)
    else:
        assert any("release-hardening failure" in failure for failure in result.failures)


@pytest.mark.asyncio
async def test_tool_timeout_and_mid_tool_user_cancellation_retain_verified_plan(
    tmp_path: Path,
) -> None:
    store, instance = await _generated_location(tmp_path)

    def controller(callback: object = None) -> AnytimeController:
        return AnytimeController(
            artifact_store=store,
            run_store=InMemoryRunStore(),
            backend=FakeModelBackend(
                [ToolCall(id="injected", name="improve", arguments={"strategy": "add_swap"})]
            ),
            tool_registry=ToolRegistry(
                (FailureInjectingImproveTool(FailureInjection(mode=FailureMode.TOOL_TIMEOUT)),)
            ),
            event_callback=callback,  # type: ignore[arg-type]
        )

    timed_out = await controller().run(_run_request(instance, "injected-tool-timeout"))
    assert timed_out.best_scorecard is not None and timed_out.best_scorecard.feasible
    assert timed_out.deadline_overshoot_ms == 0

    cancellation = CancellationToken()

    cancelled = await controller(cancel_on_tool_start(cancellation)).run(
        _run_request(instance, "injected-user-cancel"), cancellation=cancellation
    )
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.terminal_reason is TerminalReason.USER_CANCELLED
    assert cancelled.best_scorecard is not None and cancelled.best_scorecard.feasible
    assert cancelled.deadline_overshoot_ms == 0


@pytest.mark.asyncio
async def test_provider_outage_and_stale_cache_are_reproducible_and_labeled(
    tmp_path: Path,
) -> None:
    source = RetrievedSource(
        content=b"id,need\na,10\nb,20\n",
        media_type="text/csv",
        provenance=ProviderProvenance(
            provider="phase12-frozen-provider",
            source_uri="https://data.example.test/health.csv",
            retrieved_at=datetime.now(UTC),
            source_version="snapshot-v1",
            license="CC0-1.0",
        ),
    )
    provider = ScriptedSourceProvider((source, provider_outage()))
    cache = MemorySnapshotCache()
    context = ToolContext(
        run_id="phase12-stale-cache",
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        deadline_monotonic=time.monotonic() + 5,
        cancellation=CancellationToken(),
        seed=0,
        providers={SOURCE_PROVIDER: provider},
        resources={SNAPSHOT_CACHE: cache},
    )
    arguments = {
        "url": "https://data.example.test/health.csv",
        "format": "csv",
        "license": "CC0-1.0",
        "units": "people",
        "fresh_for_seconds": 0,
        "max_stale_seconds": 60,
    }
    fresh = await invoke_tool(SnapshotSourceTool(), arguments, context)
    await asyncio.sleep(0.002)
    stale = await invoke_tool(SnapshotSourceTool(), arguments, context)

    assert fresh.status is ToolResultStatus.COMPLETE
    assert stale.status is ToolResultStatus.PARTIAL
    assert stale.metrics["stale"] is True
    assert stale.metrics["cache_status"] == "stale_fallback"
    assert stale.artifacts[0].lineage.parent_ids == (fresh.artifacts[0].id,)
    assert "stale" in " ".join(stale.artifacts[0].quality.warnings)

    outage = await invoke_tool(
        SnapshotSourceTool(),
        arguments,
        ToolContext(
            run_id="phase12-provider-outage",
            artifact_store=LocalArtifactStore(tmp_path / "outage"),
            deadline_monotonic=time.monotonic() + 5,
            cancellation=CancellationToken(),
            seed=0,
            providers={SOURCE_PROVIDER: ScriptedSourceProvider((provider_outage(),))},
            resources={SNAPSHOT_CACHE: MemorySnapshotCache()},
        ),
    )
    assert outage.status is ToolResultStatus.FAILED
    assert outage.error is not None and outage.error.code.value == "provider_failure"
    assert outage.artifacts == ()


@pytest.mark.asyncio
async def test_routing_interruption_after_baseline_returns_independently_valid_plan(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    generated = await generate_instance(
        "routing-interruption",
        load_fixture(FixtureName.MOBILE_ROUTING),
        store,
    )
    assert generated.problem_artifact_id and generated.baseline_plan_artifact_id
    cancellation = CancellationToken()

    async def cancel_after_baseline(event: ControllerEvent) -> None:
        if event.kind is EventKind.BASELINE_COMMITTED:
            cancellation.cancel("interrupt after baseline")

    controller = AnytimeController(
        artifact_store=store,
        run_store=InMemoryRunStore(),
        event_callback=cancel_after_baseline,
    )
    result = await controller.run(
        RunRequest(
            run_id="routing-interruption",
            problem_artifact_id=generated.problem_artifact_id,
            baseline_plan_artifact_id=generated.baseline_plan_artifact_id,
            budget=BudgetSpec(wall_time_ms=1_000, max_tool_calls=2),
            enable_model=False,
        ),
        cancellation=cancellation,
    )
    assert result.best_plan is not None
    problem = RouteServiceProblem.model_validate(read_json(store, generated.problem_artifact_id))
    score = (
        create_builtin_problem_registry()
        .get(problem.type_id.value)
        .measure(problem, result.best_plan, store)
    )
    assert score.feasible
    assert score == result.best_scorecard


@pytest.mark.asyncio
async def test_model_token_exhaustion_invokes_deterministic_fallback(tmp_path: Path) -> None:
    store, instance = await _generated_location(tmp_path)
    runs = InMemoryRunStore()
    controller = AnytimeController(
        artifact_store=store,
        run_store=runs,
        backend=FakeModelBackend(
            [
                ToolCall(
                    id="token-limited",
                    name="improve",
                    arguments={"strategy": "add_swap", "max_candidates": 1},
                )
            ]
        ),
        policy=ControllerPolicy(one_shot_total_token_threshold=1),
    )
    request = _run_request(instance, "phase12-token-exhaustion").model_copy(
        update={
            "budget": BudgetSpec(
                wall_time_ms=30_000,
                max_total_model_tokens=35,
                max_generated_tokens=3,
                max_tool_calls=4,
            )
        }
    )

    result = await controller.run(request)

    assert result.best_scorecard is not None and result.best_scorecard.feasible
    assert result.consumed_budget.model_usage.total_tokens <= 35
    assert result.consumed_budget.model_usage.generated_tokens <= 3
    assert EventKind.FALLBACK_INVOKED in {event.kind for event in runs.read_events(result.run_id)}


@pytest.mark.asyncio
async def test_evidence_change_produces_a_distinct_problem_hash(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    original_spec = load_fixture(FixtureName.COOLING_CENTERS)
    changed_spec = original_spec.model_copy(update={"seed": original_spec.seed + 1})
    original = await generate_instance("original-evidence", original_spec, store)
    changed = await generate_instance("changed-evidence", changed_spec, store)
    assert original.problem_artifact_id and changed.problem_artifact_id
    original_problem = LocationAllocationProblem.model_validate(
        read_json(store, original.problem_artifact_id)
    )
    changed_problem = LocationAllocationProblem.model_validate(
        read_json(store, changed.problem_artifact_id)
    )

    assert original_problem.policy_hash == changed_problem.policy_hash
    assert original_problem.evidence_hash != changed_problem.evidence_hash
    assert original_problem.problem_hash != changed_problem.problem_hash
