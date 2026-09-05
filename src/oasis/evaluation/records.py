"""Derive benchmark measurements only from evaluator and controller records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from oasis.artifacts import ArtifactStore, read_json
from oasis.controller import ControllerEvent, EventKind, RunResult
from oasis.evaluation.metrics import (
    area_under_log_curve,
    incumbent_quality_at,
    normalized_gap_closed,
    relative_gain,
)
from oasis.evaluation.models import (
    BenchmarkRunRecord,
    BenchmarkTrack,
    ComparisonKind,
    GeneratedInstance,
    ProblemFamily,
    QualityCheckpoint,
    ReferenceKind,
)
from oasis.problems import (
    LocationAllocationProblem,
    RouteServiceProblem,
    Scorecard,
    create_builtin_problem_registry,
)
from oasis.runtimes import RuntimeKind, evaluation_group_key
from oasis.schemas import Plan


def independently_evaluate_plan(
    problem: LocationAllocationProblem | RouteServiceProblem,
    plan: Plan,
    store: ArtifactStore,
) -> Scorecard:
    """Validate and measure a plan without repair or use of claimed objective fields."""

    plugin = create_builtin_problem_registry().get(problem.type_id.value)
    return plugin.measure(problem, plan, store)


@dataclass(frozen=True, slots=True)
class _EventProfile:
    invalid_candidates: int
    rejected_candidates: int
    model_failures: int
    repairs: int
    tool_latency_ms: int
    tool_calls: int
    tool_failures: int
    model_input_tokens: tuple[int, ...]
    model_generated_tokens: tuple[int, ...]
    compact_context_bytes: tuple[int, ...]
    tool_latency_observations_ms: dict[str, tuple[int, ...]]
    tool_p50_estimates_ms: dict[str, int]
    tool_p95_estimates_ms: dict[str, int]


def _event_profile(events: Sequence[ControllerEvent]) -> _EventProfile:
    rejected = tuple(event for event in events if event.kind is EventKind.CANDIDATE_REJECTED)
    invalid = sum(event.payload.get("reason") == "validation_failed" for event in rejected)
    model_failures = sum(
        event.kind is EventKind.ACTION_REJECTED and event.payload.get("reason") == "model_failure"
        for event in events
    )
    repairs = sum(
        event.kind is EventKind.ACTION_REJECTED
        and event.payload.get("reason") == "malformed_action"
        for event in events
    )
    started: dict[tuple[str, int], tuple[int, str]] = {}
    observations: dict[str, list[int]] = {}
    p50_estimates: dict[str, int] = {}
    p95_estimates: dict[str, int] = {}
    latency = 0
    terminal_kinds = {
        EventKind.TOOL_COMPLETED,
        EventKind.TOOL_CANCELLED,
        EventKind.TOOL_FAILED,
    }
    for event in events:
        if event.action_id is None or event.action_generation is None:
            continue
        key = (event.action_id, event.action_generation)
        if event.kind is EventKind.TOOL_STARTED:
            tool_name = str(event.payload.get("tool", "unknown"))
            started[key] = (event.relative_monotonic_ms, tool_name)
            p50_estimate = event.payload.get("estimated_p50_ms")
            if isinstance(p50_estimate, int):
                p50_estimates[tool_name] = p50_estimate
            estimate = event.payload.get("estimated_p95_ms")
            if isinstance(estimate, int):
                p95_estimates[tool_name] = estimate
        elif event.kind in terminal_kinds and key in started:
            started_at, tool_name = started.pop(key)
            observed = max(0, event.relative_monotonic_ms - started_at)
            latency += observed
            observations.setdefault(tool_name, []).append(observed)
    tool_calls = events[-1].budget_after.tool_calls if events else 0
    tool_failures = sum(event.kind is EventKind.TOOL_FAILED for event in events)
    model_events = tuple(
        event
        for event in events
        if event.actor.value == "model"
        and event.kind in {EventKind.MODEL_ACTION_PROPOSED, EventKind.ACTION_REJECTED}
    )
    return _EventProfile(
        invalid_candidates=invalid,
        rejected_candidates=len(rejected),
        model_failures=model_failures,
        repairs=repairs,
        tool_latency_ms=latency,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        model_input_tokens=tuple(
            max(
                0,
                event.budget_after.model_usage.input_tokens
                - event.budget_before.model_usage.input_tokens,
            )
            for event in model_events
        ),
        model_generated_tokens=tuple(
            max(
                0,
                event.budget_after.model_usage.generated_tokens
                - event.budget_before.model_usage.generated_tokens,
            )
            for event in model_events
        ),
        compact_context_bytes=tuple(
            value
            for event in model_events
            if isinstance((value := event.payload.get("compact_context_bytes")), int)
        ),
        tool_latency_observations_ms={
            name: tuple(values) for name, values in sorted(observations.items())
        },
        tool_p50_estimates_ms=dict(sorted(p50_estimates.items())),
        tool_p95_estimates_ms=dict(sorted(p95_estimates.items())),
    )


def _artifact_profile(
    result: RunResult,
    events: Sequence[ControllerEvent],
    store: ArtifactStore,
) -> tuple[int, int, int]:
    pending = {
        *(event_id for event in events for event_id in event.artifact_ids),
        *(incumbent.plan_artifact_id for incumbent in result.incumbent_timeline),
        *(incumbent.scorecard_artifact_id for incumbent in result.incumbent_timeline),
    }
    if result.problem_artifact_id is not None:
        pending.add(result.problem_artifact_id)
    if result.verified_bound_artifact_id is not None:
        pending.add(result.verified_bound_artifact_id)
    visited: set[str] = set()
    sizes: list[int] = []
    while pending:
        artifact_id = pending.pop()
        if artifact_id in visited:
            continue
        reference = store.get_metadata(artifact_id)
        visited.add(artifact_id)
        sizes.append(reference.byte_size)
        pending.update(reference.lineage.parent_ids)
    return len(sizes), sum(sizes), max(sizes, default=0)


def _checkpoint_events(events: Sequence[ControllerEvent]) -> dict[str, ControllerEvent]:
    result: dict[str, ControllerEvent] = {}
    for event in events:
        if event.kind not in {EventKind.BASELINE_COMMITTED, EventKind.INCUMBENT_COMMITTED}:
            continue
        for artifact_id in event.artifact_ids:
            result[artifact_id] = event
    return result


def _checkpoints(
    result: RunResult,
    events: Sequence[ControllerEvent],
    reference_quality: float | None,
) -> tuple[QualityCheckpoint, ...]:
    if not result.incumbent_timeline:
        return ()
    baseline = result.incumbent_timeline[0].comparator_key[0]
    event_by_artifact = _checkpoint_events(events)
    checkpoints: list[QualityCheckpoint] = []
    last_tokens = 0
    last_tool_calls = 0
    for incumbent in result.incumbent_timeline:
        event = event_by_artifact.get(incumbent.plan_artifact_id)
        if event is not None:
            last_tokens = event.budget_after.model_usage.total_tokens
            last_tool_calls = event.budget_after.tool_calls
        quality = incumbent.comparator_key[0]
        checkpoints.append(
            QualityCheckpoint(
                elapsed_ms=incumbent.committed_at_ms,
                total_model_tokens=last_tokens,
                tool_calls=last_tool_calls,
                comparator_key=incumbent.comparator_key,
                primary_quality=quality,
                baseline_gain=relative_gain(quality, baseline),
                normalized_gap_closed=(
                    normalized_gap_closed(quality, baseline, reference_quality)
                    if reference_quality is not None
                    else None
                ),
                plan_artifact_id=incumbent.plan_artifact_id,
                scorecard_artifact_id=incumbent.scorecard_artifact_id,
            )
        )
    return tuple(checkpoints)


def build_run_record(
    *,
    manifest_hash: str,
    benchmark_id: str,
    run_key: str,
    pair_id: str,
    instance: GeneratedInstance,
    comparison: ComparisonKind,
    budget_id: str,
    run_seed: int,
    track: BenchmarkTrack,
    result: RunResult,
    events: Sequence[ControllerEvent],
    artifact_store: ArtifactStore,
) -> BenchmarkRunRecord:
    """Convert one controller run into independently sourced benchmark metrics."""

    reference = instance.reference
    reference_quality = (
        reference.scorecard.comparator_key[0]
        if reference is not None and reference.scorecard is not None
        else None
    )
    checkpoints = _checkpoints(result, events, reference_quality)
    score = result.best_scorecard
    baseline_quality = checkpoints[0].primary_quality if checkpoints else None
    current_quality = score.comparator_key[0] if score is not None else None
    gain = (
        relative_gain(current_quality, baseline_quality)
        if current_quality is not None and baseline_quality is not None
        else None
    )
    gap_closed = (
        normalized_gap_closed(current_quality, baseline_quality, reference_quality)
        if current_quality is not None
        and baseline_quality is not None
        and reference_quality is not None
        else None
    )
    reference_gap = (
        reference_quality - current_quality
        if reference_quality is not None and current_quality is not None
        else None
    )
    profile = _event_profile(events)
    artifact_count, artifact_bytes, largest_artifact_bytes = _artifact_profile(
        result, events, artifact_store
    )
    usage = result.consumed_budget.model_usage
    wall_horizon = result.requested_budget.wall_time_ms
    token_horizon = result.requested_budget.max_total_model_tokens
    wall_thresholds = tuple(sorted({wall_horizon // 4, wall_horizon // 2, wall_horizon}))
    token_thresholds = tuple(sorted({token_horizon // 4, token_horizon // 2, token_horizon}))
    inventory = result.compute_inventory
    group_key = (
        *evaluation_group_key(result.runtime_plan, inventory),
        f"platform={inventory.platform or 'unknown'}",
        f"cpu_count={inventory.cpu_count}",
        f"total_ram_bytes={inventory.total_ram_bytes}",
    )
    model_comparison = comparison in {
        ComparisonKind.ONE_SHOT_MODEL,
        ComparisonKind.ITERATIVE_MODEL,
        ComparisonKind.DIRECT_MODEL_CANDIDATE,
    }
    if model_comparison and result.runtime_plan.runtime is not RuntimeKind.FAKE:
        if (
            result.model_profile is None
            or result.model_id is None
            or not result.compute_inventory.library_versions
        ):
            raise ValueError("real-model comparisons require complete model/runtime metadata")
    return BenchmarkRunRecord(
        manifest_hash=manifest_hash,
        benchmark_id=benchmark_id,
        run_key=run_key,
        run_id=result.run_id,
        pair_id=pair_id,
        instance_id=instance.id,
        problem_family=instance.generator_spec.family,
        problem_type=instance.problem_type,
        track=track,
        comparison=comparison,
        budget_id=budget_id,
        run_seed=run_seed,
        generator_seed=instance.generator_spec.seed,
        problem_hash=result.problem_hash,
        evidence_hash=result.evidence_hash,
        policy_hash=result.policy_hash,
        status=result.status,
        terminal_reason=result.terminal_reason,
        feasible=score is not None and score.feasible,
        invalid_candidate_count=profile.invalid_candidates,
        rejected_candidate_count=profile.rejected_candidates,
        raw_objective=score.raw_objective if score is not None and score.feasible else {},
        comparator_key=score.comparator_key if score is not None and score.feasible else (),
        overall_metrics=score.overall_metrics if score is not None and score.feasible else {},
        group_metrics=score.group_metrics if score is not None and score.feasible else {},
        scenario_metrics=score.scenario_metrics if score is not None and score.feasible else {},
        baseline_gain=gain,
        reference_kind=reference.kind if reference is not None else None,
        reference_gap=reference_gap,
        normalized_gap_closed=gap_closed,
        time_to_first_feasible_ms=result.time_to_first_feasible_ms,
        deadline_overshoot_ms=result.deadline_overshoot_ms,
        wall_elapsed_ms=result.consumed_budget.wall_elapsed_ms,
        input_tokens=usage.input_tokens,
        generated_tokens=usage.generated_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        total_model_tokens=usage.total_tokens,
        tool_calls=profile.tool_calls,
        tool_latency_ms=profile.tool_latency_ms,
        tool_failure_count=profile.tool_failures,
        model_failure_count=profile.model_failures,
        parse_repair_count=profile.repairs,
        model_action_input_tokens=profile.model_input_tokens,
        model_action_generated_tokens=profile.model_generated_tokens,
        compact_context_bytes=profile.compact_context_bytes,
        tool_latency_observations_ms=profile.tool_latency_observations_ms,
        tool_p50_estimates_ms=profile.tool_p50_estimates_ms,
        tool_p95_estimates_ms=profile.tool_p95_estimates_ms,
        artifact_count=artifact_count,
        artifact_bytes=artifact_bytes,
        largest_artifact_bytes=largest_artifact_bytes,
        checkpoints=checkpoints,
        quality_at_wall_time_ms={
            str(key): value
            for key, value in incumbent_quality_at(
                tuple((item.elapsed_ms, item.primary_quality) for item in checkpoints),
                wall_thresholds,
            ).items()
        },
        quality_at_total_tokens={
            str(key): value
            for key, value in incumbent_quality_at(
                tuple((item.total_model_tokens, item.primary_quality) for item in checkpoints),
                token_thresholds,
            ).items()
        },
        auc_quality_log_time=area_under_log_curve(
            tuple(
                (item.elapsed_ms, item.primary_quality)
                for item in checkpoints
                if item.elapsed_ms <= wall_horizon
            ),
            wall_horizon,
        ),
        auc_quality_log_tokens=area_under_log_curve(
            tuple(
                (item.total_model_tokens, item.primary_quality)
                for item in checkpoints
                if item.total_model_tokens <= token_horizon
            ),
            token_horizon,
        ),
        runtime_plan=result.runtime_plan,
        compute_inventory=result.compute_inventory,
        runtime_group_key=group_key,
        model_profile=result.model_profile,
        model_id=result.model_id,
        evaluator_version=result.evaluator_version or instance.evaluator_version,
        problem_plugin_version=result.problem_plugin_version or "unknown",
        controller_version=result.controller_version,
        tool_versions=result.tool_versions,
        warnings=result.warnings,
        failures=result.failures,
    )


def load_scorecard(store: ArtifactStore, artifact_id: str) -> Scorecard:
    """Read a scorecard artifact through its public schema for report consumers."""

    return Scorecard.model_validate(read_json(store, artifact_id))


__all__ = [
    "ProblemFamily",
    "ReferenceKind",
    "build_run_record",
    "independently_evaluate_plan",
    "load_scorecard",
]
