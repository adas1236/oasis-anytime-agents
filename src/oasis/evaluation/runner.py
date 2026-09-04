"""Manifest-driven paired benchmark execution through public controller interfaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, cast

from oasis.artifacts import ArtifactStore, LocalArtifactStore, canonical_json_bytes, read_json
from oasis.config import BackendKind, OasisSettings
from oasis.controller import (
    AnytimeController,
    BudgetTier,
    ControllerPolicy,
    InMemoryRunStore,
    RunRequest,
)
from oasis.evaluation.fixtures import load_fixture
from oasis.evaluation.generators import generate_instance
from oasis.evaluation.models import (
    BenchmarkManifest,
    BenchmarkRunRecord,
    BenchmarkSummary,
    BenchmarkTrack,
    ComparisonKind,
    GeneratedInstance,
    ProblemFamily,
)
from oasis.evaluation.oracles import attach_reference
from oasis.evaluation.records import build_run_record
from oasis.evaluation.reporting import (
    read_run_records,
    write_aggregate_outputs,
    write_manifest_copy,
    write_run_record,
)
from oasis.llm import FakeModelBackend, ModelBackend, ToolCall
from oasis.llm.factory import create_model_backend
from oasis.llm.profiles import resolve_model_profile
from oasis.problems import (
    Comparison,
    Deadline,
    ProblemPlugin,
    ProblemRegistry,
    ResultView,
    Scorecard,
    SearchStrategy,
    ValidationReport,
    create_builtin_problem_registry,
)
from oasis.runtimes import ComputeInventory, RuntimePlan, safe_cpu_inventory
from oasis.schemas import Plan


def manifest_hash(manifest: BenchmarkManifest) -> str:
    """Hash every execution-relevant manifest field canonically."""

    return hashlib.sha256(canonical_json_bytes(manifest.model_dump(mode="json"))).hexdigest()


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Read a JSON benchmark manifest through the public versioned schema."""

    return BenchmarkManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


class _PortfolioPlugin:
    """Restrict deterministic fallback to the manifest's fixed strategy portfolio."""

    def __init__(self, delegate: ProblemPlugin, strategies: tuple[SearchStrategy, ...]) -> None:
        self._delegate = delegate
        self._strategies = strategies
        self.type_id = delegate.type_id
        self.version = delegate.version

    def validate_spec(self, spec: object, store: ArtifactStore) -> ValidationReport:
        return self._delegate.validate_spec(spec, store)

    def make_baseline(self, spec: object, store: ArtifactStore, deadline: Deadline) -> Plan:
        return self._delegate.make_baseline(spec, store, deadline)

    def validate_plan(self, spec: object, plan: Plan, store: ArtifactStore) -> ValidationReport:
        return self._delegate.validate_plan(spec, plan, store)

    def measure(self, spec: object, plan: Plan, store: ArtifactStore) -> Scorecard:
        return self._delegate.measure(spec, plan, store)

    def compare(self, left: Scorecard, right: Scorecard) -> Comparison:
        return self._delegate.compare(left, right)

    def fallback_actions(self) -> tuple[SearchStrategy, ...]:
        allowed = set(self._delegate.fallback_actions())
        return tuple(strategy for strategy in self._strategies if strategy in allowed)

    def render_result(self, spec: object, plan: Plan, scorecard: Scorecard) -> ResultView:
        return self._delegate.render_result(spec, plan, scorecard)


def _applicable_strategies(
    family: ProblemFamily,
    strategies: tuple[SearchStrategy, ...],
) -> tuple[SearchStrategy, ...]:
    location = {
        SearchStrategy.ADD_SWAP,
        SearchStrategy.MULTI_SWAP,
        SearchStrategy.LOCAL_ASSIGNMENT,
        SearchStrategy.SCENARIO_AWARE,
        SearchStrategy.EXACT_ENUMERATION,
        SearchStrategy.ORTOOLS_CP_SAT,
    }
    routing = {
        SearchStrategy.TWO_OPT,
        SearchStrategy.RELOCATE,
        SearchStrategy.SWAP,
        SearchStrategy.EXACT_ENUMERATION,
        SearchStrategy.ORTOOLS_ROUTING,
    }
    allowed = location if family is ProblemFamily.LOCATION_ALLOCATION else routing
    return tuple(strategy for strategy in strategies if strategy in allowed)


def _one_shot_strategy(
    instance: GeneratedInstance,
    strategies: tuple[SearchStrategy, ...],
    max_candidates: int,
) -> SearchStrategy:
    reference = instance.reference
    if (
        SearchStrategy.EXACT_ENUMERATION in strategies
        and reference is not None
        and reference.total_candidates is not None
        and reference.total_candidates <= max_candidates
    ):
        return SearchStrategy.EXACT_ENUMERATION
    for preferred in (
        SearchStrategy.ORTOOLS_CP_SAT,
        SearchStrategy.ORTOOLS_ROUTING,
        SearchStrategy.SCENARIO_AWARE,
        SearchStrategy.ADD_SWAP,
        SearchStrategy.RELOCATE,
        SearchStrategy.TWO_OPT,
        SearchStrategy.LOCAL_ASSIGNMENT,
        SearchStrategy.SWAP,
        SearchStrategy.MULTI_SWAP,
        SearchStrategy.EXACT_ENUMERATION,
    ):
        if preferred in strategies:
            return preferred
    raise ValueError(f"no applicable strategy exists for instance {instance.id!r}")


def _fake_backend(
    manifest: BenchmarkManifest,
    comparison: ComparisonKind,
    instance: GeneratedInstance,
    baseline: Plan | None,
    runtime_plan: RuntimePlan,
    inventory: ComputeInventory,
) -> FakeModelBackend:
    profile = resolve_model_profile(manifest.model.profile, manifest.model.model_id)
    strategies = _applicable_strategies(instance.generator_spec.family, manifest.tool_suite)
    responses: list[str | ToolCall] = []
    if comparison is ComparisonKind.ONE_SHOT_MODEL:
        strategy = _one_shot_strategy(instance, strategies, manifest.max_candidates_per_action)
        responses.append(
            ToolCall(
                id="one-shot-strategy",
                name="improve",
                arguments={
                    "strategy": strategy.value,
                    "max_candidates": manifest.max_candidates_per_action,
                },
            )
        )
    elif comparison is ComparisonKind.ITERATIVE_MODEL:
        responses.extend(
            ToolCall(
                id=f"iterative-{index}",
                name="improve",
                arguments={
                    "strategy": strategy.value,
                    "max_candidates": manifest.max_candidates_per_action,
                },
            )
            for index, strategy in enumerate(strategies)
        )
    elif comparison is ComparisonKind.DIRECT_MODEL_CANDIDATE:
        if baseline is None:
            responses.append(json.dumps({"type": "stop", "rationale": "no feasible baseline"}))
        else:
            responses.append(
                json.dumps(
                    {
                        "type": "submit_candidate",
                        "candidate": baseline.model_dump(mode="json"),
                        "rationale": "direct deterministic fake candidate",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    return FakeModelBackend(
        responses,
        profile=profile,
        inventory=inventory,
        runtime_plan=runtime_plan,
    )


def _settings(manifest: BenchmarkManifest) -> OasisSettings:
    policy = manifest.runtime_policy
    return OasisSettings.resolve(
        explicit_overrides={
            "backend": BackendKind(manifest.model.backend),
            "model_profile": manifest.model.profile,
            "model_id": manifest.model.model_id,
            "model_revision": manifest.model.revision,
            "thinking": manifest.model.thinking,
            "trust_remote_code": manifest.model.trust_remote_code,
            "device": policy.device,
            "runtime_engine": policy.engine,
            "dtype": policy.dtype,
            "quantization": policy.quantization,
            "attention_backend": policy.attention_backend,
            "memory_headroom_fraction": policy.memory_headroom_fraction,
            "allow_cpu_offload": policy.allow_cpu_offload,
            "allow_disk_offload": policy.allow_disk_offload,
            "offload_root": policy.offload_directory,
            "remote_endpoint": policy.remote_endpoint,
            "model_memory_bytes": policy.model_memory_bytes,
        }
    )


class _MetadataBackend(ModelBackend, Protocol):
    @property
    def runtime_plan(self) -> RuntimePlan: ...

    @property
    def compute_inventory(self) -> ComputeInventory: ...


def _output_state(
    output_root: str | Path,
    manifest: BenchmarkManifest,
) -> tuple[Path, str, dict[str, BenchmarkRunRecord]]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    digest = manifest_hash(manifest)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest_hash(existing) != digest:
            raise ValueError("results directory belongs to a different benchmark manifest")
    else:
        write_manifest_copy(root, manifest.model_dump_json(indent=2))
    records = {record.run_key: record for record in read_run_records(root)}
    if any(record.manifest_hash != digest for record in records.values()):
        raise ValueError("raw results contain a different manifest hash")
    return root, digest, records


def _keys(
    manifest_digest: str,
    instance_id: str,
    budget_id: str,
    comparison: ComparisonKind,
    run_seed: int,
) -> tuple[str, str]:
    pair_payload = f"{manifest_digest}:{instance_id}:{budget_id}:{run_seed}".encode()
    pair_id = hashlib.sha256(pair_payload).hexdigest()
    run_key = hashlib.sha256(f"{pair_id}:{comparison.value}".encode()).hexdigest()
    return pair_id, run_key


def _expected_run_keys(manifest: BenchmarkManifest, digest: str) -> set[str]:
    return {
        _keys(digest, instance.id, budget.id, comparison, run_seed)[1]
        for instance in manifest.instances
        for budget in manifest.budgets
        for run_seed in manifest.run_seeds
        for comparison in manifest.comparisons
    }


def _read_baseline(instance: GeneratedInstance, store: ArtifactStore) -> Plan | None:
    if instance.baseline_plan_artifact_id is None:
        return None
    return Plan.model_validate(read_json(store, instance.baseline_plan_artifact_id))


async def _prepare_instances(
    manifest: BenchmarkManifest,
    store: ArtifactStore,
) -> tuple[GeneratedInstance, ...]:
    instances: list[GeneratedInstance] = []
    for entry in manifest.instances:
        if entry.generator is not None:
            spec = entry.generator
        else:
            assert entry.fixture is not None
            spec = load_fixture(entry.fixture)
        generated = await generate_instance(entry.id, spec, store)
        generated = attach_reference(
            generated,
            store,
            max_exact_candidates=manifest.max_exact_candidates,
            max_reference_candidates=manifest.max_reference_candidates,
        )
        expected = manifest.expected_evaluator_versions.get(
            generated.problem_type, manifest.expected_evaluator_versions.get("*")
        )
        if expected is None:
            raise ValueError(
                f"manifest lacks expected evaluator version for {generated.problem_type!r}"
            )
        if expected != generated.evaluator_version:
            raise ValueError(
                f"evaluator version mismatch for {generated.problem_type!r}: "
                f"expected {expected}, found {generated.evaluator_version}"
            )
        instances.append(generated)
    return tuple(instances)


async def run_benchmark(
    manifest: BenchmarkManifest,
    output_root: str | Path,
    *,
    allow_real_model: bool = False,
    compute_inventory: ComputeInventory | None = None,
    stop_after_runs: int | None = None,
) -> BenchmarkSummary:
    """Run or resume a benchmark and publish raw plus aggregate JSON/CSV/Parquet output."""

    if manifest.model.backend == "transformers" and not allow_real_model:
        raise ValueError(
            "real-model evaluation requires an explicit approval and allow_real_model=True"
        )
    supported_tracks = {
        BenchmarkTrack.FORMAL_OPTIMIZATION,
        BenchmarkTrack.PRIMITIVE_TOOLS,
        BenchmarkTrack.EXPERT_SOLVERS,
    }
    if manifest.track not in supported_tracks:
        raise ValueError(
            f"track {manifest.track.value!r} requires a grounding/compiler runner not available "
            "in the Phase 10 formal optimization harness"
        )
    root, digest, existing_records = _output_state(output_root, manifest)
    expected_keys = _expected_run_keys(manifest, digest)
    if expected_keys == existing_records.keys():
        return write_aggregate_outputs(root, tuple(existing_records.values()))

    inventory = compute_inventory or safe_cpu_inventory()
    metadata_backend = create_model_backend(_settings(manifest), inventory=inventory)
    metadata = cast(_MetadataBackend, metadata_backend)
    runtime_plan = metadata.runtime_plan
    runtime_inventory = metadata.compute_inventory
    shared_real_backend: ModelBackend | None = (
        metadata_backend if manifest.model.backend == "transformers" else None
    )
    artifacts = LocalArtifactStore(root / "artifacts")
    instances = await _prepare_instances(manifest, artifacts)
    completed_this_call = 0
    try:
        for instance in instances:
            if instance.problem_artifact_id is None:
                raise ValueError(f"instance {instance.id!r} did not produce a problem artifact")
            applicable = _applicable_strategies(instance.generator_spec.family, manifest.tool_suite)
            delegate = create_builtin_problem_registry().get(instance.problem_type)
            registry = ProblemRegistry((_PortfolioPlugin(delegate, applicable),))
            baseline = _read_baseline(instance, artifacts)
            for budget in manifest.budgets:
                for run_seed in manifest.run_seeds:
                    for comparison in manifest.comparisons:
                        pair_id, run_key = _keys(
                            digest, instance.id, budget.id, comparison, run_seed
                        )
                        if run_key in existing_records:
                            continue
                        if stop_after_runs is not None and completed_this_call >= stop_after_runs:
                            records = tuple(existing_records.values())
                            if not records:
                                raise ValueError("benchmark stopped before producing a run record")
                            return write_aggregate_outputs(root, records)
                        model_comparison = comparison in {
                            ComparisonKind.ONE_SHOT_MODEL,
                            ComparisonKind.ITERATIVE_MODEL,
                            ComparisonKind.DIRECT_MODEL_CANDIDATE,
                        }
                        backend: ModelBackend | None = None
                        if model_comparison:
                            backend = (
                                _fake_backend(
                                    manifest,
                                    comparison,
                                    instance,
                                    baseline,
                                    runtime_plan,
                                    runtime_inventory,
                                )
                                if manifest.model.backend == "fake"
                                else shared_real_backend
                            )
                        run_store = InMemoryRunStore()
                        controller = AnytimeController(
                            artifact_store=artifacts,
                            run_store=run_store,
                            backend=backend,
                            problem_registry=registry,
                            policy=ControllerPolicy(
                                max_model_actions=max(1, len(applicable)),
                                max_no_progress_actions=max(2, len(applicable) + 1),
                                max_candidates_per_action=manifest.max_candidates_per_action,
                            ),
                        )
                        requested_tier = {
                            ComparisonKind.DETERMINISTIC_BASELINE: BudgetTier.BASELINE_ONLY,
                            ComparisonKind.DETERMINISTIC_PORTFOLIO: (
                                BudgetTier.DETERMINISTIC_IMPROVEMENT
                            ),
                            ComparisonKind.ONE_SHOT_MODEL: BudgetTier.ONE_SHOT_MODEL,
                            ComparisonKind.ITERATIVE_MODEL: BudgetTier.ITERATIVE_MODEL,
                            ComparisonKind.DIRECT_MODEL_CANDIDATE: BudgetTier.ONE_SHOT_MODEL,
                        }[comparison]
                        try:
                            result = await controller.run(
                                RunRequest(
                                    run_id=f"eval-{run_key[:32]}",
                                    problem_artifact_id=instance.problem_artifact_id,
                                    baseline_plan_artifact_id=instance.baseline_plan_artifact_id,
                                    budget=budget.resources,
                                    seed=run_seed,
                                    enable_model=model_comparison,
                                    enable_deterministic_fallback=(
                                        comparison is ComparisonKind.DETERMINISTIC_PORTFOLIO
                                    ),
                                    requested_tier=requested_tier,
                                    runtime_plan=runtime_plan,
                                    compute_inventory=runtime_inventory,
                                )
                            )
                        finally:
                            if backend is not None and backend is not shared_real_backend:
                                await backend.close()
                        record = build_run_record(
                            manifest_hash=digest,
                            benchmark_id=manifest.benchmark_id,
                            run_key=run_key,
                            pair_id=pair_id,
                            instance=instance,
                            comparison=comparison,
                            budget_id=budget.id,
                            run_seed=run_seed,
                            track=manifest.track,
                            result=result,
                            events=run_store.read_events(result.run_id),
                            artifact_store=artifacts,
                        )
                        write_run_record(root, record)
                        existing_records[run_key] = record
                        completed_this_call += 1
        return write_aggregate_outputs(root, tuple(existing_records.values()))
    finally:
        await metadata_backend.close()
