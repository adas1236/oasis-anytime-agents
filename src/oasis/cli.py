"""Command-line entry point for chat, workflows, tools, and the service API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import httpx
from shapely.geometry import Point

from oasis.anytime import run_anytime_demo
from oasis.artifacts import ArtifactProvenance, LocalArtifactStore, put_vector
from oasis.config import BackendKind, DevicePolicy, OasisSettings, RuntimeEngine
from oasis.decision import run_decision_demo
from oasis.errors import ModelBackendError
from oasis.evidence import run_evidence_demo
from oasis.llm.factory import create_model_backend
from oasis.llm.profiles import MODEL_PROFILES, resolve_model_profile
from oasis.llm.protocols import ModelBackend
from oasis.llm.schemas import ChatMessage, ChatRole, ModelRequest, TokenUsage
from oasis.providers import (
    HttpPolicy,
    HttpSourceSnapshotProvider,
    LocalSnapshotCache,
    NominatimPlaceResolver,
    OsrmRoutingMatrixProvider,
    ResilientHttpClient,
    SourceFormat,
)
from oasis.routing import run_routing_demo
from oasis.runtimes import (
    ComputeInventory,
    ConservativeRuntimePlanner,
    RemoteModelRuntime,
    RuntimePlanningError,
    inspect_cuda_inventory,
    named_fake_inventory,
    safe_cpu_inventory,
)
from oasis.schemas import ToolResult, ToolResultStatus
from oasis.tools import (
    CancellationToken,
    ToolContext,
    create_public_tool_registry,
    create_tool_registry,
    invoke_tool,
)
from oasis.tools.providers import (
    PLACE_PROVIDER,
    ROUTING_PROVIDER,
    SNAPSHOT_CACHE,
    SOURCE_PROVIDER,
)
from oasis.tools.registry import ToolRegistryError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oasis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask = subparsers.add_parser("ask", help="answer a message using the agent and server defaults")
    ask.add_argument("message", help="the question or task, in ordinary language")
    chat = subparsers.add_parser("chat", help="stream a raw, multi-turn text chat")
    chat.add_argument("--backend", choices=[kind.value for kind in BackendKind])
    chat.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    chat.add_argument("--model", dest="model_id", help="explicit Hugging Face model ID")
    chat.add_argument("--revision", dest="model_revision")
    chat.add_argument("--device", choices=[policy.value for policy in DevicePolicy])
    chat.add_argument("--engine", choices=[engine.value for engine in RuntimeEngine])
    chat.add_argument("--dtype")
    chat.add_argument("--quantization", choices=["int8", "int4"])
    chat.add_argument("--attention-backend")
    chat.add_argument(
        "--probe-cuda",
        action="store_true",
        help="explicitly inspect visible CUDA devices before resolving the runtime",
    )
    chat.add_argument("--max-generated-tokens", type=int)
    thinking = chat.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true", default=None)
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    chat.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=None,
        help="explicitly permit model repository Python code (off by default)",
    )
    chat.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="send one turn non-interactively; repeat for a multi-turn exchange",
    )

    tools = subparsers.add_parser("tools", help="list, inspect, or smoke-test registry tools")
    tools.add_argument(
        "--advanced", action="store_true", help="inspect the full low-level SDK tools"
    )
    tools.add_argument(
        "--no-plugins", action="store_true", help="do not discover third-party oasis.tools entries"
    )
    tool_commands = tools.add_subparsers(dest="tools_command", required=True)
    tool_commands.add_parser("list", help="list available tools")
    describe = tool_commands.add_parser("describe", help="print one complete ToolSpec as JSON")
    describe.add_argument("name")
    smoke = tool_commands.add_parser("smoke", help="run a tool's declared smoke input")
    smoke.add_argument("name")
    smoke.add_argument("--input", help="override smoke input with a JSON object")
    smoke.add_argument("--artifact-root", type=Path)

    evidence = subparsers.add_parser("evidence", help="run deterministic evidence workflows")
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    demo = evidence_commands.add_parser("demo", help="build the frozen Phase 3 example")
    demo.add_argument("--artifact-root", type=Path, default=Path("artifacts/evidence-demo"))
    live = evidence_commands.add_parser(
        "live-smoke",
        help="opt-in place, public-source snapshot, and routed-matrix smoke test",
    )
    live.add_argument(
        "--confirm-live-network",
        action="store_true",
        help="required acknowledgement that this command contacts configured public endpoints",
    )
    live.add_argument("--place", required=True, help="place query returned as ranked candidates")
    live.add_argument("--source-url", required=True, help="public CSV or GeoJSON URL to snapshot")
    live.add_argument(
        "--source-format", required=True, choices=[value.value for value in SourceFormat]
    )
    live.add_argument("--source-license", required=True)
    live.add_argument("--source-units", default="unitless")
    live.add_argument("--source-crs")
    live.add_argument(
        "--route-coordinate",
        action="append",
        required=True,
        metavar="LON,LAT",
        help="route point; repeat at least twice",
    )
    live.add_argument("--routing-profile", default="driving")
    live.add_argument("--nominatim-endpoint", default="https://nominatim.openstreetmap.org")
    live.add_argument("--osrm-endpoint", default="https://router.project-osrm.org")
    live.add_argument("--artifact-root", type=Path)
    live.add_argument("--cache-root", type=Path)

    decision = subparsers.add_parser("decision", help="run deterministic decision workflows")
    decision_commands = decision.add_subparsers(dest="decision_command", required=True)
    decision_demo = decision_commands.add_parser(
        "demo", help="run the frozen Phase 5 cooling-center workflow"
    )
    decision_demo.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/decision-demo")
    )
    routing_demo = decision_commands.add_parser(
        "mobile-demo", help="run the frozen Phase 6 mobile-vaccination workflow"
    )
    routing_demo.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/mobile-routing-demo")
    )
    anytime_demo = decision_commands.add_parser(
        "anytime-demo", help="run the Phase 7 fake-model anytime coverage workflow"
    )
    anytime_demo.add_argument("--artifact-root", type=Path, default=Path("artifacts/anytime-demo"))
    anytime_demo.add_argument("--wall-time-ms", type=int, default=30_000)
    anytime_demo.add_argument("--max-total-model-tokens", type=int, default=2_000)
    anytime_demo.add_argument("--max-generated-tokens", type=int, default=256)
    anytime_demo.add_argument("--max-tool-calls", type=int, default=2)

    serve = subparsers.add_parser("serve", help="serve the versioned HTTP/SSE API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--backend", choices=[kind.value for kind in BackendKind])
    serve.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    serve.add_argument("--model", dest="model_id", help="explicit Hugging Face model ID")
    serve.add_argument("--revision", dest="model_revision")
    serve.add_argument("--device", choices=[policy.value for policy in DevicePolicy])
    serve.add_argument("--engine", choices=[engine.value for engine in RuntimeEngine])
    serve.add_argument("--dtype")
    serve.add_argument("--quantization", choices=["int8", "int4"])
    serve.add_argument("--attention-backend")
    serve.add_argument("--remote-endpoint")
    serve.add_argument(
        "--probe-cuda",
        action="store_true",
        help="explicitly inspect visible CUDA devices before starting the service",
    )
    serve.add_argument("--artifact-root", type=Path)
    serve.add_argument("--run-root", type=Path)
    serve.add_argument("--max-concurrent-runs", type=int)
    serve.add_argument("--max-request-bytes", type=int)
    serve.add_argument("--max-artifact-response-bytes", type=int)
    ui = serve.add_mutually_exclusive_group()
    ui.add_argument("--serve-ui", dest="serve_ui", action="store_true", default=None)
    ui.add_argument("--no-serve-ui", dest="serve_ui", action="store_false")
    serve.add_argument("--ui-root", type=Path)

    hardware = subparsers.add_parser("hardware", help="inspect safe or explicitly probed hardware")
    hardware_commands = hardware.add_subparsers(dest="hardware_command", required=True)
    inspect = hardware_commands.add_parser("inspect", help="print a typed compute inventory")
    inventory_source = inspect.add_mutually_exclusive_group()
    inventory_source.add_argument(
        "--probe-cuda",
        action="store_true",
        help="explicitly import Torch and inspect CUDA devices visible to this process",
    )
    inventory_source.add_argument(
        "--fixture",
        choices=[
            "cpu",
            "local-5060ti-8gb",
            "local-5060ti-16gb",
            "2x24gb",
            "4x80gb",
        ],
        help="use a named fake inventory without probing real hardware",
    )

    runtime = subparsers.add_parser(
        "runtime", help="resolve runtime placement without loading a model"
    )
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    plan = runtime_commands.add_parser("plan", help="produce a dry-run typed RuntimePlan")
    plan.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    plan.add_argument("--model", dest="model_id")
    plan.add_argument("--revision", dest="model_revision")
    plan.add_argument("--device", choices=[policy.value for policy in DevicePolicy])
    plan.add_argument("--engine", choices=[engine.value for engine in RuntimeEngine])
    plan.add_argument("--dtype")
    plan.add_argument("--quantization", choices=["int8", "int4"])
    plan.add_argument("--attention-backend")
    plan.add_argument("--memory-headroom-fraction", type=float)
    plan.add_argument("--model-memory-gib", type=float)
    plan.add_argument("--allow-cpu-offload", action="store_true", default=None)
    plan.add_argument("--allow-disk-offload", action="store_true", default=None)
    plan.add_argument("--remote-endpoint")
    plan_inventory = plan.add_mutually_exclusive_group()
    plan_inventory.add_argument(
        "--probe-cuda",
        action="store_true",
        help="explicitly inspect CUDA; omit for safe CPU/environment discovery",
    )
    plan_inventory.add_argument(
        "--fixture",
        choices=[
            "cpu",
            "local-5060ti-8gb",
            "local-5060ti-16gb",
            "2x24gb",
            "4x80gb",
        ],
        help="plan against a named fake inventory",
    )

    worker = subparsers.add_parser("model-worker", help="serve or inspect a remote model worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    worker_serve = worker_commands.add_parser("serve", help="serve authenticated model inference")
    worker_serve.add_argument("--host")
    worker_serve.add_argument("--port", type=int)
    worker_serve.add_argument("--backend", choices=[kind.value for kind in BackendKind])
    worker_serve.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    worker_serve.add_argument("--model", dest="model_id")
    worker_serve.add_argument("--revision", dest="model_revision")
    worker_serve.add_argument("--device", choices=[policy.value for policy in DevicePolicy])
    worker_serve.add_argument("--engine", choices=[engine.value for engine in RuntimeEngine])
    worker_serve.add_argument("--dtype")
    worker_serve.add_argument("--quantization", choices=["int8", "int4"])
    worker_serve.add_argument("--attention-backend")
    worker_serve.add_argument(
        "--probe-cuda",
        action="store_true",
        help="explicitly inspect visible CUDA devices before starting the worker",
    )
    worker_serve.add_argument(
        "--auth-token-env",
        default="OASIS_MODEL_WORKER_AUTH_TOKEN",
        help="environment variable containing the bearer token",
    )
    for name in ("health", "capabilities"):
        command = worker_commands.add_parser(name, help=f"read worker {name}")
        command.add_argument("--endpoint", required=True)
        command.add_argument(
            "--auth-token-env",
            default="OASIS_REMOTE_AUTH_TOKEN",
            help="environment variable containing the bearer token",
        )

    evaluate = subparsers.add_parser("evaluate", help="run or resume a benchmark manifest")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("evaluation-output"))
    evaluate.add_argument(
        "--confirm-real-model-evaluation",
        action="store_true",
        help="required explicit approval for a manifest that loads real model weights",
    )
    evaluate.add_argument(
        "--probe-cuda",
        action="store_true",
        help="explicitly inspect visible CUDA for an approved real-model evaluation",
    )

    summarize = subparsers.add_parser("summarize", help="summarize benchmark raw results")
    summarize.add_argument("results", type=Path)

    showcase_help = "run or resume the complete offline Track B showcase release"
    showcase = subparsers.add_parser("demo", help=showcase_help, description=showcase_help)
    showcase.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-output/track-b-showcase-v1"),
    )
    return parser


def _settings_from_args(args: argparse.Namespace) -> OasisSettings:
    cli_values = {
        "backend": getattr(args, "backend", None),
        "model_profile": getattr(args, "profile", None),
        "model_id": getattr(args, "model_id", None),
        "model_revision": getattr(args, "model_revision", None),
        "device": getattr(args, "device", None),
        "runtime_engine": getattr(args, "engine", None),
        "dtype": getattr(args, "dtype", None),
        "quantization": getattr(args, "quantization", None),
        "attention_backend": getattr(args, "attention_backend", None),
        "max_generated_tokens": getattr(args, "max_generated_tokens", None),
        "thinking": getattr(args, "thinking", None),
        "trust_remote_code": getattr(args, "trust_remote_code", None),
        "remote_endpoint": getattr(args, "remote_endpoint", None),
    }
    return OasisSettings.resolve(cli_overrides=cli_values)


def _make_backend(
    settings: OasisSettings,
    *,
    probe_cuda: bool = False,
) -> ModelBackend:
    inventory = inspect_cuda_inventory() if probe_cuda else None
    return create_model_backend(settings, inventory=inventory)


async def _send_turn(
    backend: ModelBackend,
    history: list[ChatMessage],
    prompt: str,
    *,
    turn_number: int,
    settings: OasisSettings,
) -> TokenUsage:
    history.append(ChatMessage(role=ChatRole.USER, content=prompt))
    request = ModelRequest(
        request_id=f"chat-{turn_number}",
        messages=tuple(history),
        max_generated_tokens=settings.max_generated_tokens,
        thinking_enabled=settings.thinking,
        seed=turn_number,
    )
    response_parts: list[str] = []
    usage = TokenUsage()
    async for delta in backend.stream(request):
        if delta.text:
            response_parts.append(delta.text)
            print(delta.text, end="", flush=True)
        if delta.usage is not None:
            usage = delta.usage
    print()
    history.append(ChatMessage(role=ChatRole.ASSISTANT, content="".join(response_parts)))
    return usage


async def _chat(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    backend = _make_backend(settings, probe_cuda=args.probe_cuda)
    history: list[ChatMessage] = []
    total_usage = TokenUsage()
    try:
        if args.prompt:
            for turn_number, prompt in enumerate(args.prompt, start=1):
                print(f"You: {prompt}")
                print("Assistant: ", end="", flush=True)
                total_usage += await _send_turn(
                    backend,
                    history,
                    prompt,
                    turn_number=turn_number,
                    settings=settings,
                )
        else:
            turn_number = 0
            print("Enter /quit to stop.")
            while True:
                try:
                    # No model or tool work is active while the single-user CLI awaits a turn.
                    prompt = input("You: ")  # noqa: ASYNC250
                except EOFError:
                    break
                if prompt.strip().lower() in {"/quit", "/exit"}:
                    break
                if not prompt.strip():
                    continue
                turn_number += 1
                print("Assistant: ", end="", flush=True)
                total_usage += await _send_turn(
                    backend,
                    history,
                    prompt,
                    turn_number=turn_number,
                    settings=settings,
                )
        print(
            "Usage: "
            f"input={total_usage.input_tokens} "
            f"generated={total_usage.generated_tokens} "
            f"total={total_usage.total_tokens}"
        )
        return 0
    finally:
        await backend.close()


async def _tools(args: argparse.Namespace) -> int:
    factory = create_tool_registry if args.advanced else create_public_tool_registry
    registry = factory(discover_entry_points=not args.no_plugins)
    if args.tools_command == "list":
        for spec in registry.list():
            capabilities = ",".join(sorted(spec.capability_tags)) or "-"
            print(f"{spec.name}\t{spec.version}\t{capabilities}\t{spec.description}")
        return 0
    spec = registry.get(args.name).spec
    if args.tools_command == "describe":
        print(spec.model_dump_json(indent=2))
        return 0
    raw_arguments = json.loads(args.input) if args.input else spec.smoke_input
    if not isinstance(raw_arguments, dict):
        raise ValueError("tool smoke input must be a JSON object")
    context = ToolContext(
        run_id="cli-smoke",
        artifact_store=LocalArtifactStore(args.artifact_root or OasisSettings().artifact_root),
        deadline_monotonic=time.monotonic() + max(1.0, spec.runtime.p95_ms / 1_000 * 2),
        cancellation=CancellationToken(),
        seed=0,
    )
    result = await invoke_tool(registry.get(spec.name), raw_arguments, context)
    print(result.model_dump_json(indent=2))
    return 0 if result.status.value in {"complete", "partial"} else 1


async def _evidence(args: argparse.Namespace) -> int:
    if args.evidence_command == "demo":
        result = await run_evidence_demo(args.artifact_root)
        print(result.model_dump_json(indent=2))
        return 0
    if args.evidence_command == "live-smoke":
        return await _live_evidence_smoke(args)
    raise ValueError(f"unknown evidence command: {args.evidence_command}")


async def _decision(args: argparse.Namespace) -> int:
    if args.decision_command == "demo":
        decision_result = await run_decision_demo(args.artifact_root)
        print(decision_result.model_dump_json(indent=2))
        return 0
    if args.decision_command == "mobile-demo":
        routing_result = await run_routing_demo(args.artifact_root)
        print(routing_result.model_dump_json(indent=2))
        return 0
    if args.decision_command == "anytime-demo":
        from oasis.controller import BudgetSpec

        anytime_result = await run_anytime_demo(
            args.artifact_root,
            budget=BudgetSpec(
                wall_time_ms=args.wall_time_ms,
                max_total_model_tokens=args.max_total_model_tokens,
                max_generated_tokens=args.max_generated_tokens,
                max_tool_calls=args.max_tool_calls,
            ),
        )
        print(anytime_result.model_dump_json(indent=2))
        return 0
    raise ValueError(f"unknown decision command: {args.decision_command}")


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from oasis.api import create_app

    settings = OasisSettings.resolve(
        cli_overrides={
            "backend": args.backend,
            "model_profile": args.profile,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "device": args.device,
            "runtime_engine": args.engine,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "attention_backend": args.attention_backend,
            "remote_endpoint": args.remote_endpoint,
            "artifact_root": args.artifact_root,
            "run_root": args.run_root,
            "api_host": args.host,
            "api_port": args.port,
            "api_max_concurrent_runs": args.max_concurrent_runs,
            "api_max_request_bytes": args.max_request_bytes,
            "api_max_artifact_response_bytes": args.max_artifact_response_bytes,
            "serve_ui": args.serve_ui,
            "ui_root": args.ui_root,
        }
    )
    inventory = inspect_cuda_inventory() if args.probe_cuda else None
    uvicorn.run(
        create_app(settings, compute_inventory=inventory),
        host=settings.api_host,
        port=settings.api_port,
    )
    return 0


def _inventory_from_args(args: argparse.Namespace) -> ComputeInventory:
    if getattr(args, "fixture", None):
        return named_fake_inventory(args.fixture)
    if getattr(args, "probe_cuda", False):
        return inspect_cuda_inventory()
    return safe_cpu_inventory()


def _hardware(args: argparse.Namespace) -> int:
    if args.hardware_command != "inspect":
        raise ValueError(f"unknown hardware command: {args.hardware_command}")
    inventory = _inventory_from_args(args)
    print(inventory.model_dump_json(indent=2))
    return 0


def _runtime(args: argparse.Namespace) -> int:
    if args.runtime_command != "plan":
        raise ValueError(f"unknown runtime command: {args.runtime_command}")
    model_memory_bytes = (
        round(args.model_memory_gib * 1024**3) if args.model_memory_gib is not None else None
    )
    settings = OasisSettings.resolve(
        cli_overrides={
            "model_profile": args.profile,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "device": args.device,
            "runtime_engine": args.engine,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "attention_backend": args.attention_backend,
            "memory_headroom_fraction": args.memory_headroom_fraction,
            "model_memory_bytes": model_memory_bytes,
            "allow_cpu_offload": args.allow_cpu_offload,
            "allow_disk_offload": args.allow_disk_offload,
            "remote_endpoint": args.remote_endpoint,
        }
    )
    profile = resolve_model_profile(settings.model_profile, settings.model_id)
    inventory = _inventory_from_args(args)
    resolved = ConservativeRuntimePlanner().plan(
        profile,
        inventory,
        settings.runtime_config(),
        revision=settings.model_revision,
    )
    print(
        json.dumps(
            {
                "dry_run": not args.probe_cuda,
                "inventory": inventory.sanitized().model_dump(mode="json"),
                "plan": resolved.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return 0


def _worker_token(variable: str) -> str:
    token = os.environ.get(variable)
    if not token:
        raise ValueError(f"model-worker authentication requires environment variable {variable}")
    return token


async def _worker_client(args: argparse.Namespace) -> int:
    token = _worker_token(args.auth_token_env)
    runtime = RemoteModelRuntime(args.endpoint, auth_token=token)
    try:
        result = (
            await runtime.health()
            if args.worker_command == "health"
            else await runtime.capability_report()
        )
        print(result.model_dump_json(indent=2))
        return 0
    finally:
        await runtime.close()


async def _evaluate(args: argparse.Namespace) -> int:
    from oasis.evaluation import load_manifest, run_benchmark

    manifest = load_manifest(args.manifest)
    if args.probe_cuda and not args.confirm_real_model_evaluation:
        raise ValueError("--probe-cuda requires --confirm-real-model-evaluation")
    if args.probe_cuda and manifest.model.backend != "transformers":
        raise ValueError("--probe-cuda applies only to a real-model evaluation manifest")
    inventory = inspect_cuda_inventory() if args.probe_cuda else safe_cpu_inventory()
    summary = await run_benchmark(
        manifest,
        args.output,
        allow_real_model=args.confirm_real_model_evaluation,
        compute_inventory=inventory,
    )
    print(summary.model_dump_json(indent=2))
    return 0


def _summarize(args: argparse.Namespace) -> int:
    from oasis.evaluation import summarize_results

    print(summarize_results(args.results).model_dump_json(indent=2))
    return 0


async def _showcase(args: argparse.Namespace) -> int:
    from oasis.showcase import run_showcase

    report = await run_showcase(args.output)
    print(report.model_dump_json(indent=2))
    return 0


def _worker_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from oasis.model_worker import create_model_worker_app

    settings = OasisSettings.resolve(
        cli_overrides={
            "backend": args.backend,
            "model_profile": args.profile,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "device": args.device,
            "runtime_engine": args.engine,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "attention_backend": args.attention_backend,
            "api_host": args.host,
            "api_port": args.port,
        }
    )
    inventory = inspect_cuda_inventory() if args.probe_cuda else safe_cpu_inventory()
    backend = create_model_backend(settings, inventory=inventory)
    token = _worker_token(args.auth_token_env)
    app = create_model_worker_app(backend, auth_token=token)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
    return 0


def _route_coordinates(values: Sequence[str]) -> tuple[tuple[float, float], ...]:
    coordinates: list[tuple[float, float]] = []
    for raw in values:
        try:
            longitude_text, latitude_text = raw.split(",", 1)
            longitude, latitude = float(longitude_text), float(latitude_text)
        except ValueError as error:
            raise ValueError("route coordinates must use LON,LAT") from error
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("route coordinates must be valid longitude/latitude pairs")
        coordinates.append((longitude, latitude))
    if len(coordinates) < 2:
        raise ValueError("live smoke requires at least two route coordinates")
    if len(coordinates) > 50:
        raise ValueError("live smoke accepts at most 50 route coordinates")
    return tuple(coordinates)


def _require_result(result: ToolResult, *, allow_ambiguity: bool = False) -> ToolResult:
    allowed = {ToolResultStatus.COMPLETE, ToolResultStatus.PARTIAL}
    if allow_ambiguity:
        allowed.add(ToolResultStatus.AMBIGUOUS)
    if result.status not in allowed:
        detail = result.error.message if result.error is not None else str(result.summary)
        raise ValueError(detail)
    return result


async def _live_evidence_smoke(args: argparse.Namespace) -> int:
    if not args.confirm_live_network:
        raise ValueError("live-smoke requires --confirm-live-network")
    coordinates = _route_coordinates(args.route_coordinate)
    settings = OasisSettings()
    policy = HttpPolicy(
        user_agent=settings.provider_user_agent,
        timeout_seconds=settings.provider_timeout_seconds,
        max_attempts=settings.provider_max_attempts,
        backoff_base_seconds=settings.provider_backoff_base_seconds,
        max_response_bytes=settings.provider_max_response_bytes,
        max_pages=settings.provider_max_pages,
    )
    artifact_store = LocalArtifactStore(args.artifact_root or settings.artifact_root)
    cache = LocalSnapshotCache(args.cache_root or settings.provider_cache_root)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        transport = ResilientHttpClient(client, policy=policy)
        context = ToolContext(
            run_id="live-provider-smoke",
            artifact_store=artifact_store,
            deadline_monotonic=time.monotonic() + max(30.0, policy.timeout_seconds * 6),
            cancellation=CancellationToken(),
            seed=0,
            providers={
                PLACE_PROVIDER: NominatimPlaceResolver(transport, endpoint=args.nominatim_endpoint),
                SOURCE_PROVIDER: HttpSourceSnapshotProvider(transport),
                ROUTING_PROVIDER: OsrmRoutingMatrixProvider(transport, endpoint=args.osrm_endpoint),
            },
            resources={SNAPSHOT_CACHE: cache},
        )
        registry = create_tool_registry(discover_entry_points=False)
        resolution = _require_result(
            await invoke_tool(
                registry.get("resolve_area"),
                {"query": args.place, "limit": 5},
                context,
            ),
            allow_ambiguity=True,
        )
        snapshot = _require_result(
            await invoke_tool(
                registry.get("snapshot_source"),
                {
                    "url": args.source_url,
                    "format": args.source_format,
                    "license": args.source_license,
                    "units": args.source_units,
                    "crs": args.source_crs,
                },
                context,
            )
        )
        points = gpd.GeoDataFrame(
            {"id": [f"point-{index}" for index in range(len(coordinates))]},
            geometry=[Point(longitude, latitude) for longitude, latitude in coordinates],
            crs="EPSG:4326",
        )
        point_artifact = put_vector(
            artifact_store,
            points,
            units="degrees",
            provenance=ArtifactProvenance(
                source_uri="oasis://cli/live-smoke/route-points",
                source_provider="oasis_cli",
                license="user supplied",
            ),
        )
        route = _require_result(
            await invoke_tool(
                registry.get("travel_matrix"),
                {
                    "origins_artifact_id": point_artifact.id,
                    "destinations_artifact_id": point_artifact.id,
                    "strategy": "routed_provider",
                    "output_units": "seconds",
                    "routing_profile": args.routing_profile,
                    "route_annotation": "duration",
                },
                context,
            )
        )
    print(
        json.dumps(
            {
                "place_resolution": resolution.metrics,
                "source_snapshot": snapshot.metrics,
                "travel_matrix": route.metrics,
            },
            indent=2,
        )
    )
    return 0


async def _ask(message: str) -> int:
    from oasis.api import create_app
    from oasis.api.schemas import RunCreateRequest

    app = create_app(OasisSettings())
    async with app.router.lifespan_context(app):
        manager = app.state.run_manager
        created = await manager.start(RunCreateRequest(message=message))
        result = await manager.wait(created.run_id)
        print(result.answer)
        return 1 if result.status in {"failed", "rejected"} else 0


def _run(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ask":
            return asyncio.run(_ask(args.message))
        if args.command == "chat":
            return asyncio.run(_chat(args))
        if args.command == "tools":
            return asyncio.run(_tools(args))
        if args.command == "evidence":
            return asyncio.run(_evidence(args))
        if args.command == "decision":
            return asyncio.run(_decision(args))
        if args.command == "serve":
            return _serve(args)
        if args.command == "hardware":
            return _hardware(args)
        if args.command == "runtime":
            return _runtime(args)
        if args.command == "model-worker":
            if args.worker_command == "serve":
                return _worker_serve(args)
            return asyncio.run(_worker_client(args))
        if args.command == "evaluate":
            return asyncio.run(_evaluate(args))
        if args.command == "summarize":
            return _summarize(args)
        if args.command == "demo":
            return asyncio.run(_showcase(args))
    except (ModelBackendError, RuntimePlanningError, ToolRegistryError, ValueError) as error:
        if isinstance(error, ModelBackendError):
            print(f"error[{error.detail.code}]: {error.detail.message}", file=sys.stderr)
        elif isinstance(error, RuntimePlanningError):
            print(error.rejection.model_dump_json(), file=sys.stderr)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")


def main() -> None:
    """Console-script entry point."""

    raise SystemExit(_run())


if __name__ == "__main__":
    main()
