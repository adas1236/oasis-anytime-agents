"""Command-line entry point for chat and tool inspection."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import httpx
from shapely.geometry import Point

from oasis.artifacts import ArtifactProvenance, LocalArtifactStore, put_vector
from oasis.config import BackendKind, DevicePolicy, OasisSettings
from oasis.decision import run_decision_demo
from oasis.errors import ModelBackendError
from oasis.evidence import run_evidence_demo
from oasis.llm.fake import FakeModelBackend
from oasis.llm.profiles import MODEL_PROFILES, resolve_model_profile
from oasis.llm.protocols import ModelBackend
from oasis.llm.schemas import ChatMessage, ChatRole, ModelRequest, TokenUsage
from oasis.llm.transformers_backend import TransformersModelBackend
from oasis.providers import (
    HttpPolicy,
    HttpSourceSnapshotProvider,
    LocalSnapshotCache,
    NominatimPlaceResolver,
    OsrmRoutingMatrixProvider,
    ResilientHttpClient,
    SourceFormat,
)
from oasis.schemas import ToolResult, ToolResultStatus
from oasis.tools import CancellationToken, ToolContext, create_tool_registry, invoke_tool
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
    chat = subparsers.add_parser("chat", help="stream a raw, multi-turn text chat")
    chat.add_argument("--backend", choices=[kind.value for kind in BackendKind])
    chat.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    chat.add_argument("--model", dest="model_id", help="explicit Hugging Face model ID")
    chat.add_argument("--revision", dest="model_revision")
    chat.add_argument("--device", choices=[policy.value for policy in DevicePolicy])
    chat.add_argument("--dtype")
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
    return parser


def _settings_from_args(args: argparse.Namespace) -> OasisSettings:
    cli_values = {
        "backend": args.backend,
        "model_profile": args.profile,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "device": args.device,
        "dtype": args.dtype,
        "max_generated_tokens": args.max_generated_tokens,
        "thinking": args.thinking,
        "trust_remote_code": args.trust_remote_code,
    }
    return OasisSettings.resolve(cli_overrides=cli_values)


def _make_backend(settings: OasisSettings) -> ModelBackend:
    profile = resolve_model_profile(settings.model_profile, settings.model_id)
    if settings.backend is BackendKind.FAKE:
        return FakeModelBackend(profile=profile)
    return TransformersModelBackend(
        profile_name=settings.model_profile,
        model_id=settings.model_id,
        revision=settings.model_revision,
        device=settings.device,
        dtype=settings.dtype,
        trust_remote_code=settings.trust_remote_code,
    )


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
    backend = _make_backend(settings)
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
    registry = create_tool_registry(discover_entry_points=not args.no_plugins)
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
        result = await run_decision_demo(args.artifact_root)
        print(result.model_dump_json(indent=2))
        return 0
    raise ValueError(f"unknown decision command: {args.decision_command}")


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


def _run(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            return asyncio.run(_chat(args))
        if args.command == "tools":
            return asyncio.run(_tools(args))
        if args.command == "evidence":
            return asyncio.run(_evidence(args))
        if args.command == "decision":
            return asyncio.run(_decision(args))
    except (ModelBackendError, ToolRegistryError, ValueError) as error:
        if isinstance(error, ModelBackendError):
            print(f"error[{error.detail.code}]: {error.detail.message}", file=sys.stderr)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")


def main() -> None:
    """Console-script entry point."""

    raise SystemExit(_run())


if __name__ == "__main__":
    main()
