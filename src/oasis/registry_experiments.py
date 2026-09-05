"""Budgeted prompt-to-plan evaluation using the application's actual tool registry.

Tools receive source-data providers, never a MockCase or an oracle. The separate
observer measures candidate plans against the original case, without returning
that hidden grading information to the model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import JsonValue, ValidationError

from oasis.artifacts import LocalArtifactStore, read_json
from oasis.errors import ToolCallParseError
from oasis.llm.fake import FakeModelBackend
from oasis.llm.schemas import ChatMessage, ChatRole, ModelRequest, TokenUsage, ToolCall
from oasis.mock_experiments import (
    AgentRun,
    BudgetPoint,
    DatasetKind,
    ExperimentConfig,
    MockCase,
    OsrmMatrixStore,
    _baseline_prediction,
    _BudgetClock,
    _coverage_candidate,
    _incumbent_event,
    _wait_with_limits,
)
from oasis.providers.cache import MemorySnapshotCache
from oasis.providers.mock_dataset import DatasetEvidenceProvider, DatasetRoutingProvider
from oasis.schemas import (
    Plan,
    ToolError,
    ToolErrorCode,
    ToolEventKind,
    ToolResult,
    ToolResultStatus,
)
from oasis.tools import (
    CancellationToken,
    ToolContext,
    create_tool_registry,
    invoke_tool,
    stream_tool,
)
from oasis.tools.registry import ToolRegistryError

PROTOCOL = "live_registry_v1"


def system_prompt() -> str:
    """Common instructions; no per-case structured fields or problem-family hints."""

    return (
        "Solve the user's geospatial planning problem using the available tools. Infer the "
        "problem type, place names, depot, and constraints from the prompt. Resolve plaintext "
        "names with resolve_locations or resolve_area, inspect ambiguous candidates, and select "
        "their provider_ids explicitly with materialize_locations. Coordinates and population "
        "must come from tool results. Do not invent artifact IDs or locations. "
        "materialize_locations can copy the population metadata field and, for equal-cost "
        "facility counting, assign unit_opening_cost=1. The resulting point fields include id "
        "and name. Use build_demand and build_candidates for facility planning, travel_matrix "
        "and service_matrix for access, then compile_problem and improve. Distances in the "
        "user's prompt are kilometers. Geodesic coverage in these mock prompts means haversine "
        "distance: use travel_matrix strategy=haversine, output_units=kilometers. "
        "Driving tours use strategy=routed_provider, routing_profile=driving, "
        "route_annotation=distance, output_units=meters; these are directed OSRM road distances. "
        "For distance-only TSP the registry policy uses time_units=meters and shift_length as "
        "a distance cap; use 1e12 when the prompt imposes no cap, one vehicle, "
        "and require_return=true. "
        "For minimum facility count use min_cost_target_coverage, equal opening costs, and a "
        "coverage_target fraction and site_limit equal to the number of candidate sites when "
        "the prompt imposes no upper limit. For maximum coverage use max_weighted_coverage "
        "and site_limit. "
        "Choose improvement strategies and effort; improve returns a resumable best plan and "
        "streams feasible improvements. Use summarize_plan for verified metrics before answering. "
        "inspect_artifact reads tool-produced evidence and plans. The dataset source catalog "
        'accepts query={"names": ["place name", ...]} and returns GeoJSON snapshot URLs. '
        "Only dataset-backed sources and frozen OSRM distances are available in this evaluation."
    )


class RegistrySession:
    """One isolated public tool environment and a separate, non-model-facing grader."""

    def __init__(
        self,
        case: MockCase,
        osrm: OsrmMatrixStore | None,
        artifact_root: Path,
        seed: int,
        trace: Callable[[dict[str, Any]], None],
    ) -> None:
        self.case = case
        self.osrm = osrm
        self.store = LocalArtifactStore(artifact_root)
        self.registry = create_tool_registry(discover_entry_points=False)
        evidence = DatasetEvidenceProvider(case.locations, case.region)
        self.providers: dict[str, object] = {
            "place_resolution": evidence,
            "catalog_search": evidence,
            "source_snapshot": evidence,
        }
        if osrm is not None:
            self.providers["routing_matrix"] = DatasetRoutingProvider(osrm, case.region)
        self.resources = {"snapshot_cache": MemorySnapshotCache()}
        self.seed = seed
        self.trace = trace
        self.agent_plan_found = False

    async def prediction(self, plan: Plan) -> dict[str, Any]:
        """Independently evaluate actual decisions, never a tool's claimed objective."""

        locations = self.case.locations
        by_id = {location.location_id: i for i, location in enumerate(locations)}
        if self.case.dataset is DatasetKind.TSP:
            if plan.problem_type != "tsp" or len(plan.routes) != 1:
                raise ValueError("the requested TSP requires exactly one tour")
            ids = plan.routes[0].get("node_ids")
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise ValueError("route must contain a node_ids list")
            if not ids or ids[0] != locations[0].location_id or ids[-1] != ids[0]:
                raise ValueError("tour must start and finish at the prompt's depot")
            if len(ids) != len(locations) + 1 or set(ids[:-1]) != set(by_id):
                raise ValueError("tour must visit every requested location exactly once")
            if self.osrm is None:
                raise RuntimeError("OSRM evidence is unavailable")
            return await self.osrm.evaluate_route(
                self.case.region, [locations[by_id[str(i)]] for i in ids[:-1]]
            )
        expected_type = (
            "max_weighted_coverage"
            if self.case.dataset is DatasetKind.MAX_COVERAGE
            else "min_cost_target_coverage"
        )
        if plan.problem_type != expected_type:
            raise ValueError("candidate plan solves a different problem family")
        if len(set(plan.selected_site_ids)) != len(plan.selected_site_ids):
            raise ValueError("candidate plan repeats a facility")
        if not set(plan.selected_site_ids) <= set(by_id):
            raise ValueError("candidate plan contains sites outside the requested locations")
        if self.case.dataset is DatasetKind.MAX_COVERAGE and len(plan.selected_site_ids) > (
            self.case.centers_to_place or 0
        ):
            raise ValueError("candidate plan exceeds the prompt's center limit")
        prediction = _coverage_candidate(
            self.case.dataset,
            locations,
            [by_id[i] for i in plan.selected_site_ids],
            self.case.coverage_radius_km or 0,
            target_percent=self.case.coverage_target_percent,
        )
        if self.case.dataset is DatasetKind.MINIMUM_FACILITY and not prediction["target_reached"]:
            raise ValueError("candidate plan does not meet the prompt's population target")
        return prediction

    def comparator(self, prediction: dict[str, Any]) -> tuple[float, ...]:
        if self.case.dataset is DatasetKind.TSP:
            return (-float(prediction["distance_km"]),)
        if self.case.dataset is DatasetKind.MINIMUM_FACILITY:
            return (-float(prediction["minimum_centers"]), float(prediction["people_covered"]))
        return (float(prediction["people_covered"]),)

    async def observe(
        self,
        plan: Plan,
        name: str,
        run: AgentRun,
        clock: _BudgetClock,
        tolerance: float,
    ) -> None:
        if _time_exhausted(clock):
            return
        try:
            prediction = await self.prediction(plan)
        except ValueError as exc:
            failure = {"category": "invalid_candidate", "tool": name, "message": str(exc)}
            run.failures.append(failure)
            self.trace({"event": "candidate_rejected", **failure})
            return
        if _time_exhausted(clock):
            return
        self.agent_plan_found = True
        run.agent_plan_found = True
        if run.prediction is not None and self.comparator(prediction) < self.comparator(
            run.prediction
        ):
            return
        if (
            run.prediction_source != "baseline"
            and run.prediction is not None
            and self.comparator(prediction) == self.comparator(run.prediction)
        ):
            return
        run.prediction = prediction
        run.prediction_source = name
        run.error = None
        event = _incumbent_event(
            clock=clock, source=name, prediction=prediction, case=self.case, tolerance=tolerance
        )
        run.incumbent_timeline.append(event)
        self.trace({"event": "incumbent", **event})

    async def call(
        self,
        call: ToolCall,
        run: AgentRun,
        clock: _BudgetClock,
        tolerance: float,
        record: dict[str, Any],
    ) -> ToolResult:
        deadline = (
            math.inf
            if clock.budget.wall_time_seconds is None
            else clock.started + float(clock.budget.wall_time_seconds)
        )
        context = ToolContext(
            run_id=self.case.record_id,
            artifact_store=self.store,
            deadline_monotonic=deadline,
            cancellation=CancellationToken(),
            seed=self.seed,
            providers=self.providers,
            resources=self.resources,
        )
        try:
            tool = self.registry.get(call.name)
        except ToolRegistryError:
            return ToolResult(
                status=ToolResultStatus.FAILED,
                summary="Unknown tool.",
                error=ToolError(
                    code=ToolErrorCode.NOT_FOUND, message=f"Unknown tool {call.name!r}"
                ),
            )
        result: ToolResult | None = None
        if tool.spec.streams_candidates or tool.spec.streams_progress or tool.spec.streams_bounds:
            record["events"] = []
            async for event in stream_tool(tool, call.arguments, context):
                event_payload = {
                    "elapsed_seconds": clock.elapsed_seconds,
                    **event.model_dump(mode="json"),
                }
                record["events"].append(event_payload)
                self.trace({"event": "tool_event", "tool_call_id": call.id, **event_payload})
                if event.kind is ToolEventKind.CANDIDATE and event.candidate is not None:
                    await self.observe(event.candidate, call.name, run, clock, tolerance)
                if event.kind is ToolEventKind.RESULT:
                    result = event.result
        else:
            result = await invoke_tool(tool, call.arguments, context)
        if result is None:
            raise RuntimeError("tool completed without a terminal result")
        if result.candidate is not None:
            await self.observe(result.candidate, call.name, run, clock, tolerance)
        if call.name == "compile_problem" and result.metrics.get("baseline_plan_artifact_id"):
            plan = Plan.model_validate(
                read_json(self.store, str(result.metrics["baseline_plan_artifact_id"]))
            )
            await self.observe(plan, call.name, run, clock, tolerance)
        if call.name == "summarize_plan" and result.metrics.get("feasible"):
            plan = Plan.model_validate(
                read_json(self.store, str(call.arguments["plan_artifact_id"]))
            )
            await self.observe(plan, call.name, run, clock, tolerance)
        return result


def _time_exhausted(clock: _BudgetClock) -> bool:
    # Re-read after awaits: static narrowing must not treat time as immutable.
    return clock.time_exhausted


def _attempt_root(cell_root: Path) -> Path:
    # A retry on a fresh Pod must not overwrite the prior attempt's S3 trace.
    attempt = cell_root / "attempts" / uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


async def run_registry_case(
    case: MockCase,
    backend: Any,
    config: ExperimentConfig,
    osrm: OsrmMatrixStore | None,
    budget: BudgetPoint,
    artifact_root: Path,
) -> AgentRun:
    artifact_root = _attempt_root(artifact_root)
    trace_path = artifact_root / "trace.jsonl"
    with trace_path.open("x", encoding="utf-8") as stream:

        def trace(payload: dict[str, Any]) -> None:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        session = RegistrySession(case, osrm, artifact_root / "artifacts", config.seed, trace)
        definitions = session.registry.model_definitions()
        baseline = await _baseline_prediction(case, osrm)
        run = AgentRun(
            prediction=baseline,
            baseline_prediction=baseline,
            protocol=PROTOCOL,
            tool_names=[tool.name for tool in definitions],
            artifacts_directory=str(artifact_root),
        )
        clock = _BudgetClock(budget)
        run.incumbent_timeline.append(
            _incumbent_event(
                clock=clock,
                source="baseline",
                prediction=baseline,
                case=case,
                tolerance=config.tsp_tolerance_km,
            )
        )
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=system_prompt()),
            ChatMessage(role=ChatRole.USER, content=case.prompt),
        ]
        run.tool_spec_hash = hashlib.sha256(
            json.dumps(
                [definition.model_dump(mode="json") for definition in definitions], sort_keys=True
            ).encode()
        ).hexdigest()
        trace(
            {
                "event": "started",
                "protocol": PROTOCOL,
                "requested_budget": budget.payload(),
                "messages": [m.model_dump(mode="json") for m in messages],
                "tool_definitions": [d.model_dump(mode="json") for d in definitions],
            }
        )
        trace({"event": "incumbent", **run.incumbent_timeline[0]})

        def finish(reason: str, error: str | None = None) -> AgentRun:
            clock.sync_run(run)
            run.budget_elapsed_seconds = round(clock.elapsed_seconds, 6)
            run.terminal_reason = reason
            if error is not None:
                run.error = error
            trace(
                {
                    "event": "finished",
                    "terminal_reason": reason,
                    "error": run.error,
                    "consumed_budget": clock.consumed_payload(),
                    "prediction": run.prediction,
                }
            )
            return run

        repairs = 0
        for round_index in range(config.max_tool_rounds + 1):
            if _time_exhausted(clock):
                return finish("time_budget_exhausted")
            if clock.tokens_exhausted:
                return finish("token_budget_exhausted")
            request = ModelRequest(
                request_id=f"registry-{case.record_id}-{budget.budget_id}-{round_index}",
                messages=tuple(messages),
                tools=definitions,
                seed=config.seed + case.source_index,
                max_generated_tokens=config.max_generated_tokens,
                thinking_enabled=config.thinking,
            )
            estimated = 0
            try:
                estimated = int(
                    await _wait_with_limits(
                        backend.count_input_tokens(request), clock, config.case_timeout_seconds
                    )
                )
                if run.initial_input_tokens is None:
                    run.initial_input_tokens = estimated
                allowance = clock.generation_allowance(estimated, config.max_generated_tokens)
                context_limit = getattr(
                    getattr(backend, "capabilities", None), "context_limit", None
                )
                if isinstance(context_limit, int):
                    allowance = min(allowance, max(0, context_limit - estimated))
                if allowance < 1:
                    return finish(
                        "context_limit_exhausted"
                        if isinstance(context_limit, int) and estimated >= context_limit
                        else "token_budget_exhausted"
                    )
                request = request.model_copy(update={"max_generated_tokens": allowance})
                trace(
                    {
                        "event": "model_request",
                        "round_index": round_index,
                        "input_tokens": estimated,
                        "max_generated_tokens": allowance,
                    }
                )
                turn = await _wait_with_limits(
                    backend.generate(request), clock, config.case_timeout_seconds
                )
            except TimeoutError:
                clock.input_tokens += estimated
                run.usage_complete = False
                await backend.abort(request.request_id)
                return finish(
                    "time_budget_exhausted" if clock.time_exhausted else "model_call_timeout"
                )
            except (ToolCallParseError, ValidationError) as exc:
                raw_usage = (
                    exc.detail.context.get("token_usage")
                    if isinstance(exc, ToolCallParseError)
                    else None
                )
                if isinstance(raw_usage, dict):
                    clock.record_turn(TokenUsage.model_validate(raw_usage))
                else:
                    clock.input_tokens += estimated
                    run.usage_complete = False
                failure = {
                    "category": "malformed_model_action",
                    "message": str(exc),
                    "round_index": round_index,
                }
                if isinstance(exc, ToolCallParseError):
                    failure["output_excerpt"] = exc.detail.context.get("output_excerpt", "")
                run.failures.append(failure)
                trace({"event": "model_error", **failure})
                if repairs >= 1:
                    return finish("malformed_model_action", f"{type(exc).__name__}: {exc}")
                repairs += 1
                messages.append(
                    ChatMessage(
                        role=ChatRole.USER,
                        content=(
                            "The previous action was malformed. Return a valid tool call "
                            "using the provided schema."
                        ),
                    )
                )
                continue
            except Exception as exc:
                clock.input_tokens += estimated
                run.usage_complete = False
                error = f"{type(exc).__name__}: {exc}"
                run.failures.append({"category": "model_runtime_error", "message": error})
                return finish("model_runtime_error", error)
            clock.record_turn(turn.usage)
            run.stop_reason = turn.finish_reason.value
            messages.append(turn.message)
            trace(
                {
                    "event": "model_response",
                    "round_index": round_index,
                    **turn.model_dump(mode="json"),
                }
            )
            if (
                budget.max_total_model_tokens is not None
                and clock.input_tokens + clock.output_tokens > budget.max_total_model_tokens
            ):
                run.usage_complete = False
                return finish(
                    "token_budget_overrun", "model reported usage beyond the aggregate budget"
                )
            if _time_exhausted(clock):
                return finish("time_budget_exhausted")
            if not turn.message.tool_calls:
                run.answer_text = turn.message.content
                return finish("completed" if session.agent_plan_found else "model_stopped")
            if round_index >= config.max_tool_rounds:
                return finish("max_tool_rounds_reached")
            for call in turn.message.tool_calls:
                record = {
                    "tool_call_id": call.id,
                    "round_index": round_index,
                    "name": call.name,
                    "arguments": call.arguments,
                    "started_seconds": clock.elapsed_seconds,
                }
                run.calls.append(record)
                trace({"event": "tool_started", **record})
                if clock.time_exhausted or clock.tools_exhausted:
                    reason = (
                        "time_budget_exhausted"
                        if clock.time_exhausted
                        else "tool_call_budget_exhausted"
                    )
                    record["response"] = {"ok": False, "error": reason}
                    return finish(reason)
                clock.tool_calls += 1
                try:
                    result = await session.call(call, run, clock, config.tsp_tolerance_km, record)
                except Exception as exc:
                    result = ToolResult(
                        status=ToolResultStatus.FAILED,
                        summary="Tool execution failed.",
                        error=ToolError(code=ToolErrorCode.INTERNAL_ERROR, message=str(exc)[:1000]),
                    )
                envelope = json.loads(result.model_summary())
                record["response"] = envelope
                record["elapsed_seconds"] = clock.elapsed_seconds - record["started_seconds"]
                trace({"event": "tool_finished", **record})
                if result.error is not None:
                    category = result.error.code.value
                    failure = {
                        "category": category,
                        "tool": call.name,
                        "tool_call_id": call.id,
                        "message": result.error.message,
                        "context": result.error.context,
                    }
                    run.failures.append(failure)
                    run.error = result.error.message
                elif result.status is ToolResultStatus.INFEASIBLE:
                    run.failures.append(
                        {
                            "category": "infeasible_tool_problem",
                            "tool": call.name,
                            "tool_call_id": call.id,
                            "summary": envelope["summary"],
                            "issue_codes": result.metrics.get("issue_codes", []),
                        }
                    )
                messages.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        name=call.name,
                        tool_call_id=call.id,
                        content=result.model_summary(),
                    )
                )
                if _time_exhausted(clock):
                    return finish("time_budget_exhausted")
        return finish("max_tool_rounds_reached")


class RegistrySmokeBackend(FakeModelBackend):
    """A prompt-and-tool-results-only recipe for offline plumbing tests, not model evaluation."""

    def _next_response(self, request: ModelRequest) -> str | ToolCall:
        prompt = next(m.content or "" for m in request.messages if m.role is ChatRole.USER)
        observations = {}
        for message in request.messages:
            if message.role is ChatRole.TOOL:
                observations[message.name] = json.loads(message.content or "{}")
        for observation in observations.values():
            if observation.get("error"):
                return "The tool workflow failed: " + str(observation["error"]["message"])
            if observation.get("status") == "infeasible":
                return "The constructed problem is infeasible."
        tsp = "shortest possible total tour distance" in prompt
        minimum = "minimum number of centers" in prompt
        name = "resolve_locations"
        args: dict[str, JsonValue]
        if name not in observations:
            if tsp:
                match = re.search(r"starts at (.+?) and must visit (.+?) exactly once", prompt)
                if match is None:
                    raise ValueError("smoke recipe does not recognize this routing prompt")
                names = [match[1], *match[2].split(", ")]
            else:
                names = prompt.split(": ", 1)[1].split(". Each center", 1)[0].split(", ")
            args = {"queries": names, "limit_per_query": 1}
        elif "materialize_locations" not in observations:
            name = "materialize_locations"
            resolved = observations["resolve_locations"]
            args = {
                "resolution_artifact_id": resolved["metrics"]["resolution_artifact_id"],
                "provider_ids": [c["provider_id"] for c in resolved["summary"]["candidates"]],
                "metadata_fields": [] if tsp else ["population"],
            }
            if minimum:
                args["unit_opening_cost"] = 1
        else:
            points = observations["materialize_locations"]["metrics"]["artifact_id"]
            if not tsp and "build_demand" not in observations:
                name, args = "build_demand", {"artifact_id": points, "need_fields": ["population"]}
            elif not tsp and "build_candidates" not in observations:
                name, args = "build_candidates", {"artifact_id": points, "mode": "supplied"}
                if minimum:
                    args["opening_cost_field"] = "opening_cost"
            elif "travel_matrix" not in observations:
                name, args = (
                    "travel_matrix",
                    {
                        "origins_artifact_id": points,
                        "destinations_artifact_id": points,
                        "strategy": "routed_provider" if tsp else "haversine",
                        "output_units": "meters" if tsp else "kilometers",
                    },
                )
                if tsp:
                    args.update(routing_profile="driving", route_annotation="distance")
            elif not tsp and "service_matrix" not in observations:
                radius = re.search(r"within (\d+(?:\.\d+)?) km", prompt)
                assert radius is not None
                name, args = (
                    "service_matrix",
                    {
                        "access_matrix_artifact_id": observations["travel_matrix"]["metrics"][
                            "artifact_id"
                        ],
                        "strategy": "binary_threshold",
                        "threshold": float(radius[1]),
                    },
                )
            elif "compile_problem" not in observations:
                name = "compile_problem"
                matrix_id = observations["travel_matrix"]["metrics"]["artifact_id"]
                if tsp:
                    depot = observations["materialize_locations"]["metrics"]["location_ids"][0]
                    args = {
                        "type_id": "tsp",
                        "nodes_artifact_id": points,
                        "node_id_field": "id",
                        "travel_matrix_artifact_ids": {"roads": matrix_id},
                        "policy": {
                            "depot_ids": [depot],
                            "time_units": "meters",
                            "shift_length": 1e12,
                        },
                    }
                else:
                    match = (
                        re.search(r"at least (\d+)%", prompt)
                        if minimum
                        else re.search(r"place (\d+) centers", prompt)
                    )
                    assert match is not None
                    policy: dict[str, JsonValue] = (
                        {
                            "coverage_target": int(match[1]) / 100,
                            "site_limit": observations["materialize_locations"]["metrics"][
                                "row_count"
                            ],
                        }
                        if minimum
                        else {"site_limit": int(match[1])}
                    )
                    args = {
                        "type_id": "min_cost_target_coverage"
                        if minimum
                        else "max_weighted_coverage",
                        "demand_spec_artifact_id": observations["build_demand"]["metrics"][
                            "demand_spec_artifact_id"
                        ],
                        "candidate_spec_artifact_id": observations["build_candidates"]["metrics"][
                            "candidate_spec_artifact_id"
                        ],
                        "access_matrix_artifact_id": matrix_id,
                        "service_matrix_artifact_ids": {
                            "base": observations["service_matrix"]["metrics"]["artifact_id"]
                        },
                        "need_field": "population",
                        "policy": policy,
                    }
            elif "improve" not in observations:
                compiled = observations["compile_problem"]["metrics"]
                name, args = (
                    "improve",
                    {
                        "problem_artifact_id": compiled["problem_artifact_id"],
                        "starting_plan_artifact_id": compiled["baseline_plan_artifact_id"],
                        "strategy": "exact_enumeration",
                        "max_candidates": 100_000,
                    },
                )
            elif "summarize_plan" not in observations:
                name, args = (
                    "summarize_plan",
                    {
                        "problem_artifact_id": observations["compile_problem"]["metrics"][
                            "problem_artifact_id"
                        ],
                        "plan_artifact_id": observations["improve"]["metrics"][
                            "best_plan_artifact_id"
                        ],
                    },
                )
            else:
                return "The plan was computed and independently checked using the registry tools."
        return ToolCall(id=f"smoke-{len(observations)}-{name}", name=name, arguments=args)
