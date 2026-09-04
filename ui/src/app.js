import {
  ApiError,
  cancelRun,
  createRun,
  fetchArtifactJson,
  fetchRunMap,
  inspectRun,
  loadServiceCatalogs,
  streamRunEvents,
} from "./api.js";
import { renderQualityChart } from "./chart.js";
import { renderMapArtifact, renderMapFailure } from "./map.js";
import { loadSession, newSession, saveSession } from "./state.js";

const byId = (id) => document.getElementById(id);
const elements = {
  serviceState: document.querySelector(".service-state"),
  serviceStatus: byId("service-status"),
  form: byId("run-form"),
  formError: byId("form-error"),
  launch: byId("launch-run"),
  problem: byId("problem-example"),
  problemDescription: byId("problem-description"),
  equity: byId("equity-template"),
  floors: byId("group-floor-fields"),
  profile: byId("model-profile"),
  modelId: byId("model-id"),
  wallTime: byId("wall-time"),
  totalTokens: byId("total-tokens"),
  generatedTokens: byId("generated-tokens"),
  toolCalls: byId("tool-calls"),
  enableModel: byId("enable-model"),
  thinking: byId("thinking-enabled"),
  showTrace: byId("show-trace"),
  runtimeDevice: byId("runtime-device"),
  runtimeEngine: byId("runtime-engine"),
  runtimeDtype: byId("runtime-dtype"),
  runtimeQuantization: byId("runtime-quantization"),
  runPanel: byId("run-panel"),
  runIdentity: byId("run-identity"),
  runStatus: byId("run-status"),
  cancel: byId("cancel-run"),
  notices: byId("run-notices"),
  primaryName: byId("primary-metric-name"),
  primaryValue: byId("primary-metric-value"),
  improvement: byId("baseline-improvement"),
  wallBudget: byId("wall-budget"),
  tokenBudget: byId("token-budget"),
  map: byId("plan-map"),
  mapCaption: byId("map-caption"),
  mapFormat: byId("map-format"),
  chart: byId("quality-chart"),
  overall: byId("overall-metrics"),
  groups: byId("group-metrics"),
  scenarios: byId("scenario-metrics"),
  outcome: byId("outcome-details"),
  runtime: byId("runtime-details"),
  traceCard: byId("trace-card"),
  connection: byId("connection-state"),
  trace: byId("event-trace"),
};

let catalogs = null;
let session = null;
let streamController = null;

function humanize(value) {
  return String(value ?? "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1_000) return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);
  if (Math.abs(value) > 0 && Math.abs(value) < 0.01) return value.toExponential(2);
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value);
}

function setOptions(select, values, selected, includeServer = false) {
  select.replaceChildren();
  if (includeServer) select.add(new Option("Use server policy", "server"));
  for (const value of values) select.add(new Option(humanize(value), value));
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
  select.disabled = false;
}

function selectedExample() {
  return catalogs?.problems.examples.find((item) => item.id === elements.problem.value) ?? null;
}

function populateEquityControls() {
  const example = selectedExample();
  if (!example) return;
  elements.problemDescription.textContent = `${example.description} ${example.evidence_summary}`;
  elements.toolCalls.min = String(example.preparation_tool_calls);
  if (Number(elements.toolCalls.value) < example.preparation_tool_calls) {
    elements.toolCalls.value = String(example.preparation_tool_calls);
  }
  setOptions(elements.equity, example.equity_templates, example.default_equity_template);
  populateFloorFields();
}

function populateFloorFields() {
  const example = selectedExample();
  elements.floors.replaceChildren();
  if (!example || elements.equity.value !== "floors") return;
  for (const group of example.group_names) {
    const wrapper = document.createElement("div");
    const id = `floor-${group.replace(/[^A-Za-z0-9_-]/g, "-")}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = `${humanize(group)} minimum coverage`;
    const input = document.createElement("input");
    input.id = id;
    input.type = "number";
    input.min = "0";
    input.max = "1";
    input.step = "0.05";
    input.value = String(example.default_group_floors[group] ?? 0);
    input.dataset.group = group;
    wrapper.append(label, input);
    elements.floors.append(wrapper);
  }
}

function populateCatalogs() {
  const examples = catalogs.problems.examples;
  elements.problem.replaceChildren();
  for (const example of examples) elements.problem.add(new Option(example.name, example.id));
  elements.problem.disabled = examples.length === 0;
  populateEquityControls();

  elements.profile.replaceChildren();
  for (const model of catalogs.models.models) {
    const size = model.name.replace(/^gemma4_/, "").replace(/_it$/, "").toUpperCase();
    elements.profile.add(new Option(`Gemma 4 ${size} · ${model.model_id}`, model.name));
  }
  elements.profile.value = catalogs.models.models.some((model) => model.name === catalogs.models.active_profile)
    ? catalogs.models.active_profile
    : (catalogs.models.models.find((model) => model.is_default)?.name ?? "");
  elements.profile.disabled = false;

  const options = catalogs.runtime.options;
  setOptions(elements.runtimeDevice, options.devices, "server", true);
  setOptions(elements.runtimeEngine, options.engines, "server", true);
  setOptions(elements.runtimeDtype, options.dtypes, catalogs.runtime.requested_policy.dtype);
  setOptions(elements.runtimeQuantization, ["none", ...options.quantizations], catalogs.runtime.requested_policy.quantization ?? "none");
  elements.launch.disabled = examples.length === 0 || catalogs.models.models.length === 0;
}

function floorValues() {
  return Object.fromEntries([...elements.floors.querySelectorAll("input[data-group]")].map((input) => [input.dataset.group, Number(input.value)]));
}

function runtimePolicy() {
  if (elements.runtimeDevice.value === "server" && elements.runtimeEngine.value === "server"
      && elements.runtimeDtype.value === catalogs.runtime.requested_policy.dtype
      && elements.runtimeQuantization.value === (catalogs.runtime.requested_policy.quantization ?? "none")) {
    return null;
  }
  return {
    ...catalogs.runtime.requested_policy,
    device: elements.runtimeDevice.value === "server" ? catalogs.runtime.requested_policy.device : elements.runtimeDevice.value,
    engine: elements.runtimeEngine.value === "server" ? catalogs.runtime.requested_policy.engine : elements.runtimeEngine.value,
    dtype: elements.runtimeDtype.value,
    quantization: elements.runtimeQuantization.value === "none" ? null : elements.runtimeQuantization.value,
  };
}

function buildRequest() {
  const totalTokens = Number(elements.totalTokens.value);
  const generatedTokens = Number(elements.generatedTokens.value);
  if (generatedTokens > totalTokens) throw new Error("Generated tokens cannot exceed total model tokens.");
  const availableSearchTools = catalogs.tools.tools
    .filter((tool) => tool.capability_tags.includes("decision") && tool.capability_tags.includes("search"))
    .map((tool) => tool.name);
  return {
    source: {
      kind: "example",
      example_id: elements.problem.value,
      equity_template: elements.equity.value,
      group_floors: floorValues(),
    },
    budget: {
      wall_time_ms: Number(elements.wallTime.value),
      max_total_model_tokens: totalTokens,
      max_generated_tokens: generatedTokens,
      max_tool_calls: Number(elements.toolCalls.value),
    },
    enable_model: elements.enableModel.checked,
    enable_deterministic_fallback: true,
    allowed_tools: availableSearchTools,
    thinking_enabled: elements.thinking.checked,
    model_profile: elements.profile.value,
    model_id: elements.modelId.value.trim() || null,
    runtime_policy: runtimePolicy(),
  };
}

function showFormError(error) {
  elements.formError.textContent = error instanceof ApiError ? `${error.message} (${error.code})` : error.message;
  elements.formError.hidden = false;
}

function setRunVisible() {
  elements.runPanel.hidden = false;
  elements.runIdentity.textContent = session.runId;
  elements.traceCard.hidden = !elements.showTrace.checked;
}

function renderBudget(snapshot) {
  if (!snapshot) return;
  elements.wallBudget.textContent = `${formatNumber(snapshot.wall_remaining_ms)} ms`;
  elements.tokenBudget.textContent = `${formatNumber(snapshot.model_usage?.total_tokens ?? 0)} / ${formatNumber(session.request.budget.max_total_model_tokens)}`;
}

function metricRows(metrics, prefix = "") {
  return Object.entries(metrics ?? {}).map(([name, value]) => [prefix ? `${prefix} · ${humanize(name)}` : humanize(name), formatNumber(value)]);
}

function renderTable(container, rows, emptyMessage) {
  container.replaceChildren();
  if (rows.length === 0) {
    container.className = "metric-table empty-state";
    container.textContent = emptyMessage;
    return;
  }
  container.className = "metric-table";
  const table = document.createElement("table");
  const body = document.createElement("tbody");
  for (const [label, value] of rows) {
    const row = document.createElement("tr");
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = label;
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(heading, cell);
    body.append(row);
  }
  table.append(body);
  container.append(table);
}

function primaryMetric(scorecard) {
  const entry = Object.entries(scorecard?.raw_objective ?? {})[0] ?? Object.entries(scorecard?.overall_metrics ?? {})[0];
  return entry ?? ["Primary metric", null];
}

function renderScorecard(scorecard) {
  if (!scorecard) return;
  session.scorecard = scorecard;
  const [name, value] = primaryMetric(scorecard);
  elements.primaryName.textContent = humanize(name);
  elements.primaryValue.textContent = formatNumber(value);
  const baseline = session.incumbents[0]?.value;
  const current = session.incumbents.at(-1)?.value;
  if (Number.isFinite(baseline) && Number.isFinite(current)) {
    const delta = current - baseline;
    elements.improvement.textContent = Math.abs(baseline) > 1e-12
      ? `${delta >= 0 ? "+" : ""}${formatNumber(100 * delta / Math.abs(baseline))}%`
      : `${delta >= 0 ? "+" : ""}${formatNumber(delta)}`;
  } else if (scorecard.baseline_relative_improvement != null) {
    elements.improvement.textContent = formatNumber(scorecard.baseline_relative_improvement);
  }
  renderTable(elements.overall, metricRows(scorecard.overall_metrics), "No overall metrics reported.");
  renderTable(elements.groups, Object.entries(scorecard.group_metrics ?? {}).flatMap(([group, metrics]) => metricRows(metrics, humanize(group))), "No subgroup metrics reported.");
  renderTable(elements.scenarios, Object.entries(scorecard.scenario_metrics ?? {}).flatMap(([scenario, metrics]) => metricRows(metrics, humanize(scenario))), "No scenario metrics reported.");
}

function addKeyValue(list, key, value) {
  const term = document.createElement("dt");
  const detail = document.createElement("dd");
  term.textContent = key;
  detail.textContent = String(value ?? "—");
  list.append(term, detail);
}

function evidenceAge(problem) {
  const timestamps = [];
  const visit = (value) => {
    if (!value || typeof value !== "object") return;
    if (typeof value.retrieved_at === "string") timestamps.push(Date.parse(value.retrieved_at));
    else if (typeof value.created_at === "string" && typeof value.content_hash === "string") timestamps.push(Date.parse(value.created_at));
    for (const child of Object.values(value)) visit(child);
  };
  visit(problem);
  const valid = timestamps.filter(Number.isFinite);
  if (!valid.length) return "Not reported";
  const ageMs = Math.max(0, Date.now() - Math.min(...valid));
  if (ageMs < 60_000) return "Less than one minute";
  if (ageMs < 3_600_000) return `${Math.floor(ageMs / 60_000)} minutes`;
  if (ageMs < 86_400_000) return `${Math.floor(ageMs / 3_600_000)} hours`;
  return `${Math.floor(ageMs / 86_400_000)} days`;
}

function renderRuntime(result = null) {
  elements.runtime.replaceChildren();
  const requested = session.request.runtime_policy ?? catalogs.runtime.requested_policy;
  const resolved = result?.runtime_plan ?? catalogs.runtime.resolved_plan;
  const hardware = result?.compute_inventory ?? catalogs.runtime.inventory;
  addKeyValue(elements.runtime, "Requested model", session.request.model_id ?? session.request.model_profile);
  addKeyValue(elements.runtime, "Requested policy", `${requested.device} · ${requested.engine} · ${requested.dtype}`);
  addKeyValue(elements.runtime, "Resolved model", resolved.requested_model_id);
  addKeyValue(elements.runtime, "Resolved runtime", `${resolved.runtime} · ${(resolved.device_placement ?? []).join(", ")}`);
  addKeyValue(elements.runtime, "Hardware", `${hardware.accelerator_count ?? 0} accelerator(s), ${hardware.cpu_count ?? "?"} CPU(s)`);
  addKeyValue(elements.runtime, "Hardware validation", result?.hardware_validation ?? resolved.hardware_validation ?? "pending");
}

function renderNotices(result) {
  elements.notices.replaceChildren();
  const messages = [...(result?.warnings ?? []), ...(result?.failures ?? [])];
  for (const message of messages) {
    const notice = document.createElement("div");
    notice.className = `notice${result.failures?.includes(message) ? " error" : ""}`;
    notice.textContent = message;
    elements.notices.append(notice);
  }
}

async function renderOutcome(result) {
  elements.outcome.replaceChildren();
  addKeyValue(elements.outcome, "Status", humanize(result.status));
  addKeyValue(elements.outcome, "Termination", humanize(result.terminal_reason));
  addKeyValue(elements.outcome, "Feasible", result.best_scorecard?.feasible ? "Yes" : "No validated plan");
  addKeyValue(elements.outcome, "Deadline overshoot", `${result.deadline_overshoot_ms} ms`);
  try {
    const problem = await fetchArtifactJson(result.problem_artifact_id);
    addKeyValue(elements.outcome, "Evidence age", evidenceAge(problem));
  } catch {
    addKeyValue(elements.outcome, "Evidence age", "Unavailable");
  }
}

function appendEvent(event) {
  session.events.push(event);
  const item = document.createElement("li");
  const time = document.createElement("time");
  time.textContent = `${(event.relative_monotonic_ms / 1000).toFixed(2)}s`;
  const kind = document.createElement("span");
  kind.className = "event-kind";
  kind.textContent = humanize(event.kind);
  const detail = document.createElement("span");
  const eventDetail = event.payload?.rationale ?? event.payload?.summary ?? event.payload?.reason;
  detail.textContent = typeof eventDetail === "string"
    ? eventDetail
    : (eventDetail ? JSON.stringify(eventDetail) : humanize(event.actor));
  item.append(time, kind, detail);
  elements.trace.append(item);
  while (elements.trace.children.length > 120) elements.trace.firstElementChild.remove();
  item.scrollIntoView({ block: "nearest" });
  elements.runStatus.textContent = humanize(event.state);
  renderBudget(event.budget_after);
}

async function updateIncumbent(event) {
  const scorecardId = event.artifact_ids?.[1];
  if (!scorecardId) return;
  try {
    const scorecard = await fetchArtifactJson(scorecardId);
    const value = Number(scorecard.comparator_key?.[0]);
    if (Number.isFinite(value) && !session.incumbents.some((item) => item.eventId === event.sequence)) {
      session.incumbents.push({
        eventId: event.sequence,
        timeMs: event.relative_monotonic_ms,
        value,
        label: humanize(event.kind),
      });
    }
    renderScorecard(scorecard);
    renderQualityChart(elements.chart, session.incumbents);
    const artifact = await fetchRunMap(session.runId, elements.mapFormat.value);
    renderMapArtifact(elements.map, artifact);
    elements.mapCaption.textContent = `Verified at ${(event.relative_monotonic_ms / 1000).toFixed(2)} seconds`;
  } catch (error) {
    renderMapFailure(elements.map, error instanceof Error ? error.message : "Map unavailable.");
  }
}

async function renderFinal(result) {
  session.result = result;
  session.phase = "finalized";
  elements.runStatus.textContent = humanize(result.status);
  elements.cancel.disabled = true;
  renderBudget(result.consumed_budget);
  if (result.best_scorecard) renderScorecard(result.best_scorecard);
  if (Array.isArray(result.incumbent_timeline) && result.incumbent_timeline.length) {
    session.incumbents = result.incumbent_timeline.map((item, index) => ({
      eventId: index,
      timeMs: item.committed_at_ms,
      value: Number(item.comparator_key?.[0] ?? 0),
      label: index === 0 ? "Baseline" : "Improvement",
    }));
    renderQualityChart(elements.chart, session.incumbents);
  }
  renderRuntime(result);
  renderNotices(result);
  await renderOutcome(result);
  if (result.best_plan_artifact_id) {
    try {
      renderMapArtifact(elements.map, await fetchRunMap(session.runId, elements.mapFormat.value));
      elements.mapCaption.textContent = "Final independently verified plan";
    } catch (error) {
      renderMapFailure(elements.map, error instanceof Error ? error.message : "Map unavailable.");
    }
  }
  saveSession(session);
}

async function handleEvent(message) {
  session.lastEventId = message.id;
  const event = message.data;
  if (!session.events.some((item) => item.sequence === event.sequence)) appendEvent(event);
  if (["baseline_committed", "incumbent_committed"].includes(event.kind)) await updateIncumbent(event);
  saveSession(session);
}

function onConnection(state) {
  elements.connection.textContent = humanize(state);
}

async function connectToRun() {
  streamController?.abort();
  streamController = new AbortController();
  try {
    await streamRunEvents(session.runId, {
      afterEventId: session.lastEventId,
      signal: streamController.signal,
      onEvent: handleEvent,
      onConnection,
    });
    if (streamController.signal.aborted) return;
    const inspection = await inspectRun(session.runId);
    session.phase = inspection.phase;
    if (inspection.result) await renderFinal(inspection.result);
  } catch (error) {
    if (!streamController.signal.aborted) {
      onConnection("disconnected");
      showFormError(error);
    }
  }
}

async function launch(event) {
  event.preventDefault();
  elements.formError.hidden = true;
  elements.launch.disabled = true;
  try {
    const request = buildRequest();
    const created = await createRun(request);
    session = newSession(created.run_id, request);
    elements.trace.replaceChildren();
    elements.cancel.disabled = false;
    setRunVisible();
    renderRuntime();
    saveSession(session);
    elements.runPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    await connectToRun();
  } catch (error) {
    showFormError(error);
  } finally {
    elements.launch.disabled = false;
  }
}

async function requestCancellation() {
  elements.cancel.disabled = true;
  elements.runStatus.textContent = "Cancelling";
  try {
    const response = await cancelRun(session.runId);
    if (response.result) await renderFinal(response.result);
  } catch (error) {
    showFormError(error);
    elements.cancel.disabled = false;
  }
}

function restoreTrace() {
  elements.trace.replaceChildren();
  const storedEvents = session.events;
  session.events = [];
  for (const event of storedEvents) appendEvent(event);
  renderQualityChart(elements.chart, session.incumbents ?? []);
  if (session.scorecard) renderScorecard(session.scorecard);
}

async function initialize() {
  try {
    catalogs = await loadServiceCatalogs();
    populateCatalogs();
    elements.serviceState.classList.add("ready");
    elements.serviceStatus.textContent = `API ${catalogs.health.api_version} ready · ${catalogs.models.active_model_id}`;
    session = loadSession();
    if (session) {
      setRunVisible();
      restoreTrace();
      renderRuntime(session.result);
      if (session.scorecard) {
        try {
          renderMapArtifact(elements.map, await fetchRunMap(session.runId, elements.mapFormat.value));
          elements.mapCaption.textContent = "Restored latest independently verified plan";
        } catch (error) {
          renderMapFailure(elements.map, error instanceof Error ? error.message : "Map unavailable.");
        }
      }
      if (session.result) await renderFinal(session.result);
      else await connectToRun();
    }
  } catch (error) {
    elements.serviceState.classList.add("failed");
    elements.serviceStatus.textContent = "Planning service unavailable";
    showFormError(error);
  }
}

elements.problem.addEventListener("change", populateEquityControls);
elements.equity.addEventListener("change", populateFloorFields);
elements.form.addEventListener("submit", launch);
elements.cancel.addEventListener("click", requestCancellation);
elements.showTrace.addEventListener("change", () => {
  elements.traceCard.hidden = !elements.showTrace.checked;
});
elements.mapFormat.addEventListener("change", async () => {
  if (!session?.scorecard) return;
  try {
    renderMapArtifact(elements.map, await fetchRunMap(session.runId, elements.mapFormat.value));
  } catch (error) {
    renderMapFailure(elements.map, error instanceof Error ? error.message : "Map unavailable.");
  }
});
window.addEventListener("beforeunload", () => streamController?.abort());

initialize();
