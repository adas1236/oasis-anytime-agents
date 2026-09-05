import { checkService, createRun, inspectRun, cancelRun, streamRunEvents, fetchRunMap } from "./api.js";
import { newSession, loadSession, saveSession, clearSession } from "./state.js";
import { renderMapArtifact, renderMapFailure } from "./map.js";
import { renderQualityChart } from "./chart.js";

const byId = (id) => document.getElementById(id);
const form = byId("run-form");
const launch = byId("launch-run");
const stop = byId("cancel-run");
let session = null;
let streamController = null;
let lastMapAt = 0;

function errorMessage(error) {
  byId("form-error").textContent = error instanceof Error ? error.message : String(error);
  byId("form-error").hidden = false;
}

function visibleSession() {
  byId("run-panel").hidden = false;
  byId("request-message").textContent = session.request.message;
  byId("message").value = session.request.message;
  byId("answer").textContent = session.answer ?? "";
  byId("plan-details").hidden = !session.incumbents.length;
  byId("event-trace").replaceChildren();
  for (const event of session.events) renderEvent(event);
}

function renderEvent(event) {
  const row = document.createElement("li");
  const time = document.createElement("time");
  time.textContent = `${(event.relative_monotonic_ms / 1000).toFixed(1)}s`;
  const kind = document.createElement("span");
  kind.textContent = event.kind.replaceAll("_", " ");
  const detail = document.createElement("span");
  detail.textContent = event.payload?.tool ?? event.payload?.reason ?? "";
  row.append(time, kind, detail);
  byId("event-trace").append(row);
  while (byId("event-trace").children.length > 120) byId("event-trace").firstChild.remove();
  const budget = event.budget_after;
  if (budget) {
    byId("usage").textContent = `${(budget.wall_elapsed_ms / 1000).toFixed(1)} seconds · ` +
      `${budget.tool_calls} tool calls · ${budget.model_usage?.total_tokens ?? 0} tokens`;
  }
}

async function updateMap(runId) {
  try {
    const artifact = await fetchRunMap(runId);
    if (session?.runId === runId) renderMapArtifact(byId("plan-map"), artifact);
  } catch {
    if (session?.runId === runId) renderMapFailure(byId("plan-map"), "A map is unavailable for this answer.");
  }
}

async function receive(event) {
  const value = event.data;
  session.lastEventId = event.id;
  if (session.events.some((item) => item.sequence === value.sequence)) return;
  session.events.push(value);
  renderEvent(value);
  if (value.kind === "tool_started") byId("activity").textContent = "Gathering evidence and working through the problem…";
  if (value.kind === "model_action_proposed") byId("activity").textContent = "Considering the next step…";
  if (value.kind === "problem_compiled" && session.problemId !== value.artifact_ids?.[0]) {
    session.problemId = value.artifact_ids?.[0];
    session.incumbents = [];
    session.answer = "";
    byId("answer").textContent = "";
    byId("plan-details").hidden = true;
    byId("plan-map").replaceChildren();
    byId("quality-chart").replaceChildren();
  }
  if (value.kind === "incumbent_committed") {
    byId("activity").textContent = "A checked plan is available. Continuing to improve it…";
    if (session.problemHash !== value.payload.problem_hash) {
      session.incumbents = [];
      session.problemHash = value.payload.problem_hash;
    }
    session.incumbents.push({timeMs: value.relative_monotonic_ms,
      value: Number(value.payload.comparator_key?.[0] ?? 0), label: "Checked plan"});
    byId("plan-details").hidden = false;
    renderQualityChart(byId("quality-chart"), session.incumbents);
    if (value.payload.answer) {
      session.answer = value.payload.answer;
      byId("answer").textContent = session.answer;
    }
    if (Date.now() - lastMapAt > 1000) {
      lastMapAt = Date.now();
      await updateMap(session.runId);
    }
  }
  saveSession(session);
}

async function finish(result) {
  session.result = result;
  session.phase = "finalized";
  session.answer = result.answer ?? "This run finished without a written answer.";
  byId("answer").textContent = session.answer;
  byId("run-status").textContent = result.status === "complete" ? "Answered" : result.status;
  byId("activity").textContent = result.status === "complete" ? "" : "Showing the result available when work ended.";
  byId("run-notices").replaceChildren();
  if (result.answer_source === "model" && result.best_plan && result.status === "complete") {
    byId("activity").textContent = "Plan checked against the formulated problem.";
  }
  stop.disabled = true;
  launch.disabled = false;
  byId("plan-details").hidden = !result.best_plan;
  if (result.best_plan) {
    byId("plan-details").hidden = false;
    renderQualityChart(byId("quality-chart"), session.incumbents);
    await updateMap(session.runId);
  }
  saveSession(session);
}

async function followRun() {
  streamController?.abort();
  const controller = new AbortController();
  streamController = controller;
  const runId = session.runId;
  launch.disabled = true;
  stop.disabled = false;
  byId("run-status").textContent = "Working";
  try {
    await streamRunEvents(runId, {
      afterEventId: session.lastEventId, signal: controller.signal, onEvent: receive,
      onConnection: (state) => { byId("connection-state").textContent = state; },
    });
    if (controller.signal.aborted) return;
    const inspection = await inspectRun(runId);
    if (inspection.result) await finish(inspection.result);
  } catch (error) {
    if (!controller.signal.aborted) errorMessage(error);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = byId("message").value.trim();
  if (!message) return;
  launch.disabled = true;
  byId("form-error").hidden = true;
  try {
    const request = { message };
    const created = await createRun(request);
    session = newSession(created.run_id, request);
    visibleSession();
    byId("activity").textContent = "Working on your question…";
    saveSession(session);
    await followRun();
  } catch (error) {
    errorMessage(error);
    launch.disabled = false;
  }
});

stop.addEventListener("click", async () => {
  stop.disabled = true;
  byId("activity").textContent = "Stopping and saving the available answer…";
  try {
    const response = await cancelRun(session.runId);
    if (response.result) {
      streamController?.abort();
      await finish(response.result);
    }
  } catch (error) {
    errorMessage(error);
    stop.disabled = false;
  }
});

async function initialize() {
  try {
    await checkService();
    byId("service-status").textContent = "Ready";
    document.querySelector(".service-state").classList.add("ready");
    launch.disabled = false;
    const saved = loadSession();
    if (!saved?.request?.message) { clearSession(); return; }
    session = saved;
    visibleSession();
    const inspection = await inspectRun(session.runId);
    if (inspection.result) await finish(inspection.result);
    else await followRun();
  } catch (error) {
    byId("service-status").textContent = "Connection unavailable";
    errorMessage(error);
  }
}

initialize();
