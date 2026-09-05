const configuredBase = document.querySelector('meta[name="oasis-api-base"]')?.content ?? "/api/v1";
const API_ROOT = new URL(`${configuredBase.replace(/\/$/, "")}/`, window.location.origin);

export class ApiError extends Error {
  constructor(message, status = 0, code = "network_error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

function endpoint(relative) {
  return new URL(relative.replace(/^\//, ""), API_ROOT);
}

async function apiRequest(relative, options = {}) {
  let response;
  try {
    response = await fetch(endpoint(relative), {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch (error) {
    throw new ApiError(error instanceof Error ? error.message : "The API is unreachable.");
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.error ?? {};
    throw new ApiError(detail.message ?? `Request failed with status ${response.status}.`, response.status, detail.code);
  }
  return response;
}

async function json(relative, options) {
  return (await apiRequest(relative, options)).json();
}

export function checkService() {
  return json("health");
}

export async function loadServiceCatalogs() {
  const [health, models, runtime, tools, problems] = await Promise.all([
    json("health"),
    json("models"),
    json("runtime"),
    json("tools"),
    json("problems"),
  ]);
  return { health, models, runtime, tools, problems };
}

export function createRun(payload) {
  return json("runs", { method: "POST", body: JSON.stringify(payload) });
}

export function inspectRun(runId) {
  return json(`runs/${encodeURIComponent(runId)}`);
}

export function cancelRun(runId) {
  return json(`runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" });
}

export function fetchArtifactJson(artifactId) {
  return json(`artifacts/${encodeURIComponent(artifactId)}`);
}

export async function fetchRunMap(runId, format = "geojson") {
  const url = `runs/${encodeURIComponent(runId)}/map?format=${encodeURIComponent(format)}`;
  const response = await apiRequest(url, { headers: { Accept: "application/geo+json,image/svg+xml" } });
  const contentType = response.headers.get("content-type") ?? "application/octet-stream";
  if (contentType.includes("json")) {
    return { kind: "geojson", content: await response.json() };
  }
  return { kind: "svg", content: await response.blob() };
}

function parseEventBlock(block) {
  let id = null;
  let type = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("id:")) id = Number(line.slice(3).trim());
    if (line.startsWith("event:")) type = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!Number.isInteger(id) || data.length === 0) return null;
  return { id, type, data: JSON.parse(data.join("\n")) };
}

async function consumeEventStream(response, onEvent, signal) {
  if (!response.body) throw new ApiError("This browser cannot read streaming responses.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (!signal.aborted) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (block && !block.startsWith(":")) {
        const event = parseEventBlock(block);
        if (event) await onEvent(event);
      }
      boundary = buffer.indexOf("\n\n");
    }
    if (done) return;
  }
  await reader.cancel();
}

function delay(milliseconds, signal) {
  return new Promise((resolve) => {
    const timer = window.setTimeout(resolve, milliseconds);
    signal.addEventListener("abort", () => {
      window.clearTimeout(timer);
      resolve();
    }, { once: true });
  });
}

export async function streamRunEvents(runId, options) {
  let cursor = Number.isInteger(options.afterEventId) ? options.afterEventId : -1;
  while (!options.signal.aborted) {
    options.onConnection?.(cursor < 0 ? "connecting" : "reconnecting");
    try {
      const headers = { Accept: "text/event-stream" };
      if (cursor >= 0) headers["Last-Event-ID"] = String(cursor);
      const response = await apiRequest(`runs/${encodeURIComponent(runId)}/events`, {
        headers,
        signal: options.signal,
      });
      options.onConnection?.("connected");
      await consumeEventStream(response, async (event) => {
        if (event.id <= cursor) return;
        cursor = event.id;
        await options.onEvent(event);
      }, options.signal);
      return cursor;
    } catch (error) {
      if (options.signal.aborted) return cursor;
      if (error instanceof ApiError && error.status >= 400 && error.status < 500) throw error;
      options.onConnection?.("retrying");
      await delay(750, options.signal);
    }
  }
  return cursor;
}
