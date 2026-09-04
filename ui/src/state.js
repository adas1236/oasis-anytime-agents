const STORAGE_KEY = "oasis-anytime-ui-session-v1";

export function newSession(runId, request) {
  return {
    runId,
    request,
    phase: "preparing",
    lastEventId: -1,
    events: [],
    incumbents: [],
    scorecard: null,
    result: null,
  };
}

export function loadSession() {
  try {
    const value = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null");
    return value && typeof value.runId === "string" ? value : null;
  } catch {
    return null;
  }
}

export function saveSession(session) {
  const compact = {
    ...session,
    events: session.events.slice(-120),
    incumbents: session.incumbents.slice(-120),
  };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
}

export function clearSession() {
  window.localStorage.removeItem(STORAGE_KEY);
}
