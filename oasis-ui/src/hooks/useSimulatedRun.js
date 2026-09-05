import { useCallback, useEffect, useRef, useState } from 'react';
import {
  TASKS,
  STAGES,
  TIMELINE,
  VALIDATE_BURSTS,
  TOTAL_MS,
  DEFAULT_BUDGETS,
  EXAMPLES,
} from '../data/simulation';

const TICK_MS = 90;

function initialTaskVisuals() {
  const visuals = {};
  for (const task of TASKS) {
    visuals[task.id] = { status: 'idle', progress: 0, blurb: task.role };
  }
  return visuals;
}

function computeStageVisual(stage, elapsedMs) {
  if (elapsedMs <= stage.start) {
    return { status: 'idle', progress: 0, blurb: stage.blurbs[0] };
  }
  if (elapsedMs >= stage.end) {
    return { status: 'done', progress: 100, blurb: stage.blurbs[stage.blurbs.length - 1] };
  }
  const frac = (elapsedMs - stage.start) / (stage.end - stage.start);
  const idx = Math.min(stage.blurbs.length - 1, Math.floor(frac * stage.blurbs.length));
  return { status: 'active', progress: frac * 100, blurb: stage.blurbs[idx] };
}

function computeValidateVisual(elapsedMs) {
  for (const burst of VALIDATE_BURSTS) {
    if (elapsedMs >= burst.start && elapsedMs < burst.end) {
      const frac = (elapsedMs - burst.start) / (burst.end - burst.start);
      return { status: 'active', progress: frac * 100, blurb: 'Rechecking this one against the starting plan…' };
    }
  }
  const searchStage = STAGES.find((stage) => stage.taskId === 'search');
  if (elapsedMs >= searchStage.end) {
    return { status: 'done', progress: 100, blurb: 'Everything this run got double-checked.' };
  }
  if (elapsedMs > 0) {
    return { status: 'idle', progress: 0, blurb: 'Waiting on the next candidate…' };
  }
  return { status: 'idle', progress: 0, blurb: 'Waiting on the search to find something.' };
}

function computeBudgets(elapsedMs, budgetConfig) {
  const clamped = Math.min(elapsedMs, TOTAL_MS);
  const wallUsedMs = (clamped / TOTAL_MS) * (budgetConfig.wallTimeS * 1000);
  let modelTokens = 0;
  let generatedTokens = 0;
  let toolCalls = 0;
  for (const stage of STAGES) {
    if (clamped >= stage.end) {
      modelTokens += stage.modelTokens;
      generatedTokens += stage.generatedTokens;
      toolCalls += stage.toolCalls;
    } else if (clamped > stage.start) {
      const frac = (clamped - stage.start) / (stage.end - stage.start);
      modelTokens += stage.modelTokens * frac;
      generatedTokens += stage.generatedTokens * frac;
    }
  }
  return {
    wallTime: { used: wallUsedMs, total: budgetConfig.wallTimeS * 1000 },
    modelTokens: { used: Math.round(modelTokens), total: budgetConfig.modelTokens },
    generatedTokens: { used: Math.round(generatedTokens), total: budgetConfig.generatedTokens },
    toolCalls: { used: toolCalls, total: budgetConfig.toolCalls },
  };
}

export function useSimulatedRun() {
  const [status, setStatus] = useState('idle');
  const [elapsed, setElapsed] = useState(0);
  const [taskVisuals, setTaskVisuals] = useState(initialTaskVisuals);
  const [trace, setTrace] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [budgetConfig, setBudgetConfig] = useState(DEFAULT_BUDGETS);

  const intervalRef = useRef(null);
  const startedAtRef = useRef(0);
  const firedRef = useRef(new Set());

  const clearTimer = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const applyTick = useCallback((elapsedMs) => {
    const nextVisuals = {};
    for (const stage of STAGES) {
      nextVisuals[stage.taskId] = computeStageVisual(stage, elapsedMs);
    }
    nextVisuals.validate = computeValidateVisual(elapsedMs);
    setTaskVisuals(nextVisuals);

    TIMELINE.forEach((event, idx) => {
      if (elapsedMs >= event.time && !firedRef.current.has(idx)) {
        firedRef.current.add(idx);
        setTrace((prev) => [...prev, { id: idx, time: event.time, text: event.text, kind: event.kind }]);
      }
    });
  }, []);

  const reset = useCallback(() => {
    clearTimer();
    firedRef.current = new Set();
    setStatus('idle');
    setElapsed(0);
    setTaskVisuals(initialTaskVisuals());
    setTrace([]);
  }, [clearTimer]);

  const start = useCallback(
    (promptText) => {
      clearTimer();
      firedRef.current = new Set();
      setPrompt(promptText);
      setStatus('running');
      setElapsed(0);
      setTaskVisuals(initialTaskVisuals());
      setTrace([]);
      startedAtRef.current = performance.now();

      intervalRef.current = setInterval(() => {
        const elapsedMs = performance.now() - startedAtRef.current;
        if (elapsedMs >= TOTAL_MS) {
          applyTick(TOTAL_MS);
          setElapsed(TOTAL_MS);
          clearTimer();
          setStatus('completed');
          return;
        }
        applyTick(elapsedMs);
        setElapsed(elapsedMs);
      }, TICK_MS);
    },
    [applyTick, clearTimer],
  );

  const cancel = useCallback(() => {
    clearTimer();
    setStatus((prev) => {
      if (prev !== 'running') return prev;
      setTrace((trace) => [
        ...trace,
        { id: 'cancelled', time: performance.now() - startedAtRef.current, text: 'Stopped early — keeping the best plan found so far.', kind: 'system' },
      ]);
      return 'cancelled';
    });
  }, [clearTimer]);

  useEffect(() => () => clearTimer(), [clearTimer]);

  const budgets = computeBudgets(elapsed, budgetConfig);

  return {
    status,
    elapsed,
    totalMs: TOTAL_MS,
    taskVisuals,
    trace,
    budgets,
    budgetConfig,
    setBudgetConfig,
    prompt,
    start,
    cancel,
    reset,
    examples: EXAMPLES,
  };
}
