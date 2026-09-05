import { useState } from 'react';

export default function PromptPanel({ status, examples, onStart, onCancel, onReset }) {
  const [draft, setDraft] = useState('');
  const isRunning = status === 'running';
  const isFinished = status === 'completed' || status === 'cancelled';

  function handleSubmit(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || isRunning) return;
    onStart(text);
  }

  return (
    <section className="panel prompt-panel" aria-label="Prompt">
      <h2 className="panel-title prompt-title">What's the plan?</h2>
      <p className="panel-subtitle">
        Tell it what you're trying to plan — cooling centers, clinic access, mobile routing, or your own thing.
      </p>

      <form className="prompt-form" onSubmit={handleSubmit}>
        <textarea
          className="prompt-input"
          placeholder="e.g. figure out where to put cooling centers so they reach the most heat-vulnerable people…"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          disabled={isRunning}
          rows={6}
        />

        <div className="example-chips">
          {examples.map((example) => (
            <button
              type="button"
              key={example.id}
              className="chip"
              disabled={isRunning}
              onClick={() => setDraft(example.prompt)}
            >
              {example.label}
            </button>
          ))}
        </div>

        <div className="prompt-actions">
          {!isRunning ? (
            <button type="submit" className="btn btn-primary" disabled={!draft.trim()}>
              {isFinished ? 'Run again' : 'Run it'}
            </button>
          ) : (
            <button type="button" className="btn btn-danger" onClick={onCancel}>
              Stop
            </button>
          )}
          {isFinished && (
            <button type="button" className="btn btn-ghost" onClick={onReset}>
              Reset
            </button>
          )}
        </div>
      </form>
    </section>
  );
}
