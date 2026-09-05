const ROWS = [
  {
    key: 'wallTime',
    configKey: 'wallTimeS',
    label: 'Available time',
    unit: 's',
    min: 5,
    max: 300,
    step: 5,
    formatUsed: (ms) => `${(ms / 1000).toFixed(1)}s used`,
  },
  {
    key: 'modelTokens',
    configKey: 'modelTokens',
    label: 'Model tokens',
    unit: 'tok',
    min: 100,
    max: 20000,
    step: 100,
    formatUsed: (n) => `${n.toLocaleString()} used`,
  },
  {
    key: 'generatedTokens',
    configKey: 'generatedTokens',
    label: 'Generated tokens',
    unit: 'tok',
    min: 50,
    max: 8000,
    step: 50,
    formatUsed: (n) => `${n.toLocaleString()} used`,
  },
  {
    key: 'toolCalls',
    configKey: 'toolCalls',
    label: 'Tool calls',
    unit: 'calls',
    min: 1,
    max: 50,
    step: 1,
    formatUsed: (n) => `${n.toLocaleString()} used`,
  },
];

export default function BudgetBars({ budgets, budgetConfig, onBudgetConfigChange, status }) {
  const isRunning = status === 'running';

  function handleChange(row, rawValue) {
    const value = Number(rawValue);
    if (Number.isNaN(value)) return;
    onBudgetConfigChange({ ...budgetConfig, [row.configKey]: value });
  }

  return (
    <section className="panel budget-panel" aria-label="Budget">
      <h2 className="panel-title">Budget</h2>
      <p className="panel-subtitle">Set the ceiling before you hit run — it'll stop itself once it's spent.</p>
      <div className="budget-rows">
        {ROWS.map((row) => {
          const entry = budgets[row.key];
          const pct = Math.min(100, (entry.used / entry.total) * 100);
          const tight = pct >= 85;
          return (
            <div className="budget-row" key={row.key}>
              <div className="budget-row-head">
                <label className="budget-label" htmlFor={`budget-${row.key}`}>{row.label}</label>
                <div className="budget-input-wrap">
                  <input
                    id={`budget-${row.key}`}
                    className="budget-input"
                    type="number"
                    min={row.min}
                    max={row.max}
                    step={row.step}
                    value={budgetConfig[row.configKey]}
                    disabled={isRunning}
                    onChange={(event) => handleChange(row, event.target.value)}
                  />
                  <span className="budget-unit">{row.unit}</span>
                </div>
              </div>
              <div className="bar-track">
                <div
                  className={`bar-fill${tight ? ' bar-fill-tight' : ''}${isRunning ? ' bar-fill-active' : ''}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="budget-used-caption">{row.formatUsed(entry.used)}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
