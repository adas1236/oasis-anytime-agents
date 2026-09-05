import './App.css';
import Header from './components/Header';
import PromptPanel from './components/PromptPanel';
import BudgetBars from './components/BudgetBars';
import TaskBoard from './components/TaskBoard';
import MapPanel from './components/MapPanel';
import TraceFeed from './components/TraceFeed';
import { useSimulatedRun } from './hooks/useSimulatedRun';

export default function App() {
  const {
    status,
    taskVisuals,
    trace,
    budgets,
    budgetConfig,
    setBudgetConfig,
    start,
    cancel,
    reset,
    examples,
  } = useSimulatedRun();

  return (
    <div className="app-shell">
      <Header status={status} />

      <main className="workspace-grid">
        <div className="workspace-prompt-area">
          <PromptPanel status={status} examples={examples} onStart={start} onCancel={cancel} onReset={reset} />
          <BudgetBars
            budgets={budgets}
            budgetConfig={budgetConfig}
            onBudgetConfigChange={setBudgetConfig}
            status={status}
          />
        </div>

        <div className="workspace-map-area">
          <MapPanel status={status} />
        </div>

        <aside className="workspace-tasks">
          <TaskBoard taskVisuals={taskVisuals} />
        </aside>

        <div className="workspace-events-area">
          <TraceFeed trace={trace} />
        </div>
      </main>
    </div>
  );
}
