import TaskCard from './TaskCard';
import { TASKS } from '../data/simulation';

export default function TaskBoard({ taskVisuals }) {
  return (
    <section className="panel task-board" aria-label="Tasks">
      <h2 className="panel-title">Tasks</h2>
      <div className="task-grid">
        {TASKS.map((task) => (
          <TaskCard key={task.id} task={task} visual={taskVisuals[task.id]} />
        ))}
      </div>
    </section>
  );
}
