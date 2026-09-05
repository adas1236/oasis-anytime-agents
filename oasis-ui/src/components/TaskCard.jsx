const STATUS_LABEL = {
  idle: 'Waiting',
  active: 'In progress',
  done: 'Done',
};

export default function TaskCard({ task, visual }) {
  const status = visual?.status ?? 'idle';
  const progress = visual?.progress ?? 0;
  const blurb = visual?.blurb ?? task.role;

  return (
    <article className={`task-card task-card-${status}`}>
      <div className="task-card-top">
        <span className="task-icon" aria-hidden="true">{task.icon}</span>
        <h3 className="task-name">{task.name}</h3>
        <span className={`task-status-pill task-status-${status}`}>
          {status === 'active' && <span className="pulse-dot" />}
          {STATUS_LABEL[status]}
        </span>
      </div>

      <p className="task-blurb">{blurb}</p>

      <div className="bar-track bar-track-sm">
        <div
          className={`bar-fill${status === 'active' ? ' bar-fill-active' : ''}${status === 'done' ? ' bar-fill-done' : ''}`}
          style={{ width: `${progress}%` }}
        />
      </div>
    </article>
  );
}
