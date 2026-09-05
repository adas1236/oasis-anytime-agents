const STATUS_META = {
  idle: { label: 'Idle', dot: 'idle' },
  running: { label: 'Running', dot: 'running' },
  completed: { label: 'Completed', dot: 'completed' },
  cancelled: { label: 'Cancelled', dot: 'cancelled' },
};

export default function Header({ status }) {
  const meta = STATUS_META[status] ?? STATUS_META.idle;

  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 32 32" width="26" height="26">
            <circle cx="16" cy="16" r="14" fill="none" stroke="currentColor" strokeWidth="2.5" opacity="0.35" />
            <circle cx="16" cy="16" r="8.5" fill="none" stroke="currentColor" strokeWidth="2.5" />
            <circle cx="16" cy="16" r="2.6" fill="currentColor" />
          </svg>
        </div>
        <div className="brand-text">
          <span className="brand-name">OASIS</span>
          <span className="brand-tagline">Anytime GeoAI Agent Console</span>
        </div>
      </div>
      <div className="header-status">
        <span className={`status-pill status-${meta.dot}`}>
          <span className="status-dot" />
          {meta.label}
        </span>
      </div>
    </header>
  );
}
