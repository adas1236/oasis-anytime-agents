import { useEffect, useRef } from 'react';

const KIND_LABEL = {
  system: 'SYS',
  tool: 'TOOL',
  result: 'PLAN',
  final: 'DONE',
};

export default function TraceFeed({ trace }) {
  const listRef = useRef(null);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [trace]);

  return (
    <section className="panel trace-panel" aria-label="Event trace">
      <h2 className="panel-title">Event trace</h2>
      <ol className="trace-list" ref={listRef}>
        {trace.length === 0 && <li className="trace-empty">Trace will appear here once a run starts.</li>}
        {trace.map((entry) => (
          <li key={entry.id} className={`trace-entry trace-${entry.kind}`}>
            <span className="trace-time">{(entry.time / 1000).toFixed(1)}s</span>
            <span className={`trace-kind trace-kind-${entry.kind}`}>{KIND_LABEL[entry.kind] ?? entry.kind}</span>
            <span className="trace-text">{entry.text}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
