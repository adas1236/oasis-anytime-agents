import { useEffect, useRef, useState } from 'react';
import MapSurface from './MapSurface';

export default function MapPanel({ status }) {
  const surfaceRef = useRef(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!expanded) return undefined;
    function onKeyDown(event) {
      if (event.key === 'Escape') setExpanded(false);
    }
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [expanded]);

  return (
    <section className={`panel map-panel${expanded ? ' map-panel-expanded' : ''}`} aria-label="Results map">
      <div className="map-panel-head">
        <div>
          <h2 className="panel-title">Map</h2>
          <p className="map-panel-caption">The map's real — the pins are still made up until this connects to an actual run.</p>
        </div>
        <div className="map-toolbar">
          <button type="button" className="map-btn" onClick={() => surfaceRef.current?.zoomOut()} aria-label="Zoom out">
            −
          </button>
          <button type="button" className="map-btn" onClick={() => surfaceRef.current?.zoomIn()} aria-label="Zoom in">
            +
          </button>
          <button type="button" className="map-btn" onClick={() => surfaceRef.current?.reset()} aria-label="Reset view">
            Reset
          </button>
          <button
            type="button"
            className="map-btn map-btn-expand"
            onClick={() => setExpanded((value) => !value)}
            aria-label={expanded ? 'Collapse map' : 'Expand map'}
          >
            {expanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
      </div>
      <div className="map-frame">
        <MapSurface ref={surfaceRef} status={status} />
      </div>
    </section>
  );
}
