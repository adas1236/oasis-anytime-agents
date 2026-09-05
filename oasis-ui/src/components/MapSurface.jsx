import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { CANDIDATE_SITES, EXISTING_FACILITIES, MAP_CENTER, MAP_ZOOM } from '../data/mockMap';

// This is a REAL map (OpenStreetMap tiles via Leaflet) — pan, zoom, and drag
// all come from Leaflet itself. Only the pins are mock data; swap
// data/mockMap.js for the service's real plan geometry once that's wired up,
// and keep this component's ref API (zoomIn / zoomOut / reset) so
// MapPanel's toolbar doesn't need to change.

function siteIcon(className, size = 14) {
  return L.divIcon({
    className: 'map-marker-icon',
    html: `<span class="${className}"></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

const FACILITY_ICON = siteIcon('map-marker map-marker-facility', 13);
const CANDIDATE_ICON = siteIcon('map-marker map-marker-candidate', 13);
const PENDING_ICON = siteIcon('map-marker map-marker-candidate map-marker-pending', 13);
const SELECTED_ICON = siteIcon('map-marker map-marker-selected', 19);

const MapSurface = forwardRef(function MapSurface({ status }, ref) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);

  useEffect(() => {
    const map = L.map(containerRef.current, {
      center: MAP_CENTER,
      zoom: MAP_ZOOM,
      minZoom: 3,
      maxZoom: 18,
      zoomControl: false,
      attributionControl: true,
    });

    // Esri's public ArcGIS Online basemaps — no API key needed, and far more
    // tolerant of dev/demo traffic than OpenStreetMap's own tile servers
    // (which rate-limit/block automated or heavy use and were serving blocked
    // placeholder tiles here). Note the {z}/{y}/{x} order — Esri's REST tile
    // scheme swaps x and y relative to the usual {z}/{x}/{y} convention.
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      {
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom: 18,
        maxNativeZoom: 16,
      },
    ).addTo(map);

    // Added after the base layer so its labels render on top of it.
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18, maxNativeZoom: 16 },
    ).addTo(map);

    layerRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    const resizeObserver = new ResizeObserver(() => map.invalidateSize());
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    layer.clearLayers();

    const isDone = status === 'completed' || status === 'cancelled';
    const isRunning = status === 'running';

    EXISTING_FACILITIES.forEach((site) => {
      L.marker([site.lat, site.lng], { icon: FACILITY_ICON, title: 'Existing facility' }).addTo(layer);
    });

    CANDIDATE_SITES.forEach((site) => {
      const selected = isDone && site.selected;
      const pending = isRunning && site.selected;

      if (selected) {
        L.circle([site.lat, site.lng], {
          radius: 500,
          color: '#33d9c4',
          weight: 1,
          opacity: 0.5,
          fillColor: '#33d9c4',
          fillOpacity: 0.15,
        }).addTo(layer);
      }

      const icon = selected ? SELECTED_ICON : pending ? PENDING_ICON : CANDIDATE_ICON;
      L.marker([site.lat, site.lng], {
        icon,
        title: selected ? 'Selected site' : 'Candidate site',
      }).addTo(layer);
    });
  }, [status]);

  useImperativeHandle(
    ref,
    () => ({
      zoomIn: () => mapRef.current?.zoomIn(),
      zoomOut: () => mapRef.current?.zoomOut(),
      reset: () => mapRef.current?.setView(MAP_CENTER, MAP_ZOOM),
    }),
    [],
  );

  return (
    <div className="map-surface">
      <div ref={containerRef} className="map-leaflet" />

      <div className="map-legend">
        <span className="map-legend-item">
          <span className="map-swatch map-swatch-facility" /> Existing
        </span>
        <span className="map-legend-item">
          <span className="map-swatch map-swatch-candidate" /> Candidate
        </span>
        <span className="map-legend-item">
          <span className="map-swatch map-swatch-selected" /> Selected
        </span>
      </div>
    </div>
  );
});

export default MapSurface;
