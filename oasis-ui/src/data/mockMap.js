// Placeholder sites for the results map. The map itself is real (OpenStreetMap
// tiles via Leaflet) — only these pins are made up. Swap this file for the
// service's real plan geometry once that's wired up.

// Cambridge, MA — the same city used in the project's own live-smoke example.
export const MAP_CENTER = [42.3736, -71.1097];
export const MAP_ZOOM = 14;

export const EXISTING_FACILITIES = [
  { id: 'f1', lat: 42.3695, lng: -71.1215 },
  { id: 'f2', lat: 42.3798, lng: -71.1035 },
  { id: 'f3', lat: 42.371, lng: -71.094 },
];

export const CANDIDATE_SITES = [
  { id: 'c1', lat: 42.376, lng: -71.116, selected: true },
  { id: 'c2', lat: 42.367, lng: -71.108, selected: false },
  { id: 'c3', lat: 42.382, lng: -71.109, selected: true },
  { id: 'c4', lat: 42.365, lng: -71.099, selected: false },
  { id: 'c5', lat: 42.373, lng: -71.095, selected: true },
  { id: 'c6', lat: 42.378, lng: -71.125, selected: false },
  { id: 'c7', lat: 42.369, lng: -71.1, selected: false },
];
