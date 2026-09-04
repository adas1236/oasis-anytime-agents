const SVG_NS = "http://www.w3.org/2000/svg";

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function coordinatePairs(value, output = []) {
  if (Array.isArray(value) && value.length >= 2 && value.every((item) => typeof item === "number")) {
    output.push(value);
  } else if (Array.isArray(value)) {
    for (const item of value) coordinatePairs(item, output);
  }
  return output;
}

function geometryPath(geometry, project) {
  const line = (coordinates) => coordinates.map((point, index) => {
    const [x, y] = project(point);
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  if (geometry.type === "LineString") return line(geometry.coordinates);
  if (geometry.type === "MultiLineString") return geometry.coordinates.map(line).join(" ");
  if (geometry.type === "Polygon") return geometry.coordinates.map((ring) => `${line(ring)} Z`).join(" ");
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.flatMap((polygon) => polygon.map((ring) => `${line(ring)} Z`)).join(" ");
  }
  return "";
}

function clear(container) {
  if (container.dataset.objectUrl) URL.revokeObjectURL(container.dataset.objectUrl);
  container.replaceChildren();
  delete container.dataset.objectUrl;
}

function renderGeoJson(container, collection) {
  const features = Array.isArray(collection.features) ? collection.features : [];
  const pairs = features.flatMap((feature) => coordinatePairs(feature.geometry?.coordinates));
  if (pairs.length === 0) throw new Error("The map artifact contains no renderable coordinates.");
  const xs = pairs.map(([x]) => x);
  const ys = pairs.map(([, y]) => y);
  const bounds = {
    west: Math.min(...xs), east: Math.max(...xs),
    south: Math.min(...ys), north: Math.max(...ys),
  };
  const width = 720;
  const height = 430;
  const padding = 38;
  const project = ([longitude, latitude]) => [
    padding + ((longitude - bounds.west) / Math.max(bounds.east - bounds.west, 1)) * (width - 2 * padding),
    height - padding - ((latitude - bounds.south) / Math.max(bounds.north - bounds.south, 1)) * (height - 2 * padding),
  ];
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `Plan map with ${features.length} mapped features`,
  });
  svg.append(svgElement("rect", { width, height, fill: "transparent" }));
  for (const feature of features) {
    const geometry = feature.geometry;
    if (!geometry) continue;
    if (geometry.type === "Point") {
      const [x, y] = project(geometry.coordinates);
      const selected = Boolean(feature.properties?.selected);
      const circle = svgElement("circle", {
        cx: x,
        cy: y,
        r: selected ? 10 : 6,
        class: `map-point${selected ? " selected" : ""}`,
      });
      const title = svgElement("title");
      title.textContent = `${feature.properties?.site_id ?? "Mapped location"}${selected ? " — selected" : " — candidate"}`;
      circle.append(title);
      svg.append(circle);
      continue;
    }
    const path = geometryPath(geometry, project);
    if (path) {
      const area = geometry.type.includes("Polygon");
      svg.append(svgElement("path", { d: path, class: area ? "map-area" : "map-line" }));
    }
  }
  container.append(svg);
}

export function renderMapArtifact(container, artifact) {
  clear(container);
  if (artifact.kind === "svg") {
    const url = URL.createObjectURL(artifact.content);
    const image = document.createElement("img");
    image.src = url;
    image.alt = "Current independently verified intervention plan";
    container.dataset.objectUrl = url;
    container.append(image);
    return;
  }
  renderGeoJson(container, artifact.content);
}

export function renderMapFailure(container, message) {
  clear(container);
  const note = document.createElement("p");
  note.className = "empty-state";
  note.textContent = message;
  container.append(note);
}
