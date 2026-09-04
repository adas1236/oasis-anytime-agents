const SVG_NS = "http://www.w3.org/2000/svg";

function node(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

export function renderQualityChart(container, incumbents) {
  container.replaceChildren();
  if (!incumbents.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Incumbent checkpoints will appear here.";
    container.append(empty);
    return;
  }
  const width = 720;
  const height = 430;
  const padding = 46;
  const times = incumbents.map((item) => item.timeMs);
  const values = incumbents.map((item) => item.value);
  const minTime = Math.min(...times, 0);
  const maxTime = Math.max(...times, minTime + 1);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values, minValue + 1e-9);
  const x = (value) => padding + ((value - minTime) / (maxTime - minTime)) * (width - 2 * padding);
  const y = (value) => height - padding - ((value - minValue) / (maxValue - minValue)) * (height - 2 * padding);
  const svg = node("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `Verified quality history with ${incumbents.length} checkpoints`,
  });
  svg.append(node("line", { x1: padding, y1: height - padding, x2: width - padding, y2: height - padding, class: "chart-axis" }));
  svg.append(node("line", { x1: padding, y1: padding, x2: padding, y2: height - padding, class: "chart-axis" }));
  const points = incumbents.map((item) => `${x(item.timeMs)},${y(item.value)}`).join(" ");
  svg.append(node("polyline", { points, class: "chart-line" }));
  for (const item of incumbents) {
    const point = node("circle", { cx: x(item.timeMs), cy: y(item.value), r: 6, class: "chart-point" });
    const title = node("title");
    title.textContent = `${item.label}: ${item.value.toPrecision(5)} at ${(item.timeMs / 1000).toFixed(2)} seconds`;
    point.append(title);
    svg.append(point);
  }
  const xLabel = node("text", { x: width - padding, y: height - 12, "text-anchor": "end", fill: "#5d6a68", "font-size": 13 });
  xLabel.textContent = "elapsed seconds";
  svg.append(xLabel);
  container.append(svg);
}
