"use strict";

const el = (id) => document.getElementById(id);
const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19,
}).addTo(map);
const roads = L.layerGroup().addTo(map);
const route = L.layerGroup().addTo(map);
const gps = L.layerGroup().addTo(map);
L.control.layers({}, { "Nearby roads": roads, "GPS fixes": gps, "Predicted route": route }, { collapsed: false }).addTo(map);
L.control.scale({ imperial: false }).addTo(map);
const legend = L.control({ position: "bottomright" });
legend.onAdd = () => {
  const box = L.DomUtil.create("div", "legend");
  box.innerHTML = '<span class="swatch" style="background:#819392"></span>Nearby roads<br>' +
    '<span class="swatch" style="background:#bd4f23"></span>GPS fixes<br>' +
    '<span class="swatch" style="background:#087b79"></span>Predicted route';
  return box;
};
legend.addTo(map);

let api;
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 });

function status(text, error = false) {
  el("status").textContent = text;
  el("status").dataset.error = String(error);
}

function resetResults() {
  for (const id of ["accuracy", "continuity", "sequence", "runtime"]) el(id).textContent = "—";
  el("details").hidden = true;
  el("map-empty").hidden = false;
  [roads, route, gps].forEach(layer => layer.clearLayers());
}

async function request(base, path, timeout) {
  const response = await fetch(base + path, { signal: AbortSignal.timeout(timeout) });
  if (!response.ok) throw new Error(`API returned HTTP ${response.status}`);
  return response.json();
}

async function connect() {
  resetResults();
  const input = el("server").value.trim();
  el("connect").disabled = true;
  el("run").disabled = true;
  el("trip").disabled = true;
  el("trip").replaceChildren(new Option("Loading trips…"));
  el("connection").textContent = "Connecting…";
  status("Getting available trips…");
  try {
    const url = new URL(input);
    if (!["http:", "https:"].includes(url.protocol)) throw new Error("Use an http or https API address");
    const base = input.replace(/\/+$/, "");
    const data = await request(base, "/trips", 8000);
    if (!data.trips.length) throw new Error("No trips are available");
    api = base;
    el("trip").replaceChildren(new Option("Choose a trip", ""),
      ...data.trips.map(id => new Option(id, id)));
    el("trip").disabled = false;
    localStorage.setItem("pathsnap-api", api);
    el("connection").textContent = data.mode === "evaluator" ? "Evaluator server" : "Student test server";
    status(`${data.trips.length} trips available.`);
    return true;
  } catch (error) {
    el("connection").textContent = "Server unavailable";
    el("trip").replaceChildren(new Option("Reconnect to load trips", ""));
    status(`Could not load trips. Check the API address and that both laptops can reach the server. ${error.message}`, true);
    return false;
  } finally {
    el("connect").disabled = false;
  }
}

async function match() {
  const tripId = el("trip").value;
  if (!tripId || !api) return;
  resetResults();
  el("run").disabled = true;
  el("connect").disabled = true;
  el("trip").disabled = true;
  status(`Reconstructing trip ${tripId}…`);
  try {
    const data = await request(api, "/match/" + encodeURIComponent(tripId), 60000);
    L.polyline(data.nearby_roads_latlon, { color: "#819392", weight: 1.4, opacity: .55 }).addTo(roads);
    L.polyline(data.route_latlon, { color: "#087b79", weight: 5, opacity: .9 }).addTo(route);
    for (const point of data.gps_latlon) {
      L.circleMarker(point, { radius: 2, stroke: false, fill: true, fillColor: "#bd4f23", fillOpacity: .62 }).addTo(gps);
    }
    map.fitBounds(L.latLngBounds([...data.gps_latlon, ...data.route_latlon.flat()]),
      { padding: [28, 28], maxZoom: 17 });
    el("map-empty").hidden = true;
    const evaluation = data.evaluation;
    if (evaluation) {
      for (const [id, field] of [["accuracy", "top1_accuracy"], ["continuity", "route_continuity"], ["sequence", "sequence_f1"]]) {
        el(id).textContent = evaluation[field] == null ? "—" : percent.format(evaluation[field]);
      }
    }
    el("runtime").textContent = `${data.inference_seconds.toFixed(2)} s`;
    el("details").hidden = false;
    el("details").textContent = evaluation ? `${evaluation.aligned_observations.toLocaleString()} GPS fixes with reference labels.` : "Reference scores unavailable.";
    const info = data.observations_summary;
    status(`${info.count.toLocaleString()} GPS fixes · ${info.trajectory_parts} route part${info.trajectory_parts === 1 ? "" : "s"}.`);
  } catch (error) {
    status(`Could not reconstruct this trip. ${error.message}`, true);
  } finally {
    el("trip").disabled = false;
    el("run").disabled = false;
    el("connect").disabled = false;
  }
}

el("server-form").addEventListener("submit", event => { event.preventDefault(); connect(); });
el("match-form").addEventListener("submit", event => { event.preventDefault(); match(); });
el("trip").addEventListener("change", () => { el("run").disabled = !el("trip").value; });
const params = new URLSearchParams(location.search);
el("server").value = params.get("api") || localStorage.getItem("pathsnap-api") || el("server").value;
if (params.has("api") || localStorage.getItem("pathsnap-api")) {
  connect().then(ready => {
    const tripId = params.get("trip");
    if (ready && tripId && [...el("trip").options].some(option => option.value === tripId)) {
      el("trip").value = tripId;
      match();
    }
  });
}
