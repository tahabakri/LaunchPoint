import {
  drawMostLikely,
  drawOsmBasemap,
  drawRasterInView,
  drawSightings,
  formatArea,
  formatCoord,
  formatMeters,
  formatProbability,
  rasterCellAt,
  rasterToDataUrl,
  rasterValueAt,
} from "./rendering.js";
import { TerrainScene } from "./terrain3d.js";

const OSM_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const DEFAULT_STAGES = ["Prepare AOI", "Fetch surfaces", "Run fusion", "Serialize result", "Export ready"];

const state = {
  map: null,
  terrain: null,
  sightings: [],
  selectedSighting: null,
  selectedCell: null,
  inspectorLocked: false,
  addingPin: false,
  analysis: null,
  overlays: {
    heatmap: null,
    credible: null,
    diagnostic: null,
    markers: null,
    rays: null,
  },
  layers: {
    probability: true,
    credible: true,
    sightings: true,
  },
  activeView: "map",
  activeDiagnostic: "",
  tileSources: {
    map: OSM_TEMPLATE,
    terrain: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
  },
};

const el = {};

document.addEventListener("DOMContentLoaded", init);

function init() {
  cacheElements();
  initMap();
  initTerrain();
  bindEvents();
  refreshIcons();
  fetchDefaults();
  renderSightings();
  updateRunReadiness();
}

function cacheElements() {
  [
    "addPinButton",
    "importButton",
    "clearButton",
    "fileInput",
    "sightingCount",
    "sightingList",
    "maxRangeInput",
    "samplesInput",
    "combineSelect",
    "antennaInput",
    "gpuInput",
    "canopyInput",
    "buildingsInput",
    "cacheInput",
    "runButton",
    "runProgress",
    "runProgressBar",
    "runMessage",
    "rayModeSelect",
    "mostLikelyValue",
    "credibleAreaValue",
    "runStateBadge",
    "limitationsNote",
    "exportTiffButton",
    "exportPngButton",
    "inspectorBody",
    "timeline",
    "contributionSelect",
    "metadataList",
    "rightRail",
    "collapseRightButton",
    "terrainHost",
    "terrainStatus",
    "verticalScaleInput",
    "terrainBuildingsInput",
    "terrainCanopyInput",
    "reloadTerrainButton",
  ].forEach((id) => {
    el[id] = document.getElementById(id);
  });
}

function initMap() {
  state.map = L.map("map", {
    worldCopyJump: true,
    preferCanvas: true,
    zoomControl: true,
    minZoom: 2,
  }).setView([20, 0], 2);

  L.tileLayer(OSM_TEMPLATE, {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);

  state.map.createPane("diagnosticPane");
  state.map.createPane("heatmapPane");
  state.map.createPane("maskPane");
  state.map.getPane("diagnosticPane").style.zIndex = 340;
  state.map.getPane("heatmapPane").style.zIndex = 350;
  state.map.getPane("maskPane").style.zIndex = 360;

  state.map.on("click", handleMapClick);
  state.map.on("mousemove", handleMapHover);
}

function initTerrain() {
  state.terrain = new TerrainScene(el.terrainHost, {
    statusEl: el.terrainStatus,
    terrainTemplate: state.tileSources.terrain,
  });
}

function bindEvents() {
  el.addPinButton.addEventListener("click", () => {
    state.addingPin = !state.addingPin;
    el.addPinButton.classList.toggle("active", state.addingPin);
    el.runMessage.textContent = state.addingPin
      ? "Click the map to place a sighting."
      : statusText();
  });
  el.importButton.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", handleImport);
  el.clearButton.addEventListener("click", clearSightings);
  el.runButton.addEventListener("click", runAnalysis);
  el.exportTiffButton.addEventListener("click", exportGeotiff);
  el.exportPngButton.addEventListener("click", exportPngPreview);
  el.rayModeSelect.addEventListener("change", () => {
    renderRays();
    syncTerrainState();
  });
  el.contributionSelect.addEventListener("change", updateRasterOverlays);
  el.collapseRightButton.addEventListener("click", () => {
    el.rightRail.classList.toggle("collapsed");
  });
  el.verticalScaleInput.addEventListener("input", () => {
    state.terrain.setVerticalScale(Number(el.verticalScaleInput.value));
  });
  el.terrainBuildingsInput.addEventListener("change", syncTerrainState);
  el.terrainCanopyInput.addEventListener("change", syncTerrainState);
  el.reloadTerrainButton.addEventListener("click", loadTerrainForCurrentView);

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  document.querySelectorAll(".layer-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const layer = button.dataset.layer;
      state.layers[layer] = !state.layers[layer];
      button.classList.toggle("active", state.layers[layer]);
      updateRasterOverlays();
      renderMarkers();
      syncTerrainState();
    });
  });
  document.querySelectorAll(".diagnostic-layer").forEach((input) => {
    input.addEventListener("change", () => {
      document.querySelectorAll(".diagnostic-layer").forEach((other) => {
        if (other !== input) other.checked = false;
      });
      state.activeDiagnostic = input.checked ? input.dataset.layer : "";
      updateRasterOverlays();
      syncTerrainState();
    });
  });
}

async function fetchDefaults() {
  try {
    const response = await fetch("/api/defaults");
    const payload = await response.json();
    state.tileSources = payload.tileSources || state.tileSources;
    state.terrain.setTerrainTemplate(state.tileSources.terrain);
    if (payload.settings) {
      el.maxRangeInput.value = payload.settings.maxRangeM ?? el.maxRangeInput.value;
      el.samplesInput.value = payload.settings.samplesPerSighting ?? el.samplesInput.value;
      if (payload.settings.combine) el.combineSelect.value = payload.settings.combine;
      el.antennaInput.value = payload.settings.antennaHeightM ?? el.antennaInput.value;
      el.gpuInput.checked = payload.settings.preferGpu !== false;
    }
  } catch {
    // Defaults are already embedded for offline UI startup.
  }
}

function handleMapClick(event) {
  if (state.addingPin) {
    addSighting({
      label: `S${state.sightings.length + 1}`,
      lat: event.latlng.lat,
      lon: event.latlng.lng,
      altitude: 100,
      position_sigma_m: 250,
      altitude_sigma_m: 30,
    });
    state.addingPin = false;
    el.addPinButton.classList.remove("active");
    return;
  }
  if (state.analysis) {
    state.selectedCell = { lat: event.latlng.lat, lon: event.latlng.lng };
    state.inspectorLocked = true;
    updateInspector(event.latlng);
    renderRays();
    syncTerrainState();
  }
}

function handleMapHover(event) {
  if (!state.analysis || state.inspectorLocked) return;
  updateInspector(event.latlng);
}

function addSighting(sighting) {
  state.sightings.push(normalizeSighting(sighting, state.sightings.length));
  state.selectedSighting = state.sightings.length - 1;
  renderSightings();
  updateRunReadiness();
  renderMarkers();
}

function clearSightings() {
  state.sightings = [];
  state.selectedSighting = null;
  state.selectedCell = null;
  state.inspectorLocked = false;
  state.analysis = null;
  clearRasterOverlays();
  renderSightings();
  renderMarkers();
  renderRays();
  resetResults();
  updateRunReadiness();
}

function normalizeSighting(raw, index) {
  return {
    label: raw.label || `S${index + 1}`,
    lat: Number(raw.lat),
    lon: Number(raw.lon ?? raw.lng),
    altitude: Number(raw.altitude ?? raw.altitude_m ?? 100),
    position_sigma_m: Number(raw.position_sigma_m ?? raw.position_uncertainty_m ?? 250),
    altitude_sigma_m: Number(raw.altitude_sigma_m ?? raw.altitude_uncertainty_m ?? 30),
  };
}

function renderSightings() {
  el.sightingCount.textContent = String(state.sightings.length);
  el.sightingList.innerHTML = "";
  state.sightings.forEach((sighting, index) => {
    const card = document.createElement("article");
    card.className = `sighting-card ${index === state.selectedSighting ? "selected" : ""}`;
    card.innerHTML = `
      <div class="sighting-card-header">
        <button class="sighting-chip" type="button" data-select="${index}">${index + 1}</button>
        <input class="sighting-title-input" data-index="${index}" data-field="label" value="${escapeHtml(sighting.label)}" aria-label="Sighting label" />
        <button class="icon-button danger" type="button" data-remove="${index}" title="Remove sighting">
          <i data-lucide="x"></i>
        </button>
      </div>
      <div class="sighting-grid">
        ${numberField(index, "lat", "Lat", sighting.lat, "0.000001")}
        ${numberField(index, "lon", "Lon", sighting.lon, "0.000001")}
        ${numberField(index, "altitude", "Altitude m", sighting.altitude, "1")}
        ${numberField(index, "position_sigma_m", "Position +/- m", sighting.position_sigma_m, "1")}
        ${numberField(index, "altitude_sigma_m", "Altitude +/- m", sighting.altitude_sigma_m, "1")}
      </div>
    `;
    el.sightingList.appendChild(card);
  });

  el.sightingList.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", handleSightingInput);
  });
  el.sightingList.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", () => removeSighting(Number(button.dataset.remove)));
  });
  el.sightingList.querySelectorAll("[data-select]").forEach((button) => {
    button.addEventListener("click", () => selectSighting(Number(button.dataset.select)));
  });
  refreshIcons();
}

function numberField(index, field, label, value, step) {
  return `
    <label class="field">
      <span>${label}</span>
      <input data-index="${index}" data-field="${field}" type="number" step="${step}" value="${Number(value)}" />
    </label>
  `;
}

function handleSightingInput(event) {
  const input = event.currentTarget;
  const index = Number(input.dataset.index);
  const field = input.dataset.field;
  if (field === "label") {
    state.sightings[index][field] = input.value;
  } else {
    state.sightings[index][field] = Number(input.value);
  }
  renderMarkers();
  updateRunReadiness();
  syncTerrainState();
}

function removeSighting(index) {
  state.sightings.splice(index, 1);
  if (state.selectedSighting === index) state.selectedSighting = null;
  if (state.selectedSighting > index) state.selectedSighting -= 1;
  renderSightings();
  renderMarkers();
  updateRunReadiness();
}

function selectSighting(index) {
  state.selectedSighting = index;
  state.inspectorLocked = false;
  renderSightings();
  renderMarkers();
  renderRays();
  syncTerrainState();
}

function renderMarkers() {
  if (state.overlays.markers) {
    state.map.removeLayer(state.overlays.markers);
    state.overlays.markers = null;
  }
  if (!state.layers.sightings) return;

  const group = L.layerGroup();
  state.sightings.forEach((sighting, index) => {
    if (!isFiniteSighting(sighting)) return;
    const marker = L.marker([sighting.lat, sighting.lon], {
      draggable: true,
      icon: L.divIcon({
        className: "",
        html: `<div class="map-marker ${index === state.selectedSighting ? "selected" : ""}">${index + 1}</div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      }),
    });
    marker.on("click", () => selectSighting(index));
    marker.on("dragend", (event) => {
      const latlng = event.target.getLatLng();
      state.sightings[index].lat = latlng.lat;
      state.sightings[index].lon = latlng.lng;
      renderSightings();
      renderMarkers();
    });
    marker.addTo(group);

    L.circle([sighting.lat, sighting.lon], {
      radius: Math.max(sighting.position_sigma_m, 1),
      pane: "overlayPane",
      color: index === state.selectedSighting ? "#55d6c2" : "#69a8ff",
      weight: 1,
      opacity: 0.48,
      fillOpacity: 0.045,
    }).addTo(group);
  });
  group.addTo(state.map);
  state.overlays.markers = group;
}

function renderRays() {
  if (state.overlays.rays) {
    state.map.removeLayer(state.overlays.rays);
    state.overlays.rays = null;
  }
  const mode = el.rayModeSelect.value;
  if (mode === "off" || state.selectedSighting === null || !state.sightings[state.selectedSighting]) {
    return;
  }
  const sighting = state.sightings[state.selectedSighting];
  if (!isFiniteSighting(sighting)) return;
  const group = L.layerGroup();
  const origin = [sighting.lat, sighting.lon];
  const target = state.selectedCell || state.analysis?.mostLikely;

  if (mode === "fan") {
    L.circle(origin, {
      radius: Number(el.maxRangeInput.value),
      color: "#55d6c2",
      weight: 1,
      opacity: 0.55,
      fillColor: "#55d6c2",
      fillOpacity: 0.035,
    }).addTo(group);
  }

  if ((mode === "selected" || mode === "profile") && target) {
    L.polyline([origin, [target.lat, target.lon]], {
      color: mode === "profile" ? "#f0b54d" : "#55d6c2",
      weight: mode === "profile" ? 3 : 2,
      opacity: 0.86,
      dashArray: mode === "profile" ? "8 7" : "none",
    }).addTo(group);
  }

  group.addTo(state.map);
  state.overlays.rays = group;
}

function clearRasterOverlays() {
  ["heatmap", "credible", "diagnostic"].forEach((name) => {
    if (state.overlays[name]) {
      state.map.removeLayer(state.overlays[name]);
      state.overlays[name] = null;
    }
  });
}

function updateRasterOverlays() {
  clearRasterOverlays();
  const analysis = state.analysis;
  if (!analysis) return;

  const contributionIndex = el.contributionSelect.value;
  const baseRaster = contributionIndex === ""
    ? analysis.probability
    : analysis.perSighting[Number(contributionIndex)]?.raster;

  if (state.activeDiagnostic && analysis.surfaceLayers[state.activeDiagnostic]) {
    const layer = analysis.surfaceLayers[state.activeDiagnostic];
    state.overlays.diagnostic = L.imageOverlay(
      rasterToDataUrl(layer, { kind: state.activeDiagnostic, opacity: 0.85 }),
      leafletBounds(layer),
      { pane: "diagnosticPane", opacity: 1 }
    ).addTo(state.map);
  }

  if (state.layers.probability && baseRaster) {
    state.overlays.heatmap = L.imageOverlay(
      rasterToDataUrl(baseRaster, {
        kind: contributionIndex === "" ? "probability" : "contribution",
        opacity: 1,
      }),
      leafletBounds(baseRaster),
      { pane: "heatmapPane", opacity: 1 }
    ).addTo(state.map);
  }

  if (state.layers.credible && analysis.credibleRegion?.mask) {
    const mask = analysis.credibleRegion.mask;
    state.overlays.credible = L.imageOverlay(
      rasterToDataUrl(mask, { kind: "credible", opacity: 1 }),
      leafletBounds(mask),
      { pane: "maskPane", opacity: 1 }
    ).addTo(state.map);
  }
}

function leafletBounds(raster) {
  const b = raster.bounds;
  return [
    [b.south, b.west],
    [b.north, b.east],
  ];
}

async function handleImport(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    let sightings;
    if (file.name.toLowerCase().endsWith(".csv")) {
      const response = await fetch("/api/parse-csv", { method: "POST", body: text });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "CSV import failed");
      sightings = payload.sightings;
    } else {
      const payload = JSON.parse(text);
      sightings = Array.isArray(payload) ? payload : payload.sightings;
    }
    state.sightings = sightings.map(normalizeSighting);
    state.selectedSighting = state.sightings.length ? 0 : null;
    renderSightings();
    renderMarkers();
    fitSightings();
    updateRunReadiness();
  } catch (error) {
    el.runMessage.textContent = error.message;
  } finally {
    event.target.value = "";
  }
}

async function runAnalysis() {
  if (!validSightings()) return;
  state.inspectorLocked = false;
  setRunState("Running", "Starting analysis...");
  setTimelineRunning();
  setProgressVisible(true);
  setProgressValue(0);
  el.runButton.disabled = true;
  let hadError = false;

  try {
    const startResponse = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sightings: state.sightings,
        settings: readSettings(),
      }),
    });
    const startPayload = await startResponse.json();
    if (!startResponse.ok) throw new Error(startPayload.error || "Analysis failed");

    const runId = startPayload.runId;
    updateRunProgress(startPayload);
    const payload = await waitForRunResult(runId);

    state.analysis = payload;
    state.selectedCell = payload.mostLikely;
    setRunState("Complete", `Completed in ${Math.round(payload.metadata.elapsedMs / 1000)}s.`);
    setProgressValue(1);
    updateTimeline(payload.metadata.stages);
    updateResults();
    updateContributionOptions();
    updateRasterOverlays();
    renderMarkers();
    renderRays();
    el.exportTiffButton.disabled = false;
    el.exportPngButton.disabled = false;
    state.map.fitBounds(leafletBounds(payload.probability), { padding: [36, 36] });
    syncTerrainState();
  } catch (error) {
    hadError = true;
    setRunState("Error", error.message);
    markTimelineError();
  } finally {
    el.runButton.disabled = false;
    if (!hadError) updateRunReadiness();
  }
}

async function waitForRunResult(runId) {
  while (true) {
    await wait(1200);
    const statusResponse = await fetch(`/api/runs/status?run_id=${encodeURIComponent(runId)}`);
    const statusPayload = await statusResponse.json();
    if (!statusResponse.ok) throw new Error(statusPayload.error || "Could not read run status");
    updateRunProgress(statusPayload);

    if (statusPayload.status === "error") {
      throw new Error(statusPayload.error || statusPayload.message || "Analysis failed");
    }
    if (statusPayload.ready || statusPayload.status === "complete") {
      const resultResponse = await fetch(`/api/runs/result?run_id=${encodeURIComponent(runId)}`);
      const resultPayload = await resultResponse.json();
      if (!resultResponse.ok) throw new Error(resultPayload.error || "Could not load run result");
      return resultPayload;
    }
  }
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function updateRunProgress(status) {
  setRunState(status.status === "complete" ? "Complete" : "Running", status.message || "Running...");
  setProgressValue(status.percent || 0);
  updateTimeline(status.stages || []);
}

function readSettings() {
  return {
    max_range_m: Number(el.maxRangeInput.value),
    samples_per_sighting: Number(el.samplesInput.value),
    combine: el.combineSelect.value,
    antenna_height_m: Number(el.antennaInput.value),
    prefer_gpu: el.gpuInput.checked,
    use_canopy: el.canopyInput.checked,
    use_buildings: el.buildingsInput.checked,
    use_cache: el.cacheInput.checked,
  };
}

function updateResults() {
  const analysis = state.analysis;
  if (!analysis) return;
  el.mostLikelyValue.textContent = `${formatCoord(analysis.mostLikely.lat)}, ${formatCoord(analysis.mostLikely.lon)}`;
  el.credibleAreaValue.textContent = formatArea(analysis.credibleRegion.areaKm2);
  const layers = analysis.metadata.layers || {};
  const aoi = analysis.metadata.aoi || {};
  el.metadataList.innerHTML = `
    ${metadataRow("Samples", analysis.metadata.samples)}
    ${metadataRow("Max range", formatMeters(analysis.metadata.maxRangeM))}
    ${metadataRow("Mode", analysis.metadata.gpuMode)}
    ${metadataRow("AOI", aoi.widthM ? `${formatMeters(aoi.widthM)} x ${formatMeters(aoi.heightM)}` : "-")}
    ${metadataRow("Grid cells", aoi.cells ? `${aoi.cells.toLocaleString()} @ ${formatMeters(aoi.resolutionM)}` : "-")}
    ${metadataRow("DSM", layers.dsm ? "loaded" : "missing")}
    ${metadataRow("Canopy", layers.canopyHeight ? "loaded" : "not available")}
    ${metadataRow("Buildings", layers.buildingHeight ? "loaded" : "not available")}
    ${metadataRow("Cache", analysis.metadata.cacheDir)}
  `;
  updateInspector({ lat: analysis.mostLikely.lat, lng: analysis.mostLikely.lon });
}

function metadataRow(label, value) {
  return `<div class="metadata-row"><span>${label}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function updateContributionOptions() {
  el.contributionSelect.innerHTML = `<option value="">All sightings</option>`;
  (state.analysis?.perSighting || []).forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = item.label || `S${index + 1}`;
    el.contributionSelect.appendChild(option);
  });
}

function updateInspector(latlng) {
  const analysis = state.analysis;
  if (!analysis) return;
  const lat = latlng.lat;
  const lon = latlng.lng ?? latlng.lon;
  const prob = rasterValueAt(analysis.probability, lat, lon);
  const dsm = rasterValueAt(analysis.surfaceLayers.dsm, lat, lon);
  const bare = rasterValueAt(analysis.surfaceLayers.bareEarth, lat, lon);
  const canopy = rasterValueAt(analysis.surfaceLayers.canopyHeight, lat, lon);
  const building = rasterValueAt(analysis.surfaceLayers.buildingHeight, lat, lon);
  const launch = rasterValueAt(analysis.surfaceLayers.launchWeight, lat, lon);
  const selectedContribution = state.selectedSighting === null
    ? null
    : rasterValueAt(analysis.perSighting[state.selectedSighting]?.raster, lat, lon);
  const range = state.selectedSighting === null
    ? null
    : distanceMeters(state.sightings[state.selectedSighting], { lat, lon });
  const pathState = pathLabel(selectedContribution);

  el.inspectorBody.innerHTML = `
    ${inspectorRow("Lat/Lon", `${formatCoord(lat)}, ${formatCoord(lon)}`)}
    ${inspectorRow("Probability", formatProbability(prob))}
    ${inspectorRow("Selected path", pathState)}
    ${inspectorRow("Range", formatMeters(range))}
    ${inspectorRow("DSM", formatMeters(dsm))}
    ${inspectorRow("Bare earth", formatMeters(bare))}
    ${inspectorRow("Canopy", formatMeters(canopy))}
    ${inspectorRow("Building", formatMeters(building))}
    ${inspectorRow("Launch weight", launch === null ? "-" : launch.toFixed(2))}
  `;
}

function inspectorRow(label, value) {
  return `<div class="inspector-row"><span>${label}</span><strong>${escapeHtml(String(value))}</strong></div>`;
}

function pathLabel(contribution) {
  if (contribution === null || !Number.isFinite(contribution)) return "-";
  if (contribution >= 0.65) return "clear";
  if (contribution >= 0.2) return "partial";
  return "blocked/weak";
}

function setRunState(label, message) {
  el.runStateBadge.textContent = label;
  el.runMessage.textContent = message;
}

function setProgressVisible(visible) {
  if (!el.runProgress) return;
  el.runProgress.hidden = !visible;
}

function setProgressValue(value) {
  if (!el.runProgressBar) return;
  const percent = Math.max(0, Math.min(1, Number(value) || 0));
  el.runProgressBar.style.width = `${Math.round(percent * 100)}%`;
}

function setTimelineRunning() {
  ensureTimelineRows(DEFAULT_STAGES);
  Array.from(el.timeline.children).forEach((item, index) => {
    item.className = index === 0 ? "running" : "pending";
    item.querySelector("span").textContent = DEFAULT_STAGES[index] || "Stage";
    item.querySelector("time").textContent = "-";
    item.title = "";
  });
}

function updateTimeline(stages) {
  const names = [...DEFAULT_STAGES];
  stages.forEach((stage) => {
    if (!names.includes(stage.name)) names.push(stage.name);
  });
  ensureTimelineRows(names);
  const items = Array.from(el.timeline.children);
  stages.forEach((stage) => {
    const index = names.indexOf(stage.name);
    if (!items[index]) return;
    items[index].className = stage.status || "complete";
    items[index].querySelector("span").textContent = stage.name;
    items[index].querySelector("time").textContent = formatDuration(stage.elapsedMs || 0);
    items[index].title = stage.message || "";
  });
}

function ensureTimelineRows(names) {
  el.timeline.innerHTML = "";
  names.forEach((name) => {
    const item = document.createElement("li");
    item.className = "pending";
    item.innerHTML = `<span>${escapeHtml(name)}</span><time>-</time>`;
    el.timeline.appendChild(item);
  });
}

function formatDuration(ms) {
  const seconds = Math.max(0, Math.round(ms / 1000));
  if (seconds < 1) return `${ms}ms`;
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${minutes}m ${String(rest).padStart(2, "0")}s`;
}

function markTimelineError() {
  const running = el.timeline.querySelector(".running") || el.timeline.firstElementChild;
  if (running) {
    running.className = "running";
    running.querySelector("time").textContent = "error";
  }
}

function resetResults() {
  el.mostLikelyValue.textContent = "-";
  el.credibleAreaValue.textContent = "-";
  el.runStateBadge.textContent = "Idle";
  el.exportTiffButton.disabled = true;
  el.exportPngButton.disabled = true;
  el.inspectorBody.innerHTML = `<p class="status-message">Hover or click a probability cell.</p>`;
  el.metadataList.innerHTML = "";
  setProgressVisible(false);
  setProgressValue(0);
  setTimelineRunning();
  updateContributionOptions();
}

function updateRunReadiness() {
  el.runButton.disabled = !validSightings();
  if (!state.sightings.length) {
    el.runMessage.textContent = "Add a sighting or import a file.";
  } else if (!validSightings()) {
    el.runMessage.textContent = "Check sighting coordinates and altitude values.";
  } else if (!state.analysis) {
    el.runMessage.textContent = statusText();
  }
}

function statusText() {
  return `${state.sightings.length} sighting${state.sightings.length === 1 ? "" : "s"} ready.`;
}

function fitSightings() {
  const valid = state.sightings.filter(isFiniteSighting);
  if (!valid.length) return;
  const bounds = L.latLngBounds(valid.map((s) => [s.lat, s.lon]));
  state.map.fitBounds(bounds.pad(0.4), { maxZoom: 14 });
}

function validSightings() {
  return state.sightings.length > 0 && state.sightings.every(isFiniteSighting);
}

function isFiniteSighting(sighting) {
  return [
    sighting.lat,
    sighting.lon,
    sighting.altitude,
    sighting.position_sigma_m,
    sighting.altitude_sigma_m,
  ].every(Number.isFinite);
}

function switchView(view) {
  state.activeView = view;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".stage-view").forEach((stage) => {
    stage.classList.toggle("active", stage.id === `${view}View`);
  });
  if (view === "terrain") {
    loadTerrainForCurrentView();
  } else {
    setTimeout(() => state.map.invalidateSize(), 60);
  }
}

function loadTerrainForCurrentView() {
  const bounds = state.map.getBounds();
  state.terrain.load({
    bounds: {
      west: bounds.getWest(),
      south: bounds.getSouth(),
      east: bounds.getEast(),
      north: bounds.getNorth(),
    },
    analysis: state.analysis,
    sightings: state.sightings,
    selectedSighting: state.selectedSighting,
    selectedCell: state.selectedCell || state.analysis?.mostLikely || null,
    rayMode: el.rayModeSelect.value,
    maxRangeM: Number(el.maxRangeInput.value),
    showProbability: state.layers.probability,
    showBuildings: el.terrainBuildingsInput.checked,
    showCanopy: el.terrainCanopyInput.checked,
    verticalScale: Number(el.verticalScaleInput.value),
  });
}

function syncTerrainState() {
  if (!state.terrain) return;
  state.terrain.updateAnalysis({
    analysis: state.analysis,
    sightings: state.sightings,
    selectedSighting: state.selectedSighting,
    selectedCell: state.selectedCell || state.analysis?.mostLikely || null,
    rayMode: el.rayModeSelect.value,
    maxRangeM: Number(el.maxRangeInput.value),
    showProbability: state.layers.probability,
    showBuildings: el.terrainBuildingsInput.checked,
    showCanopy: el.terrainCanopyInput.checked,
  });
}

function exportGeotiff() {
  if (!state.analysis?.runId) return;
  window.location.href = `/api/export/geotiff?run_id=${encodeURIComponent(state.analysis.runId)}`;
}

async function exportPngPreview() {
  if (!state.analysis) return;
  el.exportPngButton.disabled = true;
  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 800;
  const ctx = canvas.getContext("2d");
  const bounds = state.map.getBounds();
  const viewBounds = {
    west: bounds.getWest(),
    south: bounds.getSouth(),
    east: bounds.getEast(),
    north: bounds.getNorth(),
  };
  const zoom = Math.round(state.map.getZoom());

  try {
    await drawOsmBasemap(ctx, viewBounds, zoom, canvas, state.tileSources.map || OSM_TEMPLATE);
    drawVisiblePreviewLayers(ctx, viewBounds, zoom, canvas);
    downloadCanvas(canvas, `launchpoint_preview_${state.analysis.runId.slice(0, 8)}.png`);
  } catch {
    ctx.fillStyle = "#071114";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawVisiblePreviewLayers(ctx, viewBounds, zoom, canvas);
    downloadCanvas(canvas, `launchpoint_preview_${state.analysis.runId.slice(0, 8)}.png`);
  } finally {
    el.exportPngButton.disabled = false;
  }
}

function drawVisiblePreviewLayers(ctx, viewBounds, zoom, canvas) {
  const analysis = state.analysis;
  if (state.activeDiagnostic && analysis.surfaceLayers[state.activeDiagnostic]) {
    drawRasterInView(ctx, analysis.surfaceLayers[state.activeDiagnostic], viewBounds, zoom, canvas, {
      kind: state.activeDiagnostic,
      opacity: 0.85,
    });
  }
  const contributionIndex = el.contributionSelect.value;
  const baseRaster = contributionIndex === ""
    ? analysis.probability
    : analysis.perSighting[Number(contributionIndex)]?.raster;
  if (state.layers.probability && baseRaster) {
    drawRasterInView(ctx, baseRaster, viewBounds, zoom, canvas, {
      kind: contributionIndex === "" ? "probability" : "contribution",
      opacity: 1,
    });
  }
  if (state.layers.credible) {
    drawRasterInView(ctx, analysis.credibleRegion.mask, viewBounds, zoom, canvas, {
      kind: "credible",
      opacity: 1,
    });
  }
  if (state.layers.sightings) {
    drawSightings(ctx, state.sightings, viewBounds, zoom, canvas, state.selectedSighting);
  }
  drawMostLikely(ctx, analysis.mostLikely, viewBounds, zoom, canvas);
}

function downloadCanvas(canvas, filename) {
  try {
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }, "image/png");
  } catch {
    el.runMessage.textContent = "PNG preview could not be exported from the current tile source.";
  }
}

function distanceMeters(a, b) {
  const earth = 6371008.8;
  const p1 = (a.lat * Math.PI) / 180;
  const p2 = (b.lat * Math.PI) / 180;
  const dp = ((b.lat - a.lat) * Math.PI) / 180;
  const dl = ((b.lon - a.lon) * Math.PI) / 180;
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * earth * Math.asin(Math.min(1, Math.sqrt(h)));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}
