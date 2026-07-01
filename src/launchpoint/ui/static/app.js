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
import { castViewshedFan, observerElevation } from "./raycast.js";

const OSM_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const DEFAULT_STAGES = ["Prepare AOI", "Fetch surfaces", "Run fusion", "Serialize result", "Export ready"];
const REVERSE_STAGES = [
  "Prepare AOI",
  "Fetch surfaces",
  "Search candidates",
  "Refine hot patches",
  "Serialize result",
  "Export ready",
];

const state = {
  map: null,
  terrain: null,
  mode: "forward",
  sightings: [],
  selectedSighting: null,
  selectedCell: null,
  inspectorLocked: false,
  addingPin: false,
  zone: null,
  addingZone: null,
  analysis: null,
  overlays: {
    heatmap: null,
    credible: null,
    diagnostic: null,
    markers: null,
    rays: null,
    zone: null,
    zonePreview: null,
  },
  diagnosticRequestId: 0,
  layers: {
    probability: true,
    credible: true,
    sightings: true,
  },
  activeView: "map",
  activeDiagnostic: "",
  diagnosticOpacity: 0.85,
  tileSources: {
    map: OSM_TEMPLATE,
    terrain: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
  },
  reverseDefaults: {
    flightAltitudeM: 100.0,
    radiusM: 300.0,
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
  renderZoneFields();
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
    "sightingsSection",
    "zoneSection",
    "drawZoneButton",
    "clearZoneButton",
    "zoneFields",
    "zoneMessage",
    "maxRangeInput",
    "forwardAdvancedFields",
    "reverseAdvancedFields",
    "samplesInput",
    "combineSelect",
    "coverageThresholdInput",
    "candidateCoarseStrideInput",
    "candidateFineStrideInput",
    "antennaInput",
    "gpuInput",
    "canopyInput",
    "canopySourceSelect",
    "resolutionField",
    "resolutionSelect",
    "resolutionHint",
    "buildingsInput",
    "cacheInput",
    "overrideSizeInput",
    "overrideSizeHint",
    "runButton",
    "runProgress",
    "runProgressBar",
    "runMessage",
    "rayModeSelect",
    "metric1Label",
    "mostLikelyValue",
    "credibleMetric",
    "credibleAreaValue",
    "minVisibilityMetric",
    "minVisibilityValue",
    "fracCoveredMetric",
    "fracCoveredValue",
    "runStateBadge",
    "limitationsNote",
    "exportTiffButton",
    "exportPngButton",
    "inspectorBody",
    "timeline",
    "contributionSelect",
    "diagnosticOpacityInput",
    "diagnosticOpacityValue",
    "metadataList",
    "rightRail",
    "collapseRightButton",
    "terrainHost",
    "terrainStatus",
    "verticalScaleInput",
    "terrainBuildingsInput",
    "terrainCanopyInput",
    "reloadTerrainButton",
    "heatmapToggle",
    "credibleToggle",
    "sightingsToggle",
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
  state.map.getPane("diagnosticPane").style.zIndex = 370;
  state.map.getPane("heatmapPane").style.zIndex = 350;
  state.map.getPane("maskPane").style.zIndex = 360;

  state.map.on("click", handleMapClick);
  state.map.on("mousemove", handleMapHover);
  state.map.on("moveend zoomend", () => {
    if (state.activeDiagnostic === "canopyHeight") updateRasterOverlays();
  });
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
  el.canopySourceSelect.addEventListener("change", syncResolutionOptions);
  el.canopyInput.addEventListener("change", syncResolutionOptions);
  el.overrideSizeInput.addEventListener("change", () => {
    el.overrideSizeHint.hidden = !el.overrideSizeInput.checked;
  });
  el.diagnosticOpacityInput.addEventListener("input", () => {
    state.diagnosticOpacity = readDiagnosticOpacity();
    updateDiagnosticOpacityLabel();
    updateRasterOverlays();
  });
  el.collapseRightButton.addEventListener("click", () => {
    el.rightRail.classList.toggle("collapsed");
  });
  el.verticalScaleInput.addEventListener("input", () => {
    state.terrain.setVerticalScale(Number(el.verticalScaleInput.value));
  });
  el.terrainBuildingsInput.addEventListener("change", syncTerrainState);
  el.terrainCanopyInput.addEventListener("change", syncTerrainState);
  el.reloadTerrainButton.addEventListener("click", loadTerrainForCurrentView);

  document.querySelectorAll(".topbar .segment").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  document.querySelectorAll(".mode-switch .segment").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  el.drawZoneButton.addEventListener("click", startDrawZone);
  el.clearZoneButton.addEventListener("click", clearZone);
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
  state.diagnosticOpacity = readDiagnosticOpacity();
  updateDiagnosticOpacityLabel();
  syncResolutionOptions();
}

// Sub-10 m model resolutions only make sense with the Meta 1 m canopy source
// (the backend rejects them otherwise), so disable those options and snap the
// selection back to 10 m when the user isn't on Meta.
function syncResolutionOptions() {
  if (!el.resolutionSelect) return;
  const fineAllowed = el.canopyInput.checked && el.canopySourceSelect.value === "meta";
  let snapped = false;
  for (const option of el.resolutionSelect.options) {
    if (option.dataset.requiresMeta === undefined) continue;
    option.disabled = !fineAllowed;
    if (!fineAllowed && option.selected) {
      el.resolutionSelect.value = "10";
      snapped = true;
    }
  }
  if (el.resolutionHint) {
    el.resolutionHint.classList.toggle("warn", snapped);
    el.resolutionHint.textContent = fineAllowed
      ? "Sub-10 m feeds the model near-native Meta 1 m detail."
      : "Sub-10 m needs the Meta 1 m canopy source.";
  }
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
    if (payload.reverseSettings) {
      const rs = payload.reverseSettings;
      state.reverseDefaults.flightAltitudeM = rs.flightAltitudeM ?? state.reverseDefaults.flightAltitudeM;
      state.reverseDefaults.radiusM = rs.radiusM ?? state.reverseDefaults.radiusM;
      el.coverageThresholdInput.value = rs.coverageThreshold ?? el.coverageThresholdInput.value;
      el.candidateCoarseStrideInput.value = rs.candidateCoarseStrideM ?? el.candidateCoarseStrideInput.value;
      el.candidateFineStrideInput.value = rs.candidateFineStrideM ?? el.candidateFineStrideInput.value;
    }
  } catch {
    // Defaults are already embedded for offline UI startup.
  }
}

function handleMapClick(event) {
  if (state.mode === "reverse" && state.addingZone) {
    if (state.addingZone === "center") {
      state.zone = normalizeZone({
        center_lat: event.latlng.lat,
        center_lon: event.latlng.lng,
        radius_m: state.zone?.radius_m ?? state.reverseDefaults.radiusM,
        flight_altitude_m: state.zone?.flight_altitude_m ?? state.reverseDefaults.flightAltitudeM,
      });
      state.addingZone = "radius";
      el.zoneMessage.textContent = "Move the mouse to size the zone, then click to confirm.";
      renderZoneFields();
      renderZoneOverlay();
      return;
    }
    // state.addingZone === "radius"
    state.zone.radius_m = Math.max(
      distanceMeters(
        { lat: state.zone.center_lat, lon: state.zone.center_lon },
        { lat: event.latlng.lat, lon: event.latlng.lng }
      ),
      10
    );
    state.addingZone = null;
    el.drawZoneButton.classList.remove("active");
    renderZoneFields();
    renderZoneOverlay();
    updateRunReadiness();
    return;
  }
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
  if (state.mode === "reverse" && state.addingZone === "radius" && state.zone) {
    const radius = distanceMeters(
      { lat: state.zone.center_lat, lon: state.zone.center_lon },
      { lat: event.latlng.lat, lon: event.latlng.lng }
    );
    renderZoneOverlay(radius);
    return;
  }
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

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  state.addingPin = false;
  state.addingZone = null;
  state.analysis = null;
  state.selectedCell = null;
  state.inspectorLocked = false;
  el.addPinButton.classList.remove("active");
  el.drawZoneButton.classList.remove("active");

  document.querySelectorAll(".mode-switch .segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  el.sightingsSection.hidden = mode !== "forward";
  el.zoneSection.hidden = mode !== "reverse";
  el.forwardAdvancedFields.hidden = mode !== "forward";
  el.reverseAdvancedFields.hidden = mode !== "reverse";
  el.resolutionField.hidden = mode !== "forward";
  el.credibleToggle.hidden = mode !== "forward";
  el.credibleMetric.hidden = mode !== "forward";
  el.minVisibilityMetric.hidden = mode !== "reverse";
  el.fracCoveredMetric.hidden = mode !== "reverse";
  el.metric1Label.textContent = mode === "reverse" ? "Best launch point" : "Most likely";
  el.heatmapToggle.textContent = mode === "reverse" ? "Coverage" : "Heatmap";
  el.sightingsToggle.textContent = mode === "reverse" ? "Zone" : "Sightings";
  el.limitationsNote.textContent = mode === "reverse"
    ? "Score is the weakest-covered point in the flight zone. A low score means part of the zone is out of view from every nearby candidate."
    : "Probability is a likelihood surface. Sparse sightings and weak terrain data widen the credible region.";

  clearRasterOverlays();
  renderMarkers();
  renderZoneOverlay();
  renderRays();
  resetResults();
  updateRunReadiness();
  syncTerrainState();
}

function normalizeZone(raw) {
  return {
    label: raw.label || "Flight zone",
    center_lat: Number(raw.center_lat),
    center_lon: Number(raw.center_lon ?? raw.center_lng),
    radius_m: Number(raw.radius_m),
    flight_altitude_m: Number(raw.flight_altitude_m),
  };
}

function isFiniteZone(zone) {
  return Boolean(zone) && [zone.center_lat, zone.center_lon, zone.radius_m, zone.flight_altitude_m]
    .every(Number.isFinite) && zone.radius_m > 0;
}

function startDrawZone() {
  state.addingZone = state.addingZone ? null : "center";
  el.drawZoneButton.classList.toggle("active", Boolean(state.addingZone));
  el.zoneMessage.textContent = state.addingZone
    ? "Click the map to place the zone center."
    : zoneStatusText();
}

function clearZone() {
  state.zone = null;
  state.addingZone = null;
  state.analysis = null;
  el.drawZoneButton.classList.remove("active");
  clearRasterOverlays();
  renderZoneFields();
  renderZoneOverlay();
  resetResults();
  updateRunReadiness();
}

function renderZoneFields() {
  const zone = state.zone || {
    center_lat: "", center_lon: "",
    radius_m: state.reverseDefaults.radiusM,
    flight_altitude_m: state.reverseDefaults.flightAltitudeM,
  };
  el.zoneFields.innerHTML = `
    <div class="sighting-grid">
      ${numberField(0, "center_lat", "Lat", zone.center_lat, "0.000001")}
      ${numberField(0, "center_lon", "Lon", zone.center_lon, "0.000001")}
      ${numberField(0, "radius_m", "Radius m", zone.radius_m, "10")}
      ${numberField(0, "flight_altitude_m", "Flight alt m (AGL)", zone.flight_altitude_m, "1")}
    </div>
  `;
  el.zoneFields.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", handleZoneInput);
  });
}

function handleZoneInput(event) {
  const field = event.currentTarget.dataset.field;
  const value = Number(event.currentTarget.value);
  const base = state.zone || {
    center_lat: NaN, center_lon: NaN,
    radius_m: state.reverseDefaults.radiusM,
    flight_altitude_m: state.reverseDefaults.flightAltitudeM,
  };
  state.zone = normalizeZone({ ...base, [field]: value });
  renderZoneOverlay();
  updateRunReadiness();
}

function renderZoneOverlay(previewRadius = null) {
  if (state.overlays.zone) {
    state.map.removeLayer(state.overlays.zone);
    state.overlays.zone = null;
  }
  if (state.overlays.zonePreview) {
    state.map.removeLayer(state.overlays.zonePreview);
    state.overlays.zonePreview = null;
  }
  if (state.mode !== "reverse" || !state.zone || !isFiniteZone(state.zone)) return;

  const group = L.layerGroup();
  L.circle([state.zone.center_lat, state.zone.center_lon], {
    radius: state.zone.radius_m,
    pane: "overlayPane",
    color: "#55d6c2",
    weight: 2,
    opacity: 0.75,
    fillOpacity: 0.08,
  }).addTo(group);

  const marker = L.marker([state.zone.center_lat, state.zone.center_lon], {
    draggable: true,
    icon: L.divIcon({
      className: "",
      html: `<div class="map-marker">Z</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
    }),
  });
  marker.on("dragend", (event) => {
    const latlng = event.target.getLatLng();
    state.zone.center_lat = latlng.lat;
    state.zone.center_lon = latlng.lng;
    renderZoneFields();
    renderZoneOverlay();
  });
  marker.addTo(group);
  group.addTo(state.map);
  state.overlays.zone = group;

  if (previewRadius !== null) {
    state.overlays.zonePreview = L.circle([state.zone.center_lat, state.zone.center_lon], {
      radius: previewRadius,
      pane: "overlayPane",
      color: "#55d6c2",
      weight: 1,
      opacity: 0.4,
      fill: false,
      dashArray: "4 6",
    }).addTo(state.map);
  }
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
  if (state.mode !== "forward" || !state.layers.sightings) return;

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

const RAY_VISIBLE_STYLE = { color: "#55d6c2", weight: 1, opacity: 0.5 };
const RAY_OCCLUDED_STYLE = { color: "#f0655a", weight: 1, opacity: 0.16 };

function renderRays() {
  if (state.overlays.rays) {
    state.map.removeLayer(state.overlays.rays);
    state.overlays.rays = null;
  }
  const rayMode = el.rayModeSelect.value;
  if (
    state.mode !== "forward" ||
    rayMode === "off" ||
    state.selectedSighting === null ||
    !state.sightings[state.selectedSighting]
  ) {
    return;
  }
  const sighting = state.sightings[state.selectedSighting];
  if (!isFiniteSighting(sighting)) return;

  const group = L.layerGroup();
  const origin = { lat: sighting.lat, lon: sighting.lon };
  const maxRange = Number(el.maxRangeInput.value) || 12000;

  // Max-range boundary so the fan reads as bounded even where it's all visible.
  L.circle([origin.lat, origin.lon], {
    radius: maxRange,
    color: "#55d6c2",
    weight: 1,
    opacity: 0.4,
    fill: false,
    dashArray: "4 6",
  }).addTo(group);

  const dsm = state.analysis?.surfaceLayers?.dsm;
  if (dsm) {
    const ground = state.analysis?.surfaceLayers?.bareEarth || dsm;
    const observerZ = observerElevation(dsm, origin.lat, origin.lon, sighting.altitude || 0);
    const targetHeight = Number(el.antennaInput.value) || 1.5;
    const fan = castViewshedFan({
      dsm,
      ground,
      origin,
      observerZ,
      targetHeight,
      maxRange,
      rayCount: 96,
    });
    drawRayFan(group, origin, fan);
  } else {
    // No analysis yet: preview the cast geometry as faint uniform spokes.
    drawPreviewFan(group, origin, maxRange);
  }

  group.addTo(state.map);
  state.overlays.rays = group;
}

// Draw each ray as polylines split into visible/occluded runs.
function drawRayFan(group, origin, fan) {
  const start = [origin.lat, origin.lon];
  fan.rays.forEach((ray) => {
    if (!ray.points.length) return;
    let runVisible = ray.points[0].visible;
    let coords = [start];
    ray.points.forEach((p) => {
      if (p.visible !== runVisible) {
        L.polyline(coords, runVisible ? RAY_VISIBLE_STYLE : RAY_OCCLUDED_STYLE).addTo(group);
        coords = [coords[coords.length - 1]]; // bridge the gap, no hole
        runVisible = p.visible;
      }
      coords.push([p.lat, p.lon]);
    });
    if (coords.length > 1) {
      L.polyline(coords, runVisible ? RAY_VISIBLE_STYLE : RAY_OCCLUDED_STYLE).addTo(group);
    }
  });
}

function drawPreviewFan(group, origin, maxRange) {
  const lat0 = origin.lat;
  const lon0 = origin.lon;
  const mPerDegLat = 111320;
  const mPerDegLon = 111320 * Math.cos((lat0 * Math.PI) / 180);
  for (let i = 0; i < 48; i += 1) {
    const az = (i / 48) * Math.PI * 2;
    const lat = lat0 + (maxRange * Math.cos(az)) / mPerDegLat;
    const lon = lon0 + (maxRange * Math.sin(az)) / mPerDegLon;
    L.polyline([[lat0, lon0], [lat, lon]], {
      color: "#55d6c2",
      weight: 1,
      opacity: 0.12,
    }).addTo(group);
  }
}

function clearRasterOverlays() {
  state.diagnosticRequestId += 1;
  ["heatmap", "credible", "diagnostic"].forEach((name) => {
    if (state.overlays[name]) {
      state.map.removeLayer(state.overlays[name]);
      state.overlays[name] = null;
    }
  });
}

async function updateRasterOverlays() {
  clearRasterOverlays();
  const analysis = state.analysis;
  if (!analysis) return;
  const requestId = state.diagnosticRequestId;

  const isReverse = state.mode === "reverse";
  const contributionIndex = el.contributionSelect.value;
  const baseRaster = isReverse
    ? analysis.coverage
    : contributionIndex === ""
      ? analysis.probability
      : analysis.perSighting[Number(contributionIndex)]?.raster;

  const fallbackDiagnostic = state.activeDiagnostic
    ? analysis.surfaceLayers[state.activeDiagnostic]
    : null;
  if (fallbackDiagnostic) renderDiagnosticOverlay(fallbackDiagnostic);

  if (state.layers.probability && baseRaster) {
    state.overlays.heatmap = L.imageOverlay(
      rasterToDataUrl(baseRaster, {
        kind: isReverse ? "coverage" : contributionIndex === "" ? "probability" : "contribution",
        opacity: 1,
      }),
      leafletBounds(baseRaster),
      { pane: "heatmapPane", opacity: 1 }
    ).addTo(state.map);
  }

  if (!isReverse && state.layers.credible && analysis.credibleRegion?.mask) {
    const mask = analysis.credibleRegion.mask;
    state.overlays.credible = L.imageOverlay(
      rasterToDataUrl(mask, { kind: "credible", opacity: 1 }),
      leafletBounds(mask),
      { pane: "maskPane", opacity: 1 }
    ).addTo(state.map);
  }

  const diagnostic = await loadVisibleDiagnosticLayer();
  if (requestId !== state.diagnosticRequestId || !diagnostic) return;
  if (state.overlays.diagnostic) {
    state.map.removeLayer(state.overlays.diagnostic);
    state.overlays.diagnostic = null;
  }
  renderDiagnosticOverlay(diagnostic);
}

function renderDiagnosticOverlay(layer) {
  state.overlays.diagnostic = L.imageOverlay(
    rasterToDataUrl(layer, {
      kind: state.activeDiagnostic,
      opacity: state.diagnosticOpacity,
      solid: true,
    }),
    leafletBounds(layer),
    { pane: "diagnosticPane", opacity: 1 }
  ).addTo(state.map);
}

function leafletBounds(raster) {
  const b = raster.bounds;
  return [
    [b.south, b.west],
    [b.north, b.east],
  ];
}

async function loadVisibleDiagnosticLayer(viewBounds = null) {
  const analysis = state.analysis;
  if (!analysis?.runId || state.activeDiagnostic !== "canopyHeight") return null;
  if (!analysis.surfaceLayers.canopyHeight) return null;
  const bounds = viewBounds || mapViewBounds();
  const params = new URLSearchParams({
    run_id: analysis.runId,
    layer: "canopyHeight",
    west: String(bounds.west),
    south: String(bounds.south),
    east: String(bounds.east),
    north: String(bounds.north),
  });
  try {
    const response = await fetch(`/api/runs/diagnostic-layer?${params.toString()}`);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function mapViewBounds() {
  const bounds = state.map.getBounds();
  return {
    west: bounds.getWest(),
    south: bounds.getSouth(),
    east: bounds.getEast(),
    north: bounds.getNorth(),
  };
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
  if (!canRun()) return;
  state.inspectorLocked = false;
  setRunState("Running", "Starting analysis...");
  setTimelineRunning();
  setProgressVisible(true);
  setProgressValue(0);
  el.runButton.disabled = true;
  let hadError = false;

  const isReverse = state.mode === "reverse";
  const endpoint = isReverse ? "/api/runs/launch" : "/api/runs";
  const body = isReverse
    ? { zone: state.zone, settings: readSettings() }
    : { sightings: state.sightings, settings: readSettings() };

  try {
    const startResponse = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const startPayload = await startResponse.json();
    if (!startResponse.ok) throw new Error(startPayload.error || "Analysis failed");

    const runId = startPayload.runId;
    updateRunProgress(startPayload);
    const payload = await waitForRunResult(runId);

    state.analysis = payload;
    state.selectedCell = isReverse ? payload.bestLaunchPoint : payload.mostLikely;
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
    const boundsRaster = isReverse ? payload.coverage : payload.probability;
    state.map.fitBounds(leafletBounds(boundsRaster), { padding: [36, 36] });
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
    canopy_source: el.canopySourceSelect.value,
    analysis_resolution_m: Number(el.resolutionSelect.value),
    use_buildings: el.buildingsInput.checked,
    use_cache: el.cacheInput.checked,
    override_cell_limit: el.overrideSizeInput.checked,
    coverage_threshold: Number(el.coverageThresholdInput.value),
    candidate_coarse_stride_m: Number(el.candidateCoarseStrideInput.value),
    candidate_fine_stride_m: Number(el.candidateFineStrideInput.value),
  };
}

function updateResults() {
  const analysis = state.analysis;
  if (!analysis) return;
  if (state.mode === "reverse") {
    updateReverseResults(analysis);
    return;
  }
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
    ${metadataRow("Canopy", layers.canopyHeight ? `loaded (${analysis.metadata.canopySource === "meta" ? "Meta 1 m" : "ETH 10 m"})` : "not available")}
    ${metadataRow("Buildings", layers.buildingHeight ? "loaded" : "not available")}
    ${metadataRow("Cache", analysis.metadata.cacheDir)}
  `;
  updateInspector({ lat: analysis.mostLikely.lat, lng: analysis.mostLikely.lon });
}

function updateReverseResults(analysis) {
  const best = analysis.bestLaunchPoint;
  el.mostLikelyValue.textContent = `${formatCoord(best.lat)}, ${formatCoord(best.lon)}`;
  el.minVisibilityValue.textContent = formatProbability(best.minVisibility);
  el.fracCoveredValue.textContent = formatProbability(best.fracCovered);
  const layers = analysis.metadata.layers || {};
  const aoi = analysis.metadata.aoi || {};
  el.metadataList.innerHTML = `
    ${metadataRow("Flight altitude", formatMeters(analysis.metadata.flightAltitudeM))}
    ${metadataRow("Zone radius", formatMeters(analysis.metadata.zoneRadiusM))}
    ${metadataRow("Max range", formatMeters(analysis.metadata.maxRangeM))}
    ${metadataRow("Mode", analysis.metadata.gpuMode)}
    ${metadataRow("Candidates evaluated", analysis.metadata.candidatesEvaluated?.toLocaleString?.() ?? "-")}
    ${metadataRow("Candidates skipped (out of range)", analysis.metadata.candidatesPrefiltered?.toLocaleString?.() ?? "-")}
    ${metadataRow("AOI", aoi.widthM ? `${formatMeters(aoi.widthM)} x ${formatMeters(aoi.heightM)}` : "-")}
    ${metadataRow("DSM", layers.dsm ? "loaded" : "missing")}
    ${metadataRow("Canopy", layers.canopyHeight ? `loaded (${analysis.metadata.canopySource === "meta" ? "Meta 1 m" : "ETH 10 m"})` : "not available")}
    ${metadataRow("Buildings", layers.buildingHeight ? "loaded" : "not available")}
    ${metadataRow("Cache", analysis.metadata.cacheDir)}
  `;
  updateInspector({ lat: best.lat, lng: best.lon });
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
  if (state.mode === "reverse") {
    updateReverseInspector(analysis, lat, lon);
    return;
  }
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

function updateReverseInspector(analysis, lat, lon) {
  const score = rasterValueAt(analysis.coverage, lat, lon);
  const minVis = rasterValueAt(analysis.minVisibility, lat, lon);
  const frac = rasterValueAt(analysis.fracCovered, lat, lon);
  const dsm = rasterValueAt(analysis.surfaceLayers.dsm, lat, lon);
  const bare = rasterValueAt(analysis.surfaceLayers.bareEarth, lat, lon);
  const canopy = rasterValueAt(analysis.surfaceLayers.canopyHeight, lat, lon);
  const building = rasterValueAt(analysis.surfaceLayers.buildingHeight, lat, lon);
  const launch = rasterValueAt(analysis.surfaceLayers.launchWeight, lat, lon);

  el.inspectorBody.innerHTML = `
    ${inspectorRow("Lat/Lon", `${formatCoord(lat)}, ${formatCoord(lon)}`)}
    ${inspectorRow("Launch coverage score", formatProbability(score))}
    ${inspectorRow("Min visibility", formatProbability(minVis))}
    ${inspectorRow("Zone covered", formatProbability(frac))}
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

function currentStages() {
  return state.mode === "reverse" ? REVERSE_STAGES : DEFAULT_STAGES;
}

function setTimelineRunning() {
  const stages = currentStages();
  ensureTimelineRows(stages);
  Array.from(el.timeline.children).forEach((item, index) => {
    item.className = index === 0 ? "running" : "pending";
    item.querySelector("span").textContent = stages[index] || "Stage";
    item.querySelector("time").textContent = "-";
    item.title = "";
  });
}

function updateTimeline(stages) {
  const names = [...currentStages()];
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
  el.minVisibilityValue.textContent = "-";
  el.fracCoveredValue.textContent = "-";
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

function canRun() {
  return state.mode === "reverse" ? isFiniteZone(state.zone) : validSightings();
}

function updateRunReadiness() {
  el.runButton.disabled = !canRun();
  if (state.mode === "reverse") {
    if (!state.zone) {
      el.runMessage.textContent = "Draw a flight zone or enter coordinates.";
    } else if (!isFiniteZone(state.zone)) {
      el.runMessage.textContent = "Check zone coordinates, radius and altitude values.";
    } else if (!state.analysis) {
      el.runMessage.textContent = zoneStatusText();
    }
    return;
  }
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

function zoneStatusText() {
  return "Flight zone ready.";
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
  document.querySelectorAll(".topbar .segment").forEach((button) => {
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
  const viewBounds = mapViewBounds();
  const zoom = Math.round(state.map.getZoom());

  try {
    await drawOsmBasemap(ctx, viewBounds, zoom, canvas, state.tileSources.map || OSM_TEMPLATE);
    await drawVisiblePreviewLayers(ctx, viewBounds, zoom, canvas);
    downloadCanvas(canvas, `launchpoint_preview_${state.analysis.runId.slice(0, 8)}.png`);
  } catch {
    ctx.fillStyle = "#071114";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    await drawVisiblePreviewLayers(ctx, viewBounds, zoom, canvas);
    downloadCanvas(canvas, `launchpoint_preview_${state.analysis.runId.slice(0, 8)}.png`);
  } finally {
    el.exportPngButton.disabled = false;
  }
}

async function drawVisiblePreviewLayers(ctx, viewBounds, zoom, canvas) {
  const analysis = state.analysis;
  const isReverse = state.mode === "reverse";
  const contributionIndex = el.contributionSelect.value;
  const baseRaster = isReverse
    ? analysis.coverage
    : contributionIndex === ""
      ? analysis.probability
      : analysis.perSighting[Number(contributionIndex)]?.raster;
  if (state.layers.probability && baseRaster) {
    drawRasterInView(ctx, baseRaster, viewBounds, zoom, canvas, {
      kind: isReverse ? "coverage" : contributionIndex === "" ? "probability" : "contribution",
      opacity: 1,
    });
  }
  if (!isReverse && state.layers.credible && analysis.credibleRegion?.mask) {
    drawRasterInView(ctx, analysis.credibleRegion.mask, viewBounds, zoom, canvas, {
      kind: "credible",
      opacity: 1,
    });
  }
  if (state.activeDiagnostic && analysis.surfaceLayers[state.activeDiagnostic]) {
    const diagnosticLayer = await loadVisibleDiagnosticLayer(viewBounds)
      || analysis.surfaceLayers[state.activeDiagnostic];
    drawRasterInView(ctx, diagnosticLayer, viewBounds, zoom, canvas, {
      kind: state.activeDiagnostic,
      opacity: state.diagnosticOpacity,
      solid: true,
    });
  }
  if (!isReverse && state.layers.sightings) {
    drawSightings(ctx, state.sightings, viewBounds, zoom, canvas, state.selectedSighting);
  }
  const bestPoint = isReverse ? analysis.bestLaunchPoint : analysis.mostLikely;
  drawMostLikely(ctx, bestPoint, viewBounds, zoom, canvas);
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

function readDiagnosticOpacity() {
  return clamp(Number(el.diagnosticOpacityInput?.value) || 0.85, 0.1, 1);
}

function updateDiagnosticOpacityLabel() {
  if (!el.diagnosticOpacityValue) return;
  el.diagnosticOpacityValue.textContent = `${Math.round(state.diagnosticOpacity * 100)}%`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
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
