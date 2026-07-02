import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";
import { clamp, heatColor, rasterValueAt } from "./rendering.js";
import { castViewshedFan, observerElevation } from "./raycast.js";

const TILE_SIZE = 256;
const SAMPLE_STEP = 8;
const METERS_PER_DEG_LAT = 111320;
const ANALYSIS_MESH_TARGET = 96;

export class TerrainScene {
  constructor(host, options = {}) {
    this.host = host;
    this.statusEl = options.statusEl || null;
    this.terrainTemplate = options.terrainTemplate;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x071114);
    this.camera = new THREE.PerspectiveCamera(48, 1, 1, 1000000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.host.appendChild(this.renderer.domElement);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.meshGroup = new THREE.Group();
    this.extrusionGroup = new THREE.Group();
    this.pinGroup = new THREE.Group();
    this.rayGroup = new THREE.Group();
    this.coverageGroup = new THREE.Group();
    this.root.add(this.meshGroup, this.extrusionGroup, this.pinGroup, this.rayGroup, this.coverageGroup);

    this.scene.add(new THREE.HemisphereLight(0xb9e8df, 0x172326, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(0.4, -0.7, 1.2);
    this.scene.add(sun);

    this.center = { lon: 0, lat: 0 };
    this.tiles = [];
    this.options = {};
    this.verticalScale = 1.4;
    this.yaw = 0.55;
    this.pitch = 0.72;
    this.distance = 65000;
    this.cameraTarget = new THREE.Vector3(0, 0, 0);
    this.dragging = false;
    this.pointerMode = "orbit";
    this.lastPointer = null;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.host);
    this.bindControls();
    this.resize();
    this.animate();
  }

  setTerrainTemplate(template) {
    this.terrainTemplate = template;
  }

  setVerticalScale(value) {
    this.verticalScale = Number(value) || 1;
    if (this.tiles.length) {
      this.rebuildMeshes();
    }
  }

  async load(options) {
    this.options = { ...this.options, ...options };
    const bounds = options.bounds;
    this.center = {
      lon: (bounds.west + bounds.east) / 2,
      lat: (bounds.south + bounds.north) / 2,
    };
    this.verticalScale = options.verticalScale || this.verticalScale;
    if (this.options.analysis?.surfaceLayers?.dsm) {
      this.tiles = [];
      this.rebuildMeshes();
      this.setStatus("Terrain: analysis DSM surface.");
      return;
    }
    const zoom = chooseTerrainZoom(bounds);
    const centerTile = lonLatToTile(this.center.lon, this.center.lat, zoom);
    const jobs = [];
    const radius = 1;

    this.setStatus("Loading terrain height tiles...");
    for (let dx = -radius; dx <= radius; dx += 1) {
      for (let dy = -radius; dy <= radius; dy += 1) {
        const x = wrapTileX(centerTile.x + dx, zoom);
        const y = centerTile.y + dy;
        if (y < 0 || y >= 2 ** zoom) continue;
        jobs.push(this.fetchTerrainTile(zoom, x, y));
      }
    }

    const tiles = (await Promise.all(jobs)).filter(Boolean);
    if (!tiles.length) {
      this.setStatus("Terrain height tiles were not available for this view.");
      return;
    }
    this.tiles = tiles;
    this.rebuildMeshes();
    this.setStatus(`Terrain: ${tiles.length} height tiles at z${zoom}.`);
  }

  updateAnalysis(options) {
    this.options = { ...this.options, ...options };
    if (this.options.analysis?.surfaceLayers?.dsm || this.tiles.length) {
      this.rebuildMeshes();
    }
  }

  async fetchTerrainTile(z, x, y) {
    const url = this.terrainTemplate
      .replace("{z}", String(z))
      .replace("{x}", String(x))
      .replace("{y}", String(y));
    try {
      const response = await fetch(url, { mode: "cors" });
      if (!response.ok) throw new Error(`tile ${response.status}`);
      const blob = await response.blob();
      const bitmap = await createImageBitmap(blob);
      const canvas = document.createElement("canvas");
      canvas.width = TILE_SIZE;
      canvas.height = TILE_SIZE;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0);
      const rgba = ctx.getImageData(0, 0, TILE_SIZE, TILE_SIZE).data;
      const heights = new Float32Array(TILE_SIZE * TILE_SIZE);
      for (let i = 0; i < heights.length; i += 1) {
        const p = i * 4;
        heights[i] = rgba[p] * 256 + rgba[p + 1] + rgba[p + 2] / 256 - 32768;
      }
      return { z, x, y, heights };
    } catch {
      return null;
    }
  }

  rebuildMeshes() {
    clearGroup(this.meshGroup);
    clearGroup(this.extrusionGroup);
    clearGroup(this.pinGroup);
    clearGroup(this.rayGroup);
    clearGroup(this.coverageGroup);

    let maxExtent = 2000;
    const dsm = this.options.analysis?.surfaceLayers?.dsm;
    if (dsm) {
      const mesh = this.buildRasterTerrainMesh(dsm);
      this.meshGroup.add(mesh);
      maxExtent = Math.max(maxExtent, mesh.userData.extent || 0);
    } else {
      this.tiles.forEach((tile) => {
        const mesh = this.buildTileMesh(tile);
        this.meshGroup.add(mesh);
        maxExtent = Math.max(maxExtent, mesh.userData.extent || 0);
      });
    }
    this.distance = clamp(maxExtent * 1.65, 3500, 180000);
    this.updateCamera();
    this.addExtrusions();
    this.addPins();
    this.addRays();
    this.addCoveragePlan();
  }

  buildRasterTerrainMesh(raster) {
    const positions = [];
    const colors = [];
    const indices = [];
    const rowStep = Math.max(1, Math.ceil(raster.rows / ANALYSIS_MESH_TARGET));
    const colStep = Math.max(1, Math.ceil(raster.cols / ANALYSIS_MESH_TARGET));
    const sampleRows = sampleIndices(raster.rows, rowStep);
    const sampleCols = sampleIndices(raster.cols, colStep);
    let maxExtent = 0;

    sampleRows.forEach((row) => {
      sampleCols.forEach((col) => {
        const lonlat = rasterCellLonLat(raster, row, col);
        const local = this.localXY(lonlat.lon, lonlat.lat);
        const raw = raster.values[row * raster.cols + col];
        const elevation = raw !== null && Number.isFinite(raw) ? raw : raster.min || 0;
        positions.push(local.x, local.y, elevation * this.verticalScale);
        const color = this.vertexColor(lonlat.lon, lonlat.lat, elevation);
        colors.push(color.r, color.g, color.b);
        maxExtent = Math.max(maxExtent, Math.hypot(local.x, local.y));
      });
    });

    const width = sampleCols.length;
    for (let row = 0; row < sampleRows.length - 1; row += 1) {
      for (let col = 0; col < sampleCols.length - 1; col += 1) {
        const a = row * width + col;
        const b = a + 1;
        const c = a + width;
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.9,
      metalness: 0.02,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.extent = maxExtent;
    return mesh;
  }

  buildTileMesh(tile) {
    const positions = [];
    const colors = [];
    const indices = [];
    const pointsPerSide = TILE_SIZE / SAMPLE_STEP + 1;
    let maxExtent = 0;
    const elevations = [];

    for (let py = 0; py <= TILE_SIZE; py += SAMPLE_STEP) {
      for (let px = 0; px <= TILE_SIZE; px += SAMPLE_STEP) {
        const sampleX = Math.min(px, TILE_SIZE - 1);
        const sampleY = Math.min(py, TILE_SIZE - 1);
        const lonlat = tilePixelToLonLat(tile.x, tile.y, tile.z, px, py);
        const local = this.localXY(lonlat.lon, lonlat.lat);
        const elevation = tile.heights[sampleY * TILE_SIZE + sampleX];
        const z = elevation * this.verticalScale;
        positions.push(local.x, local.y, z);
        elevations.push(elevation);
        maxExtent = Math.max(maxExtent, Math.hypot(local.x, local.y));

        const color = this.vertexColor(lonlat.lon, lonlat.lat, elevation);
        colors.push(color.r, color.g, color.b);
      }
    }

    for (let row = 0; row < pointsPerSide - 1; row += 1) {
      for (let col = 0; col < pointsPerSide - 1; col += 1) {
        const a = row * pointsPerSide + col;
        const b = a + 1;
        const c = a + pointsPerSide;
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.92,
      metalness: 0.03,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.extent = maxExtent;
    mesh.userData.elevations = elevations;
    return mesh;
  }

  vertexColor(lon, lat, elevation) {
    const normalizedElevation = clamp((elevation + 80) / 1400, 0, 1);
    const base = new THREE.Color().setRGB(
      0.08 + normalizedElevation * 0.33,
      0.16 + normalizedElevation * 0.28,
      0.18 + normalizedElevation * 0.2
    );
    const heatRaster = this.options.analysis?.kind === "coverage"
      ? this.options.analysis.coverage
      : this.options.analysis?.probability;
    if (this.options.showProbability && heatRaster) {
      const p = rasterValueAt(heatRaster, lat, lon);
      if (p !== null && Number.isFinite(p) && p > 0.02) {
        const [r, g, b] = heatColor(p, 1);
        const heat = new THREE.Color(r / 255, g / 255, b / 255);
        base.lerp(heat, clamp(0.25 + p * 0.62, 0, 0.86));
      }
    }
    return base;
  }

  addExtrusions() {
    const analysis = this.options.analysis;
    if (!analysis?.surfaceLayers) return;
    if (this.options.showCanopy) {
      this.addCanopyCover(analysis.surfaceLayers.canopyHeight);
    }
    if (this.options.showBuildings) {
      this.addExtrusionLayer(analysis.surfaceLayers.buildingHeight, {
        threshold: 2,
        maxItems: 420,
        color: 0x75a7ef,
      });
    }
  }

  addCanopyCover(layer) {
    if (!layer?.values?.length) return;
    const positions = [];
    const colors = [];
    const indices = [];
    const rowStep = Math.max(1, Math.ceil(layer.rows / 72));
    const colStep = Math.max(1, Math.ceil(layer.cols / 72));
    const sampleRows = sampleIndices(layer.rows, rowStep);
    const sampleCols = sampleIndices(layer.cols, colStep);

    sampleRows.forEach((row) => {
      sampleCols.forEach((col) => {
        const lonlat = rasterCellLonLat(layer, row, col);
        const canopy = layer.values[row * layer.cols + col];
        const base = this.surfaceHeight(lonlat.lon, lonlat.lat);
        const height = canopy !== null && Number.isFinite(canopy) ? Math.max(canopy, 0) : 0;
        positions.push(
          ...Object.values(this.localXY(lonlat.lon, lonlat.lat)),
          (base + height + 0.8) * this.verticalScale
        );
        const t = clamp(height / Math.max(layer.max || 35, 1), 0, 1);
        colors.push(0.18 + t * 0.22, 0.38 + t * 0.44, 0.18 + t * 0.16);
      });
    });

    const width = sampleCols.length;
    for (let row = 0; row < sampleRows.length - 1; row += 1) {
      for (let col = 0; col < sampleCols.length - 1; col += 1) {
        const values = [
          layer.values[sampleRows[row] * layer.cols + sampleCols[col]],
          layer.values[sampleRows[row] * layer.cols + sampleCols[col + 1]],
          layer.values[sampleRows[row + 1] * layer.cols + sampleCols[col]],
          layer.values[sampleRows[row + 1] * layer.cols + sampleCols[col + 1]],
        ];
        if (!values.some((value) => value !== null && Number.isFinite(value) && value > 1.0)) continue;
        const a = row * width + col;
        const b = a + 1;
        const c = a + width;
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }
    if (!indices.length) {
      this.addCanopyVolumes(layer);
      return;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.44,
      roughness: 0.96,
      metalness: 0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.extrusionGroup.add(new THREE.Mesh(geometry, material));
    this.addCanopyVolumes(layer);
  }

  addCanopyVolumes(layer) {
    const stride = Math.max(1, Math.ceil(Math.max(layer.rows, layer.cols) / 44));
    const candidates = [];
    for (let row = 0; row < layer.rows; row += stride) {
      for (let col = 0; col < layer.cols; col += stride) {
        const value = layer.values[row * layer.cols + col];
        if (value !== null && Number.isFinite(value) && value > 2.5) {
          candidates.push({ row, col, value });
        }
      }
    }
    if (!candidates.length) return;
    candidates.sort((a, b) => b.value - a.value);
    const chosen = candidates.slice(0, 900);
    const geometry = new THREE.CylinderGeometry(0.52, 0.7, 1, 9, 1, true);
    const material = new THREE.MeshStandardMaterial({
      color: 0x75c96d,
      transparent: true,
      opacity: 0.34,
      roughness: 0.98,
      metalness: 0,
      depthWrite: false,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, chosen.length);
    const matrix = new THREE.Matrix4();
    const quat = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
    const footprint = Math.max(layer.resX || 30, layer.resY || 30) * stride * 0.72;
    chosen.forEach((item, index) => {
      const lonlat = rasterCellLonLat(layer, item.row, item.col);
      const local = this.localXY(lonlat.lon, lonlat.lat);
      const base = this.surfaceHeight(lonlat.lon, lonlat.lat) * this.verticalScale;
      const height = Math.max(item.value * this.verticalScale, 4);
      matrix.compose(
        new THREE.Vector3(local.x, local.y, base + height / 2),
        quat,
        new THREE.Vector3(footprint, height, footprint)
      );
      mesh.setMatrixAt(index, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    this.extrusionGroup.add(mesh);
  }

  addExtrusionLayer(layer, config) {
    if (!layer?.values?.length) return;
    const stride = Math.max(1, Math.ceil(Math.max(layer.rows, layer.cols) / 34));
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshStandardMaterial({
      color: config.color,
      transparent: true,
      opacity: 0.58,
      roughness: 0.88,
    });
    const candidates = [];
    for (let row = 0; row < layer.rows; row += stride) {
      for (let col = 0; col < layer.cols; col += stride) {
        const value = layer.values[row * layer.cols + col];
        if (value !== null && value >= config.threshold) {
          candidates.push({ row, col, value });
        }
      }
    }
    candidates.sort((a, b) => b.value - a.value);
    const chosen = candidates.slice(0, config.maxItems);
    if (!chosen.length) return;

    const mesh = new THREE.InstancedMesh(geometry, material, chosen.length);
    const matrix = new THREE.Matrix4();
    const quat = new THREE.Quaternion();
    chosen.forEach((item, index) => {
      const lonlat = rasterCellLonLat(layer, item.row, item.col);
      const local = this.localXY(lonlat.lon, lonlat.lat);
      const base = this.surfaceHeight(lonlat.lon, lonlat.lat) * this.verticalScale;
      const height = Math.max(item.value * this.verticalScale, 3);
      const footprint = Math.max(layer.resX || 30, layer.resY || 30) * stride * 0.78;
      matrix.compose(
        new THREE.Vector3(local.x, local.y, base + height / 2),
        quat,
        new THREE.Vector3(footprint, footprint, height)
      );
      mesh.setMatrixAt(index, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    this.extrusionGroup.add(mesh);
  }

  addRays() {
    const mode = this.options.rayMode;
    const sightings = this.options.sightings || [];
    if (mode === "off" || this.options.selectedSighting === null) return;
    const sighting = sightings[this.options.selectedSighting];
    if (!sighting) return;
    const start = this.pointVector(sighting.lon, sighting.lat, sighting.altitude || 0);
    const dsm = this.options.analysis?.surfaceLayers?.dsm;
    const ground = this.options.analysis?.surfaceLayers?.bareEarth || dsm;
    const targetHeight = Number(this.options.targetHeightM) || 1.5;

    if (mode === "fan" && dsm) {
      const observerZ = observerElevation(dsm, sighting.lat, sighting.lon, sighting.altitude || 0);
      const fan = castViewshedFan({
        dsm,
        ground,
        origin: { lat: sighting.lat, lon: sighting.lon },
        observerZ,
        targetHeight,
        maxRange: this.options.maxRangeM || 12000,
        rayCount: 96,
      });
      this.addRayFanGeometry(start, fan, targetHeight);
      return;
    }

    if (mode === "fan") {
      this.addPreviewFan(start);
      return;
    }

    const target = this.options.selectedCell || this.options.analysis?.mostLikely;
    if (target) {
      const end = this.pointVector(target.lon, target.lat, 1.5);
      const material = new THREE.LineBasicMaterial({
        color: mode === "profile" ? 0xf0b54d : 0x55d6c2,
        transparent: true,
        opacity: 0.95,
      });
      const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
      this.rayGroup.add(new THREE.Line(geometry, material));
    }
  }

  addRayFanGeometry(start, fan, targetHeight) {
    const visible = [];
    const occluded = [];
    fan.rays.forEach((ray) => {
      if (!ray.points.length) return;
      let last = start;
      ray.points.forEach((point) => {
        const current = this.pointVector(point.lon, point.lat, targetHeight);
        const out = point.visible ? visible : occluded;
        out.push(last.x, last.y, last.z, current.x, current.y, current.z);
        last = current;
      });
    });
    this.addLineSegments(visible, 0x55d6c2, 0.56);
    this.addLineSegments(occluded, 0xf0655a, 0.22);
  }

  addPreviewFan(start) {
    const range = this.options.maxRangeM || 12000;
    const positions = [];
    for (let i = 0; i < 48; i += 1) {
      const angle = (i / 48) * Math.PI * 2;
      positions.push(start.x, start.y, start.z);
      positions.push(start.x + Math.cos(angle) * range, start.y + Math.sin(angle) * range, start.z);
    }
    this.addLineSegments(positions, 0x55d6c2, 0.14);
  }

  addLineSegments(positions, color, opacity) {
    if (!positions.length) return;
    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    this.rayGroup.add(new THREE.LineSegments(geometry, material));
  }

  addPins() {
    if (this.options.analysis?.kind === "coverage") return;
    if (!this.options.showSightings) return;
    const sightings = this.options.sightings || [];
    sightings.forEach((sighting, index) => {
      if (!Number.isFinite(sighting?.lat) || !Number.isFinite(sighting?.lon)) return;
      const selected = index === this.options.selectedSighting;
      const color = selected ? 0x55d6c2 : 0x69a8ff;
      const base = this.pointVector(sighting.lon, sighting.lat, 0.8);
      const top = this.pointVector(sighting.lon, sighting.lat, Math.max(sighting.altitude || 0, 2));
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(selected ? 58 : 46, 18, 12),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: selected ? 0.45 : 0.2,
          roughness: 0.42,
        })
      );
      marker.position.copy(top);
      this.pinGroup.add(marker);
      this.addLineToGroup(this.pinGroup, [base, top], color, selected ? 0.9 : 0.65);
      this.addUncertaintyRing(sighting, color, selected);
    });
  }

  addUncertaintyRing(sighting, color, selected) {
    const radius = Math.max(Number(sighting.position_sigma_m) || 0, 1);
    const points = [];
    for (let i = 0; i <= 96; i += 1) {
      const angle = (i / 96) * Math.PI * 2;
      const lonScale = METERS_PER_DEG_LAT * Math.cos((sighting.lat * Math.PI) / 180);
      const lon = sighting.lon + (Math.cos(angle) * radius) / Math.max(lonScale, 1);
      const lat = sighting.lat + (Math.sin(angle) * radius) / METERS_PER_DEG_LAT;
      points.push(this.pointVector(lon, lat, 0.7));
    }
    this.addLineToGroup(this.pinGroup, points, color, selected ? 0.7 : 0.34);
  }

  addLineToGroup(group, points, color, opacity) {
    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material));
  }

  addCoveragePlan() {
    const analysis = this.options.analysis;
    if (analysis?.kind !== "coverage") return;
    const area = this.options.flightArea || analysis.flightArea;
    if (area?.center && Number.isFinite(area.radiusM)) {
      const points = [];
      for (let i = 0; i <= 96; i += 1) {
        const angle = (i / 96) * Math.PI * 2;
        const metersPerDegLon = METERS_PER_DEG_LAT * Math.cos((area.center.lat * Math.PI) / 180);
        const lon = area.center.lon + (Math.cos(angle) * area.radiusM) / Math.max(metersPerDegLon, 1);
        const lat = area.center.lat + (Math.sin(angle) * area.radiusM) / METERS_PER_DEG_LAT;
        points.push(this.pointVector(lon, lat, Math.max(area.altitudeAglM || 0, 1)));
      }
      const material = new THREE.LineBasicMaterial({
        color: 0x55d6c2,
        transparent: true,
        opacity: 0.85,
      });
      this.coverageGroup.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        material
      ));
    }
    if (analysis.recommendedLaunch) {
      const position = this.pointVector(analysis.recommendedLaunch.lon, analysis.recommendedLaunch.lat, 5);
      const geometry = new THREE.ConeGeometry(70, 180, 3);
      const material = new THREE.MeshStandardMaterial({ color: 0x55d6c2, roughness: 0.55 });
      const marker = new THREE.Mesh(geometry, material);
      marker.position.copy(position);
      marker.rotation.x = Math.PI;
      this.coverageGroup.add(marker);
    }
  }

  pointVector(lon, lat, aboveGround = 0) {
    const local = this.localXY(lon, lat);
    const z = (this.surfaceHeight(lon, lat) + aboveGround) * this.verticalScale;
    return new THREE.Vector3(local.x, local.y, z);
  }

  surfaceHeight(lon, lat) {
    const dsm = this.options.analysis?.surfaceLayers?.dsm;
    const dsmValue = rasterValueAt(dsm, lat, lon);
    if (dsmValue !== null && Number.isFinite(dsmValue)) return dsmValue;
    for (const tile of this.tiles) {
      const pixel = lonLatToTilePixel(lon, lat, tile.z);
      if (pixel.xTile === tile.x && pixel.yTile === tile.y) {
        const x = clamp(Math.floor(pixel.x), 0, TILE_SIZE - 1);
        const y = clamp(Math.floor(pixel.y), 0, TILE_SIZE - 1);
        return tile.heights[y * TILE_SIZE + x];
      }
    }
    return 0;
  }

  localXY(lon, lat) {
    const latScale = METERS_PER_DEG_LAT;
    const lonScale = METERS_PER_DEG_LAT * Math.cos((this.center.lat * Math.PI) / 180);
    return {
      x: (lon - this.center.lon) * lonScale,
      y: (lat - this.center.lat) * latScale,
    };
  }

  bindControls() {
    const canvas = this.renderer.domElement;
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => {
      this.dragging = true;
      this.pointerMode = event.button === 1 || event.button === 2 || event.shiftKey || event.ctrlKey
        ? "pan"
        : "orbit";
      this.lastPointer = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.dragging || !this.lastPointer) return;
      const dx = event.clientX - this.lastPointer.x;
      const dy = event.clientY - this.lastPointer.y;
      if (this.pointerMode === "pan") {
        const right = new THREE.Vector3();
        const up = new THREE.Vector3();
        this.camera.matrix.extractBasis(right, up, new THREE.Vector3());
        const scale = this.distance / Math.max(this.host.clientHeight, 1);
        this.cameraTarget.addScaledVector(right, -dx * scale);
        this.cameraTarget.addScaledVector(up, dy * scale);
      } else {
        this.yaw -= dx * 0.006;
        this.pitch = clamp(this.pitch + dy * 0.004, 0.08, 1.5);
      }
      this.lastPointer = { x: event.clientX, y: event.clientY };
      this.updateCamera();
    });
    canvas.addEventListener("pointerup", (event) => {
      this.dragging = false;
      this.lastPointer = null;
      try {
        canvas.releasePointerCapture(event.pointerId);
      } catch {
        // Pointer capture may already be released by the browser.
      }
    });
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        this.distance = clamp(this.distance * (event.deltaY > 0 ? 1.1 : 0.9), 1200, 260000);
        this.updateCamera();
      },
      { passive: false }
    );
    canvas.addEventListener("dblclick", () => this.focusSelected());
  }

  updateCamera() {
    const flat = Math.cos(this.pitch) * this.distance;
    this.camera.position.set(
      Math.sin(this.yaw) * flat,
      -Math.cos(this.yaw) * flat,
      Math.sin(this.pitch) * this.distance
    );
    this.camera.position.add(this.cameraTarget);
    this.camera.lookAt(this.cameraTarget);
  }

  resetView() {
    this.yaw = 0.55;
    this.pitch = 0.72;
    this.cameraTarget.set(0, 0, 0);
    this.updateCamera();
  }

  topView() {
    this.pitch = 1.5;
    this.updateCamera();
  }

  northUp() {
    this.yaw = 0;
    this.updateCamera();
  }

  focusSelected() {
    const selectedSighting = this.options.sightings?.[this.options.selectedSighting];
    const point = this.options.analysis?.kind !== "coverage" && selectedSighting
      ? selectedSighting
      : this.options.selectedCell
      || this.options.analysis?.recommendedLaunch
      || this.options.analysis?.mostLikely
      || selectedSighting;
    if (!point) return;
    this.focusLonLat(point.lon, point.lat);
  }

  focusLonLat(lon, lat) {
    const local = this.localXY(lon, lat);
    this.cameraTarget.set(local.x, local.y, this.surfaceHeight(lon, lat) * this.verticalScale);
    this.distance = clamp(this.distance * 0.72, 1200, 260000);
    this.updateCamera();
  }

  focusBounds(bounds) {
    if (!bounds) return;
    const centerLon = (bounds.west + bounds.east) / 2;
    const centerLat = (bounds.south + bounds.north) / 2;
    const nw = this.localXY(bounds.west, bounds.north);
    const se = this.localXY(bounds.east, bounds.south);
    this.cameraTarget.set(
      (nw.x + se.x) / 2,
      (nw.y + se.y) / 2,
      this.surfaceHeight(centerLon, centerLat) * this.verticalScale
    );
    this.distance = clamp(Math.max(Math.abs(se.x - nw.x), Math.abs(nw.y - se.y)) * 1.5, 1200, 260000);
    this.updateCamera();
  }

  resize() {
    const width = Math.max(this.host.clientWidth, 1);
    const height = Math.max(this.host.clientHeight, 1);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.updateCamera();
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.renderer.render(this.scene, this.camera);
  }

  setStatus(message) {
    if (this.statusEl) {
      this.statusEl.textContent = message;
    }
  }
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children[0];
    group.remove(child);
    child.traverse?.((object) => {
      object.geometry?.dispose?.();
      if (Array.isArray(object.material)) {
        object.material.forEach((material) => material.dispose?.());
      } else {
        object.material?.dispose?.();
      }
    });
  }
}

function chooseTerrainZoom(bounds) {
  const span = Math.max(Math.abs(bounds.east - bounds.west), Math.abs(bounds.north - bounds.south));
  if (span > 28) return 6;
  if (span > 8) return 7;
  if (span > 2.5) return 8;
  if (span > 0.8) return 9;
  if (span > 0.25) return 10;
  if (span > 0.08) return 11;
  return 12;
}

function lonLatToTile(lon, lat, z) {
  const n = 2 ** z;
  const x = Math.floor(((lon + 180) / 360) * n);
  const latRad = (clamp(lat, -85.05112878, 85.05112878) * Math.PI) / 180;
  const y = Math.floor(((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n);
  return { x: wrapTileX(x, z), y: clamp(y, 0, n - 1) };
}

function lonLatToTilePixel(lon, lat, z) {
  const n = 2 ** z;
  const xFloat = ((lon + 180) / 360) * n;
  const latRad = (clamp(lat, -85.05112878, 85.05112878) * Math.PI) / 180;
  const yFloat = ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n;
  const xTile = wrapTileX(Math.floor(xFloat), z);
  const yTile = clamp(Math.floor(yFloat), 0, n - 1);
  return {
    xTile,
    yTile,
    x: (xFloat - Math.floor(xFloat)) * TILE_SIZE,
    y: (yFloat - Math.floor(yFloat)) * TILE_SIZE,
  };
}

function tilePixelToLonLat(xTile, yTile, z, px, py) {
  const n = 2 ** z;
  const x = xTile + px / TILE_SIZE;
  const y = yTile + py / TILE_SIZE;
  const lon = (x / n) * 360 - 180;
  const mercator = Math.PI * (1 - (2 * y) / n);
  const lat = (180 / Math.PI) * Math.atan(Math.sinh(mercator));
  return { lon, lat };
}

function wrapTileX(x, z) {
  const n = 2 ** z;
  return ((x % n) + n) % n;
}

function rasterCellLonLat(raster, row, col) {
  const b = raster.bounds;
  return {
    lon: b.west + ((col + 0.5) / raster.cols) * (b.east - b.west),
    lat: b.north - ((row + 0.5) / raster.rows) * (b.north - b.south),
  };
}

function sampleIndices(count, step) {
  const out = [];
  for (let i = 0; i < count; i += step) out.push(i);
  if (out[out.length - 1] !== count - 1) out.push(count - 1);
  return out;
}
