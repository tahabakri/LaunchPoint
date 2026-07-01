const HEAT_STOPS = [
  [0.0, [24, 44, 72, 0]],
  [0.08, [38, 83, 126, 70]],
  [0.28, [43, 168, 157, 118]],
  [0.52, [238, 185, 72, 158]],
  [0.76, [235, 102, 74, 205]],
  [1.0, [219, 55, 80, 235]],
];

const SURFACE_STOPS = {
  dsm: [
    [0, [20, 36, 42, 55]],
    [1, [147, 197, 190, 185]],
  ],
  bareEarth: [
    [0, [31, 42, 58, 55]],
    [1, [144, 174, 224, 180]],
  ],
  canopyHeight: [
    [0, [18, 35, 26, 0]],
    [1, [119, 198, 111, 190]],
  ],
  buildingHeight: [
    [0, [22, 35, 52, 0]],
    [1, [116, 164, 238, 190]],
  ],
  launchWeight: [
    [0, [232, 91, 86, 180]],
    [1, [85, 214, 194, 110]],
  ],
};

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function mix(a, b, t) {
  return a + (b - a) * t;
}

function colorFromStops(stops, t) {
  const n = clamp(t, 0, 1);
  for (let i = 1; i < stops.length; i += 1) {
    const [lastStop, lastColor] = stops[i - 1];
    const [nextStop, nextColor] = stops[i];
    if (n <= nextStop) {
      const local = (n - lastStop) / Math.max(nextStop - lastStop, 0.00001);
      return [
        Math.round(mix(lastColor[0], nextColor[0], local)),
        Math.round(mix(lastColor[1], nextColor[1], local)),
        Math.round(mix(lastColor[2], nextColor[2], local)),
        Math.round(mix(lastColor[3], nextColor[3], local)),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

export function heatColor(value, opacity = 1) {
  const rgba = colorFromStops(HEAT_STOPS, clamp(value, 0, 1));
  rgba[3] = Math.round(rgba[3] * opacity);
  return rgba;
}

export function rasterValueAt(raster, lat, lon) {
  if (!raster || !raster.values) return null;
  const b = raster.bounds;
  if (lon < b.west || lon > b.east || lat < b.south || lat > b.north) return null;
  const col = Math.floor(((lon - b.west) / Math.max(b.east - b.west, 1e-9)) * raster.cols);
  const row = Math.floor(((b.north - lat) / Math.max(b.north - b.south, 1e-9)) * raster.rows);
  const safeRow = clamp(row, 0, raster.rows - 1);
  const safeCol = clamp(col, 0, raster.cols - 1);
  return raster.values[safeRow * raster.cols + safeCol];
}

export function rasterCellAt(raster, lat, lon) {
  if (!raster || !raster.values) return null;
  const b = raster.bounds;
  if (lon < b.west || lon > b.east || lat < b.south || lat > b.north) return null;
  const col = Math.floor(((lon - b.west) / Math.max(b.east - b.west, 1e-9)) * raster.cols);
  const row = Math.floor(((b.north - lat) / Math.max(b.north - b.south, 1e-9)) * raster.rows);
  const safeRow = clamp(row, 0, raster.rows - 1);
  const safeCol = clamp(col, 0, raster.cols - 1);
  return {
    row: safeRow,
    col: safeCol,
    value: raster.values[safeRow * raster.cols + safeCol],
  };
}

export function rasterToCanvas(raster, options = {}) {
  const kind = options.kind || "probability";
  const opacity = options.opacity ?? 1;
  const solid = options.solid ?? false;
  const canvas = document.createElement("canvas");
  canvas.width = raster.cols;
  canvas.height = raster.rows;
  const ctx = canvas.getContext("2d", { willReadFrequently: false });
  const image = ctx.createImageData(raster.cols, raster.rows);
  const min = options.min ?? raster.min ?? 0;
  const max = options.max ?? raster.max ?? 1;
  const span = Math.max(max - min, 1e-9);
  const stops = SURFACE_STOPS[kind] || SURFACE_STOPS.dsm;

  for (let i = 0; i < raster.values.length; i += 1) {
    const value = raster.values[i];
    const px = i * 4;
    let rgba = [0, 0, 0, 0];
    if (value !== null && Number.isFinite(value)) {
      if (kind === "probability" || kind === "contribution") {
        rgba = heatColor(value, opacity);
      } else if (kind === "credible") {
        const isInside = value >= 0.5;
        rgba = isInside ? [240, 181, 77, Math.round(82 * opacity)] : [0, 0, 0, 0];
        if (isInside && isCredibleEdge(raster, i)) {
          rgba = [255, 228, 151, Math.round(210 * opacity)];
        }
      } else {
        rgba = colorFromStops(stops, clamp((value - min) / span, 0, 1));
        rgba[3] = Math.round((solid ? 255 : rgba[3]) * opacity);
      }
    }
    image.data[px] = rgba[0];
    image.data[px + 1] = rgba[1];
    image.data[px + 2] = rgba[2];
    image.data[px + 3] = rgba[3];
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function isCredibleEdge(raster, index) {
  const row = Math.floor(index / raster.cols);
  const col = index % raster.cols;
  const neighbors = [
    [row - 1, col],
    [row + 1, col],
    [row, col - 1],
    [row, col + 1],
  ];
  return neighbors.some(([r, c]) => {
    if (r < 0 || c < 0 || r >= raster.rows || c >= raster.cols) return true;
    return (raster.values[r * raster.cols + c] || 0) < 0.5;
  });
}

export function rasterToDataUrl(raster, options = {}) {
  return rasterToCanvas(raster, options).toDataURL("image/png");
}

export function formatCoord(value) {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(6);
}

export function formatProbability(value) {
  if (value === null || !Number.isFinite(value)) return "-";
  return `${Math.round(value * 1000) / 10}%`;
}

export function formatArea(km2) {
  if (!Number.isFinite(km2)) return "-";
  if (km2 < 1) return `${Math.round(km2 * 100) / 100} km2`;
  return `${Math.round(km2 * 10) / 10} km2`;
}

export function formatMeters(value) {
  if (value === null || !Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(2)} km`;
  return `${Math.round(value)} m`;
}

export function lonLatToWorldPixel(lon, lat, zoom) {
  const sinLat = Math.sin((clamp(lat, -85.05112878, 85.05112878) * Math.PI) / 180);
  const scale = 256 * 2 ** zoom;
  return {
    x: ((lon + 180) / 360) * scale,
    y: (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale,
  };
}

export function worldPixelToLonLat(x, y, zoom) {
  const scale = 256 * 2 ** zoom;
  const lon = (x / scale) * 360 - 180;
  const n = Math.PI - (2 * Math.PI * y) / scale;
  const lat = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
  return { lon, lat };
}

function boundsRect(bounds, viewBounds, zoom, canvas) {
  const viewNW = lonLatToWorldPixel(viewBounds.west, viewBounds.north, zoom);
  const viewSE = lonLatToWorldPixel(viewBounds.east, viewBounds.south, zoom);
  const bNW = lonLatToWorldPixel(bounds.west, bounds.north, zoom);
  const bSE = lonLatToWorldPixel(bounds.east, bounds.south, zoom);
  const width = Math.max(viewSE.x - viewNW.x, 1);
  const height = Math.max(viewSE.y - viewNW.y, 1);
  const sx = canvas.width / width;
  const sy = canvas.height / height;
  return {
    x: (bNW.x - viewNW.x) * sx,
    y: (bNW.y - viewNW.y) * sy,
    w: (bSE.x - bNW.x) * sx,
    h: (bSE.y - bNW.y) * sy,
  };
}

export function drawRasterInView(ctx, raster, viewBounds, zoom, canvas, options = {}) {
  if (!raster) return;
  const source = rasterToCanvas(raster, options);
  const rect = boundsRect(raster.bounds, viewBounds, zoom, canvas);
  ctx.drawImage(source, rect.x, rect.y, rect.w, rect.h);
}

function pointToCanvas(lon, lat, viewBounds, zoom, canvas) {
  const viewNW = lonLatToWorldPixel(viewBounds.west, viewBounds.north, zoom);
  const viewSE = lonLatToWorldPixel(viewBounds.east, viewBounds.south, zoom);
  const p = lonLatToWorldPixel(lon, lat, zoom);
  const sx = canvas.width / Math.max(viewSE.x - viewNW.x, 1);
  const sy = canvas.height / Math.max(viewSE.y - viewNW.y, 1);
  return {
    x: (p.x - viewNW.x) * sx,
    y: (p.y - viewNW.y) * sy,
  };
}

export function drawSightings(ctx, sightings, viewBounds, zoom, canvas, selectedIndex = null) {
  sightings.forEach((sighting, index) => {
    const p = pointToCanvas(sighting.lon, sighting.lat, viewBounds, zoom, canvas);
    const selected = index === selectedIndex;
    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x, p.y, selected ? 9 : 7, 0, Math.PI * 2);
    ctx.fillStyle = selected ? "rgba(85,214,194,0.95)" : "rgba(105,168,255,0.92)";
    ctx.strokeStyle = "rgba(8,16,19,0.92)";
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#f3faf7";
    ctx.font = "700 11px ui-monospace, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(index + 1), p.x, p.y + 0.5);
    ctx.restore();
  });
}

export function drawMostLikely(ctx, point, viewBounds, zoom, canvas) {
  if (!point) return;
  const p = pointToCanvas(point.lon, point.lat, viewBounds, zoom, canvas);
  ctx.save();
  ctx.strokeStyle = "rgba(240,181,77,0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(p.x - 11, p.y);
  ctx.lineTo(p.x + 11, p.y);
  ctx.moveTo(p.x, p.y - 11);
  ctx.lineTo(p.x, p.y + 11);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

export function drawRecommendedLaunch(ctx, point, viewBounds, zoom, canvas) {
  if (!point) return;
  const p = pointToCanvas(point.lon, point.lat, viewBounds, zoom, canvas);
  ctx.save();
  ctx.fillStyle = "rgba(85,214,194,0.96)";
  ctx.strokeStyle = "rgba(8,16,19,0.92)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(p.x, p.y - 12);
  ctx.lineTo(p.x + 10, p.y + 8);
  ctx.lineTo(p.x - 10, p.y + 8);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

export function drawFlightArea(ctx, area, viewBounds, zoom, canvas) {
  if (!area?.center || !Number.isFinite(area.radiusM)) return;
  const center = pointToCanvas(area.center.lon, area.center.lat, viewBounds, zoom, canvas);
  const metersPerDegLon = 111320 * Math.cos((area.center.lat * Math.PI) / 180);
  const edgeLon = area.center.lon + area.radiusM / Math.max(metersPerDegLon, 1);
  const edge = pointToCanvas(edgeLon, area.center.lat, viewBounds, zoom, canvas);
  const radiusPx = Math.abs(edge.x - center.x);
  ctx.save();
  ctx.fillStyle = "rgba(85,214,194,0.08)";
  ctx.strokeStyle = "rgba(85,214,194,0.88)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radiusPx, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

export async function drawOsmBasemap(ctx, viewBounds, zoom, canvas, template) {
  const viewNW = lonLatToWorldPixel(viewBounds.west, viewBounds.north, zoom);
  const viewSE = lonLatToWorldPixel(viewBounds.east, viewBounds.south, zoom);
  const minTileX = Math.floor(viewNW.x / 256);
  const maxTileX = Math.floor(viewSE.x / 256);
  const minTileY = Math.floor(viewNW.y / 256);
  const maxTileY = Math.floor(viewSE.y / 256);
  const tileCount = 2 ** zoom;
  const scaleX = canvas.width / Math.max(viewSE.x - viewNW.x, 1);
  const scaleY = canvas.height / Math.max(viewSE.y - viewNW.y, 1);

  ctx.fillStyle = "#071114";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const jobs = [];
  for (let x = minTileX; x <= maxTileX; x += 1) {
    for (let y = minTileY; y <= maxTileY; y += 1) {
      if (y < 0 || y >= tileCount) continue;
      const wrappedX = ((x % tileCount) + tileCount) % tileCount;
      const url = template
        .replace("{z}", String(zoom))
        .replace("{x}", String(wrappedX))
        .replace("{y}", String(y));
      jobs.push(loadImage(url).then((image) => ({ image, x, y })).catch(() => null));
    }
  }

  const tiles = await Promise.all(jobs);
  tiles.filter(Boolean).forEach(({ image, x, y }) => {
    const dx = (x * 256 - viewNW.x) * scaleX;
    const dy = (y * 256 - viewNW.y) * scaleY;
    ctx.drawImage(image, dx, dy, 256 * scaleX, 256 * scaleY);
  });

  ctx.fillStyle = "rgba(8,16,19,0.24)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}
