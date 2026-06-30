// Client-side port of the viewshed radial sweep (viewshed/kernels.py::_cast_ray).
//
// Given the DSM (occluder) and bare-earth (ground) rasters that the server
// already ships to the browser, this reproduces, ray by ray, the exact
// line-of-sight test the heatmap is built from: march outward from the
// observer, track the running-max terrain elevation angle (the horizon), and
// mark each sample visible when the target there rises above that horizon.
//
// The output is geometry-only (lat/lon points tagged visible/occluded) so it
// can drive both the 2D map fan and, later, the 3D terrain view.

import { rasterValueAt } from "./rendering.js";

const EARTH_RADIUS_M = 6371000;
const EPS = 1e-9;

// 1 / (2 * R_eff). With 4/3-Earth refraction R_eff = 4/3 * R (see
// viewshed/curvature.py). Multiply by d^2 for the curvature drop.
function invTwoEffectiveRadius(refraction) {
  const reff = refraction ? (4 / 3) * EARTH_RADIUS_M : EARTH_RADIUS_M;
  return 1 / (2 * reff);
}

function inBounds(raster, lat, lon) {
  const b = raster.bounds;
  return lon >= b.west && lon <= b.east && lat >= b.south && lat <= b.north;
}

// Absolute observer elevation, matching fusion/montecarlo.py::_observer_z.
export function observerElevation(dsm, lat, lon, altitude, altitudeIsAgl = true) {
  if (!altitudeIsAgl) return altitude;
  const base = rasterValueAt(dsm, lat, lon);
  if (base !== null && Number.isFinite(base)) return base + altitude;
  // Observer over nodata: fall back to the grid floor, as the kernel does.
  return (dsm?.min ?? 0) + altitude;
}

/**
 * Cast a fan of rays and classify every sample as visible or occluded.
 *
 * @returns {{origin, step, maxRange, rays: Array<{azimuth, points: Array<{lat, lon, d, visible}>}>}}
 */
export function castViewshedFan({
  dsm,
  ground,
  origin,
  observerZ,
  targetHeight = 1.5,
  maxRange = 12000,
  refraction = true,
  rayCount = 96,
  stepM = null,
}) {
  const invTwoReff = invTwoEffectiveRadius(refraction);
  const lat0 = origin.lat;
  const lon0 = origin.lon;
  const mPerDegLat = 111320;
  const mPerDegLon = 111320 * Math.cos((lat0 * Math.PI) / 180);
  // Cells the heatmap can't resolve aren't worth sampling; tie the march step
  // to the (downsampled) DSM resolution, but keep enough steps to be smooth.
  const step = stepM || Math.max(dsm?.resX || 30, maxRange / 240);
  const nSteps = Math.max(2, Math.floor(maxRange / step));
  const veryLow = (Number.isFinite(dsm?.min) ? dsm.min : 0) - 1000;

  const rays = [];
  for (let i = 0; i < rayCount; i += 1) {
    const azimuth = (i / rayCount) * Math.PI * 2; // clockwise from north
    const sinA = Math.sin(azimuth);
    const cosA = Math.cos(azimuth);
    let maxAng = -Infinity;
    const points = [];

    for (let k = 1; k <= nSteps; k += 1) {
      const d = k * step;
      if (d > maxRange) break;
      const lat = lat0 + (d * cosA) / mPerDegLat;
      const lon = lon0 + (d * sinA) / mPerDegLon;
      if (!inBounds(dsm, lat, lon)) break; // ray left the analysed area

      const occRaw = rasterValueAt(dsm, lat, lon);
      const occ = occRaw !== null && Number.isFinite(occRaw) ? occRaw : veryLow;
      const grndRaw = rasterValueAt(ground || dsm, lat, lon);
      const grnd = grndRaw !== null && Number.isFinite(grndRaw) ? grndRaw : veryLow;

      const drop = d * d * invTwoReff;
      const terrAng = (occ - drop - observerZ) / d;
      const targAng = (grnd + targetHeight - drop - observerZ) / d;
      const visible = targAng >= maxAng - EPS;
      if (terrAng > maxAng) maxAng = terrAng;

      points.push({ lat, lon, d, visible });
    }
    rays.push({ azimuth, points });
  }

  return { origin, step, maxRange, rays };
}
