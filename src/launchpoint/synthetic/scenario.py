"""Generate a synthetic terrain + controller + visible drone sightings.

The terrain is a deterministic analytic surface (a tilted plane plus a few
Gaussian hills and a blocking ridge) so that line-of-sight is genuinely
obstructed in some directions — otherwise a viewshed test is trivial. Sightings
are only kept if they have clear LOS from the controller over this surface, so
by construction the controller lies inside the true visible-from-all region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from launchpoint.core.geo import BBox, Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.core.sighting import Sighting


def _synthetic_surface(xx: np.ndarray, yy: np.ndarray, seed: int) -> np.ndarray:
    """Analytic height (metres) at local coordinates (xx, yy) in metres.

    xx, yy are offsets from the scenario centre. Returns elevation in metres.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros_like(xx, dtype=np.float64)

    # Gentle tilt so there's a regional slope.
    z += 0.01 * xx + 0.006 * yy

    # A blocking ridge running NE-SW through the middle.
    ridge = np.exp(-((0.7 * xx + 0.7 * yy) ** 2) / (2 * 350.0**2))
    z += 120.0 * ridge

    # A handful of hills at random-but-fixed locations.
    for _ in range(6):
        cx = rng.uniform(-2500, 2500)
        cy = rng.uniform(-2500, 2500)
        amp = rng.uniform(30, 90)
        sigma = rng.uniform(250, 600)
        z += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))

    # Low-amplitude roughness.
    z += 3.0 * np.sin(xx / 180.0) * np.cos(yy / 160.0)
    return z + 200.0  # lift everything above sea level


def _los_clear(
    surf: np.ndarray,
    res: float,
    r0: float,
    c0: float,
    z0: float,
    r1: float,
    c1: float,
    z1: float,
) -> bool:
    """Brute-force line-of-sight over a surface array (no curvature).

    Self-contained on purpose: the synthetic generator must not depend on the
    viewshed module it is meant to validate. Steps along the segment in pixel
    space, comparing the straight-line height against the sampled surface.
    """
    n = int(np.hypot(r1 - r0, c1 - c0)) + 1
    if n <= 1:
        return True
    ts = np.linspace(0.0, 1.0, n)
    rows = r0 + (r1 - r0) * ts
    cols = c0 + (c1 - c0) * ts
    zline = z0 + (z1 - z0) * ts
    ri = np.clip(np.round(rows).astype(int), 0, surf.shape[0] - 1)
    ci = np.clip(np.round(cols).astype(int), 0, surf.shape[1] - 1)
    ground = surf[ri, ci]
    # Ignore the two endpoints (observer/target heights handled by z0,z1).
    inner = slice(1, n - 1)
    return bool(np.all(zline[inner] >= ground[inner] - 1e-6))


@dataclass
class SyntheticScenario:
    """A fully-known test world.

    Attributes
    ----------
    surface:
        The DSM-equivalent occluding surface (RasterGrid, UTM, metres).
    controller_lonlat:
        True (lon, lat) of the planted controller.
    controller_xy:
        True (easting, northing) of the controller in the scenario UTM.
    sightings:
        Drone sightings with clear LOS from the controller.
    projector:
        WGS84 <-> scenario-UTM projector.
    antenna_height_m:
        Controller antenna height used when generating LOS.
    """

    surface: RasterGrid
    controller_lonlat: tuple[float, float]
    controller_xy: tuple[float, float]
    sightings: list[Sighting]
    projector: Projector
    antenna_height_m: float

    def controller_pixel(self) -> tuple[int, int]:
        return self.surface.world_to_pixel(*self.controller_xy)


def make_default_scenario(
    center_lon: float = 8.55,
    center_lat: float = 47.37,
    extent_m: float = 6000.0,
    resolution_m: float = 30.0,
    n_sightings: int = 4,
    drone_altitude_agl: float = 120.0,
    antenna_height_m: float = 1.5,
    position_sigma_m: float = 200.0,
    altitude_sigma_m: float = 25.0,
    seed: int = 7,
) -> SyntheticScenario:
    """Build a reproducible synthetic scenario near Zurich (arbitrary anchor).

    The anchor only matters so that WGS84<->UTM projection is realistic; the
    physics is entirely in the synthetic surface.
    """
    proj = Projector.for_point(center_lon, center_lat)
    cx, cy = proj.to_utm(center_lon, center_lat)

    half = extent_m / 2.0
    bbox = BBox(cx - half, cy - half, cx + half, cy + half)
    grid = RasterGrid.empty(bbox, resolution_m, proj.utm_crs, fill=0.0, dtype=np.float64)

    rows, cols = grid.shape
    # Local coordinates (metres relative to centre) at each pixel centre.
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    wx, wy = grid.transform * (jj + 0.5, ii + 0.5)
    lx = np.asarray(wx) - cx
    ly = np.asarray(wy) - cy
    grid.data = _synthetic_surface(lx, ly, seed)

    # Place the controller a little off-centre, on the ground.
    rng = np.random.default_rng(seed + 100)
    ctrl_x = cx + rng.uniform(-1200, 1200)
    ctrl_y = cy + rng.uniform(-1200, 1200)
    cr, cc = grid.world_to_pixel(ctrl_x, ctrl_y)
    ctrl_ground = float(grid.data[cr, cc])
    ctrl_eye = ctrl_ground + antenna_height_m
    ctrl_lon, ctrl_lat = proj.to_lonlat(ctrl_x, ctrl_y)

    sightings: list[Sighting] = []
    attempts = 0
    res = resolution_m
    while len(sightings) < n_sightings and attempts < 5000:
        attempts += 1
        dx = cx + rng.uniform(-half * 0.9, half * 0.9)
        dy = cy + rng.uniform(-half * 0.9, half * 0.9)
        dr, dc = grid.world_to_pixel(dx, dy)
        if not grid.contains_pixel(dr, dc):
            continue
        # Require some horizontal separation from the controller.
        if np.hypot(dx - ctrl_x, dy - ctrl_y) < extent_m * 0.2:
            continue
        drone_z = float(grid.data[dr, dc]) + drone_altitude_agl
        if not _los_clear(grid.data, res, cr, cc, ctrl_eye, dr, dc, drone_z):
            continue
        dlon, dlat = proj.to_lonlat(dx, dy)
        sightings.append(
            Sighting(
                lat=float(dlat),
                lon=float(dlon),
                altitude=drone_altitude_agl,  # AGL
                position_sigma_m=position_sigma_m,
                altitude_sigma_m=altitude_sigma_m,
                label=f"synthetic-{len(sightings)}",
            )
        )

    if len(sightings) < n_sightings:
        raise RuntimeError(
            f"only generated {len(sightings)}/{n_sightings} visible sightings; "
            "loosen constraints"
        )

    return SyntheticScenario(
        surface=grid,
        controller_lonlat=(float(ctrl_lon), float(ctrl_lat)),
        controller_xy=(ctrl_x, ctrl_y),
        sightings=sightings,
        projector=proj,
        antenna_height_m=antenna_height_m,
    )
