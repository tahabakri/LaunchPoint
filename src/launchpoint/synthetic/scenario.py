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
from launchpoint.core.target_zone import TargetZone


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


@dataclass
class LaunchScenario:
    """A fully-known test world for the reverse (launch-coverage) problem.

    Reuses the same ridge terrain (``_synthetic_surface``) and brute-force LOS
    oracle (``_los_clear``) as ``SyntheticScenario``. The ridge is a straight
    "wall" whose height depends only on the ``xx + yy`` coordinate (it is the
    same regardless of position along the ridge), so any straight path
    crossing ``xx + yy == 0`` passes near the ridge crest, and any path that
    stays on one side never does. The zone and ``good_launch_xy`` are planted
    on the same side (never crossing the ridge); ``bad_launch_xy`` is planted
    on the opposite side (its line of sight to the zone crosses the ridge).

    Attributes
    ----------
    surface:
        The DSM-equivalent occluding surface (RasterGrid, UTM, metres).
    zone:
        The planted flight zone.
    projector:
        WGS84 <-> scenario-UTM projector.
    good_launch_xy:
        True (easting, northing) of a candidate with clear LOS to the whole
        zone (zone centre + perimeter samples).
    bad_launch_xy:
        True (easting, northing) of a candidate whose LOS to at least part of
        the zone is blocked by the synthetic ridge.
    antenna_height_m:
        Operator antenna height used when generating LOS.
    """

    surface: RasterGrid
    zone: TargetZone
    projector: Projector
    good_launch_xy: tuple[float, float]
    bad_launch_xy: tuple[float, float]
    antenna_height_m: float


def make_launch_scenario(
    center_lon: float = 8.55,
    center_lat: float = 47.37,
    extent_m: float = 6000.0,
    resolution_m: float = 30.0,
    zone_radius_m: float = 300.0,
    flight_altitude_agl_m: float = 80.0,
    antenna_height_m: float = 1.5,
    seed: int = 11,
) -> LaunchScenario:
    """Build a reproducible synthetic launch-coverage scenario.

    Plants a flight-zone circle on one side of the synthetic ridge, then
    searches for a "good" candidate on the same side (verified via
    ``_los_clear`` to the zone centre and several perimeter samples) and a
    "bad" candidate on the opposite side (verified to be blocked to at least
    one of those same targets).
    """
    proj = Projector.for_point(center_lon, center_lat)
    cx, cy = proj.to_utm(center_lon, center_lat)

    half = extent_m / 2.0
    bbox = BBox(cx - half, cy - half, cx + half, cy + half)
    grid = RasterGrid.empty(bbox, resolution_m, proj.utm_crs, fill=0.0, dtype=np.float64)

    rows, cols = grid.shape
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    wx, wy = grid.transform * (jj + 0.5, ii + 0.5)
    lx = np.asarray(wx) - cx
    ly = np.asarray(wy) - cy
    grid.data = _synthetic_surface(lx, ly, seed)

    # Zone centred well off to one side of the NE-SW ridge (xx + yy = 0 is the
    # ridge crest line), so it sits in open terrain.
    zone_offset = extent_m * 0.22
    zone_x = cx + zone_offset
    zone_y = cy + zone_offset
    zr, zc = grid.world_to_pixel(zone_x, zone_y)
    zone_ground = float(grid.data[zr, zc])
    zone_lon, zone_lat = proj.to_lonlat(zone_x, zone_y)
    zone = TargetZone(
        center_lat=float(zone_lat),
        center_lon=float(zone_lon),
        radius_m=zone_radius_m,
        flight_altitude_m=flight_altitude_agl_m,
    )

    # LOS check targets: the zone centre plus eight perimeter samples, all at
    # the flight altitude — a candidate must see *all* of these to count as
    # "good," mirroring the strict min-visibility scoring rule under test.
    n_perimeter = 8
    targets: list[tuple[int, int, float]] = [(zr, zc, zone_ground + flight_altitude_agl_m)]
    for angle in np.linspace(0.0, 2 * np.pi, n_perimeter, endpoint=False):
        px = zone_x + zone_radius_m * np.cos(angle)
        py = zone_y + zone_radius_m * np.sin(angle)
        pr, pc = grid.world_to_pixel(px, py)
        pr = int(np.clip(pr, 0, rows - 1))
        pc = int(np.clip(pc, 0, cols - 1))
        pz = float(grid.data[pr, pc]) + flight_altitude_agl_m
        targets.append((pr, pc, pz))

    def _sees_all(cand_r: int, cand_c: int, cand_eye: float) -> bool:
        return all(
            _los_clear(grid.data, resolution_m, cand_r, cand_c, cand_eye, tr, tc, tz)
            for tr, tc, tz in targets
        )

    def _blocked_to_any(cand_r: int, cand_c: int, cand_eye: float) -> bool:
        return any(
            not _los_clear(grid.data, resolution_m, cand_r, cand_c, cand_eye, tr, tc, tz)
            for tr, tc, tz in targets
        )

    rng = np.random.default_rng(seed + 200)

    good_xy: tuple[float, float] | None = None
    attempts = 0
    while good_xy is None and attempts < 5000:
        attempts += 1
        gx = cx + rng.uniform(zone_offset * 0.3, zone_offset * 1.6)
        gy = cy + rng.uniform(zone_offset * 0.3, zone_offset * 1.6)
        gr, gc = grid.world_to_pixel(gx, gy)
        if not grid.contains_pixel(gr, gc):
            continue
        if np.hypot(gx - zone_x, gy - zone_y) < zone_radius_m * 1.5:
            continue  # keep the candidate outside the zone itself
        g_eye = float(grid.data[gr, gc]) + antenna_height_m
        if _sees_all(gr, gc, g_eye):
            good_xy = (gx, gy)

    if good_xy is None:
        raise RuntimeError("could not find a good launch candidate; loosen constraints")

    bad_xy: tuple[float, float] | None = None
    attempts = 0
    while bad_xy is None and attempts < 5000:
        attempts += 1
        bx = cx - rng.uniform(zone_offset * 0.3, zone_offset * 1.6)
        by = cy - rng.uniform(zone_offset * 0.3, zone_offset * 1.6)
        br, bc = grid.world_to_pixel(bx, by)
        if not grid.contains_pixel(br, bc):
            continue
        b_eye = float(grid.data[br, bc]) + antenna_height_m
        if _blocked_to_any(br, bc, b_eye):
            bad_xy = (bx, by)

    if bad_xy is None:
        raise RuntimeError("could not find a ridge-occluded bad launch candidate; loosen constraints")

    return LaunchScenario(
        surface=grid,
        zone=zone,
        projector=proj,
        good_launch_xy=good_xy,
        bad_launch_xy=bad_xy,
        antenna_height_m=antenna_height_m,
    )
