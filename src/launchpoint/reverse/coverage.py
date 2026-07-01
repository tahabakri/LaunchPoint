"""Single-candidate launch-coverage score: the reverse of ``fusion/montecarlo.py``.

Forward direction (``monte_carlo_sighting``): the drone/sighting is the single
point that *varies* between calls (``ViewshedInput.observer_xy``/``observer_z``),
and the controller's antenna height is a *uniform* offset applied to ``ground``
at every other cell (``target_height``) — every grid cell is a candidate
controller stance, all at the same standing height.

Reverse direction (this module): the roles swap. The *candidate launch point*
is now the single point that varies (`observer_xy`/`observer_z` = its own
ground elevation + antenna height), and the *flight altitude* (AGL, uniform
across the flight zone exactly like antenna height is uniform across the grid
today) becomes the uniform `target_height` applied to `ground` everywhere.
``ViewshedInput``/``compute_viewshed``/``RangeModel`` are reused completely
unmodified — only the semantic labels of the two roles swap.
"""

from __future__ import annotations

import numpy as np

from launchpoint.config import Config
from launchpoint.core.geo import Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.core.target_zone import TargetZone
from launchpoint.viewshed.core import ViewshedInput, compute_viewshed
from launchpoint.viewshed.range_model import RangeModel


def zone_cell_mask(grid: RasterGrid, projector: Projector, zone: TargetZone) -> np.ndarray:
    """Boolean mask (grid-shaped) of cells whose centre lies within the zone disk."""
    cx, cy = projector.to_utm(zone.center_lon, zone.center_lat)
    rows, cols = grid.shape
    rr = np.arange(rows)
    cc = np.arange(cols)
    # Cell-centre world coordinates, vectorised like viewshed/core.py's _distance_grid.
    xs = grid.transform.c + (cc + 0.5) * grid.transform.a
    ys = grid.transform.f + (rr + 0.5) * grid.transform.e
    dist = np.hypot(ys[:, None] - cy, xs[None, :] - cx)
    return dist <= zone.radius_m


def candidate_coverage(
    candidate_xy: tuple[float, float],
    zone_mask: np.ndarray,
    occluder: RasterGrid,
    flight_altitude_agl_m: float,
    config: Config,
    range_model: RangeModel,
    ground: RasterGrid | None = None,
    coverage_threshold: float = 0.5,
) -> tuple[float, float]:
    """Run one viewshed from ``candidate_xy`` and reduce it over the flight zone.

    Role assignment (the crux of the whole reverse feature):

    * ``observer_xy`` = the candidate launch point (varies per call — the same
      "outer loop" role a sighting's position plays in ``monte_carlo_sighting``).
    * ``observer_z``  = the candidate's own ground elevation + ``antenna_height_m``
      (the operator stands on the earth at the candidate, same height
      convention used for the controller today).
    * ``target_height`` = ``flight_altitude_agl_m``, applied uniformly to
      ``ground`` at *every* cell — this plays the exact role
      ``antenna_height_m`` plays in the forward direction, just swapped onto
      the drone's side of the link.

    Returns ``(min_visibility, frac_covered)``: the strict worst-case cell
    value within the zone (drives ranking, see ``search.py``), and the
    fraction of zone cells at or above ``coverage_threshold`` (a friendlier,
    non-ranking secondary metric).
    """
    r, c = occluder.world_to_pixel(*candidate_xy)
    if not occluder.contains_pixel(r, c):
        return 0.0, 0.0

    ground_grid = ground if ground is not None else occluder
    base = ground_grid.data[r, c]
    if not np.isfinite(base):
        base = occluder.data[r, c]
    if not np.isfinite(base):
        return 0.0, 0.0
    observer_z = float(base) + config.antenna_height_m

    vi = ViewshedInput(
        occluder=occluder,
        observer_xy=candidate_xy,
        observer_z=observer_z,
        target_height=flight_altitude_agl_m,
        ground=ground,
        refraction=True,
    )
    contrib = compute_viewshed(vi, range_model, prefer_gpu=config.prefer_gpu).data

    zone_vals = contrib[zone_mask]
    if zone_vals.size == 0:
        return 0.0, 0.0
    min_visibility = float(np.min(zone_vals))
    frac_covered = float(np.mean(zone_vals >= coverage_threshold))
    return min_visibility, frac_covered


def apply_launch_weight(
    score: float, launch_weight: RasterGrid | None, occluder: RasterGrid, candidate_xy: tuple[float, float]
) -> float:
    """Multiply ``score`` by the candidate's own launch-feasibility weight.

    Applied once, at the candidate's own cell only (where the operator would
    stand) — analogous to how ``fuse_sightings`` multiplies ``launch_weight``
    onto the final combined raster rather than into each per-sample viewshed.
    """
    if launch_weight is None:
        return score
    r, c = occluder.world_to_pixel(*candidate_xy)
    if not launch_weight.contains_pixel(r, c):
        return score
    w = launch_weight.data[r, c]
    return score * float(w) if np.isfinite(w) else score
