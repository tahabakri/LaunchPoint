"""High-level single-observer viewshed.

Wraps the numba kernels with georeferencing, observer-height handling, curvature
choice and the soft range model, returning a per-cell *contribution* in [0, 1]:
``visibility (0/1) * range_weight(distance)``. This is exactly what the
multi-sighting fusion step accumulates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from launchpoint.core.grid import RasterGrid
from launchpoint.viewshed.curvature import inv_two_effective_radius
from launchpoint.viewshed.kernels import oracle_viewshed, r3_viewshed
from launchpoint.viewshed.range_model import RangeModel


@dataclass
class ViewshedInput:
    """Everything one viewshed needs.

    Parameters
    ----------
    occluder:
        The DSM (occluding surface), absolute metres. NaN cells are treated as
        very low ground (non-occluding) after fill.
    observer_xy:
        Drone horizontal position (easting, northing) in the occluder CRS.
    observer_z:
        Drone elevation, absolute metres (ASL).
    target_height:
        Antenna height (metres) added to the ground surface at each candidate.
    ground:
        Target-height base surface (bare earth), absolute metres. Defaults to
        ``occluder`` when None (Phase 2 placeholder, before bare earth exists).
    refraction:
        Apply 4/3-Earth curvature+refraction correction.
    """

    occluder: RasterGrid
    observer_xy: tuple[float, float]
    observer_z: float
    target_height: float
    ground: RasterGrid | None = None
    refraction: bool = True


def _prepare_arrays(vi: ViewshedInput):
    occ = np.ascontiguousarray(vi.occluder.data, dtype=np.float64)
    ground_grid = vi.ground if vi.ground is not None else vi.occluder
    ground = np.ascontiguousarray(ground_grid.data, dtype=np.float64)
    # Replace NaN occluder cells with a very low value so they never block;
    # NaN ground cells become equally low so a target there is simply unlit.
    very_low = np.nanmin(occ) - 1000.0 if np.any(np.isfinite(occ)) else -1000.0
    occ = np.where(np.isfinite(occ), occ, very_low)
    ground = np.where(np.isfinite(ground), ground, very_low)
    return occ, ground


def _distance_grid(grid: RasterGrid, obs_r: int, obs_c: int) -> np.ndarray:
    rows, cols = grid.shape
    rr = (np.arange(rows) - obs_r) * grid.res_y
    cc = (np.arange(cols) - obs_c) * grid.res_x
    return np.hypot(rr[:, None], cc[None, :])


def _run(
    vi: ViewshedInput,
    range_model: RangeModel | None,
    kernel,
) -> RasterGrid:
    occ, ground = _prepare_arrays(vi)
    grid = vi.occluder
    obs_r, obs_c = grid.world_to_pixel(*vi.observer_xy)
    inv_two_reff = inv_two_effective_radius(vi.refraction)

    rm = range_model or RangeModel()
    max_range = rm.hard_cap_m

    vis = kernel(
        occ,
        ground,
        int(obs_r),
        int(obs_c),
        float(vi.observer_z),
        float(vi.target_height),
        float(grid.res_x),
        float(inv_two_reff),
        float(max_range),
    ).astype(np.float32)

    dist = _distance_grid(grid, obs_r, obs_c)
    weight = rm.weight(dist).astype(np.float32)
    return grid.copy_with(vis * weight)


def compute_viewshed(
    vi: ViewshedInput,
    range_model: RangeModel | None = None,
    prefer_gpu: bool = False,
) -> RasterGrid:
    """Fast radial-sweep viewshed contribution surface in [0, 1].

    With ``prefer_gpu`` the CUDA kernel is used when a device is present,
    otherwise the parallel CPU kernel (identical numerics).
    """
    if prefer_gpu:
        from launchpoint.viewshed.gpu import r3_viewshed_best

        def kernel(*args):
            return r3_viewshed_best(*args, prefer_gpu=True)

        return _run(vi, range_model, kernel)
    return _run(vi, range_model, r3_viewshed)


def compute_viewshed_oracle(
    vi: ViewshedInput, range_model: RangeModel | None = None
) -> RasterGrid:
    """Brute-force reference viewshed (slow; for validation)."""
    return _run(vi, range_model, oracle_viewshed)
