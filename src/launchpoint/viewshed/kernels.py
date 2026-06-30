"""Numba-accelerated viewshed kernels.

Two implementations of the same physics:

* ``r3_viewshed`` — radial-sweep (R3-style) algorithm. Casts a ray to every
  perimeter cell, marches outward keeping the running maximum terrain elevation
  angle, and marks a cell visible when the target point there rises above that
  angle. O(N^1.5) — the workhorse for Monte-Carlo runs.
* ``oracle_viewshed`` — brute-force per-target line of sight. O(N * path). Slow
  but obviously correct; used to validate ``r3_viewshed`` in tests.

Both share the convention:
* ``occ``      : occluding surface (DSM), absolute metres.
* ``ground``   : target-height base surface (bare earth), absolute metres.
* ``obs_z``    : observer (drone) elevation, absolute metres.
* ``target_h`` : antenna height added to ``ground`` at each candidate cell.
* ``inv_two_reff`` : 1/(2*R_eff); multiply by d^2 for the curvature drop.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

_EPS = 1e-9


@njit(cache=True, inline="always")
def _cell_distance(dr, dc, res):
    return math.sqrt((dr * res) * (dr * res) + (dc * res) * (dc * res))


@njit(cache=True)
def _cast_ray(occ, ground, obs_r, obs_c, obs_z, target_h, res, inv_two_reff,
              max_range, er, ec, out):
    rows = occ.shape[0]
    cols = occ.shape[1]
    dr = er - obs_r
    dc = ec - obs_c
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return
    max_ang = -1.0e30
    for k in range(1, steps + 1):
        rf = obs_r + dr * k / steps
        cf = obs_c + dc * k / steps
        ri = int(round(rf))
        ci = int(round(cf))
        if ri < 0 or ri >= rows or ci < 0 or ci >= cols:
            continue
        d = _cell_distance(ri - obs_r, ci - obs_c, res)
        if d <= 0.0:
            continue
        if d > max_range:
            break
        drop = d * d * inv_two_reff
        terr_ang = (occ[ri, ci] - drop - obs_z) / d
        targ = ground[ri, ci] + target_h
        targ_ang = (targ - drop - obs_z) / d
        if targ_ang >= max_ang - _EPS:
            out[ri, ci] = 1
        if terr_ang > max_ang:
            max_ang = terr_ang


@njit(cache=True, parallel=True)
def r3_viewshed(occ, ground, obs_r, obs_c, obs_z, target_h, res, inv_two_reff,
                max_range):
    rows = occ.shape[0]
    cols = occ.shape[1]
    out = np.zeros((rows, cols), dtype=np.uint8)
    if 0 <= obs_r < rows and 0 <= obs_c < cols:
        out[obs_r, obs_c] = 1
    # Rays to the top and bottom edges (parallel over columns).
    for ec in prange(cols):
        _cast_ray(occ, ground, obs_r, obs_c, obs_z, target_h, res, inv_two_reff,
                  max_range, 0, ec, out)
        _cast_ray(occ, ground, obs_r, obs_c, obs_z, target_h, res, inv_two_reff,
                  max_range, rows - 1, ec, out)
    # Rays to the left and right edges (parallel over rows).
    for er in prange(rows):
        _cast_ray(occ, ground, obs_r, obs_c, obs_z, target_h, res, inv_two_reff,
                  max_range, er, 0, out)
        _cast_ray(occ, ground, obs_r, obs_c, obs_z, target_h, res, inv_two_reff,
                  max_range, er, cols - 1, out)
    return out


@njit(cache=True, parallel=True)
def oracle_viewshed(occ, ground, obs_r, obs_c, obs_z, target_h, res, inv_two_reff,
                    max_range):
    """Brute-force: independent LOS test per candidate cell."""
    rows = occ.shape[0]
    cols = occ.shape[1]
    out = np.zeros((rows, cols), dtype=np.uint8)
    for tr in prange(rows):
        for tc in range(cols):
            d_target = _cell_distance(tr - obs_r, tc - obs_c, res)
            if d_target > max_range:
                continue
            if d_target <= 0.0:
                out[tr, tc] = 1
                continue
            targ = ground[tr, tc] + target_h
            drop_t = d_target * d_target * inv_two_reff
            targ_ang = (targ - drop_t - obs_z) / d_target
            steps = max(abs(tr - obs_r), abs(tc - obs_c))
            blocked = False
            for k in range(1, steps):  # strictly between observer and target
                rf = obs_r + (tr - obs_r) * k / steps
                cf = obs_c + (tc - obs_c) * k / steps
                ri = int(round(rf))
                ci = int(round(cf))
                d = _cell_distance(ri - obs_r, ci - obs_c, res)
                if d <= 0.0:
                    continue
                drop = d * d * inv_two_reff
                terr_ang = (occ[ri, ci] - drop - obs_z) / d
                if terr_ang > targ_ang + _EPS:
                    blocked = True
                    break
            if not blocked:
                out[tr, tc] = 1
    return out
