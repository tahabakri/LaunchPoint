"""Derive a bare-earth surface and a launch-feasibility weight.

No third-party bare-earth DEM is fetched. Following FABDEM's own idea (ML-correct
GLO-30 by subtracting vegetation/building bias) but using the layers we already
have, we compute::

    bare_earth = DSM - canopy_height - building_height   (clamped at a floor)

This keeps occlusion (DSM, untouched) and target-height correction (bare earth)
consistent — both come from the same two rasters — and avoids importing any
non-commercial data. It is a blunter correction than FABDEM's validated model
(noted in the limitations section), but well matched to the 30 m / Monte-Carlo
error budget.

The canopy layer also drives a *soft* launch-feasibility weight: you can't
realistically take off or keep a drone in view under closed canopy, but
operators do stand at forest edges and clearings — so the weight has a floor.
"""

from __future__ import annotations

import numpy as np

from launchpoint.config import CanopyConfig
from launchpoint.core.grid import RasterGrid


def apply_ground_floor(
    canopy_height: RasterGrid | None, floor_m: float
) -> RasterGrid | None:
    """Zero out canopy at/below ``floor_m`` (treat it as bare ground).

    Cancels the ETH 10 m product's positive bias over open / low-vegetation
    ground so it no longer sinks the derived bare-earth surface (see
    ``CanopyConfig.ground_floor_m``). NaN (no-data) cells are preserved as NaN so
    the "missing canopy -> 0 downstream" contract is unchanged. A ``floor_m`` of
    0 is a no-op, so the near-zero Meta product passes through untouched.
    """
    if canopy_height is None or floor_m <= 0.0:
        return canopy_height
    data = canopy_height.data
    floored = np.where(np.isfinite(data) & (data <= floor_m), 0.0, data)
    return canopy_height.copy_with(floored.astype(np.float32))


def derive_bare_earth(
    dsm: RasterGrid,
    canopy_height: RasterGrid | None,
    building_height: RasterGrid | None,
    min_clamp_below_dsm: float = 0.0,
) -> RasterGrid:
    """``bare_earth = DSM - canopy - building``, clamped so it never exceeds DSM
    and never drops more than is physically sensible below it.

    ``canopy_height`` / ``building_height`` may be None (treated as 0), in which
    case bare earth equals the DSM — the Phase 2 placeholder behaviour.
    """
    dsm_a = dsm.data.astype(np.float64)
    canopy = _aligned(canopy_height, dsm)
    building = _aligned(building_height, dsm)

    # A coarse cell is occluded by the *tallest* object in it, so canopy and
    # building heights were resampled with a max rule; subtract the larger of
    # the two rather than both (a tree growing against a wall is one obstacle).
    object_h = np.maximum(canopy, building)
    bare = dsm_a - object_h

    # Never let noise push bare earth above the DSM; allow a small floor below.
    bare = np.minimum(bare, dsm_a)
    bare = np.maximum(bare, dsm_a - np.maximum(object_h, 0.0) - min_clamp_below_dsm)
    return dsm.copy_with(bare.astype(np.float32))


def launch_feasibility_weight(
    canopy_height: RasterGrid | None,
    reference: RasterGrid,
    cfg: CanopyConfig,
) -> RasterGrid:
    """Per-cell weight in [min_launch_weight, 1] from canopy height.

    1.0 in the open; ramps down linearly between ``launch_penalty_height_m`` and
    ``launch_penalty_full_m`` to ``min_launch_weight`` under tall closed canopy.
    """
    if canopy_height is None:
        return reference.copy_with(np.ones(reference.shape, dtype=np.float32))

    ch = _aligned(canopy_height, reference)
    lo, hi = cfg.launch_penalty_height_m, cfg.launch_penalty_full_m
    frac = np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    weight = 1.0 - (1.0 - cfg.min_launch_weight) * frac
    return reference.copy_with(weight.astype(np.float32))


def _aligned(layer: RasterGrid | None, reference: RasterGrid) -> np.ndarray:
    """Return ``layer``'s data aligned to ``reference`` shape, NaN->0."""
    if layer is None:
        return np.zeros(reference.shape, dtype=np.float64)
    if layer.shape != reference.shape:
        raise ValueError(
            f"layer shape {layer.shape} != reference {reference.shape}; "
            "resample onto the reference grid before deriving bare earth"
        )
    return np.where(np.isfinite(layer.data), layer.data, 0.0).astype(np.float64)
