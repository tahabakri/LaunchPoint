"""Phase 4: enrichment changes the result in the expected direction (offline)."""

import numpy as np

from launchpoint.config import CanopyConfig, Config, MonteCarloConfig
from launchpoint.data.bare_earth import (
    apply_ground_floor,
    derive_bare_earth,
    launch_feasibility_weight,
)
from launchpoint.pipeline import find_origin
from launchpoint.synthetic import make_default_scenario


def _config():
    return Config(monte_carlo=MonteCarloConfig(samples_per_sighting=12, seed=1))


def test_bare_earth_lowers_target_under_canopy():
    sc = make_default_scenario(seed=5)
    dsm = sc.surface
    canopy = dsm.like(fill=0.0)
    # Put 25 m canopy over a block of cells.
    canopy.data[10:20, 10:20] = 25.0
    bare = derive_bare_earth(dsm, canopy, None)
    # Bare earth dropped by the canopy height exactly there...
    assert np.allclose(bare.data[10:20, 10:20], dsm.data[10:20, 10:20] - 25.0)
    # ...and equals the DSM elsewhere.
    assert np.allclose(bare.data[0, 0], dsm.data[0, 0])


def test_ground_floor_zeros_low_canopy_but_keeps_forest():
    sc = make_default_scenario(seed=5)
    canopy = sc.surface.like(fill=0.0)
    canopy.data[:] = 2.5          # ETH-style non-forest bias everywhere
    canopy.data[10:20, 10:20] = 22.0  # real forest block
    canopy.data[0, 0] = np.nan    # no-data must stay no-data

    floored = apply_ground_floor(canopy, 3.0)

    assert floored.data[30, 30] == 0.0                 # bias cleared to bare ground
    assert floored.data[15, 15] == 22.0                # forest untouched
    assert np.isnan(floored.data[0, 0])                # NaN preserved
    # A zero floor (Meta default) is a no-op passthrough.
    assert apply_ground_floor(canopy, 0.0) is canopy


def test_ground_floor_prevents_open_ground_burial():
    # The reported bug, isolated: a low-canopy baseline (ETH bias) subtracted from
    # the DSM sinks the antenna's ground below the surrounding occluder surface, so
    # grazing rays hit neighbouring ground and the cell is self-occluded. Restoring
    # the ground (what the floor does) recovers visibility. Flat occluder + a low,
    # distant observer makes the geometry unambiguous.
    from launchpoint.viewshed.kernels import r3_viewshed

    n, res = 200, 30.0
    occ = np.zeros((n, n), dtype=np.float64)  # flat true ground at 0 m
    obs_r, obs_c, obs_z = n // 2, 0, 3.0      # low observer at the west edge
    tgt_r, tgt_c = n // 2, n - 1              # far east target on the same row
    antenna = 1.5

    at_surface = np.zeros_like(occ)           # bare earth == DSM (Meta / floored)
    sunk = np.full_like(occ, -3.0)            # bare earth sunk 3 m (ETH bias)

    vis_surface = r3_viewshed(occ, at_surface, obs_r, obs_c, obs_z, antenna, res, 0.0, 1e12)
    vis_sunk = r3_viewshed(occ, sunk, obs_r, obs_c, obs_z, antenna, res, 0.0, 1e12)

    assert vis_surface[tgt_r, tgt_c] == 1  # antenna above the flat ground: visible
    assert vis_sunk[tgt_r, tgt_c] == 0     # buried below it: self-occluded, lost
    # Fewer cells overall survive when the whole surface is sunk.
    assert vis_sunk.sum() < vis_surface.sum()


def test_effective_ground_floor_is_source_specific():
    cfg = CanopyConfig()
    assert cfg.effective_ground_floor_m("eth") == 3.0
    assert cfg.effective_ground_floor_m("meta") == 0.0
    # An explicit override wins for every source.
    assert CanopyConfig(ground_floor_m=1.5).effective_ground_floor_m("eth") == 1.5


def test_launch_weight_suppresses_canopy_cells_in_heatmap():
    sc = make_default_scenario(n_sightings=4, seed=5)
    cfg = _config()

    # Canopy covering the true controller's neighbourhood.
    canopy = sc.surface.like(fill=0.0)
    cr, cc = sc.controller_pixel()
    canopy.data[max(0, cr - 3):cr + 3, max(0, cc - 3):cc + 3] = 30.0
    weight = launch_feasibility_weight(canopy, sc.surface, cfg.canopy)

    base = find_origin(sc.sightings, config=cfg, occluder=sc.surface,
                       projector=sc.projector)
    weighted = find_origin(sc.sightings, config=cfg, occluder=sc.surface,
                           projector=sc.projector, launch_weight=weight)

    # The launch penalty must reduce probability where canopy is tall.
    assert weighted.probability.data[cr, cc] < base.probability.data[cr, cc]
    # Open cells (no canopy) are unchanged.
    assert np.isclose(weighted.probability.data[0, 0], base.probability.data[0, 0])
