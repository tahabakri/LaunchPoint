"""Phase 4: enrichment changes the result in the expected direction (offline)."""

import numpy as np

from launchpoint.config import Config, MonteCarloConfig
from launchpoint.data.bare_earth import derive_bare_earth, launch_feasibility_weight
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
