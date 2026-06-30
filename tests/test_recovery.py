"""The headline synthetic test: does the pipeline recover the planted controller?

This is the objective pass/fail the whole project is built around (Phase 0 goal,
satisfied once Phase 3 fusion exists). It also checks the honesty property: as
input uncertainty grows the recovered region should get *wider*, not *wrong*.
"""

import numpy as np

from launchpoint.config import Config, MonteCarloConfig
from launchpoint.pipeline import find_origin
from launchpoint.synthetic import make_default_scenario


def _config(samples=16):
    return Config(monte_carlo=MonteCarloConfig(samples_per_sighting=samples, seed=1))


def test_recovers_controller_region():
    sc = make_default_scenario(n_sightings=4, seed=5)
    est = find_origin(
        sc.sightings,
        config=_config(),
        occluder=sc.surface,
        projector=sc.projector,
    )
    prob = est.probability.data
    cr, cc = sc.controller_pixel()

    # The true controller cell must carry high probability...
    ctrl_p = prob[cr, cc]
    assert ctrl_p > 0.5, f"controller probability too low: {ctrl_p:.3f}"

    # ...and the peak estimate must be reasonably near the true controller.
    px, py = est.argmax_lonlat()
    ex, ey = sc.projector.to_utm(px, py)
    err = np.hypot(ex - sc.controller_xy[0], ey - sc.controller_xy[1])
    assert err < 1500.0, f"argmax {err:.0f} m from true controller"

    # The controller falls inside the 50% highest-density region.
    mask = est.credible_mask(0.5).data
    assert mask[cr, cc] == 1.0


def test_degrades_gracefully_with_uncertainty():
    sc = make_default_scenario(n_sightings=4, seed=5)

    tight = find_origin(sc.sightings, config=_config(), occluder=sc.surface,
                        projector=sc.projector)

    # Inflate position uncertainty a lot.
    fuzzy_sightings = []
    for s in sc.sightings:
        s.position_sigma_m = 800.0
        s.altitude_sigma_m = 80.0
        fuzzy_sightings.append(s)
    fuzzy = find_origin(fuzzy_sightings, config=_config(), occluder=sc.surface,
                        projector=sc.projector)

    # Wider, not wrong: the high-probability region should grow in area.
    area_tight = (tight.credible_mask(0.5).data == 1.0).sum()
    area_fuzzy = (fuzzy.credible_mask(0.5).data == 1.0).sum()
    assert area_fuzzy >= area_tight

    # And the true controller should still be inside the (wider) region.
    cr, cc = sc.controller_pixel()
    assert fuzzy.credible_mask(0.7).data[cr, cc] == 1.0
