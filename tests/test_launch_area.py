"""The reverse-feature counterpart to test_recovery.py.

Does the launch-coverage search recover a planted good candidate (clear LOS to
the whole flight zone) and avoid a planted bad candidate (blocked by the
synthetic ridge)?
"""

import numpy as np
import pytest

from launchpoint.config import Config, MonteCarloConfig, RangeModelConfig
from launchpoint.core.target_zone import TargetZone
from launchpoint.reverse.coverage import candidate_coverage, zone_cell_mask
from launchpoint.reverse.pipeline import find_launch_area
from launchpoint.reverse.search import LaunchSearchConfig
from launchpoint.synthetic import make_launch_scenario
from launchpoint.viewshed.range_model import RangeModel


def _config(**overrides):
    return Config(monte_carlo=MonteCarloConfig(samples_per_sighting=1, seed=1), **overrides)


def _search(**overrides):
    defaults = dict(
        candidate_coarse_stride_m=100.0,
        candidate_fine_stride_m=25.0,
        surface_resolution_m=30.0,
    )
    defaults.update(overrides)
    return LaunchSearchConfig(**defaults)


@pytest.fixture(scope="module")
def scenario():
    return make_launch_scenario(seed=11)


@pytest.fixture(scope="module")
def search_result(scenario):
    return find_launch_area(
        scenario.zone, config=_config(), search=_search(),
        occluder=scenario.surface, projector=scenario.projector,
    )


def test_recovers_good_launch_point(search_result):
    """The search must find *some* launch point with high coverage (it need
    not be exactly the planted good candidate — other points on the same open
    side of the ridge may score just as well or better)."""
    assert search_result.best_score > 0.5
    assert search_result.best_min_visibility > 0.5


def test_good_candidate_directly_scores_high(scenario):
    """Ground-truth check: evaluating the algorithm's own scoring function
    directly at the planted good candidate (clear LOS to the whole zone by
    construction) should report high coverage."""
    config = _config()
    range_model = RangeModel(config.range_model)
    mask = zone_cell_mask(scenario.surface, scenario.projector, scenario.zone)
    min_vis, frac = candidate_coverage(
        scenario.good_launch_xy, mask, scenario.surface, scenario.zone.flight_altitude_m,
        config, range_model,
    )
    assert min_vis > 0.5, f"good candidate's min_visibility too low: {min_vis:.3f}"
    assert frac > 0.9, f"good candidate should cover nearly the whole zone: {frac:.3f}"


def test_excludes_occluded_launch_point(scenario, search_result):
    """The bad (ridge-occluded) candidate must not win the search."""
    best_xy = scenario.projector.to_utm(*search_result.best_lonlat())
    err_to_bad = np.hypot(best_xy[0] - scenario.bad_launch_xy[0], best_xy[1] - scenario.bad_launch_xy[1])
    assert err_to_bad > 250.0, "the ridge-occluded candidate should not be the winner"


def test_bad_candidate_directly_scores_low(scenario):
    """Ground-truth check: the planted bad candidate is blocked from at least
    part of the zone by construction, so its min_visibility should be low."""
    config = _config()
    range_model = RangeModel(config.range_model)
    mask = zone_cell_mask(scenario.surface, scenario.projector, scenario.zone)
    min_vis, _ = candidate_coverage(
        scenario.bad_launch_xy, mask, scenario.surface, scenario.zone.flight_altitude_m,
        config, range_model,
    )
    assert min_vis < 0.5, f"occluded candidate's min_visibility too high: {min_vis:.3f}"


def test_degrades_gracefully_with_lower_flight_altitude(scenario, search_result):
    """A lower flight altitude is strictly harder to see over terrain, so the
    best achievable score should not improve as altitude drops (honesty check,
    mirrors test_recovery.py's uncertainty-widening assertion in spirit)."""
    low_zone = TargetZone(
        center_lat=scenario.zone.center_lat, center_lon=scenario.zone.center_lon,
        radius_m=scenario.zone.radius_m, flight_altitude_m=5.0,
    )
    low = find_launch_area(
        low_zone, config=_config(), search=_search(),
        occluder=scenario.surface, projector=scenario.projector,
    )

    assert low.best_score <= search_result.best_score + 1e-6


def test_distance_prefilter_excludes_far_candidates(scenario):
    """With a tiny hard-cap range, most of the coarse candidate grid must be
    skipped by the cheap distance pre-filter before any viewshed runs.

    ``hard_cap_m = max_range_m + 4*softness_m``, so both must shrink together
    (softness_m does not scale down with max_range_m on its own) to actually
    exercise the pre-filter over this scenario's 6 km synthetic grid.
    """
    tight_config = _config(range_model=RangeModelConfig(max_range_m=200.0, softness_m=50.0))
    result = find_launch_area(
        scenario.zone, config=tight_config, search=_search(),
        occluder=scenario.surface, projector=scenario.projector,
    )
    assert result.n_candidates_prefiltered > 0
    assert result.n_candidates_prefiltered > result.n_candidates_evaluated


def test_reverse_requires_agl_altitude():
    zone = TargetZone(center_lat=47.37, center_lon=8.55, radius_m=200.0, flight_altitude_m=80.0)
    config = _config(altitude_is_agl=False)
    with pytest.raises(ValueError):
        find_launch_area(zone, config=config)


def test_target_zone_validation():
    with pytest.raises(ValueError):
        TargetZone(center_lat=95.0, center_lon=8.55, radius_m=100.0, flight_altitude_m=50.0)
    with pytest.raises(ValueError):
        TargetZone(center_lat=47.37, center_lon=200.0, radius_m=100.0, flight_altitude_m=50.0)
    with pytest.raises(ValueError):
        TargetZone(center_lat=47.37, center_lon=8.55, radius_m=-1.0, flight_altitude_m=50.0)
    with pytest.raises(ValueError):
        TargetZone(center_lat=47.37, center_lon=8.55, radius_m=100.0, flight_altitude_m=-1.0)
