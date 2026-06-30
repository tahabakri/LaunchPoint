import numpy as np

from launchpoint.synthetic import make_default_scenario
from launchpoint.viewshed.kernels import oracle_viewshed, r3_viewshed


def _observer_for_first_sighting(sc):
    s = sc.sightings[0]
    x, y = sc.projector.to_utm(s.lon, s.lat)
    r, c = sc.surface.world_to_pixel(x, y)
    obs_z = float(sc.surface.data[r, c]) + s.altitude
    return r, c, obs_z


def test_r3_matches_oracle():
    sc = make_default_scenario(n_sightings=4, seed=5)
    surf = np.ascontiguousarray(sc.surface.data, dtype=np.float64)
    r, c, obs_z = _observer_for_first_sighting(sc)
    res = sc.surface.res_x

    args = (surf, surf, r, c, obs_z, 1.5, res, 0.0, 1.0e12)
    v_r3 = r3_viewshed(*args)
    v_or = oracle_viewshed(*args)

    agree = np.mean(v_r3 == v_or)
    # R3 ray-casting under-samples far cells slightly; demand strong agreement.
    assert agree > 0.95, f"R3 vs oracle agreement only {agree:.3f}"


def test_reciprocity_controller_visible_from_drone():
    # We *constructed* clear LOS from controller to each drone; by reciprocity
    # the controller cell must be visible in each drone's viewshed.
    sc = make_default_scenario(n_sightings=4, seed=5)
    surf = np.ascontiguousarray(sc.surface.data, dtype=np.float64)
    res = sc.surface.res_x
    cr, cc = sc.controller_pixel()
    for s in sc.sightings:
        x, y = sc.projector.to_utm(s.lon, s.lat)
        r, c = sc.surface.world_to_pixel(x, y)
        obs_z = float(surf[r, c]) + s.altitude
        v = oracle_viewshed(surf, surf, r, c, obs_z, sc.antenna_height_m, res, 0.0, 1.0e12)
        assert v[cr, cc] == 1


def test_curvature_reduces_far_visibility():
    # Flat surface: with curvature, far low ground falls below the horizon.
    n = 400
    surf = np.zeros((n, n), dtype=np.float64)
    obs_r, obs_c, obs_z = n // 2, 0, 2.0  # low observer at the edge
    res = 30.0
    flat = r3_viewshed(surf, surf, obs_r, obs_c, obs_z, 0.5, res, 0.0, 1.0e12)
    from launchpoint.viewshed.curvature import inv_two_effective_radius

    curved = r3_viewshed(
        surf, surf, obs_r, obs_c, obs_z, 0.5, res,
        inv_two_effective_radius(True), 1.0e12,
    )
    # Curvature can only remove visibility on flat ground, never add it.
    assert curved.sum() < flat.sum()
