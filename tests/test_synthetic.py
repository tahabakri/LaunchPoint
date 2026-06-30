import numpy as np

from launchpoint.synthetic import make_default_scenario
from launchpoint.synthetic.scenario import _los_clear


def test_scenario_generates_visible_sightings():
    sc = make_default_scenario(n_sightings=4)
    assert len(sc.sightings) == 4
    # Controller pixel is inside the grid.
    r, c = sc.controller_pixel()
    assert sc.surface.contains_pixel(r, c)


def test_generated_sightings_have_clear_los_from_controller():
    sc = make_default_scenario(n_sightings=4, seed=3)
    cr, cc = sc.controller_pixel()
    ctrl_eye = float(sc.surface.data[cr, cc]) + sc.antenna_height_m
    surf = sc.surface.data
    for s in sc.sightings:
        x, y = sc.projector.to_utm(s.lon, s.lat)
        dr, dc = sc.surface.world_to_pixel(x, y)
        drone_z = float(surf[dr, dc]) + s.altitude
        assert _los_clear(surf, sc.surface.res_x, cr, cc, ctrl_eye, dr, dc, drone_z)


def test_scenario_is_deterministic():
    a = make_default_scenario(seed=11)
    b = make_default_scenario(seed=11)
    assert a.controller_xy == b.controller_xy
    assert np.array_equal(a.surface.data, b.surface.data)
