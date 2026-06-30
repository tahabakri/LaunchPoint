from launchpoint.viewshed.curvature import curvature_drop


def test_curvature_drop_at_12km():
    # With the 4/3-Earth convention the drop at 12 km is ~8.5 m
    # (d^2 / (2 * 4/3 * R)); without refraction it would be ~11.3 m.
    drop = float(curvature_drop(12000.0, refraction=True))
    assert 8.0 < drop < 9.0


def test_no_refraction_is_larger_drop():
    with_r = float(curvature_drop(12000.0, refraction=True))
    # Disabling the correction entirely yields 0 (we model "no correction").
    assert float(curvature_drop(12000.0, refraction=False)) == 0.0
    assert with_r > 0.0
