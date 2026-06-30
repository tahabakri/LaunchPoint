import numpy as np

from launchpoint.config import RangeModelConfig
from launchpoint.viewshed.range_model import RangeModel


def test_plateau_and_rolloff():
    rm = RangeModel(RangeModelConfig(max_range_m=12000.0))
    assert rm.weight(0.0) == 1.0
    assert rm.weight(1000.0) == 1.0  # within plateau
    mid = float(rm.weight(12000.0))
    assert 0.4 < mid < 0.6  # ~0.5 at the nominal max range
    assert float(rm.weight(rm.hard_cap_m + 1.0)) == 0.0


def test_monotonic_non_increasing():
    rm = RangeModel(RangeModelConfig(max_range_m=12000.0))
    d = np.linspace(0, rm.hard_cap_m, 200)
    w = rm.weight(d)
    assert np.all(np.diff(w) <= 1e-9)
