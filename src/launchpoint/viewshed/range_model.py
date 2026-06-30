"""Soft range model.

Real control range is not a hard 12 km disk — it collapses with interference,
antenna angle, obstructions and altitude, and there is no crisp cutoff. We model
the probability that a link can be held at horizontal distance ``d`` as a flat
plateau near the observer followed by a smooth logistic roll-off centred near
the nominal max range. Beyond a hard cap (a small multiple of max range) it is
clamped to zero so the AOI stays finite.
"""

from __future__ import annotations

import numpy as np

from launchpoint.config import RangeModelConfig


class RangeModel:
    def __init__(self, config: RangeModelConfig | None = None):
        self.cfg = config or RangeModelConfig()

    @property
    def hard_cap_m(self) -> float:
        # Where the logistic has decayed to a negligible weight.
        return self.cfg.max_range_m + 4.0 * self.cfg.softness_m

    def weight(self, distance_m: np.ndarray | float) -> np.ndarray:
        """Probability weight in [0, 1] for one or many distances (metres)."""
        d = np.asarray(distance_m, dtype=np.float64)
        plateau = self.cfg.plateau_frac * self.cfg.max_range_m
        # Logistic centred at max_range, width = softness.
        # w = 1 / (1 + exp((d - max_range) / softness)) for d > plateau,
        # and ~1 below the plateau.
        w = 1.0 / (1.0 + np.exp((d - self.cfg.max_range_m) / self.cfg.softness_m))
        w = np.where(d <= plateau, 1.0, w)
        w = np.where(d > self.hard_cap_m, 0.0, w)
        return w
