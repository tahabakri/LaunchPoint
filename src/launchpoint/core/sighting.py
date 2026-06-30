"""The Sighting: one fuzzy ground-position snapshot of the drone.

A sighting is *not* a precise fix. It is an estimate — position good to maybe a
few hundred metres, altitude a rough guess. The whole point of the tool is to
propagate that uncertainty (Phase 3 Monte Carlo) rather than fake precision, so
every sighting carries its own error model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sighting:
    """A single observed drone position with its uncertainty.

    Attributes
    ----------
    lat, lon:
        Estimated WGS84 horizontal position of the drone, decimal degrees.
    altitude:
        Estimated drone altitude in metres. Interpreted as AGL (above the
        surface under the drone) or ASL depending on ``Config.altitude_is_agl``.
    position_sigma_m:
        1-sigma horizontal position uncertainty, metres. The Monte Carlo draws
        horizontal offsets from a 2-D Gaussian with this standard deviation.
    altitude_sigma_m:
        1-sigma altitude uncertainty, metres.
    label:
        Optional human-readable identifier (for logs / UI later).
    """

    lat: float
    lon: float
    altitude: float
    position_sigma_m: float = 250.0
    altitude_sigma_m: float = 30.0
    label: str | None = None

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"lat out of range: {self.lat}")
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"lon out of range: {self.lon}")
        if self.position_sigma_m < 0 or self.altitude_sigma_m < 0:
            raise ValueError("uncertainties must be non-negative")

    def sample_positions(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``n`` plausible (east_offset_m, north_offset_m, altitude) triples.

        Horizontal offsets are in *local metres* relative to this sighting's
        nominal position (to be added after projecting to UTM); altitude is the
        sampled value in metres. Returns three 1-D arrays of length ``n``.
        With ``n == 1`` and zero sigma this collapses to the nominal value, so
        a deterministic single-sample run reproduces the nominal viewshed.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        east = rng.normal(0.0, self.position_sigma_m, size=n)
        north = rng.normal(0.0, self.position_sigma_m, size=n)
        alt = rng.normal(self.altitude, self.altitude_sigma_m, size=n)
        if n == 1:
            # Deterministic nominal sample regardless of rng draw.
            east[0] = 0.0
            north[0] = 0.0
            alt[0] = self.altitude
        return east, north, alt
