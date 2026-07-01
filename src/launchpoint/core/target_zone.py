"""The TargetZone: a user-drawn circular flight area to find launch coverage for.

Unlike a ``Sighting`` (a fuzzy *observation* of where a drone already was), a
``TargetZone`` is a *design intent*: "I plan to fly somewhere inside this
circle." There is no position/altitude uncertainty to propagate, so — unlike
``Sighting`` — it carries no sigma fields and is never Monte Carlo sampled.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetZone:
    """A circular flight area the operator intends to fly a drone within.

    Attributes
    ----------
    center_lat, center_lon:
        WGS84 centre of the intended flight area, decimal degrees.
    radius_m:
        Horizontal radius of the flight area, metres.
    flight_altitude_m:
        Intended flight altitude. Interpreted as AGL (above the surface under
        the drone) or ASL depending on ``Config.altitude_is_agl`` — the same
        flag ``Sighting.altitude`` already depends on, so a flight zone has no
        per-zone override.
    label:
        Optional human-readable identifier (for logs / UI later).
    """

    center_lat: float
    center_lon: float
    radius_m: float
    flight_altitude_m: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not (-90.0 <= self.center_lat <= 90.0):
            raise ValueError(f"center_lat out of range: {self.center_lat}")
        if not (-180.0 <= self.center_lon <= 180.0):
            raise ValueError(f"center_lon out of range: {self.center_lon}")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.flight_altitude_m < 0:
            raise ValueError("flight_altitude_m must be non-negative")
