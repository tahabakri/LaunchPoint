"""Project-wide constants and tunable parameters.

Every magic number that affects the physics or the error budget lives here so it
can be cited in the README's limitations section and overridden per run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Physical constants -----------------------------------------------------

EARTH_RADIUS_M: float = 6_371_008.8
"""Mean Earth radius (IUGG), metres."""

REFRACTION_K: float = 4.0 / 3.0
"""Effective-Earth-radius factor for standard atmospheric refraction.

Light bends down slightly, so the geometric horizon is farther than the true
one. The classic surveying/radio convention models this by pretending the Earth
is 4/3 its real radius. At 12 km the curvature+refraction drop is ~9.5 m with
this factor (vs ~11.3 m with no refraction) — not negligible for long rays.
"""


# --- Default operational parameters -----------------------------------------

DEFAULT_MAX_RANGE_M: float = 12_000.0
"""Nominal maximum control range (e.g. a DJI Mini 3 Pro under good conditions)."""

DEFAULT_ANTENNA_HEIGHT_M: float = 1.5
"""Height of the operator's controller/antenna above the bare-earth surface.

~1.5 m is a handheld controller at chest height. Configurable for a tripod or
an elevated position.
"""

DEFAULT_DSM_RESOLUTION_M: float = 30.0
"""Native resolution of the Copernicus GLO-30 DSM, metres."""

DEFAULT_ANALYSIS_RESOLUTION_M: float = 10.0
"""Default grid resolution used by the model, metres."""

DEFAULT_FINE_RESOLUTION_M: float = 2.0
"""Target resolution for the fine refinement pass (Phase 5).

The canopy source is 1 m; 2 m keeps the fine patches tractable while still
resolving individual buildings/tree crowns near the observer.
"""


@dataclass
class RangeModelConfig:
    """Soft range model: probability that the controller can hold a link at range d.

    Not a hard disk — real range collapses with interference, antenna angle and
    altitude, so we use a plateau followed by a smooth logistic roll-off.
    """

    max_range_m: float = DEFAULT_MAX_RANGE_M
    plateau_frac: float = 0.5
    """Fraction of max_range over which p stays ~1.0 before the roll-off begins."""
    softness_m: float = 1500.0
    """Logistic width (metres) of the roll-off around the effective range edge."""


@dataclass
class MonteCarloConfig:
    """Controls Monte-Carlo sampling over input uncertainty (Phase 3)."""

    samples_per_sighting: int = 64
    seed: int | None = 12345


@dataclass
class CanopyConfig:
    """How the canopy layer affects the result beyond pure occlusion (Phase 4)."""

    launch_penalty_height_m: float = 5.0
    """Canopy taller than this starts to penalise launch feasibility."""
    launch_penalty_full_m: float = 15.0
    """Canopy at/above this gets the full (but still soft) launch down-weight."""
    min_launch_weight: float = 0.15
    """Floor on the launch-feasibility weight — operators do use forest edges."""


@dataclass
class Config:
    """Top-level configuration bundle threaded through the pipeline."""

    max_range_m: float = DEFAULT_MAX_RANGE_M
    antenna_height_m: float = DEFAULT_ANTENNA_HEIGHT_M
    coarse_resolution_m: float = DEFAULT_ANALYSIS_RESOLUTION_M
    fine_resolution_m: float = DEFAULT_FINE_RESOLUTION_M

    combine: str = "min"
    """How per-sighting visibility surfaces are combined into the heatmap.

    The core assumption is a *single* launch point, so a controller location is
    only viable if **every** sighting can be seen from it.

    * ``"min"`` (default) — strict intersection: the cell value is the *weakest*
      sighting's contribution, so a single pin that cannot see a cell gates it
      to ~0. This is the faithful "all drones must be visible" rule. Monte-Carlo
      sampling inside each sighting keeps it soft (it is the min of per-sighting
      *probabilities*, not of hard booleans).
    * ``"geometric_mean"`` — softer intersection ``(prod c_i)^(1/n)``. A single
      low pin only dents the score (``0.2`` over four 1.0s still ~0.72), so weak
      corners survive; kept for comparison, not recommended as the default.
    * ``"arithmetic_mean"`` — soft *union* ``mean_i c_i``: a cell lights up if it
      sees a large *fraction* of sightings, so areas only one drone can reach
      still show. Most forgiving of a bad sighting, but blurriest.
    """

    canopy_source: str = "eth"
    """Which global canopy-height product to fetch (Phase 4).

    * ``"eth"`` (default) — ETH Sentinel-2 10 m (2020). Much faster to download;
      a smoothed 10 m canopy-top height.
    * ``"meta"`` — Meta/WRI 1 m. Per-tree detail for fine occlusion near the
      observer, at a large download cost.
    """

    altitude_is_agl: bool = True
    """If True, Sighting.altitude is metres above ground at the drone's position;
    converted to absolute (ASL) by adding the surface elevation under the drone.
    If False, altitude is already metres above sea level."""

    prefer_gpu: bool = False
    """Use the CUDA viewshed kernel when a device is available (Phase 5)."""

    range_model: RangeModelConfig = field(default_factory=RangeModelConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    canopy: CanopyConfig = field(default_factory=CanopyConfig)

    cache_dir: str = "data_cache"

    def __post_init__(self) -> None:
        # Keep the range model's max range in sync with the top-level setting
        # unless the caller explicitly customised the range model.
        if self.range_model.max_range_m == DEFAULT_MAX_RANGE_M:
            self.range_model.max_range_m = self.max_range_m
        if self.canopy_source not in ("eth", "meta"):
            self.canopy_source = "eth"
