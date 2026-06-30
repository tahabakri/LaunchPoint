"""Turn per-sighting viewsheds into the actual locator, honest about uncertainty.

Two principles from the roadmap:

* **Accumulate, don't hard-AND.** Score each candidate cell by how many
  sightings it can see (with range), as a soft mean. A strict "must see all"
  would let a single bad sighting erase the true region.
* **Monte Carlo over input uncertainty.** Each sighting is a fuzzy snapshot, so
  we sample N plausible positions/altitudes from its error model, run the
  single-observer viewshed per sample, and average. The result literally means
  "how often a controller here could see enough of the sightings, across all the
  ways the sightings might really have been."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from launchpoint.config import Config
from launchpoint.core.geo import Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.core.sighting import Sighting
from launchpoint.viewshed.core import ViewshedInput, compute_viewshed
from launchpoint.viewshed.range_model import RangeModel

ProgressCallback = Callable[[str, str, str, dict | None], None]


@dataclass
class FusionResult:
    """Output of a fusion run."""

    probability: RasterGrid
    """Normalised heatmap in [0, 1]: mean over sightings of the MC-visible
    fraction, optionally down-weighted by launch feasibility."""
    per_sighting: list[RasterGrid]
    """Per-sighting MC-mean contribution surfaces (for inspection/debugging)."""
    n_samples: int


def _observer_z(occluder: RasterGrid, x: float, y: float, altitude: float,
                altitude_is_agl: bool) -> float:
    """Convert a sampled altitude to absolute elevation at (x, y)."""
    if not altitude_is_agl:
        return altitude
    r, c = occluder.world_to_pixel(x, y)
    if occluder.contains_pixel(r, c):
        base = occluder.data[r, c]
        if np.isfinite(base):
            return float(base) + altitude
    # Observer outside grid or nodata: fall back to AOI median surface.
    finite = occluder.data[np.isfinite(occluder.data)]
    base = float(np.median(finite)) if finite.size else 0.0
    return base + altitude


def monte_carlo_sighting(
    sighting: Sighting,
    occluder: RasterGrid,
    config: Config,
    projector: Projector,
    range_model: RangeModel,
    ground: RasterGrid | None = None,
    rng: np.random.Generator | None = None,
    n_samples: int | None = None,
) -> RasterGrid:
    """MC-mean contribution surface for a single sighting.

    Samples positions/altitudes from the sighting's error model, runs the
    viewshed per sample, and averages — the per-cell value is the fraction of
    samples for which a controller at that cell could see (within range) the
    drone.
    """
    rng = rng or np.random.default_rng(config.monte_carlo.seed)
    n = n_samples or config.monte_carlo.samples_per_sighting

    nx, ny = projector.to_utm(sighting.lon, sighting.lat)
    east, north, alts = sighting.sample_positions(n, rng)

    acc = np.zeros(occluder.shape, dtype=np.float64)
    for i in range(n):
        ox = nx + east[i]
        oy = ny + north[i]
        oz = _observer_z(occluder, ox, oy, alts[i], config.altitude_is_agl)
        vi = ViewshedInput(
            occluder=occluder,
            observer_xy=(ox, oy),
            observer_z=oz,
            target_height=config.antenna_height_m,
            ground=ground,
            refraction=True,
        )
        acc += compute_viewshed(vi, range_model, prefer_gpu=config.prefer_gpu).data
    acc /= float(n)
    return occluder.copy_with(acc.astype(np.float32))


def fuse_sightings(
    sightings: list[Sighting],
    occluder: RasterGrid,
    config: Config,
    projector: Projector,
    ground: RasterGrid | None = None,
    launch_weight: RasterGrid | None = None,
    progress_callback: ProgressCallback | None = None,
) -> FusionResult:
    """Fuse all sightings into a single probability heatmap.

    ``launch_weight`` (Phase 4) is an optional per-cell multiplier in [0, 1]
    that down-weights cells where launching/operating is implausible (e.g. under
    closed canopy). It is applied once, at the end.
    """
    if not sightings:
        raise ValueError("need at least one sighting")

    range_model = RangeModel(config.range_model)
    n = config.monte_carlo.samples_per_sighting
    seed = config.monte_carlo.seed

    per_sighting: list[RasterGrid] = []
    accum = np.zeros(occluder.shape, dtype=np.float64)
    for idx, s in enumerate(sightings):
        if progress_callback is not None:
            label = s.label or f"S{idx + 1}"
            progress_callback(
                "Run fusion",
                "running",
                f"Computing viewshed {idx + 1} of {len(sightings)} ({label})",
                {"current": idx + 1, "total": len(sightings), "label": label},
            )
        # Independent, reproducible stream per sighting.
        rng = np.random.default_rng(None if seed is None else seed + idx)
        contrib = monte_carlo_sighting(
            s, occluder, config, projector, range_model, ground=ground, rng=rng
        )
        per_sighting.append(contrib)
        accum += contrib.data

    # Soft accumulation: mean over sightings -> "fraction of sightings seen".
    prob = accum / float(len(sightings))

    if launch_weight is not None:
        w = np.where(np.isfinite(launch_weight.data), launch_weight.data, 1.0)
        prob = prob * w

    if progress_callback is not None:
        progress_callback(
            "Run fusion",
            "complete",
            f"Fused {len(sightings)} sighting(s)",
            {"current": len(sightings), "total": len(sightings)},
        )

    return FusionResult(
        probability=occluder.copy_with(prob.astype(np.float32)),
        per_sighting=per_sighting,
        n_samples=n,
    )
