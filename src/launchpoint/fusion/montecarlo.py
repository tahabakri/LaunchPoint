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


def reachable_footprint(
    sightings: list[Sighting],
    occluder: RasterGrid,
    config: Config,
    projector: Projector,
    *,
    range_model: RangeModel | None = None,
    dilate_m: float | None = None,
    threshold: float = 1e-3,
    intersect: bool | None = None,
) -> np.ndarray:
    """Boolean mask of cells that can carry a non-zero final heatmap value.

    The cheap "where could the rays land?" pass that gates the expensive 1 m
    canopy fetch (N2). It runs one viewshed per sighting against the **DSM alone**
    (no canopy, no bare earth), then combines the per-sighting masks **the same
    way the heatmap combines the sightings**:

    * ``intersect=True`` (the default whenever ``config.combine == "min"``) —
      AND the masks. With a strict min, a cell is only viable if *every* sighting
      can see it, so canopy is only needed in the **intersection** of the
      viewsheds — a much smaller area than the union, and the real download win.
    * otherwise — OR the masks (union), since ``geometric_mean`` /
      ``arithmetic_mean`` keep cells only some sightings can reach.

    Correctness: each per-sighting DSM-only mask is a *superset* of that
    sighting's true canopy-aware contribution (canopy only lowers the antenna
    target and so can only shrink visibility), and we additionally lift the
    observer by ~2σ of its altitude uncertainty and dilate by ``dilate_m`` for
    position jitter. So the combined mask is a safe superset of the final
    non-zero region under either combine rule — canopy is fetched everywhere the
    heatmap could possibly be non-zero, and nowhere it can't.
    """
    range_model = range_model or RangeModel(config.range_model)
    if intersect is None:
        intersect = config.combine == "min"

    rad = 0
    if dilate_m and dilate_m > 0:
        rad = max(1, int(round(dilate_m / occluder.res_x)))

    combined: np.ndarray | None = None
    for s in sightings:
        nx, ny = projector.to_utm(s.lon, s.lat)
        # Lift the observer by ~2σ of altitude: a higher drone sees more, so this
        # keeps the per-sighting mask a superset w.r.t. altitude uncertainty.
        alt = s.altitude + 2.0 * s.altitude_sigma_m
        oz = _observer_z(occluder, nx, ny, alt, config.altitude_is_agl)
        vi = ViewshedInput(
            occluder=occluder,
            observer_xy=(nx, ny),
            observer_z=oz,
            target_height=config.antenna_height_m,
            ground=None,  # DSM-only: occluder doubles as the target surface
            refraction=True,
        )
        vs = compute_viewshed(vi, range_model, prefer_gpu=config.prefer_gpu)
        mask = np.nan_to_num(vs.data) > threshold
        if rad and mask.any():
            from scipy import ndimage

            # Dilate per sighting *before* combining so an intersection still
            # covers cells each sighting reaches only under position jitter.
            mask = ndimage.binary_dilation(mask, iterations=rad)
        if combined is None:
            combined = mask
        elif intersect:
            combined &= mask
        else:
            combined |= mask
    return combined if combined is not None else np.zeros(occluder.shape, dtype=bool)


def combine_contributions(
    contributions: list[np.ndarray], mode: str = "min"
) -> np.ndarray:
    """Combine per-sighting contribution surfaces into one heatmap.

    Each input is a per-cell value in [0, 1] (the MC-mean visible fraction for
    one sighting). Three combination rules, from strict to forgiving:

    * ``"min"`` — ``min_i c_i``. Strict intersection: the value is the weakest
      sighting's contribution, so a single sighting that cannot see a cell gates
      it to ~0. The faithful single-launch-point rule.
    * ``"geometric_mean"`` — ``(prod_i c_i)^(1/n)`` in log space. Softer: a lone
      low sighting only dents the score, so weak corners survive.
    * ``"arithmetic_mean"`` — ``mean_i c_i``. A soft *union*: a cell scores high
      if it sees a large fraction of sightings, so single-sighting areas remain.
    """
    if not contributions:
        raise ValueError("need at least one contribution surface")
    stack = np.stack([np.asarray(c, dtype=np.float64) for c in contributions])
    if mode == "min":
        return stack.min(axis=0).astype(np.float32)
    if mode == "arithmetic_mean":
        return stack.mean(axis=0).astype(np.float32)
    if mode != "geometric_mean":
        raise ValueError(f"unknown combine mode: {mode!r}")
    # Geometric mean via exp(mean(log)). log(0) -> -inf -> exp -> 0, exactly the
    # soft-AND behaviour we want; clip away tiny negatives from float noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.clip(stack, 0.0, None))
        prob = np.exp(logs.mean(axis=0))
    prob[~np.isfinite(prob)] = 0.0
    return prob.astype(np.float32)


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

    prob = combine_contributions(
        [c.data for c in per_sighting], config.combine
    )

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
