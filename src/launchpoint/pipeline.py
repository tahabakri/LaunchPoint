"""High-level orchestration: sightings in, probability heatmap out.

This is the seam between the (network-touching) data layer and the pure-compute
core. For offline/synthetic use, pass a precomputed ``occluder`` surface and
``projector`` and no network access happens at all — which is how the synthetic
recovery test and the whole test suite run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from launchpoint.config import Config
from launchpoint.core.geo import Projector, aoi_for_sightings
from launchpoint.core.grid import RasterGrid
from launchpoint.core.sighting import Sighting
from launchpoint.fusion.montecarlo import FusionResult, fuse_sightings


@dataclass
class OriginEstimate:
    """The pipeline's answer."""

    probability: RasterGrid
    fusion: FusionResult
    projector: Projector

    def argmax_lonlat(self) -> tuple[float, float]:
        """Most-likely controller location as (lon, lat)."""
        data = np.where(np.isfinite(self.probability.data), self.probability.data, -np.inf)
        r, c = np.unravel_index(int(np.argmax(data)), data.shape)
        x, y = self.probability.pixel_to_world(r, c)
        lon, lat = self.projector.to_lonlat(x, y)
        return float(lon), float(lat)

    def credible_mask(self, frac: float = 0.5) -> RasterGrid:
        """Boolean grid of the smallest set of cells holding ``frac`` of the
        total probability mass (a crude highest-density region)."""
        p = np.where(np.isfinite(self.probability.data), self.probability.data, 0.0)
        total = p.sum()
        if total <= 0:
            return self.probability.copy_with(np.zeros_like(p))
        flat = p.ravel()
        order = np.argsort(flat)[::-1]
        cum = np.cumsum(flat[order])
        keep = cum <= frac * total
        # Always include at least the peak cell.
        keep[0] = True
        mask = np.zeros_like(flat)
        mask[order[keep]] = 1.0
        return self.probability.copy_with(mask.reshape(p.shape))


def find_origin(
    sightings: list[Sighting],
    config: Config | None = None,
    *,
    occluder: RasterGrid | None = None,
    ground: RasterGrid | None = None,
    projector: Projector | None = None,
    launch_weight: RasterGrid | None = None,
) -> OriginEstimate:
    """Estimate the controller-origin probability heatmap.

    Parameters
    ----------
    sightings:
        One or more fuzzy drone-position snapshots.
    config:
        Tunables (range, antenna height, MC samples, ...). Defaults applied.
    occluder, ground, projector, launch_weight:
        Optional precomputed surfaces (offline/synthetic path). If ``occluder``
        is None, the data layer fetches a Copernicus GLO-30 DSM for the AOI
        (requires network — see launchpoint.data).
    """
    config = config or Config()

    if occluder is None:
        # Network path: build the surface stack from keyless sources.
        from launchpoint.data import build_surface_stack

        proj, _ = aoi_for_sightings(sightings, config.max_range_m)
        stack = build_surface_stack(sightings, config, projector=proj)
        occluder = stack.occluder
        ground = stack.ground if ground is None else ground
        launch_weight = stack.launch_weight if launch_weight is None else launch_weight
        projector = proj
    elif projector is None:
        projector, _ = aoi_for_sightings(sightings, config.max_range_m)

    fusion = fuse_sightings(
        sightings,
        occluder=occluder,
        config=config,
        projector=projector,
        ground=ground,
        launch_weight=launch_weight,
    )
    return OriginEstimate(
        probability=fusion.probability, fusion=fusion, projector=projector
    )
