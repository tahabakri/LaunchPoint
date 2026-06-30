"""Coarse-to-fine refinement: decouple resolution from extent.

The cost wall is ``viewshed x Monte-Carlo samples x cell count``, and cell count
explodes with resolution: a 24 km box is ~640 k cells at 30 m but ~576 M at 1 m.
But fine detail only changes the answer *near* the observer and *near* candidate
cells — a 10 m tree at 5 km subtends a negligible angle. So:

1. Coarse pass at ~30 m over the whole disk -> cheap rough heatmap.
2. Fine pass at ~1-2 m **only** in the hot candidate tiles plus a buffer around
   each sighting — a handful of small, comfortable patches.

Surfaces are obtained through a ``SurfaceProvider`` so the same machinery works
offline (resample a precomputed stack) and online (refetch COGs at fine
resolution for just the patch bounds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import ndimage

from launchpoint.config import Config
from launchpoint.core.geo import BBox, Projector, aoi_for_sightings
from launchpoint.core.grid import RasterGrid
from launchpoint.core.sighting import Sighting
from launchpoint.data.surface import SurfaceStack
from launchpoint.pipeline import OriginEstimate, find_origin


@dataclass
class MultiscaleConfig:
    coarse_resolution_m: float = 30.0
    fine_resolution_m: float = 2.0
    hot_mass_frac: float = 0.6
    """Refine the smallest set of coarse cells holding this share of the mass."""
    sighting_buffer_m: float = 500.0
    """Always refine a patch this big around each sighting (near-field detail)."""
    patch_pad_m: float = 120.0
    """Pad each hot patch so rays just outside it are still represented."""
    max_fine_patches: int = 16


class SurfaceProvider(Protocol):
    projector: Projector

    def stack(self, bbox: BBox, resolution: float) -> SurfaceStack: ...


@dataclass
class ResamplingSurfaceProvider:
    """Offline provider: resample a precomputed (coarse) stack onto any grid.

    Useful for tests and for reusing already-fetched surfaces. For genuinely new
    detail at fine resolution, use a provider that refetches the source COGs.
    """

    base: SurfaceStack
    projector: Projector

    def stack(self, bbox: BBox, resolution: float) -> SurfaceStack:
        fine = RasterGrid.empty(bbox, resolution, self.projector.utm_crs, fill=np.nan)
        return SurfaceStack(
            occluder=self.base.occluder.resample_to(fine),
            ground=self.base.ground.resample_to(fine),
            launch_weight=self.base.launch_weight.resample_to(fine),
        )


@dataclass
class MultiscaleResult:
    coarse: OriginEstimate
    fine_patches: list[OriginEstimate] = field(default_factory=list)
    patch_bounds: list[BBox] = field(default_factory=list)
    fine_cells: int = 0
    full_fine_cells: int = 0

    @property
    def speedup_vs_full_fine(self) -> float:
        return self.full_fine_cells / max(self.fine_cells, 1)


def _hot_mask(prob: RasterGrid, mass_frac: float) -> np.ndarray:
    p = np.where(np.isfinite(prob.data), prob.data, 0.0)
    total = p.sum()
    if total <= 0:
        return np.zeros_like(p, dtype=bool)
    flat = p.ravel()
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order])
    keep = cum <= mass_frac * total
    keep[0] = True
    mask = np.zeros(flat.size, dtype=bool)
    mask[order[keep]] = True
    return mask.reshape(p.shape)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a.maxx < b.minx or b.maxx < a.minx or a.maxy < b.miny or b.maxy < a.miny)


def _merge_boxes(boxes: list[BBox]) -> list[BBox]:
    """Greedily union overlapping boxes so patches aren't computed twice."""
    merged: list[BBox] = []
    for box in boxes:
        placed = False
        for i, m in enumerate(merged):
            if _boxes_overlap(box, m):
                merged[i] = BBox(
                    min(box.minx, m.minx), min(box.miny, m.miny),
                    max(box.maxx, m.maxx), max(box.maxy, m.maxy),
                )
                placed = True
                break
        if not placed:
            merged.append(box)
    return merged


def _candidate_patches(
    coarse: OriginEstimate,
    sightings: list[Sighting],
    projector: Projector,
    ms: MultiscaleConfig,
) -> list[BBox]:
    prob = coarse.probability
    boxes: list[BBox] = []

    # Hot candidate regions from the coarse heatmap (connected components).
    mask = _hot_mask(prob, ms.hot_mass_frac)
    labels, n = ndimage.label(mask)
    comps = []
    for lab in range(1, n + 1):
        rows, cols = np.where(labels == lab)
        comps.append((rows.size, rows, cols))
    comps.sort(key=lambda c: c[0], reverse=True)
    for _, rows, cols in comps[: ms.max_fine_patches]:
        x0, y0 = prob.transform * (cols.min(), rows.max() + 1)  # lower-left
        x1, y1 = prob.transform * (cols.max() + 1, rows.min())  # upper-right
        boxes.append(BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                     .buffered(ms.patch_pad_m))

    # Always refine the near-field around each sighting.
    half = ms.sighting_buffer_m
    for s in sightings:
        sx, sy = projector.to_utm(s.lon, s.lat)
        boxes.append(BBox(sx - half, sy - half, sx + half, sy + half))

    return _merge_boxes(boxes)


def _fuse_on_stack(
    sightings: list[Sighting], config: Config, projector: Projector, stack: SurfaceStack
) -> OriginEstimate:
    return find_origin(
        sightings,
        config=config,
        occluder=stack.occluder,
        ground=stack.ground,
        projector=projector,
        launch_weight=stack.launch_weight,
    )


def coarse_to_fine(
    sightings: list[Sighting],
    config: Config,
    provider: SurfaceProvider,
    ms: MultiscaleConfig | None = None,
) -> MultiscaleResult:
    """Run the coarse pass, then refine only the hot patches at fine resolution."""
    ms = ms or MultiscaleConfig()
    projector = provider.projector
    _, aoi = aoi_for_sightings(sightings, config.max_range_m)

    coarse_stack = provider.stack(aoi, ms.coarse_resolution_m)
    coarse_est = _fuse_on_stack(sightings, config, projector, coarse_stack)

    patches = _candidate_patches(coarse_est, sightings, projector, ms)

    fine_patches: list[OriginEstimate] = []
    fine_cells = 0
    for pb in patches:
        fstack = provider.stack(pb, ms.fine_resolution_m)
        fine_cells += fstack.occluder.data.size
        fine_patches.append(_fuse_on_stack(sightings, config, projector, fstack))

    full_fine_cells = int(
        (aoi.width / ms.fine_resolution_m) * (aoi.height / ms.fine_resolution_m)
    )

    return MultiscaleResult(
        coarse=coarse_est,
        fine_patches=fine_patches,
        patch_bounds=patches,
        fine_cells=fine_cells,
        full_fine_cells=full_fine_cells,
    )
