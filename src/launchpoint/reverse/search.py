"""Coarse-to-fine candidate-grid search: the reverse analogue of ``fusion/multiscale.py``.

``fusion/multiscale.py``'s ``coarse_to_fine`` decouples *terrain resolution*
from *extent* while keeping the *observer count* fixed (one viewshed per
sighting at both passes). Here the expensive step is "one viewshed per
*candidate launch cell*," and candidates can be far more numerous than
sightings ever are, so this module decouples *candidate density* from extent
instead: a sparse coarse candidate grid first, refined to a denser grid only
inside the small number of patches that scored well.

Note on why this module does **not** also refetch the terrain surface at a
finer resolution inside hot patches (unlike ``multiscale.py``'s coarse/fine
surface split): a forward hot patch is a *candidate controller* region, and
the sighting whose viewshed is being computed there can be resolved
correctly against a small patch-sized grid because the *sighting position is
the fixed input* and the patch is where the *answer* lives. In reverse, the
answer (candidate launch cell) and the reduction target (the flight zone) are
two different places that can be many kilometres apart — a fine-resolution
grid would have to span the whole corridor between them to run a single
viewshed, which reintroduces exactly the cost blowup coarse-to-fine exists to
avoid. So this module fetches **one** terrain surface, at
``surface_resolution_m``, over the whole search AOI once, and only makes the
*candidate grid itself* sparse-then-dense on top of that fixed surface. Truly
resolving finer terrain detail right at the winning candidate's own footprint
is a reasonable follow-up, not attempted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from launchpoint.config import Config
from launchpoint.core.geo import BBox, Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.core.target_zone import TargetZone
from launchpoint.fusion.montecarlo import ProgressCallback
from launchpoint.fusion.multiscale import _boxes_overlap, _merge_boxes
from launchpoint.reverse.coverage import apply_launch_weight, candidate_coverage, zone_cell_mask
from launchpoint.viewshed.range_model import RangeModel


@dataclass
class LaunchSearchConfig:
    """Tunables for the candidate-grid coarse-to-fine search."""

    coverage_threshold: float = 0.5
    """Per-cell contribution value counted as 'covered' for frac_covered."""

    candidate_coarse_stride_m: float = 60.0
    """Coarse candidate-grid spacing for the first pass."""
    candidate_fine_stride_m: float = 10.0
    """Refined candidate spacing inside hot patches."""
    hot_top_frac: float = 0.15
    """Refine candidates scoring in the top fraction of surviving (post
    pre-filter) coarse candidates, by count."""
    patch_pad_m: float = 150.0
    """Pad each hot-candidate cluster's bounding box before refining."""
    max_fine_patches: int = 12
    min_cluster_gap_m: float = 120.0
    """Coarse hot-candidates within this distance are merged into one cluster."""

    surface_resolution_m: float = 30.0
    """Terrain-surface resolution used for the whole search (see module
    docstring for why this is a single, fixed resolution rather than a
    coarse/fine split)."""


@dataclass
class LaunchCoverageResult:
    """Output of a launch-coverage search."""

    coverage: RasterGrid
    """Per-candidate-cell score (min_visibility * launch_weight), NaN where
    no candidate was evaluated."""
    min_visibility: RasterGrid
    """Per-candidate-cell raw min_visibility (pre-launch_weight)."""
    frac_covered: RasterGrid
    """Per-candidate-cell secondary metric: fraction of the zone above
    ``coverage_threshold``."""
    best_xy: tuple[float, float]
    best_score: float
    best_min_visibility: float
    best_frac_covered: float
    projector: Projector
    zone: TargetZone
    n_candidates_evaluated: int
    n_candidates_prefiltered: int
    patch_bounds: list[BBox] = field(default_factory=list)

    def best_lonlat(self) -> tuple[float, float]:
        lon, lat = self.projector.to_lonlat(*self.best_xy)
        return float(lon), float(lat)


def _candidate_grid(bbox: BBox, stride_m: float) -> list[tuple[float, float]]:
    """Enumerate candidate points on a regular grid covering ``bbox``."""
    n_cols = max(1, int(np.floor(bbox.width / stride_m)) + 1)
    n_rows = max(1, int(np.floor(bbox.height / stride_m)) + 1)
    xs = bbox.minx + np.arange(n_cols) * stride_m
    ys = bbox.miny + np.arange(n_rows) * stride_m
    return [(float(x), float(y)) for y in ys for x in xs]


def _score_candidates(
    candidates: list[tuple[float, float]],
    zone: TargetZone,
    zone_mask: np.ndarray,
    occluder: RasterGrid,
    ground: RasterGrid | None,
    launch_weight: RasterGrid | None,
    config: Config,
    range_model: RangeModel,
    search: LaunchSearchConfig,
    zone_center_xy: tuple[float, float],
) -> tuple[list[dict], int]:
    """Score every candidate that survives the cheap distance pre-filter.

    Returns (survivors, n_prefiltered) where each survivor dict has
    x, y, min_visibility, frac_covered, score.
    """
    cx, cy = zone_center_xy
    hard_cap = range_model.hard_cap_m
    survivors: list[dict] = []
    n_prefiltered = 0
    for x, y in candidates:
        # Cheap pre-filter, before any viewshed: a candidate that cannot even
        # reach the *farthest* point of the zone at non-negligible weight
        # cannot possibly achieve non-zero min_visibility.
        dist_far = float(np.hypot(x - cx, y - cy)) + zone.radius_m
        if dist_far > hard_cap:
            n_prefiltered += 1
            continue
        min_vis, frac = candidate_coverage(
            (x, y), zone_mask, occluder, zone.flight_altitude_m, config, range_model,
            ground=ground, coverage_threshold=search.coverage_threshold,
        )
        score = apply_launch_weight(min_vis, launch_weight, occluder, (x, y))
        survivors.append({"x": x, "y": y, "min_visibility": min_vis, "frac_covered": frac, "score": score})
    return survivors, n_prefiltered


def _hot_patch_boxes(
    survivors: list[dict], occluder: RasterGrid, search: LaunchSearchConfig
) -> list[BBox]:
    """Cluster the top-scoring survivors into padded, merged bounding boxes."""
    if not survivors:
        return []
    ranked = sorted(survivors, key=lambda s: s["score"], reverse=True)
    top_n = max(1, int(np.ceil(len(ranked) * search.hot_top_frac)))
    hot = ranked[:top_n]
    if all(s["score"] <= 0.0 for s in hot):
        return []

    mask = np.zeros(occluder.shape, dtype=bool)
    for s in hot:
        r, c = occluder.world_to_pixel(s["x"], s["y"])
        if occluder.contains_pixel(r, c):
            mask[r, c] = True
    if not mask.any():
        return []

    dilate_cells = max(1, int(round(search.min_cluster_gap_m / occluder.res_x)))
    dilated = ndimage.binary_dilation(mask, iterations=dilate_cells)
    labels, n_labels = ndimage.label(dilated)

    scored_boxes: list[tuple[float, BBox]] = []
    for lab in range(1, n_labels + 1):
        rows, cols = np.where(labels == lab)
        x0, y0 = occluder.transform * (cols.min(), rows.max() + 1)
        x1, y1 = occluder.transform * (cols.max() + 1, rows.min())
        box = BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)).buffered(search.patch_pad_m)
        # Best survivor score whose cell falls in this component, for ranking.
        best_in_box = max(
            (s["score"] for s in hot if _boxes_overlap(
                BBox(s["x"], s["y"], s["x"], s["y"]), box
            )),
            default=0.0,
        )
        scored_boxes.append((best_in_box, box))

    scored_boxes.sort(key=lambda t: t[0], reverse=True)
    boxes = _merge_boxes([b for _, b in scored_boxes])
    return boxes[: search.max_fine_patches]


def find_launch_coverage(
    zone: TargetZone,
    config: Config,
    search: LaunchSearchConfig,
    occluder: RasterGrid,
    projector: Projector,
    ground: RasterGrid | None = None,
    launch_weight: RasterGrid | None = None,
    progress_callback: ProgressCallback | None = None,
) -> LaunchCoverageResult:
    """Run the coarse-then-fine candidate search over a single, already-fetched
    terrain surface (``occluder``/``ground``/``launch_weight``, all sharing one
    georeferencing) spanning the launch-search AOI.
    """
    range_model = RangeModel(config.range_model)
    aoi_bbox = occluder.bounds
    zone_center_xy = projector.to_utm(zone.center_lon, zone.center_lat)
    zone_mask = zone_cell_mask(occluder, projector, zone)

    coverage = occluder.like(fill=np.nan)
    min_vis_grid = occluder.like(fill=np.nan)
    frac_grid = occluder.like(fill=np.nan)

    def _paint(x: float, y: float, min_vis: float, frac: float, score: float) -> None:
        r, c = occluder.world_to_pixel(x, y)
        if not occluder.contains_pixel(r, c):
            return
        existing = coverage.data[r, c]
        if np.isfinite(existing) and existing >= score:
            return
        coverage.data[r, c] = score
        min_vis_grid.data[r, c] = min_vis
        frac_grid.data[r, c] = frac

    if progress_callback is not None:
        progress_callback("Search candidates", "running", "Scanning coarse candidate grid", None)

    coarse_candidates = _candidate_grid(aoi_bbox, search.candidate_coarse_stride_m)
    coarse_survivors, n_prefiltered = _score_candidates(
        coarse_candidates, zone, zone_mask, occluder, ground, launch_weight,
        config, range_model, search, zone_center_xy,
    )
    for s in coarse_survivors:
        _paint(s["x"], s["y"], s["min_visibility"], s["frac_covered"], s["score"])

    if progress_callback is not None:
        progress_callback(
            "Search candidates", "complete",
            f"Evaluated {len(coarse_survivors)} candidate(s), skipped {n_prefiltered} out of range",
            {"evaluated": len(coarse_survivors), "prefiltered": n_prefiltered},
        )

    patch_bounds = _hot_patch_boxes(coarse_survivors, occluder, search)

    n_evaluated = len(coarse_survivors)
    all_survivors = list(coarse_survivors)

    if progress_callback is not None:
        progress_callback(
            "Refine hot patches", "running",
            f"Refining {len(patch_bounds)} hot patch(es)", {"patches": len(patch_bounds)},
        )

    for patch in patch_bounds:
        fine_candidates = _candidate_grid(patch, search.candidate_fine_stride_m)
        fine_survivors, fine_prefiltered = _score_candidates(
            fine_candidates, zone, zone_mask, occluder, ground, launch_weight,
            config, range_model, search, zone_center_xy,
        )
        n_prefiltered += fine_prefiltered
        n_evaluated += len(fine_survivors)
        all_survivors.extend(fine_survivors)
        for s in fine_survivors:
            _paint(s["x"], s["y"], s["min_visibility"], s["frac_covered"], s["score"])

    if progress_callback is not None:
        progress_callback(
            "Refine hot patches", "complete",
            f"Refined {len(patch_bounds)} patch(es)", {"patches": len(patch_bounds)},
        )

    if all_survivors:
        best = max(all_survivors, key=lambda s: s["score"])
        best_xy = (best["x"], best["y"])
        best_score = best["score"]
        best_min_vis = best["min_visibility"]
        best_frac = best["frac_covered"]
    else:
        best_xy = zone_center_xy
        best_score = 0.0
        best_min_vis = 0.0
        best_frac = 0.0

    return LaunchCoverageResult(
        coverage=coverage,
        min_visibility=min_vis_grid,
        frac_covered=frac_grid,
        best_xy=best_xy,
        best_score=best_score,
        best_min_visibility=best_min_vis,
        best_frac_covered=best_frac,
        projector=projector,
        zone=zone,
        n_candidates_evaluated=n_evaluated,
        n_candidates_prefiltered=n_prefiltered,
        patch_bounds=patch_bounds,
    )
