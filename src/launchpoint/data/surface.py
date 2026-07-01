"""Build the full surface stack the analysis consumes.

Combines Phase 1 (DSM) and Phase 4 (canopy, buildings, derived bare earth) into
three aligned ``RasterGrid``s:

* ``occluder``      — the GLO-30 DSM (blocks rays; already includes canopy/roofs).
* ``ground``        — derived bare earth (gives target/antenna height).
* ``launch_weight`` — soft canopy-based launch-feasibility down-weight.

Each enrichment layer degrades gracefully: if canopy or buildings can't be
fetched, the corresponding term is zero and ``ground`` falls back to the raw DSM
(the Phase 2 placeholder), so a run never hard-fails on a missing optional layer.
Results are cached to disk per AOI so repeated runs are offline.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

from launchpoint.config import Config
from launchpoint.core.geo import BBox, Projector, aoi_for_sightings
from launchpoint.core.grid import RasterGrid
from launchpoint.core.sighting import Sighting
from launchpoint.data.bare_earth import derive_bare_earth, launch_feasibility_weight
from launchpoint.data.cache import Cache, cache_key
from launchpoint.data.progress import FetchReporter, StageCallback, log


@dataclass
class SurfaceStack:
    occluder: RasterGrid
    ground: RasterGrid
    launch_weight: RasterGrid
    canopy_height: RasterGrid | None = None
    building_height: RasterGrid | None = None


def _target_grid(bbox: BBox, projector: Projector, resolution: float) -> RasterGrid:
    return RasterGrid.empty(bbox, resolution, projector.utm_crs, fill=np.nan)


# A reachable footprint covering nearly the whole AOI means the DSM occludes
# almost nothing, so gating buys no bandwidth — fall back to a plain full fetch.
_FOOTPRINT_FULL_FRAC = 0.85


def _mask_bbox(mask: np.ndarray, grid: RasterGrid) -> BBox:
    """UTM bounding box of the True cells of ``mask`` on ``grid``."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    # Corners of the inclusive pixel block (top-left of r0,c0 .. bottom-right of r1,c1).
    x0, y0 = grid.transform * (c0, r0)
    x1, y1 = grid.transform * (c1 + 1, r1 + 1)
    return BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _canopy_fetcher(source: str):
    """Resolve the canopy fetch function for a configured source.

    Both fetchers share the ``(target, projector, cache_dir=None, reporter=None)``
    signature so the footprint-gating logic below is source-agnostic.
    """
    if source == "eth":
        from launchpoint.data.eth_canopy import fetch_canopy_height_eth

        return fetch_canopy_height_eth
    from launchpoint.data.canopy import fetch_canopy_height

    return fetch_canopy_height


def _fetch_canopy_in_footprint(
    occluder: RasterGrid,
    projector: Projector,
    cache_dir: str | None,
    footprint: np.ndarray | None,
    reporter: FetchReporter | None = None,
    fetcher=None,
) -> RasterGrid:
    """Fetch the canopy, restricted to ``footprint`` when it helps (N2).

    Canopy only matters where a controller could actually stand (the reachable
    footprint); cells outside it never enter the result. So we fetch canopy only
    over the footprint's bounding box and leave the rest NaN (treated as 0
    downstream) — shrinking the read window is exactly what cuts the bytes pulled
    over the network (especially for the Meta tiles' full-width row strips).

    ``fetcher`` selects the source; it defaults to the Meta/WRI 1 m product.
    """
    if fetcher is None:
        from launchpoint.data.canopy import fetch_canopy_height

        fetcher = fetch_canopy_height

    if footprint is None or not footprint.any():
        return fetcher(occluder, projector, cache_dir, reporter=reporter)

    coverage = float(footprint.mean())
    if coverage >= _FOOTPRINT_FULL_FRAC:
        log.info(
            "canopy footprint covers %.0f%% of AOI — fetching full window", coverage * 100
        )
        return fetcher(occluder, projector, cache_dir, reporter=reporter)

    sub = occluder.subgrid(_mask_bbox(footprint, occluder))
    log.info(
        "canopy footprint gate: %.0f%% of AOI -> %d x %d cell window (rows %d->%d span)",
        coverage * 100, sub.rows, sub.cols, occluder.rows, sub.rows,
    )
    sub_canopy = fetcher(sub, projector, cache_dir, reporter=reporter)

    full = occluder.like(fill=np.nan)
    r0, c0 = occluder.world_to_pixel(sub.bounds.minx, sub.bounds.maxy)
    full.data[r0:r0 + sub.rows, c0:c0 + sub.cols] = sub_canopy.data
    return full


def build_surface_stack(
    sightings: list[Sighting],
    config: Config,
    projector: Projector | None = None,
    *,
    use_canopy: bool = True,
    use_buildings: bool = True,
    use_cache: bool = True,
    gate_canopy_with_footprint: bool = True,
    progress_callback: StageCallback | None = None,
) -> SurfaceStack:
    """Fetch/derive the occluder, bare-earth and launch-weight surfaces.

    Requires network on first call (Copernicus DSM is mandatory; canopy/buildings
    are best-effort). Subsequent calls over the same AOI read from the cache.
    """
    if projector is None:
        projector, bbox = aoi_for_sightings(sightings, config.max_range_m)
    else:
        _, bbox = aoi_for_sightings(sightings, config.max_range_m)

    epsg = projector.utm_crs.to_epsg()
    res = config.coarse_resolution_m
    cache = Cache(config.cache_dir)
    # The DSM is source-independent; bare earth and launch weight are derived
    # from the canopy, so they must not be shared across canopy sources.
    src = config.canopy_source
    keys = {
        "occluder": cache_key("dsm", bbox, epsg, res),
        "ground": cache_key(f"bare_earth_{src}", bbox, epsg, res),
        "launch": cache_key(f"launch_weight_{src}", bbox, epsg, res),
        "canopy": cache_key(f"canopy_height_{src}", bbox, epsg, res),
        "building": cache_key("building_height", bbox, epsg, res),
    }
    required_keys = ["occluder", "ground", "launch"]
    if use_canopy:
        required_keys.append("canopy")
    if use_buildings:
        required_keys.append("building")

    if use_cache and all(cache.has(keys[name]) for name in required_keys):
        log.info("surfaces served from cache for AOI epsg:%s @ %g m", epsg, res)
        return SurfaceStack(
            occluder=cache.load(keys["occluder"]),
            ground=cache.load(keys["ground"]),
            launch_weight=cache.load(keys["launch"]),
            canopy_height=cache.load(keys["canopy"]) if use_canopy else None,
            building_height=cache.load(keys["building"]) if use_buildings else None,
        )

    reporter = FetchReporter(progress_callback)
    log.info(
        "fetching surfaces for AOI %.1f x %.1f km @ %g m (canopy=%s, buildings=%s)",
        bbox.width / 1000, bbox.height / 1000, res, use_canopy, use_buildings,
    )
    target = _target_grid(bbox, projector, res)

    # --- Phase 1: mandatory DSM ---
    from launchpoint.data.copernicus import fetch_dsm

    t0 = time.perf_counter()
    occluder = fetch_dsm(target, projector, reporter=reporter)
    reporter.layer_done("DSM", time.perf_counter() - t0)

    # --- Phase 4: optional enrichment ---
    canopy = None
    if use_canopy:
        try:
            footprint = None
            if gate_canopy_with_footprint:
                from launchpoint.fusion.montecarlo import reachable_footprint

                sigmas = [s.position_sigma_m for s in sightings]
                dilate_m = 3.0 * (max(sigmas) if sigmas else 0.0) + 2.0 * res
                footprint = reachable_footprint(
                    sightings, occluder, config, projector, dilate_m=dilate_m
                )
            t0 = time.perf_counter()
            canopy = _fetch_canopy_in_footprint(
                occluder, projector, config.cache_dir, footprint, reporter=reporter,
                fetcher=_canopy_fetcher(config.canopy_source),
            )
            reporter.layer_done("canopy", time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - best-effort layer
            warnings.warn(f"canopy fetch failed, continuing without it: {exc}")

    building = None
    if use_buildings:
        try:
            from launchpoint.data.buildings import fetch_building_height

            t0 = time.perf_counter()
            building = fetch_building_height(occluder, projector, reporter=reporter)
            reporter.layer_done("buildings", time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - best-effort layer
            warnings.warn(f"building fetch failed, continuing without it: {exc}")

    reporter.done()

    ground = derive_bare_earth(occluder, canopy, building)
    launch_weight = launch_feasibility_weight(canopy, occluder, config.canopy)

    if use_cache:
        cache.save(keys["occluder"], occluder)
        cache.save(keys["ground"], ground)
        cache.save(keys["launch"], launch_weight)
        if canopy is not None:
            cache.save(keys["canopy"], canopy)
        if building is not None:
            cache.save(keys["building"], building)

    return SurfaceStack(
        occluder=occluder,
        ground=ground,
        launch_weight=launch_weight,
        canopy_height=canopy,
        building_height=building,
    )
