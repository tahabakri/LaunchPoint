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

import warnings
from dataclasses import dataclass

import numpy as np

from launchpoint.config import Config
from launchpoint.core.geo import BBox, Projector, aoi_for_sightings
from launchpoint.core.grid import RasterGrid
from launchpoint.core.sighting import Sighting
from launchpoint.data.bare_earth import derive_bare_earth, launch_feasibility_weight
from launchpoint.data.cache import Cache, cache_key


@dataclass
class SurfaceStack:
    occluder: RasterGrid
    ground: RasterGrid
    launch_weight: RasterGrid
    canopy_height: RasterGrid | None = None
    building_height: RasterGrid | None = None


def _target_grid(bbox: BBox, projector: Projector, resolution: float) -> RasterGrid:
    return RasterGrid.empty(bbox, resolution, projector.utm_crs, fill=np.nan)


def build_surface_stack(
    sightings: list[Sighting],
    config: Config,
    projector: Projector | None = None,
    *,
    use_canopy: bool = True,
    use_buildings: bool = True,
    use_cache: bool = True,
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
    keys = {
        "occluder": cache_key("dsm", bbox, epsg, res),
        "ground": cache_key("bare_earth", bbox, epsg, res),
        "launch": cache_key("launch_weight", bbox, epsg, res),
    }

    if use_cache and all(cache.has(k) for k in keys.values()):
        return SurfaceStack(
            occluder=cache.load(keys["occluder"]),
            ground=cache.load(keys["ground"]),
            launch_weight=cache.load(keys["launch"]),
        )

    target = _target_grid(bbox, projector, res)

    # --- Phase 1: mandatory DSM ---
    from launchpoint.data.copernicus import fetch_dsm

    occluder = fetch_dsm(target, projector)

    # --- Phase 4: optional enrichment ---
    canopy = None
    if use_canopy:
        try:
            from launchpoint.data.canopy import fetch_canopy_height

            canopy = fetch_canopy_height(occluder, projector, config.cache_dir)
        except Exception as exc:  # noqa: BLE001 - best-effort layer
            warnings.warn(f"canopy fetch failed, continuing without it: {exc}")

    building = None
    if use_buildings:
        try:
            from launchpoint.data.buildings import fetch_building_height

            building = fetch_building_height(occluder, projector)
        except Exception as exc:  # noqa: BLE001 - best-effort layer
            warnings.warn(f"building fetch failed, continuing without it: {exc}")

    ground = derive_bare_earth(occluder, canopy, building)
    launch_weight = launch_feasibility_weight(canopy, occluder, config.canopy)

    if use_cache:
        cache.save(keys["occluder"], occluder)
        cache.save(keys["ground"], ground)
        cache.save(keys["launch"], launch_weight)

    return SurfaceStack(
        occluder=occluder,
        ground=ground,
        launch_weight=launch_weight,
        canopy_height=canopy,
        building_height=building,
    )
