"""High-level orchestration: a flight zone in, a launch-coverage heatmap out.

Mirrors ``pipeline.find_origin``'s offline/online split: pass a precomputed
``occluder``/``projector`` for the synthetic/offline/test path (no network
touches happen), or leave them ``None`` to fetch the surface stack for the
zone's AOI via ``build_surface_stack``.
"""

from __future__ import annotations

import dataclasses

from launchpoint.config import Config
from launchpoint.core.geo import Projector, aoi_for_target_zone
from launchpoint.core.grid import RasterGrid
from launchpoint.core.target_zone import TargetZone
from launchpoint.fusion.montecarlo import ProgressCallback
from launchpoint.reverse.search import LaunchCoverageResult, LaunchSearchConfig, find_launch_coverage


def find_launch_area(
    zone: TargetZone,
    config: Config | None = None,
    search: LaunchSearchConfig | None = None,
    *,
    occluder: RasterGrid | None = None,
    ground: RasterGrid | None = None,
    projector: Projector | None = None,
    launch_weight: RasterGrid | None = None,
    progress_callback: ProgressCallback | None = None,
) -> LaunchCoverageResult:
    """Estimate the launch-coverage heatmap for a target flight zone.

    Parameters
    ----------
    zone:
        The user-drawn flight area to find launch coverage for.
    config:
        Tunables shared with the forward pipeline (range, antenna height, ...).
        ``config.altitude_is_agl`` must be True — reverse mode's viewshed
        role-swap applies ``zone.flight_altitude_m`` as a uniform offset added
        to the bare-earth ``ground`` surface, which is only meaningful for an
        AGL flight altitude (a fixed ASL altitude is not "ground + constant"
        once terrain varies across the zone).
    search:
        Candidate-grid search tunables. Defaults applied.
    occluder, ground, projector, launch_weight:
        Optional precomputed surfaces (offline/synthetic path). If ``occluder``
        is None, the data layer fetches a Copernicus GLO-30 DSM for the search
        AOI (requires network — see launchpoint.data).
    """
    config = config or Config()
    search = search or LaunchSearchConfig()

    if not config.altitude_is_agl:
        raise ValueError(
            "find_launch_area requires config.altitude_is_agl=True: flight "
            "altitude must be interpreted as height above the ground under "
            "the drone, not a fixed absolute (ASL) elevation."
        )

    if occluder is None:
        from launchpoint.data import build_surface_stack

        proj, bbox = aoi_for_target_zone(zone, config.max_range_m)
        surface_config = dataclasses.replace(config, coarse_resolution_m=search.surface_resolution_m)
        stack = build_surface_stack(
            sightings=[],
            config=surface_config,
            projector=proj,
            bbox=bbox,
            gate_canopy_with_footprint=False,
        )
        occluder = stack.occluder
        ground = stack.ground if ground is None else ground
        launch_weight = stack.launch_weight if launch_weight is None else launch_weight
        projector = proj
    elif projector is None:
        projector, _ = aoi_for_target_zone(zone, config.max_range_m)

    return find_launch_coverage(
        zone,
        config=config,
        search=search,
        occluder=occluder,
        projector=projector,
        ground=ground,
        launch_weight=launch_weight,
        progress_callback=progress_callback,
    )
