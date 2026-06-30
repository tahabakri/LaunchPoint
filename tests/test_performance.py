"""Phase 5: GPU dispatch falls back cleanly, and coarse-to-fine refines only
the hot patches while still recovering the controller."""

import numpy as np

from launchpoint.config import Config, MonteCarloConfig
from launchpoint.data.surface import SurfaceStack
from launchpoint.fusion.multiscale import (
    MultiscaleConfig,
    ResamplingSurfaceProvider,
    coarse_to_fine,
)
from launchpoint.synthetic import make_default_scenario
from launchpoint.viewshed.gpu import gpu_available, r3_viewshed_best
from launchpoint.viewshed.kernels import r3_viewshed


def test_gpu_dispatch_matches_cpu_when_no_device():
    # On a machine with no CUDA device, the "best" kernel must equal the CPU one.
    sc = make_default_scenario(seed=5)
    surf = np.ascontiguousarray(sc.surface.data, dtype=np.float64)
    r, c = sc.controller_pixel()
    args = (surf, surf, r, c, float(surf[r, c]) + 100.0, 1.5, sc.surface.res_x, 0.0, 1e12)
    best = r3_viewshed_best(*args, prefer_gpu=True)
    cpu = r3_viewshed(*args)
    assert np.array_equal(best, cpu)


def test_gpu_available_is_boolean():
    assert isinstance(gpu_available(), bool)


def _stack_from_scenario(sc):
    occ = sc.surface.copy_with(sc.surface.data.astype(np.float32))
    ground = occ
    lw = occ.like(fill=1.0)
    return SurfaceStack(occluder=occ, ground=ground, launch_weight=lw)


def test_coarse_to_fine_recovers_and_saves_work():
    sc = make_default_scenario(n_sightings=4, seed=5)
    # Keep the AOI close to the synthetic extent so the resampling provider has
    # coverage, and keep it fast.
    config = Config(
        max_range_m=2500.0,
        monte_carlo=MonteCarloConfig(samples_per_sighting=8, seed=1),
    )
    provider = ResamplingSurfaceProvider(_stack_from_scenario(sc), sc.projector)
    ms = MultiscaleConfig(coarse_resolution_m=30.0, fine_resolution_m=10.0,
                         sighting_buffer_m=400.0)

    result = coarse_to_fine(sc.sightings, config, provider, ms)

    # Coarse pass recovers the controller region.
    cr, cc = result.coarse.probability.world_to_pixel(*sc.controller_xy)
    assert result.coarse.probability.data[cr, cc] > 0.4

    # Refinement happened, and it processed far fewer cells than a full fine disk.
    assert len(result.fine_patches) >= 1
    assert result.fine_cells < result.full_fine_cells
    assert result.speedup_vs_full_fine > 2.0

    # Some fine patch covers the true controller and scores it highly.
    covered = False
    for est, bb in zip(result.fine_patches, result.patch_bounds):
        if (bb.minx <= sc.controller_xy[0] <= bb.maxx
                and bb.miny <= sc.controller_xy[1] <= bb.maxy):
            fr, fc = est.probability.world_to_pixel(*sc.controller_xy)
            if est.probability.contains_pixel(fr, fc):
                covered = True
                assert est.probability.data[fr, fc] > 0.3
    assert covered, "no fine patch covered the true controller"
