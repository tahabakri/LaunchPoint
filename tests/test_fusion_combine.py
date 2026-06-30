"""N1 + N2: cross-sighting combine rule and the canopy-fetch footprint gate.

Both are offline: ``combine_contributions`` is pure numpy, and
``reachable_footprint`` / ``_fetch_canopy_in_footprint`` run against the
synthetic surface with a stubbed canopy fetch (no network).
"""

import time

import numpy as np
import pytest
import rasterio
from pyproj import CRS

from launchpoint.config import Config, MonteCarloConfig
from launchpoint.core.geo import BBox
from launchpoint.core.grid import RasterGrid
from launchpoint.data import cog as cog_mod
from launchpoint.data import surface as surface_mod
from launchpoint.fusion.montecarlo import combine_contributions, reachable_footprint
from launchpoint.synthetic import make_default_scenario


# --- N1: combine rule -------------------------------------------------------
def test_min_gates_on_the_weakest_sighting():
    # Four sightings see a cell strongly, one barely. Strict 'min' reports the
    # weakest (0.2) — geometric mean would dilute it to ~0.72 and keep it hot.
    contribs = [np.array([[0.9]]), np.array([[1.0]]), np.array([[0.95]]),
                np.array([[1.0]]), np.array([[0.2]])]
    strict = combine_contributions(contribs, "min")
    soft = combine_contributions(contribs, "geometric_mean")
    assert strict[0, 0] == pytest.approx(0.2)
    assert soft[0, 0] > 0.6  # the dilution the screenshot showed
    # A pin that cannot see the cell at all gates it to exactly 0.
    blind = combine_contributions(contribs + [np.array([[0.0]])], "min")
    assert blind[0, 0] == 0.0


def test_geometric_mean_is_soft_intersection():
    seen_by_all = np.array([[0.9, 0.8]])
    seen_by_one = np.array([[0.0, 0.8]])  # first cell invisible to one sighting
    out = combine_contributions([seen_by_all, seen_by_one], "geometric_mean")
    # A cell any sighting cannot see collapses to 0 (the "common area" rule)...
    assert out[0, 0] == 0.0
    # ...while a cell all sightings see stays high (geomean of 0.8 and 0.8).
    assert out[0, 1] == pytest.approx(0.8, abs=1e-6)


def test_arithmetic_mean_is_soft_union():
    seen_by_all = np.array([[1.0]])
    seen_by_one = np.array([[0.0]])
    out = combine_contributions([seen_by_all, seen_by_one], "arithmetic_mean")
    assert out[0, 0] == pytest.approx(0.5)


def test_single_sighting_is_unchanged_by_mode():
    c = np.array([[0.3, 0.7, 1.0]])
    geo = combine_contributions([c], "geometric_mean")
    ari = combine_contributions([c], "arithmetic_mean")
    assert np.allclose(geo, c)
    assert np.allclose(ari, c)


def test_unknown_combine_mode_raises():
    with pytest.raises(ValueError):
        combine_contributions([np.zeros((1, 1))], "median")


# --- N2: reachable footprint + gated canopy fetch ---------------------------
def _scenario_config():
    return Config(monte_carlo=MonteCarloConfig(samples_per_sighting=4, seed=1))


def test_reachable_footprint_is_a_proper_subset():
    sc = make_default_scenario(n_sightings=4, seed=5)
    mask = reachable_footprint(
        sc.sightings, sc.surface, _scenario_config(), sc.projector
    )
    assert mask.dtype == bool
    assert mask.shape == sc.surface.shape
    # Some cells reachable, but not the whole grid (the DSM occludes something).
    assert mask.any()
    assert not mask.all()


def test_min_footprint_is_the_intersection_not_the_union():
    sc = make_default_scenario(n_sightings=4, seed=5)
    inter = reachable_footprint(
        sc.sightings, sc.surface, Config(combine="min"), sc.projector
    )
    union = reachable_footprint(
        sc.sightings, sc.surface, Config(combine="arithmetic_mean"), sc.projector
    )
    # Strict-min gates canopy to the common region, a subset of the union — so
    # it can only ever fetch *less* canopy than the union footprint.
    assert inter.sum() < union.sum()
    assert (union | inter).sum() == union.sum()  # intersection ⊆ union
    # The common region is still non-empty (the planted controller is in it).
    assert inter.any()
    cr, cc = sc.controller_pixel()
    assert inter[cr, cc]


def test_footprint_gated_fetch_only_fills_inside_footprint(monkeypatch):
    sc = make_default_scenario(n_sightings=4, seed=5)
    occluder = sc.surface

    # Stub the network canopy fetch: return a constant-height grid for whatever
    # (sub)grid it is handed, so we can see exactly where it lands.
    def fake_fetch(target, projector, cache_dir=None, reporter=None):
        return target.copy_with(np.full(target.shape, 7.0, dtype=np.float32))

    monkeypatch.setattr(surface_mod, "fetch_canopy_height", fake_fetch, raising=False)
    import launchpoint.data.canopy as canopy_mod
    monkeypatch.setattr(canopy_mod, "fetch_canopy_height", fake_fetch, raising=False)

    footprint = np.zeros(occluder.shape, dtype=bool)
    footprint[20:40, 30:55] = True  # a strict interior block

    canopy = surface_mod._fetch_canopy_in_footprint(
        occluder, sc.projector, None, footprint
    )

    filled = np.isfinite(canopy.data)
    # Every filled cell lies within the footprint's bounding box, and the
    # footprint cells themselves are filled. Cells well outside stay NaN.
    assert filled[20:40, 30:55].all()
    assert not filled[0, 0]
    assert not filled[-1, -1]
    # The fetch window is the footprint bbox, far smaller than the full grid.
    assert filled.sum() < occluder.data.size


# --- N3: parallel mosaic stays order-stable ---------------------------------
def _small_grid():
    return RasterGrid.empty(BBox(0, 0, 300, 300), 30.0, CRS.from_epsg(32632), fill=np.nan)


def test_parallel_mosaic_is_order_stable_under_reordered_completion(monkeypatch):
    target = _small_grid()

    # url0 fills everything with 1.0 but finishes slowly; url1 fills with 2.0
    # quickly. First-tile-wins must still leave the overlap as 1.0 even though
    # url1's worker completes first.
    def fake_reproject(url, tgt, *, resampling=None, band=1):
        if url == "url0":
            time.sleep(0.05)
            return np.full(tgt.shape, 1.0, dtype=np.float32)
        if url == "url1":
            return np.full(tgt.shape, 2.0, dtype=np.float32)
        raise rasterio.errors.RasterioIOError("missing")  # url2: skipped

    monkeypatch.setattr(cog_mod, "reproject_cog_onto", fake_reproject)
    out = cog_mod.mosaic_cogs_onto(["url0", "url1", "url2"], target, missing_ok=True)
    assert np.allclose(out.data, 1.0)  # earlier tile wins regardless of timing


def test_parallel_mosaic_raises_on_missing_when_not_ok(monkeypatch):
    target = _small_grid()

    def fake_reproject(url, tgt, *, resampling=None, band=1):
        raise rasterio.errors.RasterioIOError("missing")

    monkeypatch.setattr(cog_mod, "reproject_cog_onto", fake_reproject)
    with pytest.raises(rasterio.errors.RasterioIOError):
        cog_mod.mosaic_cogs_onto(["u"], target, missing_ok=False)
