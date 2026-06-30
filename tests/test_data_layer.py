import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

from launchpoint.config import CanopyConfig
from launchpoint.core.geo import BBox, Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.data.bare_earth import derive_bare_earth, launch_feasibility_weight
from launchpoint.data.buildings import _parse_height, rasterize_buildings
from launchpoint.data.copernicus import (
    glo30_tile_id,
    glo30_tile_url,
    tiles_for_lonlat_bbox,
)


# --- GLO-30 tile naming / enumeration (no network) -------------------------
def test_glo30_tile_id_naming():
    assert glo30_tile_id(47, 8) == "Copernicus_DSM_COG_10_N47_00_E008_00_DEM"
    assert glo30_tile_id(-34, -59) == "Copernicus_DSM_COG_10_S34_00_W059_00_DEM"


def test_glo30_url():
    url = glo30_tile_url(47, 8)
    assert url.startswith("https://copernicus-dem-30m.s3.amazonaws.com/")
    assert url.endswith("Copernicus_DSM_COG_10_N47_00_E008_00_DEM.tif")


def test_tiles_for_bbox_spans_integer_degrees():
    urls = tiles_for_lonlat_bbox(7.6, 46.9, 8.4, 47.3)
    # lon 7,8 x lat 46,47 -> 4 tiles
    assert len(urls) == 4


# --- bare earth derivation (no network) ------------------------------------
def _grid(fill):
    bbox = BBox(0, 0, 300, 300)
    g = RasterGrid.empty(bbox, 30.0, CRS.from_epsg(32632), fill=fill)
    return g


def test_bare_earth_subtracts_objects():
    dsm = _grid(100.0)
    canopy = _grid(0.0)
    building = _grid(0.0)
    canopy.data[0, 0] = 20.0       # a tall tree
    building.data[1, 1] = 12.0     # a building
    bare = derive_bare_earth(dsm, canopy, building)
    assert np.isclose(bare.data[0, 0], 80.0)
    assert np.isclose(bare.data[1, 1], 88.0)
    # Open ground unchanged.
    assert np.isclose(bare.data[5, 5], 100.0)


def test_bare_earth_flat_field_equals_dsm():
    # On flat open terrain (no canopy/buildings) bare earth == DSM.
    dsm = _grid(250.0)
    bare = derive_bare_earth(dsm, None, None)
    assert np.allclose(bare.data, dsm.data)


def test_bare_earth_never_exceeds_dsm():
    dsm = _grid(100.0)
    canopy = _grid(-5.0)  # noisy negative
    bare = derive_bare_earth(dsm, canopy, None)
    assert np.all(bare.data <= dsm.data + 1e-5)


def test_launch_weight_ramps_with_canopy():
    ref = _grid(0.0)
    canopy = _grid(0.0)
    canopy.data[0, 0] = 0.0     # open
    canopy.data[0, 1] = 30.0    # tall closed canopy
    cfg = CanopyConfig()
    w = launch_feasibility_weight(canopy, ref, cfg)
    assert np.isclose(w.data[0, 0], 1.0)
    assert np.isclose(w.data[0, 1], cfg.min_launch_weight)


# --- building parsing / rasterization (no network) -------------------------
def test_parse_height_variants():
    assert _parse_height({"height": "12"}) == 12.0
    assert _parse_height({"height": "9 m"}) == 9.0
    assert _parse_height({"building:levels": "4"}) == 12.0
    assert _parse_height({}) == 6.0


def test_rasterize_buildings_max_overlap():
    g = _grid(0.0)
    # Two overlapping squares of different height; the taller must win.
    short = Polygon([(0, 0), (150, 0), (150, 150), (0, 150)])
    tall = Polygon([(60, 60), (300, 60), (300, 300), (60, 300)])
    rg = rasterize_buildings([(short, 5.0), (tall, 20.0)], g)
    # A cell inside the overlap region holds the taller height.
    r, c = g.world_to_pixel(100, 100)
    assert np.isclose(rg.data[r, c], 20.0)


# --- network-gated real DSM fetch ------------------------------------------
@pytest.mark.network
def test_fetch_real_dsm_small_aoi():
    from launchpoint.data.copernicus import fetch_dsm

    proj = Projector.for_point(8.55, 47.37)
    cx, cy = proj.to_utm(8.55, 47.37)
    bbox = BBox(cx - 500, cy - 500, cx + 500, cy + 500)
    target = RasterGrid.empty(bbox, 30.0, proj.utm_crs, fill=np.nan)
    dsm = fetch_dsm(target, proj)
    finite = dsm.data[np.isfinite(dsm.data)]
    assert finite.size > 0
    # Zurich is ~400-600 m ASL; sanity-bound the values.
    assert 300 < float(np.median(finite)) < 900


@pytest.mark.network
def test_fetch_real_canopy():
    from launchpoint.data.canopy import fetch_canopy_height

    proj = Projector.for_point(8.20, 48.10)  # Black Forest
    cx, cy = proj.to_utm(8.20, 48.10)
    bbox = BBox(cx - 600, cy - 600, cx + 600, cy + 600)
    target = RasterGrid.empty(bbox, 30.0, proj.utm_crs, fill=np.nan)
    ch = fetch_canopy_height(target, proj)
    finite = ch.data[np.isfinite(ch.data)]
    assert finite.size > 0
    assert float(np.nanmax(ch.data)) > 5.0  # there are trees here


@pytest.mark.network
def test_fetch_real_osm_buildings():
    from launchpoint.data.buildings import fetch_building_height

    proj = Projector.for_point(8.54, 47.37)  # central Zurich
    cx, cy = proj.to_utm(8.54, 47.37)
    bbox = BBox(cx - 400, cy - 400, cx + 400, cy + 400)
    target = RasterGrid.empty(bbox, 30.0, proj.utm_crs, fill=np.nan)
    bh = fetch_building_height(target, proj)
    assert int((bh.data > 0).sum()) > 0  # a city has buildings
