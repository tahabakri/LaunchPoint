import os

import numpy as np
from pyproj import CRS

from launchpoint.core.geo import BBox
from launchpoint.core.grid import RasterGrid


def _grid():
    bbox = BBox(500000, 5000000, 506000, 5006000)  # 6 km box, UTM-ish
    return RasterGrid.empty(bbox, resolution=30.0, crs=CRS.from_epsg(32632), fill=0.0)


def test_empty_grid_shape_and_res():
    g = _grid()
    assert g.rows == 200 and g.cols == 200
    assert abs(g.res_x - 30.0) < 1e-9
    assert abs(g.res_y - 30.0) < 1e-9


def test_world_pixel_roundtrip():
    g = _grid()
    for (r, c) in [(0, 0), (100, 50), (199, 199)]:
        x, y = g.pixel_to_world(r, c)
        rr, cc = g.world_to_pixel(x, y)
        assert (rr, cc) == (r, c)


def test_geotiff_roundtrip(tmp_path):
    g = _grid()
    g.data[:] = np.random.default_rng(0).random(g.shape).astype(np.float32)
    path = os.path.join(tmp_path, "out.tif")
    g.write_geotiff(path)
    h = RasterGrid.read_geotiff(path)
    assert h.shape == g.shape
    assert np.allclose(h.data, g.data, atol=1e-5)
    assert h.crs.to_epsg() == 32632
