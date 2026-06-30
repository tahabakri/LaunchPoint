"""Copernicus GLO-30 DSM (~30 m) — keyless, AWS Open Data Registry.

The DSM is the *surface* model: it already includes buildings and tree canopy,
which is exactly what we want for occlusion. Tiles are 1°x1° COGs named by their
SW corner, served over anonymous HTTPS from the ``copernicus-dem-30m`` bucket.

Bucket / access (no key, no account):
    https://copernicus-dem-30m.s3.amazonaws.com/
    s3://copernicus-dem-30m/   (anonymous)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from launchpoint.core.geo import BBox, Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.data.cog import mosaic_cogs_onto

if TYPE_CHECKING:
    from launchpoint.data.progress import FetchReporter

BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def glo30_tile_id(lat_deg: int, lon_deg: int) -> str:
    """Tile id for the 1°x1° cell whose SW corner is (lat_deg, lon_deg)."""
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"


def glo30_tile_url(lat_deg: int, lon_deg: int) -> str:
    tid = glo30_tile_id(lat_deg, lon_deg)
    return f"{BASE_URL}/{tid}/{tid}.tif"


def tiles_for_lonlat_bbox(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> list[str]:
    """URLs of every 1° GLO-30 tile intersecting a lon/lat bounding box."""
    urls = []
    for lat in range(math.floor(lat_min), math.floor(lat_max) + 1):
        for lon in range(math.floor(lon_min), math.floor(lon_max) + 1):
            urls.append(glo30_tile_url(lat, lon))
    return urls


def lonlat_bbox_of_grid(grid: RasterGrid, projector: Projector, n: int = 8) -> BBox:
    """Lon/lat bounds of a UTM grid, sampling its perimeter to be safe.

    Reprojecting just the 4 corners can under-cover when the UTM box is large;
    sampling ``n`` points per edge gives a robust geographic envelope.
    """
    b = grid.bounds
    xs = np.concatenate([
        np.linspace(b.minx, b.maxx, n), np.linspace(b.minx, b.maxx, n),
        np.full(n, b.minx), np.full(n, b.maxx),
    ])
    ys = np.concatenate([
        np.full(n, b.miny), np.full(n, b.maxy),
        np.linspace(b.miny, b.maxy, n), np.linspace(b.miny, b.maxy, n),
    ])
    lon, lat = projector.to_lonlat(xs, ys)
    return BBox(float(np.min(lon)), float(np.min(lat)),
               float(np.max(lon)), float(np.max(lat)))


def fetch_dsm(
    target: RasterGrid,
    projector: Projector,
    reporter: "FetchReporter | None" = None,
) -> RasterGrid:
    """Fetch + mosaic + align the GLO-30 DSM onto ``target`` (UTM grid).

    Requires network on first call; cache the result upstream for offline reuse.
    """
    bb = lonlat_bbox_of_grid(target, projector)
    urls = tiles_for_lonlat_bbox(bb.minx, bb.miny, bb.maxx, bb.maxy)
    if reporter is not None:
        reporter.layer_start("DSM", len(urls), note="Copernicus GLO-30, ~30 m")
    return mosaic_cogs_onto(urls, target, missing_ok=True, reporter=reporter, layer="DSM")
