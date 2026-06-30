"""Meta / WRI Global Canopy Height Map (1 m) — keyless, AWS Open Data.

Per-pixel tree height usable directly as an occluder and a down-weight — single
tree resolution is already baked in, so no ML is needed to "sketch" trees.

Access (no key): the dataset publishes a tile index GeoJSON and per-tile COGs in
the public ``dataforgood-fb-forests`` bucket over anonymous HTTPS. We read the
index, pick tiles intersecting the AOI, and mosaic them onto the target grid
using a MAX resampling rule (1 m -> 30 m): a single tall tree in a coarse cell
should still occlude it.

Hansen ``treecover2000`` is the lighter-weight fallback (see ``hansen.py``) when
the 1 m tiles are impractical — but it only gives % cover, not height.
"""

from __future__ import annotations

import json
import os

import requests
from rasterio.warp import Resampling
from shapely.geometry import box, shape

from launchpoint.core.geo import Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.data.copernicus import lonlat_bbox_of_grid
from launchpoint.data.cog import mosaic_cogs_onto

# Public, keyless endpoints for the Meta/WRI canopy-height product.
# Bucket: dataforgood-fb-data (anonymous HTTPS). The tile index is a ~15 MB
# GeoJSON of 56k polygons, each with a 'tile' id; CHM COGs live under chm/.
_BASE = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"
TILE_INDEX_URL = f"{_BASE}/tiles.geojson"
TILE_URL_TEMPLATE = _BASE + "/chm/{tile}.tif"


def _load_tile_index(cache_dir: str | None = None, timeout: int = 120) -> list[dict]:
    """Load the canopy tile index, caching the ~15 MB GeoJSON to disk."""
    cache_path = os.path.join(cache_dir, "meta_canopy_tiles.geojson") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)["features"]
    resp = requests.get(TILE_INDEX_URL, timeout=timeout)
    resp.raise_for_status()
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
    return json.loads(resp.text)["features"]


def canopy_tile_urls(
    target: RasterGrid, projector: Projector, cache_dir: str | None = None
) -> list[str]:
    """URLs of canopy COG tiles intersecting the AOI (queries the tile index)."""
    bb = lonlat_bbox_of_grid(target, projector)
    aoi = box(bb.minx, bb.miny, bb.maxx, bb.maxy)
    urls = []
    for feat in _load_tile_index(cache_dir):
        if shape(feat["geometry"]).intersects(aoi):
            tile = feat["properties"].get("tile") or feat["properties"].get("quadkey")
            if tile:
                urls.append(TILE_URL_TEMPLATE.format(tile=tile))
    return urls


def fetch_canopy_height(
    target: RasterGrid, projector: Projector, cache_dir: str | None = None
) -> RasterGrid:
    """Fetch + mosaic the canopy-height layer onto ``target`` (DSM grid).

    Uses MAX resampling so the tallest 1 m tree dominates its coarse cell. Cells
    with no canopy data come back NaN and are treated as 0 downstream.
    """
    urls = canopy_tile_urls(target, projector, cache_dir)
    if not urls:
        return target.like(fill=0.0)
    return mosaic_cogs_onto(urls, target, resampling=Resampling.max, missing_ok=True)
