"""Local on-disk cache so repeated runs over the same AOI never refetch.

Cache keys are derived from the AOI bounds + CRS + resolution + layer name, so a
second run with the same parameters reads the GeoTIFF straight off disk.
"""

from __future__ import annotations

import hashlib
import os

from launchpoint.core.geo import BBox
from launchpoint.core.grid import RasterGrid


def cache_key(layer: str, bbox: BBox, epsg: int, resolution: float) -> str:
    raw = f"{layer}|{bbox.as_tuple()}|{epsg}|{resolution:.3f}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"{layer}_{epsg}_{int(resolution)}m_{digest}.tif"


class Cache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key)

    def has(self, key: str) -> bool:
        return os.path.exists(self.path(key))

    def load(self, key: str) -> RasterGrid:
        return RasterGrid.read_geotiff(self.path(key))

    def save(self, key: str, grid: RasterGrid) -> None:
        grid.write_geotiff(self.path(key))
