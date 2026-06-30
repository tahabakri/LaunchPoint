"""RasterGrid: a 2-D height/value surface tied to a CRS and an affine transform.

Thin wrapper over a numpy array plus a rasterio Affine. Everything downstream
(viewshed, fusion, enrichment) operates on aligned ``RasterGrid`` instances so
that pixel (row, col) <-> world (x, y) conversions are unambiguous and shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS

from launchpoint.core.geo import BBox


@dataclass
class RasterGrid:
    """A georeferenced 2-D array.

    Attributes
    ----------
    data:
        2-D float array, shape (rows, cols). NaN marks nodata.
    transform:
        rasterio/affine ``Affine`` mapping (col, row) -> (x, y) at pixel corners.
    crs:
        Coordinate reference system of ``transform``.
    """

    data: np.ndarray
    transform: Affine
    crs: CRS

    # --- shape helpers ------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    @property
    def res_x(self) -> float:
        return abs(self.transform.a)

    @property
    def res_y(self) -> float:
        return abs(self.transform.e)

    @property
    def bounds(self) -> BBox:
        rows, cols = self.shape
        x0, y0 = self.transform * (0, 0)
        x1, y1 = self.transform * (cols, rows)
        return BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    # --- coordinate conversions --------------------------------------------
    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) -> (row, col), floored to the containing pixel."""
        col, row = ~self.transform * (x, y)
        return int(np.floor(row)), int(np.floor(col))

    def pixel_to_world(self, row: float, col: float) -> tuple[float, float]:
        """Pixel (row, col) center -> world (x, y)."""
        return self.transform * (col + 0.5, row + 0.5)

    def contains_pixel(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    # --- factories ----------------------------------------------------------
    @classmethod
    def empty(
        cls,
        bbox: BBox,
        resolution: float,
        crs: CRS,
        fill: float = np.nan,
        dtype=np.float32,
    ) -> "RasterGrid":
        """Allocate a north-up grid covering ``bbox`` at the given resolution."""
        cols = max(1, int(np.ceil(bbox.width / resolution)))
        rows = max(1, int(np.ceil(bbox.height / resolution)))
        # Top-left origin; y decreases downward (negative e).
        transform = Affine(resolution, 0.0, bbox.minx, 0.0, -resolution, bbox.maxy)
        data = np.full((rows, cols), fill, dtype=dtype)
        return cls(data=data, transform=transform, crs=crs)

    def like(self, fill: float = np.nan, dtype=None) -> "RasterGrid":
        """A new grid with the same georeferencing, filled with ``fill``."""
        dt = dtype or self.data.dtype
        return RasterGrid(
            data=np.full(self.shape, fill, dtype=dt),
            transform=self.transform,
            crs=self.crs,
        )

    def copy_with(self, data: np.ndarray) -> "RasterGrid":
        """Same georeferencing, new data array (shape must match)."""
        if data.shape != self.shape:
            raise ValueError(f"shape mismatch: {data.shape} vs {self.shape}")
        return RasterGrid(data=data, transform=self.transform, crs=self.crs)

    def resample_to(self, like: "RasterGrid", resampling: str = "bilinear") -> "RasterGrid":
        """Resample this grid's data onto another grid's georeferencing.

        Used by the coarse-to-fine pass to lift coarse surfaces onto a fine
        patch grid (and vice versa). ``like`` may differ in resolution, extent
        and origin but must share the CRS.
        """
        from rasterio.warp import Resampling, reproject

        if self.crs != like.crs:
            raise ValueError("resample_to requires matching CRS")
        method = getattr(Resampling, resampling)
        dest = np.full(like.shape, np.nan, dtype=np.float32)
        reproject(
            source=np.ascontiguousarray(self.data, dtype=np.float32),
            destination=dest,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=np.nan,
            dst_transform=like.transform,
            dst_crs=like.crs,
            dst_nodata=np.nan,
            resampling=method,
        )
        return like.copy_with(dest)

    def subgrid(self, bbox: BBox) -> "RasterGrid":
        """A view-aligned crop covering ``bbox`` (snapped to this grid's pixels)."""
        r0, c0 = self.world_to_pixel(bbox.minx, bbox.maxy)
        r1, c1 = self.world_to_pixel(bbox.maxx, bbox.miny)
        r0, r1 = max(0, min(r0, r1)), min(self.rows, max(r0, r1) + 1)
        c0, c1 = max(0, min(c0, c1)), min(self.cols, max(c0, c1) + 1)
        sub = self.data[r0:r1, c0:c1]
        # Corner of pixel (r0, c0) is exactly transform * (c0, r0).
        corner_x, corner_y = self.transform * (c0, r0)
        transform = Affine(self.transform.a, 0.0, corner_x,
                           0.0, self.transform.e, corner_y)
        return RasterGrid(data=sub.copy(), transform=transform, crs=self.crs)

    # --- io -----------------------------------------------------------------
    def write_geotiff(self, path: str, nodata: float = np.nan) -> None:
        rows, cols = self.shape
        profile = {
            "driver": "GTiff",
            "height": rows,
            "width": cols,
            "count": 1,
            "dtype": "float32",
            "crs": self.crs,
            "transform": self.transform,
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
        }
        out = self.data.astype(np.float32)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(out, 1)

    @classmethod
    def read_geotiff(cls, path: str) -> "RasterGrid":
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None and not np.isnan(src.nodata):
                data[data == src.nodata] = np.nan
            return cls(data=data, transform=src.transform, crs=src.crs)
