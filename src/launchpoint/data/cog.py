"""Windowed Cloud-Optimized GeoTIFF reader.

COGs let GDAL/rasterio fetch only the bytes inside the area of interest via HTTP
range requests — no whole-layer download. This module reprojects/resamples the
overlapping part of one or more remote COGs onto a target ``RasterGrid``, so the
caller gets data already aligned to the common UTM analysis grid.

No credentials are used: HTTPS COGs are read through GDAL's ``/vsicurl/`` and S3
paths through anonymous access (``AWS_NO_SIGN_REQUEST=YES``).
"""

from __future__ import annotations

import contextlib

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from launchpoint.core.grid import RasterGrid

# GDAL knobs that make remote COG reads fast and keyless.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def _vsi_url(url: str) -> str:
    if url.startswith(("http://", "https://")):
        return "/vsicurl/" + url
    if url.startswith("s3://"):
        return "/vsis3/" + url[len("s3://"):]
    return url


@contextlib.contextmanager
def gdal_env():
    with rasterio.Env(**_GDAL_ENV):
        yield


def reproject_cog_onto(
    url: str,
    target: RasterGrid,
    *,
    resampling: Resampling = Resampling.bilinear,
    band: int = 1,
) -> np.ndarray:
    """Reproject the part of a remote COG overlapping ``target`` onto its grid.

    Returns a float32 array shaped like ``target.data`` with NaN where the COG
    has no data / no coverage. Only the overlapping COG blocks are fetched.
    """
    dest = np.full(target.shape, np.nan, dtype=np.float32)
    with rasterio.open(_vsi_url(url)) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=dest,
            src_crs=src.crs,
            src_transform=src.transform,
            src_nodata=src.nodata,
            dst_crs=target.crs,
            dst_transform=target.transform,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dest


def mosaic_cogs_onto(
    urls: list[str],
    target: RasterGrid,
    *,
    resampling: Resampling = Resampling.bilinear,
    missing_ok: bool = True,
) -> RasterGrid:
    """Mosaic several COG tiles onto ``target``, filling gaps tile by tile.

    Tiles are composited in order; a later tile only fills cells still NaN, so
    overlaps keep the first tile's value. Missing/unreachable tiles are skipped
    when ``missing_ok`` (ocean tiles legitimately don't exist for GLO-30).
    """
    out = np.full(target.shape, np.nan, dtype=np.float32)
    with gdal_env():
        for url in urls:
            try:
                layer = reproject_cog_onto(url, target, resampling=resampling)
            except rasterio.errors.RasterioIOError:
                if missing_ok:
                    continue
                raise
            gap = ~np.isfinite(out)
            out[gap] = layer[gap]
    return target.copy_with(out)
