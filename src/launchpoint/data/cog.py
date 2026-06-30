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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from launchpoint.core.grid import RasterGrid
from launchpoint.data.progress import FetchReporter

# Remote COG tiles are read concurrently. The S3 buckets (Copernicus, Meta) have
# no meaningful per-client rate limit and GDAL releases the GIL during I/O, so a
# small pool turns "one tile at a time" into a parallel pull. Kept modest to stay
# polite and avoid saturating a typical connection.
DEFAULT_MAX_WORKERS = 6

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


def _short_name(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def mosaic_cogs_onto(
    urls: list[str],
    target: RasterGrid,
    *,
    resampling: Resampling = Resampling.bilinear,
    missing_ok: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS,
    reporter: FetchReporter | None = None,
    layer: str = "",
) -> RasterGrid:
    """Mosaic several COG tiles onto ``target``, filling gaps tile by tile.

    Tiles are fetched **concurrently** (each worker opens its own GDAL handle and
    environment, since GDAL config is thread-local) but composited in the
    original order, so a later tile only fills cells still NaN and overlaps keep
    the first tile's value — identical to a serial mosaic. Missing/unreachable
    tiles are skipped when ``missing_ok`` (ocean tiles legitimately don't exist
    for GLO-30). ``reporter`` receives per-tile timing for logging + UI progress.
    """
    out = np.full(target.shape, np.nan, dtype=np.float32)
    n = len(urls)
    if n == 0:
        return target.copy_with(out)

    def _fetch(index: int, url: str) -> tuple[int, np.ndarray | None, float]:
        start = time.perf_counter()
        with gdal_env():
            try:
                arr = reproject_cog_onto(url, target, resampling=resampling)
            except rasterio.errors.RasterioIOError:
                if missing_ok:
                    arr = None
                else:
                    raise
        return index, arr, time.perf_counter() - start

    results: list[np.ndarray | None] = [None] * n
    workers = max(1, min(max_workers, n))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, i, url) for i, url in enumerate(urls)]
        done = 0
        for future in as_completed(futures):
            index, arr, secs = future.result()
            results[index] = arr
            done += 1
            if reporter is not None:
                reporter.tile(layer or "tiles", done, n, _short_name(urls[index]), secs)

    # Composite in the original tile order so overlap precedence is deterministic.
    for arr in results:
        if arr is None:
            continue
        gap = ~np.isfinite(out)
        out[gap] = arr[gap]
    return target.copy_with(out)
