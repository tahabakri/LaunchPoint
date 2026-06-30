"""ETH Global Sentinel-2 10 m Canopy Height (2020) — keyless COG tiles.

The lighter-weight default canopy layer. Compared with the Meta/WRI 1 m product
(``canopy.py``) it is far smaller to download — a 10 m Sentinel-2/GEDI canopy-top
height instead of 1 m per-crown detail — so a run finishes much faster. Switch to
the Meta source when fine, tree-by-tree occlusion near the observer matters.

Access (no key): ETH publishes 2651 cloud-optimized GeoTIFFs as a 3 deg x 3 deg
grid on a public Nextcloud share. Each tile name encodes its **south-west corner**
(min lat, min lon), e.g. ``N39E042`` -> lat 39..42, lon 42..45. The grid is
deterministic, so we derive the intersecting tile names straight from the AOI
bounding box — there is no tile index to download (unlike the Meta source).

Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023). A high-resolution
canopy height model of the Earth. Nature Ecology & Evolution. CC-BY-4.0.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rasterio.warp import Resampling

from launchpoint.core.geo import Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.data.cog import mosaic_cogs_onto
from launchpoint.data.copernicus import lonlat_bbox_of_grid

if TYPE_CHECKING:
    from launchpoint.data.progress import FetchReporter

# Public Nextcloud share for the ETH 10 m product. The per-tile download endpoint
# takes the file via a query string, so the URL doesn't end in a bare ".tif".
_SHARE = "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt"
TILE_URL_TEMPLATE = (
    _SHARE + "/download?path=%2F3deg_cogs"
    "&files=ETH_GlobalCanopyHeight_10m_2020_{tile}_Map.tif"
)

TILE_DEG = 3
NATIVE_RES_M = 10.0

# /vsicurl tweaks for this host: the query-string URL trips GDAL's extension
# allowlist (so we drop it), and we skip HEAD probing — the share serves ranged
# GETs fine but GDAL's default HEAD-then-GET path is unreliable against it.
_GDAL_OVERRIDES: dict[str, str | None] = {
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": None,
    "CPL_VSIL_CURL_USE_HEAD": "NO",
}


def _sw_tile_name(south: int, west: int) -> str:
    """Tile id from a tile's south-west corner, e.g. (39, 42) -> ``N39E042``."""
    ns = "N" if south >= 0 else "S"
    ew = "E" if west >= 0 else "W"
    return f"{ns}{abs(south):02d}{ew}{abs(west):03d}"


def _aligned_starts(lo: float, hi: float, step: int = TILE_DEG) -> list[int]:
    """Tile lower edges (multiples of ``step``) whose [edge, edge+step] span
    overlaps the half-open interval ``[lo, hi)``."""
    starts: list[int] = []
    v = math.floor(lo / step) * step
    while v < hi:
        if v + step > lo:
            starts.append(int(v))
        v += step
    return starts


def canopy_tile_urls(target: RasterGrid, projector: Projector) -> list[str]:
    """URLs of the ETH canopy tiles intersecting the AOI (no index lookup)."""
    bb = lonlat_bbox_of_grid(target, projector)
    souths = _aligned_starts(bb.miny, bb.maxy)
    wests = _aligned_starts(bb.minx, bb.maxx)
    return [
        TILE_URL_TEMPLATE.format(tile=_sw_tile_name(s, w))
        for s in souths
        for w in wests
    ]


def _estimate_raw_bytes(target: RasterGrid) -> float:
    """Indicative bytes a windowed read pulls: the AOI area at 10 m, uint8."""
    px = (target.bounds.width / NATIVE_RES_M) * (target.bounds.height / NATIVE_RES_M)
    return px  # ~1 byte/px


def fetch_canopy_height_eth(
    target: RasterGrid,
    projector: Projector,
    cache_dir: str | None = None,  # unused (index-free); kept for signature parity
    reporter: "FetchReporter | None" = None,
) -> RasterGrid:
    """Fetch + mosaic the ETH 10 m canopy-height layer onto ``target``.

    Uses MAX resampling so the tallest canopy in a coarse cell dominates (the
    source declares nodata=255, so missing/ocean values never leak in). Cells
    with no canopy data come back NaN and are treated as 0 downstream.
    """
    urls = canopy_tile_urls(target, projector)
    if reporter is not None:
        from launchpoint.data.progress import human_bytes

        reporter.layer_start(
            "canopy",
            len(urls),
            note=(
                f"ETH 10 m, window ~{target.bounds.width / 1000:.1f} x "
                f"{target.bounds.height / 1000:.1f} km, ~{human_bytes(_estimate_raw_bytes(target))} raw"
            ),
        )
    if not urls:
        return target.like(fill=0.0)
    return mosaic_cogs_onto(
        urls,
        target,
        resampling=Resampling.max,
        missing_ok=True,
        reporter=reporter,
        layer="canopy",
        gdal_env_overrides=_GDAL_OVERRIDES,
    )
