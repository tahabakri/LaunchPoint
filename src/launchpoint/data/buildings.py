"""OpenStreetMap building footprints -> extruded height raster — keyless.

Footprints come from the public Overpass API (no key). Each polygon is extruded
by its ``height`` tag, or ``building:levels`` x ~3 m, or a sensible default, then
rasterized (MAX rule) onto the target grid to produce a ``building_height``
raster (0 where there are no buildings). That raster feeds both occlusion-aware
bare-earth derivation and is itself already part of the DSM.

Gap-fill from Microsoft Global Building Footprints / Google Open Buildings
(bulk GeoJSON, no key) is supported via ``extra_geometries`` — see
``build_surface_stack`` — for regions where OSM coverage is sparse.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import requests
from rasterio.features import rasterize
from shapely.geometry import Polygon

from launchpoint.core.geo import Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.data.copernicus import lonlat_bbox_of_grid

if TYPE_CHECKING:
    from launchpoint.data.progress import FetchReporter

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass rejects requests without a descriptive User-Agent (HTTP 406).
HTTP_HEADERS = {"User-Agent": "LaunchPoint/0.5 (controller-origin-finder)"}
METERS_PER_LEVEL = 3.0
DEFAULT_BUILDING_HEIGHT_M = 6.0  # ~2 storeys when nothing is tagged


def _parse_height(tags: dict) -> float:
    h = tags.get("height")
    if h is not None:
        try:
            return float(str(h).split()[0].replace("m", ""))
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels is not None:
        try:
            return float(str(levels).split(";")[0]) * METERS_PER_LEVEL
        except ValueError:
            pass
    return DEFAULT_BUILDING_HEIGHT_M


def fetch_osm_buildings(
    target: RasterGrid, projector: Projector, timeout: int = 90,
    reporter: "FetchReporter | None" = None,
) -> list[tuple[Polygon, float]]:
    """Return (UTM-projected polygon, height_m) pairs for buildings in the AOI."""
    bb = lonlat_bbox_of_grid(target, projector)
    query = (
        "[out:json][timeout:{t}];"
        '(way["building"]({s},{w},{n},{e});'
        'relation["building"]({s},{w},{n},{e}););'
        "out geom;"
    ).format(t=timeout, s=bb.miny, w=bb.minx, n=bb.maxy, e=bb.maxx)

    # Overpass is rate-limited and shared, so this stays a single sequential
    # request (no parallel hammering) — only the S3 COG tiles are parallelised.
    if reporter is not None:
        reporter.layer_start("buildings", note="OSM / Overpass")
    start = time.perf_counter()
    resp = requests.post(
        OVERPASS_URL, data={"data": query}, headers=HTTP_HEADERS, timeout=timeout + 10
    )
    resp.raise_for_status()
    elements = resp.json().get("elements", [])
    if reporter is not None:
        reporter.note(
            f"overpass returned {len(elements)} building(s) in "
            f"{time.perf_counter() - start:.2f}s",
            n_bytes=len(resp.content),
        )

    out: list[tuple[Polygon, float]] = []
    for el in elements:
        geom = el.get("geometry")
        if not geom or len(geom) < 4:
            continue
        ring = [(pt["lon"], pt["lat"]) for pt in geom]
        try:
            poly_ll = Polygon(ring)
        except (ValueError, TypeError):
            continue
        if not poly_ll.is_valid or poly_ll.is_empty:
            poly_ll = poly_ll.buffer(0)
            if poly_ll.is_empty:
                continue
        # Project lon/lat -> UTM.
        lon, lat = poly_ll.exterior.coords.xy
        x, y = projector.to_utm(list(lon), list(lat))
        out.append((Polygon(zip(x, y)), _parse_height(el.get("tags", {}))))
    return out


def rasterize_buildings(
    geometries: list[tuple[Polygon, float]], target: RasterGrid
) -> RasterGrid:
    """Burn (polygon, height) pairs into a building-height raster (MAX overlap)."""
    shapes = [(poly, float(h)) for poly, h in geometries if not poly.is_empty]
    if not shapes:
        return target.like(fill=0.0)
    # rasterize's default merge is "replace" in draw order, so drawing the
    # tallest buildings last makes them win on overlapping cells (a MAX rule).
    shapes.sort(key=lambda s: s[1])
    arr = rasterize(
        shapes,
        out_shape=target.shape,
        transform=target.transform,
        fill=0.0,
        dtype="float32",
    )
    return target.copy_with(arr)


def fetch_building_height(
    target: RasterGrid, projector: Projector,
    reporter: "FetchReporter | None" = None,
) -> RasterGrid:
    """Convenience: fetch OSM footprints and rasterize to a height grid."""
    geoms = fetch_osm_buildings(target, projector, reporter=reporter)
    return rasterize_buildings(geoms, target)
