"""Geodetic helpers.

Project-wide rule: *all geometry is reprojected to a local UTM zone* so that
raycasting happens in metres, never in lat/lon degrees. A degree of longitude is
~111 km at the equator but shrinks toward the poles; doing line-of-sight math in
degrees would silently distort every distance and angle. UTM gives us a locally
near-isotropic metric grid, which is exactly what the viewshed needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS, Transformer

WGS84 = CRS.from_epsg(4326)


def utm_epsg_for(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def utm_crs_for(lon: float, lat: float) -> CRS:
    return CRS.from_epsg(utm_epsg_for(lon, lat))


@dataclass(frozen=True)
class BBox:
    """An axis-aligned bounding box in a given CRS (units follow the CRS)."""

    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    def buffered(self, pad: float) -> "BBox":
        return BBox(self.minx - pad, self.miny - pad, self.maxx + pad, self.maxy + pad)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


class Projector:
    """Bidirectional WGS84 <-> local-UTM transformer for a fixed zone.

    Build one per analysis (the zone is chosen from the sightings' centroid) and
    reuse it; ``pyproj`` transformers are relatively expensive to construct.
    """

    def __init__(self, utm_crs: CRS):
        self.utm_crs = utm_crs
        self._fwd = Transformer.from_crs(WGS84, utm_crs, always_xy=True)
        self._inv = Transformer.from_crs(utm_crs, WGS84, always_xy=True)

    @classmethod
    def for_point(cls, lon: float, lat: float) -> "Projector":
        return cls(utm_crs_for(lon, lat))

    def to_utm(self, lon, lat):
        """(lon, lat) degrees -> (easting, northing) metres. Scalars or arrays."""
        return self._fwd.transform(lon, lat)

    def to_lonlat(self, x, y):
        """(easting, northing) metres -> (lon, lat) degrees. Scalars or arrays."""
        return self._inv.transform(x, y)


def aoi_for_sightings(
    sightings, max_range_m: float, range_buffer_frac: float = 1.1
) -> tuple[Projector, BBox]:
    """Compute the area of interest in local UTM for a set of sightings.

    The AOI is the bounding box of all sighting positions, expanded by the
    maximum control range (plus a margin), because the controller can be
    anywhere within range of *any* sighting. The local UTM zone is chosen from
    the centroid of the sightings.

    Returns the ``Projector`` and the AOI ``BBox`` (in UTM metres).
    """
    if not sightings:
        raise ValueError("need at least one sighting")

    mean_lon = sum(s.lon for s in sightings) / len(sightings)
    mean_lat = sum(s.lat for s in sightings) / len(sightings)
    proj = Projector.for_point(mean_lon, mean_lat)

    xs, ys = [], []
    for s in sightings:
        x, y = proj.to_utm(s.lon, s.lat)
        xs.append(x)
        ys.append(y)

    pad = max_range_m * range_buffer_frac
    return proj, BBox(min(xs), min(ys), max(xs), max(ys)).buffered(pad)
