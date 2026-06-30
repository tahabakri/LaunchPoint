import numpy as np

from launchpoint.core.geo import (
    Projector,
    aoi_for_sightings,
    utm_epsg_for,
)
from launchpoint.core.sighting import Sighting


def test_utm_zone_selection():
    # Zurich ~ 8.5E -> zone 32 N -> EPSG 32632
    assert utm_epsg_for(8.55, 47.37) == 32632
    # Southern hemisphere flips to 327xx
    assert utm_epsg_for(8.55, -47.37) == 32732
    # New York ~ -74 -> zone 18 N -> 32618
    assert utm_epsg_for(-74.0, 40.7) == 32618


def test_projector_roundtrip():
    proj = Projector.for_point(8.55, 47.37)
    x, y = proj.to_utm(8.55, 47.37)
    lon, lat = proj.to_lonlat(x, y)
    assert abs(lon - 8.55) < 1e-7
    assert abs(lat - 47.37) < 1e-7


def test_aoi_covers_sightings_plus_range():
    sightings = [
        Sighting(47.37, 8.55, 100),
        Sighting(47.38, 8.56, 100),
    ]
    proj, bbox = aoi_for_sightings(sightings, max_range_m=12000.0)
    # The AOI must be at least ~2 * range across (each point buffered by range).
    assert bbox.width > 24000.0
    assert bbox.height > 24000.0
    # Every sighting projects inside the box.
    for s in sightings:
        x, y = proj.to_utm(s.lon, s.lat)
        assert bbox.minx <= x <= bbox.maxx
        assert bbox.miny <= y <= bbox.maxy
