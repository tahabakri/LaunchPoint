from __future__ import annotations

import numpy as np
import pytest

from launchpoint.config import Config, RangeModelConfig
from launchpoint.core.geo import BBox, Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.coverage import FlightArea, flight_area_mask, plan_launch_area
from launchpoint.ui.server import _parse_flight_area_payload


def _flat_grid(resolution: float = 30.0, extent: float = 3000.0) -> tuple[RasterGrid, Projector]:
    projector = Projector.for_point(8.55, 47.37)
    cx, cy = projector.to_utm(8.55, 47.37)
    half = extent / 2
    grid = RasterGrid.empty(
        BBox(cx - half, cy - half, cx + half, cy + half),
        resolution,
        projector.utm_crs,
        fill=0.0,
    )
    return grid, projector


def test_flight_area_mask_rasterizes_metric_circle():
    grid, projector = _flat_grid()
    area = FlightArea(center_lon=8.55, center_lat=47.37, radius_m=300, altitude_agl_m=100)

    mask = flight_area_mask(area, grid, projector)
    covered_area = float(mask.data.sum()) * grid.res_x * grid.res_y

    assert mask.data.max() == 1.0
    assert covered_area == pytest.approx(np.pi * 300**2, rel=0.18)


def test_coverage_scores_near_launch_cells_above_out_of_range_cells():
    grid, projector = _flat_grid(extent=4000)
    area = FlightArea(center_lon=8.55, center_lat=47.37, radius_m=150, altitude_agl_m=80)
    cfg = Config(
        max_range_m=700,
        prefer_gpu=False,
        range_model=RangeModelConfig(max_range_m=700, softness_m=80),
    )

    estimate = plan_launch_area(
        area,
        config=cfg,
        occluder=grid,
        ground=grid,
        projector=projector,
        sample_count=9,
    )

    center_score = estimate.coverage.data[grid.world_to_pixel(*projector.to_utm(8.55, 47.37))]
    far_score = estimate.coverage.data[0, 0]

    assert center_score > 0.7
    assert far_score == 0.0


def test_coverage_penalizes_cells_behind_synthetic_ridge():
    grid, projector = _flat_grid(extent=3000)
    cx, cy = projector.to_utm(8.55, 47.37)
    ridge = grid.world_to_pixel(cx - 350, cy)
    grid.data[:, ridge[1] : ridge[1] + 2] = 500.0
    area = FlightArea(center_lon=8.55, center_lat=47.37, radius_m=120, altitude_agl_m=60)
    cfg = Config(max_range_m=1200, prefer_gpu=False)

    estimate = plan_launch_area(
        area,
        config=cfg,
        occluder=grid,
        ground=grid.copy_with(np.zeros_like(grid.data)),
        projector=projector,
        sample_count=5,
    )

    east = grid.world_to_pixel(cx + 450, cy)
    west = grid.world_to_pixel(cx - 800, cy)

    assert estimate.coverage.data[east] > estimate.coverage.data[west]


def test_parse_flight_area_payload_validates_radius_and_altitude():
    area = _parse_flight_area_payload(
        {
            "flightArea": {
                "center": {"lat": 47.37, "lon": 8.55},
                "radiusM": 1000,
                "altitudeAglM": 120,
            }
        }
    )

    assert area.radius_m == 1000
    assert area.altitude_agl_m == 120

    try:
        _parse_flight_area_payload(
            {
                "flightArea": {
                    "center": {"lat": 47.37, "lon": 8.55},
                    "radiusM": -1,
                    "altitudeAglM": 120,
                }
            }
        )
    except ValueError as exc:
        assert "radius" in str(exc)
    else:
        raise AssertionError("negative radius should fail")
