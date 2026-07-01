from __future__ import annotations

import numpy as np
from rasterio.io import MemoryFile

from launchpoint.config import Config, MonteCarloConfig
from launchpoint.core.geo import BBox, Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.reverse.pipeline import find_launch_area
from launchpoint.reverse.search import LaunchSearchConfig
from launchpoint.synthetic import make_launch_scenario
from launchpoint.ui.server import (
    RunJob,
    RunRecord,
    _config_from_payload,
    _geotiff_bytes,
    _job_snapshot,
    _launch_payload,
    _mark_job_stage,
    _parse_csv_sightings,
    _parse_sightings_payload,
    _parse_target_zone_payload,
    _search_config_from_payload,
    _serialize_grid,
)


def test_parse_sightings_payload_accepts_existing_schema():
    sightings = _parse_sightings_payload(
        {
            "sightings": [
                {
                    "label": "north",
                    "lat": 47.39,
                    "lon": 8.55,
                    "altitude": 110,
                    "position_sigma_m": 180,
                    "altitude_sigma_m": 25,
                }
            ]
        }
    )

    assert len(sightings) == 1
    assert sightings[0].label == "north"
    assert sightings[0].position_sigma_m == 180
    assert sightings[0].altitude_sigma_m == 25


def test_parse_csv_sightings_accepts_common_column_aliases():
    sightings = _parse_csv_sightings(
        "name,latitude,longitude,altitude_m,position_uncertainty_m,altitude_uncertainty_m\n"
        "alpha,47.39,8.55,120,200,35\n"
    )

    assert len(sightings) == 1
    assert sightings[0].label == "alpha"
    assert sightings[0].lat == 47.39
    assert sightings[0].lon == 8.55
    assert sightings[0].altitude == 120


def test_serialize_grid_downsamples_and_uses_json_safe_nulls():
    projector = Projector.for_point(8.55, 47.39)
    x, y = projector.to_utm(8.55, 47.39)
    grid = RasterGrid.empty(BBox(x, y, x + 90, y + 90), 30, projector.utm_crs, fill=0.0)
    grid.data = np.array(
        [
            [0.0, 0.5, 1.0],
            [np.nan, 0.25, 0.75],
            [0.1, 0.2, 0.3],
        ],
        dtype=np.float32,
    )

    payload = _serialize_grid(grid, projector, max_dim=2)

    assert payload["rows"] == 2
    assert payload["cols"] == 2
    assert payload["stride"] == 2
    assert payload["values"][2] == 0.1
    assert all(value is None or isinstance(value, float) for value in payload["values"])


def test_geotiff_bytes_can_round_trip_probability_grid():
    projector = Projector.for_point(8.55, 47.39)
    x, y = projector.to_utm(8.55, 47.39)
    grid = RasterGrid.empty(BBox(x, y, x + 60, y + 60), 30, projector.utm_crs, fill=0.0)
    grid.data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    with MemoryFile(_geotiff_bytes(grid)) as memfile:
        with memfile.open() as dataset:
            data = dataset.read(1)
            crs = dataset.crs

    assert data.shape == (2, 2)
    assert crs == grid.crs
    np.testing.assert_allclose(data, grid.data)


def test_ui_config_defaults_prefer_gpu():
    config, _ = _config_from_payload({"settings": {}})

    assert config.prefer_gpu is True


def test_job_progress_snapshot_tracks_stage_details():
    job = RunJob(run_id="run-1")

    _mark_job_stage(job, "Run fusion", "running", "Computing viewshed 1 of 2", {
        "current": 1,
        "total": 2,
    })
    snapshot = _job_snapshot(job)

    assert snapshot["status"] == "running"
    assert snapshot["message"] == "Computing viewshed 1 of 2"
    assert snapshot["stages"][0]["name"] == "Run fusion"
    assert snapshot["stages"][0]["details"]["total"] == 2
    assert 0.0 < snapshot["percent"] < 1.0


def test_parse_target_zone_payload_accepts_schema():
    zone = _parse_target_zone_payload(
        {
            "zone": {
                "label": "field survey",
                "center_lat": 47.372,
                "center_lon": 8.542,
                "radius_m": 400.0,
                "flight_altitude_m": 100.0,
            }
        }
    )
    assert zone.label == "field survey"
    assert zone.radius_m == 400.0
    assert zone.flight_altitude_m == 100.0


def test_search_config_from_payload_uses_defaults_and_overrides():
    default_cfg = _search_config_from_payload({"settings": {}})
    assert default_cfg.coverage_threshold == LaunchSearchConfig().coverage_threshold

    overridden = _search_config_from_payload(
        {"settings": {"coverage_threshold": 0.7, "candidate_coarse_stride_m": 25.0}}
    )
    assert overridden.coverage_threshold == 0.7
    assert overridden.candidate_coarse_stride_m == 25.0


def test_launch_payload_serializes_result():
    sc = make_launch_scenario(seed=11)
    config = Config(monte_carlo=MonteCarloConfig(samples_per_sighting=1, seed=1))
    search = LaunchSearchConfig(
        candidate_coarse_stride_m=150.0, candidate_fine_stride_m=50.0, surface_resolution_m=30.0,
    )
    result = find_launch_area(
        sc.zone, config=config, search=search,
        occluder=sc.surface, projector=sc.projector,
    )

    from launchpoint.data.surface import SurfaceStack

    record = RunRecord(
        run_type="reverse",
        launch_result=result,
        zone=sc.zone,
        stack=SurfaceStack(occluder=sc.surface, ground=sc.surface, launch_weight=sc.surface.like(fill=1.0)),
        metadata={"canopySource": "eth"},
    )
    payload = _launch_payload("run-1", record)

    assert payload["runType"] == "reverse"
    assert payload["zone"]["radiusM"] == sc.zone.radius_m
    assert "lon" in payload["bestLaunchPoint"] and "lat" in payload["bestLaunchPoint"]
    assert payload["coverage"]["rows"] > 0
