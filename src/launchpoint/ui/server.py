"""Small stdlib HTTP server for the Phase 6 LaunchPoint workspace."""

from __future__ import annotations

import csv
import dataclasses
import json
import mimetypes
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from rasterio.io import MemoryFile

from launchpoint.config import Config, DEFAULT_ANALYSIS_RESOLUTION_M, MonteCarloConfig
from launchpoint.core.geo import BBox, Projector, aoi_for_sightings, aoi_for_target_zone
from launchpoint.core.grid import RasterGrid
from launchpoint.core.sighting import Sighting
from launchpoint.core.target_zone import TargetZone
from launchpoint.data.surface import SurfaceStack, build_surface_stack
from launchpoint.pipeline import OriginEstimate, find_origin
from launchpoint.reverse.pipeline import find_launch_area
from launchpoint.reverse.search import LaunchCoverageResult, LaunchSearchConfig


MAX_SERIALIZED_GRID_DIM = 420
MAX_SERIALIZED_LAYER_DIM = 220
MAX_DIAGNOSTIC_LAYER_DIM = 1024
MAX_ANALYSIS_CELLS = 25_000_000
ALLOWED_ANALYSIS_RESOLUTIONS_M = {1.0, 2.0, 5.0, 10.0, 30.0}


@dataclass
class RunRecord:
    stack: SurfaceStack
    metadata: dict
    run_type: str = "forward"
    estimate: OriginEstimate | None = None
    sightings: list[Sighting] | None = None
    launch_result: LaunchCoverageResult | None = None
    zone: TargetZone | None = None


@dataclass
class RunJob:
    run_id: str
    status: str = "queued"
    message: str = "Queued"
    stages: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    completed_at: float | None = None
    record: RunRecord | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class RunStore:
    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._jobs: dict[str, RunJob] = {}
        self._lock = threading.Lock()

    def put(self, record: RunRecord, run_id: str | None = None) -> str:
        run_id = run_id or uuid.uuid4().hex
        with self._lock:
            self._records[run_id] = record
        return run_id

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            if record is not None:
                return record
            job = self._jobs.get(run_id)
            return job.record if job is not None else None

    def create_job(self) -> RunJob:
        job = RunJob(run_id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.run_id] = job
        return job

    def get_job(self, run_id: str) -> RunJob | None:
        with self._lock:
            return self._jobs.get(run_id)


RUN_STORE = RunStore()


def _load_static_file(path: str) -> tuple[bytes, str]:
    static_root = files("launchpoint.ui").joinpath("static")
    target = static_root.joinpath("index.html" if path in {"", "/"} else path.lstrip("/"))
    target_path = Path(os.fspath(target))
    root_path = Path(os.fspath(static_root)).resolve()
    resolved = target_path.resolve()
    if root_path not in resolved.parents and resolved != root_path:
        raise FileNotFoundError(path)
    data = resolved.read_bytes()
    content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
    if resolved.suffix == ".js":
        content_type = "text/javascript"
    return data, content_type


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _error_response(handler: BaseHTTPRequestHandler, message: str, status: int = 400) -> None:
    _json_response(handler, {"error": message}, status=status)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _parse_sightings_payload(payload: dict | list) -> list[Sighting]:
    items = payload.get("sightings", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("sightings must be a list")

    sightings: list[Sighting] = []
    for idx, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"sighting {idx} must be an object")
        label = raw.get("label") or f"S{idx}"
        sightings.append(
            Sighting(
                lat=float(raw["lat"]),
                lon=float(raw["lon"]),
                altitude=float(raw["altitude"]),
                position_sigma_m=float(raw.get("position_sigma_m", 250.0)),
                altitude_sigma_m=float(raw.get("altitude_sigma_m", 30.0)),
                label=str(label),
            )
        )
    if not sightings:
        raise ValueError("add or import at least one sighting")
    return sightings


def _parse_target_zone_payload(payload: dict) -> TargetZone:
    raw = payload.get("zone") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise ValueError("zone must be an object")
    label = raw.get("label")
    return TargetZone(
        center_lat=float(raw["center_lat"]),
        center_lon=float(raw["center_lon"]),
        radius_m=float(raw["radius_m"]),
        flight_altitude_m=float(raw["flight_altitude_m"]),
        label=str(label) if label else None,
    )


def _parse_csv_sightings(text: str) -> list[Sighting]:
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV must include a header row")
    normalized = []
    for row in reader:
        normalized.append(
            {
                "label": row.get("label") or row.get("name"),
                "lat": row.get("lat") or row.get("latitude"),
                "lon": row.get("lon") or row.get("lng") or row.get("longitude"),
                "altitude": row.get("altitude") or row.get("altitude_m"),
                "position_sigma_m": row.get("position_sigma_m")
                or row.get("position_uncertainty_m"),
                "altitude_sigma_m": row.get("altitude_sigma_m")
                or row.get("altitude_uncertainty_m"),
            }
        )
    return _parse_sightings_payload(normalized)


def _config_from_payload(payload: dict) -> tuple[Config, dict]:
    settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    max_range_m = float(settings.get("max_range_m", settings.get("max_range", 12000.0)))
    samples = int(settings.get("samples_per_sighting", settings.get("samples", 64)))
    antenna_height_m = float(settings.get("antenna_height_m", 1.5))
    prefer_gpu = bool(settings.get("prefer_gpu", True))
    cache_dir = str(settings.get("cache_dir", "data_cache"))
    combine = str(settings.get("combine", "min"))
    if combine not in ("min", "geometric_mean", "arithmetic_mean"):
        combine = "min"
    canopy_source = str(settings.get("canopy_source", "eth"))
    if canopy_source not in ("eth", "meta"):
        canopy_source = "eth"
    use_canopy = bool(settings.get("use_canopy", True))
    use_buildings = bool(settings.get("use_buildings", True))
    use_cache = bool(settings.get("use_cache", True))
    resolution_raw = settings.get(
        "analysis_resolution_m",
        settings.get("coarse_resolution_m", DEFAULT_ANALYSIS_RESOLUTION_M),
    )
    analysis_resolution_m = _validate_analysis_resolution(
        float(resolution_raw), use_canopy=use_canopy, canopy_source=canopy_source
    )

    config = Config(
        max_range_m=max_range_m,
        antenna_height_m=antenna_height_m,
        coarse_resolution_m=analysis_resolution_m,
        prefer_gpu=prefer_gpu,
        cache_dir=cache_dir,
        combine=combine,
        canopy_source=canopy_source,
        monte_carlo=MonteCarloConfig(samples_per_sighting=samples),
    )
    layer_flags = {
        "use_canopy": use_canopy,
        "use_buildings": use_buildings,
        "use_cache": use_cache,
    }
    return config, layer_flags


def _search_config_from_payload(payload: dict) -> LaunchSearchConfig:
    settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    defaults = LaunchSearchConfig()
    return LaunchSearchConfig(
        coverage_threshold=float(settings.get("coverage_threshold", defaults.coverage_threshold)),
        candidate_coarse_stride_m=float(
            settings.get("candidate_coarse_stride_m", defaults.candidate_coarse_stride_m)
        ),
        candidate_fine_stride_m=float(
            settings.get("candidate_fine_stride_m", defaults.candidate_fine_stride_m)
        ),
        surface_resolution_m=float(
            settings.get("surface_resolution_m", defaults.surface_resolution_m)
        ),
    )


def _validate_analysis_resolution(
    resolution_m: float,
    *,
    use_canopy: bool,
    canopy_source: str,
) -> float:
    if not np.isfinite(resolution_m):
        raise ValueError("analysis resolution must be finite")
    selected = min(ALLOWED_ANALYSIS_RESOLUTIONS_M, key=lambda value: abs(value - resolution_m))
    if abs(selected - resolution_m) > 1e-6:
        allowed = ", ".join(f"{value:g} m" for value in sorted(ALLOWED_ANALYSIS_RESOLUTIONS_M))
        raise ValueError(f"analysis resolution must be one of: {allowed}")
    if selected < 10.0 and (not use_canopy or canopy_source != "meta"):
        raise ValueError(
            "analysis resolutions below 10 m require canopy enabled with the Meta 1 m source"
        )
    return float(selected)


def _aoi_details(
    bbox,
    resolution_m: float,
    description: str = "Bounding box of sightings padded by max range x 1.1",
) -> dict:
    rows = max(1, int(np.ceil(bbox.height / resolution_m)))
    cols = max(1, int(np.ceil(bbox.width / resolution_m)))
    cells = rows * cols
    return {
        "widthM": round(float(bbox.width), 1),
        "heightM": round(float(bbox.height), 1),
        "areaKm2": round(float(bbox.width * bbox.height / 1e6), 3),
        "resolutionM": float(resolution_m),
        "estimatedRows": rows,
        "estimatedCols": cols,
        "estimatedCells": cells,
        "rangeBufferMultiplier": 1.1,
        "description": description,
    }


def _enforce_analysis_size(aoi_details: dict, *, override: bool = False) -> None:
    cells = int(aoi_details["estimatedCells"])
    if cells <= MAX_ANALYSIS_CELLS:
        return
    if override:
        # The cap is a soft guard against accidental OOM, not a hard ceiling.
        # The caller has explicitly opted in to run an oversized grid on their
        # own hardware, so let it through (it may still fail with MemoryError).
        print(
            f"WARNING: analysis grid size cap overridden: {cells:,} cells at "
            f"{float(aoi_details['resolutionM']):g} m (cap {MAX_ANALYSIS_CELLS:,})"
        )
        return
    resolution = float(aoi_details["resolutionM"])
    raise ValueError(
        "Analysis grid is too large: "
        f"{cells:,} cells at {resolution:g} m resolution "
        f"({aoi_details['estimatedRows']:,} x {aoi_details['estimatedCols']:,}). "
        "Use a coarser model resolution, lower the max range, or enable "
        "'Override grid-size limit' in Advanced to run it anyway."
    )


def _grid_lonlat_bounds(grid: RasterGrid, projector: Projector) -> dict:
    b = grid.bounds
    xs = [b.minx, b.minx, b.maxx, b.maxx]
    ys = [b.miny, b.maxy, b.miny, b.maxy]
    lon, lat = projector.to_lonlat(xs, ys)
    return {
        "west": float(np.min(lon)),
        "south": float(np.min(lat)),
        "east": float(np.max(lon)),
        "north": float(np.max(lat)),
    }


def _downsample(data: np.ndarray, max_dim: int) -> tuple[np.ndarray, int]:
    rows, cols = data.shape
    stride = max(1, int(np.ceil(max(rows, cols) / max_dim)))
    return data[::stride, ::stride], stride


def _serialize_values(data: np.ndarray, precision: int = 5) -> list[float | None]:
    finite = np.isfinite(data)
    clean = np.where(finite, data, np.nan).astype(float).ravel()
    out: list[float | None] = []
    for value in clean:
        out.append(None if np.isnan(value) else round(float(value), precision))
    return out


def _serialize_grid(
    grid: RasterGrid,
    projector: Projector,
    *,
    max_dim: int = MAX_SERIALIZED_GRID_DIM,
    precision: int = 5,
) -> dict:
    data, stride = _downsample(grid.data, max_dim)
    finite = data[np.isfinite(data)]
    return {
        "rows": int(data.shape[0]),
        "cols": int(data.shape[1]),
        "stride": int(stride),
        "fullRows": int(grid.rows),
        "fullCols": int(grid.cols),
        "resX": float(grid.res_x * stride),
        "resY": float(grid.res_y * stride),
        "bounds": _grid_lonlat_bounds(grid, projector),
        "min": float(np.min(finite)) if finite.size else 0.0,
        "max": float(np.max(finite)) if finite.size else 0.0,
        "values": _serialize_values(data, precision=precision),
    }


def _credible_area_km2(mask: RasterGrid) -> float:
    cells = int((mask.data == 1.0).sum())
    return cells * mask.res_x * mask.res_y / 1e6


def _mark_job_stage(
    job: RunJob,
    name: str,
    status: str,
    message: str,
    details: dict | None = None,
) -> None:
    now = time.perf_counter()
    with job.lock:
        if job.status not in {"complete", "error"}:
            job.status = "running"
        job.message = message

        stage = next((item for item in job.stages if item["name"] == name), None)
        if stage is None:
            stage = {
                "name": name,
                "status": status,
                "message": message,
                "startedAt": now,
                "elapsedMs": 0,
                "details": {},
            }
            job.stages.append(stage)
        stage["status"] = status
        stage["message"] = message
        stage["elapsedMs"] = int((now - stage["startedAt"]) * 1000)
        if details is not None:
            stage["details"] = details


def _stage_percent(stages: list[dict], status: str) -> float:
    if status == "complete":
        return 1.0
    if status == "error":
        return 0.0

    stages_by_name = {stage["name"]: stage for stage in stages}
    if stages_by_name.get("Export ready", {}).get("status") == "complete":
        return 1.0
    if stages_by_name.get("Serialize result", {}).get("status") == "complete":
        return 0.97
    fusion = stages_by_name.get("Run fusion")
    if fusion:
        if fusion.get("status") == "complete":
            return 0.9
        details = fusion.get("details") or {}
        total = max(int(details.get("total") or 0), 1)
        current = min(max(int(details.get("current") or 1), 1), total)
        return 0.35 + 0.55 * ((current - 1) / total)
    refine = stages_by_name.get("Refine hot patches")
    if refine:
        return 0.9 if refine.get("status") == "complete" else 0.75
    if stages_by_name.get("Search candidates", {}).get("status") == "complete":
        return 0.75
    if stages_by_name.get("Search candidates"):
        return 0.5
    if stages_by_name.get("Fetch surfaces", {}).get("status") == "complete":
        return 0.35
    fetch = stages_by_name.get("Fetch surfaces")
    if fetch:
        # The data layer reports its own 0..1 download fraction; map it into the
        # slice of the bar reserved for fetching (0.12 -> 0.35).
        frac = float((fetch.get("details") or {}).get("fetchProgress", 0.0))
        return 0.12 + 0.23 * min(max(frac, 0.0), 1.0)
    if stages_by_name.get("Prepare AOI", {}).get("status") == "complete":
        return 0.08
    if stages_by_name.get("Prepare AOI"):
        return 0.03
    return 0.0


def _job_snapshot(job: RunJob) -> dict:
    now = time.perf_counter()
    with job.lock:
        stages = []
        for stage in job.stages:
            item = dict(stage)
            if item["status"] == "running":
                item["elapsedMs"] = int((now - item["startedAt"]) * 1000)
            item.pop("startedAt", None)
            stages.append(item)
        elapsed_ms = int(((job.completed_at or now) - job.started_at) * 1000)
        return {
            "runId": job.run_id,
            "status": job.status,
            "message": job.message,
            "stages": stages,
            "elapsedMs": elapsed_ms,
            "percent": _stage_percent(stages, job.status),
            "error": job.error,
            "ready": job.record is not None and job.status == "complete",
        }


def _surface_layers(stack: SurfaceStack, projector: Projector) -> dict:
    layers: dict[str, dict | None] = {
        "dsm": _serialize_grid(
            stack.occluder, projector, max_dim=MAX_SERIALIZED_LAYER_DIM, precision=2
        ),
        "bareEarth": _serialize_grid(
            stack.ground, projector, max_dim=MAX_SERIALIZED_LAYER_DIM, precision=2
        ),
        "launchWeight": _serialize_grid(
            stack.launch_weight, projector, max_dim=MAX_SERIALIZED_LAYER_DIM, precision=4
        ),
        "canopyHeight": None,
        "buildingHeight": None,
    }
    if stack.canopy_height is not None:
        layers["canopyHeight"] = _serialize_grid(
            stack.canopy_height, projector, max_dim=MAX_SERIALIZED_LAYER_DIM, precision=2
        )
    if stack.building_height is not None:
        layers["buildingHeight"] = _serialize_grid(
            stack.building_height, projector, max_dim=MAX_SERIALIZED_LAYER_DIM, precision=2
        )
    return layers


def _gpu_mode(prefer_gpu: bool) -> str:
    """Report the viewshed backend actually in effect, not just what was asked.

    Avoids the old misleading "gpu-preferred" label when the run silently fell
    back to CPU (the reported symptom): resolves the real GPU status so the UI
    shows "gpu" only when the CUDA kernel will run.
    """
    if not prefer_gpu:
        return "cpu"
    from launchpoint.viewshed.gpu import gpu_status

    available, reason = gpu_status()
    return "gpu" if available else f"cpu (GPU requested but unavailable: {reason})"


def _native_canopy_resolution_m(source: str) -> float:
    if source == "meta":
        from launchpoint.data.canopy import NATIVE_RES_M

        return float(NATIVE_RES_M)
    from launchpoint.data.eth_canopy import NATIVE_RES_M

    return float(NATIVE_RES_M)


def _diagnostic_bbox_from_query(params: dict[str, list[str]], projector: Projector) -> BBox:
    west = float(params.get("west", ["nan"])[0])
    south = float(params.get("south", ["nan"])[0])
    east = float(params.get("east", ["nan"])[0])
    north = float(params.get("north", ["nan"])[0])
    if not all(np.isfinite(v) for v in (west, south, east, north)):
        raise ValueError("diagnostic bounds must include west,south,east,north")
    if west >= east or south >= north:
        raise ValueError("diagnostic bounds are empty")

    xs, ys = projector.to_utm([west, west, east, east], [south, north, south, north])
    return BBox(float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))


def _intersect_bbox(a: BBox, b: BBox) -> BBox | None:
    minx = max(a.minx, b.minx)
    miny = max(a.miny, b.miny)
    maxx = min(a.maxx, b.maxx)
    maxy = min(a.maxy, b.maxy)
    if minx >= maxx or miny >= maxy:
        return None
    return BBox(minx, miny, maxx, maxy)


def _diagnostic_resolution(bbox: BBox, native_resolution_m: float) -> float:
    max_span = max(bbox.width, bbox.height)
    capped_resolution = max_span / MAX_DIAGNOSTIC_LAYER_DIM
    return max(native_resolution_m, capped_resolution)


def _record_projector(record: RunRecord) -> Projector:
    """The projector for either run type — forward's OriginEstimate or
    reverse's LaunchCoverageResult both carry one."""
    if record.run_type == "reverse":
        return record.launch_result.projector
    return record.estimate.projector


def _fetch_canopy_diagnostic_layer(record: RunRecord, bounds: BBox) -> RasterGrid:
    source = str(record.metadata.get("canopySource", "eth"))
    native_res = _native_canopy_resolution_m(source)
    resolution = _diagnostic_resolution(bounds, native_res)
    projector = _record_projector(record)
    target = RasterGrid.empty(bounds, resolution, projector.utm_crs, fill=np.nan)

    if source == "meta":
        from launchpoint.data.canopy import fetch_canopy_height

        grid = fetch_canopy_height(
            target, projector, cache_dir=record.metadata.get("cacheDir"), reporter=None
        )
    else:
        from launchpoint.data.eth_canopy import fetch_canopy_height_eth

        grid = fetch_canopy_height_eth(
            target, projector, cache_dir=record.metadata.get("cacheDir"), reporter=None
        )

    # Apply the same ground-floor cutoff the run used so the high-res diagnostic
    # matches the canopy the model actually consumed (see CanopyConfig.ground_floor_m).
    from launchpoint.config import CanopyConfig
    from launchpoint.data.bare_earth import apply_ground_floor

    floor = record.metadata.get("canopyGroundFloorM")
    if floor is None:
        floor = CanopyConfig().effective_ground_floor_m(source)
    return apply_ground_floor(grid, float(floor))


def _diagnostic_layer_payload(run_id: str, record: RunRecord, params: dict[str, list[str]]) -> dict:
    layer = params.get("layer", [""])[0]
    if layer != "canopyHeight":
        raise ValueError("high-resolution diagnostics are only available for canopyHeight")
    if not record.metadata.get("layers", {}).get("canopyHeight"):
        raise ValueError("canopy layer is not available for this run")

    projector = _record_projector(record)
    requested = _diagnostic_bbox_from_query(params, projector)
    bounds = _intersect_bbox(requested, record.stack.occluder.bounds)
    if bounds is None:
        raise ValueError("diagnostic bounds are outside the analysed area")

    grid = _fetch_canopy_diagnostic_layer(record, bounds)
    payload = _serialize_grid(
        grid, projector, max_dim=MAX_DIAGNOSTIC_LAYER_DIM, precision=2
    )
    payload["source"] = record.metadata.get("canopySource", "eth")
    payload["nativeResolutionM"] = _native_canopy_resolution_m(payload["source"])
    payload["runId"] = run_id
    payload["layer"] = layer
    return payload


def _analysis_payload(run_id: str, record: RunRecord) -> dict:
    est = record.estimate
    mask = est.credible_mask(0.5)
    lon, lat = est.argmax_lonlat()
    per_sighting = [
        {
            "label": s.label or f"S{idx + 1}",
            "raster": _serialize_grid(
                grid, est.projector, max_dim=MAX_SERIALIZED_LAYER_DIM, precision=5
            ),
        }
        for idx, (s, grid) in enumerate(zip(record.sightings, est.fusion.per_sighting))
    ]
    return {
        "runId": run_id,
        "mostLikely": {"lon": lon, "lat": lat},
        "credibleRegion": {
            "fraction": 0.5,
            "areaKm2": _credible_area_km2(mask),
            "mask": _serialize_grid(mask, est.projector, max_dim=MAX_SERIALIZED_GRID_DIM),
        },
        "probability": _serialize_grid(est.probability, est.projector),
        "perSighting": per_sighting,
        "surfaceLayers": _surface_layers(record.stack, est.projector),
        "metadata": record.metadata,
    }


def _run_analysis(payload: dict, job: RunJob | None = None) -> tuple[str, RunRecord]:
    job = job or RunJob(run_id=uuid.uuid4().hex)
    sightings = _parse_sightings_payload(payload)
    config, layer_flags = _config_from_payload(payload)
    started = time.perf_counter()
    job.started_at = started

    _mark_job_stage(job, "Prepare AOI", "running", "Preparing analysis area")
    projector, bbox = aoi_for_sightings(sightings, config.max_range_m)
    aoi_details = _aoi_details(bbox, config.coarse_resolution_m)
    settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    override_size = bool(settings.get("override_cell_limit", False))
    _enforce_analysis_size(aoi_details, override=override_size)
    _mark_job_stage(
        job,
        "Prepare AOI",
        "complete",
        (
            f"AOI {bbox.width / 1000:.1f} x {bbox.height / 1000:.1f} km "
            f"at {config.coarse_resolution_m:g} m"
        ),
        aoi_details,
    )

    _mark_job_stage(
        job,
        "Fetch surfaces",
        "running",
        "Fetching DSM, canopy, and building surfaces",
        aoi_details,
    )
    stack = build_surface_stack(
        sightings, config, projector=projector,
        progress_callback=lambda name, status, message, details: _mark_job_stage(
            job, name, status, message, details
        ),
        **layer_flags,
    )
    surface_details = {
        **aoi_details,
        "rows": stack.occluder.rows,
        "cols": stack.occluder.cols,
        "cells": int(stack.occluder.data.size),
        "layers": {
            "dsm": True,
            "bareEarth": True,
            "canopyHeight": stack.canopy_height is not None,
            "buildingHeight": stack.building_height is not None,
            "launchWeight": True,
        },
    }
    _mark_job_stage(
        job,
        "Fetch surfaces",
        "complete",
        f"Surfaces ready: {stack.occluder.rows} x {stack.occluder.cols} cells",
        surface_details,
    )

    _mark_job_stage(
        job,
        "Run fusion",
        "running",
        f"Starting viewshed fusion for {len(sightings)} sighting(s)",
        {"current": 0, "total": len(sightings)},
    )
    estimate = find_origin(
        sightings,
        config=config,
        occluder=stack.occluder,
        ground=stack.ground,
        projector=projector,
        launch_weight=stack.launch_weight,
        progress_callback=lambda name, status, message, details: _mark_job_stage(
            job, name, status, message, details
        ),
    )

    _mark_job_stage(
        job,
        "Serialize result",
        "running",
        "Preparing result metadata and exports",
    )

    metadata = {
        "samples": config.monte_carlo.samples_per_sighting,
        "combine": config.combine,
        "canopySource": config.canopy_source,
        "canopyGroundFloorM": config.canopy.effective_ground_floor_m(config.canopy_source),
        "analysisResolutionM": config.coarse_resolution_m,
        "maxRangeM": config.max_range_m,
        "gpuMode": _gpu_mode(config.prefer_gpu),
        "cacheDir": config.cache_dir,
        "aoi": surface_details,
        "layers": {
            "dsm": True,
            "bareEarth": True,
            "canopyHeight": stack.canopy_height is not None,
            "buildingHeight": stack.building_height is not None,
            "launchWeight": True,
        },
        "stages": [],
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    }
    _mark_job_stage(
        job,
        "Serialize result",
        "complete",
        "Result metadata ready",
    )
    _mark_job_stage(
        job,
        "Export ready",
        "complete",
        "GeoTIFF and PNG preview are available",
    )

    record = RunRecord(estimate=estimate, sightings=sightings, stack=stack, metadata=metadata)
    run_id = RUN_STORE.put(record, run_id=job.run_id)
    with job.lock:
        job.record = record
        job.status = "complete"
        job.message = "Analysis complete"
        job.completed_at = time.perf_counter()
    snapshot = _job_snapshot(job)
    metadata["stages"] = snapshot["stages"]
    metadata["elapsedMs"] = snapshot["elapsedMs"]
    return run_id, record


def _run_analysis_job(job: RunJob, payload: dict) -> None:
    try:
        _run_analysis(payload, job=job)
    except Exception as exc:  # noqa: BLE001 - worker boundary
        with job.lock:
            job.status = "error"
            job.message = str(exc)
            job.error = str(exc)
            job.completed_at = time.perf_counter()


def _launch_payload(run_id: str, record: RunRecord) -> dict:
    result = record.launch_result
    zone = record.zone
    lon, lat = result.best_lonlat()
    return {
        "runId": run_id,
        "runType": "reverse",
        "bestLaunchPoint": {
            "lon": lon,
            "lat": lat,
            "score": result.best_score,
            "minVisibility": result.best_min_visibility,
            "fracCovered": result.best_frac_covered,
        },
        "zone": {
            "centerLat": zone.center_lat,
            "centerLon": zone.center_lon,
            "radiusM": zone.radius_m,
            "flightAltitudeM": zone.flight_altitude_m,
        },
        "coverage": _serialize_grid(result.coverage, result.projector),
        "minVisibility": _serialize_grid(
            result.min_visibility, result.projector, max_dim=MAX_SERIALIZED_LAYER_DIM
        ),
        "fracCovered": _serialize_grid(
            result.frac_covered, result.projector, max_dim=MAX_SERIALIZED_LAYER_DIM
        ),
        "surfaceLayers": _surface_layers(record.stack, result.projector),
        "metadata": record.metadata,
    }


def _run_reverse_analysis(payload: dict, job: RunJob | None = None) -> tuple[str, RunRecord]:
    job = job or RunJob(run_id=uuid.uuid4().hex)
    zone = _parse_target_zone_payload(payload)
    config, layer_flags = _config_from_payload(payload)
    search_cfg = _search_config_from_payload(payload)
    started = time.perf_counter()
    job.started_at = started

    _mark_job_stage(job, "Prepare AOI", "running", "Preparing launch-search area")
    projector, bbox = aoi_for_target_zone(zone, config.max_range_m)
    aoi_details = _aoi_details(
        bbox, search_cfg.surface_resolution_m,
        description="Flight-zone disk padded by max range x 1.1",
    )
    settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    override_size = bool(settings.get("override_cell_limit", False))
    _enforce_analysis_size(aoi_details, override=override_size)
    _mark_job_stage(
        job,
        "Prepare AOI",
        "complete",
        (
            f"AOI {bbox.width / 1000:.1f} x {bbox.height / 1000:.1f} km "
            f"at {search_cfg.surface_resolution_m:g} m"
        ),
        aoi_details,
    )

    _mark_job_stage(
        job,
        "Fetch surfaces",
        "running",
        "Fetching DSM, canopy, and building surfaces",
        aoi_details,
    )
    surface_config = dataclasses.replace(config, coarse_resolution_m=search_cfg.surface_resolution_m)
    stack = build_surface_stack(
        sightings=[],
        config=surface_config,
        projector=projector,
        bbox=bbox,
        gate_canopy_with_footprint=False,
        progress_callback=lambda name, status, message, details: _mark_job_stage(
            job, name, status, message, details
        ),
        **layer_flags,
    )
    surface_details = {
        **aoi_details,
        "rows": stack.occluder.rows,
        "cols": stack.occluder.cols,
        "cells": int(stack.occluder.data.size),
        "layers": {
            "dsm": True,
            "bareEarth": True,
            "canopyHeight": stack.canopy_height is not None,
            "buildingHeight": stack.building_height is not None,
            "launchWeight": True,
        },
    }
    _mark_job_stage(
        job,
        "Fetch surfaces",
        "complete",
        f"Surfaces ready: {stack.occluder.rows} x {stack.occluder.cols} cells",
        surface_details,
    )

    _mark_job_stage(job, "Search candidates", "running", "Scanning coarse candidate grid")
    result = find_launch_area(
        zone,
        config=config,
        search=search_cfg,
        occluder=stack.occluder,
        ground=stack.ground,
        projector=projector,
        launch_weight=stack.launch_weight,
        progress_callback=lambda name, status, message, details: _mark_job_stage(
            job, name, status, message, details
        ),
    )

    _mark_job_stage(
        job,
        "Serialize result",
        "running",
        "Preparing result metadata and exports",
    )

    metadata = {
        "flightAltitudeM": zone.flight_altitude_m,
        "zoneRadiusM": zone.radius_m,
        "coverageThreshold": search_cfg.coverage_threshold,
        "candidatesEvaluated": result.n_candidates_evaluated,
        "candidatesPrefiltered": result.n_candidates_prefiltered,
        "canopySource": config.canopy_source,
        "canopyGroundFloorM": config.canopy.effective_ground_floor_m(config.canopy_source),
        "analysisResolutionM": search_cfg.surface_resolution_m,
        "maxRangeM": config.max_range_m,
        "gpuMode": _gpu_mode(config.prefer_gpu),
        "cacheDir": config.cache_dir,
        "aoi": surface_details,
        "layers": surface_details["layers"],
        "stages": [],
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    }
    _mark_job_stage(
        job,
        "Serialize result",
        "complete",
        "Result metadata ready",
    )
    _mark_job_stage(
        job,
        "Export ready",
        "complete",
        "GeoTIFF and PNG preview are available",
    )

    record = RunRecord(
        run_type="reverse", launch_result=result, zone=zone, stack=stack, metadata=metadata,
    )
    run_id = RUN_STORE.put(record, run_id=job.run_id)
    with job.lock:
        job.record = record
        job.status = "complete"
        job.message = "Analysis complete"
        job.completed_at = time.perf_counter()
    snapshot = _job_snapshot(job)
    metadata["stages"] = snapshot["stages"]
    metadata["elapsedMs"] = snapshot["elapsedMs"]
    return run_id, record


def _run_reverse_analysis_job(job: RunJob, payload: dict) -> None:
    try:
        _run_reverse_analysis(payload, job=job)
    except Exception as exc:  # noqa: BLE001 - worker boundary
        with job.lock:
            job.status = "error"
            job.message = str(exc)
            job.error = str(exc)
            job.completed_at = time.perf_counter()


def _geotiff_bytes(grid: RasterGrid) -> bytes:
    rows, cols = grid.shape
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(grid.data.astype(np.float32), 1)
        return memfile.read()


class LaunchPointHandler(BaseHTTPRequestHandler):
    server_version = "LaunchPointUI/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        if parsed.path == "/api/defaults":
            _json_response(
                self,
                {
                    "settings": {
                        "maxRangeM": 12000.0,
                        "samplesPerSighting": 64,
                        "antennaHeightM": 1.5,
                        "analysisResolutionM": DEFAULT_ANALYSIS_RESOLUTION_M,
                        "preferGpu": True,
                    },
                    "reverseSettings": {
                        "flightAltitudeM": 100.0,
                        "radiusM": 300.0,
                        "coverageThreshold": LaunchSearchConfig().coverage_threshold,
                        "candidateCoarseStrideM": LaunchSearchConfig().candidate_coarse_stride_m,
                        "candidateFineStrideM": LaunchSearchConfig().candidate_fine_stride_m,
                        "surfaceResolutionM": LaunchSearchConfig().surface_resolution_m,
                    },
                    "tileSources": {
                        "map": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                        "terrain": (
                            "https://s3.amazonaws.com/elevation-tiles-prod/"
                            "terrarium/{z}/{x}/{y}.png"
                        ),
                    },
                },
            )
            return
        if parsed.path == "/api/runs/status":
            params = parse_qs(parsed.query)
            run_id = params.get("run_id", [""])[0]
            job = RUN_STORE.get_job(run_id)
            if job is None:
                _error_response(self, "unknown run_id", status=HTTPStatus.NOT_FOUND)
                return
            _json_response(self, _job_snapshot(job))
            return
        if parsed.path == "/api/runs/result":
            params = parse_qs(parsed.query)
            run_id = params.get("run_id", [""])[0]
            job = RUN_STORE.get_job(run_id)
            if job is None:
                _error_response(self, "unknown run_id", status=HTTPStatus.NOT_FOUND)
                return
            snapshot = _job_snapshot(job)
            if job.status == "error":
                _json_response(self, snapshot, status=HTTPStatus.BAD_REQUEST)
                return
            if job.record is None:
                _json_response(self, snapshot, status=HTTPStatus.ACCEPTED)
                return
            if job.record.run_type == "reverse":
                _json_response(self, _launch_payload(run_id, job.record))
            else:
                _json_response(self, _analysis_payload(run_id, job.record))
            return
        if parsed.path == "/api/runs/diagnostic-layer":
            params = parse_qs(parsed.query)
            run_id = params.get("run_id", [""])[0]
            record = RUN_STORE.get(run_id)
            if record is None:
                _error_response(self, "unknown run_id", status=HTTPStatus.NOT_FOUND)
                return
            try:
                _json_response(self, _diagnostic_layer_payload(run_id, record, params))
            except ValueError as exc:
                _error_response(self, str(exc), status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # noqa: BLE001 - optional diagnostic fetch boundary
                _error_response(self, str(exc), status=HTTPStatus.BAD_GATEWAY)
            return
        if parsed.path == "/api/export/geotiff":
            params = parse_qs(parsed.query)
            run_id = params.get("run_id", [""])[0]
            record = RUN_STORE.get(run_id)
            if record is None:
                _error_response(self, "unknown run_id", status=HTTPStatus.NOT_FOUND)
                return
            if record.run_type == "reverse":
                grid = record.launch_result.coverage
                export_name = "coverage"
            else:
                grid = record.estimate.probability
                export_name = "probability"
            data = _geotiff_bytes(grid)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/tiff")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="launchpoint_{export_name}_{run_id[:8]}.tif"',
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        static_path = "/index.html" if parsed.path == "/" else parsed.path
        try:
            data, content_type = _load_static_file(static_path)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/analyze":
                payload = _read_json(self)
                run_id, record = _run_analysis(payload)
                _json_response(self, _analysis_payload(run_id, record))
                return
            if parsed.path == "/api/runs":
                payload = _read_json(self)
                job = RUN_STORE.create_job()
                thread = threading.Thread(
                    target=_run_analysis_job,
                    args=(job, payload),
                    daemon=True,
                    name=f"launchpoint-run-{job.run_id[:8]}",
                )
                thread.start()
                _json_response(self, _job_snapshot(job), status=HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/runs/launch":
                payload = _read_json(self)
                job = RUN_STORE.create_job()
                thread = threading.Thread(
                    target=_run_reverse_analysis_job,
                    args=(job, payload),
                    daemon=True,
                    name=f"launchpoint-launch-{job.run_id[:8]}",
                )
                thread.start()
                _json_response(self, _job_snapshot(job), status=HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/parse-csv":
                length = int(self.headers.get("Content-Length", "0"))
                text = self.rfile.read(length).decode("utf-8")
                sightings = _parse_csv_sightings(text)
                _json_response(
                    self,
                    {
                        "sightings": [
                            {
                                "label": s.label,
                                "lat": s.lat,
                                "lon": s.lon,
                                "altitude": s.altitude,
                                "position_sigma_m": s.position_sigma_m,
                                "altitude_sigma_m": s.altitude_sigma_m,
                            }
                            for s in sightings
                        ]
                    },
                )
                return
            _error_response(self, "unknown endpoint", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - endpoint boundary
            _error_response(self, str(exc), status=HTTPStatus.BAD_REQUEST)


def run_ui_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    mimetypes.add_type("text/javascript", ".js")
    server = ThreadingHTTPServer((host, port), LaunchPointHandler)
    print(f"LaunchPoint UI listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
