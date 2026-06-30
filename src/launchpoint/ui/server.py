"""Small stdlib HTTP server for the Phase 6 LaunchPoint workspace."""

from __future__ import annotations

import csv
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

from launchpoint.config import Config, MonteCarloConfig
from launchpoint.core.geo import Projector, aoi_for_sightings
from launchpoint.core.grid import RasterGrid
from launchpoint.core.sighting import Sighting
from launchpoint.data.surface import SurfaceStack, build_surface_stack
from launchpoint.pipeline import OriginEstimate, find_origin


MAX_SERIALIZED_GRID_DIM = 420
MAX_SERIALIZED_LAYER_DIM = 220


@dataclass
class RunRecord:
    estimate: OriginEstimate
    sightings: list[Sighting]
    stack: SurfaceStack
    metadata: dict


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

    config = Config(
        max_range_m=max_range_m,
        antenna_height_m=antenna_height_m,
        prefer_gpu=prefer_gpu,
        cache_dir=cache_dir,
        combine=combine,
        monte_carlo=MonteCarloConfig(samples_per_sighting=samples),
    )
    layer_flags = {
        "use_canopy": bool(settings.get("use_canopy", True)),
        "use_buildings": bool(settings.get("use_buildings", True)),
        "use_cache": bool(settings.get("use_cache", True)),
    }
    return config, layer_flags


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
    rows = max(1, int(np.ceil(bbox.height / config.coarse_resolution_m)))
    cols = max(1, int(np.ceil(bbox.width / config.coarse_resolution_m)))
    aoi_details = {
        "widthM": round(float(bbox.width), 1),
        "heightM": round(float(bbox.height), 1),
        "areaKm2": round(float(bbox.width * bbox.height / 1e6), 3),
        "resolutionM": float(config.coarse_resolution_m),
        "estimatedRows": rows,
        "estimatedCols": cols,
        "estimatedCells": rows * cols,
        "rangeBufferMultiplier": 1.1,
        "description": "Bounding box of sightings padded by max range x 1.1",
    }
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
        "maxRangeM": config.max_range_m,
        "gpuMode": "gpu-preferred" if config.prefer_gpu else "cpu",
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
                        "preferGpu": True,
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
            _json_response(self, _analysis_payload(run_id, job.record))
            return
        if parsed.path == "/api/export/geotiff":
            params = parse_qs(parsed.query)
            run_id = params.get("run_id", [""])[0]
            record = RUN_STORE.get(run_id)
            if record is None:
                _error_response(self, "unknown run_id", status=HTTPStatus.NOT_FOUND)
                return
            data = _geotiff_bytes(record.estimate.probability)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/tiff")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="launchpoint_probability_{run_id[:8]}.tif"',
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
