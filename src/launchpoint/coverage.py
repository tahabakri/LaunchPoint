"""Reverse coverage planner: flight area in, launch suitability out."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from launchpoint.config import Config
from launchpoint.core.geo import BBox, Projector
from launchpoint.core.grid import RasterGrid
from launchpoint.viewshed.core import ViewshedInput, compute_viewshed
from launchpoint.viewshed.range_model import RangeModel


@dataclass(frozen=True)
class FlightArea:
    """A circular drone operating area."""

    center_lon: float
    center_lat: float
    radius_m: float
    altitude_agl_m: float


@dataclass
class CoverageEstimate:
    """Launch suitability for one circular flight area."""

    coverage: RasterGrid
    flight_mask: RasterGrid
    projector: Projector
    recommended_lonlat: tuple[float, float]
    top_launch_areas: list[dict]
    sample_points: list[dict]
    metadata: dict


def flight_area_bbox(
    area: FlightArea,
    projector: Projector,
    config: Config,
    *,
    range_model: RangeModel | None = None,
) -> BBox:
    """Metric AOI covering the flight circle plus maximum useful control range."""
    rm = range_model or RangeModel(config.range_model)
    cx, cy = projector.to_utm(area.center_lon, area.center_lat)
    pad = area.radius_m + rm.hard_cap_m
    return BBox(cx - pad, cy - pad, cx + pad, cy + pad)


def flight_area_mask(area: FlightArea, grid: RasterGrid, projector: Projector) -> RasterGrid:
    """Rasterized circular flight area on ``grid``."""
    cx, cy = projector.to_utm(area.center_lon, area.center_lat)
    rows, cols = grid.shape
    rr, cc = np.indices((rows, cols))
    xs, ys = grid.transform * (cc + 0.5, rr + 0.5)
    dist = np.hypot(np.asarray(xs) - cx, np.asarray(ys) - cy)
    return grid.copy_with((dist <= area.radius_m).astype(np.float32))


def _observer_z(
    occluder: RasterGrid,
    ground: RasterGrid | None,
    x: float,
    y: float,
    altitude_agl_m: float,
) -> float:
    base_grid = ground if ground is not None else occluder
    r, c = base_grid.world_to_pixel(x, y)
    if base_grid.contains_pixel(r, c):
        base = base_grid.data[r, c]
        if np.isfinite(base):
            return float(base) + altitude_agl_m
    finite = base_grid.data[np.isfinite(base_grid.data)]
    return (float(np.median(finite)) if finite.size else 0.0) + altitude_agl_m


def _sample_flight_points(
    area: FlightArea,
    mask: RasterGrid,
    projector: Projector,
    max_samples: int,
) -> list[tuple[float, float]]:
    if max_samples <= 0:
        raise ValueError("coverage sample count must be positive")

    rows, cols = np.nonzero(mask.data >= 0.5)
    if rows.size == 0:
        raise ValueError("flight area does not cover any analysis cells")

    cx, cy = projector.to_utm(area.center_lon, area.center_lat)
    points: list[tuple[float, float]] = [(float(cx), float(cy))]
    if rows.size == 1 or max_samples == 1:
        return points[:max_samples]

    order = np.lexsort((cols, rows))
    take_count = min(max_samples - 1, rows.size)
    take = np.unique(np.linspace(0, rows.size - 1, take_count, dtype=int))
    for idx in order[take]:
        x, y = mask.pixel_to_world(float(rows[idx]), float(cols[idx]))
        if np.hypot(x - cx, y - cy) <= area.radius_m:
            points.append((float(x), float(y)))
    return points[:max_samples]


def _top_launch_cells(grid: RasterGrid, projector: Projector, count: int = 5) -> list[dict]:
    data = np.where(np.isfinite(grid.data), grid.data, -np.inf)
    if not np.isfinite(data).any():
        return []
    flat = data.ravel()
    order = np.argsort(flat)[::-1]
    out: list[dict] = []
    seen: list[tuple[int, int]] = []
    min_sep_cells = max(1, int(round(250.0 / max(grid.res_x, grid.res_y))))
    for flat_idx in order:
        value = float(flat[flat_idx])
        if not np.isfinite(value) or value <= 0:
            break
        row, col = np.unravel_index(int(flat_idx), grid.shape)
        if any(abs(row - r) <= min_sep_cells and abs(col - c) <= min_sep_cells for r, c in seen):
            continue
        x, y = grid.pixel_to_world(float(row), float(col))
        lon, lat = projector.to_lonlat(x, y)
        out.append({"lon": float(lon), "lat": float(lat), "score": value})
        seen.append((int(row), int(col)))
        if len(out) >= count:
            break
    return out


def plan_launch_area(
    area: FlightArea,
    config: Config | None = None,
    *,
    occluder: RasterGrid | None = None,
    ground: RasterGrid | None = None,
    projector: Projector | None = None,
    launch_weight: RasterGrid | None = None,
    sample_count: int = 49,
    progress_callback=None,
) -> CoverageEstimate:
    """Score candidate launch cells for control coverage over ``area``."""
    if area.radius_m <= 0 or not np.isfinite(area.radius_m):
        raise ValueError("flight area radius must be positive")
    if area.altitude_agl_m < 0 or not np.isfinite(area.altitude_agl_m):
        raise ValueError("mission altitude must be non-negative")

    config = config or Config()
    if projector is None:
        projector = Projector.for_point(area.center_lon, area.center_lat)

    if occluder is None:
        from launchpoint.data.surface import build_surface_stack_for_bbox

        bbox = flight_area_bbox(area, projector, config)
        stack = build_surface_stack_for_bbox([], config, projector, bbox)
        occluder = stack.occluder
        ground = stack.ground if ground is None else ground
        launch_weight = stack.launch_weight if launch_weight is None else launch_weight

    if progress_callback is not None:
        progress_callback("Run coverage", "running", "Rasterizing flight area", None)

    mask = flight_area_mask(area, occluder, projector)
    points = _sample_flight_points(area, mask, projector, sample_count)
    rm = RangeModel(config.range_model)
    acc = np.zeros(occluder.shape, dtype=np.float64)

    for idx, (x, y) in enumerate(points, start=1):
        if progress_callback is not None:
            progress_callback(
                "Run coverage",
                "running",
                f"Computing coverage sample {idx} of {len(points)}",
                {"current": idx, "total": len(points)},
            )
        vi = ViewshedInput(
            occluder=occluder,
            observer_xy=(x, y),
            observer_z=_observer_z(occluder, ground, x, y, area.altitude_agl_m),
            target_height=config.antenna_height_m,
            ground=ground,
            refraction=True,
        )
        acc += compute_viewshed(vi, rm, prefer_gpu=config.prefer_gpu).data

    coverage = (acc / float(len(points))).astype(np.float32)
    if launch_weight is not None:
        w = np.where(np.isfinite(launch_weight.data), launch_weight.data, 1.0)
        coverage = (coverage * w).astype(np.float32)

    result_grid = occluder.copy_with(coverage)
    top = _top_launch_cells(result_grid, projector)
    recommended = (top[0]["lon"], top[0]["lat"]) if top else (area.center_lon, area.center_lat)

    sample_payload = []
    for x, y in points:
        lon, lat = projector.to_lonlat(x, y)
        sample_payload.append({"lon": float(lon), "lat": float(lat)})

    finite = coverage[np.isfinite(coverage)]
    metadata = {
        "sampleCount": len(points),
        "missionAltitudeAglM": float(area.altitude_agl_m),
        "flightRadiusM": float(area.radius_m),
        "bestCoverage": float(top[0]["score"]) if top else 0.0,
        "meanCandidateScore": float(np.mean(finite)) if finite.size else 0.0,
    }
    if progress_callback is not None:
        progress_callback(
            "Run coverage",
            "complete",
            f"Computed {len(points)} flight-area sample(s)",
            {"current": len(points), "total": len(points)},
        )

    return CoverageEstimate(
        coverage=result_grid,
        flight_mask=mask,
        projector=projector,
        recommended_lonlat=(float(recommended[0]), float(recommended[1])),
        top_launch_areas=top,
        sample_points=sample_payload,
        metadata=metadata,
    )
