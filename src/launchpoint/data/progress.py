"""Fetch progress + logging for the data layer.

Two audiences, one source of truth:

* **Terminal** — every fetch logs through ``logging.getLogger("launchpoint.data")``
  at INFO: which layer, how many tiles, the bytes pulled where we can measure
  them, and how long each step took. This is what answers "what is slowest /
  largest" when you watch ``run.bat``.
* **UI** — the same events drive a single ``fetchProgress`` fraction in [0, 1]
  that the server maps onto the run progress bar, so the browser shows real
  download progress instead of a frozen "Fetching surfaces..." label.

``FetchReporter`` is deliberately tolerant: any callback may be None (CLI/library
use), tile totals may be unknown, and a missing layer weight just means that
layer doesn't advance the bar. It never raises into the fetch path.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger("launchpoint.data")

# Stage callback shape shared with the fusion progress (name, status, msg, details).
StageCallback = Callable[[str, str, str, dict | None], None]

# Rough share of the whole fetch phase each layer takes, for the UI bar. Canopy
# is the heavy one (1 m strips); the DSM is a handful of compressed tiles.
_LAYER_WEIGHTS = {"DSM": 0.40, "canopy": 0.45, "buildings": 0.15}


def human_bytes(n: float | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


class FetchReporter:
    """Collects fetch events, logs them, and drives a 0..1 progress fraction."""

    def __init__(self, callback: StageCallback | None = None):
        self._callback = callback
        self._completed_weight = 0.0  # sum of weights of fully-finished layers
        self.total_bytes = 0.0

    # --- per-layer lifecycle -------------------------------------------------
    def layer_start(self, layer: str, total_tiles: int | None = None, note: str = "") -> None:
        suffix = f" ({total_tiles} tile{'s' if total_tiles != 1 else ''})" if total_tiles else ""
        if note:
            suffix += f" - {note}"
        log.info("Fetching %s%s", layer, suffix)
        self._emit(layer, 0, total_tiles or 1, f"Fetching {layer}{suffix}")

    def tile(
        self, layer: str, done: int, total: int, name: str, secs: float,
        n_bytes: float | None = None,
    ) -> None:
        if n_bytes:
            self.total_bytes += n_bytes
        size = f" - {human_bytes(n_bytes)}" if n_bytes else ""
        log.info("  [%s] %d/%d  %s  in %.2fs%s", layer, done, total, name, secs, size)
        frac = self._completed_weight + _LAYER_WEIGHTS.get(layer, 0.0) * (done / max(total, 1))
        self._emit(layer, done, total, f"{layer}: {done}/{total} tiles", frac)

    def note(self, message: str, n_bytes: float | None = None) -> None:
        """A non-tile log line (tile index download, overpass query, ...)."""
        if n_bytes:
            self.total_bytes += n_bytes
        size = f" ({human_bytes(n_bytes)})" if n_bytes else ""
        log.info("  %s%s", message, size)

    def layer_done(self, layer: str, secs: float) -> None:
        self._completed_weight = min(1.0, self._completed_weight + _LAYER_WEIGHTS.get(layer, 0.0))
        log.info("Finished %s in %.2fs", layer, secs)
        self._emit(layer, 1, 1, f"{layer} ready", self._completed_weight)

    def done(self) -> None:
        log.info("All surfaces fetched — %s pulled this run", human_bytes(self.total_bytes))

    # --- internal ------------------------------------------------------------
    def _emit(self, layer: str, current: int, total: int, message: str,
              fraction: float | None = None) -> None:
        if self._callback is None:
            return
        details = {"layer": layer, "current": current, "total": total}
        if fraction is not None:
            details["fetchProgress"] = round(min(max(fraction, 0.0), 1.0), 3)
        try:
            self._callback("Fetch surfaces", "running", message, details)
        except Exception:  # noqa: BLE001 - progress is best-effort, never fatal
            pass
