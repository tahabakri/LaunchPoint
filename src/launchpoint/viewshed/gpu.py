"""GPU-accelerated viewshed kernel with a transparent CPU fallback.

Viewsheds are embarrassingly parallel — one independent ray per perimeter cell —
which is exactly the shape a GPU wants. This module provides a CUDA kernel
(via ``numba.cuda``) mirroring the CPU radial-sweep, and a dispatcher that uses
it when a CUDA device is present and silently falls back to the parallel CPU
kernel otherwise. The numerics are identical, so the GPU path is purely a
performance lever for the 1 m refinement (Phase 5), never a correctness change.

If ``cupy`` is preferred over numba.cuda it can be slotted in here; the public
surface is just ``r3_viewshed_best`` / ``gpu_available``.
"""

from __future__ import annotations

import logging

import numpy as np

from launchpoint.viewshed.kernels import r3_viewshed as _r3_cpu

log = logging.getLogger("launchpoint")

try:  # numba.cuda is optional and may not have a usable device.
    from numba import cuda

    _CUDA_IMPORTED = True
    _IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    cuda = None  # type: ignore
    _CUDA_IMPORTED = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def gpu_status() -> tuple[bool, str]:
    """``(available, reason)`` — whether the CUDA viewshed can run, and why not.

    The GPU kernel needs three things to line up: the ``numba.cuda`` bindings
    (the ``numba-cuda`` package), the CUDA toolkit compiler/runtime libraries it
    JITs against (NVVM + cudart, shipped by the ``nvidia-*-cu12`` wheels that
    ``numba-cuda[cu12]`` pulls in), and a usable device. Any missing piece leaves
    the app on the (correct, identical) CPU kernel; this reports which one so the
    fallback is never a silent mystery.
    """
    if not _CUDA_IMPORTED:
        return False, (
            f"numba.cuda import failed ({_IMPORT_ERROR}); install the GPU extra: "
            "pip install 'launchpoint[gpu]'"
        )
    try:
        if not cuda.is_available():
            # A device may be visible to the driver yet unusable because the CUDA
            # toolkit libraries (NVVM/cudart) aren't found — the usual venv cause.
            return False, (
                "no usable CUDA device (driver present but the CUDA toolkit "
                "libraries may be missing — install 'launchpoint[gpu]')"
            )
        if len(cuda.gpus) == 0:
            return False, "CUDA available but no GPUs enumerated"
    except Exception as exc:  # noqa: BLE001
        return False, f"CUDA probe raised {type(exc).__name__}: {exc}"
    name = cuda.get_current_device().name
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return True, f"CUDA device ready ({name})"


def gpu_available() -> bool:
    """True iff a CUDA device is present and usable."""
    return gpu_status()[0]


if _CUDA_IMPORTED:

    @cuda.jit(device=True, inline=True)
    def _dist(dr, dc, res):
        return (dr * res * dr * res + dc * res * dc * res) ** 0.5

    @cuda.jit
    def _r3_ray_kernel(occ, ground, obs_r, obs_c, obs_z, target_h, res,
                       inv_two_reff, max_range, perim_r, perim_c, out):
        """One thread per perimeter target; marches a ray and ORs into out."""
        idx = cuda.grid(1)
        if idx >= perim_r.size:
            return
        rows = occ.shape[0]
        cols = occ.shape[1]
        er = perim_r[idx]
        ec = perim_c[idx]
        dr = er - obs_r
        dc = ec - obs_c
        steps = max(abs(dr), abs(dc))
        if steps == 0:
            return
        max_ang = -1.0e30
        for k in range(1, steps + 1):
            rf = obs_r + dr * k / steps
            cf = obs_c + dc * k / steps
            ri = int(round(rf))
            ci = int(round(cf))
            if ri < 0 or ri >= rows or ci < 0 or ci >= cols:
                continue
            d = _dist(ri - obs_r, ci - obs_c, res)
            if d <= 0.0:
                continue
            if d > max_range:
                break
            drop = d * d * inv_two_reff
            terr_ang = (occ[ri, ci] - drop - obs_z) / d
            targ = ground[ri, ci] + target_h
            targ_ang = (targ - drop - obs_z) / d
            if targ_ang >= max_ang - 1e-9:
                # Multiple rays may touch a cell; OR semantics, race-safe for 0/1.
                out[ri, ci] = 1
            if terr_ang > max_ang:
                max_ang = terr_ang


def _perimeter_indices(rows: int, cols: int):
    rs = []
    cs = []
    for ec in range(cols):
        rs.append(0); cs.append(ec)
        rs.append(rows - 1); cs.append(ec)
    for er in range(rows):
        rs.append(er); cs.append(0)
        rs.append(er); cs.append(cols - 1)
    return np.asarray(rs, dtype=np.int32), np.asarray(cs, dtype=np.int32)


def r3_viewshed_gpu(occ, ground, obs_r, obs_c, obs_z, target_h, res,
                    inv_two_reff, max_range):
    """CUDA radial-sweep viewshed. Raises if no device is available."""
    if not gpu_available():
        raise RuntimeError("no CUDA device available")
    rows, cols = occ.shape
    out = np.zeros((rows, cols), dtype=np.uint8)
    if 0 <= obs_r < rows and 0 <= obs_c < cols:
        out[obs_r, obs_c] = 1
    perim_r, perim_c = _perimeter_indices(rows, cols)

    d_occ = cuda.to_device(np.ascontiguousarray(occ, dtype=np.float64))
    d_ground = cuda.to_device(np.ascontiguousarray(ground, dtype=np.float64))
    d_out = cuda.to_device(out)
    d_pr = cuda.to_device(perim_r)
    d_pc = cuda.to_device(perim_c)

    threads = 128
    blocks = (perim_r.size + threads - 1) // threads
    _r3_ray_kernel[blocks, threads](
        d_occ, d_ground, int(obs_r), int(obs_c), float(obs_z), float(target_h),
        float(res), float(inv_two_reff), float(max_range), d_pr, d_pc, d_out,
    )
    return d_out.copy_to_host()


_WARNED_NO_GPU = False


def r3_viewshed_best(occ, ground, obs_r, obs_c, obs_z, target_h, res,
                     inv_two_reff, max_range, prefer_gpu: bool = True):
    """Run on the GPU when available and requested, else the CPU kernel."""
    if prefer_gpu:
        available, reason = gpu_status()
        if available:
            return r3_viewshed_gpu(occ, ground, obs_r, obs_c, obs_z, target_h, res,
                                   inv_two_reff, max_range)
        # Requested but unusable: warn once (not per viewshed) so the fallback to
        # the CPU kernel is visible rather than silent — the reported symptom.
        global _WARNED_NO_GPU
        if not _WARNED_NO_GPU:
            log.warning("GPU requested but unavailable — using CPU kernel. %s", reason)
            _WARNED_NO_GPU = True
    return _r3_cpu(occ, ground, obs_r, obs_c, obs_z, target_h, res,
                   inv_two_reff, max_range)
