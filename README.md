# LaunchPoint — Controller-Origin Finder

Estimate **where a drone's operator is standing** from one or more ground-position
sightings of the drone.

The method rests on a simple fact: **line-of-sight is reciprocal.** Every place
the drone could see is a place from which the controller could see the drone — and
a controller has to keep (rough) line-of-sight to fly. So LaunchPoint intersects
the *visible regions* of several sightings on a high-fidelity height surface,
weights them by a realistic control-range model and terrain/canopy occlusion, and
propagates the (large) input uncertainty with Monte Carlo to produce a
**probability heatmap** of the operator's location.

> Output is probabilistic. ~100% accuracy is neither expected nor claimed.
> When a drone broadcasts Remote ID, the operator location is already known — this
> is the tool for when it *isn't*.

This repository implements **Phases 0–6** of the project roadmap: core science,
performance, and the first interactive command-center workspace.

---

## Quick start

### Windows (one click)

```bat
install_dependencies.bat
.venv\Scripts\activate
launchpoint demo --out demo_origin.tif
launchpoint ui
```

The installer finds a compatible Python (3.10–3.12), creates a `.venv`, and
installs everything. `demo` runs entirely offline on a synthetic scenario and
prints how close the recovered peak is to the planted controller.

### Any platform (manual)

```bash
python -m venv .venv && . .venv/bin/activate     # use a 3.10–3.12 interpreter
pip install -r requirements.txt
pip install -e .
pytest                       # 32 offline tests
launchpoint demo --out demo_origin.tif
```

> **Python version:** the geospatial stack (rasterio/GDAL, pyproj, shapely,
> geopandas) ships binary wheels for CPython **3.10–3.12**. 3.13+ is not yet
> supported by those wheels.

### Running on real sightings

```bash
launchpoint run --sightings examples/sightings_zurich.json --out origin.tif
```

On first run this fetches only the data your area needs (Copernicus DSM, plus
canopy and OSM buildings) over keyless HTTPS, caches it under `data_cache/`, and
writes a GeoTIFF probability raster. No account or API key is required for any
data source.

### Interactive workspace

```bash
launchpoint ui --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in a browser. The workspace starts as a blank
operational map. Add sightings directly on the map or import JSON/CSV matching
the existing sighting schema, then run the analysis and inspect the 2D heatmap,
50% credible region, linked 3D terrain view, per-sighting contributions, cell
diagnostics, and exports.

The 2D base map uses OpenStreetMap public tiles with attribution. The 3D terrain
view decodes keyless Terrarium elevation PNG tiles from the AWS Open Data
Terrain Tiles bucket and drapes the same probability heatmap palette used by the
2D viewport and PNG preview. GeoTIFF export is served from the full backend
probability raster.

---

## How it works

| Phase | What it does |
|------|---------------|
| **0** | Core data structures (`Sighting`, `RasterGrid`), local-UTM projection rule, and a **synthetic validation harness** — a fully-known terrain + planted controller + geometrically-visible sightings — so every later phase has objective ground truth. |
| **1** | **Keyless data layer.** Computes the area of interest (sightings + max range), then reads windowed Cloud-Optimized GeoTIFFs (HTTP range requests) for the **Copernicus GLO-30 DSM**, reprojected/mosaicked onto the common UTM grid and cached. |
| **2** | **Single-observer viewshed.** A radial-sweep (R3) line-of-sight kernel with **4/3-Earth curvature + refraction** correction and a **soft range model** (plateau → logistic roll-off, not a hard disk). A brute-force oracle validates it. |
| **3** | **Fusion + Monte Carlo.** Scores each candidate cell by visibility from every sighting, sampling each sighting's position/altitude uncertainty N times. Sightings are combined with a **strict minimum** by default (`combine="min"`) — the cell value is gated by the *weakest* sighting, so a single drone that cannot see a location drives it to ~0. This enforces the single-launch-point assumption: every drone must be visible from the operator's spot. `combine="geometric_mean"` is a softer intersection and `"arithmetic_mean"` the forgiving union. |
| **4** | **Surface enrichment.** Burns OSM building heights and the **Meta/WRI 1 m canopy** into the picture, and *derives* a bare-earth surface `bare = DSM − canopy − building` (no FABDEM dependency, all commercial-friendly). Canopy also drives a soft launch-feasibility down-weight. The 1 m canopy is fetched **only inside the DSM-reachable footprint** (see below), not the whole range disk. |
| **5** | **Performance.** Coarse-to-fine: a cheap 30 m pass over the whole disk, then fine (1–2 m) refinement **only** in hot candidate patches and around each sighting. A CUDA viewshed kernel (with automatic CPU fallback) accelerates the fine pass. |

### Geometry conventions

* All geometry is reprojected to a **local UTM zone** so raycasting happens in
  metres, never in lat/lon degrees.
* The **DSM occludes** the ray along its path (it already includes canopy and
  rooftops). The **derived bare earth** gives the *target height*: the operator
  stands on bare earth + an antenna height (~1.5 m), not on the canopy top or a
  roof.
* Observer = the drone at its (sampled) position and altitude. AGL altitudes are
  converted to absolute using the surface beneath the drone.

### Footprint-gated canopy fetch

The 1 m Meta/WRI canopy is the bandwidth-heavy layer (its tiles are full-width
single-row strips with no overviews, so a naïve windowed read of the whole range
disk can pull ~1 GB). Canopy only matters where a controller could actually
stand, so LaunchPoint first runs a cheap **bare-DSM viewshed** per sighting to
find the *reachable footprint*, then fetches high-resolution canopy **only inside
that footprint's bounding box**.

Crucially, the footprint is combined the **same way the heatmap is**. Under the
default strict `min`, a viable launch point must be seen by *every* drone, so
canopy is only needed in the **intersection** of the per-sighting viewsheds — not
their union. That is a much smaller area (roughly half the AOI even on flat
synthetic terrain, far less where terrain occludes), and it falls out of the same
single-launch-point assumption that drives the heatmap. The softer `geometric_mean`
/ `arithmetic_mean` modes fall back to the union, since they keep cells only some
drones reach.

This stays lossless: each per-sighting DSM-only viewshed is a superset of its true
canopy-aware contribution (canopy only lowers the antenna target, shrinking
visibility), and the footprint additionally lifts the observer by ~2σ of altitude
and dilates for position jitter — so canopy is fetched everywhere the heatmap
could be non-zero and nowhere it can't.

---

## Architecture

```
src/launchpoint/
  config.py            # all tunables / physical constants (one place to cite)
  core/                # Sighting, geo/UTM helpers, RasterGrid
  viewshed/            # curvature, soft range model, R3 + oracle kernels, GPU
  fusion/              # Monte-Carlo fusion, coarse-to-fine multiscale
  data/                # keyless fetchers: Copernicus DSM, canopy, OSM buildings,
                       #   derived bare earth, on-disk cache, surface stack
  synthetic/           # the validation harness
  pipeline.py          # sightings in -> OriginEstimate out
  ui/                  # Phase 6 browser workspace and shared overlay renderer
  cli.py               # `launchpoint run` / `launchpoint demo` / `launchpoint ui`
tests/                 # 32 offline tests + network-gated data tests
```

Programmatic use:

```python
from launchpoint import Sighting, find_origin

sightings = [Sighting(47.39, 8.55, altitude=110, position_sigma_m=200),
             Sighting(47.38, 8.57, altitude=95,  position_sigma_m=250)]
est = find_origin(sightings)                 # fetches keyless data for the AOI
print(est.argmax_lonlat())                   # most-likely operator location
est.probability.write_geotiff("origin.tif")  # full heatmap
```

---

## Data sources (all free, all keyless)

| Layer | Source | Access | License |
|---|---|---|---|
| DSM (surface / occlusion) | Copernicus GLO-30 | `copernicus-dem-30m` S3 / HTTPS, anonymous | Open |
| Canopy height (1 m) | Meta / WRI Global Canopy Height | `dataforgood-fb-data` S3 / HTTPS, anonymous | Free incl. commercial |
| Buildings | OpenStreetMap (Overpass API) | keyless HTTP | ODbL |
| Bare earth | **Derived** = DSM − canopy − building | computed in-house | inherits above |

No Google Earth Engine path is used (it requires an account). No FABDEM is fetched
(its bare earth is derived instead), which keeps every input commercial-friendly.

### Fetching: parallel + observable

Remote COG tiles (DSM, canopy) are pulled **concurrently** through a small thread
pool — each worker holds its own GDAL handle, and tiles are composited back in
their original order so the result is identical to a serial mosaic, just faster.
The shared, rate-limited Overpass building API stays a single sequential request.

Every fetch is logged to the terminal (the window `run.bat` opens) via the
`launchpoint.data` logger: which layer, tile-by-tile timing, bytes pulled where
measurable, and per-layer totals — so it's obvious what is slow or large. The UI
turns the same events into a live download percentage on the run progress bar.
Run the CLI with `--verbose` for debug-level detail.

---

## Tests

```bash
pytest                 # offline: synthetic recovery, R3-vs-oracle, fusion, ...
pytest -m network      # also hit the live keyless endpoints (DSM/canopy/OSM)
```

The headline test (`tests/test_recovery.py`) asserts the pipeline recovers the
planted synthetic controller and **degrades gracefully** — the credible region
gets *wider*, not *wrong*, as input uncertainty grows.

---

## Limitations / error budget

* **Probabilistic, not a fix.** The heatmap is a likelihood surface; the peak can
  be off by hundreds of metres to kilometres, especially with few sightings or
  high-altitude drones (a high drone is visible from a large area, so it weakly
  constrains the operator).
* **30 m base resolution** for the DSM — well matched to the few-hundred-metre
  position uncertainty of a single snapshot, but it is the floor on coarse-pass
  detail.
* **Derived bare earth is blunter than FABDEM's validated ML model**, especially
  in complex terrain; it is a straight `DSM − canopy − building` subtraction.
* **Soft range, soft canopy.** Control range and launch feasibility are
  *tendencies*, not rules — operators do stand at forest edges and clearings.
* **Buildings depend on OSM coverage**; sparse areas under-occlude (Microsoft /
  Google open-building gap-fill is wired as an optional extension).

---

## Status

Phases 0–6 are implemented and tested. Phase 7 (release polish) follows.

## License

MIT for the code (see `LICENSE`). Data layers retain their own licenses; note
OpenStreetMap's **ODbL** share-alike terms apply to any redistributed extracts.
