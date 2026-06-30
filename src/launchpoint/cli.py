"""Command-line interface for LaunchPoint (Phases 0-5).

Examples
--------
Run a real query from a sightings file (fetches keyless data on first run)::

    launchpoint run --sightings sightings.json --out origin.tif

Run the offline synthetic demo (no network, proves the pipeline end-to-end)::

    launchpoint demo --out demo_origin.tif
"""

from __future__ import annotations

import argparse
import json
import sys

from launchpoint.config import Config, MonteCarloConfig
from launchpoint.core.sighting import Sighting


def _load_sightings(path: str) -> list[Sighting]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    items = raw["sightings"] if isinstance(raw, dict) else raw
    out = []
    for d in items:
        out.append(
            Sighting(
                lat=float(d["lat"]),
                lon=float(d["lon"]),
                altitude=float(d["altitude"]),
                position_sigma_m=float(d.get("position_sigma_m", 250.0)),
                altitude_sigma_m=float(d.get("altitude_sigma_m", 30.0)),
                label=d.get("label"),
            )
        )
    if not out:
        raise SystemExit("no sightings found in file")
    return out


def _config_from_args(args) -> Config:
    return Config(
        max_range_m=args.max_range,
        antenna_height_m=args.antenna_height,
        prefer_gpu=args.gpu,
        cache_dir=args.cache,
        monte_carlo=MonteCarloConfig(samples_per_sighting=args.samples),
    )


def _report(est, out_path: str) -> None:
    est.probability.write_geotiff(out_path)
    lon, lat = est.argmax_lonlat()
    print(f"Wrote probability heatmap: {out_path}")
    print(f"Most-likely operator location (argmax): lat={lat:.6f}, lon={lon:.6f}")
    mask = est.credible_mask(0.5)
    import numpy as np

    cells = int((mask.data == 1.0).sum())
    area_km2 = cells * mask.res_x * mask.res_y / 1e6
    print(f"50% credible region: {cells} cells (~{area_km2:.2f} km^2)")


def _cmd_run(args) -> int:
    from launchpoint.pipeline import find_origin

    sightings = _load_sightings(args.sightings)
    config = _config_from_args(args)
    print(f"Loaded {len(sightings)} sighting(s). Building surfaces / running fusion...")
    est = find_origin(sightings, config=config)
    _report(est, args.out)
    return 0


def _cmd_demo(args) -> int:
    from launchpoint.pipeline import find_origin
    from launchpoint.synthetic import make_default_scenario
    import numpy as np

    sc = make_default_scenario(n_sightings=args.sightings_count, seed=args.seed)
    config = Config(monte_carlo=MonteCarloConfig(samples_per_sighting=args.samples))
    est = find_origin(
        sc.sightings, config=config, occluder=sc.surface, projector=sc.projector
    )
    _report(est, args.out)

    # Report recovery error against the known planted controller.
    lon, lat = est.argmax_lonlat()
    ex, ey = sc.projector.to_utm(lon, lat)
    err = float(np.hypot(ex - sc.controller_xy[0], ey - sc.controller_xy[1]))
    print(f"True controller: lat={sc.controller_lonlat[1]:.6f}, "
          f"lon={sc.controller_lonlat[0]:.6f}")
    print(f"Recovery error (argmax vs truth): {err:.0f} m")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="launchpoint", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="origin.tif", help="output GeoTIFF path")
    common.add_argument("--samples", type=int, default=64,
                        help="Monte-Carlo samples per sighting")

    pr = sub.add_parser("run", parents=[common], help="run on real sightings (network)")
    pr.add_argument("--sightings", required=True, help="JSON file of sightings")
    pr.add_argument("--max-range", type=float, default=12000.0)
    pr.add_argument("--antenna-height", type=float, default=1.5)
    pr.add_argument("--cache", default="data_cache")
    pr.add_argument("--gpu", action="store_true", help="use CUDA viewshed if present")
    pr.set_defaults(func=_cmd_run)

    pd = sub.add_parser("demo", parents=[common], help="offline synthetic demo")
    pd.add_argument("--sightings-count", type=int, default=4)
    pd.add_argument("--seed", type=int, default=5)
    pd.set_defaults(func=_cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
