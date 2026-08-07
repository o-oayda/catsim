"""Build, rebuild, or benchmark the production RACS noise/error caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter
import tracemalloc
from typing import Any

import numpy as np

from catsim import RACS_LOW3, RACS_MID1, Racs, RacsConfig


PRODUCTS = {"low3": RACS_LOW3, "mid1": RACS_MID1}


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=PRODUCTS, required=True)
    parser.add_argument(
        "--noisemap-data-dir",
        help="Directory containing RACS-low3.iqr.hpx and/or RACS-mid1.iqr.hpx",
    )
    parser.add_argument("--catalogue-path", help="Product catalogue used to rebuild the grid")
    parser.add_argument("--noise-map-nside", type=_positive_int, default=256)
    parser.add_argument("--noise-bins", type=_positive_int, default=400)
    parser.add_argument("--flux-bins", type=_positive_int, default=400)
    parser.add_argument("--min-cell-count", type=_positive_int, default=10)
    parser.add_argument(
        "--rebuild",
        choices=("none", "noise", "lookup", "all"),
        default="none",
        help=(
            "Force the selected cache to be replaced. Rebuilding noise also "
            "rebuilds its identity-dependent lookup."
        ),
    )
    parser.add_argument(
        "--benchmark-samples",
        type=_positive_int,
        help="Time this many vectorized runtime lookup samples after loading/building.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--metadata-output",
        help="JSON report path (default: beside the generated caches)",
    )
    return parser.parse_args(argv)


def precompute(args: argparse.Namespace) -> dict[str, Any]:
    cfg = RacsConfig(
        product=PRODUCTS[args.product],
        flux_min=15.0,
        catalogue_path=args.catalogue_path,
        noisemap_data_dir=args.noisemap_data_dir,
        noise_map_nside=args.noise_map_nside,
        flux_error_noise_bins=args.noise_bins,
        flux_error_flux_bins=args.flux_bins,
        flux_error_min_cell_count=args.min_cell_count,
    )
    sim = Racs(cfg)
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _commit(),
        "product": args.product,
        "rebuild": args.rebuild,
        "seed": args.seed,
        "config": {
            "noise_map_nside": cfg.noise_map_nside,
            "flux_error_noise_bins": cfg.flux_error_noise_bins,
            "flux_error_flux_bins": cfg.flux_error_flux_bins,
            "flux_error_min_cell_count": cfg.flux_error_min_cell_count,
            "source_noisemap_filename": cfg.product.source_noisemap_filename,
            "catalogue_path": cfg.catalogue_path,
            "noisemap_data_dir": cfg.noisemap_data_dir,
        },
        "timings_seconds": {},
    }

    rebuild_noise = args.rebuild in {"noise", "all"}
    rebuild_lookup = args.rebuild in {"lookup", "all"} or rebuild_noise
    tracemalloc.start()
    started = perf_counter()

    noise_started = perf_counter()
    noise_loaded = False if rebuild_noise else sim.load_cached_noise_map()
    if not noise_loaded:
        sim.build_cached_noise_map()
        sim.save_cached_noise_map(diagnostics=True)
    report["noise_cache_action"] = "loaded" if noise_loaded else "built"
    report["timings_seconds"]["noise_cache"] = perf_counter() - noise_started

    lookup_started = perf_counter()
    lookup_loaded = False if rebuild_lookup else sim.load_absolute_error_lookup()
    if not lookup_loaded:
        sim.load_catalogue()
        try:
            sim.build_absolute_error_lookup()
            sim.save_absolute_error_lookup(diagnostics=True)
        finally:
            sim.release_catalogue()
    report["lookup_cache_action"] = "loaded" if lookup_loaded else "built"
    report["timings_seconds"]["absolute_error_lookup"] = (
        perf_counter() - lookup_started
    )
    report["timings_seconds"]["total"] = perf_counter() - started

    if args.benchmark_samples is not None:
        lookup = sim.absolute_error_lookup
        rng = np.random.default_rng(args.seed)
        log_noise = rng.uniform(
            lookup.log_noise_edges[0],
            lookup.log_noise_edges[-1],
            args.benchmark_samples,
        )
        log_flux = rng.uniform(
            lookup.log_flux_edges[0],
            lookup.log_flux_edges[-1],
            args.benchmark_samples,
        )
        benchmark_started = perf_counter()
        samples = sim.sample_absolute_flux_errors(
            np.power(10.0, log_noise),
            np.power(10.0, log_flux),
            rng=rng,
        )
        elapsed = perf_counter() - benchmark_started
        report["benchmark"] = {
            "samples": args.benchmark_samples,
            "seconds": elapsed,
            "samples_per_second": args.benchmark_samples / elapsed,
            "finite_fraction": float(np.mean(np.isfinite(samples))),
        }

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    noise_path = sim._noise_map_cache_path()
    lookup_path = sim._absolute_error_lookup_cache_path()
    report["cache_files"] = {
        "noise_map": {
            "path": str(noise_path.resolve()),
            "bytes": noise_path.stat().st_size,
        },
        "absolute_error_lookup": {
            "path": str(lookup_path.resolve()),
            "bytes": lookup_path.stat().st_size,
        },
    }
    report["memory"] = {
        "tracemalloc_current_bytes": current_memory,
        "tracemalloc_peak_bytes": peak_memory,
        # Linux reports KiB. This script is a developer benchmark rather than
        # a portable process-memory API, so retain the raw unit explicitly.
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runtime_lookup_array_bytes": int(
            sim.noise_map_cache.values.nbytes
            + sim.absolute_error_lookup.log_noise_edges.nbytes
            + sim.absolute_error_lookup.log_flux_edges.nbytes
            + sim.absolute_error_lookup.cell_counts.nbytes
            + sim.absolute_error_lookup.cell_starts.nbytes
            + sim.absolute_error_lookup.resolved_cell_ids.nbytes
            + sim.absolute_error_lookup.absolute_error_values.nbytes
        ),
    }
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = precompute(args)
    if args.metadata_output is None:
        cfg = report["config"]
        cache_path = Path(report["cache_files"]["absolute_error_lookup"]["path"])
        output = cache_path.parent / (
            f"noise_lookup_precompute_{args.product}_nside{cfg['noise_map_nside']}.json"
        )
    else:
        output = Path(args.metadata_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"metadata report: {output}")


if __name__ == "__main__":
    main()
