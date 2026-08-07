"""Run small batched RACS-mid1 JAX simulations for performance checks."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import warnings

import jax
import numpy as np

from catsim import RACS_MID1, RacsConfig, RacsJax


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batched RACS-mid1 JAX map simulations and print timings."
    )
    parser.add_argument("--n-sims", type=_positive_int, default=8)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--chunk-size", type=_positive_int, default=50_000)
    parser.add_argument("--log10-n", type=float, default=4.0)
    parser.add_argument("--flux-min", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--noisemap-data-dir")
    parser.add_argument(
        "--output",
        help="Optional pinned ensemble .npz for compare_racs_noise_ensembles.py.",
    )
    parser.add_argument("--max-children", type=int, default=16)
    parser.add_argument(
        "--cluster-model",
        choices=("geometric", "poisson"),
        default="geometric",
    )
    parser.add_argument("--p-clus", type=float, default=0.0)
    parser.add_argument("--clus-stop-prob", type=float, default=1.0)
    parser.add_argument("--lambda-clus", type=float, default=0.0)
    parser.add_argument("--observer-speed", type=float, default=1.0)
    parser.add_argument("--temp-beta", type=float, default=0.0)
    parser.add_argument("--elevation-amp", type=float, default=0.0)
    parser.add_argument("--elevation-trough", type=float, default=0.0)
    parser.add_argument("--fractional-error-eta", type=float, default=0.0)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the one-simulation compile/warmup call.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar for timed batch generation.",
    )
    return parser.parse_args()


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


def _block_until_ready(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    for array in arrays:
        jax.block_until_ready(array)


def main() -> None:
    args = parse_args()
    print("JAX devices:", ", ".join(str(device) for device in jax.devices()))

    cfg = RacsConfig(
        product=RACS_MID1,
        flux_min=args.flux_min,
        chunk_size=args.chunk_size,
        store_final_samples=False,
        cluster_count_model=args.cluster_model,
        max_cluster_children_per_parent=args.max_children,
        noisemap_data_dir=args.noisemap_data_dir,
    )
    sim = RacsJax(cfg)

    t0 = perf_counter()
    sim.initialise_data()
    initialise_elapsed = perf_counter() - t0
    print(f"initialise_data: {initialise_elapsed:.3f} s")

    theta = {
        "log10_n_initial_samples": np.full(args.n_sims, args.log10_n, dtype=np.float32),
        "observer_speed": np.full(args.n_sims, args.observer_speed, dtype=np.float32),
        "temp_beta": np.full(args.n_sims, args.temp_beta, dtype=np.float32),
        "elevation_amp": np.full(args.n_sims, args.elevation_amp, dtype=np.float32),
        "elevation_trough": np.full(
            args.n_sims,
            args.elevation_trough,
            dtype=np.float32,
        ),
        "fractional_error_eta": np.full(
            args.n_sims,
            args.fractional_error_eta,
            dtype=np.float32,
        ),
    }
    if args.cluster_model == "geometric":
        theta["p_clus"] = np.full(args.n_sims, args.p_clus, dtype=np.float32)
        theta["clus_stop_prob"] = np.full(
            args.n_sims,
            args.clus_stop_prob,
            dtype=np.float32,
        )
    else:
        theta["lambda_clus"] = np.full(args.n_sims, args.lambda_clus, dtype=np.float32)

    key = jax.random.PRNGKey(args.seed)

    if not args.skip_warmup:
        warmup_count = min(args.batch_size, args.n_sims)
        warmup_theta = {name: values[:warmup_count] for name, values in theta.items()}
        t0 = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("always", RuntimeWarning)
            warmup = sim.batch_generate_dipole(
                warmup_theta,
                key,
                batch_size=warmup_count,
            )
        _block_until_ready(warmup)
        print(f"warmup/compile: {perf_counter() - t0:.3f} s")

    t0 = perf_counter()
    maps, masks = sim.batch_generate_dipole(
        theta,
        key,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    _block_until_ready((maps, masks))
    elapsed = perf_counter() - t0

    parent_sources = int(10**args.log10_n)
    total_requested = parent_sources * args.n_sims
    print(f"batch_generate_dipole: {elapsed:.3f} s")
    print(f"simulations: {args.n_sims}")
    print(f"map shape: {maps.shape}, mask shape: {masks.shape}")
    print(f"mean kept sources/map: {float(np.nanmean(np.nansum(maps, axis=1))):.3f}")
    print(f"simulations/sec: {args.n_sims / elapsed:.3f}")
    print(f"requested parent-source slots/sec: {total_requested / elapsed:.3e}")
    device_memory_stats: dict[str, object] = {}
    for device in jax.devices():
        try:
            stats = device.memory_stats()
            device_memory_stats[str(device)] = stats
            print(f"device memory stats ({device}): {stats}")
        except (AttributeError, RuntimeError, TypeError):
            device_memory_stats[str(device)] = None
            print(f"device memory stats ({device}): unavailable")

    if args.output is not None:
        if sim.last_invalid_noise_rejection_maps is None:
            raise RuntimeError("JAX simulator did not expose invalid-noise diagnostics.")
        metadata = {
            "commit": _commit(),
            "config": {
                "product": sim.product.key,
                "nside": sim.nside,
                "chunk_size": sim.chunk_size,
                "batch_size": args.batch_size,
                "flux_min": args.flux_min,
                "log10_n_initial_samples": args.log10_n,
                "cluster_count_model": args.cluster_model,
                "max_cluster_children_per_parent": args.max_children,
                "p_clus": args.p_clus,
                "clus_stop_prob": args.clus_stop_prob,
                "lambda_clus": args.lambda_clus,
                "observer_speed": args.observer_speed,
                "temp_beta": args.temp_beta,
                "elevation_amp": args.elevation_amp,
                "elevation_trough": args.elevation_trough,
                "fractional_error_eta": args.fractional_error_eta,
                "alpha_mean": cfg.alpha_mean,
                "alpha_sigma": cfg.alpha_sigma,
                "cluster_r0_arcsec": cfg.cluster_r0_arcsec,
                "cluster_r_cut_arcsec": cfg.cluster_r_cut_arcsec,
                "temperature_model": cfg.temperature_model,
                "paf_reference_temp_c": cfg.paf_reference_temp_c,
                "noise_map_nside": cfg.noise_map_nside,
                "flux_error_noise_bins": cfg.flux_error_noise_bins,
                "flux_error_flux_bins": cfg.flux_error_flux_bins,
                "flux_error_min_cell_count": cfg.flux_error_min_cell_count,
                "warmup_enabled": not args.skip_warmup,
            },
            "seeds": {"root_prng_key_seed": args.seed, "n_simulations": args.n_sims},
            "cache_identities": {
                "noise_map": sim.noise_map_cache_identity,
                "absolute_error_lookup": sim.absolute_error_lookup_identity,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "jax": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
            },
            "performance": {
                "initialise_seconds": initialise_elapsed,
                "simulation_seconds": elapsed,
                "simulations_per_second": args.n_sims / elapsed,
                "requested_parent_source_slots_per_second": (
                    total_requested / elapsed
                ),
                "device_memory_stats": device_memory_stats,
            },
            # Root seeds alone do not verify source-level pairing.
            "pairing_verified": False,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            maps=maps,
            rejected_invalid_noise_maps=sim.last_invalid_noise_rejection_maps,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
        )
        print(f"ensemble artifact: {output}")


if __name__ == "__main__":
    main()
