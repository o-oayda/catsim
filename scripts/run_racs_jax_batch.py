"""Run small batched RACS-low3 JAX simulations for performance checks."""

from __future__ import annotations

import argparse
from time import perf_counter
import warnings

import jax
import numpy as np

from catsim import RacsLow3Config, RacsLow3Jax


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batched RACS-low3 JAX map simulations and print timings."
    )
    parser.add_argument("--n-sims", type=_positive_int, default=8)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--chunk-size", type=_positive_int, default=50_000)
    parser.add_argument("--log10-n", type=float, default=4.0)
    parser.add_argument("--flux-min", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=123)
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


def _block_until_ready(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    for array in arrays:
        jax.block_until_ready(array)


def main() -> None:
    args = parse_args()
    print("JAX devices:", ", ".join(str(device) for device in jax.devices()))

    cfg = RacsLow3Config(
        flux_min=args.flux_min,
        chunk_size=args.chunk_size,
        store_final_samples=False,
        cluster_count_model=args.cluster_model,
        max_cluster_children_per_parent=args.max_children,
    )
    sim = RacsLow3Jax(cfg)

    t0 = perf_counter()
    sim.initialise_data()
    print(f"initialise_data: {perf_counter() - t0:.3f} s")

    theta = {
        "log10_n_initial_samples": np.full(args.n_sims, args.log10_n, dtype=np.float32),
        "observer_speed": np.full(args.n_sims, args.observer_speed, dtype=np.float32),
        "temp_beta": np.full(args.n_sims, args.temp_beta, dtype=np.float32),
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


if __name__ == "__main__":
    main()
