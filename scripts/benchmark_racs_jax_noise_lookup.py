"""Benchmark one RACS MID1 JAX implementation with a fixed science setup.

This entry point is intentionally usable against either the current source
tree or an exported historical source tree.  Select the implementation by
putting the desired ``src`` directory first on ``PYTHONPATH``.  The output
records the imported module paths so accidental editable-install leakage is
detectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from typing import Any
import warnings

import jax
import jaxlib
import numpy as np

import catsim
import catsim.racs_jax
from catsim import RACS_MID1, RacsConfig, RacsJax
from dipolesbi.pipelines.racs_observation_helpers import build_mask


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a RACS MID1 JAX implementation and write JSON."
    )
    parser.add_argument("--implementation", required=True, choices=("legacy", "new"))
    parser.add_argument("--code-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-batches", type=_positive_int, default=5)
    parser.add_argument("--batch-size", type=_positive_int, default=5)
    parser.add_argument("--chunk-size", type=_positive_int, default=140_000)
    parser.add_argument("--seed", type=int, default=74_000)
    parser.add_argument("--compile-seed", type=int, default=73_000)
    parser.add_argument("--log10-n", type=float, default=6.553988)
    parser.add_argument("--observer-speed", type=float, default=4.9699368)
    parser.add_argument("--dipole-longitude", type=float, default=169.235155)
    parser.add_argument("--dipole-latitude", type=float, default=43.202411)
    parser.add_argument("--temp-beta", type=float, default=0.006762472)
    parser.add_argument("--lambda-clus", type=float, default=0.579619122)
    parser.add_argument("--flux-min", type=float, default=15.0)
    parser.add_argument("--nside", type=_positive_int, default=64)
    parser.add_argument("--downscale-nside", type=_positive_int, default=4)
    parser.add_argument("--max-children", type=int, default=5)
    parser.add_argument(
        "--paf-temperature-data-dir",
        default="/home/oliver/Documents/dipole-utils/data/paf_temps",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; this benchmark is currently Linux-only.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _system_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        return None
    return None


def _device_memory_stats() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for device in jax.devices():
        try:
            result[str(device)] = _jsonable(device.memory_stats())
        except (AttributeError, RuntimeError, TypeError) as exc:
            result[str(device)] = {"available": False, "reason": str(exc)}
    return result


def _block_until_ready(outputs: tuple[np.ndarray, np.ndarray]) -> None:
    for output in outputs:
        jax.block_until_ready(output)


def _synchronise_lookups(simulator: RacsJax) -> None:
    lookup = simulator._lookup_arrays
    if lookup is None:
        raise RuntimeError("JAX lookup arrays were not initialized")
    for array in lookup.as_tuple():
        jax.block_until_ready(array)


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "sample_standard_deviation": (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        ),
        "maximum": float(np.max(array)),
    }


def _theta(args: argparse.Namespace) -> dict[str, np.ndarray]:
    def constant(value: float) -> np.ndarray:
        return np.full(args.batch_size, value, dtype=np.float32)

    return {
        "log10_n_initial_samples": constant(args.log10_n),
        "observer_speed": constant(args.observer_speed),
        "dipole_longitude": constant(args.dipole_longitude),
        "dipole_latitude": constant(args.dipole_latitude),
        "temp_beta": constant(args.temp_beta),
        "lambda_clus": constant(args.lambda_clus),
        "elevation_amp": constant(0.0),
        "elevation_trough": constant(0.0),
        "fractional_error_eta": constant(0.0),
    }


def _run_batch(
    simulator: RacsJax,
    theta: dict[str, np.ndarray],
    *,
    seed: int,
    batch_size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("always", RuntimeWarning)
        outputs = simulator.batch_generate_dipole(
            theta,
            jax.random.PRNGKey(seed),
            batch_size=batch_size,
            show_progress=False,
        )
    _block_until_ready(outputs)
    elapsed = perf_counter() - started
    return elapsed, outputs[0], outputs[1]


def main() -> None:
    args = parse_args()
    if args.max_children < 0:
        raise ValueError("--max-children must be non-negative")
    if args.batch_size != 5:
        raise ValueError("This controlled comparison requires --batch-size=5")
    if args.chunk_size != 140_000:
        raise ValueError("This controlled comparison requires --chunk-size=140000")

    mask = np.asarray(
        build_mask(args.nside, source_radii_deg={"Cygnus A": 3.0}),
        dtype=np.bool_,
    )
    cfg = RacsConfig(
        product=RACS_MID1,
        flux_min=args.flux_min,
        nside=args.nside,
        downscale_nside=args.downscale_nside,
        chunk_size=args.chunk_size,
        store_final_samples=False,
        mask_map=mask,
        cluster_count_model="poisson",
        max_cluster_children_per_parent=args.max_children,
        paf_temperature_data_dir=args.paf_temperature_data_dir,
    )
    simulator = RacsJax(cfg)
    memory_phases: dict[str, Any] = {
        "before_initialise_current_rss_bytes": _rss_bytes(),
        "before_initialise_peak_rss_bytes": _peak_rss_bytes(),
    }

    started = perf_counter()
    simulator.initialise_data()
    _synchronise_lookups(simulator)
    initialise_seconds = perf_counter() - started
    memory_phases.update(
        {
            "after_initialise_current_rss_bytes": _rss_bytes(),
            "after_initialise_peak_rss_bytes": _peak_rss_bytes(),
            "after_initialise_device_stats": _device_memory_stats(),
        }
    )

    theta = _theta(args)
    compile_seconds, compile_maps, compile_masks = _run_batch(
        simulator,
        theta,
        seed=args.compile_seed,
        batch_size=args.batch_size,
    )
    memory_phases.update(
        {
            "after_compile_current_rss_bytes": _rss_bytes(),
            "after_compile_peak_rss_bytes": _peak_rss_bytes(),
            "after_compile_device_stats": _device_memory_stats(),
        }
    )

    expected_shape = (args.batch_size, 12 * args.downscale_nside**2)
    if compile_maps.shape != expected_shape or compile_masks.shape != expected_shape:
        raise RuntimeError(
            f"unexpected compiled output shapes: {compile_maps.shape}, "
            f"{compile_masks.shape}; expected {expected_shape}"
        )

    expected_sources_per_sim = int(10.0**args.log10_n)
    parent_sources_per_sim = int(expected_sources_per_sim / (1.0 + args.lambda_clus))
    n_chunks = max(1, math.ceil(parent_sources_per_sim / args.chunk_size))
    padded_parent_slots_per_sim = n_chunks * args.chunk_size
    padded_source_slots_per_sim = padded_parent_slots_per_sim * (1 + args.max_children)
    slots_per_batch = padded_source_slots_per_sim * args.batch_size

    batch_records: list[dict[str, Any]] = []
    all_counts: list[int] = []
    all_invalid_noise_counts: list[int] = []
    reference_mask = compile_masks[0]
    for index in range(args.measured_batches):
        seed = args.seed + index
        elapsed, maps, masks = _run_batch(
            simulator,
            theta,
            seed=seed,
            batch_size=args.batch_size,
        )
        if maps.shape != expected_shape or masks.shape != expected_shape:
            raise RuntimeError(f"unexpected output shape in measured batch {index}")
        if not np.all(masks == reference_mask):
            raise RuntimeError(f"mask changed in measured batch {index}")
        retained_counts = np.nansum(maps, axis=1, dtype=np.float64).astype(np.int64)
        if np.any(retained_counts <= 0):
            raise RuntimeError(f"non-positive retained count in measured batch {index}")
        all_counts.extend(int(value) for value in retained_counts)

        invalid_counts: list[int] | None = None
        if hasattr(simulator, "last_invalid_noise_rejection_counts"):
            raw_invalid = simulator.last_invalid_noise_rejection_counts
            if raw_invalid is not None:
                invalid_counts = [int(value) for value in np.asarray(raw_invalid)]
                all_invalid_noise_counts.extend(invalid_counts)

        batch_records.append(
            {
                "batch_index": index,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "simulations_per_second": args.batch_size / elapsed,
                "expected_sources_per_second": (
                    expected_sources_per_sim * args.batch_size / elapsed
                ),
                "padded_parent_slots_per_second": (
                    padded_parent_slots_per_sim * args.batch_size / elapsed
                ),
                "padded_source_slots_per_second": slots_per_batch / elapsed,
                "retained_source_counts": [int(value) for value in retained_counts],
                "invalid_noise_rejection_counts": invalid_counts,
            }
        )

    memory_phases.update(
        {
            "final_current_rss_bytes": _rss_bytes(),
            "final_peak_rss_bytes": _peak_rss_bytes(),
            "final_device_stats": _device_memory_stats(),
        }
    )
    elapsed_values = [record["elapsed_seconds"] for record in batch_records]
    simulations_per_second = [
        record["simulations_per_second"] for record in batch_records
    ]
    source_slots_per_second = [
        record["padded_source_slots_per_second"] for record in batch_records
    ]

    lookup_shapes = {
        f"array_{index}": list(np.shape(array))
        for index, array in enumerate(simulator._lookup_arrays.as_tuple())
    }
    result = {
        "schema_version": 1,
        "implementation": args.implementation,
        "code_identity": args.code_identity,
        "imported_modules": {
            "catsim": str(Path(catsim.__file__).resolve()),
            "catsim.racs_jax": str(Path(catsim.racs_jax.__file__).resolve()),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "system_memory_bytes": _system_memory_bytes(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "environment_variables": {
                name: os.environ.get(name)
                for name in (
                    "JAX_PLATFORMS",
                    "JAX_ENABLE_X64",
                    "XLA_FLAGS",
                    "OMP_NUM_THREADS",
                )
            },
        },
        "configuration": {
            "product": "mid1",
            "flux_min_mjy": args.flux_min,
            "nside": args.nside,
            "downscale_nside": args.downscale_nside,
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "cluster_count_model": "poisson",
            "max_cluster_children_per_parent": args.max_children,
            "paf_temperature_data_dir": str(
                Path(args.paf_temperature_data_dir).expanduser().resolve()
            ),
            "mask": {
                "builder": "dipolesbi.pipelines.racs_observation_helpers.build_mask",
                "source_radii_deg": {"Cygnus A": 3.0},
                "kept_pixels": int(np.count_nonzero(mask)),
                "total_pixels": int(mask.size),
                "sha256": _sha256_array(mask),
            },
            "theta": {
                "log10_n_initial_samples": args.log10_n,
                "observer_speed": args.observer_speed,
                "dipole_longitude_deg_galactic": args.dipole_longitude,
                "dipole_latitude_deg_galactic": args.dipole_latitude,
                "temp_beta": args.temp_beta,
                "lambda_clus": args.lambda_clus,
                "elevation_amp": 0.0,
                "elevation_trough": 0.0,
                "fractional_error_eta": 0.0,
            },
            "parameter_conventions": {
                "observer_speed": (
                    "dimensionless multiplier of catsim.utils.constants.CMB_BETA"
                ),
                "dipole_direction": (
                    "Galactic longitude/latitude in degrees, transformed internally "
                    "to equatorial coordinates"
                ),
            },
        },
        "workload": {
            "compile_batches": 1,
            "measured_batches": args.measured_batches,
            "measured_simulations": args.measured_batches * args.batch_size,
            "expected_sources_per_simulation": expected_sources_per_sim,
            "parent_sources_per_simulation": parent_sources_per_sim,
            "chunks_per_simulation": n_chunks,
            "padded_parent_slots_per_simulation": padded_parent_slots_per_sim,
            "padded_source_slots_per_simulation": padded_source_slots_per_sim,
        },
        "lookup_array_shapes": lookup_shapes,
        "timings": {
            "initialise_data_seconds": initialise_seconds,
            "first_full_batch_compile_and_generate_seconds": compile_seconds,
            "measured_batches": batch_records,
            "steady_state_batch_seconds": _summary(elapsed_values),
            "steady_state_simulations_per_second": _summary(simulations_per_second),
            "steady_state_padded_source_slots_per_second": _summary(
                source_slots_per_second
            ),
        },
        "sanity_checks": {
            "output_shape": list(expected_shape),
            "output_mask_kept_pixels": int(np.count_nonzero(reference_mask)),
            "compile_retained_source_counts": [
                int(value)
                for value in np.nansum(
                    compile_maps, axis=1, dtype=np.float64
                ).astype(np.int64)
            ],
            "measured_retained_source_counts": all_counts,
            "measured_retained_source_count_summary": _summary(
                [float(value) for value in all_counts]
            ),
            "measured_invalid_noise_rejection_counts": (
                all_invalid_noise_counts if all_invalid_noise_counts else None
            ),
            "all_measured_masks_equal": True,
            "all_retained_counts_positive": True,
        },
        "memory": memory_phases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
