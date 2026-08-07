"""Compare pinned legacy and noise-conditioned RACS ensemble artifacts.

The script is deliberately branch-agnostic: generate each artifact from a
pinned commit/worktree, then pass the two saved ``.npz`` files here. Required
artifact arrays are ``maps`` and ``rejected_invalid_noise_maps``, both shaped
``(n_simulations, 49152)`` in NESTED nside-64 ordering. The latter should be an
explicit zero array for an implementation which never rejects invalid noise.

Each artifact must also contain scalar JSON ``metadata_json`` pinning the
commit, complete effective science configuration, seeds, cache identities,
and environment. Optional ``dipole`` and ``angular_power`` arrays may be
embedded or supplied separately on the command line.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import warnings

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


REQUIRED_METADATA = (
    "commit",
    "config",
    "seeds",
    "cache_identities",
    "environment",
)
REQUIRED_SCIENCE_CONFIG = (
    "product",
    "nside",
    "chunk_size",
    "batch_size",
    "flux_min",
    "log10_n_initial_samples",
    "cluster_count_model",
    "max_cluster_children_per_parent",
    "p_clus",
    "clus_stop_prob",
    "lambda_clus",
    "observer_speed",
    "temp_beta",
    "elevation_amp",
    "elevation_trough",
    "fractional_error_eta",
    "alpha_mean",
    "alpha_sigma",
    "cluster_r0_arcsec",
    "cluster_r_cut_arcsec",
    "temperature_model",
    "paf_reference_temp_c",
)


def _load_metadata(value: NDArray[Any]) -> dict[str, Any]:
    try:
        metadata = json.loads(str(np.asarray(value).item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json must be a scalar JSON object") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode a JSON object")
    return metadata


def load_artifact(
    path: str | Path,
    *,
    allow_incomplete_metadata: bool = False,
) -> dict[str, Any]:
    """Load and validate one pinned nside-64 ensemble artifact."""
    artifact_path = Path(path)
    try:
        with np.load(artifact_path, allow_pickle=False) as data:
            missing = {
                "maps",
                "rejected_invalid_noise_maps",
                "metadata_json",
            }.difference(data.files)
            if missing:
                raise ValueError(
                    f"{artifact_path} is missing required arrays: {sorted(missing)}"
                )
            maps = np.asarray(data["maps"], dtype=np.float64)
            rejected = np.asarray(
                data["rejected_invalid_noise_maps"], dtype=np.float64
            )
            metadata = _load_metadata(data["metadata_json"])
            optional = {
                name: np.asarray(data[name], dtype=np.float64)
                for name in ("dipole", "angular_power")
                if name in data.files
            }
    except OSError as exc:
        raise ValueError(f"Could not load ensemble artifact {artifact_path}") from exc

    expected_npix = hp.nside2npix(64)
    if maps.ndim != 2 or maps.shape[1] != expected_npix:
        raise ValueError(
            f"{artifact_path}: maps must have shape (n, {expected_npix}) for "
            f"NESTED nside=64, found {maps.shape}"
        )
    if rejected.shape != maps.shape:
        raise ValueError(
            f"{artifact_path}: rejected_invalid_noise_maps must match maps shape"
        )
    if not allow_incomplete_metadata:
        missing_metadata = [name for name in REQUIRED_METADATA if name not in metadata]
        if missing_metadata:
            raise ValueError(
                f"{artifact_path}: metadata_json is missing {missing_metadata}; "
                "artifacts must pin commit, configuration, and seeds"
            )
        config = metadata.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{artifact_path}: metadata config must be an object")
        missing_config = [name for name in REQUIRED_SCIENCE_CONFIG if name not in config]
        if missing_config:
            raise ValueError(
                f"{artifact_path}: metadata config does not pin {missing_config}"
            )
    return {
        "path": artifact_path.resolve(),
        "maps": maps,
        "rejected": rejected,
        "metadata": metadata,
        **optional,
    }


def downgrade_nested_count_maps(
    maps: NDArray[np.floating], target_nside: int
) -> NDArray[np.float64]:
    """Sum NESTED child counts while preserving wholly invalid coarse pixels."""
    values = np.asarray(maps, dtype=np.float64)
    source_nside = int(hp.npix2nside(values.shape[-1]))
    if target_nside > source_nside or source_nside % target_nside:
        raise ValueError("target_nside must divide and not exceed source nside")
    ratio = source_nside // target_nside
    if ratio & (ratio - 1):
        raise ValueError("target_nside must be a power-of-two downgrade")
    child_count = ratio * ratio
    grouped = values.reshape(*values.shape[:-1], hp.nside2npix(target_nside), child_count)
    finite = np.isfinite(grouped)
    output = np.nansum(grouped, axis=-1)
    output[~np.any(finite, axis=-1)] = np.nan
    return output


def ensemble_comparison(
    old_maps: NDArray[np.floating],
    new_maps: NDArray[np.floating],
    *,
    paired: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Return mean, difference, uncertainty, variance, and total-count products."""
    old = np.asarray(old_maps, dtype=np.float64)
    new = np.asarray(new_maps, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        old_mean = np.nanmean(old, axis=0)
        new_mean = np.nanmean(new, axis=0)
        old_variance = np.nanvar(old, axis=0, ddof=1)
        new_variance = np.nanvar(new, axis=0, ddof=1)
    difference = new_mean - old_mean
    old_n = np.count_nonzero(np.isfinite(old), axis=0)
    new_n = np.count_nonzero(np.isfinite(new), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fractional_difference = difference / old_mean
        independent_uncertainty = np.sqrt(
            old_variance / old_n + new_variance / new_n
        )
    if paired:
        if old.shape != new.shape:
            raise ValueError("Paired ensembles must have identical shapes")
        pair_valid = np.isfinite(old) & np.isfinite(new)
        pair_count = np.count_nonzero(pair_valid, axis=0)
        paired_differences = np.where(pair_valid, new - old, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            paired_variance = np.nanvar(paired_differences, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            paired_uncertainty = np.sqrt(paired_variance / pair_count)
        difference_uncertainty = paired_uncertainty
    else:
        paired_uncertainty = np.full_like(independent_uncertainty, np.nan)
        difference_uncertainty = independent_uncertainty
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized_difference = difference / difference_uncertainty
    fractional_difference[~np.isfinite(fractional_difference)] = np.nan
    independent_uncertainty[~np.isfinite(independent_uncertainty)] = np.nan
    paired_uncertainty[~np.isfinite(paired_uncertainty)] = np.nan
    difference_uncertainty[~np.isfinite(difference_uncertainty)] = np.nan
    standardized_difference[~np.isfinite(standardized_difference)] = np.nan
    return {
        "old_mean": old_mean,
        "new_mean": new_mean,
        "difference": difference,
        "fractional_difference": fractional_difference,
        "difference_uncertainty": difference_uncertainty,
        "difference_uncertainty_independent": independent_uncertainty,
        "difference_uncertainty_paired": paired_uncertainty,
        "standardized_difference": standardized_difference,
        "old_variance": old_variance,
        "new_variance": new_variance,
        "variance_difference": new_variance - old_variance,
        "old_total_counts": np.nansum(old, axis=1),
        "new_total_counts": np.nansum(new, axis=1),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _running_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_optional(path: str | None, embedded: Any) -> NDArray[np.float64] | None:
    if path is None:
        return None if embedded is None else np.asarray(embedded, dtype=np.float64)
    source = Path(path)
    if source.suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False), dtype=np.float64)
    with np.load(source, allow_pickle=False) as data:
        if len(data.files) != 1:
            raise ValueError(f"{source} must contain exactly one array")
        return np.asarray(data[data.files[0]], dtype=np.float64)


def _plot_healpix_panels(
    comparison: dict[str, NDArray[np.float64]], path: Path, nside: int
) -> None:
    names = (
        ("old_mean", "legacy ensemble mean"),
        ("new_mean", "new ensemble mean"),
        ("difference", "new - legacy mean"),
        ("fractional_difference", "fractional mean difference"),
        ("difference_uncertainty", "MC uncertainty of difference"),
        ("standardized_difference", "standardized difference"),
        ("old_variance", "legacy variance"),
        ("new_variance", "new variance"),
        ("variance_difference", "new - legacy variance"),
    )
    fig = plt.figure(figsize=(18, 14))
    for index, (name, title) in enumerate(names, start=1):
        hp.mollview(
            comparison[name],
            nest=True,
            fig=fig.number,
            sub=(3, 3, index),
            title=title,
            hold=True,
        )
    fig.suptitle(f"RACS error-model ensemble comparison, nside={nside}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_rejected(
    comparison: dict[str, NDArray[np.float64]], path: Path, nside: int
) -> None:
    fig = plt.figure(figsize=(16, 5))
    for index, (name, title) in enumerate(
        (
            ("old_mean", "legacy mean rejected"),
            ("new_mean", "new mean rejected"),
            ("difference", "new - legacy rejected"),
        ),
        start=1,
    ):
        hp.mollview(
            comparison[name],
            nest=True,
            fig=fig.number,
            sub=(1, 3, index),
            title=title,
            hold=True,
        )
    fig.suptitle(f"Sources rejected for invalid local noise, nside={nside}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_total_counts(
    comparison: dict[str, NDArray[np.float64]], path: Path, title: str
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(comparison["old_total_counts"], bins="auto", alpha=0.6, label="legacy")
    axis.hist(comparison["new_total_counts"], bins="auto", alpha=0.6, label="new")
    axis.set_xlabel("total count per realization")
    axis.set_ylabel("realizations")
    axis.set_title(title)
    axis.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _array_summary(old: NDArray[np.floating], new: NDArray[np.floating]) -> dict[str, Any]:
    return {
        "old_shape": list(old.shape),
        "new_shape": list(new.shape),
        "old_mean": np.nanmean(old, axis=0).tolist(),
        "new_mean": np.nanmean(new, axis=0).tolist(),
        "mean_difference": (np.nanmean(new, axis=0) - np.nanmean(old, axis=0)).tolist(),
    }


def _plot_optional_ensemble(
    old: NDArray[np.floating],
    new: NDArray[np.floating],
    path: Path,
    title: str,
) -> None:
    old_values = np.atleast_2d(np.asarray(old, dtype=np.float64))
    new_values = np.atleast_2d(np.asarray(new, dtype=np.float64))
    if old_values.shape[1:] != new_values.shape[1:]:
        raise ValueError(f"Legacy and new {title} arrays have incompatible shapes")
    old_flat = old_values.reshape(old_values.shape[0], -1)
    new_flat = new_values.reshape(new_values.shape[0], -1)
    coordinate = np.arange(old_flat.shape[1])
    fig, axis = plt.subplots(figsize=(9, 5))
    for values, label in ((old_flat, "legacy"), (new_flat, "new")):
        mean = np.nanmean(values, axis=0)
        finite_n = np.count_nonzero(np.isfinite(values), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            standard_error = np.nanstd(values, axis=0, ddof=1) / np.sqrt(finite_n)
        axis.plot(coordinate, mean, label=label)
        axis.fill_between(
            coordinate,
            mean - standard_error,
            mean + standard_error,
            alpha=0.2,
        )
    axis.set_xlabel("component / multipole index")
    axis.set_ylabel("ensemble mean +/- standard error")
    axis.set_title(title)
    axis.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="Pinned legacy ensemble .npz")
    parser.add_argument("--new", required=True, help="Pinned new ensemble .npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-dipole")
    parser.add_argument("--new-dipole")
    parser.add_argument("--old-power")
    parser.add_argument("--new-power")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--allow-incomplete-metadata",
        action="store_true",
        help="Permit exploratory artifacts without commit/config/seeds metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    old = load_artifact(args.old, allow_incomplete_metadata=args.allow_incomplete_metadata)
    new = load_artifact(args.new, allow_incomplete_metadata=args.allow_incomplete_metadata)
    old_config = old["metadata"].get("config", {})
    new_config = new["metadata"].get("config", {})
    mismatched_config = {
        name: (old_config.get(name), new_config.get(name))
        for name in REQUIRED_SCIENCE_CONFIG
        if name in old_config
        and name in new_config
        and old_config[name] != new_config[name]
    }
    if mismatched_config:
        raise ValueError(
            "Legacy and new artifacts have incompatible science configuration: "
            f"{mismatched_config}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_script_commit": _running_commit(),
        "ordering": "NESTED",
        "old_artifact": {
            "path": str(old["path"]),
            "sha256": _sha256(old["path"]),
            "metadata": old["metadata"],
        },
        "new_artifact": {
            "path": str(new["path"]),
            "sha256": _sha256(new["path"]),
            "metadata": new["metadata"],
        },
        "levels": {},
    }
    # Equal root seeds do not establish pairing when an implementation changes
    # its RNG split/call sequence. Paired uncertainty is enabled only for
    # artifacts which explicitly attest identical pre-noise sources and
    # Gaussian deviates and carry the same per-realization pairing IDs.
    old_pair_ids = old["metadata"].get("paired_realization_ids")
    new_pair_ids = new["metadata"].get("paired_realization_ids")
    paired = bool(
        old["maps"].shape == new["maps"].shape
        and old["metadata"].get("pairing_verified") is True
        and new["metadata"].get("pairing_verified") is True
        and isinstance(old_pair_ids, list)
        and len(old_pair_ids) == old["maps"].shape[0]
        and old_pair_ids == new_pair_ids
    )
    summary["verified_realization_paired_uncertainty"] = paired

    for nside in (64, 4):
        old_maps = old["maps"] if nside == 64 else downgrade_nested_count_maps(old["maps"], nside)
        new_maps = new["maps"] if nside == 64 else downgrade_nested_count_maps(new["maps"], nside)
        old_rejected = old["rejected"] if nside == 64 else downgrade_nested_count_maps(old["rejected"], nside)
        new_rejected = new["rejected"] if nside == 64 else downgrade_nested_count_maps(new["rejected"], nside)
        comparison = ensemble_comparison(old_maps, new_maps, paired=paired)
        rejected_comparison = ensemble_comparison(
            old_rejected, new_rejected, paired=paired
        )
        np.savez_compressed(
            output_dir / f"comparison_nside{nside}.npz",
            **comparison,
            **{f"rejected_{name}": value for name, value in rejected_comparison.items()},
        )
        summary["levels"][str(nside)] = {
            "old_total_count_mean": float(np.mean(comparison["old_total_counts"])),
            "new_total_count_mean": float(np.mean(comparison["new_total_counts"])),
            "old_rejected_total_mean": float(np.mean(rejected_comparison["old_total_counts"])),
            "new_rejected_total_mean": float(np.mean(rejected_comparison["new_total_counts"])),
        }
        if not args.no_plots:
            _plot_healpix_panels(comparison, output_dir / f"counts_nside{nside}.png", nside)
            _plot_rejected(rejected_comparison, output_dir / f"rejected_invalid_noise_nside{nside}.png", nside)
            _plot_total_counts(
                comparison,
                output_dir / f"total_counts_nside{nside}.png",
                f"Total-count distributions, nside={nside}",
            )

    optional_specs = (
        ("dipole", args.old_dipole, args.new_dipole),
        ("angular_power", args.old_power, args.new_power),
    )
    for name, old_path, new_path in optional_specs:
        old_array = _load_optional(old_path, old.get(name))
        new_array = _load_optional(new_path, new.get(name))
        if (old_array is None) != (new_array is None):
            raise ValueError(f"Supply both legacy and new {name} arrays, or neither")
        if old_array is not None and new_array is not None:
            summary[name] = _array_summary(old_array, new_array)
            np.savez_compressed(
                output_dir / f"{name}_comparison.npz",
                old=old_array,
                new=new_array,
                mean_difference=np.nanmean(new_array, axis=0) - np.nanmean(old_array, axis=0),
            )
            if not args.no_plots:
                _plot_optional_ensemble(
                    old_array,
                    new_array,
                    output_dir / f"{name}_comparison.png",
                    name.replace("_", " ").title() + " comparison",
                )

    (output_dir / "comparison_metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
