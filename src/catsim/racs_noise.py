"""Cached RACS noise maps and conditional absolute flux-error lookups.

The runtime representation in this module is deliberately NumPy-only and
compact.  In particular, conditional samples are stored as one stable-sorted
flat array; sparse grid cells are routed to an eligible cell during cache
construction rather than by a runtime nearest-neighbour query.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import healpy as hp
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


NOISE_MAP_CACHE_FORMAT_VERSION = 1
ABSOLUTE_ERROR_LOOKUP_FORMAT_VERSION = 1
NOISE_UNIT = "uJy/beam"
FLUX_UNIT = "mJy"
ERROR_UNIT = "mJy"
NOISE_DOWNSCALING_METHOD = "valid-subpixel arithmetic mean (pess=False, power=0)"
NOISE_INVALID_PIXEL_POLICY = "UNSEEN, non-finite, zero, and negative become NaN"


class RacsCacheValidationError(ValueError):
    """A cache exists, but does not satisfy the requested data contract."""


def _normalise_header(header: list[tuple[Any, ...]]) -> dict[str, Any]:
    return {str(item[0]).upper(): item[1] for item in header if len(item) >= 2}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_checksum(metadata: Mapping[str, Any], *arrays: NDArray[Any]) -> str:
    digest = hashlib.sha256(_canonical_json(metadata).encode("utf-8"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 identity of a source file without loading it twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: str | Path, **arrays: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".npz",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_power_of_two_nside(nside: int, *, name: str) -> None:
    if isinstance(nside, bool) or not isinstance(nside, (int, np.integer)):
        raise ValueError(f"{name} must be a positive power-of-two integer.")
    if nside <= 0 or (int(nside) & (int(nside) - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of two.")


def normalise_noise_values(values: NDArray[np.floating]) -> NDArray[np.float32]:
    """Represent all invalid noise values, including HEALPix UNSEEN, as NaN."""
    noise = np.asarray(values, dtype=np.float32).copy()
    invalid = ~np.isfinite(noise) | (noise <= 0) | np.isclose(noise, hp.UNSEEN)
    noise[invalid] = np.nan
    return noise


def downscale_noise_map(
    source_values: NDArray[np.floating],
    target_nside: int,
    *,
    source_ordering: str = "RING",
) -> NDArray[np.float32]:
    """Average valid source subpixels and return a NESTED float32 map.

    ``healpy.ud_grade(..., pess=False, power=0)`` supplies the required
    arithmetic mean over valid subpixels.  Invalid output pixels are always
    normalised back to NaN.
    """
    _validate_power_of_two_nside(target_nside, name="target_nside")
    source = normalise_noise_values(source_values)
    if source.ndim != 1 or not hp.isnpixok(source.size):
        raise ValueError("Source noise map must be a one-dimensional full-sky HEALPix map.")
    source_nside = int(hp.npix2nside(source.size))
    if target_nside > source_nside:
        raise ValueError(
            f"target_nside ({target_nside}) must not exceed source nside ({source_nside})."
        )
    ordering = source_ordering.upper()
    if ordering not in {"RING", "NESTED", "NEST"}:
        raise ValueError("source_ordering must be RING or NESTED.")
    input_values = np.where(np.isfinite(source), source, hp.UNSEEN)
    output = hp.ud_grade(
        input_values,
        nside_out=target_nside,
        order_in="NESTED" if ordering in {"NESTED", "NEST"} else "RING",
        order_out="NESTED",
        pess=False,
        power=0,
        dtype=np.float64,
    )
    return normalise_noise_values(output)


@dataclass(frozen=True)
class NoiseMapCache:
    values: NDArray[np.float32]
    metadata: dict[str, Any]
    identity: str

    @property
    def nside(self) -> int:
        return int(self.metadata["target_nside"])

    def query(self, ra_deg: NDArray[np.floating], dec_deg: NDArray[np.floating]) -> NDArray[np.float32]:
        ra, dec = np.broadcast_arrays(
            np.asarray(ra_deg, dtype=np.float64),
            np.asarray(dec_deg, dtype=np.float64),
        )
        pixels = hp.ang2pix(self.nside, ra, dec, lonlat=True, nest=True)
        return self.values[pixels]


def build_noise_map_cache(
    source_path: str | Path,
    *,
    product_key: str,
    target_nside: int,
) -> NoiseMapCache:
    """Read and validate a RING/equatorial ``.hpx`` map and downscale it."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"RACS source noisemap does not exist: {source}")
    values, raw_header = hp.read_map(source, field=0, dtype=np.float32, h=True, nest=False)
    header = _normalise_header(raw_header)
    ordering = str(header.get("ORDERING", "")).strip().upper()
    if ordering != "RING":
        raise ValueError(f"RACS source noisemap must use RING ordering, found {ordering!r}.")
    coordinates = str(
        header.get("COORDSYS", header.get("COORD", header.get("COORDTYPE", "")))
    ).strip().upper()
    if not coordinates.startswith("C"):
        raise ValueError(
            "RACS source noisemap must use equatorial/celestial coordinates "
            f"(C), found {coordinates!r}."
        )
    if np.asarray(values).ndim != 1 or not hp.isnpixok(np.asarray(values).size):
        raise ValueError("RACS source noisemap is not a full-sky HEALPix array.")
    source_nside = int(hp.npix2nside(np.asarray(values).size))
    _validate_power_of_two_nside(target_nside, name="noise_map_nside")
    if target_nside > source_nside:
        raise ValueError(
            f"noise_map_nside ({target_nside}) must not exceed source nside ({source_nside})."
        )

    cached_values = downscale_noise_map(values, target_nside, source_ordering="RING")
    stat = source.stat()
    metadata: dict[str, Any] = {
        "format_version": NOISE_MAP_CACHE_FORMAT_VERSION,
        "product_key": str(product_key),
        "source_filename": source.name,
        "source_sha256": file_sha256(source),
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_nside": source_nside,
        "source_ordering": "RING",
        "source_coordinates": "C",
        "unit": NOISE_UNIT,
        "target_nside": int(target_nside),
        "target_ordering": "NESTED",
        "downscaling_method": NOISE_DOWNSCALING_METHOD,
        "invalid_pixel_policy": NOISE_INVALID_PIXEL_POLICY,
        "dtype": "float32",
    }
    identity = _payload_checksum(metadata, cached_values)
    return NoiseMapCache(cached_values, metadata, identity)


def save_noise_map_cache(
    cache: NoiseMapCache,
    path: str | Path,
    *,
    diagnostics: bool = True,
) -> None:
    metadata = dict(cache.metadata)
    identity = _payload_checksum(metadata, cache.values)
    _atomic_savez(
        path,
        metadata_json=np.asarray(_canonical_json(metadata)),
        cache_identity=np.asarray(identity),
        noise_values=np.asarray(cache.values, dtype=np.float32),
    )
    if diagnostics:
        _save_noise_diagnostics(cache, Path(path))


def load_noise_map_cache(
    path: str | Path,
    *,
    product_key: str,
    target_nside: int,
    source_filename: str | None = None,
) -> NoiseMapCache:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"RACS cached noisemap does not exist: {input_path}")
    try:
        with np.load(input_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if not isinstance(metadata, dict):
                raise ValueError("metadata_json must encode an object")
            identity = str(data["cache_identity"])
            values = np.asarray(data["noise_values"], dtype=np.float32)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RacsCacheValidationError(f"Invalid RACS noise-map cache: {input_path}") from exc

    expected = {
        "format_version": NOISE_MAP_CACHE_FORMAT_VERSION,
        "product_key": product_key,
        "source_ordering": "RING",
        "source_coordinates": "C",
        "target_nside": int(target_nside),
        "target_ordering": "NESTED",
        "downscaling_method": NOISE_DOWNSCALING_METHOD,
        "invalid_pixel_policy": NOISE_INVALID_PIXEL_POLICY,
        "unit": NOISE_UNIT,
        "dtype": "float32",
    }
    if source_filename is not None:
        expected["source_filename"] = source_filename
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    expected_shape = (hp.nside2npix(target_nside),)
    if values.shape != expected_shape:
        mismatches.append(f"shape={values.shape!r} (expected {expected_shape!r})")
    source_nside = metadata.get("source_nside")
    if (
        isinstance(source_nside, bool)
        or not isinstance(source_nside, int)
        or source_nside < target_nside
        or source_nside <= 0
        or (source_nside & (source_nside - 1)) != 0
    ):
        mismatches.append(
            f"source_nside={source_nside!r} is not a compatible power-of-two nside"
        )
    invalid_stored = ~np.isnan(values) & (~np.isfinite(values) | (values <= 0))
    if np.any(invalid_stored):
        mismatches.append("noise values must be positive finite values or NaN")
    computed_identity = _payload_checksum(metadata, values)
    if computed_identity != identity:
        mismatches.append("payload checksum does not match")
    if mismatches:
        raise RacsCacheValidationError(
            "Incompatible RACS noise-map cache: " + "; ".join(mismatches)
        )
    return NoiseMapCache(normalise_noise_values(values), metadata, identity)


def _save_noise_diagnostics(cache: NoiseMapCache, cache_path: Path) -> None:
    stem = cache_path.with_suffix("")
    finite = np.isfinite(cache.values) & (cache.values > 0)
    positive = cache.values[finite].astype(np.float64)
    percentiles = (
        np.percentile(positive, [0, 1, 5, 16, 50, 84, 95, 99, 100]).tolist()
        if positive.size
        else [None] * 9
    )
    summary = {
        **cache.metadata,
        "cache_identity": cache.identity,
        "finite_pixels": int(np.count_nonzero(finite)),
        "invalid_pixels": int(np.count_nonzero(~finite)),
        "finite_fraction": float(np.mean(finite)),
        "positive_noise_percentiles_0_1_5_16_50_84_95_99_100": percentiles,
        "positive_noise_percentiles_1_5_16_50_84_95_99": percentiles[1:-1],
    }
    _atomic_json(Path(f"{stem}.summary.json"), summary)

    plot_values = np.where(finite, np.log10(cache.values), np.nan)
    for suffix, values, title, unit, cmap in (
        ("map", plot_values, "log10 local noise", "log10(uJy/beam)", "viridis"),
        ("coverage", finite.astype(np.float32), "finite noise-map coverage", "valid", "gray_r"),
    ):
        fig = plt.figure(figsize=(10, 6))
        hp.mollview(values, nest=True, fig=fig.number, title=f"{cache.metadata['product_key']}: {title}", unit=unit, cmap=cmap)
        fig.savefig(Path(f"{stem}.{suffix}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if positive.size:
        full_low = float(np.min(positive))
        full_high = float(np.max(positive))
        if full_high <= full_low:
            full_high = full_low * (1.0 + 1e-6)
        full_bins = np.geomspace(full_low, full_high, 101)
        axes[0].hist(positive, bins=full_bins)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")

        robust_low, robust_high = np.percentile(positive, [0.5, 99.5])
        if robust_high <= robust_low:
            robust_high = robust_low * (1.0 + 1e-6)
        robust_bins = np.geomspace(robust_low, robust_high, 101)
        robust_values = positive[
            (positive >= robust_low) & (positive <= robust_high)
        ]
        axes[1].hist(robust_values, bins=robust_bins)
        axes[1].set_xscale("log")
        axes[1].set_title(
            f"central 99% ({robust_low:.1f}–{robust_high:.1f} {NOISE_UNIT})"
        )
    axes[0].set_title("full range (log-spaced bins; log counts)")
    for axis in axes:
        axis.set_xlabel(NOISE_UNIT)
        axis.set_ylabel("HEALPix pixels")
    fig.suptitle(f"{cache.metadata['product_key']}: positive local noise")
    fig.savefig(Path(f"{stem}.hist.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _regular_log_edges(values: NDArray[np.float64], bins: int) -> NDArray[np.float64]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        padding = max(abs(lower) * 1e-6, 1e-6)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, bins + 1, dtype=np.float64)


def conditional_cell_ids(
    log_noise: NDArray[np.floating],
    log_flux: NDArray[np.floating],
    log_noise_edges: NDArray[np.floating],
    log_flux_edges: NDArray[np.floating],
) -> NDArray[np.int64]:
    """Return clipped row-major cells (noise bin first, flux bin second)."""
    noise_edges = np.asarray(log_noise_edges, dtype=np.float64)
    flux_edges = np.asarray(log_flux_edges, dtype=np.float64)
    noise_bin = np.searchsorted(noise_edges, log_noise, side="right") - 1
    flux_bin = np.searchsorted(flux_edges, log_flux, side="right") - 1
    noise_bin = np.clip(noise_bin, 0, noise_edges.size - 2)
    flux_bin = np.clip(flux_bin, 0, flux_edges.size - 2)
    return (noise_bin * (flux_edges.size - 1) + flux_bin).astype(np.int64, copy=False)


def _resolve_sparse_cells(
    noise_edges: NDArray[np.float64],
    flux_edges: NDArray[np.float64],
    eligible: NDArray[np.bool_],
) -> NDArray[np.int64]:
    noise_centres = 0.5 * (noise_edges[:-1] + noise_edges[1:])
    flux_centres = 0.5 * (flux_edges[:-1] + flux_edges[1:])
    coordinates = np.stack(
        np.meshgrid(noise_centres, flux_centres, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    eligible_ids = np.flatnonzero(eligible).astype(np.int64)
    resolved = np.arange(eligible.size, dtype=np.int64)
    ineligible_ids = np.flatnonzero(~eligible).astype(np.int64)
    tree = cKDTree(coordinates[eligible_ids])
    if eligible_ids.size == 1:
        resolved[ineligible_ids] = eligible_ids[0]
        return resolved

    distances, neighbours = tree.query(coordinates[ineligible_ids], k=2, workers=-1)
    resolved[ineligible_ids] = eligible_ids[neighbours[:, 0]]

    # cKDTree is deterministic for ordinary cases.  Define the exact-distance
    # tie rule explicitly: choose the lowest flattened eligible cell ID.
    tied = np.isclose(distances[:, 0], distances[:, 1], rtol=0.0, atol=1e-12)
    for query_index in np.flatnonzero(tied):
        coordinate = coordinates[ineligible_ids[query_index]]
        radius = np.nextafter(distances[query_index, 0], np.inf)
        candidates = eligible_ids[tree.query_ball_point(coordinate, radius)]
        squared = np.sum((coordinates[candidates] - coordinate) ** 2, axis=1)
        minimum = np.min(squared)
        nearest = candidates[np.isclose(squared, minimum, rtol=0.0, atol=1e-14)]
        resolved[ineligible_ids[query_index]] = int(np.min(nearest))
    return resolved


@dataclass(frozen=True)
class ConditionalErrorLookup:
    log_noise_edges: NDArray[np.float64]
    log_flux_edges: NDArray[np.float64]
    cell_counts: NDArray[np.int64]
    cell_starts: NDArray[np.int64]
    resolved_cell_ids: NDArray[np.int64]
    absolute_error_values: NDArray[np.float32]
    metadata: dict[str, Any]
    identity: str
    # Build-only diagnostic statistic. It is deliberately excluded from the
    # compact runtime cache and is therefore normally absent after cache load.
    diagnostic_median_fractional_error: NDArray[np.float64] | None = None

    def resolve_cells(
        self,
        noise_ujy_beam: NDArray[np.floating],
        flux_mjy: NDArray[np.floating],
    ) -> NDArray[np.int64]:
        noise, flux = np.broadcast_arrays(
            np.asarray(noise_ujy_beam, dtype=np.float64),
            np.asarray(flux_mjy, dtype=np.float64),
        )
        raw = conditional_cell_ids(
            np.log10(noise),
            np.log10(flux),
            self.log_noise_edges,
            self.log_flux_edges,
        )
        return self.resolved_cell_ids[raw]

    def sample(
        self,
        noise_ujy_beam: NDArray[np.floating],
        flux_mjy: NDArray[np.floating],
        *,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float32]:
        """Vectorized empirical absolute-error sampling; invalid queries return NaN."""
        if rng is None:
            rng = np.random.default_rng()
        noise, flux = np.broadcast_arrays(
            np.asarray(noise_ujy_beam, dtype=np.float64),
            np.asarray(flux_mjy, dtype=np.float64),
        )
        output = np.full(noise.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(noise) & (noise > 0) & np.isfinite(flux) & (flux > 0)
        if not np.any(valid):
            return output
        cells = self.resolve_cells(noise[valid], flux[valid])
        counts = self.cell_counts[cells]
        if np.any(counts <= 0):
            raise RuntimeError("Resolved conditional-error cell has no samples.")
        offsets = rng.integers(0, counts, dtype=np.int64)
        output[valid] = self.absolute_error_values[self.cell_starts[cells] + offsets]
        return output


def build_conditional_error_lookup(
    noise_ujy_beam: NDArray[np.floating],
    flux_mjy: NDArray[np.floating],
    absolute_error_mjy: NDArray[np.floating],
    *,
    product_key: str,
    noise_map_identity: str,
    noise_bins: int,
    flux_bins: int,
    min_cell_count: int,
    catalogue_columns: Mapping[str, str] | None = None,
) -> ConditionalErrorLookup:
    """Build the compact empirical noise/flux -> absolute-error distribution."""
    if noise_bins < 2 or flux_bins < 2:
        raise ValueError("noise_bins and flux_bins must each be at least 2.")
    if min_cell_count < 1:
        raise ValueError("min_cell_count must be at least 1.")
    noise, flux, error = np.broadcast_arrays(
        np.asarray(noise_ujy_beam, dtype=np.float64),
        np.asarray(flux_mjy, dtype=np.float64),
        np.asarray(absolute_error_mjy, dtype=np.float64),
    )
    valid_noise = np.isfinite(noise) & (noise > 0)
    valid_flux = np.isfinite(flux) & (flux > 0)
    valid_error = np.isfinite(error) & (error > 0)
    valid = valid_noise & valid_flux & valid_error
    if not np.any(valid):
        raise ValueError("No finite positive rows are available for the absolute-error lookup.")

    log_noise = np.log10(noise[valid])
    log_flux = np.log10(flux[valid])
    noise_edges = _regular_log_edges(log_noise, noise_bins)
    flux_edges = _regular_log_edges(log_flux, flux_bins)
    cell_ids = conditional_cell_ids(log_noise, log_flux, noise_edges, flux_edges)
    order = np.argsort(cell_ids, kind="stable")
    sorted_cells = cell_ids[order]
    sorted_errors = error[valid][order].astype(np.float32, copy=False)
    sorted_fractional_errors = (error[valid][order] / flux[valid][order]).astype(
        np.float64,
        copy=False,
    )
    n_cells = noise_bins * flux_bins
    counts = np.bincount(sorted_cells, minlength=n_cells).astype(np.int64, copy=False)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    eligible = counts >= min_cell_count
    if not np.any(eligible):
        raise ValueError(
            "No conditional error-grid cell reaches "
            f"flux_error_min_cell_count={min_cell_count}."
        )
    resolved = _resolve_sparse_cells(noise_edges, flux_edges, eligible)
    median_fractional_error = np.full(n_cells, np.nan, dtype=np.float64)
    for cell in np.flatnonzero(counts):
        start = starts[cell]
        stop = start + counts[cell]
        median_fractional_error[cell] = np.median(
            sorted_fractional_errors[start:stop]
        )
    metadata: dict[str, Any] = {
        "format_version": ABSOLUTE_ERROR_LOOKUP_FORMAT_VERSION,
        "product_key": str(product_key),
        "noise_map_identity": str(noise_map_identity),
        "noise_unit": NOISE_UNIT,
        "flux_unit": FLUX_UNIT,
        "absolute_error_unit": ERROR_UNIT,
        "noise_bins": int(noise_bins),
        "flux_bins": int(flux_bins),
        "min_cell_count": int(min_cell_count),
        "flattening_convention": "cell = noise_bin * flux_bins + flux_bin",
        "edge_coordinate_dtype": "float64",
        "value_dtype": "float32",
        "catalogue_columns": dict(catalogue_columns or {}),
        "training_rows_total": int(noise.size),
        "training_rows_accepted": int(np.count_nonzero(valid)),
        "training_rows_rejected": int(np.count_nonzero(~valid)),
        "rows_invalid_noise": int(np.count_nonzero(~valid_noise)),
        "rows_invalid_flux": int(np.count_nonzero(~valid_flux)),
        "rows_invalid_absolute_error": int(np.count_nonzero(~valid_error)),
        # These first-failure counts are disjoint and therefore sum to the
        # total rejected count. The invalid-* counts above intentionally
        # retain the useful, possibly overlapping, per-field diagnostics.
        "rows_rejected_first_reason_noise": int(np.count_nonzero(~valid_noise)),
        "rows_rejected_first_reason_flux": int(
            np.count_nonzero(valid_noise & ~valid_flux)
        ),
        "rows_rejected_first_reason_absolute_error": int(
            np.count_nonzero(valid_noise & valid_flux & ~valid_error)
        ),
        "noise_ujy_beam_percentiles_0_1_5_16_50_84_95_99_100": np.percentile(
            noise[valid], [0, 1, 5, 16, 50, 84, 95, 99, 100]
        ).tolist(),
        "flux_mjy_percentiles_0_1_5_16_50_84_95_99_100": np.percentile(
            flux[valid], [0, 1, 5, 16, 50, 84, 95, 99, 100]
        ).tolist(),
        "absolute_error_mjy_percentiles_0_1_5_16_50_84_95_99_100": np.percentile(
            error[valid], [0, 1, 5, 16, 50, 84, 95, 99, 100]
        ).tolist(),
    }
    identity = _payload_checksum(
        metadata,
        noise_edges,
        flux_edges,
        counts,
        starts,
        resolved,
        sorted_errors,
    )
    return ConditionalErrorLookup(
        noise_edges,
        flux_edges,
        counts,
        starts,
        resolved,
        sorted_errors,
        metadata,
        identity,
        median_fractional_error,
    )


def save_conditional_error_lookup(
    lookup: ConditionalErrorLookup,
    path: str | Path,
    *,
    diagnostics: bool = True,
) -> None:
    arrays = (
        lookup.log_noise_edges,
        lookup.log_flux_edges,
        lookup.cell_counts,
        lookup.cell_starts,
        lookup.resolved_cell_ids,
        lookup.absolute_error_values,
    )
    identity = _payload_checksum(lookup.metadata, *arrays)
    _atomic_savez(
        path,
        metadata_json=np.asarray(_canonical_json(lookup.metadata)),
        cache_identity=np.asarray(identity),
        log_noise_edges=lookup.log_noise_edges,
        log_flux_edges=lookup.log_flux_edges,
        cell_counts=lookup.cell_counts,
        cell_starts=lookup.cell_starts,
        resolved_cell_ids=lookup.resolved_cell_ids,
        absolute_error_values=lookup.absolute_error_values,
    )
    if diagnostics:
        _save_grid_diagnostics(lookup, Path(path))


def load_conditional_error_lookup(
    path: str | Path,
    *,
    product_key: str,
    noise_map_identity: str,
    noise_bins: int,
    flux_bins: int,
    min_cell_count: int,
    catalogue_columns: Mapping[str, str] | None = None,
) -> ConditionalErrorLookup:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"RACS conditional-error cache does not exist: {input_path}")
    try:
        with np.load(input_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if not isinstance(metadata, dict):
                raise ValueError("metadata_json must encode an object")
            identity = str(data["cache_identity"])
            arrays = [
                np.asarray(data["log_noise_edges"], dtype=np.float64),
                np.asarray(data["log_flux_edges"], dtype=np.float64),
                np.asarray(data["cell_counts"], dtype=np.int64),
                np.asarray(data["cell_starts"], dtype=np.int64),
                np.asarray(data["resolved_cell_ids"], dtype=np.int64),
                np.asarray(data["absolute_error_values"], dtype=np.float32),
            ]
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RacsCacheValidationError(
            f"Invalid RACS conditional absolute-error cache: {input_path}"
        ) from exc
    expected = {
        "format_version": ABSOLUTE_ERROR_LOOKUP_FORMAT_VERSION,
        "product_key": product_key,
        "noise_map_identity": noise_map_identity,
        "noise_bins": int(noise_bins),
        "flux_bins": int(flux_bins),
        "min_cell_count": int(min_cell_count),
        "noise_unit": NOISE_UNIT,
        "flux_unit": FLUX_UNIT,
        "absolute_error_unit": ERROR_UNIT,
        "flattening_convention": "cell = noise_bin * flux_bins + flux_bin",
        "edge_coordinate_dtype": "float64",
        "value_dtype": "float32",
    }
    if catalogue_columns is not None:
        expected["catalogue_columns"] = dict(catalogue_columns)
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    n_cells = noise_bins * flux_bins
    if arrays[0].shape != (noise_bins + 1,):
        mismatches.append("log-noise edge shape does not match configured bins")
    if arrays[1].shape != (flux_bins + 1,):
        mismatches.append("log-flux edge shape does not match configured bins")
    if any(array.shape != (n_cells,) for array in arrays[2:5]):
        mismatches.append("cell array shape does not match configured grid")
    if _payload_checksum(metadata, *arrays) != identity:
        mismatches.append("payload checksum does not match")
    counts, starts, resolved, values = arrays[2], arrays[3], arrays[4], arrays[5]
    for name, edges in (
        ("log-noise", arrays[0]),
        ("log-flux", arrays[1]),
    ):
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            mismatches.append(f"{name} edges must be finite and strictly increasing")
    if np.any(counts < 0) or int(np.sum(counts)) != values.size:
        mismatches.append("cell counts are inconsistent with flat values")
    if not np.array_equal(starts, np.cumsum(counts, dtype=np.int64) - counts):
        mismatches.append("cell starts are inconsistent with cell counts")
    if np.any((resolved < 0) | (resolved >= n_cells)) or np.any(counts[resolved] < min_cell_count):
        mismatches.append("resolved cell IDs do not all reference eligible cells")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        mismatches.append("absolute-error samples must be finite and positive")
    if mismatches:
        raise RacsCacheValidationError(
            "Incompatible RACS conditional absolute-error cache: " + "; ".join(mismatches)
        )
    return ConditionalErrorLookup(*arrays, metadata, identity)


def _save_grid_diagnostics(lookup: ConditionalErrorLookup, cache_path: Path) -> None:
    noise_bins = lookup.log_noise_edges.size - 1
    flux_bins = lookup.log_flux_edges.size - 1
    shape = (noise_bins, flux_bins)
    counts = lookup.cell_counts.reshape(shape)
    eligible = counts >= int(lookup.metadata["min_cell_count"])
    median_error = np.full(counts.size, np.nan)
    q16_error = np.full(counts.size, np.nan)
    q84_error = np.full(counts.size, np.nan)
    median_fractional = (
        np.asarray(lookup.diagnostic_median_fractional_error, dtype=np.float64)
        if lookup.diagnostic_median_fractional_error is not None
        else np.full(counts.size, np.nan, dtype=np.float64)
    )
    for cell in np.flatnonzero(lookup.cell_counts):
        start = lookup.cell_starts[cell]
        stop = start + lookup.cell_counts[cell]
        values = lookup.absolute_error_values[start:stop].astype(np.float64)
        q16_error[cell], median_error[cell], q84_error[cell] = np.percentile(values, [16, 50, 84])
    noise_centres = 0.5 * (lookup.log_noise_edges[:-1] + lookup.log_noise_edges[1:])
    log_flux_centres = 0.5 * (lookup.log_flux_edges[:-1] + lookup.log_flux_edges[1:])
    coordinates = np.stack(np.meshgrid(noise_centres, log_flux_centres, indexing="ij"), axis=-1).reshape(-1, 2)
    fallback_distance = np.linalg.norm(
        coordinates - coordinates[lookup.resolved_cell_ids], axis=1
    ).reshape(shape)

    def logarithmic_panel(values: NDArray[np.floating]) -> tuple[NDArray, LogNorm]:
        plotted = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        positive = plotted[np.isfinite(plotted)]
        if positive.size == 0:
            # These panels are expected to contain positive data, but retaining
            # a harmless unit interval makes diagnostic failures intelligible.
            return plotted, LogNorm(vmin=1.0, vmax=10.0)
        lower = float(np.min(positive))
        upper = float(np.max(positive))
        if upper <= lower:
            upper = lower * (1.0 + 1e-6)
        return plotted, LogNorm(vmin=lower, vmax=upper)

    occupancy_values, occupancy_norm = logarithmic_panel(counts.astype(np.float64))
    median_error_values, median_error_norm = logarithmic_panel(
        median_error.reshape(shape)
    )
    spread_values, spread_norm = logarithmic_panel(
        (q84_error - q16_error).reshape(shape)
    )
    fractional_values, fractional_norm = logarithmic_panel(
        median_fractional.reshape(shape)
    )
    panels = (
        (occupancy_values, "raw occupancy (log colour)", occupancy_norm),
        (eligible.astype(float), "eligible cell", None),
        (median_error_values, "median absolute error (mJy; log colour)", median_error_norm),
        (spread_values, "84th - 16th error percentile (mJy; log colour)", spread_norm),
        (fractional_values, "median derived fractional error (log colour)", fractional_norm),
        (fallback_distance, "fallback distance (dex)", None),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (values, title, norm) in zip(axes.flat, panels):
        image = axis.pcolormesh(
            lookup.log_flux_edges,
            lookup.log_noise_edges,
            values,
            shading="auto",
            norm=norm,
        )
        axis.set_xlabel("log10(total flux / mJy)")
        axis.set_ylabel("log10(noise / uJy beam$^{-1}$)")
        axis.set_title(title)
        fig.colorbar(image, ax=axis)
    stem = cache_path.with_suffix("")
    fig.suptitle(f"{lookup.metadata['product_key']} conditional absolute-error lookup")
    fig.savefig(Path(f"{stem}.diagnostics.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Cell occupancies are sufficient to recover the exact grid-bin
    # marginals without retaining per-row conditioning coordinates in the
    # runtime cache. Plot every bin so the full training range and tail cells
    # remain visible; robust ranges are recorded in the sidecar below.
    noise_marginal = np.sum(counts, axis=1)
    flux_marginal = np.sum(counts, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].stairs(noise_marginal, lookup.log_noise_edges, fill=True)
    axes[0].set_xlabel("log10(noise / uJy beam$^{-1}$)")
    axes[0].set_ylabel("accepted catalogue rows")
    axes[0].set_title("full-range noise marginal")
    axes[1].stairs(flux_marginal, lookup.log_flux_edges, fill=True)
    axes[1].set_xlabel("log10(total flux / mJy)")
    axes[1].set_ylabel("accepted catalogue rows")
    axes[1].set_title("full-range flux marginal")
    fig.suptitle(
        f"{lookup.metadata['product_key']} lookup training marginals "
        "(end bins include all outliers)"
    )
    fig.savefig(Path(f"{stem}.marginals.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    occupied = lookup.cell_counts > 0
    accepted_in_eligible = int(np.sum(lookup.cell_counts[eligible.reshape(-1)]))
    distances = fallback_distance.reshape(-1)
    summary = {
        **lookup.metadata,
        "cache_identity": lookup.identity,
        "nonempty_cells": int(np.count_nonzero(occupied)),
        "nonempty_cell_fraction": float(np.mean(occupied)),
        "eligible_cells": int(np.count_nonzero(eligible)),
        "eligible_cell_fraction": float(np.mean(eligible)),
        "accepted_rows_in_eligible_cells": accepted_in_eligible,
        "accepted_rows_in_eligible_cell_fraction": float(
            accepted_in_eligible / lookup.absolute_error_values.size
        ),
        "fallback_distance_dex_percentiles_50_84_95_99_max": np.percentile(
            distances, [50, 84, 95, 99, 100]
        ).tolist(),
        "log_noise_full_range": [float(lookup.log_noise_edges[0]), float(lookup.log_noise_edges[-1])],
        "log_flux_full_range": [float(lookup.log_flux_edges[0]), float(lookup.log_flux_edges[-1])],
        "log_noise_range": [float(lookup.log_noise_edges[0]), float(lookup.log_noise_edges[-1])],
        "log_flux_range": [float(lookup.log_flux_edges[0]), float(lookup.log_flux_edges[-1])],
        "absolute_error_mjy_percentiles_0_1_5_16_50_84_95_99_100": np.percentile(
            lookup.absolute_error_values, [0, 1, 5, 16, 50, 84, 95, 99, 100]
        ).tolist(),
        "absolute_error_percentiles_1_16_50_84_99": np.percentile(
            lookup.absolute_error_values, [1, 16, 50, 84, 99]
        ).tolist(),
        "noise_marginal_counts_by_grid_bin": noise_marginal.tolist(),
        "flux_marginal_counts_by_grid_bin": flux_marginal.tolist(),
    }
    _atomic_json(Path(f"{stem}.summary.json"), summary)
