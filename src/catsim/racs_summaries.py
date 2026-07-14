from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def validate_bin_edges(
    values: Sequence[float] | NDArray[np.floating],
    name: str = "bin_edges",
) -> NDArray[np.float64]:
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values.")
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return edges


def validate_quantiles(
    values: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    quantiles = np.asarray(values, dtype=np.float64)
    if quantiles.ndim != 1 or quantiles.size == 0:
        raise ValueError("quantiles must be a non-empty one-dimensional array.")
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("quantiles must lie in [0, 1].")
    return quantiles


def empirical_bin_edges(
    values: Sequence[float] | NDArray[np.floating],
    n_bins: int,
    *,
    value_name: str,
) -> NDArray[np.float64]:
    if n_bins < 1:
        raise ValueError(f"{value_name} summary requires at least one bin.")
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(f"{value_name} summary found no finite values.")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if not lower < upper:
        raise ValueError(f"{value_name} summary requires a non-zero range.")
    return np.linspace(lower, upper, n_bins + 1, dtype=np.float64)


def binned_flux_quantiles_exact(
    observed_flux: Sequence[float] | NDArray[np.floating],
    bin_values: Sequence[float] | NDArray[np.floating],
    *,
    bin_edges: Sequence[float] | NDArray[np.floating],
    quantiles: Sequence[float] | NDArray[np.floating],
    value_name: str = "value",
) -> NDArray[np.float32]:
    flux = np.asarray(observed_flux, dtype=np.float64)
    values = np.asarray(bin_values, dtype=np.float64)
    edges = validate_bin_edges(bin_edges)
    quantile_values = validate_quantiles(quantiles)
    if flux.shape != values.shape:
        raise ValueError("observed_flux and bin_values must have matching shapes.")

    valid = np.isfinite(flux) & np.isfinite(values)
    flux = flux[valid]
    values = values[valid]
    if flux.size == 0:
        raise ValueError("Flux summary has no finite flux/bin-value pairs.")

    features: list[float] = []
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index == edges.size - 2:
            in_bin = (values >= lower) & (values <= upper)
        else:
            in_bin = (values >= lower) & (values < upper)
        if not np.any(in_bin):
            closing = "]" if bin_index == edges.size - 2 else ")"
            raise ValueError(
                f"Flux summary has an empty {value_name} bin "
                f"[{lower:.6g}, {upper:.6g}{closing}."
            )
        features.extend(np.quantile(flux[in_bin], quantile_values))
    return np.asarray(features, dtype=np.float32)


def binned_flux_quantiles_histogram(
    observed_flux: Sequence[float] | NDArray[np.floating],
    bin_values: Sequence[float] | NDArray[np.floating],
    *,
    bin_edges: Sequence[float] | NDArray[np.floating],
    quantiles: Sequence[float] | NDArray[np.floating],
    flux_min_mjy: float,
    flux_max_mjy: float,
    n_flux_bins: int = 128,
    empty_value: float = 0.0,
) -> NDArray[np.float32]:
    flux = np.asarray(observed_flux, dtype=np.float64)
    values = np.asarray(bin_values, dtype=np.float64)
    edges = validate_bin_edges(bin_edges)
    quantile_values = validate_quantiles(quantiles)
    if flux.shape != values.shape:
        raise ValueError("observed_flux and bin_values must have matching shapes.")
    if not np.isfinite(flux_min_mjy) or flux_min_mjy <= 0:
        raise ValueError("flux_min_mjy must be positive and finite.")
    if not np.isfinite(flux_max_mjy) or flux_max_mjy <= flux_min_mjy:
        raise ValueError("flux_max_mjy must be finite and greater than flux_min_mjy.")
    if n_flux_bins < 1:
        raise ValueError("n_flux_bins must be at least 1.")

    valid = (
        np.isfinite(flux)
        & np.isfinite(values)
        & (flux >= flux_min_mjy)
    )
    flux = flux[valid]
    values = values[valid]
    if flux.size == 0:
        raise ValueError("Flux summary has no finite flux/bin-value pairs.")

    z_max = np.log10(flux_max_mjy / flux_min_mjy)
    z = np.log10(flux / flux_min_mjy)
    z = np.clip(z, 0.0, np.nextafter(z_max, 0.0))
    z_edges = np.linspace(0.0, z_max, n_flux_bins + 1, dtype=np.float64)
    histogram = np.zeros((edges.size - 1, n_flux_bins), dtype=np.float64)
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index == edges.size - 2:
            in_bin = (values >= lower) & (values <= upper)
        else:
            in_bin = (values >= lower) & (values < upper)
        if np.any(in_bin):
            flux_bins = np.searchsorted(z_edges, z[in_bin], side="right") - 1
            histogram[bin_index] = np.bincount(
                np.clip(flux_bins, 0, n_flux_bins - 1),
                minlength=n_flux_bins,
            )

    features: list[float] = []
    for row in histogram:
        total = float(np.sum(row))
        if total <= 0:
            features.extend([empty_value] * quantile_values.size)
            continue
        cumulative = np.cumsum(row)
        for quantile in quantile_values:
            target = np.finfo(np.float64).eps if quantile <= 0 else quantile * total
            flux_bin = int(np.searchsorted(cumulative, target, side="left"))
            flux_bin = int(np.clip(flux_bin, 0, n_flux_bins - 1))
            previous = 0.0 if flux_bin == 0 else float(cumulative[flux_bin - 1])
            current = float(cumulative[flux_bin])
            fraction = np.clip(
                (target - previous) / max(current - previous, np.finfo(np.float64).eps),
                0.0,
                1.0,
            )
            z_quantile = z_edges[flux_bin] + fraction * (
                z_edges[flux_bin + 1] - z_edges[flux_bin]
            )
            features.append(float(flux_min_mjy * np.power(10.0, z_quantile)))
    return np.asarray(features, dtype=np.float32)


def binned_flux_quantile_ndim(n_value_bins: int, quantiles: Sequence[float]) -> int:
    if n_value_bins < 1:
        raise ValueError("n_value_bins must be at least 1.")
    return n_value_bins * validate_quantiles(quantiles).size
