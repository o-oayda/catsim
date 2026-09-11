from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys
import types

import healpy as hp
import numpy as np
import pytest


if "dipoleutils.utils.data_loader" not in sys.modules:
    dipoleutils_module = types.ModuleType("dipoleutils")
    utils_module = types.ModuleType("dipoleutils.utils")
    data_loader_module = types.ModuleType("dipoleutils.utils.data_loader")

    class _TestDataLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            raise RuntimeError("catalogue access is forbidden in this test")

    data_loader_module.DataLoader = _TestDataLoader
    sys.modules["dipoleutils"] = dipoleutils_module
    sys.modules["dipoleutils.utils"] = utils_module
    sys.modules["dipoleutils.utils.data_loader"] = data_loader_module


from catsim import RACS_LOW3, RACS_MID1, Racs, RacsConfig
from catsim.racs_products import RACS_LOW2, RACS_LOW2_25AS, RACS_LOW2_45AS
from catsim.racs_noise import (
    ABSOLUTE_ERROR_LOOKUP_FORMAT_VERSION,
    BOUNDED_ABSOLUTE_ERROR_LOOKUP_FORMAT_VERSION,
    NOISE_MAP_CACHE_FORMAT_VERSION,
    RacsCacheValidationError,
    build_conditional_error_lookup,
    build_noise_map_cache,
    load_conditional_error_lookup,
    load_noise_map_cache,
    save_conditional_error_lookup,
    save_noise_map_cache,
)


def _write_ring_equatorial_noise_map(path: Path) -> np.ndarray:
    nside = 4
    nested = np.full(hp.nside2npix(nside), 8.0, dtype=np.float32)
    nested[:4] = np.asarray([1.0, 3.0, hp.UNSEEN, -2.0], dtype=np.float32)
    ring = hp.reorder(nested, n2r=True)
    hp.write_map(path, ring, nest=False, coord="C", dtype=np.float32, overwrite=True)
    return nested


def _lookup(*, min_cell_count: int = 2):
    # The ranges deliberately give noise bins width 1.5 dex and flux bins
    # width 0.5 dex, making nearest-cell routes unambiguous in raw dex space.
    return build_conditional_error_lookup(
        np.asarray([1.0, 1.0, 1000.0, 1000.0]),
        np.asarray([1.0, 1.0, 10.0, 10.0]),
        np.asarray([1.25, 1.5, 7.0, 8.0]),
        product_key="low3",
        noise_map_identity="synthetic-noise",
        noise_bins=2,
        flux_bins=2,
        min_cell_count=min_cell_count,
        catalogue_columns={"total_flux": "Total_flux", "total_flux_error": "E_Total_flux"},
    )


def test_product_noisemap_contract_and_config_validation():
    assert RACS_LOW3.source_noisemap_filename == "RACS-low3.iqr.hpx"
    assert RACS_MID1.source_noisemap_filename == "RACS-mid1.iqr.hpx"
    assert RACS_LOW2.source_noisemap_filename == "RACS-low2.iqr.hpx"
    assert RACS_LOW2_25AS.source_noisemap_filename == "RACS-low2.iqr.hpx"
    assert RACS_LOW2_45AS.source_noisemap_filename == "RACS-low2.iqr.hpx"
    low3_cfg = RacsConfig(product=RACS_LOW3, flux_min=1.0)
    assert low3_cfg.flux_error_noise_bins == 200
    assert low3_cfg.flux_error_flux_bins == 300
    np.testing.assert_allclose(
        low3_cfg.flux_error_noise_bounds_ujy_beam,
        (10.0**1.9, 1000.0),
    )
    assert low3_cfg.flux_error_flux_bounds_mjy == (0.1, 10_000.0)
    mid1_cfg = RacsConfig(product=RACS_MID1, flux_min=1.0)
    assert mid1_cfg.flux_error_noise_bins == 200
    assert mid1_cfg.flux_error_flux_bins == 300
    assert mid1_cfg.flux_error_noise_bounds_ujy_beam == (100.0, 1000.0)
    assert mid1_cfg.flux_error_flux_bounds_mjy == (0.1, 10_000.0)
    low2_cfg = RacsConfig(product=RACS_LOW2, flux_min=1.0)
    assert low2_cfg.flux_error_noise_bins == 200
    assert low2_cfg.flux_error_flux_bins == 300
    np.testing.assert_allclose(
        low2_cfg.flux_error_noise_bounds_ujy_beam,
        (10.0**1.9, 1000.0),
    )
    assert low2_cfg.flux_error_flux_bounds_mjy == (0.1, 10_000.0)
    for product in (RACS_LOW2_25AS, RACS_LOW2_45AS):
        cfg = RacsConfig(product=product, flux_min=1.0)
        assert cfg.flux_error_noise_bins == 200
        assert cfg.flux_error_flux_bins == 300
        np.testing.assert_allclose(
            cfg.flux_error_noise_bounds_ujy_beam,
            (10.0**1.9, 1000.0),
        )
        assert cfg.flux_error_flux_bounds_mjy == (0.1, 10_000.0)
    override = RacsConfig(
        product=RACS_MID1,
        flux_min=1.0,
        flux_error_noise_bins=20,
        flux_error_flux_bins=30,
        flux_error_noise_bounds_ujy_beam=(90.0, 900.0),
        flux_error_flux_bounds_mjy=None,
    )
    assert override.flux_error_noise_bins == 20
    assert override.flux_error_flux_bins == 30
    assert override.flux_error_noise_bounds_ujy_beam == (90.0, 900.0)
    assert override.flux_error_flux_bounds_mjy is None
    with pytest.raises(ValueError, match="positive power of two"):
        RacsConfig(flux_min=1.0, noise_map_nside=3)
    with pytest.raises(ValueError, match="noise_bins must be at least 2"):
        RacsConfig(flux_min=1.0, flux_error_noise_bins=1)
    with pytest.raises(ValueError, match="flux_bins must be at least 2"):
        RacsConfig(flux_min=1.0, flux_error_flux_bins=1)
    with pytest.raises(ValueError, match="min_cell_count must be at least 1"):
        RacsConfig(flux_min=1.0, flux_error_min_cell_count=0)
    for field_name in (
        "flux_error_noise_bins",
        "flux_error_flux_bins",
        "flux_error_min_cell_count",
    ):
        with pytest.raises(ValueError, match=f"{field_name} must be an integer"):
            RacsConfig(flux_min=1.0, **{field_name: 2.5})
        with pytest.raises(ValueError, match=f"{field_name} must be an integer"):
            RacsConfig(flux_min=1.0, **{field_name: True})
    with pytest.raises(TypeError):
        RacsConfig(flux_min=1.0, fractional_error_flux_min_mjy=10.0)
    for field_name in (
        "flux_error_noise_bounds_ujy_beam",
        "flux_error_flux_bounds_mjy",
    ):
        for bad_bounds in ((0.0, 1.0), (2.0, 1.0), (1.0, np.inf), (1.0,)):
            with pytest.raises(ValueError, match=field_name):
                RacsConfig(flux_min=1.0, **{field_name: bad_bounds})


def test_noise_map_build_uses_valid_subpixel_mean_and_nested_cache(tmp_path: Path):
    source = tmp_path / "RACS-low3.iqr.hpx"
    _write_ring_equatorial_noise_map(source)

    cache = build_noise_map_cache(source, product_key="low3", target_nside=2)

    assert cache.values.dtype == np.float32
    assert cache.values.shape == (hp.nside2npix(2),)
    assert cache.values[0] == pytest.approx(2.0)
    assert np.all(cache.values[1:] == np.float32(8.0))
    assert cache.metadata["source_ordering"] == "RING"
    assert cache.metadata["source_coordinates"] == "C"
    assert cache.metadata["target_ordering"] == "NESTED"
    assert cache.metadata["unit"] == "uJy/beam"
    assert cache.metadata["format_version"] == NOISE_MAP_CACHE_FORMAT_VERSION

    output = tmp_path / "noise-cache.npz"
    save_noise_map_cache(cache, output, diagnostics=True)
    restored = load_noise_map_cache(
        output,
        product_key="low3",
        target_nside=2,
        source_filename=source.name,
    )
    np.testing.assert_array_equal(restored.values, cache.values)
    assert restored.identity == cache.identity
    for suffix in ("map.png", "coverage.png", "hist.png", "summary.json"):
        assert (tmp_path / f"noise-cache.{suffix}").is_file()


def test_noise_map_cache_mismatch_is_rejected_without_source_access(tmp_path: Path):
    source = tmp_path / "RACS-low3.iqr.hpx"
    _write_ring_equatorial_noise_map(source)
    cache = build_noise_map_cache(source, product_key="low3", target_nside=2)
    output = tmp_path / "noise.npz"
    save_noise_map_cache(cache, output, diagnostics=False)
    source.unlink()

    # A valid cache has no runtime dependency on the deleted source map.
    restored = load_noise_map_cache(output, product_key="low3", target_nside=2)
    assert restored.identity == cache.identity
    with pytest.raises(RacsCacheValidationError, match="product_key"):
        load_noise_map_cache(output, product_key="mid1", target_nside=2)
    with pytest.raises(RacsCacheValidationError, match="target_nside"):
        load_noise_map_cache(output, product_key="low3", target_nside=1)

    incompatible = replace(
        cache,
        metadata={**cache.metadata, "source_ordering": "NESTED"},
    )
    save_noise_map_cache(incompatible, output, diagnostics=False)
    with pytest.raises(RacsCacheValidationError, match="source_ordering"):
        load_noise_map_cache(output, product_key="low3", target_nside=2)

    np.savez_compressed(
        output,
        metadata_json=np.asarray("[]"),
        cache_identity=np.asarray("irrelevant"),
        noise_values=cache.values,
    )
    with pytest.raises(RacsCacheValidationError, match="Invalid RACS noise-map cache"):
        load_noise_map_cache(output, product_key="low3", target_nside=2)


def test_conditional_grid_stores_absolute_errors_and_routes_in_dex():
    lookup = _lookup()

    # Stable sorting preserves row order within both eligible cells.
    np.testing.assert_array_equal(
        lookup.absolute_error_values,
        np.asarray([1.25, 1.5, 7.0, 8.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(lookup.cell_counts, [2, 0, 0, 2])
    # Cell 1 is closer to cell 0; cell 2 is closer to cell 3 in unscaled dex.
    np.testing.assert_array_equal(lookup.resolved_cell_ids, [0, 0, 3, 3])
    assert lookup.metadata["training_rows_accepted"] == 4
    assert lookup.metadata["absolute_error_unit"] == "mJy"
    assert lookup.metadata["format_version"] == ABSOLUTE_ERROR_LOOKUP_FORMAT_VERSION
    assert lookup.metadata[
        "noise_ujy_beam_percentiles_0_1_5_16_50_84_95_99_100"
    ][0] == pytest.approx(1.0)
    assert lookup.metadata[
        "noise_ujy_beam_percentiles_0_1_5_16_50_84_95_99_100"
    ][-1] == pytest.approx(1000.0)
    np.testing.assert_allclose(
        lookup.diagnostic_median_fractional_error[[0, 3]],
        np.asarray([1.375, 0.75]),
    )

    # Exact lower/upper edges and out-of-range inputs clip to boundary cells.
    cells = lookup.resolve_cells(
        np.asarray([1.0, 1000.0, 1e-20, 1e20]),
        np.asarray([1.0, 10.0, 1e-20, 1e20]),
    )
    np.testing.assert_array_equal(cells, [0, 3, 0, 3])

    samples = lookup.sample(
        np.asarray([2.0] * 100 + [500.0] * 100),
        np.asarray([5.0] * 100 + [2.0] * 100),
        rng=np.random.default_rng(4),
    )
    assert set(np.unique(samples[:100])).issubset({1.25, 1.5})
    assert set(np.unique(samples[100:])).issubset({7.0, 8.0})


def test_grid_includes_low_flux_rows_and_uses_inclusive_minimum_count():
    noise = np.ones(19)
    flux = np.concatenate([np.full(10, 1e-4), np.full(9, 100.0)])
    error = np.arange(1, 20, dtype=np.float64)
    lookup = build_conditional_error_lookup(
        noise,
        flux,
        error,
        product_key="low3",
        noise_map_identity="noise",
        noise_bins=2,
        flux_bins=2,
        min_cell_count=10,
    )
    assert lookup.metadata["training_rows_accepted"] == 19
    assert sorted(lookup.cell_counts.tolist()) == [0, 0, 9, 10]
    eligible_cell = int(np.flatnonzero(lookup.cell_counts == 10)[0])
    sparse_cell = int(np.flatnonzero(lookup.cell_counts == 9)[0])
    assert lookup.resolved_cell_ids[sparse_cell] == eligible_cell
    assert np.float32(1.0) in lookup.absolute_error_values

    with pytest.raises(ValueError, match="No conditional error-grid cell reaches"):
        build_conditional_error_lookup(
            noise,
            flux,
            error,
            product_key="low3",
            noise_map_identity="noise",
            noise_bins=2,
            flux_bins=2,
            min_cell_count=11,
        )


def test_bounded_grid_excludes_training_outliers_but_clips_runtime_queries():
    noise = np.asarray([50.0, 100.0, 200.0, 1000.0, 2000.0, 200.0, 200.0])
    flux = np.asarray([1.0, 0.05, 1.0, 10_000.0, 1.0, 20_000.0, 1.0])
    error = np.asarray([5001.0, 5002.0, 1.0, 2.0, 5003.0, 5004.0, np.nan])
    lookup = build_conditional_error_lookup(
        noise,
        flux,
        error,
        product_key="mid1",
        noise_map_identity="noise",
        noise_bins=2,
        flux_bins=2,
        min_cell_count=1,
        noise_bounds_ujy_beam=(100.0, 1000.0),
        flux_bounds_mjy=(0.1, 10_000.0),
    )

    np.testing.assert_array_equal(lookup.absolute_error_values, [1.0, 2.0])
    np.testing.assert_allclose(lookup.log_noise_edges[[0, -1]], [2.0, 3.0])
    np.testing.assert_allclose(lookup.log_flux_edges[[0, -1]], [-1.0, 4.0])
    assert lookup.metadata["format_version"] == BOUNDED_ABSOLUTE_ERROR_LOOKUP_FORMAT_VERSION
    assert lookup.metadata["training_rows_total"] == 7
    assert lookup.metadata["training_rows_finite_positive_candidates"] == 6
    assert lookup.metadata["training_rows_rejected_invalid"] == 1
    assert lookup.metadata["rows_below_noise_bound"] == 1
    assert lookup.metadata["rows_above_noise_bound"] == 1
    assert lookup.metadata["rows_below_flux_bound"] == 1
    assert lookup.metadata["rows_above_flux_bound"] == 1
    assert lookup.metadata["rows_excluded_by_bounds_union"] == 4
    assert lookup.metadata["training_rows_accepted"] == 2
    assert lookup.metadata["training_rows_rejected"] == 5

    # Inclusive extrema stay in training; finite-positive runtime values beyond
    # them clip to boundary cells without rejection or retries.
    edge_cells = lookup.resolve_cells(
        np.asarray([100.0, 1000.0, 1.0, 1e6]),
        np.asarray([0.1, 10_000.0, 1e-6, 1e9]),
    )
    np.testing.assert_array_equal(edge_cells, [0, 3, 0, 3])
    counts = lookup.query_range_counts(
        np.asarray([100.0, 1000.0, 1.0, 1e6, np.nan]),
        np.asarray([0.1, 10_000.0, 1e-6, 1e9, 1.0]),
    )
    assert counts == {
        "queries_total": 5,
        "queries_finite_positive": 4,
        "queries_below_noise_bound": 1,
        "queries_above_noise_bound": 1,
        "queries_below_flux_bound": 1,
        "queries_above_flux_bound": 1,
        "queries_outside_bounds_union": 2,
    }


def test_bounded_cache_rejects_bounds_and_policy_mismatches(tmp_path: Path):
    lookup = build_conditional_error_lookup(
        np.asarray([100.0, 1000.0]),
        np.asarray([0.1, 10_000.0]),
        np.asarray([1.0, 2.0]),
        product_key="mid1",
        noise_map_identity="noise",
        noise_bins=2,
        flux_bins=2,
        min_cell_count=1,
        noise_bounds_ujy_beam=(100.0, 1000.0),
        flux_bounds_mjy=(0.1, 10_000.0),
    )
    output = tmp_path / "bounded.npz"
    save_conditional_error_lookup(lookup, output, diagnostics=False)
    restored = load_conditional_error_lookup(
        output,
        product_key="mid1",
        noise_map_identity="noise",
        noise_bins=2,
        flux_bins=2,
        min_cell_count=1,
        noise_bounds_ujy_beam=(100.0, 1000.0),
        flux_bounds_mjy=(0.1, 10_000.0),
    )
    assert restored.identity == lookup.identity

    with pytest.raises(RacsCacheValidationError, match="noise_bounds_ujy_beam"):
        load_conditional_error_lookup(
            output,
            product_key="mid1",
            noise_map_identity="noise",
            noise_bins=2,
            flux_bins=2,
            min_cell_count=1,
            noise_bounds_ujy_beam=(90.0, 1000.0),
            flux_bounds_mjy=(0.1, 10_000.0),
        )
    with pytest.raises(RacsCacheValidationError, match="format_version"):
        load_conditional_error_lookup(
            output,
            product_key="mid1",
            noise_map_identity="noise",
            noise_bins=2,
            flux_bins=2,
            min_cell_count=1,
        )


def test_grid_cache_round_trip_mismatch_and_diagnostics(tmp_path: Path):
    lookup = _lookup()
    output = tmp_path / "absolute.npz"
    save_conditional_error_lookup(lookup, output, diagnostics=True)
    restored = load_conditional_error_lookup(
        output,
        product_key="low3",
        noise_map_identity="synthetic-noise",
        noise_bins=2,
        flux_bins=2,
        min_cell_count=2,
    )
    np.testing.assert_array_equal(restored.resolved_cell_ids, lookup.resolved_cell_ids)
    assert restored.diagnostic_median_fractional_error is None
    assert set(
        restored.sample(
            np.full(50, 2.0),
            np.full(50, 5.0),
            rng=np.random.default_rng(8),
        )
    ).issubset({1.25, 1.5})
    assert (tmp_path / "absolute.diagnostics.png").is_file()
    assert (tmp_path / "absolute.marginals.png").is_file()
    assert (tmp_path / "absolute.summary.json").is_file()
    summary = json.loads((tmp_path / "absolute.summary.json").read_text())
    assert summary["noise_marginal_counts_by_grid_bin"] == [2, 2]
    assert summary["flux_marginal_counts_by_grid_bin"] == [2, 2]
    assert summary[
        "absolute_error_mjy_percentiles_0_1_5_16_50_84_95_99_100"
    ][0] == pytest.approx(1.25)
    with pytest.raises(RacsCacheValidationError, match="noise_map_identity"):
        load_conditional_error_lookup(
            output,
            product_key="low3",
            noise_map_identity="different",
            noise_bins=2,
            flux_bins=2,
            min_cell_count=2,
        )

    wrong_columns = replace(
        lookup,
        metadata={**lookup.metadata, "catalogue_columns": {"total_flux": "wrong"}},
    )
    save_conditional_error_lookup(wrong_columns, output, diagnostics=False)
    with pytest.raises(RacsCacheValidationError, match="catalogue_columns"):
        load_conditional_error_lookup(
            output,
            product_key="low3",
            noise_map_identity="synthetic-noise",
            noise_bins=2,
            flux_bins=2,
            min_cell_count=2,
            catalogue_columns={
                "total_flux": "Total_flux",
                "total_flux_error": "E_Total_flux",
            },
        )

    invalid_edges = lookup.log_noise_edges.copy()
    invalid_edges[1] = invalid_edges[0]
    save_conditional_error_lookup(
        replace(lookup, log_noise_edges=invalid_edges),
        output,
        diagnostics=False,
    )
    with pytest.raises(RacsCacheValidationError, match="strictly increasing"):
        load_conditional_error_lookup(
            output,
            product_key="low3",
            noise_map_identity="synthetic-noise",
            noise_bins=2,
            flux_bins=2,
            min_cell_count=2,
        )


def test_racs_helpers_load_both_caches_without_catalogue_or_source(tmp_path: Path):
    source = tmp_path / "RACS-low3.iqr.hpx"
    _write_ring_equatorial_noise_map(source)
    noise_cache = build_noise_map_cache(source, product_key="low3", target_nside=2)
    lookup = build_conditional_error_lookup(
        np.asarray([2.0, 2.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([0.25, 0.5]),
        product_key="low3",
        noise_map_identity=noise_cache.identity,
        noise_bins=2,
        flux_bins=2,
        min_cell_count=2,
        catalogue_columns={
            "ra": "RA",
            "dec": "Dec",
            "total_flux": "Total_flux",
            "total_flux_error": "E_Total_flux",
        },
    )
    sim = Racs(
        RacsConfig(
            flux_min=1.0,
            noise_map_nside=2,
            flux_error_noise_bins=2,
            flux_error_flux_bins=2,
            flux_error_min_cell_count=2,
            noisemap_data_dir=None,
            flux_error_noise_bounds_ujy_beam=None,
            flux_error_flux_bounds_mjy=None,
        )
    )
    sim._cache_dir = lambda: tmp_path  # type: ignore[method-assign]
    save_noise_map_cache(noise_cache, sim._noise_map_cache_path(), diagnostics=False)
    save_conditional_error_lookup(
        lookup,
        sim._absolute_error_lookup_cache_path(),
        diagnostics=False,
    )
    source.unlink()

    assert sim.load_cached_noise_map()
    assert sim.load_absolute_error_lookup()
    assert not sim.catalogue_is_loaded
    samples = sim.sample_absolute_flux_errors(
        np.asarray([2.0, 2.0]),
        np.asarray([1.0, 1.0]),
        rng=np.random.default_rng(9),
    )
    assert set(samples).issubset({0.25, 0.5})
