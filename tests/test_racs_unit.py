import unittest
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import sys
import types
from unittest.mock import patch

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
import healpy as hp
import numpy as np

if "dipoleutils.utils.data_loader" not in sys.modules:
    dipoleutils_module = types.ModuleType("dipoleutils")
    utils_module = types.ModuleType("dipoleutils.utils")
    data_loader_module = types.ModuleType("dipoleutils.utils.data_loader")

    class _TestDataLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def load(self):
            raise RuntimeError("Test stub DataLoader.load() should not be called.")

    data_loader_module.DataLoader = _TestDataLoader
    sys.modules["dipoleutils"] = dipoleutils_module
    sys.modules["dipoleutils.utils"] = utils_module
    sys.modules["dipoleutils.utils.data_loader"] = data_loader_module

from catsim import RACS_MID1, Racs, RacsConfig, RacsLow3, RacsLow3Config
from catsim.racs import LOW3_TEMPERATURE_EPSILON_FLOOR
from catsim.racs_products import RacsCatalogueColumns, RacsProductSpec, resolve_racs_product
from catsim.utils import weather


def _write_paf_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text(
        "\n".join(
            [
                '"Time","temperature"',
                *[f"{timestamp},{value}" for timestamp, value in rows],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _make_full_antenna_set(
    tmp_path: Path,
    minute_offset: int = 0,
    temperature_offset: float = 0.0,
) -> None:
    base_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=minute_offset)
    for antenna_index in range(1, 37):
        antenna_name = f"ak{antenna_index:02d}"
        _write_paf_csv(
            tmp_path / f"{antenna_name} ctrl_adc1_pafAvTemp-data.csv",
            [
                (
                    base_time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{antenna_index + temperature_offset:.1f}",
                ),
                (
                    (base_time + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                    f"{antenna_index + 100 + temperature_offset:.1f}",
                ),
            ],
        )


class RacsFluxErrorTests(unittest.TestCase):
    def setUp(self):
        self.sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))

    def _configure_minimal_generate_dipole_sim(
        self,
        n_samples: int,
        sim: Optional[RacsLow3] = None,
    ) -> RacsLow3:
        if sim is None:
            sim = self.sim
        sim.lookups_are_initialised = True
        n_pix = hp.nside2npix(sim.nside)
        sim.mask_map = np.ones(n_pix, dtype=bool)
        sim.tile_lookup_map = np.zeros(n_pix, dtype=np.int32)
        sim.tile_temperature_by_index = np.array([30.0], dtype=np.float64)
        sim.sample_points = lambda n, dtype=np.float64, rng=None: (
            np.linspace(0.0, 90.0, n, dtype=dtype),
            np.zeros(n, dtype=dtype),
        )
        sim.sample_spectral_indices = lambda n, rng=None: np.full(n, 0.8, dtype=np.float32)
        sim.aberrate_points = lambda ra, dec, dtype=np.float64: (
            np.asarray(ra, dtype=dtype),
            np.asarray(dec, dtype=dtype),
            np.zeros_like(ra, dtype=dtype),
        )
        sim.boost_fluxes = lambda flux, angle_to_dipole_deg, spectral_index, dtype=np.float64: (
            np.asarray(flux, dtype=dtype)
        )
        sim._source_isin_mask = lambda ra, dec: (
            np.ones(ra.shape[0], dtype=bool),
            np.arange(ra.shape[0], dtype=np.int64) % n_pix,
        )
        sim.sample_tiles_for_pixels = lambda pixel_indices, rng=None: np.zeros(
            pixel_indices.shape[0],
            dtype=np.int32,
        )
        sim.evaluate_temperature_enhancement = lambda tile_indices, temp_beta: (
            np.ones(tile_indices.shape[0], dtype=np.float64),
            np.full(tile_indices.shape[0], 30.0, dtype=np.float32),
        )
        sim.apply_temperature_enhancement = lambda flux, enhancement, dtype=np.float64: (
            np.asarray(flux, dtype=dtype)
        )
        sim.sample_elevations = lambda pixel_indices, rng=None: np.full(
            pixel_indices.shape[0],
            60.0,
            dtype=np.float32,
        )
        sim.evaluate_elevation_enhancement = lambda elevations, elevation_amp, elevation_trough: (
            np.ones(elevations.shape[0], dtype=np.float64)
        )
        sim.add_flux_error = lambda flux, flux_error, rng=None, dtype=np.float64: (
            np.asarray(flux, dtype=dtype)
        )
        sim.sample_fractional_errors = lambda pixel_indices, rng=None: np.full(
            pixel_indices.shape[0],
            0.1,
            dtype=np.float32,
        )
        sim.sample_fluxes = lambda n, rng=None: np.full(n, 100.0, dtype=np.float64)
        return sim

    def test_sample_fractional_errors_draws_from_pixel_lookup(self):
        # Pixel 0 has one possible value; pixel 1 has two values; pixel 2 is empty.
        self.sim.error_lookup_pixel_counts = np.array([1, 2, 0], dtype=np.int64)
        self.sim.error_lookup_pixel_starts = np.array([0, 1, 3], dtype=np.int64)
        self.sim.error_lookup_fractional_values = np.array(
            [0.1, 0.2, 0.3],
            dtype=np.float32,
        )

        rng = np.random.default_rng(123)
        samples = self.sim.sample_fractional_errors(
            np.array([0, 1, 1, 2], dtype=np.int64),
            rng=rng,
        )

        self.assertEqual(samples[0], np.float32(0.1))
        self.assertIn(samples[1], (np.float32(0.2), np.float32(0.3)))
        self.assertIn(samples[2], (np.float32(0.2), np.float32(0.3)))
        self.assertIn(samples[3], (np.float32(0.1), np.float32(0.2), np.float32(0.3)))

    def test_compute_total_flux_error_eta_scales_raw_sigma_by_sqrt_one_plus_eta(self):
        flux = np.full(8, 100.0, dtype=np.float64)
        fractional_error = np.full(8, 0.1, dtype=np.float64)

        sigma_base = self.sim.compute_total_flux_error(
            flux,
            fractional_error,
            fractional_error_eta=0.0,
            dtype=np.float64,
        )
        sigma_eta = self.sim.compute_total_flux_error(
            flux,
            fractional_error,
            fractional_error_eta=3.0,
            dtype=np.float64,
        )

        np.testing.assert_allclose(sigma_base, np.full(8, 10.0))
        np.testing.assert_allclose(sigma_eta, np.full(8, 20.0))

    def test_temperature_flux_summary_uses_independent_cut_without_changing_map(self):
        self._configure_minimal_generate_dipole_sim(n_samples=4)
        self.sim.cfg.flux_temperature_min_mjy = 2.0
        self.sim.sample_fluxes = lambda n, rng=None: np.asarray(
            [2.0, 10.0, 20.0, 30.0],
            dtype=np.float64,
        )[:n]

        density_map, mask, summaries = self.sim.generate_dipole_with_flux_summaries(
            log10_n_initial_samples=np.log10(4.0),
            temperature_edges=np.asarray([0.0, 40.0]),
            temperature_quantiles=(0.5,),
        )

        self.assertEqual(float(np.nansum(density_map[mask])), 2.0)
        np.testing.assert_allclose(summaries["temperature"], np.asarray([15.0]))

    def test_generate_dipole_keeps_two_value_return_api(self):
        self._configure_minimal_generate_dipole_sim(n_samples=4)

        result = self.sim.generate_dipole(log10_n_initial_samples=np.log10(4.0))

        self.assertEqual(len(result), 2)

    def test_compute_total_flux_error_rejects_negative_eta(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            self.sim.compute_total_flux_error(
                np.array([100.0]),
                np.array([0.1]),
                fractional_error_eta=-0.1,
            )

    def test_generate_dipole_rejects_invalid_clustering_parameters(self):
        self._configure_minimal_generate_dipole_sim(n_samples=8)

        with self.assertRaisesRegex(
            ValueError,
            "cluster_count_model='geometric'.*p_clus must lie in \\[0, 1\\]",
        ):
            self.sim.generate_dipole(
                log10_n_initial_samples=np.log10(8.0),
                p_clus=-0.1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "cluster_count_model='geometric'.*p_clus must lie in \\[0, 1\\]",
        ):
            self.sim.generate_dipole(
                log10_n_initial_samples=np.log10(8.0),
                p_clus=1.1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "cluster_count_model='geometric'.*clus_stop_prob must lie in \\(0, 1\\]",
        ):
            self.sim.generate_dipole(
                log10_n_initial_samples=np.log10(8.0),
                p_clus=0.5,
                clus_stop_prob=0.0,
            )

        with self.assertRaisesRegex(
            ValueError,
            "cluster_count_model='geometric'.*clus_stop_prob must lie in \\(0, 1\\]",
        ):
            self.sim.generate_dipole(
                log10_n_initial_samples=np.log10(8.0),
                p_clus=0.5,
                clus_stop_prob=1.1,
            )

        with self.assertRaisesRegex(ValueError, "lambda_clus is only valid"):
            self.sim.generate_dipole(
                log10_n_initial_samples=np.log10(8.0),
                lambda_clus=0.1,
            )

    def test_generate_dipole_rejects_invalid_poisson_clustering_parameters(self):
        sim = RacsLow3(
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                cluster_count_model="poisson",
            )
        )
        self._configure_minimal_generate_dipole_sim(n_samples=8, sim=sim)

        with self.assertRaisesRegex(ValueError, "lambda_clus must be non-negative"):
            sim.generate_dipole(
                log10_n_initial_samples=np.log10(8.0),
                lambda_clus=-0.1,
            )

        with self.assertRaisesRegex(ValueError, "p_clus is only valid"):
            sim.generate_dipole(
                log10_n_initial_samples=np.log10(8.0),
                p_clus=0.1,
                lambda_clus=0.1,
            )

        with self.assertRaisesRegex(ValueError, "clus_stop_prob is only valid"):
            sim.generate_dipole(
                log10_n_initial_samples=np.log10(8.0),
                lambda_clus=0.1,
                clus_stop_prob=0.5,
            )

    def test_add_flux_error_uses_precomputed_raw_sigma(self):
        flux = np.full(8, 100.0, dtype=np.float64)
        sigma = np.full(8, 10.0, dtype=np.float64)

        noisy_base = self.sim.add_flux_error(
            flux,
            sigma,
            rng=np.random.default_rng(7),
            dtype=np.float64,
        )
        noisy_double_sigma = self.sim.add_flux_error(
            flux,
            2.0 * sigma,
            rng=np.random.default_rng(7),
            dtype=np.float64,
        )

        np.testing.assert_allclose(noisy_double_sigma - flux, 2.0 * (noisy_base - flux))

    def test_generate_dipole_stores_effective_fractional_errors_after_eta_scaling(self):
        sim = self.sim
        self._configure_minimal_generate_dipole_sim(n_samples=10)

        n_samples = 10
        base_fractional_error = np.full(n_samples, 0.1, dtype=np.float32)

        sim.sample_fractional_errors = lambda pixel_indices, rng=None: base_fractional_error[
            : pixel_indices.shape[0]
        ]

        dmap, mask = sim.generate_dipole(
            log10_n_initial_samples=1.0,
            fractional_error_eta=3.0,
            temp_beta=0.0,
        )

        expected_effective = np.full(n_samples, 0.2, dtype=np.float32)
        np.testing.assert_allclose(sim.final_base_fractional_error_samples, base_fractional_error)
        np.testing.assert_allclose(sim.final_fractional_error_samples, expected_effective)
        self.assertIsNone(sim.final_elevation_samples)

        sampled_map = sim.sampled_fractional_error_map
        self.assertIsNotNone(sampled_map)
        finite = np.isfinite(sampled_map)
        self.assertTrue(np.any(finite))
        np.testing.assert_allclose(sampled_map[finite], np.full(np.count_nonzero(finite), 0.2))
        self.assertEqual(dmap.shape, mask.shape)

    def test_evaluate_temperature_enhancement_uses_hot_paf_suppression(self):
        self.sim.tile_temperature_by_index = np.array([24.0, 25.0, 30.0], dtype=np.float64)

        enhancement, temperatures = self.sim.evaluate_temperature_enhancement(
            tile_indices=np.array([0, 1, 2], dtype=np.int32),
            temp_beta=0.02,
        )

        np.testing.assert_array_equal(
            temperatures,
            np.array([24.0, 25.0, 30.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            enhancement,
            np.asarray([1.0, 1.0, 0.9], dtype=self.sim.dtype),
        )

    def test_evaluate_temperature_enhancement_supports_hot_quadratic(self):
        self.sim.cfg.temperature_model = "hot_quadratic"
        self.sim.tile_temperature_by_index = np.array(
            [24.0, 25.0, 30.0],
            dtype=np.float64,
        )

        enhancement, _ = self.sim.evaluate_temperature_enhancement(
            tile_indices=np.array([0, 1, 2], dtype=np.int32),
            temp_beta=0.02,
        )

        np.testing.assert_allclose(
            enhancement,
            np.asarray([1.0, 1.0, 0.99], dtype=self.sim.dtype),
        )

    def test_evaluate_temperature_enhancement_clips_to_positive_floor(self):
        self.sim.tile_temperature_by_index = np.array([60.0], dtype=np.float64)

        enhancement, temperatures = self.sim.evaluate_temperature_enhancement(
            tile_indices=np.array([0], dtype=np.int32),
            temp_beta=1.0,
        )

        self.assertEqual(temperatures[0], np.float32(60.0))
        self.assertEqual(
            enhancement[0],
            np.asarray(LOW3_TEMPERATURE_EPSILON_FLOOR, dtype=self.sim.dtype),
        )

    def test_evaluate_temperature_enhancement_rejects_invalid_temp_beta(self):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            self.sim.evaluate_temperature_enhancement(
                tile_indices=np.array([0], dtype=np.int32),
                temp_beta=-0.1,
            )

        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            self.sim.evaluate_temperature_enhancement(
                tile_indices=np.array([0], dtype=np.int32),
                temp_beta=np.nan,
            )

    def test_evaluate_elevation_enhancement_uses_degree_angles(self):
        enhancement = self.sim.evaluate_elevation_enhancement(
            np.array([30.0, 120.0, 210.0], dtype=np.float32),
            elevation_amp=0.5,
            elevation_trough=30.0,
        )

        np.testing.assert_allclose(
            enhancement,
            np.array([1.0, 1.5, 2.0], dtype=self.sim.dtype),
            rtol=1e-6,
        )

    def test_evaluate_elevation_enhancement_rejects_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "elevation_amp must be finite"):
            self.sim.evaluate_elevation_enhancement(
                np.array([60.0], dtype=np.float32),
                elevation_amp=-0.1,
                elevation_trough=30.0,
            )

        with self.assertRaisesRegex(ValueError, "elevation_trough must be finite"):
            self.sim.evaluate_elevation_enhancement(
                np.array([60.0], dtype=np.float32),
                elevation_amp=0.1,
                elevation_trough=np.nan,
            )

    def test_generate_dipole_remains_finite_when_linear_enhancement_hits_floor(self):
        sim = self.sim
        self._configure_minimal_generate_dipole_sim(n_samples=8)
        sim.tile_temperature_by_index = np.array([60.0], dtype=np.float64)

        n_samples = 8
        sim.sample_fractional_errors = lambda pixel_indices, rng=None: np.full(
            pixel_indices.shape[0],
            0.1,
            dtype=np.float32,
        )

        dmap, mask = sim.generate_dipole(
            log10_n_initial_samples=np.log10(float(n_samples)),
            fractional_error_eta=0.0,
            temp_beta=1.0,
        )

        self.assertEqual(dmap.shape, mask.shape)
        self.assertTrue(np.all(np.isfinite(sim.final_observed_flux_samples)))
        self.assertTrue(np.all(np.isfinite(sim.final_flux_error_samples)))
        self.assertTrue(np.all(sim.final_flux_error_samples >= 0.0))

    def test_generate_dipole_applies_elevation_enhancement(self):
        sim = self.sim
        sim.product = RACS_MID1
        n_samples = 6
        self._configure_minimal_generate_dipole_sim(n_samples=n_samples)
        sim.sample_elevations = lambda pixel_indices, rng=None: np.full(
            pixel_indices.shape[0],
            90.0,
            dtype=np.float32,
        )
        sim.evaluate_elevation_enhancement = Racs.evaluate_elevation_enhancement.__get__(
            sim,
            Racs,
        )
        sim.sample_fractional_errors = lambda pixel_indices, rng=None: np.zeros(
            pixel_indices.shape[0],
            dtype=np.float32,
        )

        sim.generate_dipole(
            log10_n_initial_samples=np.log10(float(n_samples)),
            elevation_amp=1.0,
            elevation_trough=0.0,
        )

        np.testing.assert_allclose(
            sim.final_observed_flux_samples,
            np.full(n_samples, 200.0, dtype=np.float32),
        )
        np.testing.assert_allclose(
            sim.final_elevation_samples,
            np.full(n_samples, 90.0, dtype=np.float32),
        )

    def test_generate_dipole_skips_clustering_when_p_clus_zero(self):
        sim = self.sim
        n_samples = 10
        self._configure_minimal_generate_dipole_sim(n_samples=n_samples)

        def _unexpected_cluster_sample(*args, **kwargs):
            raise AssertionError("sample_clustered_points() should not be called")

        sim.sample_clustered_points = _unexpected_cluster_sample

        sim.generate_dipole(
            log10_n_initial_samples=np.log10(float(n_samples)),
            p_clus=0.0,
        )

        self.assertEqual(sim.final_intrinsic_flux_samples.shape[0], n_samples)
        self.assertEqual(sim.final_longitudes.shape[0], n_samples)

    def test_generate_dipole_adds_cluster_components_on_top_of_parents(self):
        sim = self.sim
        n_expected_sources = 8.5
        n_parent_samples = 5
        self._configure_minimal_generate_dipole_sim(n_samples=n_parent_samples)

        flux_call_sizes: list[int] = []

        def _sample_fluxes(n, rng=None):
            flux_call_sizes.append(int(n))
            return np.full(n, 100.0 + len(flux_call_sizes), dtype=np.float64)

        sim.sample_fluxes = _sample_fluxes
        sim.sample_clustered_points = lambda parent_ra, parent_dec, counts, rng=None, dtype=np.float64: (
            (np.repeat(np.asarray(parent_ra, dtype=np.float64), counts) + 0.01).astype(
                dtype,
                copy=False,
            ),
            np.repeat(np.asarray(parent_dec, dtype=np.float64), counts).astype(dtype, copy=False),
        )

        class _FixedClusterRng:
            def __init__(self):
                self._delegate = np.random.default_rng(123)

            def random(self, size=None):
                if size == n_parent_samples:
                    return np.array([0.1, 0.9, 0.2, 0.95, 0.8], dtype=np.float64)
                return self._delegate.random(size=size)

            def geometric(self, p, size=None):
                if size == 2 and np.isscalar(p):
                    return np.array([2, 1], dtype=np.int64)
                return self._delegate.geometric(p, size=size)

            def __getattr__(self, name):
                return getattr(self._delegate, name)

        class _FixedClusterKey:
            def _generator(self):
                return _FixedClusterRng()

        sim.generate_dipole(
            log10_n_initial_samples=np.log10(float(n_expected_sources)),
            p_clus=0.5,
            clus_stop_prob=1.0,
            rng_key=_FixedClusterKey(),
        )

        self.assertEqual(flux_call_sizes, [n_parent_samples, 3])
        self.assertEqual(sim.final_intrinsic_flux_samples.shape[0], n_parent_samples + 3)
        np.testing.assert_allclose(
            sim.final_longitudes[:n_parent_samples],
            np.linspace(0.0, 90.0, n_parent_samples, dtype=np.float32),
        )
        self.assertTrue(np.all(sim.final_longitudes[n_parent_samples:] >= 0.01))

    def test_generate_dipole_poisson_model_adds_cluster_components_on_top_of_parents(self):
        sim = RacsLow3(
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                cluster_count_model="poisson",
            )
        )
        n_expected_sources = 10
        n_parent_samples = 5
        self._configure_minimal_generate_dipole_sim(n_samples=n_parent_samples, sim=sim)

        flux_call_sizes: list[int] = []

        def _sample_fluxes(n, rng=None):
            flux_call_sizes.append(int(n))
            return np.full(n, 100.0 + len(flux_call_sizes), dtype=np.float64)

        sim.sample_fluxes = _sample_fluxes
        sim.sample_clustered_points = lambda parent_ra, parent_dec, counts, rng=None, dtype=np.float64: (
            (np.repeat(np.asarray(parent_ra, dtype=np.float64), counts) + 0.01).astype(
                dtype,
                copy=False,
            ),
            np.repeat(np.asarray(parent_dec, dtype=np.float64), counts).astype(dtype, copy=False),
        )

        class _FixedPoissonRng:
            def __init__(self):
                self._delegate = np.random.default_rng(123)

            def poisson(self, lam, size=None):
                if size == n_parent_samples and np.isscalar(lam):
                    return np.array([2, 0, 1, 0, 0], dtype=np.int64)
                return self._delegate.poisson(lam, size=size)

            def __getattr__(self, name):
                return getattr(self._delegate, name)

        class _FixedPoissonKey:
            def _generator(self):
                return _FixedPoissonRng()

        sim.generate_dipole(
            log10_n_initial_samples=np.log10(float(n_expected_sources)),
            lambda_clus=1.0,
            rng_key=_FixedPoissonKey(),
        )

        self.assertEqual(flux_call_sizes, [n_parent_samples, 3])
        self.assertEqual(sim.final_intrinsic_flux_samples.shape[0], n_parent_samples + 3)
        np.testing.assert_allclose(
            sim.final_longitudes[:n_parent_samples],
            np.linspace(0.0, 90.0, n_parent_samples, dtype=np.float32),
        )
        self.assertTrue(np.all(sim.final_longitudes[n_parent_samples:] >= 0.01))

    def test_generate_dipole_normalizes_parent_count_by_expected_cluster_multiplicity(self):
        geometric_sim = self._configure_minimal_generate_dipole_sim(n_samples=8)
        geometric_flux_call_sizes: list[int] = []
        geometric_sim.sample_fluxes = lambda n, rng=None: (
            geometric_flux_call_sizes.append(int(n)) or np.full(n, 100.0, dtype=np.float64)
        )
        geometric_sim.generate_dipole(
            log10_n_initial_samples=np.log10(12.0),
            p_clus=0.5,
            clus_stop_prob=1.0,
        )

        poisson_sim = RacsLow3(
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                cluster_count_model="poisson",
            )
        )
        self._configure_minimal_generate_dipole_sim(n_samples=6, sim=poisson_sim)
        poisson_flux_call_sizes: list[int] = []
        poisson_sim.sample_fluxes = lambda n, rng=None: (
            poisson_flux_call_sizes.append(int(n)) or np.full(n, 100.0, dtype=np.float64)
        )
        poisson_sim.generate_dipole(
            log10_n_initial_samples=np.log10(12.0),
            lambda_clus=1.0,
        )

        self.assertEqual(geometric_flux_call_sizes[0], 8)
        self.assertEqual(poisson_flux_call_sizes[0], 6)

    def test_generate_dipole_with_unit_stop_probability_adds_one_component_per_clustered_parent(self):
        sim = self.sim
        n_expected_sources = 8
        n_parent_samples = 4
        self._configure_minimal_generate_dipole_sim(n_samples=n_parent_samples)

        sim.sample_clustered_points = lambda parent_ra, parent_dec, counts, rng=None, dtype=np.float64: (
            (np.repeat(np.asarray(parent_ra, dtype=np.float64), counts) + 0.01).astype(
                dtype,
                copy=False,
            ),
            np.repeat(np.asarray(parent_dec, dtype=np.float64), counts).astype(dtype, copy=False),
        )

        class _UnitStopRng:
            def __init__(self):
                self._delegate = np.random.default_rng(123)

            def random(self, size=None):
                if size == n_parent_samples:
                    return np.array([0.2, 0.8, 0.1, 0.7], dtype=np.float64)
                return self._delegate.random(size=size)

            def geometric(self, p, size=None):
                if size == 2 and p == 1.0:
                    return np.ones(2, dtype=np.int64)
                return self._delegate.geometric(p, size=size)

            def __getattr__(self, name):
                return getattr(self._delegate, name)

        class _UnitStopKey:
            def _generator(self):
                return _UnitStopRng()

        sim.generate_dipole(
            log10_n_initial_samples=np.log10(float(n_expected_sources)),
            p_clus=0.5,
            clus_stop_prob=1.0,
            rng_key=_UnitStopKey(),
        )

        self.assertEqual(sim.final_intrinsic_flux_samples.shape[0], n_parent_samples + 2)

    def test_sample_clustered_points_enforces_minimum_offset(self):
        sim = RacsLow3(
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                cluster_r0_arcsec=0.01,
                cluster_r_cut_arcsec=20.0,
            )
        )

        parent_ra = np.array([120.0], dtype=np.float64)
        parent_dec = np.array([-30.0], dtype=np.float64)
        child_ra, child_dec = sim.sample_clustered_points(
            parent_ra,
            parent_dec,
            np.array([256], dtype=np.int64),
            rng=np.random.default_rng(123),
            dtype=np.float64,
        )

        parent_coord = SkyCoord(ra=parent_ra[0] * u.deg, dec=parent_dec[0] * u.deg, frame="icrs")
        child_coord = SkyCoord(ra=child_ra * u.deg, dec=child_dec * u.deg, frame="icrs")
        separations_arcsec = child_coord.separation(parent_coord).arcsec

        self.assertEqual(child_ra.shape[0], 256)
        self.assertTrue(np.all(np.isfinite(child_ra)))
        self.assertTrue(np.all(np.isfinite(child_dec)))
        self.assertTrue(np.all(separations_arcsec >= 20.0 - 1e-6))

    def test_sample_tiles_for_pixels_draws_from_pixel_mixture(self):
        self.sim.sbid_mixture_counts = np.array([2], dtype=np.int64)
        self.sim.sbid_mixture_starts = np.array([0], dtype=np.int64)
        self.sim.sbid_mixture_tile_indices = np.array([0, 1], dtype=np.int32)
        self.sim.sbid_mixture_probabilities = np.array([0.75, 0.25], dtype=np.float64)

        sampled = self.sim.sample_tiles_for_pixels(
            np.zeros(2000, dtype=np.int64),
            rng=np.random.default_rng(123),
        )
        counts = np.bincount(sampled, minlength=2).astype(np.float64)
        frequencies = counts / counts.sum()

        self.assertTrue(np.all(sampled >= 0))
        self.assertAlmostEqual(frequencies[0], 0.75, delta=0.05)
        self.assertAlmostEqual(frequencies[1], 0.25, delta=0.05)

    def test_build_temperature_map_uses_pixel_mixture_mean(self):
        self.sim.tile_temperature_by_index = np.array([20.0, 30.0], dtype=np.float64)
        self.sim.sbid_mixture_counts = np.array([2], dtype=np.int64)
        self.sim.sbid_mixture_starts = np.array([0], dtype=np.int64)
        self.sim.sbid_mixture_tile_indices = np.array([0, 1], dtype=np.int32)
        self.sim.sbid_mixture_probabilities = np.array([0.75, 0.25], dtype=np.float64)

        self.sim.build_temperature_map()

        self.assertAlmostEqual(float(self.sim.temperature_map[0]), 22.5)


class PafWeatherLookupTests(unittest.TestCase):
    def test_get_paf_antenna_temperatures_for_observation_returns_all_36_temperatures(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _make_full_antenna_set(tmp_path)
            obs_mjd = Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd

            result = weather.get_paf_antenna_temperatures_for_observation(
                obs_mjd,
                data_dir=tmp_path,
                max_interpolation_gap_minutes=20.0,
            )

            self.assertEqual(
                result.antenna_names,
                tuple(f"ak{antenna_index:02d}" for antenna_index in range(1, 37)),
            )
            self.assertEqual(result.temperatures_c.shape, (36,))
            np.testing.assert_allclose(result.temperatures_c, np.arange(41.0, 77.0))
            np.testing.assert_allclose(
                result.matched_time_offsets_seconds,
                np.full(36, 240.0),
            )

    def test_get_paf_antenna_temperatures_for_observation_marks_large_interpolation_gaps_as_nan(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _make_full_antenna_set(tmp_path, minute_offset=30)
            obs_mjd = Time(datetime(2023, 12, 31, 16, 0, tzinfo=UTC)).mjd

            result = weather.get_paf_antenna_temperatures_for_observation(
                obs_mjd,
                data_dir=tmp_path,
                max_interpolation_gap_minutes=20.0,
            )

            self.assertTrue(np.all(np.isnan(result.temperatures_c)))
            self.assertTrue(np.all(np.isnan(result.matched_time_offsets_seconds)))
            self.assertTrue(np.all(np.isnan(result.matched_unix_seconds)))

    def test_get_mean_paf_temperatures_for_observations_reuses_unique_timestamps(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _make_full_antenna_set(tmp_path)
            repeated_mjd = Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd

            temperatures = weather.get_mean_paf_temperatures_for_observations(
                [repeated_mjd, repeated_mjd],
                data_dir=tmp_path,
                max_interpolation_gap_minutes=20.0,
            )

            np.testing.assert_allclose(
                temperatures,
                np.full(2, np.mean(np.arange(41.0, 77.0))),
            )

    def test_get_open_meteo_temperatures_for_mjd_linearly_interpolates_hourly_data(self):
        hourly_unix = np.array(
            [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC).timestamp(),
                datetime(2024, 1, 1, 1, 0, tzinfo=UTC).timestamp(),
                datetime(2024, 1, 1, 2, 0, tzinfo=UTC).timestamp(),
            ],
            dtype=float,
        )
        hourly_temperatures = np.array([10.0, 14.0, 18.0], dtype=float)
        mjd_values = [
            Time(datetime(2024, 1, 1, 1, 0, tzinfo=UTC)).mjd,
            Time(datetime(2024, 1, 1, 1, 30, tzinfo=UTC)).mjd,
        ]

        with patch(
            "catsim.utils.weather._fetch_open_meteo_hourly_temperature",
            return_value=(hourly_unix, hourly_temperatures),
        ) as mocked_fetch:
            temperatures = weather.get_open_meteo_temperatures_for_mjd(
                mjd_values,
                latitude_deg=-1.0,
                longitude_deg=2.0,
                timeout=3.0,
            )

        np.testing.assert_allclose(temperatures, np.array([14.0, 16.0]))
        mocked_fetch.assert_called_once()
        self.assertEqual(mocked_fetch.call_args.kwargs["latitude_deg"], -1.0)
        self.assertEqual(mocked_fetch.call_args.kwargs["longitude_deg"], 2.0)
        self.assertEqual(mocked_fetch.call_args.kwargs["timeout"], 3.0)


class RacsInitialiseDataTests(unittest.TestCase):
    def test_product_registry_resolves_low3_and_mid1_metadata(self):
        low3 = resolve_racs_product("low3")
        mid1 = resolve_racs_product("racs_mid1")

        self.assertEqual(low3.data_loader_args, ("racs", "low3"))
        self.assertEqual(low3.data_dir_name, "racs_low3")
        self.assertEqual(low3.columns.dec, "Dec")
        self.assertEqual(low3.columns.source_name, "Name")
        self.assertIsNone(low3.columns.elevation)
        self.assertEqual(mid1, RACS_MID1)
        self.assertEqual(mid1.data_loader_args, ("racs", "mid1"))
        self.assertEqual(mid1.data_dir_name, "racs_mid1")
        self.assertEqual(mid1.columns.dec, "DEC")
        self.assertEqual(mid1.columns.field_id, "Tile_ID")
        self.assertEqual(mid1.columns.source_name, "Source_Name")
        self.assertEqual(mid1.columns.elevation, "ALT")

        with self.assertRaisesRegex(ValueError, "Unknown RACS product"):
            resolve_racs_product("not-a-product")

    def test_initialise_data_skips_elevation_lookup_for_low3(self):
        sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))

        with (
            patch.object(sim, "load_flux_distribution", return_value=True),
            patch.object(sim, "load_tile_metadata", return_value=True),
            patch.object(sim, "load_tile_lookup", return_value=True),
            patch.object(sim, "load_sbid_mixture_lookup", return_value=True),
            patch.object(sim, "load_fractional_error_lookup", return_value=True),
            patch.object(
                sim,
                "load_elevation_lookup",
                side_effect=AssertionError("LOW3 must not load elevation data"),
            ),
            patch.object(sim, "load_mask_map"),
            patch.object(sim, "load_temperature_table"),
        ):
            sim.initialise_data()

        self.assertTrue(sim.lookups_are_initialised)

    def test_low3_rejects_nonzero_elevation_model_at_simulation_time(self):
        sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))
        sim.lookups_are_initialised = True

        with self.assertRaisesRegex(ValueError, "does not define an elevation column"):
            sim.generate_dipole(
                log10_n_initial_samples=1.0,
                elevation_amp=0.1,
            )

    def test_lookup_builders_use_product_column_mapping(self):
        product = RacsProductSpec(
            key="synthetic",
            label="Synthetic RACS",
            data_loader_catalogue="racs",
            data_loader_variant="synthetic",
            data_dir_name="racs_synthetic",
            columns=RacsCatalogueColumns(
                ra="source_ra",
                dec="source_dec",
                tile_id="tile_sbid",
                total_flux="flux_total",
                total_flux_error="flux_error_total",
                scan_start_mjd="scan_mjd",
                scan_length="scan_duration",
                field_id="field_name",
                source_name="source_name",
                elevation="source_alt",
            ),
        )
        sim = Racs(
            RacsConfig(
                flux_min=1.0,
                product=product,
                nside=1,
                chunk_size=16,
                fractional_error_flux_min_mjy=1.0,
            )
        )
        sim.catalogue = Table(
            {
                "source_ra": np.array([0.0, 90.0, 180.0, 270.0]),
                "source_dec": np.array([0.0, 20.0, -20.0, 40.0]),
                "tile_sbid": np.array([101, 101, 202, 303]),
                "flux_total": np.array([1.0, 2.0, 4.0, 8.0]),
                "flux_error_total": np.array([0.1, 0.2, 0.4, 0.8]),
                "scan_mjd": np.array([60000.0, 60000.0, 60001.0, 60002.0]),
                "scan_duration": np.array([10.0, 10.0, 12.0, 14.0]),
                "field_name": np.array(["a", "a", "b", "c"]),
                "source_alt": np.array([40.0, 45.0, 50.0, 55.0]),
            }
        )
        sim.catalogue_is_loaded = True

        sim.build_flux_distribution()
        sim.build_tile_metadata()
        sim.build_tile_lookup()
        sim.build_fractional_error_lookup()
        sim.build_elevation_lookup()
        sim.load_mask_map()

        np.testing.assert_array_equal(sim.tile_sbids, np.array([101, 202, 303], dtype=np.int32))
        self.assertEqual(sim.log_flux_bin_cdf[-1], 1.0)
        self.assertGreater(sim.error_lookup_fractional_values.size, 0)
        self.assertGreater(sim.elevation_lookup_values.size, 0)
        np.testing.assert_array_equal(sim.mask_map, sim.tile_lookup_map >= 0)

    def test_build_elevation_lookup_rejects_missing_elevation_column(self):
        sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))
        sim.catalogue = Table(
            {
                "RA": np.array([0.0]),
                "Dec": np.array([0.0]),
            }
        )
        sim.catalogue_is_loaded = True

        with self.assertRaisesRegex(ValueError, "does not define an elevation column"):
            sim.build_elevation_lookup()

    def test_sample_elevations_uses_pixel_values_and_global_fallback(self):
        product = RacsProductSpec(
            key="synthetic",
            label="Synthetic RACS",
            data_loader_catalogue="racs",
            data_loader_variant="synthetic",
            data_dir_name="racs_synthetic",
            columns=RacsCatalogueColumns(
                ra="RA",
                dec="Dec",
                tile_id="SBID",
                total_flux="Total_flux",
                total_flux_error="E_Total_flux",
                scan_start_mjd="Scan_start_MJD",
                scan_length="Scan_length",
                field_id="Field_ID",
                source_name="Name",
                elevation="ALT",
            ),
        )
        sim = Racs(RacsConfig(product=product, flux_min=15.0, nside=1, chunk_size=16))
        n_pix = hp.nside2npix(sim.nside)
        sim.elevation_lookup_pixel_counts = np.zeros(n_pix, dtype=np.int64)
        sim.elevation_lookup_pixel_counts[0] = 2
        sim.elevation_lookup_pixel_starts = np.zeros(n_pix, dtype=np.int64)
        sim.elevation_lookup_values = np.array([10.0, 20.0], dtype=np.float32)

        samples = sim.sample_elevations(
            np.array([0, 1], dtype=np.int64),
            rng=np.random.default_rng(3),
        )

        self.assertIn(float(samples[0]), {10.0, 20.0})
        self.assertIn(float(samples[1]), {10.0, 20.0})

    def test_config_rejects_invalid_clustering_parameters(self):
        with self.assertRaisesRegex(ValueError, "cluster_r0_arcsec must be positive"):
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                cluster_r0_arcsec=0.0,
            )

        with self.assertRaisesRegex(ValueError, "cluster_r_cut_arcsec must be non-negative"):
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                cluster_r_cut_arcsec=-1.0,
            )

        with self.assertRaisesRegex(ValueError, "cluster_count_model must be either"):
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                cluster_count_model="unknown",
            )

        with self.assertRaisesRegex(ValueError, "paf_reference_temp_c must be finite"):
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                paf_reference_temp_c=np.nan,
            )

        self.assertEqual(RacsLow3Config(flux_min=15.0).temperature_model, "hot_linear")
        with self.assertRaisesRegex(ValueError, "temperature_model must be either"):
            RacsLow3Config(
                flux_min=15.0,
                temperature_model="unknown",
            )

    def test_initialise_data_uses_cached_lookups_without_loading_catalogue(self):
        with TemporaryDirectory() as tmpdir:
            product = RacsProductSpec(
                key="synthetic",
                label="Synthetic RACS",
                data_loader_catalogue="racs",
                data_loader_variant="synthetic",
                data_dir_name="racs_synthetic",
                default_mask_filename="mask.npy",
                columns=RacsCatalogueColumns(
                    ra="RA",
                    dec="Dec",
                    tile_id="SBID",
                    total_flux="Total_flux",
                    total_flux_error="E_Total_flux",
                    scan_start_mjd="Scan_start_MJD",
                    scan_length="Scan_length",
                    field_id="Field_ID",
                    source_name="Name",
                    elevation="ALT",
                ),
            )
            sim = Racs(
                RacsConfig(
                    product=product,
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=tmpdir,
                )
            )
            cache_dir = Path(tmpdir)
            _write_paf_csv(cache_dir / "ak01.csv", [])
            n_pix = hp.nside2npix(sim.nside)

            sim._cache_dir = lambda: cache_dir
            sim._mask_map_path = lambda: cache_dir / "mask.npy"
            np.save(cache_dir / "mask.npy", np.ones(n_pix, dtype=np.uint8))

            sim.log_flux_bin_edges = np.array([0.0, 1.0, 2.0], dtype=np.float64)
            sim.log_flux_bin_probabilities = np.array([0.25, 0.75], dtype=np.float64)
            sim.log_flux_bin_cdf = np.array([0.25, 1.0], dtype=np.float64)
            sim.save_flux_distribution()

            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)
            sim.tile_scan_length = np.array([10.0, 11.0], dtype=np.float64)
            sim.tile_field_id = np.array(["field-a", "field-b"])
            sim._tile_index_from_sbid = {101: 0, 202: 1}
            sim.save_tile_metadata()

            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[:2] = np.array([101, 202], dtype=np.int32)
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[:2] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_starts[1] = 1
            sim.sbid_mixture_tile_indices = np.array([0, 1], dtype=np.int32)
            sim.sbid_mixture_probabilities = np.array([1.0, 1.0], dtype=np.float64)
            sim.save_tile_lookup()
            sim.save_sbid_mixture_lookup()
            self.assertTrue((cache_dir / "sbid_lookup_nside64.png").exists())
            self.assertTrue((cache_dir / "sbid_mixture_lookup_nside64.npz").exists())

            sim.error_lookup_pixel_counts = np.zeros(n_pix, dtype=np.int64)
            sim.error_lookup_pixel_starts = np.zeros(n_pix, dtype=np.int64)
            sim.error_lookup_fractional_values = np.array([0.1], dtype=np.float32)
            sim.fractional_error_map = np.full(n_pix, 0.1, dtype=np.float32)
            sim.save_fractional_error_lookup()
            self.assertTrue(
                (
                    cache_dir
                    / "fractional_error_lookup_nside64_fluxmin10p0mjy.png"
                ).exists()
            )

            sim.tile_temperature_by_index = np.array([20.0, 21.0], dtype=np.float64)
            sim.temperature_map = np.full(n_pix, np.nan, dtype=np.float32)
            sim.temperature_map[:2] = np.array([20.0, 21.0], dtype=np.float32)
            sim.save_temperature_lookup()
            self.assertTrue((cache_dir / "temperature_lookup_nside64.png").exists())

            sim.elevation_lookup_pixel_counts = np.zeros(n_pix, dtype=np.int64)
            sim.elevation_lookup_pixel_counts[:2] = 1
            sim.elevation_lookup_pixel_starts = np.zeros(n_pix, dtype=np.int64)
            sim.elevation_lookup_pixel_starts[1] = 1
            sim.elevation_lookup_values = np.array([50.0, 60.0], dtype=np.float32)
            sim.elevation_map = np.full(n_pix, np.nan, dtype=np.float32)
            sim.elevation_map[:2] = np.array([50.0, 60.0], dtype=np.float32)
            sim.save_elevation_lookup()
            self.assertTrue((cache_dir / "elevation_lookup_nside64.png").exists())
            self.assertTrue((cache_dir / "elevation_lookup_eq.png").exists())
            self.assertTrue((cache_dir / "elevation_lookup_gal.png").exists())

            sim.load_catalogue = lambda: (_ for _ in ()).throw(
                AssertionError("initialise_data() unexpectedly loaded the catalogue")
            )

            sim.initialise_data()

            self.assertTrue(sim.lookups_are_initialised)
            self.assertFalse(sim.catalogue_is_loaded)
            self.assertFalse(hasattr(sim, "catalogue"))
            np.testing.assert_array_equal(sim.tile_sbids, np.array([101, 202], dtype=np.int32))
            np.testing.assert_allclose(
                sim.temperature_map[:2],
                np.array([20.0, 21.0], dtype=np.float32),
            )
            np.testing.assert_allclose(
                sim.elevation_map[:2],
                np.array([50.0, 60.0], dtype=np.float32),
            )
            np.testing.assert_array_equal(sim.sbid_mixture_tile_indices, np.array([0, 1], dtype=np.int32))

    def test_load_temperature_table_raises_when_paf_directory_missing(self):
        sim = RacsLow3(
            RacsLow3Config(
                flux_min=15.0,
                nside=64,
                chunk_size=16,
                paf_temperature_data_dir="/definitely/missing/paf_temps",
            )
        )
        sim.tile_sbids = np.array([101], dtype=np.int32)
        sim.tile_scan_start_mjd = np.array([60000.0], dtype=np.float64)
        sim.tile_lookup_map = np.array([101], dtype=np.int32)
        sim._tile_index_from_sbid = {101: 0}
        sim.sbid_mixture_counts = np.array([1], dtype=np.int64)
        sim.sbid_mixture_starts = np.array([0], dtype=np.int64)
        sim.sbid_mixture_tile_indices = np.array([0], dtype=np.int32)
        sim.sbid_mixture_probabilities = np.array([1.0], dtype=np.float64)

        with self.assertRaises(FileNotFoundError):
            sim.load_temperature_table()

    def test_load_temperature_table_builds_from_paf_lookup_and_saves_new_cache(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            paf_dir = tmp_path / "paf"
            n_pix = hp.nside2npix(64)
            cache_dir.mkdir()
            paf_dir.mkdir()
            _make_full_antenna_set(paf_dir)

            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=str(paf_dir),
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array(
                [
                    Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd,
                    Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd,
                ],
                dtype=np.float64,
            )
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[:2] = np.array([101, 202], dtype=np.int32)
            sim._tile_index_from_sbid = {101: 0, 202: 1}
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[:2] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_starts[1] = 1
            sim.sbid_mixture_tile_indices = np.array([0, 1], dtype=np.int32)
            sim.sbid_mixture_probabilities = np.array([1.0, 1.0], dtype=np.float64)

            with self.assertLogs("catsim.racs", level="WARNING") as logs:
                sim.load_temperature_table()

            expected_temperature = np.mean(np.arange(41.0, 77.0))
            np.testing.assert_allclose(
                sim.tile_temperature_by_index,
                np.full(2, expected_temperature),
            )
            np.testing.assert_allclose(
                sim.temperature_map[:2],
                np.full(2, expected_temperature, dtype=np.float32),
            )
            self.assertTrue((cache_dir / "temperature_lookup_nside64.npz").exists())
            self.assertTrue((cache_dir / "temperature_lookup_nside64.png").exists())
            self.assertTrue(
                any("legacy flat directory" in message for message in logs.output)
            )

    def test_load_temperature_table_uses_product_paf_subdirectory(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paf_root = tmp_path / "paf_temps"
            low3_paf_dir = paf_root / "low3"
            mid1_paf_dir = paf_root / "mid1"
            cache_dir = tmp_path / "cache"
            n_pix = hp.nside2npix(64)
            low3_paf_dir.mkdir(parents=True)
            mid1_paf_dir.mkdir(parents=True)
            _make_full_antenna_set(low3_paf_dir, temperature_offset=0.0)
            _make_full_antenna_set(mid1_paf_dir, temperature_offset=1000.0)

            sim = Racs(
                RacsConfig(
                    product=RACS_MID1,
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=str(paf_root),
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array(
                [Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd],
                dtype=np.float64,
            )
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[0] = 101
            sim._tile_index_from_sbid = {101: 0}
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[0] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_tile_indices = np.array([0], dtype=np.int32)
            sim.sbid_mixture_probabilities = np.array([1.0], dtype=np.float64)

            with self.assertLogs("catsim.racs", level="INFO") as logs:
                sim.load_temperature_table()

            expected_temperature = np.mean(np.arange(1041.0, 1077.0))
            np.testing.assert_allclose(
                sim.tile_temperature_by_index,
                np.array([expected_temperature]),
            )
            self.assertTrue(any(str(mid1_paf_dir) in message for message in logs.output))
            self.assertFalse(any(str(low3_paf_dir) in message for message in logs.output))
            with np.load(cache_dir / "temperature_lookup_nside64.npz") as data:
                self.assertEqual(str(data["product_key"]), "mid1")
                self.assertEqual(str(data["paf_temperature_data_dir"]), str(mid1_paf_dir))

    def test_configured_paf_root_ignores_legacy_cache_without_source_metadata(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            paf_root = tmp_path / "paf_temps"
            paf_dir = paf_root / "low3"
            n_pix = hp.nside2npix(64)
            cache_dir.mkdir()
            paf_dir.mkdir(parents=True)
            _make_full_antenna_set(paf_dir)

            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=str(paf_root),
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array(
                [Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd],
                dtype=np.float64,
            )
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[0] = 101
            sim._tile_index_from_sbid = {101: 0}
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[0] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_tile_indices = np.array([0], dtype=np.int32)
            sim.sbid_mixture_probabilities = np.array([1.0], dtype=np.float64)
            np.savez_compressed(
                sim._temperature_lookup_cache_path(),
                nside=np.asarray(sim.nside, dtype=np.int64),
                tile_sbids=sim.tile_sbids.astype(np.int32, copy=False),
                tile_temperature_by_index=np.array([999.0], dtype=np.float64),
            )

            sim.load_temperature_table()

            expected_temperature = np.mean(np.arange(41.0, 77.0))
            np.testing.assert_allclose(
                sim.tile_temperature_by_index,
                np.array([expected_temperature]),
            )

    def test_load_temperature_table_raises_when_any_sbid_temperature_is_nan(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            paf_dir = tmp_path / "paf"
            cache_dir.mkdir()
            paf_dir.mkdir()
            _make_full_antenna_set(paf_dir, minute_offset=120)

            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=str(paf_dir),
                    paf_max_interpolation_gap_minutes=1.0,
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array(
                [
                    Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd,
                    Time(datetime(2023, 12, 31, 16, 4, tzinfo=UTC)).mjd,
                ],
                dtype=np.float64,
            )

            with self.assertRaisesRegex(ValueError, "non-finite temperatures.*101, 202"):
                sim.load_temperature_table()

            self.assertFalse((cache_dir / "temperature_lookup_nside64.npz").exists())

    def test_reference_fallback_requires_explicit_positive_tile_limit(self):
        for value in (None, 0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "max_reference_fallback_tiles.*positive integer"
                ):
                    RacsConfig(
                        flux_min=15.0,
                        temperature_fallback="reference",
                        max_reference_fallback_tiles=value,
                    )

    def test_reference_fallback_fills_only_missing_paf_temperature(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paf_dir = tmp_path / "paf"
            cache_dir = tmp_path / "cache"
            paf_dir.mkdir()
            _write_paf_csv(paf_dir / "ak01.csv", [])
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    paf_temperature_data_dir=str(paf_dir),
                    paf_reference_temp_c=25.0,
                    temperature_fallback="reference",
                    max_reference_fallback_tiles=1,
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202, 303], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0, 60002.0])

            with (
                self.assertLogs("catsim.racs", level="WARNING") as logs,
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations",
                    return_value=np.array([10.0, np.nan, 30.0]),
                ),
                patch.object(sim, "build_temperature_map"),
            ):
                sim.load_temperature_table()

            np.testing.assert_allclose(
                sim.tile_temperature_by_index, np.array([10.0, 25.0, 30.0])
            )
            self.assertTrue(
                any(
                    "reference temperature 25.000 C" in message
                    for message in logs.output
                )
            )
            self.assertTrue(any("202" in message for message in logs.output))
            with np.load(cache_dir / "temperature_lookup_nside64.npz") as data:
                np.testing.assert_array_equal(
                    data["tile_temperature_sources"],
                    np.array(["mean_paf", "reference", "mean_paf"]),
                )

    def test_reference_fallback_rejects_missing_count_above_limit(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paf_dir = tmp_path / "paf"
            paf_dir.mkdir()
            _write_paf_csv(paf_dir / "ak01.csv", [])
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    paf_temperature_data_dir=str(paf_dir),
                    temperature_fallback="reference",
                    max_reference_fallback_tiles=1,
                )
            )
            sim._cache_dir = lambda: tmp_path / "cache"
            sim.tile_sbids = np.array([101, 202, 303], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0, 60002.0])

            with (
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations",
                    return_value=np.array([10.0, np.nan, np.nan]),
                ),
                self.assertRaisesRegex(
                    ValueError, "exceeding max_reference_fallback_tiles=1.*202, 303"
                ),
            ):
                sim.load_temperature_table()

    def test_reference_fallback_does_not_mask_complete_paf_failure(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            missing_paf_dir = tmp_path / "missing"
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    paf_temperature_data_dir=str(missing_paf_dir),
                    temperature_fallback="reference",
                    max_reference_fallback_tiles=1,
                )
            )
            sim._cache_dir = lambda: tmp_path / "cache"
            sim.tile_sbids = np.array([101], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0])

            with self.assertRaisesRegex(FileNotFoundError, "missing directory"):
                sim.load_temperature_table()

    def test_reference_fallback_rejects_all_nonfinite_paf_lookup(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paf_dir = tmp_path / "paf"
            paf_dir.mkdir()
            _write_paf_csv(paf_dir / "ak01.csv", [])
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    paf_temperature_data_dir=str(paf_dir),
                    temperature_fallback="reference",
                    max_reference_fallback_tiles=2,
                )
            )
            sim._cache_dir = lambda: tmp_path / "cache"
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0])

            with (
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations",
                    return_value=np.array([np.nan, np.nan]),
                ),
                self.assertRaisesRegex(ValueError, "no finite temperatures"),
            ):
                sim.load_temperature_table()

    def test_reference_cache_reuse_and_config_invalidation(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paf_dir = tmp_path / "paf"
            cache_dir = tmp_path / "cache"
            paf_dir.mkdir()
            _write_paf_csv(paf_dir / "ak01.csv", [])
            base_kwargs = {
                "flux_min": 15.0,
                "paf_temperature_data_dir": str(paf_dir),
                "temperature_fallback": "reference",
                "max_reference_fallback_tiles": 1,
            }
            sim = RacsLow3(RacsLow3Config(**base_kwargs, paf_reference_temp_c=25.0))
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0])
            sim.tile_temperature_by_index = np.array([10.0, 25.0])
            sim.save_temperature_lookup(
                tile_temperature_sources=np.array(["mean_paf", "reference"])
            )
            with patch.object(sim, "build_temperature_map"):
                self.assertTrue(sim.load_temperature_lookup())

            changed = RacsLow3(
                RacsLow3Config(**base_kwargs, paf_reference_temp_c=24.0)
            )
            changed._cache_dir = lambda: cache_dir
            changed.tile_sbids = sim.tile_sbids
            changed.tile_scan_start_mjd = sim.tile_scan_start_mjd
            with patch.object(changed, "build_temperature_map"):
                self.assertFalse(changed.load_temperature_lookup())

            self.assertEqual(
                sorted(path.name for path in cache_dir.glob("temperature_lookup*.npz")),
                ["temperature_lookup_nside64.npz"],
            )

    def test_reference_cache_does_not_mask_unavailable_paf_source(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paf_dir = tmp_path / "paf"
            cache_dir = tmp_path / "cache"
            paf_dir.mkdir()
            paf_path = paf_dir / "ak01.csv"
            _write_paf_csv(paf_path, [])
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    paf_temperature_data_dir=str(paf_dir),
                    temperature_fallback="reference",
                    max_reference_fallback_tiles=1,
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0])
            sim.tile_temperature_by_index = np.array([10.0, 25.0])
            sim.save_temperature_lookup(
                tile_temperature_sources=np.array(["mean_paf", "reference"])
            )

            paf_path.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "temperature files"):
                sim.load_temperature_table()

    def test_load_temperature_table_falls_back_to_open_meteo_when_configured(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            n_pix = hp.nside2npix(64)
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    temperature_fallback="open_meteo",
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[:2] = np.array([101, 202], dtype=np.int32)
            sim._tile_index_from_sbid = {101: 0, 202: 1}
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[:2] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_starts[1] = 1
            sim.sbid_mixture_tile_indices = np.array([0, 1], dtype=np.int32)
            sim.sbid_mixture_probabilities = np.array([1.0, 1.0], dtype=np.float64)

            with (
                self.assertLogs("catsim.racs", level="WARNING") as logs,
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations",
                    return_value=np.array([np.nan, np.nan], dtype=np.float64),
                ) as paf_lookup,
                patch(
                    "catsim.racs.get_open_meteo_temperatures_for_mjd",
                    return_value=np.array([20.0, 21.0], dtype=np.float64),
                ) as meteo_lookup,
            ):
                sim.load_temperature_table()

            paf_lookup.assert_called_once()
            meteo_lookup.assert_called_once()
            self.assertIn("falling back to Open-Meteo", logs.output[0])
            self.assertIn("no finite temperatures", logs.output[0])
            np.testing.assert_allclose(
                sim.tile_temperature_by_index,
                np.array([20.0, 21.0], dtype=np.float64),
            )
            self.assertTrue(
                (cache_dir / "temperature_lookup_nside64.npz").exists()
            )
            self.assertFalse(
                (cache_dir / "temperature_lookup_nside64_mean_paf.npz").exists()
            )

            with np.load(cache_dir / "temperature_lookup_nside64.npz") as data:
                np.testing.assert_array_equal(
                    data["tile_temperature_sources"],
                    np.array(["open_meteo", "open_meteo"]),
                )
                np.testing.assert_array_equal(data["tile_sbids"], sim.tile_sbids)

    def test_load_temperature_table_fills_only_missing_paf_temperatures(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            paf_dir = tmp_path / "paf"
            cache_dir.mkdir()
            paf_dir.mkdir()
            _write_paf_csv(paf_dir / "ak01.csv", [])
            n_pix = hp.nside2npix(64)
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=str(paf_dir),
                    temperature_fallback="open_meteo",
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202, 303], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array(
                [60000.0, 60001.0, 60002.0],
                dtype=np.float64,
            )
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[:3] = sim.tile_sbids
            sim._tile_index_from_sbid = {101: 0, 202: 1, 303: 2}
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[:3] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_starts[:3] = np.arange(3)
            sim.sbid_mixture_tile_indices = np.arange(3, dtype=np.int32)
            sim.sbid_mixture_probabilities = np.ones(3, dtype=np.float64)

            with (
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations",
                    return_value=np.array([10.0, np.nan, 30.0], dtype=np.float64),
                ),
                patch(
                    "catsim.racs.get_open_meteo_temperatures_for_mjd",
                    return_value=np.array([21.0], dtype=np.float64),
                ) as meteo_lookup,
            ):
                sim.load_temperature_table()

            np.testing.assert_array_equal(
                meteo_lookup.call_args.args[0],
                np.array([60001.0], dtype=np.float64),
            )
            np.testing.assert_allclose(
                sim.tile_temperature_by_index,
                np.array([10.0, 21.0, 30.0], dtype=np.float64),
            )
            hybrid_cache = cache_dir / "temperature_lookup_nside64.npz"
            self.assertTrue(hybrid_cache.exists())
            self.assertFalse(
                (cache_dir / "temperature_lookup_nside64_open_meteo.npz").exists()
            )
            with np.load(hybrid_cache) as data:
                np.testing.assert_array_equal(
                    data["tile_temperature_sources"],
                    np.array(["mean_paf", "open_meteo", "mean_paf"]),
                )
                self.assertEqual(str(data["paf_temperature_data_dir"]), str(paf_dir))
                np.testing.assert_array_equal(
                    data["tile_scan_start_mjd"],
                    sim.tile_scan_start_mjd,
                )

    def test_cached_open_meteo_does_not_preempt_newly_available_paf_data(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            paf_dir = tmp_path / "paf"
            paf_dir.mkdir()
            _write_paf_csv(paf_dir / "ak01.csv", [])
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=str(paf_dir),
                    temperature_fallback="open_meteo",
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)
            sim.tile_temperature_by_index = np.array([90.0, 91.0], dtype=np.float64)
            sim.save_temperature_lookup(
                tile_temperature_sources=np.array(["open_meteo", "open_meteo"])
            )

            with (
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations",
                    return_value=np.array([10.0, 11.0], dtype=np.float64),
                ) as paf_lookup,
                patch(
                    "catsim.racs.get_open_meteo_temperatures_for_mjd"
                ) as meteo_lookup,
                patch.object(sim, "build_temperature_map"),
            ):
                sim.load_temperature_table()

            paf_lookup.assert_called_once()
            meteo_lookup.assert_not_called()
            np.testing.assert_allclose(
                sim.tile_temperature_by_index,
                np.array([10.0, 11.0], dtype=np.float64),
            )
            self.assertTrue(
                (cache_dir / "temperature_lookup_nside64.npz").exists()
            )

    def test_load_temperature_table_reuses_matching_hybrid_cache(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            paf_dir = tmp_path / "paf"
            paf_dir.mkdir()
            _write_paf_csv(paf_dir / "ak01.csv", [])
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=str(paf_dir),
                    temperature_fallback="open_meteo",
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)
            sim.tile_temperature_by_index = np.array([10.0, 21.0], dtype=np.float64)
            sim.save_temperature_lookup(
                tile_temperature_sources=np.array(["mean_paf", "open_meteo"]),
            )
            sim.tile_temperature_by_index = None

            with (
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations"
                ) as paf_lookup,
                patch(
                    "catsim.racs.get_open_meteo_temperatures_for_mjd"
                ) as meteo_lookup,
                patch.object(sim, "build_temperature_map"),
            ):
                sim.load_temperature_table()

            paf_lookup.assert_not_called()
            meteo_lookup.assert_not_called()
            np.testing.assert_allclose(
                sim.tile_temperature_by_index,
                np.array([10.0, 21.0], dtype=np.float64),
            )

    def test_load_temperature_lookup_rejects_malformed_metadata(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            paf_dir = tmp_path / "paf"
            cache_dir.mkdir()
            paf_dir.mkdir()
            _write_paf_csv(paf_dir / "ak01.csv", [])
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    paf_temperature_data_dir=str(paf_dir),
                    temperature_fallback="open_meteo",
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)
            sim.tile_temperature_by_index = np.array([10.0, 21.0])
            sim.save_temperature_lookup(
                tile_temperature_sources=np.array(["mean_paf", "open_meteo"])
            )
            cache_path = sim._temperature_lookup_cache_path()
            with np.load(cache_path) as data:
                base_payload = {name: data[name] for name in data.files}

            malformed_values = (
                ("missing format version", "format_version", None),
                ("wrong format version", "format_version", np.asarray(999)),
                ("missing provenance", "tile_temperature_sources", None),
                (
                    "wrong provenance shape",
                    "tile_temperature_sources",
                    np.array(["mean_paf"]),
                ),
                (
                    "unknown provenance",
                    "tile_temperature_sources",
                    np.array(["mean_paf", "unknown"]),
                ),
            )

            for name, field, value in malformed_values:
                with self.subTest(name=name):
                    payload = dict(base_payload)
                    if value is None:
                        payload.pop(field)
                    else:
                        payload[field] = value
                    np.savez_compressed(cache_path, **payload)
                    self.assertFalse(sim.load_temperature_lookup())

    def test_load_temperature_table_uses_cached_open_meteo_fallback(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            n_pix = hp.nside2npix(64)
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    temperature_fallback="open_meteo",
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[:2] = np.array([101, 202], dtype=np.int32)
            sim._tile_index_from_sbid = {101: 0, 202: 1}
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[:2] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_starts[1] = 1
            sim.sbid_mixture_tile_indices = np.array([0, 1], dtype=np.int32)
            sim.sbid_mixture_probabilities = np.array([1.0, 1.0], dtype=np.float64)
            sim.tile_temperature_by_index = np.array([20.0, 21.0], dtype=np.float64)
            sim.temperature_map = np.full(n_pix, np.nan, dtype=np.float32)
            sim.temperature_map[:2] = np.array([20.0, 21.0], dtype=np.float32)
            sim.save_temperature_lookup(
                tile_temperature_sources=np.array(["open_meteo", "open_meteo"])
            )

            with (
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations",
                    side_effect=FileNotFoundError("no paf"),
                ) as paf_lookup,
                patch("catsim.racs.get_open_meteo_temperatures_for_mjd") as meteo_lookup,
            ):
                sim.load_temperature_table()

            paf_lookup.assert_called_once()
            meteo_lookup.assert_not_called()
            np.testing.assert_allclose(
                sim.temperature_map[:2],
                np.array([20.0, 21.0], dtype=np.float32),
            )

    def test_load_temperature_table_rejects_nan_open_meteo_fallback(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            sim = RacsLow3(
                RacsLow3Config(
                    flux_min=15.0,
                    nside=64,
                    chunk_size=16,
                    temperature_fallback="open_meteo",
                )
            )
            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)

            with (
                patch(
                    "catsim.racs.get_mean_paf_temperatures_for_observations",
                    side_effect=FileNotFoundError("no paf"),
                ),
                patch(
                    "catsim.racs.get_open_meteo_temperatures_for_mjd",
                    return_value=np.array([20.0, np.nan], dtype=np.float64),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Open-Meteo.*202"):
                    sim.load_temperature_table()

            self.assertFalse(
                (cache_dir / "temperature_lookup_nside64.npz").exists()
            )

    def test_mid1_open_meteo_temperature_lookup_path_is_product_specific(self):
        sim = Racs(RacsConfig(product=RACS_MID1, flux_min=15.0))

        self.assertIn(
            "racs_mid1/lookups/temperature_lookup_nside64.npz",
            sim._temperature_lookup_cache_path().as_posix(),
        )

    def test_load_temperature_table_uses_cached_paf_lookup_when_present(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            n_pix = hp.nside2npix(64)
            sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))

            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[:2] = np.array([101, 202], dtype=np.int32)
            sim._tile_index_from_sbid = {101: 0, 202: 1}
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[:2] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_starts[1] = 1
            sim.sbid_mixture_tile_indices = np.array([0, 1], dtype=np.int32)
            sim.sbid_mixture_probabilities = np.array([1.0, 1.0], dtype=np.float64)
            sim.tile_temperature_by_index = np.array([20.0, 21.0], dtype=np.float64)
            sim.save_temperature_lookup()

            with patch("catsim.racs.get_mean_paf_temperatures_for_observations") as mocked_lookup:
                sim.load_temperature_table()

            mocked_lookup.assert_not_called()
            np.testing.assert_allclose(
                sim.temperature_map[:2],
                np.array([20.0, 21.0], dtype=np.float32),
            )

    def test_load_temperature_table_rejects_cached_nan_temperature(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            n_pix = hp.nside2npix(64)
            sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))

            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_scan_start_mjd = np.array([60000.0, 60001.0], dtype=np.float64)
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[:2] = np.array([101, 202], dtype=np.int32)
            sim._tile_index_from_sbid = {101: 0, 202: 1}
            sim.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_counts[:2] = 1
            sim.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            sim.sbid_mixture_starts[1] = 1
            sim.sbid_mixture_tile_indices = np.array([0, 1], dtype=np.int32)
            sim.sbid_mixture_probabilities = np.array([1.0, 1.0], dtype=np.float64)
            sim.tile_temperature_by_index = np.array([20.0, 21.0], dtype=np.float64)
            sim.save_temperature_lookup()
            cache_path = sim._temperature_lookup_cache_path()
            with np.load(cache_path) as data:
                payload = {name: data[name] for name in data.files}
            payload["tile_temperature_by_index"] = np.array([20.0, np.nan])
            np.savez_compressed(cache_path, **payload)

            with self.assertRaisesRegex(ValueError, "non-finite temperatures.*202"):
                sim.load_temperature_table()

    def test_pickle_excludes_catalogue_payload(self):
        sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))
        large_catalogue = Table(
            {
                "RA": np.linspace(0.0, 359.0, 20_000, dtype=np.float64),
                "Dec": np.linspace(-89.0, 89.0, 20_000, dtype=np.float64),
                "Total_flux": np.linspace(1.0, 10.0, 20_000, dtype=np.float64),
            }
        )

        sim.catalogue = large_catalogue
        sim.catalogue_is_loaded = True
        raw_state_payload = len(pickle.dumps(sim.__dict__, protocol=pickle.HIGHEST_PROTOCOL))
        object_payload = len(pickle.dumps(sim, protocol=pickle.HIGHEST_PROTOCOL))

        sim.release_catalogue()
        payload_after_release = len(pickle.dumps(sim, protocol=pickle.HIGHEST_PROTOCOL))

        self.assertFalse(hasattr(sim, "catalogue"))
        self.assertEqual(payload_after_release, object_payload)
        self.assertLess(object_payload, raw_state_payload)
        self.assertLess(object_payload, raw_state_payload // 10)


if __name__ == "__main__":
    unittest.main()
