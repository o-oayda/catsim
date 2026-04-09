import unittest
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import types
from unittest.mock import patch

from astropy.time import Time
from astropy.table import Table
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

from catsim import RacsLow3, RacsLow3Config
from catsim.racs import LOW3_TEMPERATURE_EPSILON_FLOOR
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


def _make_full_antenna_set(tmp_path: Path, minute_offset: int = 0) -> None:
    base_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=minute_offset)
    for antenna_index in range(1, 37):
        antenna_name = f"ak{antenna_index:02d}"
        _write_paf_csv(
            tmp_path / f"{antenna_name} ctrl_adc1_pafAvTemp-data.csv",
            [
                (base_time.strftime("%Y-%m-%d %H:%M:%S"), f"{antenna_index:.1f}"),
                (
                    (base_time + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                    f"{antenna_index + 100:.1f}",
                ),
            ],
        )


class RacsFluxErrorTests(unittest.TestCase):
    def setUp(self):
        self.sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))

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

    def test_compute_total_flux_error_rejects_negative_eta(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            self.sim.compute_total_flux_error(
                np.array([100.0]),
                np.array([0.1]),
                fractional_error_eta=-0.1,
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
        sim.lookups_are_initialised = True
        n_pix = hp.nside2npix(sim.nside)
        sim.mask_map = np.ones(n_pix, dtype=bool)
        sim.tile_lookup_map = np.zeros(n_pix, dtype=np.int32)
        sim.tile_temperature_by_index = np.array([30.0], dtype=np.float64)

        n_samples = 10
        base_fractional_error = np.full(n_samples, 0.1, dtype=np.float32)

        sim.sample_fluxes = lambda n, rng=None: np.full(n, 100.0, dtype=np.float64)
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
        sim.assign_tiles = lambda ra, dec: np.zeros(ra.shape[0], dtype=np.int32)
        sim.evaluate_temperature_enhancement = lambda tile_indices, temp_slope, temp_intercept, temp_pivot_c: (
            np.ones(tile_indices.shape[0], dtype=np.float64),
            np.full(tile_indices.shape[0], 30.0, dtype=np.float32),
        )
        sim.apply_temperature_enhancement = lambda flux, enhancement, dtype=np.float64: (
            np.asarray(flux, dtype=dtype)
        )
        sim.sample_fractional_errors = lambda pixel_indices, rng=None: base_fractional_error[
            : pixel_indices.shape[0]
        ]

        dmap, mask = sim.generate_dipole(
            log10_n_initial_samples=1.0,
            fractional_error_eta=3.0,
            temp_slope=0.0,
            temp_intercept=1.0,
        )

        expected_effective = np.full(n_samples, 0.2, dtype=np.float32)
        np.testing.assert_allclose(sim.final_base_fractional_error_samples, base_fractional_error)
        np.testing.assert_allclose(sim.final_fractional_error_samples, expected_effective)

        sampled_map = sim.sampled_fractional_error_map
        self.assertIsNotNone(sampled_map)
        finite = np.isfinite(sampled_map)
        self.assertTrue(np.any(finite))
        np.testing.assert_allclose(sampled_map[finite], np.full(np.count_nonzero(finite), 0.2))
        self.assertEqual(dmap.shape, mask.shape)

    def test_evaluate_temperature_enhancement_clips_to_positive_floor(self):
        self.sim.tile_temperature_by_index = np.array([60.0], dtype=np.float64)

        enhancement, temperatures = self.sim.evaluate_temperature_enhancement(
            tile_indices=np.array([0], dtype=np.int32),
            temp_slope=-1.0,
            temp_intercept=0.1,
            temp_pivot_c=30.0,
        )

        self.assertEqual(temperatures[0], np.float32(60.0))
        self.assertEqual(enhancement[0], np.asarray(LOW3_TEMPERATURE_EPSILON_FLOOR, dtype=self.sim.dtype))

    def test_evaluate_temperature_enhancement_rejects_non_positive_pivot(self):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            self.sim.evaluate_temperature_enhancement(
                tile_indices=np.array([0], dtype=np.int32),
                temp_slope=0.0,
                temp_intercept=1.0,
                temp_pivot_c=0.0,
            )

    def test_generate_dipole_remains_finite_when_linear_enhancement_hits_floor(self):
        sim = self.sim
        sim.lookups_are_initialised = True
        n_pix = hp.nside2npix(sim.nside)
        sim.mask_map = np.ones(n_pix, dtype=bool)
        sim.tile_lookup_map = np.zeros(n_pix, dtype=np.int32)
        sim.tile_temperature_by_index = np.array([60.0], dtype=np.float64)

        n_samples = 8
        sim.sample_fluxes = lambda n, rng=None: np.full(n, 100.0, dtype=np.float64)
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
        sim.assign_tiles = lambda ra, dec: np.zeros(ra.shape[0], dtype=np.int32)
        sim.sample_fractional_errors = lambda pixel_indices, rng=None: np.full(
            pixel_indices.shape[0],
            0.1,
            dtype=np.float32,
        )

        dmap, mask = sim.generate_dipole(
            log10_n_initial_samples=np.log10(float(n_samples)),
            fractional_error_eta=0.0,
            temp_slope=-1.0,
            temp_intercept=0.1,
            temp_pivot_c=30.0,
        )

        self.assertEqual(dmap.shape, mask.shape)
        self.assertTrue(np.all(np.isfinite(sim.final_observed_flux_samples)))
        self.assertTrue(np.all(np.isfinite(sim.final_flux_error_samples)))
        self.assertTrue(np.all(sim.final_flux_error_samples >= 0.0))


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


class RacsInitialiseDataTests(unittest.TestCase):
    def test_initialise_data_uses_cached_lookups_without_loading_catalogue(self):
        with TemporaryDirectory() as tmpdir:
            sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))
            cache_dir = Path(tmpdir)
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
            sim.save_tile_lookup()
            self.assertTrue((cache_dir / "sbid_lookup_nside64.png").exists())

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
            self.assertTrue((cache_dir / "temperature_lookup_nside64_mean_paf.png").exists())

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
            self.assertTrue((cache_dir / "temperature_lookup_nside64_mean_paf.npz").exists())
            self.assertTrue((cache_dir / "temperature_lookup_nside64_mean_paf.png").exists())

    def test_load_temperature_table_uses_cached_paf_lookup_when_present(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            n_pix = hp.nside2npix(64)
            sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=64, chunk_size=16))

            sim._cache_dir = lambda: cache_dir
            sim.tile_sbids = np.array([101, 202], dtype=np.int32)
            sim.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
            sim.tile_lookup_map[:2] = np.array([101, 202], dtype=np.int32)
            sim._tile_index_from_sbid = {101: 0, 202: 1}
            sim.tile_temperature_by_index = np.array([20.0, 21.0], dtype=np.float64)
            sim.save_temperature_lookup()

            with patch("catsim.racs.get_mean_paf_temperatures_for_observations") as mocked_lookup:
                sim.load_temperature_table()

            mocked_lookup.assert_not_called()
            np.testing.assert_allclose(
                sim.temperature_map[:2],
                np.array([20.0, 21.0], dtype=np.float32),
            )

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
