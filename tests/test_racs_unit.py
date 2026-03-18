import unittest

import numpy as np

from catsim import RacsLow3, RacsLow3Config


class RacsFluxErrorTests(unittest.TestCase):
    def setUp(self):
        self.sim = RacsLow3(RacsLow3Config(flux_min=15.0, nside=1, chunk_size=16))

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
        sim.mask_map = np.ones(12, dtype=bool)
        sim.tile_lookup_map = np.zeros(12, dtype=np.int32)
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
            np.arange(ra.shape[0], dtype=np.int64) % sim.mask_map.size,
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


if __name__ == "__main__":
    unittest.main()
