import os
from pathlib import Path
import subprocess
import sys
import unittest

import healpy as hp
import numpy as np

from catsim import RACS_MID1, Racs, RacsConfig, RacsLow3, RacsLow3Config, batch_simulate
from catsim.utils.rng import prng_key

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - depends on optional extra availability.
    jax = None
    jnp = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _racs_config(**kwargs) -> RacsConfig:
    """Use external noisemaps only when a test needs to build missing caches."""
    configured = os.environ.get("RACS_NOISEMAP_DATA_DIR")
    default = Path.home() / "catalogue_data" / "racs" / "noisemaps"
    if configured is None and default.is_dir():
        configured = str(default)
    return RacsConfig(noisemap_data_dir=configured, **kwargs)


def _coarsen_nested_count_maps(
    maps: np.ndarray,
    *,
    input_nside: int,
    output_nside: int,
) -> np.ndarray:
    if input_nside % output_nside != 0:
        raise ValueError("output_nside must divide input_nside.")
    ratio = input_nside // output_nside
    child_pixels_per_parent = ratio * ratio
    output_pixels = np.arange(hp.nside2npix(input_nside)) // child_pixels_per_parent
    return np.stack(
        [
            np.bincount(
                output_pixels,
                weights=np.nan_to_num(density_map, nan=0.0),
                minlength=hp.nside2npix(output_nside),
            )
            for density_map in maps
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _centered_mean_map_residual_rms(
    left_maps: np.ndarray,
    right_maps: np.ndarray,
) -> tuple[float, float]:
    valid_pixels = (
        np.any(np.isfinite(left_maps), axis=0)
        & np.any(np.isfinite(right_maps), axis=0)
    )
    if not np.any(valid_pixels):
        raise ValueError("No common finite pixels are available for residual comparison.")
    left_mean = np.nanmean(left_maps[:, valid_pixels], axis=0)
    right_mean = np.nanmean(right_maps[:, valid_pixels], axis=0)
    residual = left_mean - right_mean
    raw_mean_residual = float(np.nanmean(residual))
    centered_residual = residual - raw_mean_residual
    return raw_mean_residual, float(np.sqrt(np.nanmean(centered_residual**2)))


def _mean_count_over_finite_pixels(maps: np.ndarray) -> float:
    valid_pixels = np.any(np.isfinite(maps), axis=0)
    if not np.any(valid_pixels):
        raise ValueError("No finite pixels are available.")
    return float(np.mean(np.nanmean(maps[:, valid_pixels], axis=0)))


def _numpy_split_null_residual_rms(
    numpy_maps: np.ndarray,
    *,
    group_size: int,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        permutation = rng.permutation(numpy_maps.shape[0])
        left = numpy_maps[permutation[:group_size]]
        right = numpy_maps[permutation[group_size : 2 * group_size]]
        _, out[i] = _centered_mean_map_residual_rms(left, right)
    return out


class RacsJaxImportTests(unittest.TestCase):
    def test_base_import_does_not_eagerly_import_jax(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_repo_root() / "src")
        env["MPLCONFIGDIR"] = "/tmp"
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import sys; import catsim; print('jax' in sys.modules)",
            ],
            cwd=_repo_root(),
            env=env,
            text=True,
        )
        self.assertEqual(output.strip(), "False")

    def test_racs_jax_missing_dependency_message_is_clear(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_repo_root() / "src")
        env["MPLCONFIGDIR"] = "/tmp"
        code = """
import builtins
import catsim

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "jax" or name.startswith("jax."):
        raise ModuleNotFoundError("No module named 'jax'", name="jax")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
try:
    catsim.RacsLow3Jax
except ImportError as exc:
    print(str(exc))
"""
        output = subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=_repo_root(),
            env=env,
            text=True,
        )
        self.assertIn("RacsLow3Jax requires the optional JAX dependencies", output)
        self.assertIn("catsim[jax]", output)


@unittest.skipIf(jax is None, "RacsLow3Jax requires the optional JAX dependencies.")
class RacsJaxNoiseLookupTests(unittest.TestCase):
    @staticmethod
    def _lookup_components():
        return (
            jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32),
            jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32),
            jnp.asarray([2, 0, 0, 3], dtype=jnp.int32),
            jnp.asarray([0, 2, 2, 2], dtype=jnp.int32),
            jnp.asarray([0, 0, 3, 3], dtype=jnp.int32),
            jnp.asarray([1.25, 1.5, 7.0, 8.0, 9.0], dtype=jnp.float32),
        )

    @staticmethod
    def _minimal_kernel_lookup(noise_map: np.ndarray) -> tuple:
        n_pix = hp.nside2npix(1)
        return (
            jnp.asarray([-1.0, 1.0], dtype=jnp.float32),
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.ones(n_pix, dtype=jnp.bool_),
            jnp.ones(n_pix, dtype=jnp.int32),
            jnp.zeros((n_pix, 1), dtype=jnp.int32),
            jnp.ones((n_pix, 1), dtype=jnp.float32),
            jnp.asarray(noise_map, dtype=jnp.float32),
            jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32),
            jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32),
            jnp.asarray([1, 0, 0, 0], dtype=jnp.int32),
            jnp.zeros(4, dtype=jnp.int32),
            jnp.zeros(4, dtype=jnp.int32),
            jnp.asarray([0.25], dtype=jnp.float32),
            jnp.zeros(n_pix, dtype=jnp.int32),
            jnp.zeros((n_pix, 1), dtype=jnp.float32),
            jnp.asarray([45.0], dtype=jnp.float32),
            jnp.asarray([25.0], dtype=jnp.float32),
        )

    def test_healpix_ang2pix_matches_configured_noise_nside_at_boundaries_and_random(self):
        from catsim.racs_jax import jax_ang2pix_nest_lonlat

        rng = np.random.default_rng(42)
        transition = np.rad2deg(np.arcsin(2.0 / 3.0))
        boundary_lon, boundary_lat = np.meshgrid(
            np.asarray(
                [0.001, 89.999, 90.001, 179.999, 180.001, 269.999, 270.001, 359.999],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    -89.999,
                    -transition - 0.01,
                    -transition + 0.01,
                    -0.001,
                    0.001,
                    transition - 0.01,
                    transition + 0.01,
                    89.999,
                ],
                dtype=np.float32,
            ),
        )
        lon = np.concatenate(
            [
                boundary_lon.reshape(-1),
                rng.uniform(0.0, 360.0, 256),
            ]
        ).astype(np.float32)
        lat = np.concatenate(
            [
                boundary_lat.reshape(-1),
                np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, 256))),
            ]
        ).astype(np.float32)
        nside = 256
        expected = hp.ang2pix(nside, lon, lat, lonlat=True, nest=True)
        actual = jax.jit(
            lambda query_lon, query_lat: jax_ang2pix_nest_lonlat(
                nside,
                query_lon,
                query_lat,
            )
        )(jnp.asarray(lon), jnp.asarray(lat))

        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_flat_ragged_absolute_error_sampling_is_jittable_and_vmappable(self):
        from catsim.racs_jax import _sample_absolute_flux_errors_jax

        lookup = self._lookup_components()
        noise = jnp.asarray([1.0, 2.0, 100.0, 1000.0], dtype=jnp.float32)
        flux = jnp.asarray([1.0, 2.0, 1.0, 10.0], dtype=jnp.float32)
        sample, cells, valid = jax.jit(_sample_absolute_flux_errors_jax)(
            jax.random.PRNGKey(8),
            noise,
            flux,
            *lookup,
        )

        np.testing.assert_array_equal(np.asarray(cells), [0, 0, 3, 3])
        self.assertTrue(np.all(np.asarray(valid)))
        self.assertTrue(set(np.asarray(sample[:2])).issubset({1.25, 1.5}))
        self.assertTrue(set(np.asarray(sample[2:])).issubset({7.0, 8.0, 9.0}))

        keys = jax.random.split(jax.random.PRNGKey(9), noise.size)
        vmapped = jax.jit(
            jax.vmap(
                lambda one_key, one_noise, one_flux: _sample_absolute_flux_errors_jax(
                    one_key,
                    one_noise,
                    one_flux,
                    *lookup,
                )[0]
            )
        )(keys, noise, flux)
        self.assertEqual(vmapped.shape, noise.shape)
        self.assertTrue(set(np.asarray(vmapped[:2])).issubset({1.25, 1.5}))
        self.assertTrue(set(np.asarray(vmapped[2:])).issubset({7.0, 8.0, 9.0}))

    def test_bounded_cell_resolution_and_sampling_match_numpy_lookup(self):
        from catsim.racs_jax import _sample_absolute_flux_errors_jax
        from catsim.racs_noise import build_conditional_error_lookup

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
        noise = np.asarray([100.0, 1000.0, 1.0, 1e6], dtype=np.float32)
        flux = np.asarray([0.1, 10_000.0, 1e-6, 1e9], dtype=np.float32)
        expected_cells = lookup.resolve_cells(noise, flux)
        samples, cells, valid = jax.jit(_sample_absolute_flux_errors_jax)(
            jax.random.PRNGKey(91),
            jnp.asarray(noise),
            jnp.asarray(flux),
            jnp.asarray(lookup.log_noise_edges, dtype=jnp.float32),
            jnp.asarray(lookup.log_flux_edges, dtype=jnp.float32),
            jnp.asarray(lookup.cell_counts, dtype=jnp.int32),
            jnp.asarray(lookup.cell_starts, dtype=jnp.int32),
            jnp.asarray(lookup.resolved_cell_ids, dtype=jnp.int32),
            jnp.asarray(lookup.absolute_error_values, dtype=jnp.float32),
        )

        np.testing.assert_array_equal(np.asarray(cells), expected_cells)
        np.testing.assert_array_equal(np.asarray(valid), np.ones(4, dtype=bool))
        np.testing.assert_array_equal(np.asarray(samples), [1.0, 2.0, 1.0, 2.0])

    def test_cell_resolution_absolute_eta_and_invalid_queries_match_numpy_semantics(self):
        from catsim.racs_jax import (
            _sample_absolute_flux_errors_jax,
            _scale_absolute_flux_errors_jax,
        )

        lookup = self._lookup_components()
        noise = jnp.asarray(
            [1.0, 1000.0, np.nan, hp.UNSEEN, 0.0, -1.0],
            dtype=jnp.float32,
        )
        flux = jnp.asarray([1.0, 10.0, 2.0, 2.0, 2.0, 2.0], dtype=jnp.float32)
        base_sigma, cells, valid = _sample_absolute_flux_errors_jax(
            jax.random.PRNGKey(10),
            noise,
            flux,
            *lookup,
        )
        effective_sigma = _scale_absolute_flux_errors_jax(
            base_sigma,
            jnp.asarray(3.0, dtype=jnp.float32),
        )

        np.testing.assert_array_equal(np.asarray(cells[:2]), [0, 3])
        np.testing.assert_array_equal(
            np.asarray(valid),
            [True, True, False, False, False, False],
        )
        np.testing.assert_allclose(
            np.asarray(effective_sigma[:2]),
            2.0 * np.asarray(base_sigma[:2]),
        )
        self.assertTrue(np.all(np.isnan(np.asarray(base_sigma[2:]))))
        # The absolute lookup sigma is independent of the queried flux scale.
        self.assertLessEqual(float(base_sigma[1]), 9.0)

    def test_invalid_noise_is_excluded_from_ordinary_and_summary_kernels(self):
        from catsim.racs_jax import (
            _simulate_one_jax,
            _simulate_one_jax_with_flux_summaries,
        )

        n_pix = hp.nside2npix(1)
        lookup = self._minimal_kernel_lookup(np.full(n_pix, np.nan, dtype=np.float32))
        common = dict(
            key=jax.random.PRNGKey(11),
            parent_count=jnp.asarray(4, dtype=jnp.int32),
            flux_min=jnp.asarray(0.01, dtype=jnp.float32),
            p_clus=jnp.asarray(0.0, dtype=jnp.float32),
            clus_stop_prob=jnp.asarray(1.0, dtype=jnp.float32),
            lambda_clus=jnp.asarray(0.0, dtype=jnp.float32),
            observer_beta=jnp.asarray(0.0, dtype=jnp.float32),
            forward_matrix=jnp.eye(3, dtype=jnp.float32),
            inverse_matrix=jnp.eye(3, dtype=jnp.float32),
            temp_beta=jnp.asarray(0.0, dtype=jnp.float32),
            elevation_amp=jnp.asarray(0.0, dtype=jnp.float32),
            elevation_trough=jnp.asarray(45.0, dtype=jnp.float32),
            fractional_error_eta=jnp.asarray(0.0, dtype=jnp.float32),
            lookup_tuple=lookup,
            cluster_model_code=0,
            nside=1,
            noise_map_nside=1,
            n_chunks=jnp.asarray(1, dtype=jnp.int32),
            chunk_size=4,
            max_children=0,
            alpha_mean=0.8,
            alpha_sigma=0.2,
            cluster_r0_arcsec=100.0,
            cluster_r_cut_arcsec=20.0,
            paf_reference_temp_c=25.0,
            temperature_model="hot_linear",
            use_elevation=False,
        )
        density, mask, rejected = _simulate_one_jax(**common)
        np.testing.assert_array_equal(np.asarray(density[mask]), 0.0)
        self.assertEqual(float(np.sum(np.asarray(rejected))), 4.0)

        summary_common = dict(common)
        summary_common.update(
            temperature_flux_min=jnp.asarray(0.01, dtype=jnp.float32),
            flux_max=jnp.asarray(10.0, dtype=jnp.float32),
            temperature_edges=jnp.asarray([0.0, 50.0], dtype=jnp.float32),
            temperature_quantiles=jnp.asarray([0.5], dtype=jnp.float32),
            elevation_edges=jnp.asarray([0.0, 90.0], dtype=jnp.float32),
            elevation_quantiles=jnp.asarray([0.5], dtype=jnp.float32),
            n_flux_bins=4,
            include_temperature=True,
            include_elevation=True,
        )
        density, mask, temperature_summary, elevation_summary, rejected = (
            _simulate_one_jax_with_flux_summaries(**summary_common)
        )
        np.testing.assert_array_equal(np.asarray(density[mask]), 0.0)
        np.testing.assert_array_equal(np.asarray(temperature_summary), 0.0)
        np.testing.assert_array_equal(np.asarray(elevation_summary), 0.0)
        self.assertEqual(float(np.sum(np.asarray(rejected))), 4.0)


@unittest.skipIf(jax is None, "RacsLow3Jax requires the optional JAX dependencies.")
class RacsJaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from catsim import RacsJax

        cfg = _racs_config(
            product=RACS_MID1,
            flux_min=0.001,
            chunk_size=4,
            store_final_samples=False,
            max_cluster_children_per_parent=4,
        )
        cls.sim = RacsJax(cfg)
        cls.sim.initialise_data()

    def test_healpix_ang2pix_matches_healpy_for_representative_points(self):
        from catsim.racs_jax import jax_ang2pix_nest_lonlat

        lon = np.array([0.1, 45.3, 90.2, 179.8, 359.7, 12.3, 210.4], dtype=np.float32)
        lat = np.array([0.2, 45.1, -45.2, 88.7, -89.5, 10.1, -20.2], dtype=np.float32)
        expected = hp.ang2pix(64, lon, lat, lonlat=True, nest=True)
        actual = np.asarray(
            jax_ang2pix_nest_lonlat(64, jnp.asarray(lon), jnp.asarray(lat))
        )

        np.testing.assert_array_equal(actual, expected)

    def test_initialisation_transfers_compact_shared_noise_error_cache(self):
        lookup = self.sim._lookup_arrays
        self.assertIsNotNone(lookup)
        self.assertEqual(
            lookup.noise_map.shape,
            (hp.nside2npix(self.sim.cfg.noise_map_nside),),
        )
        self.assertEqual(lookup.absolute_error_values.ndim, 1)
        self.assertEqual(lookup.error_cell_counts.ndim, 1)
        self.assertEqual(lookup.error_cell_counts.shape, lookup.error_cell_starts.shape)
        self.assertEqual(
            lookup.error_cell_counts.shape,
            lookup.error_resolved_cell_ids.shape,
        )
        self.assertFalse(hasattr(lookup, "error_values_by_pixel"))
        self.assertFalse(hasattr(lookup, "global_error_values"))

    def test_generate_dipole_returns_expected_shapes_and_is_deterministic(self):
        key = jax.random.PRNGKey(123)
        first_map, first_mask = self.sim.generate_dipole(np.log10(8.0), key=key)
        second_map, second_mask = self.sim.generate_dipole(np.log10(8.0), key=key)

        self.assertEqual(first_map.shape, (hp.nside2npix(64),))
        self.assertEqual(first_mask.shape, (hp.nside2npix(64),))
        self.assertEqual(first_map.dtype, np.float32)
        self.assertEqual(first_mask.dtype, np.bool_)
        np.testing.assert_array_equal(first_map, second_map)
        np.testing.assert_array_equal(first_mask, second_mask)
        self.assertEqual(
            self.sim.last_invalid_noise_rejection_maps.shape,
            (1, hp.nside2npix(64)),
        )
        np.testing.assert_array_equal(
            self.sim.last_invalid_noise_rejection_counts,
            np.sum(
                self.sim.last_invalid_noise_rejection_maps,
                axis=1,
                dtype=np.float64,
            ).astype(np.int64),
        )

    def test_low3_initialises_without_elevation_and_rejects_nonzero_amplitude(self):
        from catsim import RacsJax

        sim = RacsJax(
            _racs_config(
                flux_min=15.0,
                chunk_size=4,
                store_final_samples=False,
            )
        )
        sim.initialise_data()

        self.assertFalse(sim.elevation_is_available)
        self.assertIsNone(sim.elevation_lookup_values)
        density_map, mask = sim.generate_dipole(
            np.log10(8.0),
            key=jax.random.PRNGKey(321),
        )
        self.assertEqual(density_map.shape, mask.shape)

        with self.assertRaisesRegex(ValueError, "does not define an elevation column"):
            sim.generate_dipole(
                np.log10(8.0),
                elevation_amp=0.1,
                key=jax.random.PRNGKey(322),
            )

    def test_low3_rejects_elevation_flux_summary(self):
        from catsim import RacsJax

        sim = RacsJax(
            _racs_config(
                flux_min=15.0,
                chunk_size=4,
                store_final_samples=False,
            )
        )
        sim.initialise_data()

        with self.assertRaisesRegex(ValueError, "does not define an elevation column"):
            sim.batch_generate_dipole_with_flux_summaries(
                theta={"log10_n_initial_samples": np.asarray([np.log10(8.0)])},
                key=jax.random.PRNGKey(323),
                batch_size=1,
                elevation_edges=np.asarray([0.0, 90.0], dtype=np.float32),
                elevation_quantiles=(0.5,),
            )

    def test_batch_generate_dipole_returns_stacked_maps_and_masks(self):
        theta = {
            "log10_n_initial_samples": np.log10(np.array([4.0, 8.0])),
            "p_clus": np.array([0.0, 0.5]),
            "clus_stop_prob": np.array([1.0, 0.8]),
            "elevation_amp": np.array([0.0, 0.1]),
            "elevation_trough": np.array([45.0, 50.0]),
        }
        maps, masks = self.sim.batch_generate_dipole(
            theta,
            jax.random.PRNGKey(7),
            batch_size=2,
        )

        self.assertEqual(maps.shape, (2, hp.nside2npix(64)))
        self.assertEqual(masks.shape, (2, hp.nside2npix(64)))
        self.assertEqual(maps.dtype, np.float32)
        self.assertEqual(masks.dtype, np.bool_)

    def test_histogram_flux_quantiles_interpolates_raw_flux_values(self):
        from catsim.racs_jax import _histogram_flux_quantiles_jax

        hist = jnp.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=jnp.float32)
        features = _histogram_flux_quantiles_jax(
            hist,
            jnp.asarray(10.0, dtype=jnp.float32),
            jnp.asarray(1000.0, dtype=jnp.float32),
            jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32),
        )

        expected = np.asarray([10.0, 100.0, 1000.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(features), expected, rtol=1e-5)

    def test_temperature_models_have_numpy_jax_parity(self):
        from catsim.racs_temperature import evaluate_temperature_response

        temperatures = np.asarray([20.0, 25.0, 30.0, 80.0], dtype=np.float32)
        for model in ("hot_linear", "hot_quadratic"):
            expected = evaluate_temperature_response(
                temperatures,
                0.02,
                25.0,
                model=model,
                xp=np,
            )
            actual = jax.jit(
                lambda values: evaluate_temperature_response(
                    values,
                    0.02,
                    25.0,
                    model=model,
                    xp=jnp,
                )
            )(jnp.asarray(temperatures))

            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)

    def test_batch_generate_dipole_with_flux_temperature_summary_returns_features(self):
        temperatures = self.sim.tile_temperature_by_index
        self.assertIsNotNone(temperatures)
        finite_temperatures = temperatures[np.isfinite(temperatures)]
        self.assertGreater(finite_temperatures.size, 1)
        temperature_edges = np.linspace(
            float(np.min(finite_temperatures)),
            float(np.max(finite_temperatures)),
            4,
            dtype=np.float32,
        )

        maps, masks, summaries = self.sim.batch_generate_dipole_with_flux_temperature_summary(
            theta={
                "log10_n_initial_samples": np.log10(np.asarray([8.0, 8.0])),
                "observer_speed": np.zeros(2, dtype=np.float32),
                "p_clus": np.zeros(2, dtype=np.float32),
                "clus_stop_prob": np.ones(2, dtype=np.float32),
            },
            key=jax.random.PRNGKey(123),
            batch_size=2,
            temperature_edges=temperature_edges,
            quantiles=(0.25, 0.5),
            n_flux_bins=8,
            flux_max_mjy=1000.0,
        )

        self.assertEqual(maps.shape, (2, hp.nside2npix(64)))
        self.assertEqual(masks.shape, (2, hp.nside2npix(64)))
        self.assertEqual(summaries.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(summaries)))

    def test_hot_quadratic_runs_in_ordinary_and_summary_kernels(self):
        temperatures = self.sim.tile_temperature_by_index
        finite_temperatures = temperatures[np.isfinite(temperatures)]
        temperature_edges = np.asarray(
            [np.min(finite_temperatures), np.max(finite_temperatures)],
            dtype=np.float32,
        )
        original_model = self.sim.cfg.temperature_model
        self.sim.cfg.temperature_model = "hot_quadratic"
        try:
            density_map, mask = self.sim.generate_dipole(
                np.log10(8.0),
                temp_beta=0.02,
                key=jax.random.PRNGKey(811),
            )
            maps, masks, summaries = (
                self.sim.batch_generate_dipole_with_flux_temperature_summary(
                    theta={
                        "log10_n_initial_samples": np.asarray([np.log10(8.0)]),
                        "temp_beta": np.asarray([0.02]),
                    },
                    key=jax.random.PRNGKey(812),
                    batch_size=1,
                    temperature_edges=temperature_edges,
                    quantiles=(0.5,),
                    n_flux_bins=8,
                    flux_max_mjy=1000.0,
                )
            )
        finally:
            self.sim.cfg.temperature_model = original_model

        self.assertEqual(density_map.shape, mask.shape)
        self.assertEqual(maps.shape, masks.shape)
        self.assertEqual(summaries.shape, (1, 1))
        self.assertTrue(np.all(np.isfinite(summaries)))

    def test_batch_generate_dipole_with_combined_flux_summaries(self):
        temperatures = self.sim.tile_temperature_by_index
        elevations = self.sim.elevation_lookup_values
        temperature_edges = np.linspace(
            float(np.nanmin(temperatures)),
            float(np.nanmax(temperatures)),
            4,
            dtype=np.float32,
        )
        elevation_edges = np.linspace(
            float(np.nanmin(elevations)),
            float(np.nanmax(elevations)),
            5,
            dtype=np.float32,
        )
        theta = {
            "log10_n_initial_samples": np.log10(np.asarray([8.0, 8.0])),
            "observer_speed": np.zeros(2, dtype=np.float32),
            "p_clus": np.zeros(2, dtype=np.float32),
            "clus_stop_prob": np.ones(2, dtype=np.float32),
        }
        key = jax.random.PRNGKey(321)

        maps, masks, combined = self.sim.batch_generate_dipole_with_flux_summaries(
            theta,
            key,
            batch_size=2,
            temperature_edges=temperature_edges,
            temperature_quantiles=(0.25, 0.5),
            elevation_edges=elevation_edges,
            elevation_quantiles=(0.5,),
            n_flux_bins=8,
            flux_max_mjy=1000.0,
        )
        elevation_maps, elevation_masks, elevation_only = (
            self.sim.batch_generate_dipole_with_flux_summaries(
                theta,
                key,
                batch_size=2,
                elevation_edges=elevation_edges,
                elevation_quantiles=(0.5,),
                n_flux_bins=8,
                flux_max_mjy=1000.0,
            )
        )

        self.assertEqual(set(combined), {"temperature", "elevation"})
        self.assertEqual(combined["temperature"].shape, (2, 6))
        self.assertEqual(combined["elevation"].shape, (2, 4))
        np.testing.assert_array_equal(maps, elevation_maps)
        np.testing.assert_array_equal(masks, elevation_masks)
        np.testing.assert_allclose(combined["elevation"], elevation_only["elevation"])

    def test_batch_generate_dipole_with_flux_summaries_requires_complete_pair(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            self.sim.batch_generate_dipole_with_flux_summaries(
                {"log10_n_initial_samples": np.log10(np.asarray([8.0]))},
                jax.random.PRNGKey(1),
                batch_size=1,
                elevation_edges=np.asarray([0.0, 90.0]),
            )

    def test_batch_generate_dipole_accepts_dynamic_source_counts(self):
        low_maps, low_masks = self.sim.batch_generate_dipole(
            {"log10_n_initial_samples": np.full(2, 1.0)},
            jax.random.PRNGKey(701),
            batch_size=2,
        )
        high_maps, high_masks = self.sim.batch_generate_dipole(
            {"log10_n_initial_samples": np.full(2, 2.0)},
            jax.random.PRNGKey(702),
            batch_size=2,
        )

        self.assertEqual(low_maps.shape, high_maps.shape)
        self.assertEqual(low_masks.shape, high_masks.shape)
        np.testing.assert_array_equal(low_masks, high_masks)

    def test_batch_generate_dipole_rejects_invalid_elevation_amp(self):
        with self.assertRaisesRegex(ValueError, "elevation_amp must be finite"):
            self.sim.batch_generate_dipole(
                {
                    "log10_n_initial_samples": np.full(1, 1.0),
                    "elevation_amp": np.array([-0.1], dtype=np.float32),
                },
                jax.random.PRNGKey(704),
                batch_size=1,
            )

    def test_batch_generate_dipole_pads_final_host_batch(self):
        maps, masks = self.sim.batch_generate_dipole(
            {"log10_n_initial_samples": np.full(3, 1.0)},
            jax.random.PRNGKey(703),
            batch_size=2,
        )

        self.assertEqual(maps.shape, (3, hp.nside2npix(64)))
        self.assertEqual(masks.shape, (3, hp.nside2npix(64)))

    def test_geometric_clustering_path_runs(self):
        density_map, mask = self.sim.generate_dipole(
            np.log10(8.0),
            p_clus=1.0,
            clus_stop_prob=1.0,
            key=jax.random.PRNGKey(11),
        )

        self.assertEqual(density_map.shape, mask.shape)
        self.assertTrue(np.all(np.isfinite(density_map[mask])))

    def test_poisson_clustering_path_runs(self):
        from catsim import RacsJax

        cfg = _racs_config(
            product=RACS_MID1,
            flux_min=0.001,
            chunk_size=4,
            store_final_samples=False,
            cluster_count_model="poisson",
            max_cluster_children_per_parent=6,
        )
        sim = RacsJax(cfg)
        sim.initialise_data()

        density_map, mask = sim.generate_dipole(
            np.log10(8.0),
            lambda_clus=0.5,
            key=jax.random.PRNGKey(12),
        )

        self.assertEqual(density_map.shape, mask.shape)
        self.assertTrue(np.all(np.isfinite(density_map[mask])))

    def test_overfill_probability_warning_is_explicit(self):
        from catsim import RacsJax

        cfg = _racs_config(
            product=RACS_MID1,
            flux_min=0.001,
            chunk_size=4,
            store_final_samples=False,
            max_cluster_children_per_parent=2,
        )
        sim = RacsJax(cfg)
        sim.initialise_data()

        with self.assertWarnsRegex(
            RuntimeWarning,
            r"P\(children_per_parent > 2\).*geometric",
        ):
            sim.generate_dipole(
                np.log10(4.0),
                p_clus=1.0,
                clus_stop_prob=0.5,
                key=jax.random.PRNGKey(13),
            )

    def test_store_final_samples_is_not_supported(self):
        from catsim import RacsJax

        cfg = _racs_config(
            product=RACS_MID1,
            flux_min=0.001,
            chunk_size=4,
            store_final_samples=True,
        )
        sim = RacsJax(cfg)
        sim.initialise_data()

        with self.assertRaisesRegex(NotImplementedError, "store_final_samples=True"):
            sim.generate_dipole(np.log10(4.0), key=jax.random.PRNGKey(14))

    def test_wrong_clustering_arguments_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "lambda_clus is only valid"):
            self.sim.generate_dipole(
                np.log10(4.0),
                lambda_clus=0.5,
                key=jax.random.PRNGKey(15),
            )

    def test_integration_mean_counts_and_residual_rms_match_numpy_reference(self):
        """Compare JAX mean maps to a NumPy-only finite-sample null.

        The observed statistic is the RMS of the centered residual between the
        100-simulation JAX mean map and a 100-simulation NumPy mean map. The
        null distribution is the same statistic computed from random 100-vs-100
        splits of 200 independent NumPy reference maps.
        """
        from catsim import RacsJax

        n_numpy = 200
        n_jax = 100
        log10_n = 6.0

        cfg = _racs_config(
            product=RACS_MID1,
            flux_min=15,
            chunk_size=1_000,
            store_final_samples=False,
            max_cluster_children_per_parent=4,
        )

        numpy_sim = Racs(cfg)
        numpy_sim.initialise_data()
        numpy_maps, numpy_masks = batch_simulate(
            theta={
                "log10_n_initial_samples": np.full(n_numpy, log10_n, dtype=np.float32),
                "observer_speed": np.zeros(n_numpy, dtype=np.float32),
                "p_clus": np.zeros(n_numpy, dtype=np.float32),
                "clus_stop_prob": np.ones(n_numpy, dtype=np.float32),
            },
            model_callable=numpy_sim.generate_dipole,
            n_workers=12,
            rng_key=prng_key(20_000),
        )

        jax_sim = RacsJax(cfg)
        jax_sim.initialise_data()
        jax_maps, jax_masks = jax_sim.batch_generate_dipole(
            theta={
                "log10_n_initial_samples": np.full(n_jax, log10_n, dtype=np.float32),
                "observer_speed": np.zeros(n_jax, dtype=np.float32),
                "p_clus": np.zeros(n_jax, dtype=np.float32),
                "clus_stop_prob": np.ones(n_jax, dtype=np.float32),
            },
            key=jax.random.PRNGKey(30_000),
            batch_size=10,
        )

        np.testing.assert_array_equal(numpy_masks[0], numpy_sim.mask_map)
        np.testing.assert_array_equal(jax_masks[0], numpy_sim.mask_map)

        numpy_total_counts = np.nansum(numpy_maps[:n_jax], axis=1)
        jax_total_counts = np.nansum(jax_maps, axis=1)
        numpy_total_mean = float(np.mean(numpy_total_counts))
        jax_total_mean = float(np.mean(jax_total_counts))
        relative_total_count_delta = abs(jax_total_mean - numpy_total_mean) / max(
            numpy_total_mean,
            1.0,
        )
        self.assertLess(
            relative_total_count_delta,
            0.05,
            msg=(
                "JAX and NumPy total mean counts differ by "
                f"{relative_total_count_delta:.3g}; "
                f"numpy_mean={numpy_total_mean:.3f}, jax_mean={jax_total_mean:.3f}"
            ),
        )

        raw_mean_residual, observed_rms = _centered_mean_map_residual_rms(
            jax_maps,
            numpy_maps[:n_jax],
        )
        mean_pixel_count = _mean_count_over_finite_pixels(numpy_maps[:n_jax])
        self.assertLess(
            abs(raw_mean_residual),
            0.05 * max(mean_pixel_count, 1.0),
            msg=(
                "Mean JAX-minus-NumPy native-pixel residual is too large; "
                f"raw_mean_residual={raw_mean_residual:.3g}, "
                f"mean_pixel_count={mean_pixel_count:.3g}"
            ),
        )

        null_rms = _numpy_split_null_residual_rms(
            numpy_maps,
            group_size=n_jax,
            n_resamples=100,
            seed=40_000,
        )
        null_95 = float(np.percentile(null_rms, 95.0))
        self.assertLessEqual(
            observed_rms,
            1.5 * null_95,
            msg=(
                "JAX-vs-NumPy centered residual RMS exceeds the NumPy-only "
                "split-half null tolerance; "
                f"observed_rms={observed_rms:.3g}, null_95={null_95:.3g}"
            ),
        )
