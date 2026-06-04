import os
from pathlib import Path
import subprocess
import sys
import unittest

import healpy as hp
import numpy as np

from catsim import Catwise, CatwiseConfig, batch_simulate
from catsim.utils.rng import prng_key

try:
    import jax
except ImportError:  # pragma: no cover - depends on optional extra availability.
    jax = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


class _FakeAlphaLookup:
    def fit_alpha(self, w12_colour, out=None):
        colour = np.asarray(w12_colour, dtype=np.float32)
        if out is None:
            return np.full(colour.shape, 0.8, dtype=np.float32)
        out[:] = 0.8
        return out


class _FakeHashGrid:
    def sample(self, grid_coords, rng=None, batch_size=2_000_000, report_success=False):
        first = next(iter(grid_coords.values()))
        return np.full((np.asarray(first).shape[0], 2), 0.01, dtype=np.float32)


def _configure_minimal_catwise_sim(chunk_size: int = 16) -> tuple[Catwise, list[int]]:
    cfg = CatwiseConfig(
        cat_w1_max=17.0,
        cat_w12_min=0.5,
        magnitude_error_dist="gaussian",
        chunk_size=chunk_size,
        store_final_samples=False,
    )
    sim = Catwise(cfg)
    sim.lookups_are_initialised = True
    n_pix = hp.nside2npix(sim.nside)
    sim.mask_map = np.zeros(n_pix, dtype=np.float64)
    sim.fill_value = np.nan
    sim.log_w1cov_map = np.zeros(n_pix, dtype=np.float64)
    sim.log_w2cov_map = np.zeros(n_pix, dtype=np.float64)
    sim.spectral_lookup = _FakeAlphaLookup()
    sim.hashgrid = _FakeHashGrid()

    magnitude_call_sizes: list[int] = []

    def _sample_magnitudes(n, dtype=np.float64, rng=None):
        magnitude_call_sizes.append(int(n))
        return (
            np.full(n, 10.0, dtype=dtype),
            np.full(n, 9.0, dtype=dtype),
        )

    sim.sample_magnitudes = _sample_magnitudes
    sim.sample_points = lambda n, dtype=np.float64, **kwargs: (
        np.linspace(0.0, 90.0, n, dtype=dtype),
        np.zeros(n, dtype=dtype),
    )
    sim.aberrate_points = lambda lon, lat, dtype=np.float64: (
        np.asarray(lon, dtype=dtype),
        np.asarray(lat, dtype=dtype),
        np.zeros(np.asarray(lon).shape, dtype=dtype),
    )
    sim.boost_magnitudes = lambda magnitudes, angle, spectral_index, dtype=np.float64: (
        np.asarray(magnitudes, dtype=dtype)
    )
    sim._source_isin_mask = lambda lon, lat: (
        np.ones(np.asarray(lon).shape[0], dtype=bool),
        np.arange(np.asarray(lon).shape[0], dtype=np.int64) % n_pix,
    )
    sim.add_error = lambda w1, w2, **kwargs: (
        np.asarray(w1[0]),
        np.asarray(w2[0]),
        np.asarray(w1[1]),
        np.asarray(w2[1]),
    )
    return sim, magnitude_call_sizes


class CatwisePoissonClusteringTests(unittest.TestCase):
    def test_generate_dipole_rejects_invalid_lambda_clus(self):
        sim, _ = _configure_minimal_catwise_sim()
        with self.assertRaisesRegex(ValueError, "lambda_clus must be non-negative"):
            sim.generate_dipole(np.log10(8.0), lambda_clus=-0.1)

    def test_generate_dipole_rejects_lambda_clus_with_correlated_points(self):
        cfg = CatwiseConfig(
            cat_w1_max=17.0,
            cat_w12_min=0.5,
            magnitude_error_dist="gaussian",
            generate_correlated_points=True,
        )
        sim = Catwise(cfg)
        sim.lookups_are_initialised = True
        with self.assertRaisesRegex(ValueError, "generate_correlated_points=True"):
            sim.generate_dipole(np.log10(8.0), lambda_clus=0.5)

    def test_generate_dipole_normalizes_parent_count_by_expected_multiplicity(self):
        sim, magnitude_call_sizes = _configure_minimal_catwise_sim()
        sim.generate_dipole(np.log10(12.0), lambda_clus=1.0)
        self.assertEqual(magnitude_call_sizes[0], 6)

    def test_generate_dipole_adds_poisson_children_on_top_of_parents(self):
        sim, magnitude_call_sizes = _configure_minimal_catwise_sim()

        sim.sample_clustered_points = lambda parent_lon, parent_lat, counts, rng=None, dtype=np.float64: (
            (np.repeat(np.asarray(parent_lon, dtype=np.float64), counts) + 0.01).astype(
                dtype,
                copy=False,
            ),
            np.repeat(np.asarray(parent_lat, dtype=np.float64), counts).astype(
                dtype,
                copy=False,
            ),
        )

        class _FixedClusterRng:
            def __init__(self):
                self._delegate = np.random.default_rng(123)

            def poisson(self, lam, size=None):
                if size == 5 and np.isscalar(lam):
                    return np.array([2, 0, 1, 0, 0], dtype=np.int64)
                return self._delegate.poisson(lam=lam, size=size)

            def __getattr__(self, name):
                return getattr(self._delegate, name)

        class _FixedClusterKey:
            def _generator(self):
                return _FixedClusterRng()

        density_map, _ = sim.generate_dipole(
            np.log10(10.0),
            lambda_clus=1.0,
            rng_key=_FixedClusterKey(),
        )

        self.assertEqual(magnitude_call_sizes[:2], [5, 3])
        self.assertEqual(float(np.nansum(density_map)), 8.0)


class CatwiseJaxImportTests(unittest.TestCase):
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

    def test_catwise_jax_missing_dependency_message_is_clear(self):
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
    catsim.CatwiseJax
except ImportError as exc:
    print(str(exc))
"""
        output = subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=_repo_root(),
            env=env,
            text=True,
        )
        self.assertIn("CatwiseJax requires the optional JAX dependencies", output)
        self.assertIn("catsim[jax]", output)


@unittest.skipIf(jax is None, "CatwiseJax requires the optional JAX dependencies.")
class CatwiseJaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from catsim import CatwiseJax

        cfg = CatwiseConfig(
            cat_w1_max=17.0,
            cat_w12_min=0.5,
            magnitude_error_dist="gaussian",
            chunk_size=8,
            store_final_samples=False,
        )
        cls.sim = CatwiseJax(cfg)
        cls.sim.initialise_data()

    def test_generate_dipole_returns_expected_shapes_and_is_deterministic(self):
        key = jax.random.PRNGKey(123)
        first_map, first_mask = self.sim.generate_dipole(
            np.log10(16.0),
            observer_speed=0.0,
            key=key,
        )
        second_map, second_mask = self.sim.generate_dipole(
            np.log10(16.0),
            observer_speed=0.0,
            key=key,
        )

        self.assertEqual(first_map.shape, (hp.nside2npix(64),))
        self.assertEqual(first_mask.shape, (hp.nside2npix(64),))
        self.assertEqual(first_map.dtype, np.float32)
        self.assertEqual(first_mask.dtype, np.bool_)
        np.testing.assert_array_equal(first_map, second_map)
        np.testing.assert_array_equal(first_mask, second_mask)

    def test_batch_generate_dipole_returns_stacked_maps_and_masks(self):
        maps, masks = self.sim.batch_generate_dipole(
            {
                "log10_n_initial_samples": np.log10(np.array([8.0, 16.0])),
                "observer_speed": np.zeros(2, dtype=np.float32),
            },
            jax.random.PRNGKey(7),
            batch_size=2,
        )

        self.assertEqual(maps.shape, (2, hp.nside2npix(64)))
        self.assertEqual(masks.shape, (2, hp.nside2npix(64)))
        self.assertEqual(maps.dtype, np.float32)
        self.assertEqual(masks.dtype, np.bool_)

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

    def test_batch_generate_dipole_pads_final_host_batch(self):
        maps, masks = self.sim.batch_generate_dipole(
            {"log10_n_initial_samples": np.full(3, 1.0)},
            jax.random.PRNGKey(703),
            batch_size=2,
        )

        self.assertEqual(maps.shape, (3, hp.nside2npix(64)))
        self.assertEqual(masks.shape, (3, hp.nside2npix(64)))

    def test_students_t_error_path_runs(self):
        from catsim import CatwiseJax

        cfg = CatwiseConfig(
            cat_w1_max=17.0,
            cat_w12_min=0.5,
            magnitude_error_dist="students-t",
            chunk_size=8,
            store_final_samples=False,
        )
        sim = CatwiseJax(cfg)
        sim.initialise_data()

        density_map, mask = sim.generate_dipole(
            np.log10(16.0),
            observer_speed=0.0,
            log10_magnitude_error_shape_param=np.log10(5.0),
            key=jax.random.PRNGKey(704),
        )

        self.assertEqual(density_map.shape, mask.shape)
        self.assertTrue(np.all(np.isfinite(density_map[mask])))

    def test_poisson_clustering_path_runs(self):
        density_map, mask = self.sim.generate_dipole(
            np.log10(16.0),
            observer_speed=0.0,
            lambda_clus=0.5,
            key=jax.random.PRNGKey(705),
        )

        self.assertEqual(density_map.shape, mask.shape)
        self.assertTrue(np.all(np.isfinite(density_map[mask])))

    def test_batch_generate_dipole_accepts_vector_lambda_clus(self):
        maps, masks = self.sim.batch_generate_dipole(
            {
                "log10_n_initial_samples": np.full(2, 2.0, dtype=np.float32),
                "lambda_clus": np.array([0.0, 0.5], dtype=np.float32),
            },
            jax.random.PRNGKey(706),
            batch_size=2,
        )

        self.assertEqual(maps.shape, (2, hp.nside2npix(64)))
        self.assertEqual(masks.shape, (2, hp.nside2npix(64)))

    def test_zero_lambda_clus_uses_no_child_slots(self):
        parameters = self.sim._normalise_theta(
            {
                "log10_n_initial_samples": np.full(2, 2.0),
                "lambda_clus": np.zeros(2),
            }
        )
        self.assertEqual(self.sim._active_max_children(parameters), 0)

    def test_positive_lambda_clus_uses_configured_child_cap(self):
        parameters = self.sim._normalise_theta(
            {
                "log10_n_initial_samples": np.full(2, 2.0),
                "lambda_clus": np.array([0.0, 0.5]),
            }
        )
        self.assertEqual(
            self.sim._active_max_children(parameters),
            self.sim.max_cluster_children_per_parent,
        )

    def test_negative_lambda_clus_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "lambda_clus must be non-negative"):
            self.sim.generate_dipole(
                np.log10(16.0),
                lambda_clus=-0.1,
                key=jax.random.PRNGKey(707),
            )

    def test_overfill_probability_warning_is_explicit(self):
        from catsim import CatwiseJax

        cfg = CatwiseConfig(
            cat_w1_max=17.0,
            cat_w12_min=0.5,
            magnitude_error_dist="gaussian",
            chunk_size=8,
            store_final_samples=False,
            max_cluster_children_per_parent=1,
        )
        sim = CatwiseJax(cfg)
        sim.initialise_data()

        with self.assertWarnsRegex(
            RuntimeWarning,
            r"P\(children_per_parent > 1\)",
        ):
            sim.generate_dipole(
                np.log10(16.0),
                lambda_clus=1.0,
                key=jax.random.PRNGKey(708),
            )

    def test_unsupported_options_raise_clear_errors(self):
        from catsim import CatwiseJax

        unsupported_configs = [
            (
                CatwiseConfig(
                    cat_w1_max=17.0,
                    cat_w12_min=0.5,
                    magnitude_error_dist="gaussian",
                    store_final_samples=True,
                ),
                "store_final_samples=True",
            ),
            (
                CatwiseConfig(
                    cat_w1_max=17.0,
                    cat_w12_min=0.5,
                    magnitude_error_dist="gaussian",
                    generate_correlated_points=True,
                ),
                "generate_correlated_points=True",
            ),
            (
                CatwiseConfig(
                    cat_w1_max=17.0,
                    cat_w12_min=0.5,
                    magnitude_error_dist="gaussian",
                    add_confusion_noise=True,
                ),
                "add_confusion_noise=True",
            ),
        ]

        for cfg, message in unsupported_configs:
            with self.subTest(message=message):
                sim = CatwiseJax(cfg)
                with self.assertRaisesRegex(NotImplementedError, message):
                    sim.initialise_data()

    def test_integration_mean_counts_and_residual_rms_match_numpy_reference(self):
        from catsim import CatwiseJax

        n_numpy = 80
        n_jax = 40
        log10_n = 4.0

        cfg = CatwiseConfig(
            cat_w1_max=17.0,
            cat_w12_min=0.5,
            magnitude_error_dist="gaussian",
            chunk_size=1_000,
            store_final_samples=False,
        )

        numpy_sim = Catwise(cfg)
        numpy_sim.initialise_data()
        numpy_maps, numpy_masks = batch_simulate(
            theta={
                "log10_n_initial_samples": np.full(n_numpy, log10_n, dtype=np.float32),
                "observer_speed": np.zeros(n_numpy, dtype=np.float32),
                "lambda_clus": np.full(n_numpy, 0.5, dtype=np.float32),
            },
            model_callable=numpy_sim.generate_dipole,
            n_workers=1,
            rng_key=prng_key(20_000),
        )

        jax_sim = CatwiseJax(cfg)
        jax_sim.initialise_data()
        jax_maps, jax_masks = jax_sim.batch_generate_dipole(
            theta={
                "log10_n_initial_samples": np.full(n_jax, log10_n, dtype=np.float32),
                "observer_speed": np.zeros(n_jax, dtype=np.float32),
                "lambda_clus": np.full(n_jax, 0.5, dtype=np.float32),
            },
            key=jax.random.PRNGKey(30_000),
            batch_size=10,
        )

        np.testing.assert_array_equal(numpy_masks[0], numpy_sim.native_mask)
        np.testing.assert_array_equal(jax_masks[0], numpy_sim.native_mask)

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
            0.08,
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
            0.08 * max(mean_pixel_count, 1.0),
            msg=(
                "Mean JAX-minus-NumPy native-pixel residual is too large; "
                f"raw_mean_residual={raw_mean_residual:.3g}, "
                f"mean_pixel_count={mean_pixel_count:.3g}"
            ),
        )

        null_rms = _numpy_split_null_residual_rms(
            numpy_maps,
            group_size=n_jax,
            n_resamples=50,
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
