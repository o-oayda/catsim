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
