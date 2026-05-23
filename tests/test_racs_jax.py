import os
from pathlib import Path
import subprocess
import sys
import unittest

import healpy as hp
import numpy as np

from catsim import RacsLow3, RacsLow3Config, batch_simulate
from catsim.utils.rng import prng_key

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - depends on optional extra availability.
    jax = None
    jnp = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
class RacsJaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from catsim import RacsLow3Jax

        cfg = RacsLow3Config(
            flux_min=0.001,
            chunk_size=4,
            store_final_samples=False,
            max_cluster_children_per_parent=4,
        )
        cls.sim = RacsLow3Jax(cfg)
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

    def test_batch_generate_dipole_returns_stacked_maps_and_masks(self):
        theta = {
            "log10_n_initial_samples": np.log10(np.array([4.0, 8.0])),
            "p_clus": np.array([0.0, 0.5]),
            "clus_stop_prob": np.array([1.0, 0.8]),
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
        from catsim import RacsLow3Jax

        cfg = RacsLow3Config(
            flux_min=0.001,
            chunk_size=4,
            store_final_samples=False,
            cluster_count_model="poisson",
            max_cluster_children_per_parent=6,
        )
        sim = RacsLow3Jax(cfg)
        sim.initialise_data()

        density_map, mask = sim.generate_dipole(
            np.log10(8.0),
            lambda_clus=0.5,
            key=jax.random.PRNGKey(12),
        )

        self.assertEqual(density_map.shape, mask.shape)
        self.assertTrue(np.all(np.isfinite(density_map[mask])))

    def test_overfill_probability_warning_is_explicit(self):
        from catsim import RacsLow3Jax

        cfg = RacsLow3Config(
            flux_min=0.001,
            chunk_size=4,
            store_final_samples=False,
            max_cluster_children_per_parent=2,
        )
        sim = RacsLow3Jax(cfg)
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
        from catsim import RacsLow3Jax

        cfg = RacsLow3Config(
            flux_min=0.001,
            chunk_size=4,
            store_final_samples=True,
        )
        sim = RacsLow3Jax(cfg)
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
        from catsim import RacsLow3Jax

        n_numpy = 200
        n_jax = 100
        log10_n = 6.0

        cfg = RacsLow3Config(
            flux_min=15,
            chunk_size=1_000,
            store_final_samples=False,
            max_cluster_children_per_parent=4,
        )

        numpy_sim = RacsLow3(cfg)
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

        jax_sim = RacsLow3Jax(cfg)
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
