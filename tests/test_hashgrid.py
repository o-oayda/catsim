import unittest
import numpy as np
import tempfile
from pathlib import Path

from catsim.utils.hashgrid import HashGrid


class HashGridInitTests(unittest.TestCase):
    def test_coordinate_lengths_must_match(self):
        grid_coords = {"x": np.array([0.0, 1.0]), "y": np.array([0.0])}
        grid_values = {"sigma": np.array([1.0, 2.0])}

        with self.assertRaisesRegex(ValueError, "coordinate arrays must share"):
            HashGrid(grid_coords=grid_coords, grid_values=grid_values, grid_step=[1.0, 1.0])

    def test_value_lengths_must_match_coordinates(self):
        grid_coords = {"x": np.array([0.0, 1.0])}
        grid_values = {"sigma": np.array([5.0])}

        with self.assertRaisesRegex(ValueError, "must match the length"):
            HashGrid(grid_coords=grid_coords, grid_values=grid_values, grid_step=[1.0])

    def test_grid_bins_are_integer_typed(self):
        grid_coords = {"x": np.array([0.0, 5.0]), "y": np.array([0.0, 5.0])}
        grid_values = {"sigma": np.array([1.0, 2.0])}

        hashgrid = HashGrid(
            grid_coords=grid_coords,
            grid_values=grid_values,
            grid_step=[0.5, 0.5],
        )

        self.assertEqual(hashgrid.grid_nbins.dtype, np.int64)


class HashGridSampleTests(unittest.TestCase):
    def setUp(self):
        self.grid_coords = {"x": np.array([0.0, 5.0]), "y": np.array([0.0, 5.0])}
        self.grid_values = {"sigma": np.array([10.0, 20.0])}
        self.grid_step = [1.0, 1.0]
        self.hashgrid = HashGrid(
            grid_coords=self.grid_coords,
            grid_values=self.grid_values,
            grid_step=self.grid_step,
        )

    def test_sample_requires_all_dimensions(self):
        query = {"x": np.array([0.0])}
        with self.assertRaisesRegex(ValueError, "Missing required query dimensions"):
            self.hashgrid.sample(query)

    def test_sample_rejects_extra_dimensions(self):
        query = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
        }
        with self.assertRaisesRegex(ValueError, "unexpected dimensions"):
            self.hashgrid.sample(query)

    def test_sample_returns_bucket_values_for_hits(self):
        rng = np.random.default_rng(0)
        query = {"x": np.array([0.0, 5.0]), "y": np.array([0.0, 5.0])}

        samples = self.hashgrid.sample(query, rng=rng)

        np.testing.assert_allclose(samples[:, 0], self.grid_values["sigma"], rtol=0, atol=0)

    def test_sample_fallback_draws_from_global_values(self):
        rng = np.random.default_rng(123)
        query = {"x": np.array([100.0, 200.0]), "y": np.array([100.0, 200.0])}

        samples = self.hashgrid.sample(query, rng=rng)

        for value in samples[:, 0]:
            self.assertIn(value, self.grid_values["sigma"])


class HashGridLinearizationTests(unittest.TestCase):
    def test_grid_coords_to_scalar_matches_numpy_ravel(self):
        x = np.arange(3)
        y = np.arange(2)
        z = np.arange(4)
        mesh = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
        grid_coords = {"x": mesh[:, 0].astype(float), "y": mesh[:, 1].astype(float), "z": mesh[:, 2].astype(float)}
        grid_values = {"sigma": np.arange(mesh.shape[0], dtype=float)}
        hashgrid = HashGrid(grid_coords, grid_values, grid_step=[1.0, 1.0, 1.0])

        int_coords = mesh.astype(np.int64)
        scalars = hashgrid._grid_coords_to_scalar(int_coords)

        expected = np.ravel_multi_index(
            (int_coords[:, 0], int_coords[:, 1], int_coords[:, 2]),
            dims=tuple(hashgrid.grid_nbins.astype(int)),
        )

        np.testing.assert_array_equal(scalars, expected)
        self.assertEqual(len(np.unique(scalars)), int_coords.shape[0])


class HashGridSerializationTests(unittest.TestCase):
    def setUp(self):
        self.grid_coords = {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([3.0, 4.0, 5.0])}
        self.grid_values = {
            "sigma": np.array([1.1, 2.2, 3.3]),
            "tau": np.array([4.4, 5.5, 6.6]),
        }
        self.hashgrid = HashGrid(self.grid_coords, self.grid_values, grid_step=[0.5, 0.5])

    def test_save_and_load_preserves_sampling(self):
        rng = np.random.default_rng(123)
        query = {"x": np.array([0.0, 2.0]), "y": np.array([3.0, 5.0])}
        expected = self.hashgrid.sample(query, rng=rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hashgrid.npz"
            self.hashgrid.save(path)
            loaded = HashGrid.load(path)

        rng_loaded = np.random.default_rng(123)
        actual = loaded.sample(query, rng=rng_loaded)

        np.testing.assert_allclose(actual, expected)
        self.assertEqual(loaded.grid_dim_labels, self.hashgrid.grid_dim_labels)
        self.assertEqual(loaded.grid_value_labels, self.hashgrid.grid_value_labels)
        np.testing.assert_array_equal(loaded.grid_values, self.hashgrid.grid_values)
        np.testing.assert_array_equal(loaded.grid_step, self.hashgrid.grid_step)


if __name__ == "__main__":
    unittest.main()
