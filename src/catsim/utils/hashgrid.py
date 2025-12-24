from typing import Optional
from numpy.typing import NDArray
import numpy as np


class HashGrid:
    def __init__(
            self, 
            grid_coords: dict[str, NDArray], 
            grid_values: dict[str, NDArray],
            grid_step: list[float]
    ) -> None:
        self.positional_data = np.column_stack(list(grid_coords.values()))
        self.grid_dim_labels = list(grid_coords.keys())
        self.ndim = self.positional_data.shape[-1]
        self.n_points = self.positional_data.shape[0]

        self.grid_values = np.column_stack(list(grid_values.values()))
        self.grid_value_labels = list(grid_values.keys())
        self.grid_values_ndim = self.grid_values.shape[-1]
        self.n_grid_values = self.grid_values.shape[0]

        self.mins = self.positional_data.min(axis=0)
        self.maxs = self.positional_data.max(axis=0)

        self.grid_step = np.asarray(grid_step)
        self._check_gridstep_isok(self.grid_step)

        self._build_grid()

    def _check_gridstep_isok(self, grid_step):
        assert len(grid_step) == self.ndim, (
            "Specify a grid step for each dimension of the grid."
        )
        if np.any(grid_step <= 0):
            raise ValueError("All grid_step values must be > 0.")

    def _build_grid(self):
        self.grid_span, self.grid_nbins = self._determine_grid_nbins()
        grid_ids = self._point_to_grid_id(self.positional_data)
        self.unique_grid_ids, self.offsets, self.members = self._build_lookup(
            grid_ids
        )

    def _determine_grid_nbins(self) -> tuple[NDArray, NDArray]:
        grid_span = self.maxs - self.mins
        grid_nbins = np.floor(grid_span / self.grid_step) + 1
        return grid_span, grid_nbins

    def _point_to_grid_id(self, positional_data: NDArray) -> NDArray:
        grid_coords = np.floor( (positional_data - self.mins) / self.grid_step)
        grid_coords = np.clip(grid_coords, 0, self.grid_nbins - 1).astype(np.int64)
        return self._grid_coords_to_scalar(grid_coords)

    def _grid_coords_to_scalar(self, grid_nd_integer_coords: NDArray) -> NDArray:
        '''
        linearization: (((i0 * n1 + i1) * n2 + i2) * n3 + i3)
        '''
        coords = grid_nd_integer_coords
        cur_dim_scalar = coords[:, 0]
        for i in range(1, self.ndim):
            cur_dim_scalar = cur_dim_scalar * self.grid_nbins[i] + coords[:, i]
        return cur_dim_scalar
        # n1, n2, n3 = self.grid_nbins[1], self.grid_nbins[2], self.grid_nbins[3]
        # scalar_id = (
        #     (coords[:, 0] * n1 + coords[:, 1]) * n2 + coords[:, 2]
        # ) * n3 + coords[:, 3]
        # return scalar_id

    def _build_lookup(self, grid_ids: NDArray):
        order = np.argsort(grid_ids, kind="mergesort")  # stable
        cell_sorted = grid_ids[order]
        members = (
            order.astype(np.int32) if self.n_points < (1 << 31) 
            else order.astype(np.int64)
        )
        unique_ids, first_idx, counts = np.unique(
            cell_sorted, 
            return_index=True, 
            return_counts=True
        )
        offsets = np.empty(unique_ids.size + 1, dtype=np.int64)
        offsets[0] = 0
        offsets[1:] = np.cumsum(counts, dtype=np.int64)

        return unique_ids.astype(np.int64), offsets, members

    def sample(
            self,
            grid_coords: dict[str, NDArray],
            rng: Optional[np.random.Generator] = None,
            batch_size: int = 2_000_000,
    ) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()

        grid_query = np.column_stack(
            list(grid_coords.values())
        ).astype(np.float32, copy=False)
        n_query = grid_query.shape[0]
        out = np.empty((n_query, self.grid_values_ndim), dtype=np.float32)

        for start in range(0, n_query, batch_size):
            end = min(n_query, start + batch_size)
            cur_query = grid_query[start:end]
            batchsize = cur_query.shape[0]

            unresolved = np.ones(batchsize, dtype=bool)
            out_block = np.empty(
                (batchsize, self.grid_values_ndim), dtype=np.float32
            )

            queried_cell_id = self._point_to_grid_id(cur_query)

            # membership via searchsorted (unique_cell_ids sorted)
            pos = np.searchsorted(self.unique_grid_ids, queried_cell_id)
            hit = np.zeros(batchsize, dtype=bool)
            in_bounds = pos < self.unique_grid_ids.size
            if np.any(in_bounds):
                idx_in_bounds = np.where(in_bounds)[0]
                hit[idx_in_bounds] = (
                    self.unique_grid_ids[pos[idx_in_bounds]]
                    == queried_cell_id[idx_in_bounds]
                )

            # bucket ranges
            idx_hit = np.where(hit)[0]
            bucket = pos[idx_hit].astype(np.int64)
            start_off = self.offsets[bucket]
            end_off = self.offsets[bucket + 1]
            lengths = end_off - start_off

            # sample index within each bucket: start + floor(u * len)
            u = rng.random(lengths.size)
            pick = start_off + np.floor(u * lengths).astype(np.int64)

            point_idx = self.members[pick]  # indices into model.Y
            if idx_hit.size:
                out_block[idx_hit, :] = self.grid_values[point_idx, :]

            unresolved[idx_hit] = False

            # final fallback (global) for anything still unresolved
            if np.any(unresolved):
                idx_u = np.where(unresolved)[0]
                global_point_idx = rng.integers(
                    0, self.grid_values.shape[0], size=idx_u.size, dtype=np.int64
                )
                out_block[idx_u, :] = self.grid_values[global_point_idx, :]

            out[start:end, :] = out_block
            
        return out


if __name__ == '__main__':
    w1 = np.asarray([1,2])
    w1cov = np.asarray([3,3])
    sigmaw1 = np.asarray([0.4, 0.3])
    grid_data = {
        'w1': w1,
        'w1cov': w1cov
    }
    grid_values = {'sigmaw1': sigmaw1}
    
    hgrid = HashGrid(
        grid_coords=grid_data, 
        grid_values=grid_values,
        grid_step=[0.1, 0.1]
    )
    nice = hgrid.sample(grid_coords=grid_data)
