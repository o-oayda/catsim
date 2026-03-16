from typing import Optional
from os import PathLike
from numpy.typing import NDArray
import numpy as np


class HashGrid:
    def __init__(
            self, 
            grid_coords: dict[str, NDArray], 
            grid_values: dict[str, NDArray],
            grid_step: list[float]
    ) -> None:
        self.grid_dim_labels, coord_arrays = self._prepare_coord_arrays(grid_coords)
        self.positional_data = np.column_stack(coord_arrays)
        self.ndim = self.positional_data.shape[-1]
        self.n_points = self.positional_data.shape[0]

        self.grid_value_labels, value_arrays = self._prepare_value_arrays(
            grid_values, expected_length=self.n_points
        )
        self.grid_values = np.column_stack(value_arrays)
        self.grid_values_ndim = self.grid_values.shape[-1]
        self.n_grid_values = self.grid_values.shape[0]

        self.mins = self.positional_data.min(axis=0)
        self.maxs = self.positional_data.max(axis=0)

        self.grid_step = np.asarray(grid_step)
        self._check_gridstep_isok(self.grid_step)

        self._build_grid()

    def __repr__(self) -> str:
        state = {
            "grid_dim_labels": np.array(self.grid_dim_labels, dtype="U"),
            "grid_value_labels": np.array(self.grid_value_labels, dtype="U"),
            "grid_step": self.grid_step,
            "positional_data": self.positional_data,
            "grid_values": self.grid_values,
            "mins": self.mins,
            "maxs": self.maxs,
            "grid_span": self.grid_span,
            "grid_nbins": self.grid_nbins,
            "unique_grid_ids": self.unique_grid_ids,
            "offsets": self.offsets,
            "members": self.members,
        }
        strrepr = 'HashGrid('
        for key, val in state.items():
            strrepr += f'\n\t{key}: {val}'
        strrepr += '\n)'
        return strrepr

    def _check_gridstep_isok(self, grid_step):
        assert len(grid_step) == self.ndim, (
            "Specify a grid step for each dimension of the grid."
        )
        if np.any(grid_step <= 0):
            raise ValueError("All grid_step values must be > 0.")

    def _prepare_coord_arrays(
        self, grid_coords: dict[str, NDArray]
    ) -> tuple[list[str], list[NDArray]]:
        labels = list(grid_coords.keys())
        if not labels:
            raise ValueError("grid_coords must contain at least one dimension.")

        coord_arrays = [np.asarray(grid_coords[label]) for label in labels]
        coord_lengths = [arr.shape[0] for arr in coord_arrays]
        if len(set(coord_lengths)) != 1:
            raise ValueError(
                "All coordinate arrays must share the same length; "
                f"got lengths {coord_lengths}."
            )
            
        return labels, coord_arrays

    def _prepare_value_arrays(
        self,
        grid_values: dict[str, NDArray],
        *,
        expected_length: int,
    ) -> tuple[list[str], list[NDArray]]:
        labels = list(grid_values.keys())
        if not labels:
            raise ValueError("grid_values must contain at least one value column.")
            
        value_arrays = [np.asarray(grid_values[label]) for label in labels]
        value_lengths = [arr.shape[0] for arr in value_arrays]
        if len(set(value_lengths)) != 1 or value_lengths[0] != expected_length:
            raise ValueError(
                "Each value array must match the length of the coordinate arrays; "
                f"coords length={expected_length}, value lengths={value_lengths}."
            )

        return labels, value_arrays

    def _build_grid(self):
        self.grid_span, self.grid_nbins = self._determine_grid_nbins()
        grid_ids = self._point_to_grid_id(self.positional_data)
        self.unique_grid_ids, self.offsets, self.members = self._build_lookup(
            grid_ids
        )

    def _determine_grid_nbins(self) -> tuple[NDArray, NDArray]:
        grid_span = self.maxs - self.mins
        grid_nbins = np.floor(grid_span / self.grid_step).astype(np.int64) + 1
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
            *,
            report_success: bool = False,
    ) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()

        missing = [label for label in self.grid_dim_labels if label not in grid_coords]
        if missing:
            raise ValueError(
                f"Missing required query dimensions: {missing}."
            )

        extra = [label for label in grid_coords.keys() if label not in self.grid_dim_labels]
        if extra:
            raise ValueError(
                f"Query provided unexpected dimensions: {extra}."
            )

        query_arrays = [
            np.asarray(grid_coords[label], dtype=np.float32)
            for label in self.grid_dim_labels
        ]
        query_lengths = [arr.shape[0] for arr in query_arrays]
        if len(set(query_lengths)) != 1:
            raise ValueError(
                "All query coordinate arrays must share the same length; "
                f"got lengths {query_lengths}."
            )

        grid_query = np.column_stack(query_arrays).astype(np.float32, copy=False)
        n_query = grid_query.shape[0]
        out = np.empty((n_query, self.grid_values_ndim), dtype=np.float32)
        total_hits = 0
        total_queries = n_query

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
            total_hits += idx_hit.size

            # final fallback (global) for anything still unresolved
            if np.any(unresolved):
                idx_u = np.where(unresolved)[0]
                global_point_idx = rng.integers(
                    0, self.grid_values.shape[0], size=idx_u.size, dtype=np.int64
                )
                out_block[idx_u, :] = self.grid_values[global_point_idx, :]

            out[start:end, :] = out_block
            
        if report_success and total_queries > 0:
            success_pct = 100.0 * total_hits / total_queries
            print(f"HashGrid sampling success rate: {success_pct:.2f}%")

        return out

    def save(self, path: str | PathLike[str]) -> None:
        state = {
            "grid_dim_labels": np.array(self.grid_dim_labels, dtype="U"),
            "grid_value_labels": np.array(self.grid_value_labels, dtype="U"),
            "grid_step": self.grid_step,
            "positional_data": self.positional_data,
            "grid_values": self.grid_values,
            "mins": self.mins,
            "maxs": self.maxs,
            "grid_span": self.grid_span,
            "grid_nbins": self.grid_nbins,
            "unique_grid_ids": self.unique_grid_ids,
            "offsets": self.offsets,
            "members": self.members,
        }
        np.savez_compressed(path, **state)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> "HashGrid":
        with np.load(path, allow_pickle=True) as data:
            obj = cls.__new__(cls)
            obj.grid_dim_labels = data["grid_dim_labels"].astype(str).tolist()
            obj.grid_value_labels = data["grid_value_labels"].astype(str).tolist()
            obj.grid_step = data["grid_step"]
            obj.positional_data = data["positional_data"]
            obj.grid_values = data["grid_values"]
            obj.grid_values_ndim = obj.grid_values.shape[-1]
            obj.n_grid_values = obj.grid_values.shape[0]
            obj.mins = data["mins"]
            obj.maxs = data["maxs"]
            obj.grid_span = data["grid_span"]
            obj.grid_nbins = data["grid_nbins"]
            obj.unique_grid_ids = data["unique_grid_ids"]
            obj.offsets = data["offsets"]
            obj.members = data["members"]
            obj.ndim = obj.positional_data.shape[-1]
            obj.n_points = obj.positional_data.shape[0]
        return obj


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
