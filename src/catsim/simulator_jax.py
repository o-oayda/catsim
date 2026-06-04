from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Optional
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via package import.
    if exc.name and (exc.name == "jax" or exc.name.startswith("jax.")):
        raise ImportError(
            "CatwiseJax requires the optional JAX dependencies. "
            "Install them with `pip install 'catsim[jax]'`."
        ) from exc
    raise

from .configs import CatwiseConfig
from .racs_jax import (
    _aberrate_points_jax,
    _sample_clustered_positions_jax,
    jax_ang2pix_nest_lonlat,
)
from .simulator import Catwise
from .utils.constants import CMB_B, CMB_BETA, CMB_L
from .utils.healsphere import downgrade_ignore_nan
from .utils.physics import rotation_matrices_for_dipole


_GAUSSIAN_ERROR = 0
_STUDENTS_T_ERROR = 1
_OVERFILL_WARNING_THRESHOLD = 0.01


@dataclass(frozen=True)
class _CatwiseLookupArrays:
    mag_x_flat: jax.Array
    mag_y_flat: jax.Array
    mag_x_widths_flat: jax.Array
    mag_y_widths_flat: jax.Array
    mag_cdf: jax.Array
    mask_map: jax.Array
    log_w1cov_map: jax.Array
    log_w2cov_map: jax.Array
    alpha_coefficients: jax.Array
    hashgrid_mins: jax.Array
    hashgrid_grid_step: jax.Array
    hashgrid_grid_nbins: jax.Array
    hashgrid_unique_ids: jax.Array
    hashgrid_offsets: jax.Array
    hashgrid_members: jax.Array
    hashgrid_values: jax.Array

    def as_tuple(self) -> tuple[jax.Array, ...]:
        return (
            self.mag_x_flat,
            self.mag_y_flat,
            self.mag_x_widths_flat,
            self.mag_y_widths_flat,
            self.mag_cdf,
            self.mask_map,
            self.log_w1cov_map,
            self.log_w2cov_map,
            self.alpha_coefficients,
            self.hashgrid_mins,
            self.hashgrid_grid_step,
            self.hashgrid_grid_nbins,
            self.hashgrid_unique_ids,
            self.hashgrid_offsets,
            self.hashgrid_members,
            self.hashgrid_values,
        )


def _hashgrid_ids_jax(
    coordinates: jax.Array,
    mins: jax.Array,
    grid_step: jax.Array,
    grid_nbins: jax.Array,
) -> jax.Array:
    grid_coordinates = jnp.floor((coordinates - mins) / grid_step).astype(jnp.int32)
    grid_coordinates = jnp.clip(grid_coordinates, 0, grid_nbins - 1)
    scalar = grid_coordinates[:, 0]
    for dim in range(1, coordinates.shape[1]):
        scalar = scalar * grid_nbins[dim] + grid_coordinates[:, dim]
    return scalar.astype(jnp.int32)


def _sample_magnitudes_jax(
    key: jax.Array,
    shape: tuple[int, ...],
    x_flat: jax.Array,
    y_flat: jax.Array,
    x_widths_flat: jax.Array,
    y_widths_flat: jax.Array,
    cdf: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    key_bin, key_x_jitter, key_y_jitter = jax.random.split(key, 3)
    u_bin = jax.random.uniform(key_bin, shape, dtype=jnp.float32)
    bin_idx = jnp.searchsorted(cdf, u_bin, side="right")
    bin_idx = jnp.clip(bin_idx, 0, cdf.shape[0] - 1)

    x_jitter = jax.random.uniform(
        key_x_jitter,
        shape,
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    )
    y_jitter = jax.random.uniform(
        key_y_jitter,
        shape,
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    )
    w1 = x_flat[bin_idx] + x_jitter * x_widths_flat[bin_idx]
    w2 = y_flat[bin_idx] + y_jitter * y_widths_flat[bin_idx]
    return w1, w2


def _fit_alpha_jax(colour: jax.Array, coefficients: jax.Array) -> jax.Array:
    result = jnp.full_like(colour, coefficients[0], dtype=jnp.float32)
    for coefficient in coefficients[1:]:
        result = result * colour + coefficient
    return result


def _boost_magnitudes_jax(
    magnitudes: jax.Array,
    angle_to_source: jax.Array,
    observer_beta: jax.Array,
    spectral_index: jax.Array,
) -> jax.Array:
    gamma = 1.0 / jnp.sqrt(1.0 - observer_beta**2)
    delta = gamma * (1.0 + observer_beta * jnp.cos(jnp.deg2rad(angle_to_source)))
    return magnitudes - 2.5 * (1.0 + spectral_index) * jnp.log10(delta)


def _sample_hashgrid_jax(
    key: jax.Array,
    query_coordinates: jax.Array,
    mins: jax.Array,
    grid_step: jax.Array,
    grid_nbins: jax.Array,
    unique_ids: jax.Array,
    offsets: jax.Array,
    members: jax.Array,
    values: jax.Array,
) -> jax.Array:
    key_bucket, key_global = jax.random.split(key)
    queried_ids = _hashgrid_ids_jax(query_coordinates, mins, grid_step, grid_nbins)

    pos = jnp.searchsorted(unique_ids, queried_ids)
    safe_pos = jnp.minimum(pos, unique_ids.shape[0] - 1)
    in_bounds = pos < unique_ids.shape[0]
    hit = in_bounds & (unique_ids[safe_pos] == queried_ids)

    start = offsets[safe_pos]
    end = offsets[safe_pos + 1]
    lengths = jnp.maximum(end - start, 1)
    u = jax.random.uniform(key_bucket, queried_ids.shape, dtype=jnp.float32)
    local_pick = jnp.floor(u * lengths.astype(jnp.float32)).astype(jnp.int32)
    member_pick = jnp.minimum(start + local_pick, members.shape[0] - 1)
    bucket_member = members[member_pick]

    global_member = jax.random.randint(
        key_global,
        queried_ids.shape,
        minval=0,
        maxval=values.shape[0],
        dtype=jnp.int32,
    )
    sampled_member = jnp.where(hit, bucket_member, global_member)
    return values[sampled_member]


def _total_errors_jax(
    formal_w1_error: jax.Array,
    formal_w2_error: jax.Array,
    w1_extra_error: jax.Array,
    w2_extra_error: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    w1_factor = jnp.where(
        jnp.isfinite(w1_extra_error),
        jnp.sqrt(1.0 + w1_extra_error),
        1.0,
    )
    w2_factor = jnp.where(
        jnp.isfinite(w2_extra_error),
        jnp.sqrt(1.0 + w2_extra_error),
        1.0,
    )
    return formal_w1_error * w1_factor, formal_w2_error * w2_factor


def _add_magnitude_error_jax(
    key: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_error: jax.Array,
    w2_error: jax.Array,
    log10_shape_param: jax.Array,
    *,
    error_model_code: int,
) -> tuple[jax.Array, jax.Array]:
    key_w1, key_w2 = jax.random.split(key)
    if error_model_code == _GAUSSIAN_ERROR:
        noise_w1 = jax.random.normal(key_w1, w1.shape, dtype=jnp.float32)
        noise_w2 = jax.random.normal(key_w2, w2.shape, dtype=jnp.float32)
    else:
        shape_param = jnp.power(10.0, log10_shape_param)
        noise_w1 = jax.random.t(key_w1, shape_param, shape=w1.shape, dtype=jnp.float32)
        noise_w2 = jax.random.t(key_w2, shape_param, shape=w2.shape, dtype=jnp.float32)
    return w1 + noise_w1 * w1_error, w2 + noise_w2 * w2_error


def _simulate_one_catwise_jax(
    key: jax.Array,
    parent_count: jax.Array,
    lambda_clus: jax.Array,
    w1_max: jax.Array,
    w1_min: jax.Array,
    w12_min: jax.Array,
    observer_beta: jax.Array,
    forward_matrix: jax.Array,
    inverse_matrix: jax.Array,
    w1_extra_error: jax.Array,
    w2_extra_error: jax.Array,
    log10_magnitude_error_shape_param: jax.Array,
    lookup_tuple: tuple[jax.Array, ...],
    *,
    error_model_code: int,
    nside: int,
    n_chunks: jax.Array,
    chunk_size: int,
    max_children: int,
    cluster_r0_arcsec: float,
    cluster_r_cut_arcsec: float,
) -> tuple[jax.Array, jax.Array]:
    (
        mag_x_flat,
        mag_y_flat,
        mag_x_widths_flat,
        mag_y_widths_flat,
        mag_cdf,
        mask_map,
        log_w1cov_map,
        log_w2cov_map,
        alpha_coefficients,
        hashgrid_mins,
        hashgrid_grid_step,
        hashgrid_grid_nbins,
        hashgrid_unique_ids,
        hashgrid_offsets,
        hashgrid_members,
        hashgrid_values,
    ) = lookup_tuple

    n_pix = mask_map.shape[0]
    source_slots = 1 + max_children
    child_ordinals = jnp.arange(max_children, dtype=jnp.int32)

    def chunk_body(chunk_index: jax.Array, current_density: jax.Array) -> jax.Array:
        chunk_key = jax.random.fold_in(key, chunk_index)
        (
            key_position,
            key_child_counts,
            key_child_position,
            key_magnitude,
            key_hashgrid,
            key_formal_jitter,
            key_noise,
        ) = jax.random.split(chunk_key, 7)

        source_indices = chunk_index * chunk_size + jnp.arange(chunk_size, dtype=jnp.int32)
        parent_valid = source_indices < parent_count

        key_lon, key_lat = jax.random.split(key_position)
        parent_lon = 360.0 * jax.random.uniform(key_lon, (chunk_size,), dtype=jnp.float32)
        parent_lat = jnp.rad2deg(
            jnp.arcsin(
                2.0 * jax.random.uniform(key_lat, (chunk_size,), dtype=jnp.float32)
                - 1.0
            )
        )

        if max_children == 0:
            child_lon = jnp.empty((chunk_size, 0), dtype=jnp.float32)
            child_lat = jnp.empty((chunk_size, 0), dtype=jnp.float32)
            child_valid = jnp.empty((chunk_size, 0), dtype=jnp.bool_)
        else:
            child_counts = jax.random.poisson(
                key_child_counts,
                lam=lambda_clus,
                shape=(chunk_size,),
            ).astype(jnp.int32)
            child_counts = jnp.where(parent_valid, child_counts, 0)
            child_counts = jnp.minimum(child_counts, jnp.int32(max_children))
            child_valid = (
                (child_ordinals[None, :] < child_counts[:, None])
                & parent_valid[:, None]
            )
            child_lon, child_lat = _sample_clustered_positions_jax(
                key_child_position,
                parent_lon,
                parent_lat,
                max_children=max_children,
                cluster_r0_arcsec=cluster_r0_arcsec,
                cluster_r_cut_arcsec=cluster_r_cut_arcsec,
            )

        lon = jnp.concatenate((parent_lon[:, None], child_lon), axis=1).reshape(-1)
        lat = jnp.concatenate((parent_lat[:, None], child_lat), axis=1).reshape(-1)
        source_valid = jnp.concatenate(
            (parent_valid[:, None], child_valid),
            axis=1,
        ).reshape(-1)

        rest_w1, rest_w2 = _sample_magnitudes_jax(
            key_magnitude,
            (chunk_size, source_slots),
            mag_x_flat,
            mag_y_flat,
            mag_x_widths_flat,
            mag_y_widths_flat,
            mag_cdf,
        )
        rest_w1 = rest_w1.reshape(-1)
        rest_w2 = rest_w2.reshape(-1)

        boosted_lon, boosted_lat, angle_to_dipole = _aberrate_points_jax(
            lon,
            lat,
            observer_beta,
            forward_matrix,
            inverse_matrix,
        )
        pixel_indices = jax_ang2pix_nest_lonlat(nside, boosted_lon, boosted_lat)
        in_mask = mask_map[pixel_indices]

        rest_colour = rest_w1 - rest_w2
        spectral_indices = -_fit_alpha_jax(rest_colour, alpha_coefficients)
        boosted_w1 = _boost_magnitudes_jax(
            rest_w1,
            angle_to_dipole,
            observer_beta,
            spectral_indices,
        )
        boosted_w2 = _boost_magnitudes_jax(
            rest_w2,
            angle_to_dipole,
            observer_beta,
            spectral_indices,
        )

        source_logw1_cov = log_w1cov_map[pixel_indices]
        source_logw2_cov = log_w2cov_map[pixel_indices]
        cov_delta = source_logw1_cov - source_logw2_cov
        cov_mean = 0.5 * (source_logw1_cov + source_logw2_cov)
        query = jnp.stack((boosted_w1, cov_delta, boosted_w2, cov_mean), axis=1)
        formal_errors = _sample_hashgrid_jax(
            key_hashgrid,
            query,
            hashgrid_mins,
            hashgrid_grid_step,
            hashgrid_grid_nbins,
            hashgrid_unique_ids,
            hashgrid_offsets,
            hashgrid_members,
            hashgrid_values,
        )
        jitter_w1, jitter_w2 = jax.random.split(key_formal_jitter)
        formal_w1_error = formal_errors[:, 0] + 0.001 * jax.random.normal(
            jitter_w1,
            pixel_indices.shape,
            dtype=jnp.float32,
        )
        formal_w2_error = formal_errors[:, 1] + 0.001 * jax.random.normal(
            jitter_w2,
            pixel_indices.shape,
            dtype=jnp.float32,
        )
        total_w1_error, total_w2_error = _total_errors_jax(
            formal_w1_error,
            formal_w2_error,
            w1_extra_error,
            w2_extra_error,
        )
        observed_w1, observed_w2 = _add_magnitude_error_jax(
            key_noise,
            boosted_w1,
            boosted_w2,
            total_w1_error,
            total_w2_error,
            log10_magnitude_error_shape_param,
            error_model_code=error_model_code,
        )
        observed_w12 = observed_w1 - observed_w2

        keep = (
            source_valid
            & in_mask
            & (observed_w1 < w1_max)
            & (observed_w1 > w1_min)
            & (observed_w12 > w12_min)
        )
        return current_density.at[pixel_indices].add(keep.astype(jnp.float32))

    density = jnp.zeros((n_pix,), dtype=jnp.float32)
    density = jax.lax.fori_loop(
        jnp.asarray(0, dtype=jnp.int32),
        n_chunks,
        chunk_body,
        density,
    )
    return jnp.where(mask_map, density, jnp.nan), mask_map


@partial(
    jax.jit,
    static_argnames=(
        "error_model_code",
        "nside",
        "chunk_size",
        "max_children",
        "cluster_r0_arcsec",
        "cluster_r_cut_arcsec",
    ),
)
def _simulate_catwise_batch_jax(
    keys: jax.Array,
    parent_counts: jax.Array,
    lambda_clus: jax.Array,
    w1_max: jax.Array,
    w1_min: jax.Array,
    w12_min: jax.Array,
    observer_beta: jax.Array,
    forward_matrices: jax.Array,
    inverse_matrices: jax.Array,
    w1_extra_error: jax.Array,
    w2_extra_error: jax.Array,
    log10_magnitude_error_shape_param: jax.Array,
    lookup_tuple: tuple[jax.Array, ...],
    *,
    error_model_code: int,
    nside: int,
    n_chunks: jax.Array,
    chunk_size: int,
    max_children: int,
    cluster_r0_arcsec: float,
    cluster_r_cut_arcsec: float,
) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(
        partial(
            _simulate_one_catwise_jax,
            lookup_tuple=lookup_tuple,
            error_model_code=error_model_code,
            nside=nside,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            max_children=max_children,
            cluster_r0_arcsec=cluster_r0_arcsec,
            cluster_r_cut_arcsec=cluster_r_cut_arcsec,
        )
    )(
        keys,
        parent_counts,
        lambda_clus,
        w1_max,
        w1_min,
        w12_min,
        observer_beta,
        forward_matrices,
        inverse_matrices,
        w1_extra_error,
        w2_extra_error,
        log10_magnitude_error_shape_param,
    )


class CatwiseJax:
    """Fixed-shape JAX implementation of the CatWISE map simulator."""

    def __init__(self, config: CatwiseConfig):
        self.cfg = config
        self.nside = 64
        self.chunk_size = config.chunk_size
        self.downscale_nside = config.downscale_nside
        self.max_cluster_children_per_parent = config.max_cluster_children_per_parent
        self.lookups_are_initialised = False
        self._lookup_arrays: Optional[_CatwiseLookupArrays] = None
        self.mask_map: Optional[NDArray[np.bool_]] = None

        if self.downscale_nside is not None:
            if self.downscale_nside > self.nside:
                raise ValueError("downscale_nside must be <= native nside (64).")
            ratio = self.nside // self.downscale_nside
            if (self.nside % self.downscale_nside) != 0 or (ratio & (ratio - 1)) != 0:
                raise ValueError(
                    "downscale_nside must be a power-of-two divisor of the native nside."
                )

    def initialise_data(self) -> None:
        self._validate_supported_config()
        reference = Catwise(self.cfg)
        reference.initialise_data()

        native_mask = (reference.mask_map == 0).astype(np.bool_, copy=False)
        self.mask_map = native_mask

        sampler = reference.colour_mag_sampler
        mag_cdf = np.cumsum(sampler.probs_flat.astype(np.float32), dtype=np.float32)
        if mag_cdf.size:
            mag_cdf[-1] = 1.0

        hashgrid_unique_ids = reference.hashgrid.unique_grid_ids.astype(np.int32, copy=False)
        hashgrid_offsets = reference.hashgrid.offsets.astype(np.int32, copy=False)
        hashgrid_members = reference.hashgrid.members.astype(np.int32, copy=False)

        self._lookup_arrays = _CatwiseLookupArrays(
            mag_x_flat=jnp.asarray(sampler.x_flat, dtype=jnp.float32),
            mag_y_flat=jnp.asarray(sampler.y_flat, dtype=jnp.float32),
            mag_x_widths_flat=jnp.asarray(sampler.x_widths_flat, dtype=jnp.float32),
            mag_y_widths_flat=jnp.asarray(sampler.y_widths_flat, dtype=jnp.float32),
            mag_cdf=jnp.asarray(mag_cdf, dtype=jnp.float32),
            mask_map=jnp.asarray(native_mask, dtype=jnp.bool_),
            log_w1cov_map=jnp.asarray(reference.log_w1cov_map, dtype=jnp.float32),
            log_w2cov_map=jnp.asarray(reference.log_w2cov_map, dtype=jnp.float32),
            alpha_coefficients=jnp.asarray(
                reference.spectral_lookup.p_W12.c.astype(np.float32),
                dtype=jnp.float32,
            ),
            hashgrid_mins=jnp.asarray(reference.hashgrid.mins, dtype=jnp.float32),
            hashgrid_grid_step=jnp.asarray(reference.hashgrid.grid_step, dtype=jnp.float32),
            hashgrid_grid_nbins=jnp.asarray(
                reference.hashgrid.grid_nbins.astype(np.int32, copy=False),
                dtype=jnp.int32,
            ),
            hashgrid_unique_ids=jnp.asarray(hashgrid_unique_ids, dtype=jnp.int32),
            hashgrid_offsets=jnp.asarray(hashgrid_offsets, dtype=jnp.int32),
            hashgrid_members=jnp.asarray(hashgrid_members, dtype=jnp.int32),
            hashgrid_values=jnp.asarray(reference.hashgrid.grid_values, dtype=jnp.float32),
        )
        self.lookups_are_initialised = True

    def generate_dipole(
        self,
        log10_n_initial_samples: float,
        w1_max: float = 16.4,
        w1_min: float = 9.0,
        w12_min: float = 0.8,
        observer_speed: float = 1.0,
        dipole_longitude: float = CMB_L,
        dipole_latitude: float = CMB_B,
        w1_extra_error: Optional[float] = 1.0,
        w2_extra_error: Optional[float] = 1.0,
        w12_extra_error: Optional[float] = None,
        log10_magnitude_error_shape_param: float = 0.0,
        cluster_rate_param: Optional[float] = 10.0,
        log10_cluster_scale_param: Optional[float] = 3.0,
        lambda_clus: float = 0.0,
        log10_w1conf_scale: Optional[float] = None,
        log10_w2conf_scale: Optional[float] = None,
        key: Optional[jax.Array] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if key is None:
            key = jax.random.PRNGKey(0)
        theta = {
            "log10_n_initial_samples": np.asarray([log10_n_initial_samples]),
            "w1_max": np.asarray([w1_max]),
            "w1_min": np.asarray([w1_min]),
            "w12_min": np.asarray([w12_min]),
            "observer_speed": np.asarray([observer_speed]),
            "dipole_longitude": np.asarray([dipole_longitude]),
            "dipole_latitude": np.asarray([dipole_latitude]),
            "w1_extra_error": np.asarray([np.nan if w1_extra_error is None else w1_extra_error]),
            "w2_extra_error": np.asarray([np.nan if w2_extra_error is None else w2_extra_error]),
            "w12_extra_error": np.asarray([np.nan if w12_extra_error is None else w12_extra_error]),
            "log10_magnitude_error_shape_param": np.asarray(
                [log10_magnitude_error_shape_param]
            ),
            "cluster_rate_param": np.asarray(
                [np.nan if cluster_rate_param is None else cluster_rate_param]
            ),
            "log10_cluster_scale_param": np.asarray(
                [np.nan if log10_cluster_scale_param is None else log10_cluster_scale_param]
            ),
            "lambda_clus": np.asarray([lambda_clus]),
            "log10_w1conf_scale": np.asarray(
                [np.nan if log10_w1conf_scale is None else log10_w1conf_scale]
            ),
            "log10_w2conf_scale": np.asarray(
                [np.nan if log10_w2conf_scale is None else log10_w2conf_scale]
            ),
        }
        maps, masks = self.batch_generate_dipole(theta, key, batch_size=1)
        return maps[0], masks[0]

    def batch_generate_dipole(
        self,
        theta: dict[str, np.ndarray | jax.Array],
        key: jax.Array,
        batch_size: int,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.lookups_are_initialised or self._lookup_arrays is None:
            raise RuntimeError("Run initialise_data() before generating maps.")
        self._validate_supported_config()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        parameters = self._normalise_theta(theta)
        self._validate_parameters(parameters)
        self._warn_if_cluster_cap_overfills(parameters)

        n_sims = parameters["log10_n_initial_samples"].shape[0]
        root_keys = jax.random.split(key, n_sims)
        maps: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        batch_starts = range(0, n_sims, batch_size)
        if show_progress:
            from tqdm import tqdm

            batch_starts = tqdm(
                batch_starts,
                total=(n_sims + batch_size - 1) // batch_size,
                unit="batch",
                desc="simulating",
            )

        for start in batch_starts:
            stop = min(start + batch_size, n_sims)
            chunk = {name: values[start:stop] for name, values in parameters.items()}
            parent_counts = self._parent_counts(chunk)
            n_chunks = max(1, int(math.ceil(int(parent_counts.max(initial=0)) / self.chunk_size)))
            active_max_children = self._active_max_children(chunk)
            forward, inverse = self._rotation_matrices_for_parameters(chunk)
            actual_batch_size = stop - start

            if actual_batch_size < batch_size:
                pad_count = batch_size - actual_batch_size
                chunk = {
                    name: np.pad(values, (0, pad_count), mode="edge")
                    for name, values in chunk.items()
                }
                parent_counts = np.pad(parent_counts, (0, pad_count), mode="constant")
                forward = np.pad(forward, ((0, pad_count), (0, 0), (0, 0)), mode="edge")
                inverse = np.pad(inverse, ((0, pad_count), (0, 0), (0, 0)), mode="edge")
                batch_keys = jax.random.split(jax.random.fold_in(key, start), batch_size)
            else:
                batch_keys = root_keys[start:stop]

            batch_maps, batch_masks = _simulate_catwise_batch_jax(
                keys=jnp.asarray(batch_keys),
                parent_counts=jnp.asarray(parent_counts, dtype=jnp.int32),
                lambda_clus=jnp.asarray(chunk["lambda_clus"], dtype=jnp.float32),
                w1_max=jnp.asarray(chunk["w1_max"], dtype=jnp.float32),
                w1_min=jnp.asarray(chunk["w1_min"], dtype=jnp.float32),
                w12_min=jnp.asarray(chunk["w12_min"], dtype=jnp.float32),
                observer_beta=jnp.asarray(
                    chunk["observer_speed"] * CMB_BETA,
                    dtype=jnp.float32,
                ),
                forward_matrices=jnp.asarray(forward, dtype=jnp.float32),
                inverse_matrices=jnp.asarray(inverse, dtype=jnp.float32),
                w1_extra_error=jnp.asarray(chunk["w1_extra_error"], dtype=jnp.float32),
                w2_extra_error=jnp.asarray(chunk["w2_extra_error"], dtype=jnp.float32),
                log10_magnitude_error_shape_param=jnp.asarray(
                    chunk["log10_magnitude_error_shape_param"],
                    dtype=jnp.float32,
                ),
                lookup_tuple=self._lookup_arrays.as_tuple(),
                error_model_code=self._error_model_code(),
                nside=self.nside,
                n_chunks=jnp.asarray(n_chunks, dtype=jnp.int32),
                chunk_size=self.chunk_size,
                max_children=active_max_children,
                cluster_r0_arcsec=float(self.cfg.cluster_r0_arcsec),
                cluster_r_cut_arcsec=float(self.cfg.cluster_r_cut_arcsec),
            )
            maps.append(np.asarray(batch_maps[:actual_batch_size], dtype=np.float32))
            masks.append(np.asarray(batch_masks[:actual_batch_size], dtype=np.bool_))
            if show_progress:
                batch_starts.set_postfix(completed=stop, total=n_sims)

        out_maps = np.concatenate(maps, axis=0)
        out_masks = np.concatenate(masks, axis=0)
        if self.downscale_nside is not None:
            coarse_maps = []
            coarse_masks = []
            for density_map, mask in zip(out_maps, out_masks):
                coarse_map, coarse_mask = downgrade_ignore_nan(
                    density_map,
                    mask,
                    self.downscale_nside,
                )
                coarse_map = coarse_map.astype(np.float32, copy=False)
                coarse_mask = coarse_mask.astype(np.bool_, copy=False)
                coarse_map = coarse_map.copy()
                coarse_map[~coarse_mask] = np.nan
                coarse_maps.append(coarse_map)
                coarse_masks.append(coarse_mask)
            out_maps = np.stack(coarse_maps, axis=0)
            out_masks = np.stack(coarse_masks, axis=0)
        return out_maps, out_masks

    def _validate_supported_config(self) -> None:
        if self.cfg.store_final_samples:
            raise NotImplementedError(
                "CatwiseJax does not support store_final_samples=True. "
                "Use CatwiseConfig(..., store_final_samples=False)."
            )
        if self.cfg.generate_correlated_points:
            raise NotImplementedError(
                "CatwiseJax does not yet support generate_correlated_points=True."
            )
        if self.cfg.add_confusion_noise:
            raise NotImplementedError(
                "CatwiseJax does not yet support add_confusion_noise=True."
            )

    def _error_model_code(self) -> int:
        if self.cfg.magnitude_error_dist == "gaussian":
            return _GAUSSIAN_ERROR
        if self.cfg.magnitude_error_dist == "students-t":
            return _STUDENTS_T_ERROR
        raise ValueError("magnitude_error_dist must be either 'gaussian' or 'students-t'.")

    def _normalise_theta(
        self,
        theta: dict[str, np.ndarray | jax.Array],
    ) -> dict[str, NDArray[np.float64]]:
        if "log10_n_initial_samples" not in theta:
            raise ValueError("theta must include 'log10_n_initial_samples'.")

        log10_samples = np.asarray(theta["log10_n_initial_samples"], dtype=np.float64)
        if log10_samples.ndim == 0:
            log10_samples = log10_samples[None]
        n_sims = log10_samples.shape[0]

        defaults = {
            "w1_max": 16.4,
            "w1_min": 9.0,
            "w12_min": 0.8,
            "observer_speed": 1.0,
            "dipole_longitude": CMB_L,
            "dipole_latitude": CMB_B,
            "w1_extra_error": 1.0,
            "w2_extra_error": 1.0,
            "w12_extra_error": np.nan,
            "log10_magnitude_error_shape_param": 0.0,
            "cluster_rate_param": 10.0,
            "log10_cluster_scale_param": 3.0,
            "lambda_clus": 0.0,
            "log10_w1conf_scale": np.nan,
            "log10_w2conf_scale": np.nan,
        }
        out: dict[str, NDArray[np.float64]] = {
            "log10_n_initial_samples": log10_samples.astype(np.float64, copy=False),
        }
        for name, default in defaults.items():
            values = np.asarray(theta.get(name, default), dtype=np.float64)
            if values.ndim == 0:
                values = np.full(n_sims, float(values), dtype=np.float64)
            if values.shape != (n_sims,):
                raise ValueError(
                    f"theta['{name}'] must be scalar or have shape ({n_sims},)."
                )
            out[name] = values.astype(np.float64, copy=False)

        if self.cfg.use_common_extra_error:
            out["w2_extra_error"] = out["w1_extra_error"]
        return out

    def _validate_parameters(self, parameters: dict[str, NDArray[np.float64]]) -> None:
        if np.any(~np.isfinite(parameters["log10_n_initial_samples"])):
            raise ValueError("log10_n_initial_samples must be finite.")
        if np.any(parameters["w1_max"] <= parameters["w1_min"]):
            raise ValueError("w1_max must be greater than w1_min.")
        if np.any(~np.isfinite(parameters["w1_max"])):
            raise ValueError("w1_max must be finite.")
        if np.any(~np.isfinite(parameters["w1_min"])):
            raise ValueError("w1_min must be finite.")
        if np.any(~np.isfinite(parameters["w12_min"])):
            raise ValueError("w12_min must be finite.")
        if np.any(~np.isfinite(parameters["observer_speed"])):
            raise ValueError("observer_speed must be finite.")
        if np.any(np.abs(parameters["observer_speed"] * CMB_BETA) >= 1.0):
            raise ValueError("observer_speed produces an invalid beta >= 1.")
        for name in ("w1_extra_error", "w2_extra_error"):
            values = parameters[name]
            finite = np.isfinite(values)
            if np.any(values[finite] < -1.0):
                raise ValueError(f"{name} must be >= -1 when provided.")
        if self.cfg.magnitude_error_dist == "students-t":
            values = parameters["log10_magnitude_error_shape_param"]
            if np.any(~np.isfinite(values)):
                raise ValueError("log10_magnitude_error_shape_param must be finite.")
        if np.any(parameters["lambda_clus"] < 0):
            raise ValueError("lambda_clus must be non-negative.")

    def _warn_if_cluster_cap_overfills(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> None:
        cap = self.max_cluster_children_per_parent
        active = parameters["lambda_clus"] > 0
        if not np.any(active):
            return
        probabilities = poisson.sf(cap, parameters["lambda_clus"])
        max_probability = float(np.max(probabilities[active]))
        if max_probability > _OVERFILL_WARNING_THRESHOLD:
            warnings.warn(
                "CatwiseJax clustering parameters have "
                f"P(children_per_parent > {cap}) = {max_probability:.3g}; "
                "excess children will be truncated.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _active_max_children(self, parameters: dict[str, NDArray[np.float64]]) -> int:
        if np.any(parameters["lambda_clus"] > 0):
            return self.max_cluster_children_per_parent
        return 0

    def _parent_counts(self, parameters: dict[str, NDArray[np.float64]]) -> NDArray[np.int32]:
        expected_counts = np.power(
            10.0,
            parameters["log10_n_initial_samples"],
        ).astype(np.int64)
        expected_multiplicity = 1.0 + parameters["lambda_clus"]
        parent_counts = (expected_counts / expected_multiplicity).astype(np.int64)
        parent_counts = np.maximum(parent_counts, 0)
        if np.any(parent_counts > np.iinfo(np.int32).max):
            raise ValueError("parent source count exceeds int32 range.")
        return parent_counts.astype(np.int32, copy=False)

    def _rotation_matrices_for_parameters(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        forward_matrices = []
        inverse_matrices = []
        for longitude, latitude in zip(
            parameters["dipole_longitude"],
            parameters["dipole_latitude"],
        ):
            forward, inverse = rotation_matrices_for_dipole(
                float(longitude),
                float(latitude),
            )
            forward_matrices.append(forward.astype(np.float32, copy=False))
            inverse_matrices.append(inverse.astype(np.float32, copy=False))
        return np.stack(forward_matrices, axis=0), np.stack(inverse_matrices, axis=0)
