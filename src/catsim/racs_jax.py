from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Optional
import warnings

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via package import.
    if exc.name and (exc.name == "jax" or exc.name.startswith("jax.")):
        raise ImportError(
            "RacsLow3Jax requires the optional JAX dependencies. "
            "Install them with `pip install 'catsim[jax]'`."
        ) from exc
    raise

from .racs import RACS_TEMPERATURE_EPSILON_FLOOR, RACS_LOW3, Racs, RacsConfig
from .utils.constants import CMB_B, CMB_BETA, CMB_L
from .utils.healsphere import downgrade_ignore_nan
from .utils.physics import rotation_matrices_for_dipole


_GEOMETRIC_MODEL = 0
_POISSON_MODEL = 1
_OVERFILL_WARNING_THRESHOLD = 0.01


@dataclass(frozen=True)
class _LookupArrays:
    log_flux_edges: jax.Array
    log_flux_cdf: jax.Array
    mask_map: jax.Array
    tile_counts: jax.Array
    tile_indices: jax.Array
    tile_cdf: jax.Array
    error_counts: jax.Array
    error_values_by_pixel: jax.Array
    global_error_values: jax.Array
    tile_temperature_by_index: jax.Array

    def as_tuple(self) -> tuple[jax.Array, ...]:
        return (
            self.log_flux_edges,
            self.log_flux_cdf,
            self.mask_map,
            self.tile_counts,
            self.tile_indices,
            self.tile_cdf,
            self.error_counts,
            self.error_values_by_pixel,
            self.global_error_values,
            self.tile_temperature_by_index,
        )


def _pad_flat_lookup(
    counts: NDArray[np.integer],
    starts: NDArray[np.integer],
    values: NDArray,
    *,
    fill_value: float | int,
    dtype: np.dtype,
) -> NDArray:
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    max_count = max(int(np.max(counts, initial=0)), 1)
    offsets = np.arange(max_count, dtype=np.int64)[None, :]
    valid = offsets < counts[:, None]
    value_indices = starts[:, None] + offsets

    padded = np.full((counts.size, max_count), fill_value, dtype=dtype)
    if np.any(valid):
        padded[valid] = np.asarray(values, dtype=dtype)[value_indices[valid]]
    return padded


def _spread_bits_jax(values: jax.Array, order: int) -> jax.Array:
    values = values.astype(jnp.int32)
    out = jnp.zeros_like(values, dtype=jnp.int32)
    for bit in range(order):
        out = out | (((values >> bit) & jnp.int32(1)) << jnp.int32(2 * bit))
    return out


def jax_ang2pix_nest_lonlat(
    nside: int,
    lon_deg: jax.Array,
    lat_deg: jax.Array,
) -> jax.Array:
    """JAX-compatible HEALPix NESTED ang2pix for lon/lat degrees."""
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of two.")

    order = int(math.log2(nside))
    nside_i = jnp.int32(nside)
    nside2 = nside_i * nside_i

    phi = jnp.deg2rad(jnp.mod(lon_deg, 360.0))
    z = jnp.sin(jnp.deg2rad(lat_deg))
    za = jnp.abs(z)
    tt = phi / (0.5 * jnp.pi)

    temp1 = nside * (0.5 + tt)
    temp2 = nside * (0.75 * z)
    jp = jnp.floor(temp1 - temp2).astype(jnp.int32)
    jm = jnp.floor(temp1 + temp2).astype(jnp.int32)
    ifp = jp >> order
    ifm = jm >> order
    face_equ = jnp.where(
        ifp == ifm,
        (ifp & jnp.int32(3)) + jnp.int32(4),
        jnp.where(ifp < ifm, ifp & jnp.int32(3), (ifm & jnp.int32(3)) + jnp.int32(8)),
    )
    ix_equ = jm & (nside_i - jnp.int32(1))
    iy_equ = nside_i - (jp & (nside_i - jnp.int32(1))) - jnp.int32(1)
    ipf_equ = _spread_bits_jax(ix_equ, order) | (_spread_bits_jax(iy_equ, order) << 1)
    pix_equ = face_equ * nside2 + ipf_equ

    ntt = jnp.minimum(jnp.floor(tt).astype(jnp.int32), jnp.int32(3))
    tp = tt - ntt.astype(tt.dtype)
    tmp = nside * jnp.sqrt(3.0 * (1.0 - za))
    jp_pol = jnp.minimum(jnp.floor(tp * tmp).astype(jnp.int32), nside_i - jnp.int32(1))
    jm_pol = jnp.minimum(
        jnp.floor((1.0 - tp) * tmp).astype(jnp.int32),
        nside_i - jnp.int32(1),
    )
    north = z > 0.0
    face_pol = jnp.where(north, ntt, ntt + jnp.int32(8))
    ix_pol = jnp.where(north, nside_i - jm_pol - jnp.int32(1), jp_pol)
    iy_pol = jnp.where(north, nside_i - jp_pol - jnp.int32(1), jm_pol)
    ipf_pol = _spread_bits_jax(ix_pol, order) | (_spread_bits_jax(iy_pol, order) << 1)
    pix_pol = face_pol * nside2 + ipf_pol

    return jnp.where(za <= (2.0 / 3.0), pix_equ, pix_pol).astype(jnp.int32)


def _sample_fluxes_jax(
    key: jax.Array,
    shape: tuple[int, ...],
    log_flux_edges: jax.Array,
    log_flux_cdf: jax.Array,
) -> jax.Array:
    key_bin, key_pos = jax.random.split(key)
    u_bin = jax.random.uniform(key_bin, shape, dtype=jnp.float32)
    u_pos = jax.random.uniform(key_pos, shape, dtype=jnp.float32)
    bin_idx = jnp.searchsorted(log_flux_cdf, u_bin, side="right")
    bin_idx = jnp.clip(bin_idx, 0, log_flux_edges.shape[0] - 2)
    low = log_flux_edges[bin_idx]
    high = log_flux_edges[bin_idx + 1]
    return jnp.power(10.0, low + (high - low) * u_pos)


def _aberrate_points_jax(
    lon_deg: jax.Array,
    lat_deg: jax.Array,
    observer_beta: jax.Array,
    forward_matrix: jax.Array,
    inverse_matrix: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    lon_rad = jnp.deg2rad(lon_deg)
    lat_rad = jnp.deg2rad(lat_deg)
    cos_lat = jnp.cos(lat_rad)
    x = cos_lat * jnp.cos(lon_rad)
    y = cos_lat * jnp.sin(lon_rad)
    z = jnp.sin(lat_rad)

    dipole_x = forward_matrix[0, 0] * x + forward_matrix[0, 1] * y + forward_matrix[0, 2] * z
    dipole_y = forward_matrix[1, 0] * x + forward_matrix[1, 1] * y + forward_matrix[1, 2] * z
    dipole_z = forward_matrix[2, 0] * x + forward_matrix[2, 1] * y + forward_matrix[2, 2] * z

    dipole_lon = jnp.mod(jnp.rad2deg(jnp.arctan2(dipole_y, dipole_x)) + 360.0, 360.0)
    source_to_dipole_angle = jnp.rad2deg(jnp.arccos(jnp.clip(dipole_z, -1.0, 1.0)))
    source_angle_rad = jnp.deg2rad(source_to_dipole_angle)
    cos_source_angle = jnp.cos(source_angle_rad)
    boosted_angle = jnp.rad2deg(
        jnp.arccos(
            jnp.clip(
                (observer_beta + cos_source_angle)
                / (observer_beta * cos_source_angle + 1.0),
                -1.0,
                1.0,
            )
        )
    )
    boosted_lat = 90.0 - boosted_angle
    boosted_lat_rad = jnp.deg2rad(boosted_lat)
    boosted_lon_rad = jnp.deg2rad(dipole_lon)
    cos_boosted_lat = jnp.cos(boosted_lat_rad)
    boosted_x = cos_boosted_lat * jnp.cos(boosted_lon_rad)
    boosted_y = cos_boosted_lat * jnp.sin(boosted_lon_rad)
    boosted_z = jnp.sin(boosted_lat_rad)

    native_x = (
        inverse_matrix[0, 0] * boosted_x
        + inverse_matrix[0, 1] * boosted_y
        + inverse_matrix[0, 2] * boosted_z
    )
    native_y = (
        inverse_matrix[1, 0] * boosted_x
        + inverse_matrix[1, 1] * boosted_y
        + inverse_matrix[1, 2] * boosted_z
    )
    native_z = (
        inverse_matrix[2, 0] * boosted_x
        + inverse_matrix[2, 1] * boosted_y
        + inverse_matrix[2, 2] * boosted_z
    )
    out_lon = jnp.mod(jnp.rad2deg(jnp.arctan2(native_y, native_x)) + 360.0, 360.0)
    out_lat = jnp.rad2deg(jnp.arcsin(jnp.clip(native_z, -1.0, 1.0)))
    return out_lon, out_lat, source_to_dipole_angle


def _cluster_counts_jax(
    key: jax.Array,
    parent_valid: jax.Array,
    p_clus: jax.Array,
    clus_stop_prob: jax.Array,
    lambda_clus: jax.Array,
    *,
    cluster_model_code: int,
    max_children: int,
) -> jax.Array:
    if max_children == 0:
        return jnp.zeros(parent_valid.shape, dtype=jnp.int32)

    if cluster_model_code == _GEOMETRIC_MODEL:
        key_select, key_geom = jax.random.split(key)
        selected = jax.random.uniform(key_select, parent_valid.shape) < p_clus
        u = jax.random.uniform(key_geom, parent_valid.shape, minval=0.0, maxval=1.0)
        raw_counts = jnp.floor(jnp.log1p(-u) / jnp.log1p(-clus_stop_prob)).astype(jnp.int32) + 1
        counts = jnp.where(selected, raw_counts, 0)
    else:
        counts = jax.random.poisson(
            key,
            lam=lambda_clus,
            shape=parent_valid.shape,
        ).astype(jnp.int32)

    counts = jnp.where(parent_valid, counts, 0)
    return jnp.minimum(counts, jnp.int32(max_children))


def _sample_clustered_positions_jax(
    key: jax.Array,
    parent_ra_deg: jax.Array,
    parent_dec_deg: jax.Array,
    *,
    max_children: int,
    cluster_r0_arcsec: float,
    cluster_r_cut_arcsec: float,
) -> tuple[jax.Array, jax.Array]:
    shape = (parent_ra_deg.shape[0], max_children)
    key_phi, key_radial = jax.random.split(key)
    phi = jax.random.uniform(key_phi, shape, minval=0.0, maxval=2.0 * jnp.pi)
    u_radial = jax.random.uniform(key_radial, shape, minval=0.0, maxval=1.0)
    radial_arcsec = cluster_r_cut_arcsec - cluster_r0_arcsec * jnp.log1p(-u_radial)
    distance_rad = jnp.deg2rad(radial_arcsec / 3600.0)

    parent_ra_rad = jnp.deg2rad(parent_ra_deg)[:, None]
    parent_dec_rad = jnp.deg2rad(parent_dec_deg)[:, None]
    sin_parent_dec = jnp.sin(parent_dec_rad)
    cos_parent_dec = jnp.cos(parent_dec_rad)
    sin_distance = jnp.sin(distance_rad)
    cos_distance = jnp.cos(distance_rad)

    child_dec_rad = jnp.arcsin(
        jnp.clip(
            sin_parent_dec * cos_distance
            + cos_parent_dec * sin_distance * jnp.cos(phi),
            -1.0,
            1.0,
        )
    )
    child_ra_rad = parent_ra_rad + jnp.arctan2(
        jnp.sin(phi) * sin_distance * cos_parent_dec,
        cos_distance - sin_parent_dec * jnp.sin(child_dec_rad),
    )
    child_ra_deg = jnp.mod(jnp.rad2deg(child_ra_rad), 360.0)
    child_dec_deg = jnp.rad2deg(child_dec_rad)
    return child_ra_deg, child_dec_deg


def _simulate_one_jax(
    key: jax.Array,
    parent_count: jax.Array,
    flux_min: jax.Array,
    p_clus: jax.Array,
    clus_stop_prob: jax.Array,
    lambda_clus: jax.Array,
    observer_beta: jax.Array,
    forward_matrix: jax.Array,
    inverse_matrix: jax.Array,
    temp_beta: jax.Array,
    fractional_error_eta: jax.Array,
    lookup_tuple: tuple[jax.Array, ...],
    *,
    cluster_model_code: int,
    nside: int,
    n_chunks: jax.Array,
    chunk_size: int,
    max_children: int,
    alpha_mean: float,
    alpha_sigma: float,
    cluster_r0_arcsec: float,
    cluster_r_cut_arcsec: float,
    paf_reference_temp_c: float,
) -> tuple[jax.Array, jax.Array]:
    (
        log_flux_edges,
        log_flux_cdf,
        mask_map,
        tile_counts,
        tile_indices,
        tile_cdf,
        error_counts,
        error_values_by_pixel,
        global_error_values,
        tile_temperature_by_index,
    ) = lookup_tuple

    n_pix = mask_map.shape[0]
    source_slots = 1 + max_children
    child_ordinals = jnp.arange(max_children, dtype=jnp.int32)
    error_max_count = error_values_by_pixel.shape[1]

    def chunk_body(accumulator: jax.Array, chunk_index: jax.Array) -> tuple[jax.Array, None]:
        chunk_key = jax.random.fold_in(key, chunk_index)
        (
            key_parent_pos,
            key_counts,
            key_child_pos,
            key_flux,
            key_alpha,
            key_tile,
            key_error,
            key_global_error,
            key_noise,
        ) = jax.random.split(chunk_key, 9)

        parent_indices = chunk_index * chunk_size + jnp.arange(chunk_size, dtype=jnp.int32)
        parent_valid = parent_indices < parent_count

        key_ra, key_dec = jax.random.split(key_parent_pos)
        parent_ra = 360.0 * jax.random.uniform(key_ra, (chunk_size,), dtype=jnp.float32)
        parent_dec = jnp.rad2deg(
            jnp.arcsin(
                2.0 * jax.random.uniform(key_dec, (chunk_size,), dtype=jnp.float32) - 1.0
            )
        )

        child_counts = _cluster_counts_jax(
            key_counts,
            parent_valid,
            p_clus,
            clus_stop_prob,
            lambda_clus,
            cluster_model_code=cluster_model_code,
            max_children=max_children,
        )
        child_valid = (child_ordinals[None, :] < child_counts[:, None]) & parent_valid[:, None]
        child_ra, child_dec = _sample_clustered_positions_jax(
            key_child_pos,
            parent_ra,
            parent_dec,
            max_children=max_children,
            cluster_r0_arcsec=cluster_r0_arcsec,
            cluster_r_cut_arcsec=cluster_r_cut_arcsec,
        )

        source_ra = jnp.concatenate((parent_ra[:, None], child_ra), axis=1).reshape(-1)
        source_dec = jnp.concatenate((parent_dec[:, None], child_dec), axis=1).reshape(-1)
        source_valid = jnp.concatenate((parent_valid[:, None], child_valid), axis=1).reshape(-1)
        source_shape = (chunk_size, source_slots)

        intrinsic_flux = _sample_fluxes_jax(
            key_flux,
            source_shape,
            log_flux_edges,
            log_flux_cdf,
        ).reshape(-1)
        alpha = (
            alpha_mean
            + alpha_sigma * jax.random.normal(key_alpha, source_shape, dtype=jnp.float32)
        ).reshape(-1)

        boosted_ra, boosted_dec, angle_to_dipole = _aberrate_points_jax(
            source_ra,
            source_dec,
            observer_beta,
            forward_matrix,
            inverse_matrix,
        )
        gamma = 1.0 / jnp.sqrt(1.0 - observer_beta**2)
        delta = gamma * (1.0 + observer_beta * jnp.cos(jnp.deg2rad(angle_to_dipole)))
        dipole_flux = intrinsic_flux * jnp.power(delta, 1.0 + alpha)

        pixel_indices = jax_ang2pix_nest_lonlat(nside, boosted_ra, boosted_dec)
        in_mask = mask_map[pixel_indices]

        tile_count = tile_counts[pixel_indices]
        safe_tile_count = jnp.maximum(tile_count, 1)
        tile_u = jax.random.uniform(key_tile, pixel_indices.shape, dtype=jnp.float32)
        pixel_tile_cdf = tile_cdf[pixel_indices]
        tile_choice = jnp.sum(tile_u[:, None] > pixel_tile_cdf, axis=1).astype(jnp.int32)
        tile_choice = jnp.minimum(tile_choice, safe_tile_count - 1)
        sampled_tile = tile_indices[pixel_indices, tile_choice]
        sampled_tile = jnp.where(tile_count > 0, sampled_tile, -1)

        safe_tile = jnp.maximum(sampled_tile, 0)
        temperatures = tile_temperature_by_index[safe_tile]
        valid_temperature = (sampled_tile >= 0) & jnp.isfinite(temperatures)
        hot_temperature = jnp.maximum(temperatures - paf_reference_temp_c, 0.0)
        enhancement = jnp.where(valid_temperature, 1.0 - temp_beta * hot_temperature, 1.0)
        enhancement = jnp.maximum(enhancement, RACS_TEMPERATURE_EPSILON_FLOOR)
        systematics_flux = dipole_flux * enhancement

        error_count = error_counts[pixel_indices]
        safe_error_count = jnp.maximum(error_count, 1)
        error_u = jax.random.uniform(key_error, pixel_indices.shape, dtype=jnp.float32)
        error_choice = jnp.floor(error_u * safe_error_count.astype(jnp.float32)).astype(jnp.int32)
        error_choice = jnp.minimum(error_choice, error_max_count - 1)
        pixel_fractional_error = error_values_by_pixel[pixel_indices, error_choice]

        global_choice = jax.random.randint(
            key_global_error,
            pixel_indices.shape,
            minval=0,
            maxval=global_error_values.shape[0],
            dtype=jnp.int32,
        )
        global_fractional_error = global_error_values[global_choice]
        base_fractional_error = jnp.where(
            error_count > 0,
            pixel_fractional_error,
            global_fractional_error,
        )
        flux_sigma = (
            base_fractional_error
            * systematics_flux
            * jnp.sqrt(1.0 + fractional_error_eta)
        )
        observed_flux = systematics_flux + jax.random.normal(
            key_noise,
            pixel_indices.shape,
            dtype=jnp.float32,
        ) * flux_sigma

        keep = source_valid & in_mask & (sampled_tile >= 0) & (observed_flux >= flux_min)
        accumulator = accumulator.at[pixel_indices].add(keep.astype(jnp.float32))
        return accumulator, None

    density = jnp.zeros((n_pix,), dtype=jnp.float32)

    def fori_body(chunk_index: jax.Array, current_density: jax.Array) -> jax.Array:
        current_density, _ = chunk_body(current_density, chunk_index)
        return current_density

    density = jax.lax.fori_loop(
        jnp.asarray(0, dtype=jnp.int32),
        n_chunks,
        fori_body,
        density,
    )
    return jnp.where(mask_map, density, jnp.nan), mask_map


@partial(
    jax.jit,
    static_argnames=(
        "cluster_model_code",
        "nside",
        "chunk_size",
        "max_children",
        "alpha_mean",
        "alpha_sigma",
        "cluster_r0_arcsec",
        "cluster_r_cut_arcsec",
        "paf_reference_temp_c",
    ),
)
def _simulate_batch_jax(
    keys: jax.Array,
    parent_counts: jax.Array,
    flux_mins: jax.Array,
    p_clus: jax.Array,
    clus_stop_prob: jax.Array,
    lambda_clus: jax.Array,
    observer_beta: jax.Array,
    forward_matrices: jax.Array,
    inverse_matrices: jax.Array,
    temp_beta: jax.Array,
    fractional_error_eta: jax.Array,
    lookup_tuple: tuple[jax.Array, ...],
    *,
    cluster_model_code: int,
    nside: int,
    n_chunks: jax.Array,
    chunk_size: int,
    max_children: int,
    alpha_mean: float,
    alpha_sigma: float,
    cluster_r0_arcsec: float,
    cluster_r_cut_arcsec: float,
    paf_reference_temp_c: float,
) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(
        partial(
            _simulate_one_jax,
            lookup_tuple=lookup_tuple,
            cluster_model_code=cluster_model_code,
            nside=nside,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            max_children=max_children,
            alpha_mean=alpha_mean,
            alpha_sigma=alpha_sigma,
            cluster_r0_arcsec=cluster_r0_arcsec,
            cluster_r_cut_arcsec=cluster_r_cut_arcsec,
            paf_reference_temp_c=paf_reference_temp_c,
        )
    )(
        keys,
        parent_counts,
        flux_mins,
        p_clus,
        clus_stop_prob,
        lambda_clus,
        observer_beta,
        forward_matrices,
        inverse_matrices,
        temp_beta,
        fractional_error_eta,
    )


class RacsJax:
    """Fixed-shape JAX implementation of the RACS map simulator."""

    def __init__(self, config: RacsConfig):
        self.cfg = config
        self.nside = config.nside
        self.chunk_size = config.chunk_size
        self.downscale_nside = config.downscale_nside
        self.max_cluster_children_per_parent = config.max_cluster_children_per_parent
        self.lookups_are_initialised = False
        self._lookup_arrays: Optional[_LookupArrays] = None
        self.mask_map: Optional[NDArray[np.bool_]] = None

    def initialise_data(self) -> None:
        reference = Racs(self.cfg)
        reference.initialise_data()

        self.mask_map = reference.mask_map.astype(np.bool_, copy=False)

        tile_counts = reference.sbid_mixture_counts.astype(np.int32, copy=False)
        tile_indices = _pad_flat_lookup(
            tile_counts,
            reference.sbid_mixture_starts,
            reference.sbid_mixture_tile_indices,
            fill_value=-1,
            dtype=np.int32,
        )
        tile_probabilities = _pad_flat_lookup(
            tile_counts,
            reference.sbid_mixture_starts,
            reference.sbid_mixture_probabilities,
            fill_value=0.0,
            dtype=np.float32,
        )
        tile_cdf = np.cumsum(tile_probabilities, axis=1, dtype=np.float32)

        error_counts = reference.error_lookup_pixel_counts.astype(np.int32, copy=False)
        error_values_by_pixel = _pad_flat_lookup(
            error_counts,
            reference.error_lookup_pixel_starts,
            reference.error_lookup_fractional_values,
            fill_value=0.0,
            dtype=np.float32,
        )

        if reference.tile_temperature_by_index is None:
            tile_temperature_by_index = np.full(
                max(int(reference.tile_sbids.size), 1),
                np.nan,
                dtype=np.float32,
            )
        else:
            tile_temperature_by_index = reference.tile_temperature_by_index.astype(
                np.float32,
                copy=False,
            )

        self._lookup_arrays = _LookupArrays(
            log_flux_edges=jnp.asarray(reference.log_flux_bin_edges, dtype=jnp.float32),
            log_flux_cdf=jnp.asarray(reference.log_flux_bin_cdf, dtype=jnp.float32),
            mask_map=jnp.asarray(self.mask_map, dtype=jnp.bool_),
            tile_counts=jnp.asarray(tile_counts, dtype=jnp.int32),
            tile_indices=jnp.asarray(tile_indices, dtype=jnp.int32),
            tile_cdf=jnp.asarray(tile_cdf, dtype=jnp.float32),
            error_counts=jnp.asarray(error_counts, dtype=jnp.int32),
            error_values_by_pixel=jnp.asarray(error_values_by_pixel, dtype=jnp.float32),
            global_error_values=jnp.asarray(
                reference.error_lookup_fractional_values,
                dtype=jnp.float32,
            ),
            tile_temperature_by_index=jnp.asarray(
                tile_temperature_by_index,
                dtype=jnp.float32,
            ),
        )
        self.lookups_are_initialised = True

    def generate_dipole(
        self,
        log10_n_initial_samples: float,
        flux_min: Optional[float] = None,
        p_clus: float = 0.0,
        clus_stop_prob: float = 1.0,
        lambda_clus: float = 0.0,
        observer_speed: float = 1.0,
        dipole_longitude: float = CMB_L,
        dipole_latitude: float = CMB_B,
        temp_beta: float = 0.0,
        fractional_error_eta: float = 0.0,
        key: Optional[jax.Array] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if key is None:
            key = jax.random.PRNGKey(0)
        theta = {
            "log10_n_initial_samples": np.asarray([log10_n_initial_samples]),
            "flux_min": np.asarray([self.cfg.flux_min if flux_min is None else flux_min]),
            "p_clus": np.asarray([p_clus]),
            "clus_stop_prob": np.asarray([clus_stop_prob]),
            "lambda_clus": np.asarray([lambda_clus]),
            "observer_speed": np.asarray([observer_speed]),
            "dipole_longitude": np.asarray([dipole_longitude]),
            "dipole_latitude": np.asarray([dipole_latitude]),
            "temp_beta": np.asarray([temp_beta]),
            "fractional_error_eta": np.asarray([fractional_error_eta]),
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
        if self.cfg.store_final_samples:
            raise NotImplementedError(
                "RacsJax does not support store_final_samples=True. "
                "Use RacsConfig(..., store_final_samples=False)."
            )
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

            batch_maps, batch_masks = _simulate_batch_jax(
                keys=jnp.asarray(batch_keys),
                parent_counts=jnp.asarray(parent_counts, dtype=jnp.int32),
                flux_mins=jnp.asarray(chunk["flux_min"], dtype=jnp.float32),
                p_clus=jnp.asarray(chunk["p_clus"], dtype=jnp.float32),
                clus_stop_prob=jnp.asarray(chunk["clus_stop_prob"], dtype=jnp.float32),
                lambda_clus=jnp.asarray(chunk["lambda_clus"], dtype=jnp.float32),
                observer_beta=jnp.asarray(chunk["observer_speed"] * CMB_BETA, dtype=jnp.float32),
                forward_matrices=jnp.asarray(forward, dtype=jnp.float32),
                inverse_matrices=jnp.asarray(inverse, dtype=jnp.float32),
                temp_beta=jnp.asarray(chunk["temp_beta"], dtype=jnp.float32),
                fractional_error_eta=jnp.asarray(
                    chunk["fractional_error_eta"],
                    dtype=jnp.float32,
                ),
                lookup_tuple=self._lookup_arrays.as_tuple(),
                cluster_model_code=self._cluster_model_code(),
                nside=self.nside,
                n_chunks=jnp.asarray(n_chunks, dtype=jnp.int32),
                chunk_size=self.chunk_size,
                max_children=self.max_cluster_children_per_parent,
                alpha_mean=float(self.cfg.alpha_mean),
                alpha_sigma=float(self.cfg.alpha_sigma),
                cluster_r0_arcsec=float(self.cfg.cluster_r0_arcsec),
                cluster_r_cut_arcsec=float(self.cfg.cluster_r_cut_arcsec),
                paf_reference_temp_c=float(self.cfg.paf_reference_temp_c),
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

    def _cluster_model_code(self) -> int:
        if self.cfg.cluster_count_model == "geometric":
            return _GEOMETRIC_MODEL
        if self.cfg.cluster_count_model == "poisson":
            return _POISSON_MODEL
        raise ValueError("cluster_count_model must be either 'geometric' or 'poisson'.")

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
            "flux_min": self.cfg.flux_min,
            "p_clus": 0.0,
            "clus_stop_prob": 1.0,
            "lambda_clus": 0.0,
            "observer_speed": 1.0,
            "dipole_longitude": CMB_L,
            "dipole_latitude": CMB_B,
            "temp_beta": 0.0,
            "fractional_error_eta": 0.0,
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
        return out

    def _validate_parameters(self, parameters: dict[str, NDArray[np.float64]]) -> None:
        if np.any(~np.isfinite(parameters["log10_n_initial_samples"])):
            raise ValueError("log10_n_initial_samples must be finite.")
        if np.any(parameters["flux_min"] <= 0):
            raise ValueError("flux_min must be positive.")
        if np.any(parameters["temp_beta"] < 0) or np.any(~np.isfinite(parameters["temp_beta"])):
            raise ValueError("temp_beta must be finite and non-negative.")
        if np.any(parameters["fractional_error_eta"] < 0):
            raise ValueError("fractional_error_eta must be non-negative.")

        if self.cfg.cluster_count_model == "geometric":
            if np.any((parameters["p_clus"] < 0) | (parameters["p_clus"] > 1)):
                raise ValueError(
                    "For cluster_count_model='geometric', p_clus must lie in [0, 1]."
                )
            if np.any(
                (parameters["clus_stop_prob"] <= 0) | (parameters["clus_stop_prob"] > 1)
            ):
                raise ValueError(
                    "For cluster_count_model='geometric', "
                    "clus_stop_prob must lie in (0, 1]."
                )
            if np.any(parameters["lambda_clus"] != 0):
                raise ValueError(
                    "lambda_clus is only valid for cluster_count_model='poisson'."
                )
        else:
            if np.any(parameters["lambda_clus"] < 0):
                raise ValueError(
                    "For cluster_count_model='poisson', lambda_clus must be non-negative."
                )
            if np.any(parameters["p_clus"] != 0):
                raise ValueError(
                    "p_clus is only valid for cluster_count_model='geometric'."
                )
            if np.any(parameters["clus_stop_prob"] != 1.0):
                raise ValueError(
                    "clus_stop_prob is only valid for cluster_count_model='geometric'."
                )

    def _warn_if_cluster_cap_overfills(
        self,
        parameters: dict[str, NDArray[np.float64]],
    ) -> None:
        cap = self.max_cluster_children_per_parent
        if self.cfg.cluster_count_model == "geometric":
            probabilities = parameters["p_clus"] * np.power(
                1.0 - parameters["clus_stop_prob"],
                cap,
            )
            active = parameters["p_clus"] > 0
            model_name = "geometric"
        else:
            probabilities = poisson.sf(cap, parameters["lambda_clus"])
            active = parameters["lambda_clus"] > 0
            model_name = "poisson"

        if not np.any(active):
            return
        max_probability = float(np.max(probabilities[active]))
        if max_probability > _OVERFILL_WARNING_THRESHOLD:
            warnings.warn(
                "RacsJax clustering parameters have "
                f"P(children_per_parent > {cap}) = {max_probability:.3g} "
                f"for the {model_name} model; excess children will be truncated.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _parent_counts(self, parameters: dict[str, NDArray[np.float64]]) -> NDArray[np.int32]:
        n_expected = np.power(10.0, parameters["log10_n_initial_samples"]).astype(np.int64)
        if self.cfg.cluster_count_model == "geometric":
            expected_multiplicity = 1.0 + (
                parameters["p_clus"] / parameters["clus_stop_prob"]
            )
        else:
            expected_multiplicity = 1.0 + parameters["lambda_clus"]
        parent_counts = (n_expected / expected_multiplicity).astype(np.int64)
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
            dipole_ra, dipole_dec = self._galactic_to_equatorial(
                float(longitude),
                float(latitude),
            )
            forward, inverse = rotation_matrices_for_dipole(dipole_ra, dipole_dec)
            forward_matrices.append(forward.astype(np.float32, copy=False))
            inverse_matrices.append(inverse.astype(np.float32, copy=False))
        return np.stack(forward_matrices, axis=0), np.stack(inverse_matrices, axis=0)

    @staticmethod
    def _galactic_to_equatorial(
        galactic_longitude: float,
        galactic_latitude: float,
    ) -> tuple[float, float]:
        coord = SkyCoord(
            l=galactic_longitude * u.deg,
            b=galactic_latitude * u.deg,
            frame="galactic",
        )
        equatorial = coord.icrs
        return float(equatorial.ra.deg), float(equatorial.dec.deg)


class RacsLow3Jax(RacsJax):
    """Backwards-compatible LOW3 JAX simulator wrapper."""

    def __init__(self, config: RacsConfig):
        if config.product != RACS_LOW3:
            raise ValueError("RacsLow3Jax requires a LOW3 RACS product configuration.")
        super().__init__(config)
