from typing import Optional, Literal
from astropy.table import Table, join
from catsim.utils.hashgrid import HashGrid
from catsim.utils.plotting import plot_adaptive_bins
from .configs import CatwiseConfig
from .utils.hists import (
    MultinomialSample2DHistogram,
    build_bin_lookup_grid, 
    kdtree_binned_distribution_2d, 
    load_sigma_bins,
    save_sigma_bins, 
    sample_sigma_w1w2_from_bins_vectorized_fast
)
from .utils.physics import (
    generate_clusters, sample_spherical_points, aberrate_points, boost_magnitudes,
    rotation_matrices_for_dipole, spherical_to_cart_deg
)
from .utils.constants import *
from .utils.spec_idx import AlphaLookup
import numpy as np
import healpy as hp
from tqdm import tqdm
import os
from pathlib import Path
from .utils.healsphere import Mask, ParameterMap, downgrade_ignore_nan
from .utils.package_data import data_path
from collections import defaultdict
import pickle
from scipy.stats import binned_statistic_2d
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import RegularGridInterpolator
from numpy.typing import NDArray
from datetime import datetime
import matplotlib.pyplot as plt
from .utils.rng import NPKey


class Catwise:
    def __init__(self, config: CatwiseConfig):
        '''
        :param config: Instance of a CatwiseConfig; determines the behaviour of
            the `generate_dipole` method.
        '''
        self.nside = 64
        self.cfg = config

        self.downscale_nside = self.cfg.downscale_nside
        if self.downscale_nside is not None:
            if self.downscale_nside > self.nside:
                raise ValueError('downscale_nside must be ≤ native nside (64).')
            ratio = self.nside // self.downscale_nside
            if (self.nside % self.downscale_nside) != 0 or (ratio & (ratio - 1)) != 0:
                raise ValueError(
                    'downscale_nside must be a power-of-two divisor of the native nside.'
                )

        self.dtype = np.float32 if self.cfg.use_float32 else np.float64
        print(f'Using {self.dtype} for intermediate variables...')

        self.magnitude_error_dist: Literal['gaussian', 'students-t'] = (
            self.cfg.magnitude_error_dist
        )
        print(f'Using {self.magnitude_error_dist} distribution for mag errors...')
        
        self.store_final_samples = self.cfg.store_final_samples
        self.chunk_size = self.cfg.chunk_size
        self.use_common_extra_error = self.cfg.use_common_extra_error
        self.use_noecl_mask = self.cfg.use_noecl_mask

        self.generate_correlated_points = self.cfg.generate_correlated_points
        self.add_confusion_noise = self.cfg.add_confusion_noise

        self.dipole_longitude = CMB_L
        self.dipole_latitude = CMB_B
        self.observer_speed = CMB_BETA
        self.cat_w1_max = self.cfg.cat_w1_max
        self.cat_w12_min = self.cfg.cat_w12_min
        self.cut_path = self._get_cut_path(self.cat_w1_max, self.cat_w12_min)
        self.file_name = (
            f'catwise2020_corr_w12{self.cat_w12_min_str}_w1{self.cat_w1_max_str}.fits'
        )
        self.catalogue_is_loaded = False
        self.lookups_are_initialised = False
        self.s21_cat_fname = (
            'catwise_agns_masked_final_w1lt16p5_alpha.fits'
        )
        self.s21_catalogue_path = self._resolve_s21_path()

        self._coarse_density_map: Optional[NDArray[np.float32]] = None
        self._coarse_mask: Optional[NDArray[np.bool_]] = None
        self._coarse_real_density_map: Optional[NDArray[np.float32]] = None
        self._coarse_real_mask: Optional[NDArray[np.bool_]] = None
    
    def _get_cut_path(self,
            cat_w1_max: float,
            cat_w12_min: float
        ) -> str:
        self.cat_w1_max_str = str(cat_w1_max).replace('.', 'p')
        self.cat_w12_min_str = str(cat_w12_min).replace('.', 'p')
        return f'{self.cat_w12_min_str}_{self.cat_w1_max_str}'

    def _resolve_s21_path(self) -> Path:
        env_override = os.environ.get('CATSIM_S21_PATH')
        if env_override:
            override_path = Path(env_override).expanduser()
            if override_path.exists():
                return override_path
            raise FileNotFoundError(
                f"CATSIM_S21_PATH points to missing file: {override_path}"
            )

        if self.cfg.s21_catalogue_path is not None:
            config_path = Path(self.cfg.s21_catalogue_path).expanduser()
            if config_path.exists():
                return config_path
            raise FileNotFoundError(
                'CatwiseConfig.s21_catalogue_path points to missing file: '
                f'{config_path}'
            )

        try:
            with data_path(self.s21_cat_fname) as bundled_path:
                if bundled_path.exists():
                    return bundled_path
        except FileNotFoundError:
            pass

        raise FileNotFoundError(
            'CatWISE S21 catalogue not found. Provide the FITS file via '
            'CatwiseConfig.s21_catalogue_path or set CATSIM_S21_PATH to the '
            f'full path of {self.s21_cat_fname}.'
        )

    def load_catalogue(self):
        self.file_path = f'src/catsim/data/{self.file_name}'
        print('Loading CatWISE2020...')
        self.catalogue = Table.read(
            self.file_path,
            unit_parse_strict='silent' # supress unit warning printouts
        )
        print('Finished loading CatWISE2020.')
        self.catalogue_is_loaded = True
    
    def generate_dipole(self,
            log10_n_initial_samples: float,
            w1_max: float = 16.4,
            w1_min: float = 9.,
            w12_min: float = 0.8,
            observer_speed: float = 1.,
            dipole_longitude: float = CMB_L,
            dipole_latitude: float = CMB_B,
            w1_extra_error: Optional[float] = 1.,
            w2_extra_error: Optional[float] = 1.,
            w12_extra_error: Optional[float] = None,
            log10_magnitude_error_shape_param: float = 0.,
            cluster_rate_param: Optional[float] = 10.,
            log10_cluster_scale_param: Optional[float] = 3.,
            lambda_clus: float = 0.0,
            log10_w1conf_scale: Optional[float] = None,
            log10_w2conf_scale: Optional[float] = None,
            rng_key: Optional[NPKey] = None,
        ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        '''
        :param observer_speed: Observer speed in units of CMB-derived speed.
        '''
        assert self.lookups_are_initialised, (
            "Lookup tables must be initialised before generating dipoles. "
            "Run this class's `initialise_data` method first."
        )

        self.observer_speed = observer_speed * CMB_BETA
        self.dipole_longitude = dipole_longitude
        self.dipole_latitude = dipole_latitude
        self._rotation_matrices = rotation_matrices_for_dipole(
            dipole_longitude=self.dipole_longitude,
            dipole_latitude=self.dipole_latitude
        )

        n_expected_sources = int(10 ** log10_n_initial_samples)
        if n_expected_sources < 0:
            raise ValueError('n_initial_samples must be non-negative.')
        if lambda_clus < 0:
            raise ValueError('lambda_clus must be non-negative.')
        if lambda_clus > 0 and self.generate_correlated_points:
            raise ValueError(
                'lambda_clus cannot be used when generate_correlated_points=True.'
            )
        self.n_samples = int(n_expected_sources / (1.0 + lambda_clus))

        chunk_size = self.chunk_size
        store_final_samples = self.store_final_samples
        magnitude_error_dist = self.magnitude_error_dist

        common_extra_error = self.use_common_extra_error
        rng = rng_key._generator() if rng_key is not None else np.random.default_rng()

        if self.n_samples == 0:
            n_pix = hp.nside2npix(self.nside)
            self._density_map = np.zeros(n_pix, dtype=np.float32)
            if store_final_samples:
                self.final_w1_samples = np.empty(0, dtype=np.float32)
                self.final_w2_samples = np.empty(0, dtype=np.float32)
                self.final_w12_samples = np.empty(0, dtype=np.float32)
                self.final_pixel_indices = np.empty(0, dtype=np.int32)
            else:
                self.final_w1_samples = None
                self.final_w2_samples = None
                self.final_w12_samples = None
                self.final_pixel_indices = None
            output_map, output_mask = self._prepare_map_output(
                self._density_map,
                cache='simulation'
            )
            if self.downscale_nside is None:
                self._coarse_density_map = None
                self._coarse_mask = None
            return output_map, output_mask


        n_pix = hp.nside2npix(self.nside)
        density_accumulator = np.zeros(n_pix, dtype=np.float64)

        final_w1_list: list[NDArray[np.float32]] = []
        final_w2_list: list[NDArray[np.float32]] = []
        final_w12_list: list[NDArray[np.float32]] = []
        final_w1e_list: list[NDArray[np.float32]] = []
        final_w2e_list: list[NDArray[np.float32]] = []
        final_indices_list: list[NDArray[np.int32]] = []
        final_alpha_list: list[NDArray[np.float32]] = []
        final_measured_alpha_list: list[NDArray[np.float32]] = []

        self.final_w1_samples = None
        self.final_w2_samples = None
        self.final_w1e_samples = None
        self.final_w2e_samples = None
        self.final_w12_samples = None
        self.final_pixel_indices = None

        coverage_query_buffer = np.empty((chunk_size, 2), dtype=self.dtype)
        colour_buffer = np.empty(chunk_size, dtype=self.dtype)
        spectral_buffer = np.empty(chunk_size, dtype=np.float32)
        noise_buffer_w1 = np.empty(chunk_size, dtype=np.float64)
        noise_buffer_w2 = np.empty(chunk_size, dtype=np.float64)
        noise_buffer_w12 = np.empty(chunk_size, dtype=np.float64)

        for start in range(0, self.n_samples, chunk_size):
            current_chunk = min(chunk_size, self.n_samples - start)

            rest_w1_samples, rest_w2_samples = self.sample_magnitudes(
                current_chunk, dtype=self.dtype, rng=rng
            )
            rest_source_lon_deg, rest_source_lat_deg = self.sample_points(
                current_chunk,
                generate_correlated_points=self.generate_correlated_points,
                cluster_rate_param=cluster_rate_param,
                log10_cluster_scale_param=log10_cluster_scale_param,
                rng=rng
            )

            if lambda_clus > 0:
                per_parent_n_components = rng.poisson(
                    lambda_clus,
                    size=current_chunk,
                ).astype(np.int64, copy=False)
                total_n_components = int(per_parent_n_components.sum())
                if total_n_components > 0:
                    child_lon_deg, child_lat_deg = self.sample_clustered_points(
                        rest_source_lon_deg,
                        rest_source_lat_deg,
                        per_parent_n_components,
                        rng=rng,
                        dtype=self.dtype,
                    )
                    child_w1_samples, child_w2_samples = self.sample_magnitudes(
                        total_n_components,
                        dtype=self.dtype,
                        rng=rng,
                    )
                    rest_w1_samples = np.concatenate(
                        (rest_w1_samples, child_w1_samples)
                    ).astype(self.dtype, copy=False)
                    rest_w2_samples = np.concatenate(
                        (rest_w2_samples, child_w2_samples)
                    ).astype(self.dtype, copy=False)
                    rest_source_lon_deg = np.concatenate(
                        (rest_source_lon_deg, child_lon_deg)
                    ).astype(self.dtype, copy=False)
                    rest_source_lat_deg = np.concatenate(
                        (rest_source_lat_deg, child_lat_deg)
                    ).astype(self.dtype, copy=False)

            boosted_source_lon_deg, boosted_source_lat_deg, \
                rest_source_to_dipole_angle_deg = self.aberrate_points(
                    rest_source_lon_deg, rest_source_lat_deg, dtype=self.dtype
                )

            mask_slice, source_pixel_indices = self._source_isin_mask(
                boosted_source_lon_deg,
                boosted_source_lat_deg
            )

            if not mask_slice.any():
                continue

            rest_w1_samples = rest_w1_samples[mask_slice]
            rest_w2_samples = rest_w2_samples[mask_slice]
            boosted_source_lon_deg = boosted_source_lon_deg[mask_slice]
            boosted_source_lat_deg = boosted_source_lat_deg[mask_slice]
            rest_source_to_dipole_angle_deg = rest_source_to_dipole_angle_deg[mask_slice]
            source_pixel_indices = source_pixel_indices[mask_slice]

            if rest_w1_samples.size == 0:
                continue

            if rest_w1_samples.size <= colour_buffer.size:
                current_colour = colour_buffer[:rest_w1_samples.size]
            else:
                current_colour = np.empty(rest_w1_samples.size, dtype=self.dtype)
            np.subtract(rest_w1_samples, rest_w2_samples, out=current_colour)

            if current_colour.size <= spectral_buffer.size:
                spectral_indices = spectral_buffer[:current_colour.size]
            else:
                spectral_indices = np.empty(current_colour.size, dtype=np.float32)
            self.spectral_lookup.fit_alpha(
                w12_colour=current_colour,
                out=spectral_indices
            )
            # do not drop -ve sign
            np.negative(spectral_indices, out=spectral_indices)

            boosted_w1_samples = self.boost_magnitudes(
                rest_w1_samples, rest_source_to_dipole_angle_deg, spectral_indices,
                dtype=self.dtype
            )
            boosted_w2_samples = self.boost_magnitudes(
                rest_w2_samples, rest_source_to_dipole_angle_deg, spectral_indices,
                dtype=self.dtype
            )

            source_logw1_cov = self.log_w1cov_map[source_pixel_indices]
            source_logw2_cov = self.log_w2cov_map[source_pixel_indices]

            cov_delta = source_logw1_cov - source_logw2_cov
            cov_mean = 0.5 * (source_logw1_cov + source_logw2_cov)

            hashgrid_out = self.hashgrid.sample(
                grid_coords={
                    'w1': boosted_w1_samples,
                    'delta_cov': cov_delta,
                    # 'w1cov': source_logw1_cov,
                    'w2': boosted_w2_samples,
                    'mean_cov': cov_mean
                    # 'w2cov': source_logw2_cov
                },
                rng=rng,
                report_success=False
            )
            formal_w1_error = hashgrid_out[:, 0]; formal_w2_error = hashgrid_out[:, 1]
            formal_w1_error += rng.normal(loc=0, scale=0.001, size=len(formal_w1_error))
            formal_w2_error += rng.normal(loc=0, scale=0.001, size=len(formal_w2_error))

            if self.add_confusion_noise:
                assert log10_w1conf_scale is not None; assert log10_w2conf_scale is not None
                w1conf_scale = 10 ** log10_w1conf_scale
                w2conf_scale = 10 ** log10_w2conf_scale

                w1_confusion_proxy, w2_confusion_proxy = self.sample_confusion(
                    source_pixel_indices, rng
                )
            else:
                w1_confusion_proxy = None
                w2_confusion_proxy = None
                w1conf_scale = None
                w2conf_scale = None

            total_w1_error, total_w2_error = self.compute_total_error(
                formal_error=(formal_w1_error, formal_w2_error),
                extra_formal_error=(w1_extra_error, w2_extra_error),
                common_extra_formal_error=common_extra_error,
                confusion_error=(w1_confusion_proxy, w2_confusion_proxy),
                confusion_scale=(w1conf_scale, w2conf_scale)
            )

            if boosted_w1_samples.size <= noise_buffer_w1.size:
                current_noise_buffers = (
                    noise_buffer_w1[:boosted_w1_samples.size],
                    noise_buffer_w2[:boosted_w2_samples.size]
                )
            else:
                current_noise_buffers = (
                    np.empty(boosted_w1_samples.size, dtype=np.float64),
                    np.empty(boosted_w2_samples.size, dtype=np.float64)
                )

            boosted_w1_samples, boosted_w2_samples, w1e, w2e = self.add_error(
                w1=(boosted_w1_samples, total_w1_error),
                w2=(boosted_w2_samples, total_w2_error),
                error_dist=magnitude_error_dist,
                log10_shape_param=log10_magnitude_error_shape_param,
                rng=rng,
                noise_buffers=current_noise_buffers
            )

            # after adding error
            boosted_w12_samples = boosted_w1_samples - boosted_w2_samples
            cur_buffer_w12 = noise_buffer_w12[:boosted_w12_samples.size]
            # w12_formal_error = np.hypot(formal_w1_error, formal_w2_error)
            #
            # # if w12_extra_error is None, no colour error is added
            # boosted_w12_samples = self.maybe_add_colour_error(
            #     w12_magnitudes=boosted_w12_samples,
            #     w12_formal_error=w12_formal_error,
            #     w12_extra_error=w12_extra_error,
            #     noise_buffer=cur_buffer_w12,
            #     rng=rng
            # )

            cut = self.magnitude_cut_boolean(
                w1_magnitudes=boosted_w1_samples,
                w12_magnitudes=boosted_w12_samples,
                w1_max=w1_max,
                w1_min=w1_min,
                w12_min=w12_min
            )

            if not cut.any():
                continue

            cut_boosted_w1_samples = boosted_w1_samples[cut]
            cut_boosted_w2_samples = boosted_w2_samples[cut]
            cut_w1e_samples = w1e[cut]
            cut_w2e_samples = w2e[cut]
            cut_boosted_w12_samples = boosted_w12_samples[cut]
            cut_source_pixel_indices = source_pixel_indices[cut].astype(np.int32, copy=False)

            # these are the 'true' spectral indices, i.e. the ones we actually
            # measure are after error has been added to w1 and w2
            true_cut_spectral_indices = spectral_indices[cut]
            measured_cut_spectral_indices = -self.spectral_lookup.fit_alpha(
                cut_boosted_w12_samples
            )

            chunk_density = np.bincount(
                cut_source_pixel_indices,
                minlength=n_pix
            )
            density_accumulator += chunk_density

            if store_final_samples:
                final_w1_list.append(cut_boosted_w1_samples.astype(np.float32)) 
                final_w2_list.append(cut_boosted_w2_samples.astype(np.float32))
                final_w12_list.append(cut_boosted_w12_samples.astype(np.float32))
                final_w1e_list.append(cut_w1e_samples.astype(np.float32)) 
                final_w2e_list.append(cut_w2e_samples.astype(np.float32))
                final_indices_list.append(cut_source_pixel_indices)
                final_alpha_list.append(true_cut_spectral_indices)
                final_measured_alpha_list.append(measured_cut_spectral_indices)

        self._density_map = density_accumulator.astype(np.float32)

        if store_final_samples:
            if final_w1_list:
                self.final_w1_samples = np.concatenate(final_w1_list)
                self.final_w2_samples = np.concatenate(final_w2_list)
                self.final_w1e_samples = np.concatenate(final_w1e_list)
                self.final_w2e_samples = np.concatenate(final_w2e_list)
                self.final_w12_samples = np.concatenate(final_w12_list)
                self.final_pixel_indices = np.concatenate(final_indices_list)
                self.final_alpha_samples = np.concatenate(final_alpha_list)
                self.final_measured_alpha_samples = np.concatenate(
                    final_measured_alpha_list
                )
            else:
                self.final_w1_samples = np.empty(0, dtype=np.float32)
                self.final_w2_samples = np.empty(0, dtype=np.float32)
                self.final_w1e_samples = np.empty(0, dtype=np.float32)
                self.final_w2e_samples = np.empty(0, dtype=np.float32)
                self.final_w12_samples = np.empty(0, dtype=np.float32)
                self.final_pixel_indices = np.empty(0, dtype=np.int32)
                self.final_alpha_samples = np.empty(0, dtype=np.float32)
                self.final_measured_alpha_samples = np.empty(0, np.float32)

        output_map, output_mask = self._prepare_map_output(
            self._density_map,
            cache='simulation'
        )
        if self.downscale_nside is None:
            self._coarse_density_map = None
            self._coarse_mask = None
        return output_map, output_mask
    
    def make_real_sample(
            self, 
            mask_catalogue: bool = False
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        print(f'Reading in CatWISE2020 from {self.s21_catalogue_path}...')

        self.real_catalogue = Table.read(self.s21_catalogue_path)

        print(f'Loaded CatWISE2020.')

        print('Making flux cuts...')
        flux_cuts = (
                (self.real_catalogue['w1'] < 16.4)
              & (self.real_catalogue['w1'] > 9)
        )
        self.real_catalogue = self.real_catalogue[flux_cuts]
        
        self._real_density_map = self.make_density_map(
            longitudes=self.real_catalogue['l'].data,
            latitudes=self.real_catalogue['b'].data
        )
        output_map, output_mask = self._prepare_map_output(
            self._real_density_map,
            cache='real'
        )
        if self.downscale_nside is None:
            self._coarse_real_density_map = None
            self._coarse_real_mask = None

        if mask_catalogue:
            assert hasattr(self, 'masked_pixel_indices_set'), 'Load mask first.'
            
            print('Generating masked catalogue...')
            all_pixel_indices = hp.ang2pix(
                self.nside,
                self.real_catalogue['l'],
                self.real_catalogue['b'],
                lonlat=True,
                nest=True
            )
            self.real_catalogue_mask = [
                idx not in self.masked_pixel_indices_set
                for idx in all_pixel_indices
            ]
            self.real_catalogue = self.real_catalogue[self.real_catalogue_mask]
            print('Done.')

        return output_map, output_mask

    def make_density_map(self,
        longitudes: NDArray,
        latitudes: NDArray
    ) -> NDArray[np.float32]:
        source_indices = hp.ang2pix(
            self.nside, longitudes, latitudes, lonlat=True, nest=True
        ).astype(np.int32)
        return np.bincount(
            source_indices,
            minlength=hp.nside2npix(self.nside)
        ).astype(np.float32)

    def _native_mask(self) -> NDArray[np.bool_]:
        if not hasattr(self, 'mask_map'):
            raise AttributeError('Mask not initialised; call determine_masked_pixels() first.')
        return (self.mask_map == 0).astype(np.bool_)

    @property
    def native_mask(self) -> NDArray[np.bool_]:
        """Return the native-resolution boolean mask (nside=64)."""
        return self._native_mask()

    def _prepare_map_output(
            self,
            map_values: NDArray,
            *,
            cache: Optional[str] = None
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        native_mask = self._native_mask()
        fill_value = getattr(self, 'fill_value', np.nan)

        map_with_mask = np.asarray(map_values, dtype=np.float32).copy()
        map_with_mask[~native_mask] = fill_value

        if self.downscale_nside is None:
            if cache == 'simulation':
                self._coarse_density_map = None
                self._coarse_mask = None
            elif cache == 'real':
                self._coarse_real_density_map = None
                self._coarse_real_mask = None
            return map_with_mask, native_mask

        coarse_map, coarse_mask = downgrade_ignore_nan(
            map_with_mask,
            native_mask,
            self.downscale_nside
        )

        coarse_map = coarse_map.astype(np.float32, copy=False)
        coarse_mask = coarse_mask.astype(np.bool_, copy=False)

        if not np.isnan(fill_value):
            coarse_map = coarse_map.copy()
            coarse_map[~coarse_mask] = fill_value

        if cache == 'simulation':
            self._coarse_density_map = coarse_map
            self._coarse_mask = coarse_mask
        elif cache == 'real':
            self._coarse_real_density_map = coarse_map
            self._coarse_real_mask = coarse_mask

        return coarse_map, coarse_mask
    
    @property
    def density_map(self) -> NDArray[np.float32]:
        if self.downscale_nside is not None and self._coarse_density_map is not None:
            return self._coarse_density_map

        out = np.asarray(self._density_map, dtype=np.float32).copy()
        fill_value = getattr(self, 'fill_value', np.nan)
        out[self.mask_map == 1] = fill_value
        return out

    @property
    def binary_mask(self) -> NDArray[np.bool_]:
        if self.downscale_nside is not None and self._coarse_mask is not None:
            return self._coarse_mask
        return self._native_mask()

    @property
    def real_density_map(self) -> NDArray[np.float32]:
        if self.downscale_nside is not None and self._coarse_real_density_map is not None:
            return self._coarse_real_density_map

        out = np.asarray(self._real_density_map, dtype=np.float32).copy()
        fill_value = getattr(self, 'fill_value', np.nan)
        out[self.mask_map == 1] = fill_value
        return out

    def determine_masked_pixels(self,
            fill_value = None,
            mask_north_ecliptic: bool = True
    ) -> None:
        self.mask = Mask(nside=self.nside)
        self.mask_map = np.zeros(self.mask.npix)

        assert self.nside == 64, 'CatWISE mask requires nside=64.'
        masked_pixel_indices = set(
            self.mask.catwise_mask(self.cfg.base_mask_version)
        )
        
        if mask_north_ecliptic:
            north_pole_pixels = self.mask.north_ecliptic_mask()
            masked_pixel_indices.update(north_pole_pixels)
        
        self.masked_pixel_indices_set = masked_pixel_indices
        self.masked_pixel_indices_list = list(masked_pixel_indices)
        self.mask_map[self.masked_pixel_indices_list] = 1

        if fill_value == None:
            self.fill_value = np.nan
        else:
            self.fill_value = fill_value

    def _source_isin_mask(
            self,
            longitudes: NDArray,
            latitudes: NDArray
    ) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
        source_pixel_indices = hp.ang2pix(
            self.nside,
            longitudes,
            latitudes,
            lonlat=True,
            nest=True
        ).astype(np.uint16) # unsigned 16-bit can represent all pixels in nside=64 map
        mask_slice = self.mask_map[source_pixel_indices] == 0
        return mask_slice, source_pixel_indices

    def compute_total_error(
        self,
        formal_error: tuple[NDArray, NDArray],
        extra_formal_error: tuple[float | None, float | None],
        common_extra_formal_error: bool | None,
        confusion_error: tuple[NDArray | None, NDArray | None],
        confusion_scale: tuple[float | None, float | None],
    ) -> tuple[NDArray, NDArray]:
        '''
        Should follow relationship sigm_tot^2 = sigm_totformal^2 + sigm_totconf^2
        where sigm_totformal^2 = sigm_formal^2 + eta_extra * sigm_formal^2
        and sigm_totconf^2 = (k sigm_conf)^2
        '''
        w1_formal, w2_formal = formal_error
        w1_extra_err, w2_extra_err = extra_formal_error

        if common_extra_formal_error and w1_extra_err is not None:
            w2_extra_err = w1_extra_err

        # sigma^2 = sigma_b^2 + extra * sigma_b^2 => sigma = sigma_b sqrt( 1 + extra )
        if w1_extra_err is not None:
            w1_formal = np.multiply(
                w1_formal,
                np.sqrt(np.array(1.0 + w1_extra_err))
            )
        if w2_extra_err is not None:
            w2_formal = np.multiply(
                w2_formal,
                np.sqrt(np.array(1.0 + w2_extra_err))
            )

        if self.add_confusion_noise:
            w1_confusion, w2_confusion = confusion_error
            w1conf_scale, w2conf_scale = confusion_scale

            assert (w1_confusion is not None) and (w2_confusion is not None)
            assert (w1conf_scale is not None) and (w2conf_scale is not None)

            w1_confusion *= w1conf_scale
            w2_confusion *= w2conf_scale

            total_w1_error = np.hypot(w1_formal, w1_confusion)
            total_w2_error = np.hypot(w2_formal, w2_confusion)
        else:
            total_w1_error = w1_formal
            total_w2_error = w2_formal

        return total_w1_error, total_w2_error

    def add_error(self,
            w1: tuple[NDArray, NDArray],
            w2: tuple[NDArray, NDArray],
            error_dist: Literal['gaussian', 'students-t'] = 'gaussian',
            log10_shape_param: float = 0.,
            rng: Optional[np.random.Generator] = None,
            noise_buffers: Optional[tuple[NDArray[np.float64], NDArray[np.float64]]] = None
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Adds random photometric errors to W1 and W2 magnitudes.

        :param w1: Tuple of (magnitudes, errors) for the W1 band.
        :param w2: Tuple of (magnitudes, errors) for the W2 band.
        :param w1_extra_error: Optional extra error (added in quadrature) for W1.
        :param w2_extra_error: Optional extra error (added in quadrature) for W2.
        :param error_dist: Distribution to sample errors from
            ('gaussian' or 'students-t').
        :param log10_shape_param: Log10 of shape parameter (degrees of freedom)
            for Student's t-distribution.
        :returns: Tuple of arrays: (noisy_w1_magnitudes, noisy_w2_magnitudes)
        """
        w1_magnitudes, w1_error = w1
        w2_magnitudes, w2_error = w2

        w1_dtype = w1_magnitudes.dtype
        w2_dtype = w2_magnitudes.dtype

        if rng is None:
            rng = np.random.default_rng()

        if noise_buffers is not None:
            noise_w1_buf, noise_w2_buf = noise_buffers
            noise_w1 = noise_w1_buf[:w1_error.size]
            noise_w2 = noise_w2_buf[:w2_error.size]
        else:
            noise_w1 = None
            noise_w2 = None

        if error_dist == 'gaussian':
            if noise_w1 is not None and noise_w2 is not None:
                rng.standard_normal(out=noise_w1)
                rng.standard_normal(out=noise_w2)
            else:
                noise_w1 = rng.standard_normal(size=w1_error.shape)
                noise_w2 = rng.standard_normal(size=w2_error.shape)
        else:
            shape_param = float(10 ** log10_shape_param)
            if shape_param <= 0:
                raise ValueError("Student's t shape parameter must be positive.")
            if noise_w1 is not None and noise_w2 is not None:
                noise_w1[:] = rng.standard_t(df=shape_param, size=w1_error.shape)
                noise_w2[:] = rng.standard_t(df=shape_param, size=w2_error.shape)
            else:
                noise_w1 = rng.standard_t(df=shape_param, size=w1_error.shape)
                noise_w2 = rng.standard_t(df=shape_param, size=w2_error.shape)

        calc_dtype_w1 = np.float64 if w1_dtype == np.float64 else np.float32
        calc_dtype_w2 = np.float64 if w2_dtype == np.float64 else np.float32

        noisy_w1 = (
            np.asarray(w1_magnitudes, dtype=calc_dtype_w1)
          + noise_w1.astype(calc_dtype_w1, copy=False)
          * np.asarray(w1_error, dtype=calc_dtype_w1)
        ).astype(w1_dtype, copy=False)

        noisy_w2 = (
            np.asarray(w2_magnitudes, dtype=calc_dtype_w2)
          + noise_w2.astype(calc_dtype_w2, copy=False)
          * np.asarray(w2_error, dtype=calc_dtype_w2)
        ).astype(w2_dtype, copy=False)

        return noisy_w1, noisy_w2, w1_error, w2_error

    def maybe_add_colour_error(
        self,
        w12_magnitudes: NDArray,
        w12_formal_error: NDArray,
        w12_extra_error: Optional[float],
        rng: Optional[np.random.Generator] = None,
        noise_buffer: Optional[NDArray[np.float64]] = None
    ) -> NDArray:
        '''
        Add Gaussian magnitude error to the W12 colour (W1 - W2).

        :param w12_magnitudes: Array of W12 magnitudes to perturb.
        :param w12_formal_error: Array of formal errors propagated after
            subtracting W2 from W1 (assuming no correlation for now, which is
            handled here).
        :param w12_extra_error: The fractional extra error to apply to the W12
            magnitudes, i.e. eta. If ``None`` no extra noise is added.
        :param rng: Optional numpy random generator to use. A default RNG is
            created when not provided.
        :param noise_buffer: Optional buffer reused to store the generated
            noise. When supplied it must match the shape of ``w12_magnitudes``
            and is overwritten in place.
        :returns: ``w12_magnitudes`` with an independent Gaussian error added
            to each element.
        '''
        if w12_extra_error is None:
            return w12_magnitudes

        if noise_buffer is not None:
            noise_w12 = noise_buffer
        else:
            noise_w12 = None

        if rng is None:
            rng = np.random.default_rng()

        if noise_w12 is None:
            noise_w12 = rng.normal(
                scale=w12_extra_error * w12_formal_error,
                size=w12_magnitudes.shape
            )
        else:
            rng.standard_normal( # pyright: ignore[reportCallIssue]
                size=w12_magnitudes.shape,
                out=noise_w12
            )
            noise_w12 *= (w12_extra_error * w12_formal_error)
        
        w12_with_added_error = w12_magnitudes + noise_w12

        return w12_with_added_error
    
    def magnitude_cut_boolean(self,
            w1_magnitudes: NDArray,
            w12_magnitudes: NDArray,
            w1_max: float,
            w1_min: float,
            w12_min: float
        ) -> NDArray:
        condition = np.logical_and(
            np.logical_and(
                w1_magnitudes < w1_max,
                w1_magnitudes > w1_min
            ),
            w12_magnitudes > w12_min
        )
        return condition

    def precompute_data(self, mask_north_ecliptic: bool = True) -> None: 
        # load catalogue and mask
        self.northecl_is_masked = mask_north_ecliptic
        if not self.catalogue_is_loaded:
            self.load_catalogue()

        self.determine_masked_pixels(mask_north_ecliptic=mask_north_ecliptic)
        self.make_masked_catalogue()

        self.create_confusion_skylookup()
        self.create_w1_w2_distribution()
        self.create_coverage_maps()
        self.create_mag_cov_hashgrid(
            grid_step=[0.1, 0.01, 0.1, 0.02], 
            project_coverage=True
        )
    
    def make_masked_catalogue(self):
        assert self.catalogue_is_loaded, 'Load catalogue first.'
        assert hasattr(self, 'masked_pixel_indices_set'), 'Load mask first.'
        
        print('Generating masked catalogue...')
        all_pixel_indices = hp.ang2pix(
            self.nside,
            self.catalogue['l'],
            self.catalogue['b'],
            lonlat=True,
            nest=True
        )
        self.catalogue_mask = [
            idx not in self.masked_pixel_indices_set
            for idx in all_pixel_indices
        ]
        self.masked_catalogue = self.catalogue[self.catalogue_mask]
        print('Done.')

    def create_confusion_skylookup(self):
        assert self.catalogue_is_loaded
        assert hasattr(self, 'masked_catalogue')

        print('Joining supplementary confusion data...')
        masked_len = len(self.masked_catalogue)
        with data_path() as data_dir:
            conf_table = Table.read(data_dir / 'catwise_w120p5_confdata.fits')

        # strict intersection operation at source_id
        out = join(self.masked_catalogue, conf_table, keys="source_id")
        consolidated_len = len(out)
        del conf_table

        assert consolidated_len == masked_len

        # make vectorized lookup arrays (sorted by pixel index)
        pixel_indices = hp.ang2pix(self.nside, out['l'], out['b'], lonlat=True, nest=True)
        n_pix = hp.nside2npix(self.nside)
        order = np.argsort(pixel_indices)
        pix_sorted = np.asarray(pixel_indices, dtype=np.int64)[order]
        w1conf_sorted = np.asarray(out['w1conf'] / out['w1flux'], dtype=np.float32)[order]
        w2conf_sorted = np.asarray(out['w2conf'] / out['w2flux'], dtype=np.float32)[order]

        counts = np.bincount(pix_sorted, minlength=n_pix).astype(np.int64)
        starts = np.cumsum(counts) - counts

        self.confusion_pixel_counts = counts
        self.confusion_pixel_starts = starts
        self.confusion_w1_values = w1conf_sorted
        self.confusion_w2_values = w2conf_sorted

        with data_path(self.cut_path, 'confusion') as conf_dir:
            conf_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                conf_dir / 'confusion_skylookup.npz',
                nside=np.array(self.nside, dtype=np.int64),
                pix_ordered=pix_sorted,
                counts=counts,
                starts=starts,
                w1conf=w1conf_sorted,
                w2conf=w2conf_sorted,
            )
            mean_w1conf = np.full(n_pix, np.nan, dtype=np.float32)
            mean_w2conf = np.full(n_pix, np.nan, dtype=np.float32)
            nonzero_pix = np.where(counts > 0)[0]
            for pix in nonzero_pix:
                start = starts[pix]
                length = counts[pix]
                mean_w1conf[pix] = np.mean(w1conf_sorted[start:start + length])
                mean_w2conf[pix] = np.mean(w2conf_sorted[start:start + length])

            plt.figure()
            hp.projview(mean_w1conf, nest=True)
            plt.savefig(conf_dir / 'mean_w1conf_map.png', dpi=300)
            plt.close()

            plt.figure()
            hp.projview(mean_w2conf, nest=True)
            plt.savefig(conf_dir / 'mean_w2conf_map.png', dpi=300)
            plt.close()

            print(f'Saved confusion lookup at {conf_dir}.')

    def load_confusion_skylookup(self) -> None:
        with data_path(self.cut_path, 'confusion') as conf_dir:
            lookup_path = conf_dir / 'confusion_skylookup.npz'
            with np.load(lookup_path) as data:
                lookup_nside = int(data['nside'])
                if lookup_nside != self.nside:
                    raise ValueError(
                        'Confusion lookup nside does not match Catwise nside: '
                        f'{lookup_nside} != {self.nside}'
                    )
                self.confusion_pixel_counts = data['counts'].astype(np.int64)
                self.confusion_pixel_starts = data['starts'].astype(np.int64)
                self.confusion_w1_values = data['w1conf'].astype(np.float32)
                self.confusion_w2_values = data['w2conf'].astype(np.float32)

    def sample_confusion(
            self,
            pixel_indices: NDArray[np.int_],
            rng: Optional[np.random.Generator] = None
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        assert hasattr(self, 'confusion_pixel_counts'), 'Load confusion lookup first.'
        if rng is None:
            rng = np.random.default_rng()

        pix = np.asarray(pixel_indices, dtype=np.int64)
        counts = self.confusion_pixel_counts[pix]
        starts = self.confusion_pixel_starts[pix]

        out_w1 = np.empty(pix.shape[0], dtype=np.float32)
        out_w2 = np.empty(pix.shape[0], dtype=np.float32)

        valid = counts > 0
        if np.any(valid):
            rand_offsets = rng.integers(0, counts[valid], dtype=np.int64)
            pick = starts[valid] + rand_offsets
            out_w1[valid] = self.confusion_w1_values[pick]
            out_w2[valid] = self.confusion_w2_values[pick]

        if np.any(~valid):
            # fallback: global sampling for empty pixels
            pick = rng.integers(
                0, self.confusion_w1_values.size, size=np.count_nonzero(~valid)
            )
            out_w1[~valid] = self.confusion_w1_values[pick]
            out_w2[~valid] = self.confusion_w2_values[pick]

        return out_w1, out_w2

    def create_coverage_maps(self, use_mask: bool = False):
        if use_mask:
            cat = self.masked_catalogue
            file_descriptor = 'coverage_map'
        else:
            cat = self.catalogue
            file_descriptor = 'coverage_map_unmasked'

        pixel_indices = hp.ang2pix(
            self.nside,
            cat['l'],
            cat['b'],
            lonlat=True,
            nest=True
        )
        print('Building coverage maps...')
        w1_covmap = ParameterMap(
            pixel_indices=pixel_indices,
            parameter=cat['w1cov'],
            nside=self.nside
        ).get_map()
        w2_covmap = ParameterMap(
            pixel_indices=pixel_indices,
            parameter=cat['w2cov'],
            nside=self.nside
        ).get_map()
        
        with data_path(self.cut_path, 'coverage_map') as coverage_dir:
            coverage_dir.mkdir(parents=True, exist_ok=True)

            w1_npy = coverage_dir / f'w1_{file_descriptor}.npy'
            w2_npy = coverage_dir / f'w2_{file_descriptor}.npy'
            np.save(w1_npy, w1_covmap.astype(np.float32, copy=False))
            np.save(w2_npy, w2_covmap.astype(np.float32, copy=False))
            print(f'Saved coverage maps at {coverage_dir}.')

            plt.figure()
            hp.projview(w1_covmap, nest=True, norm='log')
            plt.savefig(coverage_dir / f'w1_{file_descriptor}.png', dpi=300)

            plt.figure()
            hp.projview(w2_covmap, nest=True, norm='log')
            plt.savefig(coverage_dir / f'w2_{file_descriptor}.png', dpi=300)

            print(f'Saved coverage map figures at {coverage_dir}.')

    def create_mag_cov_hashgrid(
            self, 
            grid_step: list[float], 
            project_coverage: bool = False
    ) -> None:
        cat = self.masked_catalogue
        logw1cov = np.log10(cat['w1cov'])
        logw2cov = np.log10(cat['w2cov'])

        if project_coverage:
            delta_cov = logw1cov - logw2cov
            mean_cov = 0.5 * ( logw1cov + logw2cov )
            grid_coords = {
                'w1': cat['w1'],
                'delta_cov': delta_cov,
                'w2': cat['w2'],
                'mean_cov': mean_cov
            }
        else:
            grid_coords = {
                'w1': cat['w1'],
                'w1cov': logw1cov,
                'w2': cat['w2'],
                'w2cov': logw2cov
            }

        grid_values = {
            'w1e': cat['w1e'],
            'w2e': cat['w2e']
        }
        hashgrid = HashGrid(
            grid_coords=grid_coords,
            grid_values=grid_values,
            grid_step=grid_step
        )

        with data_path(self.cut_path, 'mag_coverage') as file_dir:
            if self.northecl_is_masked:
                hashgrid.save(file_dir / f'w1w2_magcov_hashgrid.npz')
            else:
                hashgrid.save(file_dir / f'w1w2_magcov_hashgrid_no_eclmask.npz')

    def create_magnitude_coverage_cell_dist(self):
        cat = self.masked_catalogue
        x = cat['w1']
        y = np.log10(cat['w1cov'])
        z = np.concatenate([cat['w1e'][:, None], cat['w2e'][:, None]], axis=1)

        bin_bounds, bin_values = kdtree_binned_distribution_2d(
            x, y, z, target_count=10000, max_factor=2, min_count=5000
        )

        plot_adaptive_bins(
            bin_bounds,
            x=x,
            y=y,
            show_counts=False,
            bin_values=bin_values,
        )

        with data_path(self.cut_path, 'mag_coverage') as file_dir:
            save_sigma_bins(
                filename=file_dir / f'error_bins_w1_and_w2.npz',
                bin_bounds=bin_bounds,
                bin_values=bin_values
            )
            plt.savefig(file_dir / f'adaptive_bins_w1_and_w2.png')
            plt.close()

    def load_magnitude_coverage_cell_dist(self, file_path: str | Path):
        bin_bounds, bin_values = load_sigma_bins(file_path)
        return bin_bounds, bin_values

    def create_magnitude_coverage_function(self, statistic: str = 'median'):
        N_1D_BINS = 200

        # define magnitude-coverage grid bins, same for w1 and w2 for simplicity
        magnitude_bins = np.linspace(
            np.min(self.masked_catalogue['w1']),
            self.cat_w1_max,
            N_1D_BINS
        )
        coverage_bins = np.linspace(1.5, 4., N_1D_BINS)
        magnitude_centres = 0.5 * (magnitude_bins[:-1] + magnitude_bins[1:])
        coverage_centres = 0.5 * (coverage_bins[:-1] + coverage_bins[1:])

        for band in ['w1', 'w2']:
            print(f'Building {band} mag-coverage-error relation...')
                
            # compute median raw photometric across all sources in each cell
            median_error_grid, *_ = binned_statistic_2d(
                self.masked_catalogue[f'{band}'],
                np.log10(self.masked_catalogue[f'{band}cov']),
                self.masked_catalogue[f'{band}e'],
                statistic=statistic,
                bins=[magnitude_bins, coverage_bins] # type: ignore
            )
            n_sources, *_ = binned_statistic_2d(
                self.masked_catalogue[f'{band}'],
                np.log10(self.masked_catalogue[f'{band}cov']),
                self.masked_catalogue[f'{band}e'],
                statistic='count',
                bins=[magnitude_bins, coverage_bins] # type: ignore
            )

            # to remove noisy cells
            median_error_grid[n_sources < 10] = np.nan

            # do nearest neighbour interpolation to fill nan cells
            magnitude_grid, coverage_grid = np.meshgrid(
                magnitude_centres, coverage_centres, indexing='ij'
            )
            mask = ~np.isnan(median_error_grid)
            valid_indices = np.where(mask)
            X_train = np.column_stack(
                [magnitude_grid[valid_indices], coverage_grid[valid_indices]]
            )
            y_train = median_error_grid[valid_indices]

            nbrs = NearestNeighbors(
                n_neighbors=4,
                algorithm='kd_tree',
                leaf_size=30
            )
            nbrs.fit(X_train)

            def knn_interpolate(X_pred, nbrs: NearestNeighbors):
                '''
                KNN interpolation with inverse distance weighting.
                '''
                distances, indices = nbrs.kneighbors(X_pred)
                
                # Inverse distance weighting
                weights = 1 / (distances + 1e-8) # Eps. to avoid division by zero
                weights = weights / weights.sum(axis=1)[:, np.newaxis]
                
                # Weighted prediction
                return np.sum(weights * y_train[indices], axis=1)
            
            filled_median_error_grid = median_error_grid.copy()
            nan_indices = np.where(~mask)
            X_predict = np.column_stack(
                [magnitude_grid[nan_indices], coverage_grid[nan_indices]]
            )
            filled_errors = knn_interpolate(X_predict, nbrs)
            filled_median_error_grid[~mask] = filled_errors

            rgi = RegularGridInterpolator(
                (magnitude_centres, coverage_centres), 
                filled_median_error_grid,
                method='linear',
                bounds_error=False,
                fill_value=None # extrapolation for 16.95 -> 17 mag # type: ignore
            )
            with data_path(self.cut_path, 'mag_coverage') as file_path:
                self.save_interpolator(
                    band=band,
                    interpolator=rgi,
                    mag_bins=magnitude_bins,
                    cov_bins=coverage_bins,
                    filled_grid=filled_median_error_grid,
                    statistic=statistic,
                    file_path=file_path
                )
                plt.figure()
                plt.pcolormesh(
                    magnitude_bins,
                    coverage_bins,
                    filled_median_error_grid.T,
                    shading='auto'
                )
                plt.colorbar()
                plt.savefig(f'{file_path}/{band}_matrix_plot.png', dpi=300)
                plt.close()

    def save_interpolator(self,
            band: str,
            interpolator: RegularGridInterpolator,
            mag_bins: NDArray[np.float64],
            cov_bins: NDArray[np.float64],
            filled_grid: NDArray[np.float64],
            statistic: str,
            file_path: str | Path
    ) -> bool:
        '''
        Save RegularGridInterpolator and metadata for use in batch simulations.
        '''
        save_data = {
            'interpolator': interpolator,
            'band': band,
            'mag_bins': mag_bins,
            'cov_bins': cov_bins,
            'filled_grid': filled_grid,
            'statistic': statistic,
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'mag_range': (mag_bins.min(), mag_bins.max()),
                'cov_range': (cov_bins.min(), cov_bins.max()),
                'grid_shape': filled_grid.shape,
                'interpolation_method': 'linear'
            }
        }
        try:
            full_path = f'{file_path}/{band}_median_error_interpolator.pkl'

            if not os.path.exists(file_path):
                os.makedirs(file_path)

            with open(full_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Interpolator saved to: {full_path}")
            return True
        
        except Exception as e:
            print(f"✗ Error saving interpolator: {e}")
            return False
        
    def load_interpolator(self, full_path: os.PathLike[str] | str) -> RegularGridInterpolator:
        file_path = os.fspath(full_path)
        try:
            with open(file_path, 'rb') as f:
                save_data = pickle.load(f)

            print(f"Interpolator loaded from: {file_path}")
            print(f"Created: {save_data['metadata']['creation_date']}")
            print(f"Mag range: {save_data['metadata']['mag_range']}")
            print(f"Cov range: {save_data['metadata']['cov_range']}")
            print(f"Grid shape: {save_data['metadata']['grid_shape']}")
            
            return save_data['interpolator']
        
        except Exception as e:
            print(f"Error loading interpolator: {e}")
            raise Exception(e)

    def initialise_data(self):
        if self.use_noecl_mask:
            fname_append = '_no_eclmask'
        else:
            fname_append = ''

        self.colour_mag_sampler = MultinomialSample2DHistogram()
        with data_path(self.cut_path, 'colour_mag') as colour_mag_dir:
            self.colour_mag_sampler.load_data(
                colour_mag_dir,
                fname_append=fname_append if self.use_noecl_mask else None
            )

        with data_path(
            self.cut_path, 
            'mag_coverage', 
            f'w1w2_magcov_hashgrid{fname_append}.npz'
        ) as filepath:
            self.hashgrid = HashGrid.load(filepath)

        # loads things back into numpy
        # for now we just hardcode the use of the unmasked covmaps;
        # their masked counterparts still remain in the data dir
        with data_path(
            self.cut_path,
            'coverage_map',
            'w1_coverage_map_unmasked.npy'
        ) as w1_cov_path:
            self.w1cov_map = np.load(
                w1_cov_path,
                allow_pickle=False
            ).astype(np.float32, copy=False)

        with data_path(
            self.cut_path,
            'coverage_map',
            'w2_coverage_map_unmasked.npy'
        ) as w2_cov_path:
            self.w2cov_map = np.load(
                w2_cov_path,
                allow_pickle=False
            ).astype(np.float32, copy=False)

        self.log_w1cov_map = np.log10(self.w1cov_map).astype(self.dtype)
        self.log_w2cov_map = np.log10(self.w2cov_map).astype(self.dtype)

        self.load_confusion_skylookup()

        # initialise AlphaLookup so table is not read in at each simulation
        self.spectral_lookup = AlphaLookup()
        
        # mask now instead of during each loop
        self.determine_masked_pixels(mask_north_ecliptic=not self.use_noecl_mask)

        self.lookups_are_initialised = True
        
    def create_error_map(self) -> None:
        assert self.catalogue_is_loaded

        l, b = self.masked_catalogue['l'], self.masked_catalogue['b']
        source_pixel_indices = hp.ang2pix(self.nside, l, b, lonlat=True, nest=True)
        n_pixels = hp.nside2npix(self.nside)
        
        w1_error_map = np.empty(n_pixels)
        w2_error_map = np.empty(n_pixels)
        w12_error_map = np.empty(n_pixels)
        w1_error_dict = defaultdict(list)
        w2_error_dict = defaultdict(list)
        w12_error_dict = defaultdict(list)

        for pix_ind in tqdm(range(n_pixels)):
            active_pixel = source_pixel_indices == pix_ind

            w1_fractional_error = (
                self.masked_catalogue['w1e'][active_pixel]
                / self.masked_catalogue['w1'][active_pixel]
            )
            w2_fractional_error = (
                self.masked_catalogue['w2e'][active_pixel]
                / self.masked_catalogue['w2'][active_pixel]
            )
            w12_fractional_error = (
                self.masked_catalogue['w12e'][active_pixel]
                / self.masked_catalogue['w12'][active_pixel]
            )

            w1_error_dict[pix_ind] = w1_fractional_error
            w2_error_dict[pix_ind] = w2_fractional_error
            w12_error_dict[pix_ind] = w12_fractional_error
            w1_error_map[pix_ind] = np.median( w1_fractional_error )
            w2_error_map[pix_ind] = np.median( w2_fractional_error )
            w12_error_map[pix_ind] = np.median( w12_fractional_error )
        
        file_path = f'dipolesbi/catwise/{self.cut_path}/data/error_map/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        np.save(
            f'{file_path}w1_error_map.npy',
            w1_error_map.astype(np.float32, copy=False)
        )
        np.save(
            f'{file_path}w2_error_map.npy',
            w2_error_map.astype(np.float32, copy=False)
        )
        np.save(
            f'{file_path}w12_error_map.npy',
            w12_error_map.astype(np.float32, copy=False)
        )
        with open(f'{file_path}w1_error_dict.pt', 'wb') as handle:
            pickle.dump(w1_error_dict, handle)
        
        with open(f'{file_path}w2_error_dict.pt', 'wb') as handle:
            pickle.dump(w2_error_dict, handle)

        with open(f'{file_path}w12_error_dict.pt', 'wb') as handle:
            pickle.dump(w12_error_dict, handle)

    def create_w1_w2_distribution(self,
            bins: int = 200,
            **hist_kwargs
        ) -> None:
        assert self.catalogue_is_loaded
        
        w1_mags = self.masked_catalogue['w1']
        w2_mags = self.masked_catalogue['w2']

        sampler = MultinomialSample2DHistogram()
        sampler.build(
            w1_mags,
            w2_mags,
            **{
                'bins': bins,
                **hist_kwargs
            }
        )

        if not self.northecl_is_masked:
            fname_append = '_no_eclmask'
        else:
            fname_append = None

        with data_path(self.cut_path, 'colour_mag') as colour_mag_dir:
            sampler.save_data(colour_mag_dir, fname_append=fname_append)
            print(f'Saved W1-W2 distribution to {colour_mag_dir}.')

            w1_samples, w2_samples = sampler.sample(n_samples=30_000_000)
            plt.hist2d(w1_samples, w2_samples, bins=400, norm='log')

            if fname_append:
                plt.savefig(colour_mag_dir / f'w1_w2_dist{fname_append}.png', dpi=300)
            else:
                plt.savefig(colour_mag_dir / 'w1_w2_dist.png', dpi=300)

            print(f'Generated W1-W2 distribution plot...')
    
    def resample_catwise_magnitudes(
            self,
            n_samples: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not self.catalogue_is_loaded:
            self.load_catalogue()
        
        if not hasattr(self, 'masked_catalogue'):
            self.determine_masked_pixels()
            self.make_masked_catalogue()
        
        w1_real, w2_real = self.masked_catalogue['w1'], self.masked_catalogue['w2']
        resampled_indexes = np.random.choice(len(w1_real), n_samples)
        w1_resampled = w1_real[resampled_indexes].astype('float32')
        w2_resampled = w2_real[resampled_indexes].astype('float32')
        
        return w1_resampled, w2_resampled
    
    def resample_colour_mag_distribution(
            self,
            n_samples: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not self.catalogue_is_loaded:
            self.load_catalogue()
        
        w1_real, w12_real = self.catalogue['w1'], self.catalogue['w12']
        resampled_indexes = np.random.choice(len(w1_real), n_samples)
        w1_resampled = w1_real[resampled_indexes].astype('float32')
        w12_resampled = w12_real[resampled_indexes].astype('float32')

        return w1_resampled, w12_resampled

    def sample_magnitudes(
            self,
            n_samples: int,
            dtype: type = np.float64,
            rng: Optional[np.random.Generator] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''
        :return: Tuple of 64-bit numpy arrays representing the w1 and w2 mag samples.
        '''
        w1_samples, w2_samples = self.colour_mag_sampler.sample(n_samples, rng=rng)
        return w1_samples.astype(dtype), w2_samples.astype(dtype)

    def sample_clustered_points(
            self,
            parent_lon_deg: NDArray[np.floating],
            parent_lat_deg: NDArray[np.floating],
            per_parent_n_components: NDArray[np.integer],
            rng: Optional[np.random.Generator] = None,
            dtype: type = np.float64,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Sample clustered component positions around parent sources."""
        if rng is None:
            rng = np.random.default_rng()

        counts = np.asarray(per_parent_n_components, dtype=np.int64)
        total_n_components = int(counts.sum())
        if total_n_components == 0:
            empty = np.empty(0, dtype=dtype)
            return empty, empty

        parent_indices = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        parent_lon_rad = np.deg2rad(
            np.asarray(parent_lon_deg, dtype=np.float64)[parent_indices]
        )
        parent_lat_rad = np.deg2rad(
            np.asarray(parent_lat_deg, dtype=np.float64)[parent_indices]
        )

        phi = rng.uniform(0.0, 2.0 * np.pi, size=total_n_components)
        radial_arcsec = self.cfg.cluster_r_cut_arcsec + rng.exponential(
            scale=self.cfg.cluster_r0_arcsec,
            size=total_n_components,
        )
        angular_distance_rad = np.deg2rad(radial_arcsec / 3600.0)

        sin_parent_lat = np.sin(parent_lat_rad)
        cos_parent_lat = np.cos(parent_lat_rad)
        sin_distance = np.sin(angular_distance_rad)
        cos_distance = np.cos(angular_distance_rad)

        child_lat_rad = np.arcsin(
            sin_parent_lat * cos_distance
            + cos_parent_lat * sin_distance * np.cos(phi)
        )
        child_lon_rad = parent_lon_rad + np.arctan2(
            np.sin(phi) * sin_distance * cos_parent_lat,
            cos_distance - sin_parent_lat * np.sin(child_lat_rad),
        )

        child_lon_deg = np.mod(np.rad2deg(child_lon_rad), 360.0)
        child_lat_deg = np.rad2deg(child_lat_rad)
        return (
            child_lon_deg.astype(dtype, copy=False),
            child_lat_deg.astype(dtype, copy=False),
        )
    
    def sample_points(
            self,
            n_points: int,
            dtype: type = np.float64,
            generate_correlated_points: bool = False,
            log10_cluster_scale_param: Optional[float] = None,
            cluster_rate_param: Optional[float] = None,
            rng: Optional[np.random.Generator] = None
    ) -> tuple[NDArray, NDArray]:
        if generate_correlated_points:
            assert log10_cluster_scale_param is not None
            assert cluster_rate_param is not None

            if rng is None:
                rng = np.random.default_rng()

            longitudes_deg_list: list[NDArray] = []
            latitudes_deg_list: list[NDArray] = []
            total = 0

            kappa = 10 ** log10_cluster_scale_param

            while total < n_points:
                remaining = n_points - total
                n_parents = max(1, int(np.ceil(remaining / cluster_rate_param)))
                long_parent_deg, lat_parent_deg = sample_spherical_points(
                    n_parents, rng=rng
                )
                xyz = spherical_to_cart_deg(long_parent_deg, lat_parent_deg)

                child_lon_deg, child_lat_deg = generate_clusters(
                    parent_unit_vectors=xyz,
                    cluster_rate_param=cluster_rate_param,
                    kappa=kappa,
                    rng=rng
                )

                if child_lon_deg.size == 0:
                    continue

                longitudes_deg_list.append(child_lon_deg)
                latitudes_deg_list.append(child_lat_deg)
                total += child_lon_deg.size

            longitudes_deg = np.concatenate(longitudes_deg_list)
            latitudes_deg = np.concatenate(latitudes_deg_list)

            if total > n_points:
                keep_idx = rng.choice(total, n_points, replace=False)
                longitudes_deg = longitudes_deg[keep_idx]
                latitudes_deg = latitudes_deg[keep_idx]
        else:
            longitudes_deg, latitudes_deg = sample_spherical_points(n_points, rng=rng)

        return longitudes_deg.astype(dtype), latitudes_deg.astype(dtype)
    
    def aberrate_points(self,
            longitudes_deg: NDArray,
            latitudes_deg: NDArray,
            dtype: type = np.float64
        ) -> tuple[NDArray, NDArray, NDArray]:

        # Convert to float64 for calculations if using float32, then convert back
        if dtype == np.float32:
            calc_lon = longitudes_deg.astype(np.float64)
            calc_lat = latitudes_deg.astype(np.float64)
        else:
            calc_lon = longitudes_deg
            calc_lat = latitudes_deg
            
        boosted_lon_deg, boosted_lat_deg, rest_source_to_dipole_angle = aberrate_points(
            rest_longitudes=calc_lon,
            rest_latitudes=calc_lat,
            observer_direction=(self.dipole_longitude, self.dipole_latitude),
            observer_speed=self.observer_speed,
            rotation_matrices=self._rotation_matrices
        )
        
        # Convert back to requested dtype
        if dtype == np.float32:
            return (
                boosted_lon_deg.astype(dtype), 
                boosted_lat_deg.astype(dtype), 
                rest_source_to_dipole_angle.astype(dtype)
            )
        else:
            return boosted_lon_deg, boosted_lat_deg, rest_source_to_dipole_angle

    def boost_magnitudes(self,
            magnitudes: NDArray,
            rest_source_to_dipole_angle: NDArray,
            spectral_index: NDArray,
            dtype: type = np.float64
        ) -> NDArray[np.float64]:
         # Convert to float64 for calculations if using float32, then convert back
        if dtype == np.float32:
            calc_mags = magnitudes.astype(np.float64)
            calc_angles = rest_source_to_dipole_angle.astype(np.float64)
            calc_spectral = spectral_index.astype(np.float64)
        else:
            calc_mags = magnitudes
            calc_angles = rest_source_to_dipole_angle
            calc_spectral = spectral_index
             
        boosted_magnitudes = boost_magnitudes(
            magnitudes=calc_mags,
            angle_to_source=calc_angles,
            observer_speed=self.observer_speed,
            spectral_index=calc_spectral
        )
        
        # Convert back to requested dtype
        if dtype == np.float32:
            return boosted_magnitudes.astype(dtype)
        else:
            return boosted_magnitudes
