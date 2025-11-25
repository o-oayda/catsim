from typing import Optional, Literal
from astropy.table import Table
from .configs import CatwiseConfig
from .utils.hists import MultinomialSample2DHistogram
from .utils.physics import (
    sample_spherical_points, aberrate_points, boost_magnitudes,
    rotation_matrices_for_dipole
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
        self.real_file_path = (
            'dipolesbi/catwise/catwise_agns_masked_final_w1lt16p5_alpha.fits'
        )

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
            log10_magnitude_error_shape_param: float = 0.,
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

        self.n_samples = int(10 ** log10_n_initial_samples)
        if self.n_samples < 0:
            raise ValueError('n_initial_samples must be non-negative.')

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
        final_indices_list: list[NDArray[np.int32]] = []
        final_alpha_list: list[NDArray[np.float32]] = []
        final_measured_alpha_list: list[NDArray[np.float32]] = []

        self.final_w1_samples = None
        self.final_w2_samples = None
        self.final_w12_samples = None
        self.final_pixel_indices = None

        coverage_query_buffer = np.empty((chunk_size, 2), dtype=self.dtype)
        colour_buffer = np.empty(chunk_size, dtype=self.dtype)
        spectral_buffer = np.empty(chunk_size, dtype=np.float32)
        noise_buffer_w1 = np.empty(chunk_size, dtype=np.float64)
        noise_buffer_w2 = np.empty(chunk_size, dtype=np.float64)

        for start in range(0, self.n_samples, chunk_size):
            current_chunk = min(chunk_size, self.n_samples - start)

            rest_w1_samples, rest_w2_samples = self.sample_magnitudes(
                current_chunk, dtype=self.dtype, rng=rng
            )
            rest_source_lon_deg, rest_source_lat_deg = self.sample_points(
                current_chunk,
                rng=rng
            )

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

            current_colour = colour_buffer[:rest_w1_samples.size]
            np.subtract(rest_w1_samples, rest_w2_samples, out=current_colour)

            spectral_indices = spectral_buffer[:current_colour.size]
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

            coverage_query = coverage_query_buffer[:boosted_w1_samples.size]
            coverage_query[:, 0] = boosted_w1_samples
            coverage_query[:, 1] = source_logw1_cov
            w1_error = self.w1mag_coverage_rgi(
                coverage_query
            ).astype(np.float32)

            coverage_query[:, 0] = boosted_w2_samples
            coverage_query[:, 1] = source_logw2_cov
            w2_error = self.w2mag_coverage_rgi(
                coverage_query
            ).astype(np.float32)

            current_noise_buffers = (
                noise_buffer_w1[:boosted_w1_samples.size],
                noise_buffer_w2[:boosted_w2_samples.size]
            )

            boosted_w1_samples, boosted_w2_samples = self.add_error(
                w1=(boosted_w1_samples, w1_error),
                w2=(boosted_w2_samples, w2_error),
                w1_extra_error=w1_extra_error,
                w2_extra_error=w2_extra_error,
                common_extra_error=common_extra_error,
                error_dist=magnitude_error_dist,
                log10_shape_param=log10_magnitude_error_shape_param,
                rng=rng,
                noise_buffers=current_noise_buffers
            )

            # after adding error
            boosted_w12_samples = boosted_w1_samples - boosted_w2_samples

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
                final_indices_list.append(cut_source_pixel_indices)
                final_alpha_list.append(true_cut_spectral_indices)
                final_measured_alpha_list.append(measured_cut_spectral_indices)

        self._density_map = density_accumulator.astype(np.float32)

        if store_final_samples:
            if final_w1_list:
                self.final_w1_samples = np.concatenate(final_w1_list)
                self.final_w2_samples = np.concatenate(final_w2_list)
                self.final_w12_samples = np.concatenate(final_w12_list)
                self.final_pixel_indices = np.concatenate(final_indices_list)
                self.final_alpha_samples = np.concatenate(final_alpha_list)
                self.final_measured_alpha_samples = np.concatenate(
                    final_measured_alpha_list
                )
            else:
                self.final_w1_samples = np.empty(0, dtype=np.float32)
                self.final_w2_samples = np.empty(0, dtype=np.float32)
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
    
    def make_real_sample(self) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        print(f'Reading in CatWISE2020 from {self.real_file_path}...')
        self.real_catalogue = Table.read(self.real_file_path)
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

    def add_error(self,
            w1: tuple[NDArray, NDArray],
            w2: tuple[NDArray, NDArray],
            w1_extra_error: Optional[float] = None,
            w2_extra_error: Optional[float] = None,
            common_extra_error: Optional[bool] = False,
            error_dist: Literal['gaussian', 'students-t'] = 'gaussian',
            log10_shape_param: float = 0.,
            rng: Optional[np.random.Generator] = None,
            noise_buffers: Optional[tuple[NDArray[np.float64], NDArray[np.float64]]] = None
    ) -> tuple[NDArray, NDArray]:
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

        if common_extra_error and w1_extra_error is not None:
            w2_extra_error = w1_extra_error

        w1_sigma = np.array(w1_error, dtype=w1_dtype, copy=True)
        w2_sigma = np.array(w2_error, dtype=w2_dtype, copy=True)

        # sigma^2 = sigma_b^2 + extra * sigma_b^2 => sigma = sigma_b sqrt( 1 + extra )
        if w1_extra_error is not None:
            np.multiply(
                w1_sigma,
                np.sqrt(np.array(1.0 + w1_extra_error, dtype=w1_dtype)),
                out=w1_sigma,
                casting='unsafe'
            )

        if w2_extra_error is not None:
            np.multiply(
                w2_sigma,
                np.sqrt(np.array(1.0 + w2_extra_error, dtype=w2_dtype)),
                out=w2_sigma,
                casting='unsafe'
            )

        if rng is None:
            rng = np.random.default_rng()

        if noise_buffers is not None:
            noise_w1_buf, noise_w2_buf = noise_buffers
            noise_w1 = noise_w1_buf[:w1_sigma.size]
            noise_w2 = noise_w2_buf[:w2_sigma.size]
        else:
            noise_w1 = None
            noise_w2 = None

        if error_dist == 'gaussian':
            if noise_w1 is not None and noise_w2 is not None:
                rng.standard_normal(out=noise_w1)
                rng.standard_normal(out=noise_w2)
            else:
                noise_w1 = rng.standard_normal(size=w1_sigma.shape)
                noise_w2 = rng.standard_normal(size=w2_sigma.shape)
        else:
            shape_param = float(10 ** log10_shape_param)
            if shape_param <= 0:
                raise ValueError("Student's t shape parameter must be positive.")
            if noise_w1 is not None and noise_w2 is not None:
                noise_w1[:] = rng.standard_t(df=shape_param, size=w1_sigma.shape)
                noise_w2[:] = rng.standard_t(df=shape_param, size=w2_sigma.shape)
            else:
                noise_w1 = rng.standard_t(df=shape_param, size=w1_sigma.shape)
                noise_w2 = rng.standard_t(df=shape_param, size=w2_sigma.shape)

        calc_dtype_w1 = np.float64 if w1_dtype == np.float64 else np.float32
        calc_dtype_w2 = np.float64 if w2_dtype == np.float64 else np.float32

        noisy_w1 = (
            np.asarray(w1_magnitudes, dtype=calc_dtype_w1)
          + noise_w1.astype(calc_dtype_w1, copy=False)
          * np.asarray(w1_sigma, dtype=calc_dtype_w1)
        ).astype(w1_dtype, copy=False)

        noisy_w2 = (
            np.asarray(w2_magnitudes, dtype=calc_dtype_w2)
          + noise_w2.astype(calc_dtype_w2, copy=False)
          * np.asarray(w2_sigma, dtype=calc_dtype_w2)
        ).astype(w2_dtype, copy=False)

        return noisy_w1, noisy_w2
    
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

    def precompute_data(self): 
        # load catalogue and mask
        if not self.catalogue_is_loaded:
            self.load_catalogue()
        self.determine_masked_pixels()
        self.make_masked_catalogue()

        self.create_w1_w2_distribution()
        # self.create_spectral_index_distribution(no_check=no_check)
        # self.create_error_map()
        self.create_coverage_maps()
        self.create_magnitude_coverage_function()
    
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

    def create_magnitude_coverage_function(self):
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
                statistic='median',
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
            file_path = f'dipolesbi/catwise/{self.cut_path}/data/mag_coverage'
            self.save_interpolator(
                band=band,
                interpolator=rgi,
                mag_bins=magnitude_bins,
                cov_bins=coverage_bins,
                filled_grid=filled_median_error_grid,
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

    def save_interpolator(self,
            band: str,
            interpolator: RegularGridInterpolator,
            mag_bins: NDArray[np.float64],
            cov_bins: NDArray[np.float64],
            filled_grid: NDArray[np.float64],
            file_path: str
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
        self.colour_mag_sampler = MultinomialSample2DHistogram()
        with data_path(self.cut_path, 'colour_mag') as colour_mag_dir:
            self.colour_mag_sampler.load_data(colour_mag_dir)
        with data_path(
            self.cut_path,
            'mag_coverage',
            'w1_median_error_interpolator.pkl'
        ) as w1_mag_interpolator:
            self.w1mag_coverage_rgi = self.load_interpolator(w1_mag_interpolator)
        with data_path(
            self.cut_path,
            'mag_coverage',
            'w2_median_error_interpolator.pkl'
        ) as w2_mag_interpolator:
            self.w2mag_coverage_rgi = self.load_interpolator(w2_mag_interpolator)

        # loads things back into numpy
        with data_path(
            self.cut_path,
            'coverage_map',
            'w1_coverage_map.npy'
        ) as w1_cov_path:
            self.w1cov_map = np.load(
                w1_cov_path,
                allow_pickle=False
            ).astype(np.float32, copy=False)

        with data_path(
            self.cut_path,
            'coverage_map',
            'w2_coverage_map.npy'
        ) as w2_cov_path:
            self.w2cov_map = np.load(
                w2_cov_path,
                allow_pickle=False
            ).astype(np.float32, copy=False)

        self.log_w1cov_map = np.log10(self.w1cov_map).astype(self.dtype)
        self.log_w2cov_map = np.log10(self.w2cov_map).astype(self.dtype)

        # initialise AlphaLookup so table is not read in at each simulation
        self.spectral_lookup = AlphaLookup()
        
        # mask now instead of during each loop
        self.determine_masked_pixels()

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
        path = f'dipolesbi/catwise/{self.cut_path}/data/colour_mag/'
        sampler.save_data(path)
        print(f'Saved W1-W2 distribution to {path}.')

        print(f'Generating W1-W2 distribution plot...')
        w1_samples, w2_samples = sampler.sample(n_samples=30_000_000)
        plt.hist2d(w1_samples, w2_samples, bins=400, norm='log')
        plt.savefig(f'{path}w1_w2_dist.png', dpi=300)
    
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
    
    def sample_points(
            self,
            n_points: int,
            dtype: type = np.float64,
            rng: Optional[np.random.Generator] = None
    ) -> tuple[NDArray, NDArray]:
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
