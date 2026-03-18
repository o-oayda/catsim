from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from dipoleutils.utils.data_loader import DataLoader
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from .utils.constants import CMB_BETA, CMB_L, CMB_B
from .utils.healsphere import downgrade_ignore_nan
from .utils.physics import (
    aberrate_points as aberrate_spherical_points,
    doppler_shift_factor,
    rotation_matrices_for_dipole,
    sample_spherical_points,
)
from .utils.rng import NPKey
from .utils.weather import get_temperatures_for_mjd


@dataclass
class RacsLow3Config:
    flux_min: float
    nside: int = 64
    chunk_size: int = 50_000
    use_float32: bool = False
    downscale_nside: Optional[int] = None
    store_final_samples: bool = True
    catalogue_path: Optional[str] = None
    flux_hist_bins: int = 200
    alpha_mean: float = 0.8
    alpha_sigma: float = 0.2
    fractional_error_flux_min_mjy: float = 10.0

    def __post_init__(self) -> None:
        if self.flux_min <= 0:
            raise ValueError("flux_min must be positive.")
        if self.nside != 64:
            raise ValueError("RacsLow3 currently requires nside=64 to use the packaged mask.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if self.flux_hist_bins < 2:
            raise ValueError("flux_hist_bins must be at least 2.")
        if self.alpha_sigma <= 0:
            raise ValueError("alpha_sigma must be positive.")
        if self.fractional_error_flux_min_mjy <= 0:
            raise ValueError("fractional_error_flux_min_mjy must be positive.")
        if self.downscale_nside is not None:
            if self.downscale_nside > self.nside:
                raise ValueError("downscale_nside must be <= nside.")
            ratio = self.nside // self.downscale_nside
            if (self.nside % self.downscale_nside) != 0 or (ratio & (ratio - 1)) != 0:
                raise ValueError(
                    "downscale_nside must be a power-of-two divisor of nside."
                )


class RacsLow3:
    """Skeleton RACS-low3 simulator following the Catwise initialise/simulate split."""

    def __init__(self, config: RacsLow3Config):
        self.cfg = config
        self.nside = config.nside
        self.dtype = np.float32 if config.use_float32 else np.float64
        self.chunk_size = config.chunk_size
        self.downscale_nside = config.downscale_nside
        self.store_final_samples = config.store_final_samples

        self.catalogue_is_loaded = False
        self.lookups_are_initialised = False

        self.observer_speed = CMB_BETA
        self.dipole_longitude = CMB_L
        self.dipole_latitude = CMB_B
        self.dipole_ra, self.dipole_dec = self._galactic_to_equatorial(
            self.dipole_longitude,
            self.dipole_latitude,
        )
        self._rotation_matrices = rotation_matrices_for_dipole(
            dipole_longitude=self.dipole_ra,
            dipole_latitude=self.dipole_dec,
        )

        self._density_map: Optional[NDArray[np.float32]] = None
        self._coarse_density_map: Optional[NDArray[np.float32]] = None
        self._coarse_mask: Optional[NDArray[np.bool_]] = None
        self.temperature_map: Optional[NDArray[np.float32]] = None
        self.fractional_error_map: Optional[NDArray[np.float32]] = None
        self.sampled_fractional_error_map: Optional[NDArray[np.float32]] = None

        self.final_intrinsic_flux_samples: Optional[NDArray[np.float32]] = None
        self.final_observed_flux_samples: Optional[NDArray[np.float32]] = None
        self.final_alpha_samples: Optional[NDArray[np.float32]] = None
        self.final_flux_error_samples: Optional[NDArray[np.float32]] = None
        self.final_fractional_error_samples: Optional[NDArray[np.float32]] = None
        self.final_base_fractional_error_samples: Optional[NDArray[np.float32]] = None
        self.final_pixel_indices: Optional[NDArray[np.int32]] = None
        self.final_tile_indices: Optional[NDArray[np.int32]] = None
        self.final_longitudes: Optional[NDArray[np.float32]] = None
        self.final_latitudes: Optional[NDArray[np.float32]] = None
        self.final_temperature_samples: Optional[NDArray[np.float32]] = None

    def _cache_dir(self) -> Path:
        return Path(__file__).resolve().parent / "data" / "racs_low3" / "lookups"

    def _sbid_lookup_cache_path(self) -> Path:
        return self._cache_dir() / f"sbid_lookup_nside{self.nside}.npz"

    def _temperature_lookup_cache_path(self) -> Path:
        return self._cache_dir() / f"temperature_lookup_nside{self.nside}_openmeteo_askap.npz"

    def _fractional_error_lookup_cache_path(self) -> Path:
        flux_token = str(self.cfg.fractional_error_flux_min_mjy).replace(".", "p")
        return self._cache_dir() / (
            f"fractional_error_lookup_nside{self.nside}_fluxmin{flux_token}mjy.npz"
        )

    def _mask_map_path(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "data"
            / "racs_low3"
            / "racs-low3_mask_nside64_ring.npy"
        )

    def load_catalogue(self) -> None:
        """Load the real RACS-low3 catalogue from a configured path or dipole-utils."""
        if self.cfg.catalogue_path is not None:
            catalogue_path = Path(self.cfg.catalogue_path).expanduser()
            if not catalogue_path.exists():
                raise FileNotFoundError(
                    f"RacsLow3Config.catalogue_path points to missing file: {catalogue_path}"
                )
            self.catalogue = Table.read(catalogue_path, unit_parse_strict="silent")
        else:
            self.catalogue = DataLoader("racs", "low3").load()

        self.catalogue_is_loaded = True

    def _galactic_to_equatorial(
        self,
        galactic_longitude: float,
        galactic_latitude: float,
    ) -> tuple[float, float]:
        """Convert Galactic ``l,b`` in degrees to equatorial ``RA,Dec`` in degrees."""
        coord = SkyCoord(
            l=galactic_longitude * u.deg,
            b=galactic_latitude * u.deg,
            frame="galactic",
        )
        equatorial = coord.icrs
        return float(equatorial.ra.deg), float(equatorial.dec.deg)

    def build_flux_distribution(self) -> None:
        """Build the empirical 1D log-flux sampler used by ``sample_fluxes``."""
        assert self.catalogue_is_loaded, "Load the catalogue before building flux lookups."

        flux = np.asarray(self.catalogue["Total_flux"], dtype=np.float64)
        flux = flux[np.isfinite(flux) & (flux > 0)]
        if flux.size == 0:
            raise ValueError("No positive finite Total_flux values available.")

        log_flux = np.log10(flux)
        counts, edges = np.histogram(log_flux, bins=self.cfg.flux_hist_bins)
        if not np.any(counts > 0):
            raise ValueError("Flux histogram contains no populated bins.")

        probabilities = counts.astype(np.float64)
        probabilities /= probabilities.sum()

        self.log_flux_bin_edges = edges.astype(np.float64, copy=False)
        self.log_flux_bin_probabilities = probabilities
        self.log_flux_bin_cdf = np.cumsum(probabilities)

    def build_tile_lookup(self) -> None:
        """Build a simple HEALPix survey mask and per-pixel dominant SBID lookup."""
        assert self.catalogue_is_loaded, "Load the catalogue before building tile lookups."

        ra = np.asarray(self.catalogue["RA"], dtype=np.float64)
        dec = np.asarray(self.catalogue["Dec"], dtype=np.float64)
        sbid = np.asarray(self.catalogue["SBID"], dtype=np.int64)

        pixel_indices = hp.ang2pix(self.nside, ra, dec, lonlat=True, nest=True)
        n_pix = hp.nside2npix(self.nside)

        self.tile_lookup_map = np.full(n_pix, -1, dtype=np.int32)
        order = np.argsort(pixel_indices, kind="stable")
        pix_sorted = pixel_indices[order]
        sbid_sorted = sbid[order]

        unique_pixels, starts, counts = np.unique(
            pix_sorted,
            return_index=True,
            return_counts=True,
        )
        for pix, start, count in zip(unique_pixels, starts, counts):
            sbid_values = sbid_sorted[start:start + count]
            sbid_unique, sbid_counts = np.unique(sbid_values, return_counts=True)
            self.tile_lookup_map[pix] = int(sbid_unique[np.argmax(sbid_counts)])

        self.mask_map = self.tile_lookup_map >= 0

    def save_tile_lookup(self) -> None:
        """Persist the HEALPix SBID lookup derived from the uncut catalogue."""
        cache_path = self._sbid_lookup_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            nside=np.asarray(self.nside, dtype=np.int64),
            tile_lookup_map=self.tile_lookup_map.astype(np.int32, copy=False),
        )

    def load_tile_lookup(self) -> bool:
        """Load a cached HEALPix SBID lookup if one exists and matches this config."""
        cache_path = self._sbid_lookup_cache_path()
        if not cache_path.exists():
            return False

        with np.load(cache_path) as data:
            cache_nside = int(data["nside"])
            if cache_nside != self.nside:
                return False
            self.tile_lookup_map = data["tile_lookup_map"].astype(np.int32, copy=False)

        return True

    def load_mask_map(self) -> None:
        """Load the packaged equatorial RACS-low3 mask and convert RING -> NEST."""
        mask_path = self._mask_map_path()
        if not mask_path.exists():
            raise FileNotFoundError(f"Packaged RACS-low3 mask not found: {mask_path}")

        mask_map_ring = np.load(mask_path, allow_pickle=False)
        mask_map = hp.reorder(mask_map_ring, r2n=True)
        if mask_map.shape != (hp.nside2npix(self.nside),):
            raise ValueError(
                "Packaged RACS-low3 mask has unexpected shape: "
                f"{mask_map.shape}, expected {(hp.nside2npix(self.nside),)}"
            )

        self.mask_map = np.asarray(mask_map == 1, dtype=np.bool_)

    def build_tile_metadata(self) -> None:
        """Collect one row of metadata per SBID for later tile-level systematics."""
        assert self.catalogue_is_loaded, "Load the catalogue before building tile metadata."

        sbid = np.asarray(self.catalogue["SBID"], dtype=np.int64)
        field_id = np.asarray(self.catalogue["Field_ID"])
        scan_start_mjd = np.asarray(self.catalogue["Scan_start_MJD"], dtype=np.float64)
        scan_length = np.asarray(self.catalogue["Scan_length"], dtype=np.float64)

        unique_sbid, first_indices = np.unique(sbid, return_index=True)
        self.tile_sbids = unique_sbid.astype(np.int32, copy=False)
        self.tile_scan_start_mjd = scan_start_mjd[first_indices].astype(np.float64, copy=False)
        self.tile_scan_length = scan_length[first_indices].astype(np.float64, copy=False)
        self.tile_field_id = field_id[first_indices]
        self._tile_index_from_sbid = {
            int(tile_sbid): int(tile_index)
            for tile_index, tile_sbid in enumerate(self.tile_sbids)
        }

    def build_temperature_map(self) -> None:
        """Project tile temperatures onto the HEALPix survey footprint."""
        n_pix = hp.nside2npix(self.nside)
        temperature_map = np.full(n_pix, np.nan, dtype=np.float32)

        if self.tile_temperature_by_index is None:
            self.temperature_map = temperature_map
            return

        valid_pixels = self.tile_lookup_map >= 0
        if np.any(valid_pixels):
            tile_indices = np.array(
                [
                    self._tile_index_from_sbid[int(tile_sbid)]
                    for tile_sbid in self.tile_lookup_map[valid_pixels]
                ],
                dtype=np.int32,
            )
            temperature_map[valid_pixels] = self.tile_temperature_by_index[
                tile_indices
            ].astype(np.float32, copy=False)

        self.temperature_map = temperature_map

    def save_temperature_lookup(self) -> None:
        """Persist the per-tile and per-pixel temperature lookup."""
        if self.tile_temperature_by_index is None:
            return

        cache_path = self._temperature_lookup_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            nside=np.asarray(self.nside, dtype=np.int64),
            tile_sbids=self.tile_sbids.astype(np.int32, copy=False),
            tile_temperature_by_index=self.tile_temperature_by_index.astype(
                np.float64, copy=False
            ),
        )

    def load_temperature_lookup(self) -> bool:
        """Load a cached per-tile temperature lookup if it matches the tile metadata."""
        cache_path = self._temperature_lookup_cache_path()
        if not cache_path.exists():
            return False

        with np.load(cache_path) as data:
            cache_nside = int(data["nside"])
            if cache_nside != self.nside:
                return False

            cache_tile_sbids = data["tile_sbids"].astype(np.int32, copy=False)
            if cache_tile_sbids.shape != self.tile_sbids.shape:
                return False
            if not np.array_equal(cache_tile_sbids, self.tile_sbids):
                return False

            self.tile_temperature_by_index = data["tile_temperature_by_index"].astype(
                np.float64,
                copy=False,
            )

        self.build_temperature_map()
        return True

    def build_fractional_error_lookup(self) -> None:
        """Build a per-pixel empirical lookup of fractional flux errors."""
        assert self.catalogue_is_loaded, "Load the catalogue before building error lookups."

        flux = np.asarray(self.catalogue["Total_flux"], dtype=np.float64)
        flux_error = np.asarray(self.catalogue["E_Total_flux"], dtype=np.float64)
        ra = np.asarray(self.catalogue["RA"], dtype=np.float64)
        dec = np.asarray(self.catalogue["Dec"], dtype=np.float64)

        valid = (
            np.isfinite(flux)
            & np.isfinite(flux_error)
            & (flux > 0)
            & (flux >= self.cfg.fractional_error_flux_min_mjy)
        )
        if not np.any(valid):
            raise ValueError("No valid sources available to build fractional-error lookup.")

        pixel_indices = hp.ang2pix(
            self.nside,
            ra[valid],
            dec[valid],
            lonlat=True,
            nest=True,
        ).astype(np.int64, copy=False)
        fractional_error = (flux_error[valid] / flux[valid]).astype(np.float32, copy=False)

        order = np.argsort(pixel_indices, kind="stable")
        pix_sorted = pixel_indices[order]
        frac_sorted = fractional_error[order]

        n_pix = hp.nside2npix(self.nside)
        counts = np.bincount(pix_sorted, minlength=n_pix).astype(np.int64)
        starts = np.cumsum(counts, dtype=np.int64) - counts

        self.error_lookup_pixel_counts = counts
        self.error_lookup_pixel_starts = starts
        self.error_lookup_fractional_values = frac_sorted

        fractional_error_map = np.full(n_pix, np.nan, dtype=np.float32)
        populated = counts > 0
        if np.any(populated):
            populated_pixels = np.flatnonzero(populated)
            for pix in populated_pixels:
                start = starts[pix]
                count = counts[pix]
                fractional_error_map[pix] = np.median(frac_sorted[start:start + count])
        self.fractional_error_map = fractional_error_map

    def save_fractional_error_lookup(self) -> None:
        """Persist the per-pixel fractional-error lookup."""
        cache_path = self._fractional_error_lookup_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            nside=np.asarray(self.nside, dtype=np.int64),
            flux_min_mjy=np.asarray(self.cfg.fractional_error_flux_min_mjy, dtype=np.float64),
            counts=self.error_lookup_pixel_counts.astype(np.int64, copy=False),
            starts=self.error_lookup_pixel_starts.astype(np.int64, copy=False),
            fractional_error=self.error_lookup_fractional_values.astype(np.float32, copy=False),
        )

    def load_fractional_error_lookup(self) -> bool:
        """Load the cached per-pixel fractional-error lookup if available."""
        cache_path = self._fractional_error_lookup_cache_path()
        if not cache_path.exists():
            return False

        with np.load(cache_path) as data:
            cache_nside = int(data["nside"])
            if cache_nside != self.nside:
                return False
            self.error_lookup_pixel_counts = data["counts"].astype(np.int64, copy=False)
            self.error_lookup_pixel_starts = data["starts"].astype(np.int64, copy=False)
            self.error_lookup_fractional_values = data["fractional_error"].astype(
                np.float32,
                copy=False,
            )

        n_pix = hp.nside2npix(self.nside)
        fractional_error_map = np.full(n_pix, np.nan, dtype=np.float32)
        populated = self.error_lookup_pixel_counts > 0
        if np.any(populated):
            populated_pixels = np.flatnonzero(populated)
            for pix in populated_pixels:
                start = self.error_lookup_pixel_starts[pix]
                count = self.error_lookup_pixel_counts[pix]
                fractional_error_map[pix] = np.median(
                    self.error_lookup_fractional_values[start:start + count]
                )
        self.fractional_error_map = fractional_error_map
        return True

    def sample_fractional_errors(
        self,
        pixel_indices: NDArray[np.int_],
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.float32]:
        """Sample fractional flux errors from the empirical distribution of each pixel."""
        assert hasattr(self, "error_lookup_pixel_counts"), "Run initialise_data() first."
        if rng is None:
            rng = np.random.default_rng()

        pix = np.asarray(pixel_indices, dtype=np.int64)
        counts = self.error_lookup_pixel_counts[pix]
        starts = self.error_lookup_pixel_starts[pix]

        out = np.empty(pix.shape[0], dtype=np.float32)
        valid = counts > 0
        if np.any(valid):
            rand_offsets = rng.integers(0, counts[valid], dtype=np.int64)
            pick = starts[valid] + rand_offsets
            out[valid] = self.error_lookup_fractional_values[pick]

        if np.any(~valid):
            pick = rng.integers(
                0,
                self.error_lookup_fractional_values.size,
                size=np.count_nonzero(~valid),
                dtype=np.int64,
            )
            out[~valid] = self.error_lookup_fractional_values[pick]

        return out

    def compute_total_flux_error(
        self,
        flux_density: NDArray[np.floating],
        fractional_error: NDArray[np.floating],
        fractional_error_eta: float = 0.0,
        dtype: type = np.float64,
    ) -> NDArray[np.floating]:
        """Convert sampled fractional errors into raw flux-error sigmas."""
        if fractional_error_eta < 0:
            raise ValueError("fractional_error_eta must be non-negative.")

        flux = np.asarray(flux_density, dtype=np.float64)
        frac = np.asarray(fractional_error, dtype=np.float64)
        sigma = frac * flux
        sigma *= np.sqrt(1.0 + fractional_error_eta)
        return sigma.astype(dtype, copy=False)

    def add_flux_error(
        self,
        flux_density: NDArray[np.floating],
        flux_error: NDArray[np.floating],
        rng: Optional[np.random.Generator] = None,
        dtype: type = np.float64,
    ) -> NDArray[np.floating]:
        """Apply Gaussian flux noise with a precomputed raw flux-error sigma."""
        if rng is None:
            rng = np.random.default_rng()

        flux = np.asarray(flux_density, dtype=np.float64)
        sigma = np.asarray(flux_error, dtype=np.float64)
        noisy_flux = flux + rng.normal(loc=0.0, scale=sigma, size=flux.shape)
        return noisy_flux.astype(dtype, copy=False)

    def load_temperature_table(self) -> None:
        """Load or derive per-SBID temperatures and project them onto the sky."""
        self.tile_temperature_by_index = None

        if self.load_temperature_lookup():
            return

        fetched_temperatures_ok = True
        try:
            self.tile_temperature_by_index = np.asarray(
                get_temperatures_for_mjd(self.tile_scan_start_mjd),
                dtype=np.float64,
            )
        except Exception as exc:
            fetched_temperatures_ok = False
            warnings.warn(
                f"Unable to fetch ASKAP temperatures during initialise_data(): {exc}"
            )
            self.tile_temperature_by_index = np.full(
                self.tile_sbids.shape,
                np.nan,
                dtype=np.float64,
            )

        self.build_temperature_map()
        if fetched_temperatures_ok and np.any(np.isfinite(self.tile_temperature_by_index)):
            self.save_temperature_lookup()

    def initialise_data(self) -> None:
        """Initialise the catalogue-derived lookup tables used during simulation."""
        if not self.catalogue_is_loaded:
            self.load_catalogue()

        self.build_flux_distribution()
        self.build_tile_metadata()
        if not self.load_tile_lookup():
            self.build_tile_lookup()
            self.save_tile_lookup()
        self.load_mask_map()
        self.load_temperature_table()
        if not self.load_fractional_error_lookup():
            self.build_fractional_error_lookup()
            self.save_fractional_error_lookup()
        self.lookups_are_initialised = True

    def sample_fluxes(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.floating]:
        """Draw intrinsic fluxes from the empirical 1D log-flux histogram."""
        assert hasattr(self, "log_flux_bin_cdf"), "Run initialise_data() first."
        if rng is None:
            rng = np.random.default_rng()

        u = rng.random(n_samples)
        bin_indices = np.searchsorted(self.log_flux_bin_cdf, u, side="right")
        bin_indices = np.clip(bin_indices, 0, self.log_flux_bin_edges.size - 2)

        log_flux_low = self.log_flux_bin_edges[bin_indices]
        log_flux_high = self.log_flux_bin_edges[bin_indices + 1]
        sampled_log_flux = rng.uniform(log_flux_low, log_flux_high)
        flux = np.power(10.0, sampled_log_flux)
        return flux.astype(self.dtype, copy=False)

    def sample_points(
        self,
        n_points: int,
        dtype: type = np.float64,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[NDArray, NDArray]:
        """Sample isotropic sky positions in equatorial coordinates."""
        ra_deg, dec_deg = sample_spherical_points(n_points, rng=rng)
        return ra_deg.astype(dtype), dec_deg.astype(dtype)

    def sample_spectral_indices(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.float32]:
        """Draw per-source radio spectral indices using the current Gaussian model."""
        if rng is None:
            rng = np.random.default_rng()

        alpha = rng.normal(
            loc=self.cfg.alpha_mean,
            scale=self.cfg.alpha_sigma,
            size=n_samples,
        )
        return alpha.astype(np.float32, copy=False)

    def aberrate_points(
        self,
        ra_deg: NDArray,
        dec_deg: NDArray,
        dtype: type = np.float64,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Apply the cosmic-dipole aberration step used in CatSIM."""
        out_ra, out_dec, source_to_dipole_angle_deg = aberrate_spherical_points(
            rest_longitudes=ra_deg,
            rest_latitudes=dec_deg,
            observer_direction=(self.dipole_ra, self.dipole_dec),
            observer_speed=self.observer_speed,
            rotation_matrices=self._rotation_matrices,
        )
        return (
            out_ra.astype(dtype, copy=False),
            out_dec.astype(dtype, copy=False),
            source_to_dipole_angle_deg.astype(dtype, copy=False),
        )

    def boost_fluxes(
        self,
        flux_density: NDArray,
        angle_to_dipole_deg: NDArray,
        spectral_index: NDArray | float,
        dtype: type = np.float64,
    ) -> NDArray[np.floating]:
        """Apply the radio-flux dipole boost using ``S_nu ∝ nu^(-alpha)``."""
        delta = doppler_shift_factor(self.observer_speed, angle_to_dipole_deg)
        boosted_flux = np.asarray(flux_density, dtype=np.float64) * np.power(
            delta,
            1.0 + np.asarray(spectral_index, dtype=np.float64),
        )
        return boosted_flux.astype(dtype, copy=False)

    def assign_tiles(
        self,
        ra_deg: NDArray[np.floating],
        dec_deg: NDArray[np.floating],
    ) -> NDArray[np.int32]:
        """Assign each source to the dominant observed SBID in its HEALPix pixel."""
        assert hasattr(self, "tile_lookup_map"), "Run initialise_data() first."

        pixel_indices = hp.ang2pix(self.nside, ra_deg, dec_deg, lonlat=True, nest=True)
        tile_sbids = self.tile_lookup_map[pixel_indices]
        tile_indices = np.full(tile_sbids.shape, -1, dtype=np.int32)

        valid = tile_sbids >= 0
        if np.any(valid):
            tile_indices[valid] = np.array(
                [self._tile_index_from_sbid[int(sbid)] for sbid in tile_sbids[valid]],
                dtype=np.int32,
            )

        return tile_indices

    def evaluate_temperature_enhancement(
        self,
        tile_indices: NDArray[np.int32],
        temp_slope: float,
        temp_intercept: float,
        temp_pivot_c: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.float32]]:
        """Evaluate ``epsilon(T) = a (T / T_0) + b`` at the tile level."""
        if temp_pivot_c == 0:
            raise ValueError("temp_pivot_c must be non-zero.")

        enhancement = np.full(tile_indices.shape, temp_intercept, dtype=np.float64)
        temperatures = np.full(tile_indices.shape, np.nan, dtype=np.float32)

        if self.tile_temperature_by_index is None:
            return enhancement.astype(self.dtype, copy=False), temperatures

        valid = tile_indices >= 0
        if np.any(valid):
            tile_temperatures = self.tile_temperature_by_index[tile_indices[valid]]
            temperatures[valid] = tile_temperatures.astype(np.float32, copy=False)
            valid_temperature = np.isfinite(tile_temperatures)
            if np.any(valid_temperature):
                enhancement_valid = (
                    temp_slope
                    * (tile_temperatures[valid_temperature] / temp_pivot_c)
                    + temp_intercept
                )
                enhancement_indices = np.flatnonzero(valid)[valid_temperature]
                enhancement[enhancement_indices] = enhancement_valid

        return enhancement.astype(self.dtype, copy=False), temperatures

    def apply_temperature_enhancement(
        self,
        flux_density: NDArray[np.floating],
        enhancement: NDArray[np.floating],
        dtype: type = np.float64,
    ) -> NDArray[np.floating]:
        """Apply the ASKAP tile-level multiplicative systematic after dipole boosting."""
        observed_flux = np.asarray(flux_density, dtype=np.float64) * np.asarray(
            enhancement, dtype=np.float64
        )
        return observed_flux.astype(dtype, copy=False)

    def flux_cut_boolean(
        self,
        flux_density: NDArray[np.floating],
        flux_min: float,
    ) -> NDArray[np.bool_]:
        """Apply the survey flux threshold."""
        return np.asarray(flux_density >= flux_min, dtype=np.bool_)

    def _source_isin_mask(
        self,
        ra_deg: NDArray[np.floating],
        dec_deg: NDArray[np.floating],
    ) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
        """Return the survey-footprint mask and output pixel index for each source."""
        pixel_indices = hp.ang2pix(self.nside, ra_deg, dec_deg, lonlat=True, nest=True)
        mask_slice = self.mask_map[pixel_indices]
        return mask_slice.astype(np.bool_, copy=False), pixel_indices.astype(np.int64, copy=False)

    def _prepare_map_output(
        self,
        map_values: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Apply the survey mask and optional nside downgrading."""
        native_mask = self.mask_map.astype(np.bool_, copy=False)
        map_with_mask = np.asarray(map_values, dtype=np.float32).copy()
        map_with_mask[~native_mask] = np.nan

        if self.downscale_nside is None:
            self._coarse_density_map = None
            self._coarse_mask = None
            return map_with_mask, native_mask

        coarse_map, coarse_mask = downgrade_ignore_nan(
            map_with_mask,
            native_mask,
            self.downscale_nside,
        )
        coarse_map = coarse_map.astype(np.float32, copy=False)
        coarse_mask = coarse_mask.astype(np.bool_, copy=False)
        coarse_map = coarse_map.copy()
        coarse_map[~coarse_mask] = np.nan

        self._coarse_density_map = coarse_map
        self._coarse_mask = coarse_mask
        return coarse_map, coarse_mask

    def generate_dipole(
        self,
        log10_n_initial_samples: float,
        flux_min: Optional[float] = None,
        observer_speed: float = 1.0,
        dipole_longitude: float = CMB_L,
        dipole_latitude: float = CMB_B,
        temp_slope: float = 0.0,
        temp_intercept: float = 1.0,
        temp_pivot_c: float = 30.0,
        fractional_error_eta: float = 0.0,
        rng_key: Optional[NPKey] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Coordinate the CatSIM-like simulation pipeline for RACS-low3."""
        assert self.lookups_are_initialised, (
            "Lookup tables must be initialised before generating maps. "
            "Run initialise_data() first."
        )

        self.observer_speed = observer_speed * CMB_BETA
        self.dipole_longitude = dipole_longitude
        self.dipole_latitude = dipole_latitude
        self.dipole_ra, self.dipole_dec = self._galactic_to_equatorial(
            dipole_longitude,
            dipole_latitude,
        )
        self._rotation_matrices = rotation_matrices_for_dipole(
            dipole_longitude=self.dipole_ra,
            dipole_latitude=self.dipole_dec,
        )

        n_samples = int(10 ** log10_n_initial_samples)
        if n_samples < 0:
            raise ValueError("n_initial_samples must be non-negative.")

        active_flux_min = self.cfg.flux_min if flux_min is None else flux_min
        rng = rng_key._generator() if rng_key is not None else np.random.default_rng()

        n_pix = hp.nside2npix(self.nside)
        density_accumulator = np.zeros(n_pix, dtype=np.float64)
        fractional_error_sum = np.zeros(n_pix, dtype=np.float64)
        fractional_error_count = np.zeros(n_pix, dtype=np.int64)

        final_intrinsic_flux: list[NDArray[np.float32]] = []
        final_observed_flux: list[NDArray[np.float32]] = []
        final_alpha: list[NDArray[np.float32]] = []
        final_flux_error: list[NDArray[np.float32]] = []
        final_base_fractional_error: list[NDArray[np.float32]] = []
        final_fractional_error: list[NDArray[np.float32]] = []
        final_pixels: list[NDArray[np.int32]] = []
        final_tiles: list[NDArray[np.int32]] = []
        final_ra: list[NDArray[np.float32]] = []
        final_dec: list[NDArray[np.float32]] = []
        final_temperature: list[NDArray[np.float32]] = []

        for start in range(0, n_samples, self.chunk_size):
            current_chunk = min(self.chunk_size, n_samples - start)

            intrinsic_flux = self.sample_fluxes(current_chunk, rng=rng)
            rest_ra_deg, rest_dec_deg = self.sample_points(
                current_chunk, dtype=self.dtype, rng=rng
            )
            alpha = self.sample_spectral_indices(current_chunk, rng=rng)

            boosted_ra_deg, boosted_dec_deg, angle_to_dipole_deg = self.aberrate_points(
                rest_ra_deg,
                rest_dec_deg,
                dtype=self.dtype,
            )
            dipole_flux = self.boost_fluxes(
                intrinsic_flux,
                angle_to_dipole_deg,
                spectral_index=alpha,
                dtype=self.dtype,
            )

            mask_slice, pixel_indices = self._source_isin_mask(boosted_ra_deg, boosted_dec_deg)
            tile_indices = self.assign_tiles(boosted_ra_deg, boosted_dec_deg)
            enhancement, temperatures = self.evaluate_temperature_enhancement(
                tile_indices=tile_indices,
                temp_slope=temp_slope,
                temp_intercept=temp_intercept,
                temp_pivot_c=temp_pivot_c,
            )
            systematics_flux = self.apply_temperature_enhancement(
                dipole_flux,
                enhancement,
                dtype=self.dtype,
            )
            base_fractional_error = self.sample_fractional_errors(pixel_indices, rng=rng)
            flux_error = self.compute_total_flux_error(
                systematics_flux,
                base_fractional_error,
                fractional_error_eta=fractional_error_eta,
                dtype=self.dtype,
            )
            safe_flux = np.clip(
                np.asarray(systematics_flux, dtype=np.float64),
                np.finfo(np.float64).tiny,
                None,
            )
            fractional_error = (
                np.asarray(flux_error, dtype=np.float64) / safe_flux
            ).astype(np.float32, copy=False)
            observed_flux = self.add_flux_error(
                systematics_flux,
                flux_error,
                rng=rng,
                dtype=self.dtype,
            )

            cut_slice = self.flux_cut_boolean(observed_flux, active_flux_min)
            keep = mask_slice & cut_slice & (tile_indices >= 0)
            if not np.any(keep):
                continue

            kept_pixels = pixel_indices[keep]
            np.add.at(density_accumulator, kept_pixels, 1)
            np.add.at(fractional_error_sum, kept_pixels, fractional_error[keep].astype(np.float64))
            np.add.at(fractional_error_count, kept_pixels, 1)

            if self.store_final_samples:
                final_intrinsic_flux.append(
                    intrinsic_flux[keep].astype(np.float32, copy=False)
                )
                final_observed_flux.append(
                    observed_flux[keep].astype(np.float32, copy=False)
                )
                final_alpha.append(alpha[keep].astype(np.float32, copy=False))
                final_flux_error.append(
                    flux_error[keep].astype(np.float32, copy=False)
                )
                final_base_fractional_error.append(
                    base_fractional_error[keep].astype(np.float32, copy=False)
                )
                final_fractional_error.append(
                    fractional_error[keep].astype(np.float32, copy=False)
                )
                final_pixels.append(kept_pixels.astype(np.int32, copy=False))
                final_tiles.append(tile_indices[keep].astype(np.int32, copy=False))
                final_ra.append(boosted_ra_deg[keep].astype(np.float32, copy=False))
                final_dec.append(boosted_dec_deg[keep].astype(np.float32, copy=False))
                final_temperature.append(temperatures[keep].astype(np.float32, copy=False))

        self._density_map = density_accumulator.astype(np.float32, copy=False)
        sampled_fractional_error_map = np.full(n_pix, np.nan, dtype=np.float32)
        valid_error_pixels = fractional_error_count > 0
        if np.any(valid_error_pixels):
            sampled_fractional_error_map[valid_error_pixels] = (
                fractional_error_sum[valid_error_pixels]
                / fractional_error_count[valid_error_pixels]
            ).astype(np.float32, copy=False)
        sampled_fractional_error_map[~self.mask_map.astype(bool)] = np.nan
        self.sampled_fractional_error_map = sampled_fractional_error_map
        output_map, output_mask = self._prepare_map_output(self._density_map)

        if self.store_final_samples:
            self.final_intrinsic_flux_samples = (
                np.concatenate(final_intrinsic_flux) if final_intrinsic_flux else np.empty(0, dtype=np.float32)
            )
            self.final_observed_flux_samples = (
                np.concatenate(final_observed_flux) if final_observed_flux else np.empty(0, dtype=np.float32)
            )
            self.final_alpha_samples = (
                np.concatenate(final_alpha) if final_alpha else np.empty(0, dtype=np.float32)
            )
            self.final_flux_error_samples = (
                np.concatenate(final_flux_error)
                if final_flux_error else np.empty(0, dtype=np.float32)
            )
            self.final_base_fractional_error_samples = (
                np.concatenate(final_base_fractional_error)
                if final_base_fractional_error else np.empty(0, dtype=np.float32)
            )
            self.final_fractional_error_samples = (
                np.concatenate(final_fractional_error)
                if final_fractional_error else np.empty(0, dtype=np.float32)
            )
            self.final_pixel_indices = (
                np.concatenate(final_pixels) if final_pixels else np.empty(0, dtype=np.int32)
            )
            self.final_tile_indices = (
                np.concatenate(final_tiles) if final_tiles else np.empty(0, dtype=np.int32)
            )
            self.final_longitudes = (
                np.concatenate(final_ra) if final_ra else np.empty(0, dtype=np.float32)
            )
            self.final_latitudes = (
                np.concatenate(final_dec) if final_dec else np.empty(0, dtype=np.float32)
            )
            self.final_temperature_samples = (
                np.concatenate(final_temperature) if final_temperature else np.empty(0, dtype=np.float32)
            )
        else:
            self.final_intrinsic_flux_samples = None
            self.final_observed_flux_samples = None
            self.final_alpha_samples = None
            self.final_flux_error_samples = None
            self.final_base_fractional_error_samples = None
            self.final_fractional_error_samples = None
            self.final_pixel_indices = None
            self.final_tile_indices = None
            self.final_longitudes = None
            self.final_latitudes = None
            self.final_temperature_samples = None

        return output_map, output_mask
