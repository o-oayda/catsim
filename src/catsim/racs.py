from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile
from typing import Literal, Optional

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from dipoleutils.utils.data_loader import DataLoader
import healpy as hp
import matplotlib.pyplot as plt
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
from .utils.weather import (
    ASKAP_LATITUDE_DEG,
    ASKAP_LONGITUDE_DEG,
    get_mean_paf_temperatures_for_observations,
    get_open_meteo_temperatures_for_mjd,
)
from .racs_products import RACS_LOW3, RacsProductSpec, resolve_racs_product
from .racs_noise import (
    RacsCacheValidationError,
    build_conditional_error_lookup,
    build_noise_map_cache,
    load_conditional_error_lookup,
    load_noise_map_cache,
    save_conditional_error_lookup,
    save_noise_map_cache,
)
from .racs_summaries import (
    binned_flux_quantiles_exact,
    validate_bin_edges,
    validate_quantiles,
)
from .racs_temperature import (
    RACS_TEMPERATURE_EPSILON_FLOOR,
    TEMPERATURE_MODELS,
    TemperatureModel,
    evaluate_temperature_response,
)

LOW3_TEMPERATURE_EPSILON_FLOOR = RACS_TEMPERATURE_EPSILON_FLOOR
LOGGER = logging.getLogger(__name__)
TemperatureSource = Literal["mean_paf", "open_meteo", "reference"]
TEMPERATURE_CACHE_FORMAT_VERSION = 2


@dataclass
class RacsConfig:
    """Configuration for the RACS simulator.

    ``cluster_count_model`` selects how many component sources each parent can
    add. ``"geometric"`` uses Bernoulli source selection plus geometric
    component counts. ``"poisson"`` draws a Poisson component count for every
    parent source.

    Clustering offsets use
    ``r = cluster_r_cut_arcsec + Exponential(scale=cluster_r0_arcsec)``
    in arcseconds, with a random position angle ``phi ~ Uniform(0, 2pi)``.

    If ``mask_map`` is provided, it must be a 1D HEALPix mask matching
    ``nside`` in NEST ordering, with ``1`` for kept pixels and ``0`` for
    masked pixels.

    Conditional error-grid dimensions and physical bounds default from the
    selected product. Passing explicit dimensions or bound pairs overrides the
    product values; passing ``None`` for a bound makes that axis unbounded.
    """
    flux_min: float
    product: RacsProductSpec | str = RACS_LOW3
    nside: int = 64
    chunk_size: int = 50_000
    use_float32: bool = False
    downscale_nside: Optional[int] = None
    store_final_samples: bool = True
    catalogue_path: Optional[str] = None
    mask_map: Optional[NDArray[np.generic]] = None
    flux_hist_bins: int = 200
    alpha_mean: float = 0.8
    alpha_sigma: float = 0.2
    cluster_count_model: Literal['geometric', 'poisson'] = "geometric"
    max_cluster_children_per_parent: int = 16
    cluster_r0_arcsec: float = 100.0
    cluster_r_cut_arcsec: float = 20.0
    noisemap_data_dir: Optional[str] = None
    noise_map_nside: int = 256
    flux_error_noise_bins: Optional[int] = None
    flux_error_flux_bins: Optional[int] = None
    flux_error_min_cell_count: int = 10
    flux_error_noise_bounds_ujy_beam: (
        tuple[float, float] | None | Literal["product_default"]
    ) = "product_default"
    flux_error_flux_bounds_mjy: (
        tuple[float, float] | None | Literal["product_default"]
    ) = "product_default"
    flux_temperature_min_mjy: Optional[float] = None
    paf_temperature_data_dir: Optional[str] = None
    paf_reference_temp_c: float = 25.0
    temperature_model: TemperatureModel = "hot_linear"
    paf_max_interpolation_gap_minutes: float = 20.0
    temperature_fallback: Literal["none", "open_meteo", "reference"] = "none"
    max_reference_fallback_tiles: Optional[int] = None
    open_meteo_cache_dir: Optional[str] = None
    open_meteo_timeout_seconds: float = 60.0
    open_meteo_latitude_deg: float = ASKAP_LATITUDE_DEG
    open_meteo_longitude_deg: float = ASKAP_LONGITUDE_DEG

    def __post_init__(self) -> None:
        self.product = resolve_racs_product(self.product)
        if self.flux_error_noise_bins is None:
            self.flux_error_noise_bins = self.product.default_flux_error_noise_bins
        if self.flux_error_flux_bins is None:
            self.flux_error_flux_bins = self.product.default_flux_error_flux_bins
        if (
            isinstance(self.flux_error_noise_bounds_ujy_beam, str)
            and self.flux_error_noise_bounds_ujy_beam == "product_default"
        ):
            self.flux_error_noise_bounds_ujy_beam = (
                self.product.default_flux_error_noise_bounds_ujy_beam
            )
        if (
            isinstance(self.flux_error_flux_bounds_mjy, str)
            and self.flux_error_flux_bounds_mjy == "product_default"
        ):
            self.flux_error_flux_bounds_mjy = (
                self.product.default_flux_error_flux_bounds_mjy
            )
        if self.flux_min <= 0:
            raise ValueError("flux_min must be positive.")
        if (
            self.mask_map is None
            and self.product.default_mask_filename is not None
            and self.nside != 64
        ):
            raise ValueError(
                f"{self.product.label} currently requires nside=64 when using "
                "the packaged mask."
            )
        if self.mask_map is not None:
            if self.mask_map.ndim != 1:
                raise ValueError("mask_map must be a 1D HEALPix array.")
            expected_shape = (hp.nside2npix(self.nside),)
            if self.mask_map.shape != expected_shape:
                raise ValueError(
                    "mask_map has unexpected shape: "
                    f"{self.mask_map.shape}, expected {expected_shape}"
                )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if self.flux_hist_bins < 2:
            raise ValueError("flux_hist_bins must be at least 2.")
        if self.alpha_sigma <= 0:
            raise ValueError("alpha_sigma must be positive.")
        if self.cluster_count_model not in {"geometric", "poisson"}:
            raise ValueError(
                "cluster_count_model must be either 'geometric' or 'poisson'."
            )
        if self.max_cluster_children_per_parent < 0:
            raise ValueError("max_cluster_children_per_parent must be non-negative.")
        if self.cluster_r0_arcsec <= 0:
            raise ValueError("cluster_r0_arcsec must be positive.")
        if self.cluster_r_cut_arcsec < 0:
            raise ValueError("cluster_r_cut_arcsec must be non-negative.")
        if (
            isinstance(self.noise_map_nside, bool)
            or not isinstance(self.noise_map_nside, (int, np.integer))
            or self.noise_map_nside <= 0
            or (int(self.noise_map_nside) & (int(self.noise_map_nside) - 1)) != 0
        ):
            raise ValueError("noise_map_nside must be a positive power of two.")
        for field_name, value, minimum in (
            ("flux_error_noise_bins", self.flux_error_noise_bins, 2),
            ("flux_error_flux_bins", self.flux_error_flux_bins, 2),
            ("flux_error_min_cell_count", self.flux_error_min_cell_count, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{field_name} must be an integer of at least {minimum}.")
            if value < minimum:
                raise ValueError(f"{field_name} must be at least {minimum}.")
        for field_name, bounds in (
            (
                "flux_error_noise_bounds_ujy_beam",
                self.flux_error_noise_bounds_ujy_beam,
            ),
            ("flux_error_flux_bounds_mjy", self.flux_error_flux_bounds_mjy),
        ):
            if bounds is None:
                continue
            if (
                not isinstance(bounds, (tuple, list))
                or len(bounds) != 2
                or not np.all(np.isfinite(bounds))
                or float(bounds[0]) <= 0
                or float(bounds[1]) <= float(bounds[0])
            ):
                raise ValueError(
                    f"{field_name} must be None or two finite positive values "
                    "in strictly increasing order."
                )
            # Canonical tuples keep cache metadata and filenames deterministic.
            setattr(self, field_name, (float(bounds[0]), float(bounds[1])))
        if self.flux_temperature_min_mjy is not None and (
            not np.isfinite(self.flux_temperature_min_mjy)
            or self.flux_temperature_min_mjy <= 0
        ):
            raise ValueError("flux_temperature_min_mjy must be positive and finite.")
        if not np.isfinite(self.paf_reference_temp_c):
            raise ValueError("paf_reference_temp_c must be finite.")
        if self.temperature_model not in TEMPERATURE_MODELS:
            raise ValueError(
                "temperature_model must be either 'hot_linear' or 'hot_quadratic'."
            )
        if self.paf_max_interpolation_gap_minutes <= 0:
            raise ValueError("paf_max_interpolation_gap_minutes must be positive.")
        if self.temperature_fallback not in {"none", "open_meteo", "reference"}:
            raise ValueError(
                "temperature_fallback must be 'none', 'open_meteo', or 'reference'."
            )
        if self.temperature_fallback == "reference":
            if (
                isinstance(self.max_reference_fallback_tiles, bool)
                or not isinstance(self.max_reference_fallback_tiles, int)
                or self.max_reference_fallback_tiles <= 0
            ):
                raise ValueError(
                    "max_reference_fallback_tiles must be explicitly set to a "
                    "positive integer when temperature_fallback='reference'."
                )
        if self.open_meteo_timeout_seconds <= 0:
            raise ValueError("open_meteo_timeout_seconds must be positive.")
        if not np.isfinite(self.open_meteo_latitude_deg):
            raise ValueError("open_meteo_latitude_deg must be finite.")
        if not np.isfinite(self.open_meteo_longitude_deg):
            raise ValueError("open_meteo_longitude_deg must be finite.")
        if self.downscale_nside is not None:
            if self.downscale_nside > self.nside:
                raise ValueError("downscale_nside must be <= nside.")
            ratio = self.nside // self.downscale_nside
            if (self.nside % self.downscale_nside) != 0 or (ratio & (ratio - 1)) != 0:
                raise ValueError(
                    "downscale_nside must be a power-of-two divisor of nside."
                )


RacsLow3Config = RacsConfig


class Racs:
    """RACS simulator following the Catwise initialise/simulate split."""

    def __init__(self, config: RacsConfig):
        self.cfg = config
        self.product = config.product
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
        self.elevation_map: Optional[NDArray[np.float32]] = None
        self.sampled_flux_error_map: Optional[NDArray[np.float32]] = None
        self.sampled_fractional_error_map: Optional[NDArray[np.float32]] = None
        # All-sky source counts at the simulator nside. These diagnostics are
        # intentionally not survey-masked so exterior no-map coverage remains
        # visible; the second map isolates sources otherwise in the footprint.
        self.invalid_noise_rejection_map: Optional[NDArray[np.float32]] = None
        self.invalid_noise_rejection_in_footprint_map: Optional[
            NDArray[np.float32]
        ] = None
        self.invalid_noise_rejection_count = 0
        self.invalid_noise_rejection_in_footprint_count = 0

        self.final_intrinsic_flux_samples: Optional[NDArray[np.float32]] = None
        self.final_observed_flux_samples: Optional[NDArray[np.float32]] = None
        self.final_alpha_samples: Optional[NDArray[np.float32]] = None
        self.final_base_flux_error_samples: Optional[NDArray[np.float32]] = None
        self.final_flux_error_samples: Optional[NDArray[np.float32]] = None
        self.final_fractional_error_samples: Optional[NDArray[np.float32]] = None
        self.final_base_fractional_error_samples: Optional[NDArray[np.float32]] = None
        self.final_pixel_indices: Optional[NDArray[np.int32]] = None
        self.final_tile_indices: Optional[NDArray[np.int32]] = None
        self.final_longitudes: Optional[NDArray[np.float32]] = None
        self.final_latitudes: Optional[NDArray[np.float32]] = None
        self.final_temperature_samples: Optional[NDArray[np.float32]] = None
        self.final_elevation_samples: Optional[NDArray[np.float32]] = None

    def _cache_dir(self) -> Path:
        return Path(__file__).resolve().parent / "data" / self.product.data_dir_name / "lookups"

    def _sbid_lookup_cache_path(self) -> Path:
        return self._cache_dir() / f"sbid_lookup_nside{self.nside}.npz"

    def _sbid_mixture_lookup_cache_path(self) -> Path:
        return self._cache_dir() / f"sbid_mixture_lookup_nside{self.nside}.npz"

    def _flux_distribution_cache_path(self) -> Path:
        return self._cache_dir() / f"flux_distribution_bins{self.cfg.flux_hist_bins}.npz"

    def _tile_metadata_cache_path(self) -> Path:
        return self._cache_dir() / "tile_metadata.npz"

    def _temperature_lookup_cache_path(self) -> Path:
        return self._cache_dir() / f"temperature_lookup_nside{self.nside}.npz"

    def _elevation_lookup_cache_path(self) -> Path:
        return self._cache_dir() / f"elevation_lookup_nside{self.nside}.npz"

    def _noise_map_cache_path(self) -> Path:
        return self._cache_dir() / (
            f"noise_map_nside{self.cfg.noise_map_nside}_nested_v1.npz"
        )

    def _absolute_error_lookup_cache_path(self) -> Path:
        noise_bounds = self.cfg.flux_error_noise_bounds_ujy_beam
        flux_bounds = self.cfg.flux_error_flux_bounds_mjy
        if noise_bounds is None and flux_bounds is None:
            # Explicitly unbounded user overrides retain the established v1
            # contract. Both production product defaults use bounded v2 caches.
            return self._cache_dir() / (
                "absolute_error_lookup_"
                f"noise{self.cfg.noise_map_nside}_"
                f"grid{self.cfg.flux_error_noise_bins}x{self.cfg.flux_error_flux_bins}_"
                f"min{self.cfg.flux_error_min_cell_count}_v1.npz"
            )

        def cache_number(value: float) -> str:
            return format(value, ".12g").replace(".", "p").replace("-", "m")

        noise_tag = (
            "unbounded"
            if noise_bounds is None
            else f"{cache_number(noise_bounds[0])}to{cache_number(noise_bounds[1])}"
        )
        flux_tag = (
            "unbounded"
            if flux_bounds is None
            else f"{cache_number(flux_bounds[0])}to{cache_number(flux_bounds[1])}"
        )
        return self._cache_dir() / (
            "absolute_error_lookup_"
            f"noise{self.cfg.noise_map_nside}_"
            f"grid{self.cfg.flux_error_noise_bins}x{self.cfg.flux_error_flux_bins}_"
            f"min{self.cfg.flux_error_min_cell_count}_"
            f"bounds-noise{noise_tag}_flux{flux_tag}_v2.npz"
        )

    def _source_noisemap_path(self) -> Path:
        filename = self.product.source_noisemap_filename
        if filename is None:
            raise ValueError(
                f"{self.product.label} does not support the noise/flux error lookup."
            )
        if self.cfg.noisemap_data_dir is None:
            raise FileNotFoundError(
                "RacsConfig.noisemap_data_dir is required to build a missing "
                f"{self.product.label} noise-map cache."
            )
        directory = Path(self.cfg.noisemap_data_dir).expanduser()
        source = directory / filename
        if not source.is_file():
            raise FileNotFoundError(f"RACS source noisemap does not exist: {source}")
        return source

    def build_cached_noise_map(self) -> None:
        """Build the configured NESTED noise-map cache in memory."""
        self.noise_map_cache = build_noise_map_cache(
            self._source_noisemap_path(),
            product_key=self.product.key,
            target_nside=self.cfg.noise_map_nside,
        )
        self.noise_map = self.noise_map_cache.values

    def save_cached_noise_map(self, *, diagnostics: bool = True) -> None:
        if not hasattr(self, "noise_map_cache"):
            raise RuntimeError("Build or load the cached noise map before saving it.")
        save_noise_map_cache(
            self.noise_map_cache,
            self._noise_map_cache_path(),
            diagnostics=diagnostics,
        )

    def load_cached_noise_map(self) -> bool:
        """Load a compatible cached map without consulting its external source."""
        filename = self.product.source_noisemap_filename
        if filename is None:
            raise ValueError(
                f"{self.product.label} does not support the noise/flux error lookup."
            )
        path = self._noise_map_cache_path()
        if not path.exists():
            return False
        try:
            cache = load_noise_map_cache(
                path,
                product_key=self.product.key,
                target_nside=self.cfg.noise_map_nside,
                source_filename=filename,
            )
        except RacsCacheValidationError as exc:
            LOGGER.warning("Ignoring incompatible noise-map cache %s: %s", path, exc)
            return False
        self.noise_map_cache = cache
        self.noise_map = cache.values
        return True

    def query_local_noise(
        self,
        ra_deg: NDArray[np.floating],
        dec_deg: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        if not hasattr(self, "noise_map_cache"):
            raise RuntimeError("Build or load the cached noise map before querying it.")
        return self.noise_map_cache.query(ra_deg, dec_deg)

    def build_absolute_error_lookup(self) -> None:
        """Build the catalogue noise/flux-conditioned absolute-error grid."""
        if not self.catalogue_is_loaded:
            raise RuntimeError("Load the catalogue before building the error lookup.")
        if not hasattr(self, "noise_map_cache"):
            raise RuntimeError("Build or load the cached noise map before the error lookup.")
        columns = self.product.columns
        ra = np.asarray(self.catalogue[columns.ra], dtype=np.float64)
        dec = np.asarray(self.catalogue[columns.dec], dtype=np.float64)
        noise = self.query_local_noise(ra, dec)
        flux = np.asarray(self.catalogue[columns.total_flux], dtype=np.float64)
        error = np.asarray(self.catalogue[columns.total_flux_error], dtype=np.float64)
        self.absolute_error_lookup = build_conditional_error_lookup(
            noise,
            flux,
            error,
            product_key=self.product.key,
            noise_map_identity=self.noise_map_cache.identity,
            noise_bins=self.cfg.flux_error_noise_bins,
            flux_bins=self.cfg.flux_error_flux_bins,
            min_cell_count=self.cfg.flux_error_min_cell_count,
            noise_bounds_ujy_beam=self.cfg.flux_error_noise_bounds_ujy_beam,
            flux_bounds_mjy=self.cfg.flux_error_flux_bounds_mjy,
            catalogue_columns={
                "ra": columns.ra,
                "dec": columns.dec,
                "total_flux": columns.total_flux,
                "total_flux_error": columns.total_flux_error,
            },
        )

    def save_absolute_error_lookup(self, *, diagnostics: bool = True) -> None:
        if not hasattr(self, "absolute_error_lookup"):
            raise RuntimeError("Build or load the absolute-error lookup before saving it.")
        save_conditional_error_lookup(
            self.absolute_error_lookup,
            self._absolute_error_lookup_cache_path(),
            diagnostics=diagnostics,
        )

    def load_absolute_error_lookup(self) -> bool:
        if not hasattr(self, "noise_map_cache"):
            raise RuntimeError("Load the cached noise map before the error lookup.")
        path = self._absolute_error_lookup_cache_path()
        if not path.exists():
            return False
        try:
            lookup = load_conditional_error_lookup(
                path,
                product_key=self.product.key,
                noise_map_identity=self.noise_map_cache.identity,
                noise_bins=self.cfg.flux_error_noise_bins,
                flux_bins=self.cfg.flux_error_flux_bins,
                min_cell_count=self.cfg.flux_error_min_cell_count,
                noise_bounds_ujy_beam=self.cfg.flux_error_noise_bounds_ujy_beam,
                flux_bounds_mjy=self.cfg.flux_error_flux_bounds_mjy,
                catalogue_columns={
                    "ra": self.product.columns.ra,
                    "dec": self.product.columns.dec,
                    "total_flux": self.product.columns.total_flux,
                    "total_flux_error": self.product.columns.total_flux_error,
                },
            )
        except RacsCacheValidationError as exc:
            LOGGER.warning("Ignoring incompatible absolute-error cache %s: %s", path, exc)
            return False
        self.absolute_error_lookup = lookup
        return True

    def sample_absolute_flux_errors(
        self,
        noise_ujy_beam: NDArray[np.floating],
        pre_noise_flux_mjy: NDArray[np.floating],
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.float32]:
        if not hasattr(self, "absolute_error_lookup"):
            raise RuntimeError("Build or load the absolute-error lookup before sampling it.")
        return self.absolute_error_lookup.sample(
            noise_ujy_beam,
            pre_noise_flux_mjy,
            rng=rng,
        )

    def _save_lookup_map_png(
        self,
        map_values: NDArray[np.floating],
        output_path: Path,
        title: str,
        unit: str = "",
        cmap: str = "viridis",
        **kwargs
    ) -> None:
        fig = plt.figure(figsize=(10, 6))
        hp.projview(
            np.asarray(map_values, dtype=np.float64),
            nest=True,
            fig=fig.number,
            title=title,
            unit=unit,
            cmap=cmap,
            hold=True,
            **kwargs
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _mask_map_path(self) -> Path:
        if self.product.default_mask_filename is None:
            raise FileNotFoundError(
                f"No packaged mask is configured for {self.product.label}."
            )
        return (
            Path(__file__).resolve().parent
            / "data"
            / self.product.data_dir_name
            / self.product.default_mask_filename
        )

    def load_catalogue(self) -> None:
        """Load the real RACS catalogue from a configured path or dipole-utils."""
        if self.cfg.catalogue_path is not None:
            catalogue_path = Path(self.cfg.catalogue_path).expanduser()
            if not catalogue_path.exists():
                raise FileNotFoundError(
                    f"RacsConfig.catalogue_path points to missing file: {catalogue_path}"
                )
            self.catalogue = Table.read(catalogue_path, unit_parse_strict="silent")
        else:
            self.catalogue = DataLoader(*self.product.data_loader_args).load()

        self.catalogue_is_loaded = True

    def release_catalogue(self) -> None:
        """Drop the in-memory catalogue once lookup tables have been derived."""
        if hasattr(self, "catalogue"):
            del self.catalogue
        self.catalogue_is_loaded = False

    def __getstate__(self) -> dict:
        """Exclude the raw catalogue from pickle payloads."""
        state = self.__dict__.copy()
        state.pop("catalogue", None)
        state["catalogue_is_loaded"] = False
        return state

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

        flux = np.asarray(self.catalogue[self.product.columns.total_flux], dtype=np.float64)
        flux = flux[np.isfinite(flux) & (flux > 0)]
        if flux.size == 0:
            raise ValueError(
                f"No positive finite {self.product.columns.total_flux} values available."
            )

        log_flux = np.log10(flux)
        counts, edges = np.histogram(log_flux, bins=self.cfg.flux_hist_bins)
        if not np.any(counts > 0):
            raise ValueError("Flux histogram contains no populated bins.")

        probabilities = counts.astype(np.float64)
        probabilities /= probabilities.sum()

        self.log_flux_bin_edges = edges.astype(np.float64, copy=False)
        self.log_flux_bin_probabilities = probabilities
        self.log_flux_bin_cdf = np.cumsum(probabilities)

    def save_flux_distribution(self) -> None:
        """Persist the empirical log-flux histogram."""
        cache_path = self._flux_distribution_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            flux_hist_bins=np.asarray(self.cfg.flux_hist_bins, dtype=np.int64),
            log_flux_bin_edges=self.log_flux_bin_edges.astype(np.float64, copy=False),
            log_flux_bin_probabilities=self.log_flux_bin_probabilities.astype(
                np.float64,
                copy=False,
            ),
            log_flux_bin_cdf=self.log_flux_bin_cdf.astype(np.float64, copy=False),
        )

    def load_flux_distribution(self) -> bool:
        """Load the cached empirical log-flux histogram if available."""
        cache_path = self._flux_distribution_cache_path()
        if not cache_path.exists():
            return False

        with np.load(cache_path) as data:
            cache_bins = int(data["flux_hist_bins"])
            if cache_bins != self.cfg.flux_hist_bins:
                return False
            self.log_flux_bin_edges = data["log_flux_bin_edges"].astype(
                np.float64,
                copy=False,
            )
            self.log_flux_bin_probabilities = data["log_flux_bin_probabilities"].astype(
                np.float64,
                copy=False,
            )
            self.log_flux_bin_cdf = data["log_flux_bin_cdf"].astype(np.float64, copy=False)

        return True

    def build_tile_lookup(self) -> None:
        """Build per-pixel dominant-SBID and SBID-mixture lookup products."""
        assert self.catalogue_is_loaded, "Load the catalogue before building tile lookups."

        ra = np.asarray(self.catalogue[self.product.columns.ra], dtype=np.float64)
        dec = np.asarray(self.catalogue[self.product.columns.dec], dtype=np.float64)
        sbid = np.asarray(self.catalogue[self.product.columns.tile_id], dtype=np.int64)

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
        mixture_counts = np.zeros(n_pix, dtype=np.int64)
        mixture_starts = np.zeros(n_pix, dtype=np.int64)
        mixture_tile_indices: list[NDArray[np.int32]] = []
        mixture_sbid_probabilities: list[NDArray[np.float64]] = []
        mixture_offset = 0

        for pix, start, count in zip(unique_pixels, starts, counts):
            sbid_values = sbid_sorted[start:start + count]
            sbid_unique, sbid_counts = np.unique(sbid_values, return_counts=True)
            self.tile_lookup_map[pix] = int(sbid_unique[np.argmax(sbid_counts)])
            mixture_starts[pix] = mixture_offset
            mixture_counts[pix] = sbid_unique.size
            mixture_tile_indices.append(
                np.asarray(
                    [self._tile_index_from_sbid[int(tile_sbid)] for tile_sbid in sbid_unique],
                    dtype=np.int32,
                )
            )
            mixture_sbid_probabilities.append(
                (sbid_counts.astype(np.float64) / sbid_counts.sum()).astype(
                    np.float64,
                    copy=False,
                )
            )
            mixture_offset += sbid_unique.size

        self.mask_map = self.tile_lookup_map >= 0
        if mixture_tile_indices:
            self.sbid_mixture_counts = mixture_counts
            self.sbid_mixture_starts = mixture_starts
            self.sbid_mixture_tile_indices = np.concatenate(mixture_tile_indices).astype(
                np.int32,
                copy=False,
            )
            self.sbid_mixture_probabilities = np.concatenate(
                mixture_sbid_probabilities
            ).astype(np.float64, copy=False)
        else:
            self.sbid_mixture_counts = np.zeros(n_pix, dtype=np.int64)
            self.sbid_mixture_starts = np.zeros(n_pix, dtype=np.int64)
            self.sbid_mixture_tile_indices = np.empty(0, dtype=np.int32)
            self.sbid_mixture_probabilities = np.empty(0, dtype=np.float64)

    def save_tile_lookup(self) -> None:
        """Persist the HEALPix dominant-SBID lookup derived from the uncut catalogue."""
        cache_path = self._sbid_lookup_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            nside=np.asarray(self.nside, dtype=np.int64),
            tile_lookup_map=self.tile_lookup_map.astype(np.int32, copy=False),
        )
        tile_lookup_map = self.tile_lookup_map.astype(np.float64, copy=False)
        tile_lookup_map = np.where(tile_lookup_map >= 0, tile_lookup_map, np.nan)
        self._save_lookup_map_png(
            tile_lookup_map,
            cache_path.with_suffix(".png"),
            title=f"{self.product.label} Dominant SBID Lookup (nside={self.nside})",
            unit="SBID",
            cmap="viridis",
        )

    def save_sbid_mixture_lookup(self) -> None:
        """Persist the per-pixel SBID mixture lookup used during simulation."""
        cache_path = self._sbid_mixture_lookup_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            nside=np.asarray(self.nside, dtype=np.int64),
            tile_sbids=self.tile_sbids.astype(np.int32, copy=False),
            counts=self.sbid_mixture_counts.astype(np.int64, copy=False),
            starts=self.sbid_mixture_starts.astype(np.int64, copy=False),
            tile_indices=self.sbid_mixture_tile_indices.astype(np.int32, copy=False),
            probabilities=self.sbid_mixture_probabilities.astype(np.float64, copy=False),
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

    def load_sbid_mixture_lookup(self) -> bool:
        """Load the cached per-pixel SBID mixture lookup if it matches the tile metadata."""
        cache_path = self._sbid_mixture_lookup_cache_path()
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
            if "tile_indices" not in data.files:
                return False
            self.sbid_mixture_counts = data["counts"].astype(np.int64, copy=False)
            self.sbid_mixture_starts = data["starts"].astype(np.int64, copy=False)
            self.sbid_mixture_tile_indices = data["tile_indices"].astype(np.int32, copy=False)
            self.sbid_mixture_probabilities = data["probabilities"].astype(
                np.float64,
                copy=False,
            )

        return True

    def load_mask_map(self) -> None:
        """Load a custom mask, packaged product mask, or empirical footprint mask."""
        expected_shape = (hp.nside2npix(self.nside),)

        if self.cfg.mask_map is not None:
            mask_map = np.asarray(self.cfg.mask_map)
            if mask_map.shape != expected_shape:
                raise ValueError(
                    f"Custom {self.product.label} mask has unexpected shape: "
                    f"{mask_map.shape}, expected {expected_shape}"
                )
        elif self.product.default_mask_filename is not None:
            mask_path = self._mask_map_path()
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Packaged {self.product.label} mask not found: {mask_path}"
                )

            mask_map_ring = np.load(mask_path, allow_pickle=False)
            mask_map = hp.reorder(mask_map_ring, r2n=True)
            if mask_map.shape != expected_shape:
                raise ValueError(
                    f"Packaged {self.product.label} mask has unexpected shape: "
                    f"{mask_map.shape}, expected {expected_shape}"
                )
        elif hasattr(self, "tile_lookup_map"):
            mask_map = self.tile_lookup_map >= 0
        else:
            raise FileNotFoundError(
                f"No mask_map or packaged mask is available for {self.product.label}; "
                "run initialise_data() so the empirical tile footprint can be used."
            )

        self.mask_map = np.asarray(mask_map == 1, dtype=np.bool_)

    def build_tile_metadata(self) -> None:
        """Collect one row of metadata per SBID for later tile-level systematics."""
        assert self.catalogue_is_loaded, "Load the catalogue before building tile metadata."

        sbid = np.asarray(self.catalogue[self.product.columns.tile_id], dtype=np.int64)
        field_id = np.asarray(self.catalogue[self.product.columns.field_id])
        scan_start_mjd = np.asarray(
            self.catalogue[self.product.columns.scan_start_mjd],
            dtype=np.float64,
        )
        scan_length = np.asarray(
            self.catalogue[self.product.columns.scan_length],
            dtype=np.float64,
        )

        unique_sbid, first_indices = np.unique(sbid, return_index=True)
        self.tile_sbids = unique_sbid.astype(np.int32, copy=False)
        self.tile_scan_start_mjd = scan_start_mjd[first_indices].astype(np.float64, copy=False)
        self.tile_scan_length = scan_length[first_indices].astype(np.float64, copy=False)
        self.tile_field_id = field_id[first_indices]
        self._tile_index_from_sbid = {
            int(tile_sbid): int(tile_index)
            for tile_index, tile_sbid in enumerate(self.tile_sbids)
        }

    def save_tile_metadata(self) -> None:
        """Persist one-row-per-SBID tile metadata."""
        cache_path = self._tile_metadata_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            tile_sbids=self.tile_sbids.astype(np.int32, copy=False),
            tile_scan_start_mjd=self.tile_scan_start_mjd.astype(np.float64, copy=False),
            tile_scan_length=self.tile_scan_length.astype(np.float64, copy=False),
            tile_field_id=np.asarray(self.tile_field_id),
        )

    def load_tile_metadata(self) -> bool:
        """Load cached one-row-per-SBID tile metadata if available."""
        cache_path = self._tile_metadata_cache_path()
        if not cache_path.exists():
            return False

        with np.load(cache_path, allow_pickle=False) as data:
            self.tile_sbids = data["tile_sbids"].astype(np.int32, copy=False)
            self.tile_scan_start_mjd = data["tile_scan_start_mjd"].astype(
                np.float64,
                copy=False,
            )
            self.tile_scan_length = data["tile_scan_length"].astype(np.float64, copy=False)
            self.tile_field_id = data["tile_field_id"]

        self._tile_index_from_sbid = {
            int(tile_sbid): int(tile_index)
            for tile_index, tile_sbid in enumerate(self.tile_sbids)
        }
        return True

    def build_temperature_map(self) -> None:
        """Project the mixture-mean tile temperatures onto the HEALPix survey footprint."""
        n_pix = hp.nside2npix(self.nside)
        temperature_map = np.full(n_pix, np.nan, dtype=np.float32)

        if self.tile_temperature_by_index is None:
            self.temperature_map = temperature_map
            return

        assert hasattr(self, "sbid_mixture_counts"), "Run initialise_data() first."
        valid_pixels = self.sbid_mixture_counts > 0
        if np.any(valid_pixels):
            for pix in np.flatnonzero(valid_pixels):
                start = self.sbid_mixture_starts[pix]
                count = self.sbid_mixture_counts[pix]
                pixel_tile_indices = self.sbid_mixture_tile_indices[start:start + count]
                pixel_probabilities = self.sbid_mixture_probabilities[start:start + count]
                pixel_temperatures = self.tile_temperature_by_index[pixel_tile_indices]
                finite = np.isfinite(pixel_temperatures)
                if np.any(finite):
                    probs = pixel_probabilities[finite]
                    probs /= probs.sum()
                    temperature_map[pix] = float(
                        np.sum(pixel_temperatures[finite] * probs, dtype=np.float64)
                    )

        self.temperature_map = temperature_map

    def save_temperature_lookup(
        self,
        tile_temperature_sources: NDArray[np.str_] | None = None,
    ) -> None:
        """Persist the per-tile and per-pixel temperature lookup."""
        if self.tile_temperature_by_index is None:
            return

        cache_path = self._temperature_lookup_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if tile_temperature_sources is None:
            tile_temperature_sources = np.full(
                self.tile_temperature_by_index.shape, "mean_paf", dtype="<U10"
            )
        per_tile_sources = np.asarray(tile_temperature_sources, dtype=np.str_)
        if per_tile_sources.shape != self.tile_temperature_by_index.shape:
            raise ValueError(
                "tile_temperature_sources must match tile_temperature_by_index."
            )
        allowed_sources = self._allowed_temperature_sources()
        if not np.all(np.isin(per_tile_sources, np.asarray(allowed_sources))):
            raise ValueError(
                "tile_temperature_sources contains values incompatible with "
                f"temperature_fallback={self.cfg.temperature_fallback!r}."
            )

        paf_data_dir = self._configured_paf_temperature_data_dir_for_cache()
        payload = {
            "format_version": np.asarray(TEMPERATURE_CACHE_FORMAT_VERSION, dtype=np.int64),
            "nside": np.asarray(self.nside, dtype=np.int64),
            "product_key": np.asarray(self.product.key),
            "temperature_fallback": np.asarray(self.cfg.temperature_fallback),
            "tile_sbids": self.tile_sbids.astype(np.int32, copy=False),
            "tile_scan_start_mjd": self.tile_scan_start_mjd.astype(np.float64, copy=False),
            "tile_temperature_by_index": self.tile_temperature_by_index.astype(
                np.float64,
                copy=False,
            ),
            "tile_temperature_sources": per_tile_sources,
            "paf_temperature_data_dir": np.asarray(paf_data_dir),
            "paf_max_interpolation_gap_minutes": np.asarray(
                self.cfg.paf_max_interpolation_gap_minutes, dtype=np.float64
            ),
            "paf_reference_temp_c": np.asarray(
                self.cfg.paf_reference_temp_c, dtype=np.float64
            ),
            "max_reference_fallback_tiles": np.asarray(
                -1 if self.cfg.max_reference_fallback_tiles is None
                else self.cfg.max_reference_fallback_tiles,
                dtype=np.int64,
            ),
            "open_meteo_latitude_deg": np.asarray(
                self.cfg.open_meteo_latitude_deg, dtype=np.float64
            ),
            "open_meteo_longitude_deg": np.asarray(
                self.cfg.open_meteo_longitude_deg, dtype=np.float64
            ),
            "open_meteo_timeout_seconds": np.asarray(
                self.cfg.open_meteo_timeout_seconds, dtype=np.float64
            ),
            "open_meteo_cache_dir": np.asarray(
                "" if self.cfg.open_meteo_cache_dir is None
                else str(Path(self.cfg.open_meteo_cache_dir).expanduser().resolve())
            ),
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.stem}.",
                suffix=".npz",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_path, **payload)
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        if self.temperature_map is not None:
            unique_sources = set(per_tile_sources.tolist())
            source_label = " + ".join(
                label
                for source, label in (
                    ("mean_paf", "Mean PAF"),
                    ("open_meteo", "Open-Meteo"),
                    ("reference", "Reference"),
                )
                if source in unique_sources
            )
            self._save_lookup_map_png(
                self.temperature_map,
                cache_path.with_suffix(".png"),
                title=(
                    f"{self.product.label} {source_label} Temperature Lookup "
                    f"(nside={self.nside})"
                ),
                unit="deg C",
                cmap="coolwarm",
            )
            for coord, coord_str in zip([['C'], ['C', 'G']], ['eq', 'gal']):
                self._save_lookup_map_png(
                    self.temperature_map,
                    self._cache_dir() / f"temperature_lookup_nside{self.nside}_{coord_str}.png",
                    title=(
                        f"{self.product.label} {source_label} Temperature Lookup "
                        f"(nside={self.nside})"
                    ),
                    unit="deg C",
                    cmap="coolwarm",
                    coord=coord
                )

    def load_temperature_lookup(self, *, allow_complete_open_meteo: bool = True) -> bool:
        """Load a cached per-tile temperature lookup if it matches the tile metadata."""
        cache_path = self._temperature_lookup_cache_path()
        if not cache_path.exists():
            return False

        with np.load(cache_path) as data:
            required_fields = {
                "format_version", "nside", "product_key", "temperature_fallback",
                "tile_sbids", "tile_scan_start_mjd", "tile_temperature_by_index",
                "tile_temperature_sources", "paf_temperature_data_dir",
                "paf_max_interpolation_gap_minutes", "paf_reference_temp_c",
                "max_reference_fallback_tiles", "open_meteo_latitude_deg",
                "open_meteo_longitude_deg", "open_meteo_timeout_seconds",
                "open_meteo_cache_dir",
            }
            if not required_fields.issubset(data.files):
                return False
            if int(data["format_version"]) != TEMPERATURE_CACHE_FORMAT_VERSION:
                return False
            if int(data["nside"]) != self.nside or str(data["product_key"]) != self.product.key:
                return False
            if str(data["temperature_fallback"]) != self.cfg.temperature_fallback:
                return False

            cache_tile_sbids = data["tile_sbids"].astype(np.int32, copy=False)
            cached_scan_start_mjd = data["tile_scan_start_mjd"].astype(np.float64, copy=False)
            if not np.array_equal(cache_tile_sbids, self.tile_sbids):
                return False
            if not np.array_equal(cached_scan_start_mjd, self.tile_scan_start_mjd):
                return False
            if not self._temperature_lookup_cache_matches_config(data):
                return False
            cached_temperatures = data["tile_temperature_by_index"]
            cached_sources = data["tile_temperature_sources"]
            if cached_temperatures.shape != cache_tile_sbids.shape:
                return False
            if cached_sources.shape != cached_temperatures.shape:
                return False
            if not np.all(np.isin(cached_sources, np.asarray(self._allowed_temperature_sources()))):
                return False
            if not allow_complete_open_meteo and np.all(cached_sources == "open_meteo"):
                return False
            if np.any(cached_sources != "open_meteo"):
                try:
                    self._resolve_paf_temperature_data_dir()
                except FileNotFoundError:
                    return False
            if self.cfg.temperature_fallback == "reference":
                reference_mask = cached_sources == "reference"
                assert self.cfg.max_reference_fallback_tiles is not None
                if np.count_nonzero(reference_mask) > self.cfg.max_reference_fallback_tiles:
                    return False
                if np.all(reference_mask):
                    return False
                if not np.allclose(
                    cached_temperatures[reference_mask],
                    self.cfg.paf_reference_temp_c,
                ):
                    return False

            self.tile_temperature_by_index = cached_temperatures.astype(
                np.float64,
                copy=False,
            )

        self._validate_tile_temperatures(
            source=self._temperature_source_for_validation(cached_sources)
        )
        fallback_mask = cached_sources != "mean_paf"
        if np.any(fallback_mask):
            fallback_sbids = self.tile_sbids[fallback_mask]
            preview = ", ".join(str(int(sbid)) for sbid in fallback_sbids[:10])
            if fallback_sbids.size > 10:
                preview += ", ..."
            if np.any(cached_sources == "reference"):
                LOGGER.warning(
                    "%s using cached temperatures with reference fallback %.3f C "
                    "for %d SBID(s): %s",
                    self.product.label,
                    self.cfg.paf_reference_temp_c,
                    fallback_sbids.size,
                    preview,
                )
            else:
                LOGGER.warning(
                    "%s using cached temperatures with Open-Meteo fallback for "
                    "%d SBID(s): %s",
                    self.product.label,
                    fallback_sbids.size,
                    preview,
                )
        self.build_temperature_map()
        return True

    def _allowed_temperature_sources(self) -> tuple[str, ...]:
        if self.cfg.temperature_fallback == "open_meteo":
            return ("mean_paf", "open_meteo")
        if self.cfg.temperature_fallback == "reference":
            return ("mean_paf", "reference")
        return ("mean_paf",)

    @staticmethod
    def _temperature_source_for_validation(sources: NDArray[np.str_]) -> TemperatureSource:
        if np.any(sources == "open_meteo"):
            return "open_meteo"
        if np.any(sources == "reference"):
            return "reference"
        return "mean_paf"

    def build_elevation_lookup(self) -> None:
        """Build a per-pixel empirical lookup of source elevations in degrees."""
        assert self.catalogue_is_loaded, "Load the catalogue before building elevation lookups."
        if self.product.columns.elevation is None:
            raise ValueError(
                f"{self.product.label} does not define an elevation column; "
                "source-elevation systematics require catalogue ALT data."
            )
        if self.product.columns.elevation not in self.catalogue.colnames:
            raise ValueError(
                f"{self.product.label} catalogue is missing elevation column "
                f"{self.product.columns.elevation!r}."
            )

        ra = np.asarray(self.catalogue[self.product.columns.ra], dtype=np.float64)
        dec = np.asarray(self.catalogue[self.product.columns.dec], dtype=np.float64)
        elevation = np.asarray(
            self.catalogue[self.product.columns.elevation],
            dtype=np.float64,
        )

        valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(elevation)
        if not np.any(valid):
            raise ValueError("No valid source elevations available to build elevation lookup.")

        pixel_indices = hp.ang2pix(
            self.nside,
            ra[valid],
            dec[valid],
            lonlat=True,
            nest=True,
        ).astype(np.int64, copy=False)
        elevation_values = elevation[valid].astype(np.float32, copy=False)

        order = np.argsort(pixel_indices, kind="stable")
        pix_sorted = pixel_indices[order]
        elevation_sorted = elevation_values[order]

        n_pix = hp.nside2npix(self.nside)
        counts = np.bincount(pix_sorted, minlength=n_pix).astype(np.int64)
        starts = np.cumsum(counts, dtype=np.int64) - counts

        self.elevation_lookup_pixel_counts = counts
        self.elevation_lookup_pixel_starts = starts
        self.elevation_lookup_values = elevation_sorted

        elevation_map = np.full(n_pix, np.nan, dtype=np.float32)
        populated = counts > 0
        if np.any(populated):
            for pix in np.flatnonzero(populated):
                start = starts[pix]
                count = counts[pix]
                elevation_map[pix] = np.median(elevation_sorted[start:start + count])
        self.elevation_map = elevation_map

    def save_elevation_lookup(self) -> None:
        """Persist the per-pixel source-elevation lookup."""
        cache_path = self._elevation_lookup_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            nside=np.asarray(self.nside, dtype=np.int64),
            elevation_column=np.asarray(self.product.columns.elevation or ""),
            counts=self.elevation_lookup_pixel_counts.astype(np.int64, copy=False),
            starts=self.elevation_lookup_pixel_starts.astype(np.int64, copy=False),
            elevation=self.elevation_lookup_values.astype(np.float32, copy=False),
        )
        if self.elevation_map is not None:
            self._save_lookup_map_png(
                self.elevation_map,
                cache_path.with_suffix(".png"),
                title=f"{self.product.label} Elevation Lookup (nside={self.nside})",
                unit="deg",
                cmap="viridis",
            )
            for coord, coord_str in zip([['C'], ['C', 'G']], ['eq', 'gal']):
                self._save_lookup_map_png(
                    self.elevation_map,
                    self._cache_dir() / f"elevation_lookup_{coord_str}.png",
                    title=f"{self.product.label} Elevation Lookup (nside={self.nside})",
                    unit="deg",
                    cmap="viridis",
                    coord=coord,
                )

    def load_elevation_lookup(self) -> bool:
        """Load the cached per-pixel elevation lookup if available."""
        if self.product.columns.elevation is None:
            raise ValueError(
                f"{self.product.label} does not define an elevation column; "
                "source-elevation systematics require catalogue ALT data."
            )

        cache_path = self._elevation_lookup_cache_path()
        if not cache_path.exists():
            return False

        with np.load(cache_path) as data:
            cache_nside = int(data["nside"])
            if cache_nside != self.nside:
                return False
            if "elevation_column" in data.files:
                cache_column = str(data["elevation_column"])
                if cache_column != self.product.columns.elevation:
                    return False
            self.elevation_lookup_pixel_counts = data["counts"].astype(np.int64, copy=False)
            self.elevation_lookup_pixel_starts = data["starts"].astype(np.int64, copy=False)
            self.elevation_lookup_values = data["elevation"].astype(np.float32, copy=False)

        if self.elevation_lookup_values.size == 0:
            raise ValueError("Cached elevation lookup contains no elevation samples.")

        n_pix = hp.nside2npix(self.nside)
        elevation_map = np.full(n_pix, np.nan, dtype=np.float32)
        populated = self.elevation_lookup_pixel_counts > 0
        if np.any(populated):
            for pix in np.flatnonzero(populated):
                start = self.elevation_lookup_pixel_starts[pix]
                count = self.elevation_lookup_pixel_counts[pix]
                elevation_map[pix] = np.median(
                    self.elevation_lookup_values[start:start + count]
                )
        self.elevation_map = elevation_map
        return True

    def _configured_paf_temperature_data_dir_for_cache(self) -> str:
        """Return a stable representation of the configured PAF source."""
        try:
            return str(self._resolve_paf_temperature_data_dir())
        except FileNotFoundError:
            if self.cfg.paf_temperature_data_dir is None:
                return ""
            return str(Path(self.cfg.paf_temperature_data_dir).expanduser().resolve())

    def _temperature_lookup_cache_matches_config(
        self,
        data: np.lib.npyio.NpzFile,
    ) -> bool:
        """Return whether cached temperature inputs match the current config."""
        if str(data["paf_temperature_data_dir"]) != (
            self._configured_paf_temperature_data_dir_for_cache()
        ):
            return False
        if not np.isclose(
            float(data["paf_max_interpolation_gap_minutes"]),
            self.cfg.paf_max_interpolation_gap_minutes,
        ):
            return False
        if self.cfg.temperature_fallback == "reference":
            if not np.isclose(
                float(data["paf_reference_temp_c"]), self.cfg.paf_reference_temp_c
            ):
                return False
            assert self.cfg.max_reference_fallback_tiles is not None
            if (
                int(data["max_reference_fallback_tiles"])
                != self.cfg.max_reference_fallback_tiles
            ):
                return False
        if self.cfg.temperature_fallback == "open_meteo":
            if not np.isclose(
                float(data["open_meteo_latitude_deg"]),
                self.cfg.open_meteo_latitude_deg,
            ):
                return False
            if not np.isclose(
                float(data["open_meteo_longitude_deg"]),
                self.cfg.open_meteo_longitude_deg,
            ):
                return False
            if not np.isclose(
                float(data["open_meteo_timeout_seconds"]),
                self.cfg.open_meteo_timeout_seconds,
            ):
                return False
            expected_cache_dir = (
                "" if self.cfg.open_meteo_cache_dir is None
                else str(Path(self.cfg.open_meteo_cache_dir).expanduser().resolve())
            )
            if str(data["open_meteo_cache_dir"]) != expected_cache_dir:
                return False
        return True

    def sample_elevations(
        self,
        pixel_indices: NDArray[np.int_],
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.float32]:
        """Sample source elevations in degrees from each pixel's empirical distribution."""
        assert hasattr(self, "elevation_lookup_pixel_counts"), "Run initialise_data() first."
        if rng is None:
            rng = np.random.default_rng()

        pix = np.asarray(pixel_indices, dtype=np.int64)
        counts = self.elevation_lookup_pixel_counts[pix]
        starts = self.elevation_lookup_pixel_starts[pix]

        out = np.empty(pix.shape[0], dtype=np.float32)
        valid = counts > 0
        if np.any(valid):
            rand_offsets = rng.integers(0, counts[valid], dtype=np.int64)
            pick = starts[valid] + rand_offsets
            out[valid] = self.elevation_lookup_values[pick]

        if np.any(~valid):
            if self.elevation_lookup_values.size == 0:
                raise ValueError("Elevation lookup contains no global fallback samples.")
            pick = rng.integers(
                0,
                self.elevation_lookup_values.size,
                size=np.count_nonzero(~valid),
                dtype=np.int64,
            )
            out[~valid] = self.elevation_lookup_values[pick]

        return out

    def evaluate_elevation_enhancement(
        self,
        elevations_deg: NDArray[np.floating],
        elevation_amp: float,
        elevation_trough: float,
    ) -> NDArray[np.floating]:
        """Evaluate the source-elevation flux enhancement for degree-valued angles."""
        if not np.isfinite(elevation_amp) or elevation_amp < 0:
            raise ValueError("elevation_amp must be finite and non-negative.")
        if not np.isfinite(elevation_trough):
            raise ValueError("elevation_trough must be finite.")

        elevations = np.asarray(elevations_deg, dtype=np.float64)
        delta_rad = np.deg2rad(elevations - elevation_trough)
        enhancement = 1.0 + elevation_amp * (1.0 - np.cos(delta_rad))
        return enhancement.astype(self.dtype, copy=False)

    def scale_absolute_flux_error(
        self,
        base_flux_error: NDArray[np.floating],
        fractional_error_eta: float = 0.0,
        dtype: type = np.float64,
    ) -> NDArray[np.floating]:
        """Apply eta variance scaling to sampled absolute error sigmas.

        ``base_flux_error`` is already an absolute total-flux error in mJy.
        It must not be multiplied by source flux.
        """
        if fractional_error_eta < 0:
            raise ValueError("fractional_error_eta must be non-negative.")

        sigma = np.asarray(base_flux_error, dtype=np.float64)
        sigma = sigma * np.sqrt(1.0 + fractional_error_eta)
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

        if self.load_temperature_lookup(allow_complete_open_meteo=False):
            return

        paf_failure: Exception | None = None
        try:
            paf_data_dir = self._resolve_paf_temperature_data_dir()
            self.tile_temperature_by_index = np.asarray(
                get_mean_paf_temperatures_for_observations(
                    self.tile_scan_start_mjd,
                    data_dir=paf_data_dir,
                    max_interpolation_gap_minutes=self.cfg.paf_max_interpolation_gap_minutes,
                ),
                dtype=np.float64,
            )
        except Exception as exc:
            paf_failure = exc
            self.tile_temperature_by_index = None
            if self.cfg.temperature_fallback in {"none", "reference"}:
                raise
        else:
            if self.tile_temperature_by_index.shape != self.tile_sbids.shape:
                raise ValueError(
                    "PAF temperature lookup shape does not match the tile SBID list."
                )
            invalid_paf = ~np.isfinite(self.tile_temperature_by_index)
            if np.any(invalid_paf):
                invalid_sbids = self.tile_sbids[invalid_paf]
                if self.cfg.temperature_fallback == "none":
                    self._validate_tile_temperatures(source="mean_paf")
                if np.all(invalid_paf):
                    paf_failure = ValueError(
                        "PAF lookup contains no finite temperatures."
                    )
                    self.tile_temperature_by_index = None
                    if self.cfg.temperature_fallback == "reference":
                        raise paf_failure
                else:
                    tile_temperature_sources = np.full(
                        self.tile_temperature_by_index.shape,
                        "mean_paf",
                        dtype="<U10",
                    )
                    sbid_preview = ", ".join(
                        str(int(sbid)) for sbid in invalid_sbids[:10]
                    )
                    if invalid_sbids.size > 10:
                        sbid_preview += ", ..."
                    if self.cfg.temperature_fallback == "reference":
                        assert self.cfg.max_reference_fallback_tiles is not None
                        if invalid_sbids.size > self.cfg.max_reference_fallback_tiles:
                            raise ValueError(
                                f"{self.product.label} PAF temperature lookup is missing "
                                f"{invalid_sbids.size} SBID(s), exceeding "
                                "max_reference_fallback_tiles="
                                f"{self.cfg.max_reference_fallback_tiles}: {sbid_preview}."
                            )
                        LOGGER.warning(
                            "%s PAF temperature lookup is missing %d SBID(s); "
                            "explicitly filling only those with reference temperature "
                            "%.3f C: %s",
                            self.product.label,
                            invalid_sbids.size,
                            self.cfg.paf_reference_temp_c,
                            sbid_preview,
                        )
                        self.tile_temperature_by_index[invalid_paf] = (
                            self.cfg.paf_reference_temp_c
                        )
                        tile_temperature_sources[invalid_paf] = "reference"
                        validation_source: TemperatureSource = "reference"
                    else:
                        LOGGER.warning(
                            "%s PAF temperature lookup is missing %d SBID(s); filling "
                            "only those temperatures from Open-Meteo: %s",
                            self.product.label,
                            invalid_sbids.size,
                            sbid_preview,
                        )
                        open_meteo_temperatures = np.asarray(
                            get_open_meteo_temperatures_for_mjd(
                                self.tile_scan_start_mjd[invalid_paf],
                                latitude_deg=self.cfg.open_meteo_latitude_deg,
                                longitude_deg=self.cfg.open_meteo_longitude_deg,
                                timeout=self.cfg.open_meteo_timeout_seconds,
                                cache_dir=self.cfg.open_meteo_cache_dir,
                            ),
                            dtype=np.float64,
                        )
                        self.tile_temperature_by_index[invalid_paf] = open_meteo_temperatures
                        tile_temperature_sources[invalid_paf] = "open_meteo"
                        validation_source = "open_meteo"
                    self._validate_tile_temperatures(source=validation_source)
                    self.build_temperature_map()
                    self.save_temperature_lookup(
                        tile_temperature_sources=tile_temperature_sources,
                    )
                    return
            else:
                self._validate_tile_temperatures(source="mean_paf")
                self.build_temperature_map()
                self.save_temperature_lookup()
                return

        if (
            self.cfg.temperature_fallback == "open_meteo"
            and self.load_temperature_lookup(allow_complete_open_meteo=True)
        ):
            LOGGER.warning(
                "%s PAF temperature lookup failed; using cached Open-Meteo "
                "ambient temperature fallback lookup: %s",
                self.product.label,
                paf_failure,
            )
            return

        LOGGER.warning(
            "%s PAF temperature lookup failed; falling back to Open-Meteo "
            "ambient temperatures for %d SBID(s) (%s): %s",
            self.product.label,
            self.tile_sbids.size,
            ", ".join(str(int(sbid)) for sbid in self.tile_sbids[:10])
            + (", ..." if self.tile_sbids.size > 10 else ""),
            paf_failure,
        )
        self.tile_temperature_by_index = np.asarray(
            get_open_meteo_temperatures_for_mjd(
                self.tile_scan_start_mjd,
                latitude_deg=self.cfg.open_meteo_latitude_deg,
                longitude_deg=self.cfg.open_meteo_longitude_deg,
                timeout=self.cfg.open_meteo_timeout_seconds,
                cache_dir=self.cfg.open_meteo_cache_dir,
            ),
            dtype=np.float64,
        )
        self._validate_tile_temperatures(source="open_meteo")
        self.build_temperature_map()
        self.save_temperature_lookup(
            tile_temperature_sources=np.full(
                self.tile_temperature_by_index.shape,
                "open_meteo",
                dtype="<U10",
            )
        )

    def _validate_tile_temperatures(
        self,
        source: TemperatureSource = "mean_paf",
    ) -> None:
        """Fail initialisation if any SBID has no finite PAF temperature."""
        if self.tile_temperature_by_index is None:
            return

        temperatures = np.asarray(self.tile_temperature_by_index, dtype=np.float64)
        invalid = ~np.isfinite(temperatures)
        if not np.any(invalid):
            return

        invalid_indices = np.flatnonzero(invalid)
        invalid_sbids = self.tile_sbids[invalid_indices]
        preview = ", ".join(str(int(sbid)) for sbid in invalid_sbids[:10])
        if invalid_sbids.size > 10:
            preview += ", ..."
        source_label = {
            "mean_paf": "PAF",
            "open_meteo": "Open-Meteo",
            "reference": "hybrid PAF and reference",
        }[source]
        raise ValueError(
            f"{self.product.label} {source_label} temperature lookup contains non-finite "
            f"temperatures for {invalid_sbids.size} SBID(s): {preview}."
        )

    def _resolve_paf_temperature_data_dir(self) -> Path:
        if self.cfg.paf_temperature_data_dir is not None:
            root_dir = Path(self.cfg.paf_temperature_data_dir).expanduser().resolve()
            if not root_dir.exists():
                raise FileNotFoundError(
                    "RacsConfig.paf_temperature_data_dir points to missing directory: "
                    f"{root_dir}"
                )
            return self._select_product_paf_temperature_data_dir(root_dir)

        repo_root = Path(__file__).resolve().parents[2]
        default_root = repo_root.parent / "dipole-utils" / "data" / "paf_temps"
        if default_root.exists():
            return self._select_product_paf_temperature_data_dir(default_root)

        raise FileNotFoundError(
            "Could not find PAF temperature data. Set "
            "RacsConfig.paf_temperature_data_dir or provide "
            f"{default_root}."
        )

    def _select_product_paf_temperature_data_dir(self, root_dir: Path) -> Path:
        product_dir = root_dir / self.product.key
        if product_dir.is_dir():
            LOGGER.info(
                "%s loading PAF temperatures from product directory: %s",
                self.product.label,
                product_dir,
            )
            return product_dir

        if any(root_dir.glob("ak*csv")):
            LOGGER.warning(
                "%s loading PAF temperatures from legacy flat directory: %s",
                self.product.label,
                root_dir,
            )
            return root_dir

        raise FileNotFoundError(
            f"Could not find PAF temperature files for {self.product.label}. "
            f"Expected product directory {product_dir} containing ak*csv files, "
            f"or a legacy flat directory {root_dir} containing ak*csv files."
        )

    def initialise_data(self) -> None:
        """Initialise the catalogue-derived lookup tables used during simulation."""
        # The external source noisemap is needed only when its compact runtime
        # cache is absent or incompatible. Load/build it before deciding
        # whether the catalogue-conditioned absolute-error cache is usable.
        if not self.load_cached_noise_map():
            self.build_cached_noise_map()
            self.save_cached_noise_map()

        need_absolute_error_lookup = not self.load_absolute_error_lookup()
        need_flux_distribution = not self.load_flux_distribution()
        need_tile_metadata = not self.load_tile_metadata()
        need_tile_lookup = not self.load_tile_lookup()
        need_sbid_mixture_lookup = False
        need_elevation_lookup = (
            self.product.columns.elevation is not None
            and not self.load_elevation_lookup()
        )

        if not need_tile_metadata:
            need_sbid_mixture_lookup = not self.load_sbid_mixture_lookup()

        if (
            need_flux_distribution
            or need_tile_metadata
            or need_tile_lookup
            or need_sbid_mixture_lookup
            or need_absolute_error_lookup
            or need_elevation_lookup
        ):
            if not self.catalogue_is_loaded:
                self.load_catalogue()

            try:
                if need_flux_distribution:
                    self.build_flux_distribution()
                    self.save_flux_distribution()
                if need_tile_metadata:
                    self.build_tile_metadata()
                    self.save_tile_metadata()
                if need_tile_lookup:
                    self.build_tile_lookup()
                    self.save_tile_lookup()
                    self.save_sbid_mixture_lookup()
                elif need_sbid_mixture_lookup:
                    self.build_tile_lookup()
                    self.save_sbid_mixture_lookup()
                if need_absolute_error_lookup:
                    self.build_absolute_error_lookup()
                    self.save_absolute_error_lookup()
                if need_elevation_lookup:
                    self.build_elevation_lookup()
                    self.save_elevation_lookup()
            finally:
                self.release_catalogue()

        self.load_mask_map()
        self.load_temperature_table()
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

    def sample_clustered_points(
        self,
        parent_ra_deg: NDArray[np.floating],
        parent_dec_deg: NDArray[np.floating],
        per_parent_n_components: NDArray[np.integer],
        rng: Optional[np.random.Generator] = None,
        dtype: type = np.float64,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Sample clustered component positions around parent sources.

        For each added component, draw ``phi ~ Uniform(0, 2pi)`` and
        ``r = cluster_r_cut_arcsec + Exponential(scale=cluster_r0_arcsec)``,
        then offset the parent source by ``(r, phi)`` on the sphere.
        """
        if rng is None:
            rng = np.random.default_rng()

        counts = np.asarray(per_parent_n_components, dtype=np.int64)
        total_n_components = int(counts.sum())
        if total_n_components == 0:
            empty = np.empty(0, dtype=dtype)
            return empty, empty

        parent_indices = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        parent_ra_rad = np.deg2rad(np.asarray(parent_ra_deg, dtype=np.float64)[parent_indices])
        parent_dec_rad = np.deg2rad(np.asarray(parent_dec_deg, dtype=np.float64)[parent_indices])

        phi = rng.uniform(0.0, 2.0 * np.pi, size=total_n_components)
        radial_arcsec = self.cfg.cluster_r_cut_arcsec + rng.exponential(
            scale=self.cfg.cluster_r0_arcsec,
            size=total_n_components,
        )
        angular_distance_rad = np.deg2rad(radial_arcsec / 3600.0)

        sin_parent_dec = np.sin(parent_dec_rad)
        cos_parent_dec = np.cos(parent_dec_rad)
        sin_distance = np.sin(angular_distance_rad)
        cos_distance = np.cos(angular_distance_rad)

        child_dec_rad = np.arcsin(
            sin_parent_dec * cos_distance
            + cos_parent_dec * sin_distance * np.cos(phi)
        )
        child_ra_rad = parent_ra_rad + np.arctan2(
            np.sin(phi) * sin_distance * cos_parent_dec,
            cos_distance - sin_parent_dec * np.sin(child_dec_rad),
        )

        child_ra_deg = np.mod(np.rad2deg(child_ra_rad), 360.0)
        child_dec_deg = np.rad2deg(child_dec_rad)
        return child_ra_deg.astype(dtype, copy=False), child_dec_deg.astype(dtype, copy=False)

    def sample_spectral_indices(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
        alpha_mean: Optional[float] = None,
        alpha_sigma: Optional[float] = None,
    ) -> NDArray[np.float32]:
        """Draw per-source radio spectral indices from a Gaussian model.

        Per-simulation parameters default to the values in ``RacsConfig``.
        """
        if rng is None:
            rng = np.random.default_rng()

        active_alpha_mean = self.cfg.alpha_mean if alpha_mean is None else alpha_mean
        active_alpha_sigma = self.cfg.alpha_sigma if alpha_sigma is None else alpha_sigma

        alpha = rng.normal(
            loc=active_alpha_mean,
            scale=active_alpha_sigma,
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

    def sample_tiles_for_pixels(
        self,
        pixel_indices: NDArray[np.int_],
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.int32]:
        """Sample a tile assignment from each pixel's empirical SBID mixture."""
        assert hasattr(self, "sbid_mixture_counts"), "Run initialise_data() first."
        if rng is None:
            rng = np.random.default_rng()

        pix = np.asarray(pixel_indices, dtype=np.int64)
        out = np.full(pix.shape[0], -1, dtype=np.int32)
        valid_pixels = self.sbid_mixture_counts[pix] > 0
        if not np.any(valid_pixels):
            return out

        valid_positions = np.flatnonzero(valid_pixels)
        valid_pix = pix[valid_positions]
        order = np.argsort(valid_pix, kind="stable")
        valid_positions_sorted = valid_positions[order]
        valid_pix_sorted = valid_pix[order]
        unique_pix, starts, counts = np.unique(
            valid_pix_sorted,
            return_index=True,
            return_counts=True,
        )
        for pixel, start_idx, count in zip(unique_pix, starts, counts):
            pixel_output_positions = valid_positions_sorted[start_idx:start_idx + count]
            start = self.sbid_mixture_starts[pixel]
            mixture_count = self.sbid_mixture_counts[pixel]
            pixel_tile_indices = self.sbid_mixture_tile_indices[start:start + mixture_count]
            pixel_probabilities = self.sbid_mixture_probabilities[start:start + mixture_count]
            if mixture_count == 1:
                out[pixel_output_positions] = pixel_tile_indices[0]
                continue
            if mixture_count == 2:
                draws = rng.random(count)
                sampled_tile_indices = np.where(
                    draws < pixel_probabilities[0],
                    pixel_tile_indices[0],
                    pixel_tile_indices[1],
                )
                out[pixel_output_positions] = sampled_tile_indices.astype(
                    np.int32,
                    copy=False,
                )
                continue
            sampled_tile_indices = rng.choice(
                pixel_tile_indices,
                size=count,
                p=pixel_probabilities,
            )
            out[pixel_output_positions] = sampled_tile_indices.astype(np.int32, copy=False)

        return out

    def evaluate_temperature_enhancement(
        self,
        tile_indices: NDArray[np.int32],
        temp_beta: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.float32]]:
        """Evaluate hot-PAF flux suppression at the tile level.

        Temperatures at or below ``cfg.paf_reference_temp_c`` have no flux
        correction. Hotter observations use the response selected by
        ``cfg.temperature_model``.
        """
        if not np.isfinite(temp_beta) or temp_beta < 0:
            raise ValueError("temp_beta must be finite and non-negative.")

        enhancement = np.ones(tile_indices.shape, dtype=np.float64)
        temperatures = np.full(tile_indices.shape, np.nan, dtype=np.float32)

        if self.tile_temperature_by_index is None:
            enhancement = np.maximum(enhancement, RACS_TEMPERATURE_EPSILON_FLOOR)
            return enhancement.astype(self.dtype, copy=False), temperatures

        valid = tile_indices >= 0
        if np.any(valid):
            tile_temperatures = self.tile_temperature_by_index[tile_indices[valid]]
            temperatures[valid] = tile_temperatures.astype(np.float32, copy=False)
            valid_temperature = np.isfinite(tile_temperatures)
            if np.any(valid_temperature):
                enhancement_valid = evaluate_temperature_response(
                    tile_temperatures[valid_temperature],
                    temp_beta,
                    self.cfg.paf_reference_temp_c,
                    model=self.cfg.temperature_model,
                    xp=np,
                )
                enhancement_indices = np.flatnonzero(valid)[valid_temperature]
                enhancement[enhancement_indices] = enhancement_valid

        enhancement = np.maximum(enhancement, RACS_TEMPERATURE_EPSILON_FLOOR)
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
        p_clus: float = 0.0,
        clus_stop_prob: float = 1.0,
        lambda_clus: float = 0.0,
        observer_speed: float = 1.0,
        dipole_longitude: float = CMB_L,
        dipole_latitude: float = CMB_B,
        temp_beta: float = 0.0,
        elevation_amp: float = 0.0,
        elevation_trough: float = 0.0,
        fractional_error_eta: float = 0.0,
        rng_key: Optional[NPKey] = None,
        alpha_mean: Optional[float] = None,
        alpha_sigma: Optional[float] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        output_map, output_mask, _ = self._generate_dipole_impl(
            log10_n_initial_samples=log10_n_initial_samples,
            flux_min=flux_min,
            p_clus=p_clus,
            clus_stop_prob=clus_stop_prob,
            lambda_clus=lambda_clus,
            observer_speed=observer_speed,
            dipole_longitude=dipole_longitude,
            dipole_latitude=dipole_latitude,
            temp_beta=temp_beta,
            elevation_amp=elevation_amp,
            elevation_trough=elevation_trough,
            fractional_error_eta=fractional_error_eta,
            alpha_mean=alpha_mean,
            alpha_sigma=alpha_sigma,
            rng_key=rng_key,
        )
        return output_map, output_mask

    def generate_dipole_with_flux_summaries(
        self,
        *args,
        temperature_edges: Optional[NDArray[np.floating]] = None,
        temperature_quantiles: Optional[tuple[float, ...] | NDArray[np.floating]] = None,
        elevation_edges: Optional[NDArray[np.floating]] = None,
        elevation_quantiles: Optional[tuple[float, ...] | NDArray[np.floating]] = None,
        **kwargs,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool_],
        dict[str, NDArray[np.float32]],
    ]:
        """Generate a map plus exact source-level flux summaries."""
        return self._generate_dipole_impl(
            *args,
            temperature_edges=temperature_edges,
            temperature_quantiles=temperature_quantiles,
            elevation_edges=elevation_edges,
            elevation_quantiles=elevation_quantiles,
            **kwargs,
        )

    def _generate_dipole_impl(
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
        elevation_amp: float = 0.0,
        elevation_trough: float = 0.0,
        fractional_error_eta: float = 0.0,
        rng_key: Optional[NPKey] = None,
        alpha_mean: Optional[float] = None,
        alpha_sigma: Optional[float] = None,
        *,
        temperature_edges: Optional[NDArray[np.floating]] = None,
        temperature_quantiles: Optional[tuple[float, ...] | NDArray[np.floating]] = None,
        elevation_edges: Optional[NDArray[np.floating]] = None,
        elevation_quantiles: Optional[tuple[float, ...] | NDArray[np.floating]] = None,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool_],
        dict[str, NDArray[np.float32]],
    ]:
        """Coordinate the CatSIM-like simulation pipeline for RACS.

        ``log10_n_initial_samples`` sets the expected total number of
        pre-selection sources after clustering. The simulator derives the
        parent-anchor count by dividing by the selected model's expected
        multiplicity.

        The clustering count model is selected by
        ``RacsConfig.cluster_count_model``. ``"geometric"`` draws
        ``X ~ Bernoulli(p_clus)`` for each parent and, if ``X = 1``, draws
        ``K ~ Geometric(clus_stop_prob)`` on support ``1, 2, 3, ...``.
        ``"poisson"`` draws ``K ~ Poisson(lambda_clus)`` for every parent.
        The parent source is retained in both models, and the ``K`` added
        components are given clustered positions, independent fluxes, and
        independent spectral indices before entering the ordinary downstream
        simulation pipeline.
        """
        assert self.lookups_are_initialised, (
            "Lookup tables must be initialised before generating maps. "
            "Run initialise_data() first."
        )

        if not np.isfinite(elevation_amp) or elevation_amp < 0:
            raise ValueError("elevation_amp must be finite and non-negative.")
        if not np.isfinite(elevation_trough):
            raise ValueError("elevation_trough must be finite.")
        active_alpha_mean = self.cfg.alpha_mean if alpha_mean is None else alpha_mean
        active_alpha_sigma = self.cfg.alpha_sigma if alpha_sigma is None else alpha_sigma
        if not np.isfinite(active_alpha_mean):
            raise ValueError("alpha_mean must be finite.")
        if not np.isfinite(active_alpha_sigma) or active_alpha_sigma <= 0:
            raise ValueError("alpha_sigma must be finite and positive.")
        include_temperature_summary = (
            temperature_edges is not None or temperature_quantiles is not None
        )
        include_elevation_summary = (
            elevation_edges is not None or elevation_quantiles is not None
        )
        if (temperature_edges is None) != (temperature_quantiles is None):
            raise ValueError(
                "temperature_edges and temperature_quantiles must be provided together."
            )
        if (elevation_edges is None) != (elevation_quantiles is None):
            raise ValueError(
                "elevation_edges and elevation_quantiles must be provided together."
            )
        if include_temperature_summary:
            temperature_edges = validate_bin_edges(temperature_edges, "temperature_edges")
            temperature_quantiles = validate_quantiles(temperature_quantiles)
        if include_elevation_summary:
            elevation_edges = validate_bin_edges(elevation_edges, "elevation_edges")
            elevation_quantiles = validate_quantiles(elevation_quantiles)

        use_elevation = elevation_amp > 0.0
        elevation_is_available = self.product.columns.elevation is not None
        if use_elevation and not elevation_is_available:
            raise ValueError(
                f"{self.product.label} does not define an elevation column; "
                "source-elevation systematics require catalogue ALT data."
            )
        sample_elevation = elevation_is_available and (
            use_elevation or self.store_final_samples or include_elevation_summary
        )
        if include_elevation_summary and not elevation_is_available:
            raise ValueError(
                f"{self.product.label} does not define an elevation column; "
                "flux-elevation summaries require catalogue ALT data."
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

        n_expected_sources = int(10 ** log10_n_initial_samples)
        if n_expected_sources < 0:
            raise ValueError("n_initial_samples must be non-negative.")
        if self.cfg.cluster_count_model == "geometric":
            if p_clus < 0 or p_clus > 1:
                raise ValueError(
                    "For cluster_count_model='geometric', p_clus must lie in [0, 1]."
                )
            if clus_stop_prob <= 0 or clus_stop_prob > 1:
                raise ValueError(
                    "For cluster_count_model='geometric', "
                    "clus_stop_prob must lie in (0, 1]."
                )
            if lambda_clus != 0:
                raise ValueError(
                    "lambda_clus is only valid for cluster_count_model='poisson'."
                )
            expected_multiplicity = 1.0 + p_clus / clus_stop_prob
        elif self.cfg.cluster_count_model == "poisson":
            if lambda_clus < 0:
                raise ValueError(
                    "For cluster_count_model='poisson', lambda_clus must be non-negative."
                )
            if p_clus != 0:
                raise ValueError(
                    "p_clus is only valid for cluster_count_model='geometric'."
                )
            if clus_stop_prob != 1.0:
                raise ValueError(
                    "clus_stop_prob is only valid for cluster_count_model='geometric'."
                )
            expected_multiplicity = 1.0 + lambda_clus

        n_samples = int(n_expected_sources / expected_multiplicity)

        active_flux_min = self.cfg.flux_min if flux_min is None else flux_min
        active_temperature_flux_min = (
            active_flux_min
            if self.cfg.flux_temperature_min_mjy is None
            else self.cfg.flux_temperature_min_mjy
        )
        rng = rng_key._generator() if rng_key is not None else np.random.default_rng()

        n_pix = hp.nside2npix(self.nside)
        density_accumulator = np.zeros(n_pix, dtype=np.float64)
        flux_error_sum = np.zeros(n_pix, dtype=np.float64)
        fractional_error_sum = np.zeros(n_pix, dtype=np.float64)
        error_sample_count = np.zeros(n_pix, dtype=np.int64)
        invalid_noise_accumulator = np.zeros(n_pix, dtype=np.float64)
        invalid_noise_in_footprint_accumulator = np.zeros(n_pix, dtype=np.float64)
        noise_query_count = 0
        invalid_noise_count = 0
        invalid_noise_in_footprint_count = 0

        final_intrinsic_flux: list[NDArray[np.float32]] = []
        final_observed_flux: list[NDArray[np.float32]] = []
        final_alpha: list[NDArray[np.float32]] = []
        final_base_flux_error: list[NDArray[np.float32]] = []
        final_flux_error: list[NDArray[np.float32]] = []
        final_base_fractional_error: list[NDArray[np.float32]] = []
        final_fractional_error: list[NDArray[np.float32]] = []
        final_pixels: list[NDArray[np.int32]] = []
        final_tiles: list[NDArray[np.int32]] = []
        final_ra: list[NDArray[np.float32]] = []
        final_dec: list[NDArray[np.float32]] = []
        final_temperature: list[NDArray[np.float32]] = []
        final_elevation: list[NDArray[np.float32]] = []
        summary_temperature_flux: list[NDArray[np.float32]] = []
        summary_temperature: list[NDArray[np.float32]] = []
        summary_elevation_flux: list[NDArray[np.float32]] = []
        summary_elevation: list[NDArray[np.float32]] = []

        for start in range(0, n_samples, self.chunk_size):
            current_chunk = min(self.chunk_size, n_samples - start)

            intrinsic_flux = self.sample_fluxes(current_chunk, rng=rng)
            rest_ra_deg, rest_dec_deg = self.sample_points(
                current_chunk, dtype=self.dtype, rng=rng
            )
            alpha_kwargs = {}
            if alpha_mean is not None:
                alpha_kwargs["alpha_mean"] = active_alpha_mean
            if alpha_sigma is not None:
                alpha_kwargs["alpha_sigma"] = active_alpha_sigma
            alpha = self.sample_spectral_indices(
                current_chunk,
                rng=rng,
                **alpha_kwargs,
            )

            if (
                (self.cfg.cluster_count_model == "geometric" and p_clus > 0)
                or (self.cfg.cluster_count_model == "poisson" and lambda_clus > 0)
            ):
                if self.cfg.cluster_count_model == "geometric":
                    clustered_mask = rng.random(current_chunk) < p_clus
                    per_parent_n_components = np.zeros(current_chunk, dtype=np.int64)
                    n_clustered = int(np.count_nonzero(clustered_mask))
                    if n_clustered > 0:
                        per_parent_n_components[clustered_mask] = rng.geometric(
                            clus_stop_prob,
                            size=n_clustered,
                        ).astype(np.int64, copy=False)
                else:
                    per_parent_n_components = rng.poisson(
                        lambda_clus,
                        size=current_chunk,
                    ).astype(np.int64, copy=False)
                total_n_components = int(per_parent_n_components.sum())
                if total_n_components > 0:
                    cluster_ra_deg, cluster_dec_deg = self.sample_clustered_points(
                        rest_ra_deg,
                        rest_dec_deg,
                        per_parent_n_components,
                        rng=rng,
                        dtype=self.dtype,
                    )
                    cluster_flux = self.sample_fluxes(total_n_components, rng=rng)
                    cluster_alpha = self.sample_spectral_indices(
                        total_n_components,
                        rng=rng,
                        **alpha_kwargs,
                    )
                    intrinsic_flux = np.concatenate((intrinsic_flux, cluster_flux)).astype(
                        self.dtype,
                        copy=False,
                    )
                    rest_ra_deg = np.concatenate((rest_ra_deg, cluster_ra_deg)).astype(
                        self.dtype,
                        copy=False,
                    )
                    rest_dec_deg = np.concatenate((rest_dec_deg, cluster_dec_deg)).astype(
                        self.dtype,
                        copy=False,
                    )
                    alpha = np.concatenate((alpha, cluster_alpha)).astype(
                        np.float32,
                        copy=False,
                    )

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
            tile_indices = self.sample_tiles_for_pixels(pixel_indices, rng=rng)
            enhancement, temperatures = self.evaluate_temperature_enhancement(
                tile_indices=tile_indices,
                temp_beta=temp_beta,
            )
            systematics_flux = self.apply_temperature_enhancement(
                dipole_flux,
                enhancement,
                dtype=self.dtype,
            )
            elevations = None
            if sample_elevation:
                elevations = self.sample_elevations(pixel_indices, rng=rng)
            if use_elevation:
                assert elevations is not None
                elevation_enhancement = self.evaluate_elevation_enhancement(
                    elevations,
                    elevation_amp=elevation_amp,
                    elevation_trough=elevation_trough,
                )
                systematics_flux = (
                    np.asarray(systematics_flux, dtype=np.float64)
                    * np.asarray(elevation_enhancement, dtype=np.float64)
                ).astype(self.dtype, copy=False)

            # Local survey noise is queried at the aberrated source position.
            # Invalid-noise sources are removed from every downstream product;
            # they are not sent through the conditional lookup or Gaussian draw.
            local_noise = self.query_local_noise(boosted_ra_deg, boosted_dec_deg)
            valid_noise = np.isfinite(local_noise) & (local_noise > 0)
            noise_query_count += int(local_noise.size)
            invalid_noise_count += int(np.count_nonzero(~valid_noise))
            invalid_noise_in_footprint_count += int(
                np.count_nonzero(mask_slice & (tile_indices >= 0) & ~valid_noise)
            )
            if np.any(~valid_noise):
                np.add.at(
                    invalid_noise_accumulator,
                    pixel_indices[~valid_noise],
                    1,
                )
            invalid_in_footprint = (
                mask_slice & (tile_indices >= 0) & ~valid_noise
            )
            if np.any(invalid_in_footprint):
                np.add.at(
                    invalid_noise_in_footprint_accumulator,
                    pixel_indices[invalid_in_footprint],
                    1,
                )

            base_flux_error = np.full(systematics_flux.shape, np.nan, dtype=self.dtype)
            flux_error = np.full(systematics_flux.shape, np.nan, dtype=self.dtype)
            observed_flux = np.full(systematics_flux.shape, np.nan, dtype=self.dtype)
            if np.any(valid_noise):
                sampled_base_error = self.sample_absolute_flux_errors(
                    local_noise[valid_noise],
                    systematics_flux[valid_noise],
                    rng=rng,
                )
                base_flux_error[valid_noise] = sampled_base_error
                flux_error[valid_noise] = self.scale_absolute_flux_error(
                    sampled_base_error,
                    fractional_error_eta=fractional_error_eta,
                    dtype=self.dtype,
                )
                observed_flux[valid_noise] = self.add_flux_error(
                    systematics_flux[valid_noise],
                    flux_error[valid_noise],
                    rng=rng,
                    dtype=self.dtype,
                )

            safe_flux = np.maximum(
                np.asarray(systematics_flux, dtype=np.float64),
                np.finfo(np.float64).tiny,
            )
            base_fractional_error = (
                np.asarray(base_flux_error, dtype=np.float64) / safe_flux
            ).astype(np.float32, copy=False)
            fractional_error = (
                np.asarray(flux_error, dtype=np.float64) / safe_flux
            ).astype(np.float32, copy=False)

            base_keep = mask_slice & (tile_indices >= 0) & valid_noise
            if include_temperature_summary:
                temperature_keep = (
                    base_keep
                    & np.isfinite(temperatures)
                    & (observed_flux >= active_temperature_flux_min)
                )
                if np.any(temperature_keep):
                    summary_temperature_flux.append(
                        observed_flux[temperature_keep].astype(np.float32, copy=False)
                    )
                    summary_temperature.append(
                        temperatures[temperature_keep].astype(np.float32, copy=False)
                    )

            cut_slice = self.flux_cut_boolean(observed_flux, active_flux_min)
            keep = base_keep & cut_slice
            if include_elevation_summary and elevations is not None:
                elevation_keep = keep & np.isfinite(elevations)
                if np.any(elevation_keep):
                    summary_elevation_flux.append(
                        observed_flux[elevation_keep].astype(np.float32, copy=False)
                    )
                    summary_elevation.append(
                        elevations[elevation_keep].astype(np.float32, copy=False)
                    )
            if not np.any(keep):
                continue

            kept_pixels = pixel_indices[keep]
            np.add.at(density_accumulator, kept_pixels, 1)
            np.add.at(flux_error_sum, kept_pixels, flux_error[keep].astype(np.float64))
            np.add.at(fractional_error_sum, kept_pixels, fractional_error[keep].astype(np.float64))
            np.add.at(error_sample_count, kept_pixels, 1)

            if self.store_final_samples:
                final_intrinsic_flux.append(
                    intrinsic_flux[keep].astype(np.float32, copy=False)
                )
                final_observed_flux.append(
                    observed_flux[keep].astype(np.float32, copy=False)
                )
                final_alpha.append(alpha[keep].astype(np.float32, copy=False))
                final_base_flux_error.append(
                    base_flux_error[keep].astype(np.float32, copy=False)
                )
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
                if elevations is not None:
                    final_elevation.append(
                        elevations[keep].astype(np.float32, copy=False)
                    )

        if invalid_noise_in_footprint_count:
            LOGGER.warning(
                "%s removed %d source(s) with invalid local noise inside the "
                "survey/tile footprint (%d invalid of %d noise queries overall).",
                self.product.label,
                invalid_noise_in_footprint_count,
                invalid_noise_count,
                noise_query_count,
            )
        else:
            LOGGER.info(
                "%s noise-map coverage: %d valid of %d generated source queries.",
                self.product.label,
                noise_query_count - invalid_noise_count,
                noise_query_count,
            )

        self._density_map = density_accumulator.astype(np.float32, copy=False)
        self.invalid_noise_rejection_map = invalid_noise_accumulator.astype(
            np.float32,
            copy=False,
        )
        self.invalid_noise_rejection_in_footprint_map = (
            invalid_noise_in_footprint_accumulator.astype(np.float32, copy=False)
        )
        self.invalid_noise_rejection_count = invalid_noise_count
        self.invalid_noise_rejection_in_footprint_count = (
            invalid_noise_in_footprint_count
        )
        sampled_flux_error_map = np.full(n_pix, np.nan, dtype=np.float32)
        sampled_fractional_error_map = np.full(n_pix, np.nan, dtype=np.float32)
        valid_error_pixels = error_sample_count > 0
        if np.any(valid_error_pixels):
            sampled_flux_error_map[valid_error_pixels] = (
                flux_error_sum[valid_error_pixels]
                / error_sample_count[valid_error_pixels]
            ).astype(np.float32, copy=False)
            sampled_fractional_error_map[valid_error_pixels] = (
                fractional_error_sum[valid_error_pixels]
                / error_sample_count[valid_error_pixels]
            ).astype(np.float32, copy=False)
        outside_mask = ~self.mask_map.astype(bool)
        sampled_flux_error_map[outside_mask] = np.nan
        sampled_fractional_error_map[outside_mask] = np.nan
        self.sampled_flux_error_map = sampled_flux_error_map
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
            self.final_base_flux_error_samples = (
                np.concatenate(final_base_flux_error)
                if final_base_flux_error else np.empty(0, dtype=np.float32)
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
            if sample_elevation:
                self.final_elevation_samples = (
                    np.concatenate(final_elevation)
                    if final_elevation
                    else np.empty(0, dtype=np.float32)
                )
            else:
                self.final_elevation_samples = None
        else:
            self.final_intrinsic_flux_samples = None
            self.final_observed_flux_samples = None
            self.final_alpha_samples = None
            self.final_base_flux_error_samples = None
            self.final_flux_error_samples = None
            self.final_base_fractional_error_samples = None
            self.final_fractional_error_samples = None
            self.final_pixel_indices = None
            self.final_tile_indices = None
            self.final_longitudes = None
            self.final_latitudes = None
            self.final_temperature_samples = None
            self.final_elevation_samples = None

        summaries: dict[str, NDArray[np.float32]] = {}
        if include_temperature_summary:
            temperature_flux_values = (
                np.concatenate(summary_temperature_flux)
                if summary_temperature_flux
                else np.empty(0, dtype=np.float32)
            )
            temperature_values = (
                np.concatenate(summary_temperature)
                if summary_temperature
                else np.empty(0, dtype=np.float32)
            )
            summaries["temperature"] = binned_flux_quantiles_exact(
                temperature_flux_values,
                temperature_values,
                bin_edges=temperature_edges,
                quantiles=temperature_quantiles,
            )
        if include_elevation_summary:
            elevation_flux_values = (
                np.concatenate(summary_elevation_flux)
                if summary_elevation_flux
                else np.empty(0, dtype=np.float32)
            )
            elevation_values = (
                np.concatenate(summary_elevation)
                if summary_elevation
                else np.empty(0, dtype=np.float32)
            )
            summaries["elevation"] = binned_flux_quantiles_exact(
                elevation_flux_values,
                elevation_values,
                bin_edges=elevation_edges,
                quantiles=elevation_quantiles,
            )

        return output_map, output_mask, summaries


class RacsLow3(Racs):
    """Backwards-compatible LOW3 simulator wrapper."""

    def __init__(self, config: RacsConfig):
        if config.product != RACS_LOW3:
            raise ValueError("RacsLow3 requires a LOW3 RACS product configuration.")
        super().__init__(config)
