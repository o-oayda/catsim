from .simulator import Catwise
from .configs import CatwiseConfig
from .racs import Racs, RacsConfig, RacsLow3, RacsLow3Config
from .racs_products import (
    RACS_LOW2,
    RACS_LOW2_25AS,
    RACS_LOW2_45AS,
    RACS_LOW3,
    RACS_MID1,
    RACS_PRODUCTS,
    RacsProductSpec,
)
from .racs_summaries import (
    binned_flux_quantile_ndim,
    binned_flux_quantiles_exact,
    binned_flux_quantiles_histogram,
    empirical_bin_edges,
)
from .racs_temperature import TEMPERATURE_MODELS, TemperatureModel
from .utils.batch_simulate import batch_simulate
from .utils.plotting import smooth_map
from .utils.rng import prng_key


def __getattr__(name: str):
    if name == "CatwiseJax":
        try:
            from .simulator_jax import CatwiseJax
        except ModuleNotFoundError as exc:
            if exc.name and (exc.name == "jax" or exc.name.startswith("jax.")):
                raise ImportError(
                    "CatwiseJax requires the optional JAX dependencies. "
                    "Install them with `pip install 'catsim[jax]'`."
                ) from exc
            raise
        return CatwiseJax
    if name == "RacsLow3Jax":
        try:
            from .racs_jax import RacsLow3Jax
        except ModuleNotFoundError as exc:
            if exc.name and (exc.name == "jax" or exc.name.startswith("jax.")):
                raise ImportError(
                    "RacsLow3Jax requires the optional JAX dependencies. "
                    "Install them with `pip install 'catsim[jax]'`."
                ) from exc
            raise
        return RacsLow3Jax
    if name == "RacsJax":
        try:
            from .racs_jax import RacsJax
        except ModuleNotFoundError as exc:
            if exc.name and (exc.name == "jax" or exc.name.startswith("jax.")):
                raise ImportError(
                    "RacsJax requires the optional JAX dependencies. "
                    "Install them with `pip install 'catsim[jax]'`."
                ) from exc
            raise
        return RacsJax
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Catwise",
    "CatwiseJax",
    "CatwiseConfig",
    "Racs",
    "RacsJax",
    "RacsConfig",
    "RacsLow3",
    "RacsLow3Jax",
    "RacsLow3Config",
    "RACS_LOW2",
    "RACS_LOW2_25AS",
    "RACS_LOW2_45AS",
    "RACS_LOW3",
    "RACS_MID1",
    "RACS_PRODUCTS",
    "RacsProductSpec",
    "binned_flux_quantile_ndim",
    "binned_flux_quantiles_exact",
    "binned_flux_quantiles_histogram",
    "empirical_bin_edges",
    "TEMPERATURE_MODELS",
    "TemperatureModel",
    "batch_simulate",
    "smooth_map",
    "prng_key",
]
