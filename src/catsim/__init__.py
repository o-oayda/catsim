from .simulator import Catwise
from .configs import CatwiseConfig
from .racs import RacsLow3, RacsLow3Config
from .utils.batch_simulate import batch_simulate
from .utils.plotting import smooth_map
from .utils.rng import prng_key


def __getattr__(name: str):
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Catwise",
    "CatwiseConfig",
    "RacsLow3",
    "RacsLow3Jax",
    "RacsLow3Config",
    "batch_simulate",
    "smooth_map",
    "prng_key",
]
