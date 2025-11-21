from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence
import numpy as np
from numpy.typing import DTypeLike


@dataclass(frozen=True)
class NPKey:
    """
    A JAX-like, immutable PRNG key built on NumPy's SeedSequence.

    - Splittable: key.split(n) -> list[NPKey]
    - Fold-in:    key.fold_in(i) -> NPKey   (mix in an int, e.g., step id or device id)
    - Stateless sampling helpers: normal, uniform, integers, choice, permutation

    Design: each sampling call constructs a *fresh* Generator from the SeedSequence,
    so results are fully deterministic and independent of call order (like JAX).
    """
    _ss: np.random.SeedSequence

    # ---- constructors --------------------------------------------------------
    @staticmethod
    def from_seed(seed: int | Sequence[int]) -> "NPKey":
        return NPKey(np.random.SeedSequence(seed))

    def fold_in(self, data: int) -> "NPKey":
        """
        Mix an integer into the key (similar to jax.random.fold_in).
        We create a new SeedSequence with a nested entropy tuple to keep it deterministic.
        """
        # SeedSequence accepts any hashable sequence for entropy; nesting is fine.
        entropy: Tuple = (self._ss.entropy, int(data))
        return NPKey(np.random.SeedSequence(entropy))

    def split(self, n: int = 2) -> list["NPKey"]:
        """
        Deterministically spawn n independent child keys.
        """
        return [NPKey(ss) for ss in self._ss.spawn(n)]

    # ---- low-level: build a fresh, stateless Generator ----------------------
    def _generator(self) -> np.random.Generator:
        bitgen = np.random.PCG64(self._ss)
        return np.random.Generator(bitgen)

    # ---- stateless sampling helpers (JAX-like) ------------------------------
    def normal(self, shape: Iterable[int] = (), loc=0.0, scale=1.0, dtype=np.float32):
        g = self._generator()
        return g.normal(loc=loc, scale=scale, size=tuple(shape)).astype(dtype, copy=False)

    def uniform(self, shape: Iterable[int] = (), low=0.0, high=1.0, dtype: DTypeLike=np.float32):
        g = self._generator()
        return g.uniform(low=low, high=high, size=tuple(shape)).astype(dtype, copy=False)

    def poisson(self, lam, shape=(), dtype=np.int32):
        """
        Draw Poisson(lam) samples.
        - lam can be scalar or array-like (broadcasts to 'shape').
        - Returns np.ndarray with dtype (default int32).
        """
        g = self._generator()
        out = g.poisson(lam=lam, size=tuple(shape))
        return out.astype(dtype, copy=False)

    def integers(self, low: int, high: int | None = None, shape: Iterable[int] = (), dtype=np.int32):
        g = self._generator()
        return g.integers(low, high, size=tuple(shape), dtype=dtype)

    def choice(self, a, shape: Iterable[int] = (), replace=True, p=None):
        g = self._generator()
        return g.choice(a, size=tuple(shape), replace=replace, p=p)

    def permutation(self, x):
        g = self._generator()
        return g.permutation(x)

class NPKeySequence:
    """
    Iterator that mimics hk.PRNGSequence, but using NPKey.
    Yields a fresh NPKey each time.
    """
    def __init__(self, seed: int | np.ndarray | Sequence[int] | NPKey):
        if isinstance(seed, NPKey):
            self._key = seed
        else:
            self._key = NPKey.from_seed(seed) # type: ignore
        self._count = 0

    def __iter__(self):
        return self

    def __next__(self) -> NPKey:
        # Fold in the step counter for determinism
        child = self._key.fold_in(self._count)
        self._count += 1
        return child

    def take(self, n: int) -> list[NPKey]:
        return [next(self) for _ in range(n)]

    # convenience: JAX-like module-level functions
def prng_key(seed: int | Sequence[int]) -> NPKey:
    return NPKey.from_seed(seed)

def split(key: NPKey, n: int = 2) -> list[NPKey]:
    return key.split(n)

def fold_in(key: NPKey, data: int) -> NPKey:
    return key.fold_in(data)

def normal(key: NPKey, shape=(), loc=0.0, scale=1.0, dtype=np.float32):
    return key.normal(shape, loc, scale, dtype)

def uniform(key: NPKey, shape=(), low=0.0, high=1.0, dtype=np.float32):
    return key.uniform(shape, low, high, dtype)

def poisson(key: NPKey, lam, shape=(), dtype=np.int32):
    return key.poisson(lam, shape, dtype)

def integers(key: NPKey, low, high=None, shape=(), dtype=np.int32):
    return key.integers(low, high, shape, dtype)
