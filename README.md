# catsim

## Installation

Install catsim like any ordinary package into your virtual environment.
For example, with pip:
```bash
pip install git+https://github.com/o-oayda/catsim.git
```

Or if you want to edit and work with catsim, do:

```bash
git clone git+https://github.com/o-oayda/catsim.git
cd catsim
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If you prefer Poetry, the configuration is already present—you can just run
`poetry install`.

### CatWISE S21 catalogue

The CatWISE S21 FITS catalogue (`catwise_agns_masked_final_w1lt16p5_alpha.fits`)
is too large to ship with this repository. Download it separately and point
CatSIM to the local path via either the `CATSIM_S21_PATH` environment variable
or the `CatwiseConfig.s21_catalogue_path` option, for example:

```python
config = CatwiseConfig(
    cat_w12_min=0.5,
    cat_w1_max=17.0,
    magnitude_error_dist='gaussian',
    s21_catalogue_path='/path/to/catwise_agns_masked_final_w1lt16p5_alpha.fits'
)
```

If neither is supplied and the file is not bundled, CatSIM will raise a clear
`FileNotFoundError` describing how to provide it.

## Example Scripts

The scripts below show to generate CatSIMS. We need to instantiate the CatWISE
class with an instance of CatWISEConfig, which sets the limiting magnitudes and
colours from which sources are drawn. We also need to specify the distribution
of the photometric errors. Other options are described in CatwiseConfig.

### `scripts/test_catsim.py`

This shows how to generate one CatSIM realisation. Note that `generate_dipole`
returns a tuple containing the source density map and the mask.

```python
from catsim import CatwiseConfig, Catwise, smooth_map
import healpy as hp
import matplotlib.pyplot as plt

config = CatwiseConfig(
    cat_w12_min=0.5,
    cat_w1_max=17.0,
    magnitude_error_dist='gaussian'
)
sim = Catwise(config)

sim.initialise_data()
dmap, mask = sim.generate_dipole(log10_n_initial_samples=7.5)

hp.projview(dmap, nest=True)
smooth_map(dmap)
plt.show()
```

### `scripts/batch_catsim.py`

Since CatSIM is slow but easily parallelisable, we use joblib to distribute
simulation jobs across multiple workers. This script demonstrates how to do that,
using 10 different values for `log10_n_initial_samples` to imply 10 different
simulations. Each simulation shares the same initialised `Catwise` instance
but receives a different `log10_n_initial_samples` value.

The `batch_simulate` function executes this parallelisation. The `theta` 
dictionary defines the batched parameter inputs, which are turned into 
per-simulation argument dictionaries inside `batch_simulate`. We can also specify
a custom `NPKey`, which is just a wrapper on numpy's rng generator utilities
made to behave like jax's `PRNGKey`. This custom key is folded into unique 
per-simulation keys so each worker produces a decorrelated result.

```python
from catsim import CatwiseConfig, Catwise, batch_simulate, prng_key
import numpy as np

N_SIMS = 10
rng_key = prng_key(42)
config = CatwiseConfig(
    cat_w12_min=0.5,
    cat_w1_max=17.0,
    magnitude_error_dist='gaussian'
)
sim = Catwise(config)
sim.initialise_data()

simulator_function = sim.generate_dipole
theta = {'log10_n_initial_samples': np.linspace(7, 8, N_SIMS)}
dmap, mask = batch_simulate(
    theta=theta,
    model_callable=simulator_function,
    n_workers=32,
    rng_key=rng_key
)
```

## RACS LOW1 tile identity

LOW1 uses a dense integer encoding of catalogue `Tile_ID`, because its
multi-field scheduling blocks do not identify unique sky tiles. See
[the LOW1 tile-indexing note](docs/RACS_LOW1_TILE_INDEXING.md) for the verified
catalogue invariants, compatibility details, and remaining temperature-data
limitation.

## CatWISE source catalogue and hashgrid cache

The source catalogue and generated hashgrids are not distributed through Git LFS.
Supply `CatwiseConfig.source_catalogue_path` when building lookup data, including
on the first `initialise_data()` call without a compatible hashgrid cache.
The expected filename and SHA-256 for each supported set of cuts are tracked in
`src/catsim/data/catwise_catalogues.json`. Files with a different checksum are
rejected before loading. The S21 sample catalogue (`s21_catalogue_path`) is a
separate input.

```python
config = CatwiseConfig(
    cat_w1_max=17.0, cat_w12_min=0.5, magnitude_error_dist="gaussian",
    source_catalogue_path="/data/catwise2020_corr_w120p5_w117p0.fits",
)
model = Catwise(config)  # CatwiseJax accepts the same config
model.initialise_data()
```

Hashgrids are built with the defaults used by `precompute_data()` and saved
under `$XDG_CACHE_HOME/catsim/catwise` (normally `~/.cache/catsim/catwise`).
Set `CatwiseConfig.hashgrid_cache_dir` to override that directory. Cache metadata
records the source digest, mask version, ecliptic mask setting, and grid settings.
Compatible caches are reused without requiring the source catalogue. Legacy grids
without provenance, LFS pointers, and incompatible caches are rebuilt.
The other precomputed CatWISE data products must still be available.

Removing LFS tracking does not rewrite old commits; historical LFS references
remain in repository history.
