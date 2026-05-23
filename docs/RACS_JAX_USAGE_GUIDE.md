# RACS-low3 JAX Usage Guide

This guide explains how the JAX RACS-low3 simulator is generated and executed
in the current implementation. The NumPy simulator in `src/catsim/racs.py`
remains the reference implementation.

## Basic Usage

```python
import jax
import numpy as np

from catsim import RacsLow3Config, RacsLow3Jax

cfg = RacsLow3Config(
    flux_min=15.0,
    chunk_size=50_000,
    store_final_samples=False,
    max_cluster_children_per_parent=16,
)

sim = RacsLow3Jax(cfg)
sim.initialise_data()

maps, masks = sim.batch_generate_dipole(
    theta={
        "log10_n_initial_samples": np.full(8, 5.0),
        "p_clus": np.zeros(8),
        "clus_stop_prob": np.ones(8),
    },
    key=jax.random.PRNGKey(123),
    batch_size=4,
)
```

For `nside=64`, `maps.shape == (8, 49152)` and
`masks.shape == (8, 49152)`.

## Generation Flow

`RacsLow3Jax.initialise_data()` reuses the existing NumPy `RacsLow3` lookup
initialization, then converts the final lookup products into JAX-friendly
arrays:

- log-flux histogram edges and CDF;
- survey mask;
- per-pixel SBID/tile mixture tables;
- tile temperature table;
- per-pixel fractional-error lookup tables.

Simulation then runs through a stateless JAX path:

1. Normalize scalar or batched `theta` parameters.
2. Validate clustering arguments according to `cluster_count_model`.
3. Convert `log10_n_initial_samples` to parent counts after expected clustering
   multiplicity.
4. Split the PRNG key per simulation.
5. Process host-side batches of size `batch_size`.
6. Inside compiled JAX, `vmap` over simulations.
7. Inside each simulation, `lax.scan` over fixed-size source chunks.
8. For each chunk:
   - sample parent fluxes and isotropic positions;
   - sample fixed-shape geometric or Poisson child counts;
   - truncate child counts above `max_cluster_children_per_parent`;
   - sample padded child positions;
   - sample spectral indices;
   - apply aberration and Doppler flux boosting;
   - compute HEALPix NESTED pixels with a JAX implementation;
   - sample tile assignment and fractional flux error;
   - apply temperature suppression and Gaussian flux noise;
   - apply mask, tile validity, and flux threshold;
   - scatter-add kept sources into the density map.
9. Return `(maps, masks)` as NumPy arrays on the host.

## Clustering

Both clustering models are supported.

Geometric clustering:

```python
cfg = RacsLow3Config(
    flux_min=15.0,
    store_final_samples=False,
    cluster_count_model="geometric",
)

theta = {
    "log10_n_initial_samples": np.full(n_sims, 5.0),
    "p_clus": np.full(n_sims, 0.3),
    "clus_stop_prob": np.full(n_sims, 0.8),
}
```

Poisson clustering:

```python
cfg = RacsLow3Config(
    flux_min=15.0,
    store_final_samples=False,
    cluster_count_model="poisson",
)

theta = {
    "log10_n_initial_samples": np.full(n_sims, 5.0),
    "lambda_clus": np.full(n_sims, 0.5),
}
```

If

```text
P(children_per_parent > max_cluster_children_per_parent) > 0.01
```

the simulator raises a `RuntimeWarning`. Excess children are deterministically
truncated to preserve fixed JAX shapes.

## Performance Script

Use `scripts/run_racs_jax_batch.py` for quick performance checks:

```bash
uv run python scripts/run_racs_jax_batch.py \
  --n-sims 16 \
  --batch-size 4 \
  --chunk-size 50000 \
  --log10-n 5 \
  --flux-min 15
```

Poisson clustering example:

```bash
uv run python scripts/run_racs_jax_batch.py \
  --n-sims 16 \
  --batch-size 4 \
  --log10-n 5 \
  --cluster-model poisson \
  --lambda-clus 0.5 \
  --max-children 16
```

The script prints:

- JAX devices;
- lookup initialization time;
- warmup/compile time;
- timed batch generation time;
- output shapes;
- mean kept sources per map;
- simulations per second;
- requested parent-source slots per second.

## Batch Size

`batch_size` controls how many independent simulations are executed together in
one compiled JAX `vmap` call.

For example:

```text
n_sims=100, batch_size=10
```

runs 10 host-side batches, each containing 10 simulations. The final output is
concatenated on the host into shape `(100, npix)`.

Larger `batch_size` can improve GPU utilization and reduce Python overhead, but
uses more device memory. Smaller `batch_size` reduces memory pressure and is
safer for large `log10_n_initial_samples`, but may reduce throughput.

Changing `batch_size` can trigger a new JAX compilation because the compiled
batch shape changes.

## Changes To The NumPy Branch

The NumPy simulator behavior is intended to remain unchanged.

The only shared NumPy-side change is the new config field in
`RacsLow3Config`:

```python
max_cluster_children_per_parent: int = 16
```

This field is used by `RacsLow3Jax` for fixed-shape clustering. The existing
`RacsLow3.generate_dipole` path does not use the cap and still performs dynamic
NumPy clustering as before.

JAX is now an optional dependency extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
jax = [
    "jax[cuda13]>=0.10.1,<0.11.0",
]
```

`from catsim import RacsLow3` remains JAX-free. `RacsLow3Jax` is loaded lazily.
