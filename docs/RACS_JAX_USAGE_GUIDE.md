# RACS MID1 JAX Usage Guide

This guide explains how the JAX RACS MID1 simulator is generated and executed
in the current implementation. The NumPy simulator in `src/catsim/racs.py`
remains the reference implementation.

## Basic Usage

```python
import jax
import numpy as np

from catsim import RACS_MID1, RacsConfig, RacsJax

cfg = RacsConfig(
    product=RACS_MID1,
    flux_min=15.0,
    chunk_size=50_000,
    store_final_samples=False,
    max_cluster_children_per_parent=16,
    # Needed only if the feature caches have not been built:
    noisemap_data_dir="/path/to/racs/noisemaps",
    noise_map_nside=256,
    flux_error_noise_bins=200,
    flux_error_flux_bins=300,
    flux_error_min_cell_count=10,
    flux_error_noise_bounds_ujy_beam=(100.0, 1000.0),
    flux_error_flux_bounds_mjy=(0.1, 10_000.0),
)

sim = RacsJax(cfg)
sim.initialise_data()

maps, masks = sim.batch_generate_dipole(
    theta={
        "log10_n_initial_samples": np.full(8, 5.0),
        "p_clus": np.zeros(8),
        "clus_stop_prob": np.ones(8),
        "elevation_amp": np.zeros(8),
        "elevation_trough": np.zeros(8),
    },
    key=jax.random.PRNGKey(123),
    batch_size=4,
)
```

For `nside=64`, `maps.shape == (8, 49152)` and
`masks.shape == (8, 49152)`.

## Generation Flow

`RacsJax.initialise_data()` reuses the existing NumPy `Racs` lookup
initialization, then converts the final lookup products into JAX-friendly
arrays:

- log-flux histogram edges and CDF;
- survey mask;
- per-pixel SBID/tile mixture tables;
- tile temperature table;
- the product's flat NESTED local-noise map;
- compact noise/flux-conditioned absolute-error values and cell-routing arrays;
- per-pixel source-elevation lookup tables.

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
   - sample tile assignment and source elevation;
   - apply temperature suppression and elevation enhancement;
   - query local noise at the aberrated coordinates and sample an absolute
     flux-error sigma conditioned on that noise and the post-physics,
     pre-noise flux;
   - reject invalid-noise sources and apply Gaussian flux noise;
   - apply mask, tile validity, and flux threshold;
   - scatter-add kept sources into the density map.
9. Return `(maps, masks)` as NumPy arrays on the host.

## Clustering

Both clustering models are supported.

Geometric clustering:

```python
cfg = RacsConfig(
    product=RACS_MID1,
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
cfg = RacsConfig(
    product=RACS_MID1,
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

For the controlled JAX-only legacy-versus-new benchmark of the MID1
noise/error lookup, including the posterior-median configuration, raw batch
timings, memory measurements, and reproducibility protocol, see
[RACS MID1 JAX noise-lookup performance](RACS_JAX_NOISE_LOOKUP_PERFORMANCE.md).

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

## Noise-map and absolute-error caches

LOW2, LOW3, and MID1 use these external source maps:

| Product | Filename | Native geometry | Physical unit |
| --- | --- | --- | --- |
| LOW2 | `RACS-low2.iqr.hpx` | nside 2048, RING, equatorial | uJy/beam |
| LOW3 | `RACS-low3.iqr.hpx` | nside 2048, RING, equatorial | uJy/beam |
| MID1 | `RACS-mid1.iqr.hpx` | nside 1024, RING, equatorial | uJy/beam |

The source headers have no unit keyword. CatSIM assigns `uJy/beam` by contract,
validates the geometry, averages valid fine pixels to the configured nside, and
caches float32 values in NESTED order. The LOW2/MID1 defaults are:

```python
noise_map_nside = 256
flux_error_noise_bins = 200
flux_error_flux_bins = 300
flux_error_min_cell_count = 10
flux_error_noise_bounds_ujy_beam = (100.0, 1000.0)
flux_error_flux_bounds_mjy = (0.1, 10_000.0)
noisemap_data_dir = None
```

LOW3 uses the same settings except that its lower noise bound is exactly 0.1
dex lower: `flux_error_noise_bounds_ujy_beam = (10**1.9, 1000.0)`, or about
79.43--1,000 uJy/beam.

`noisemap_data_dir` is an input directory, not a cache directory. Generated
files are stored beneath the selected product's package data directory:

```text
src/catsim/data/racs_low3/lookups/
  absolute_error_lookup_noise256_grid200x300_min10_bounds-noise79p4328234724to1000_flux0p1to10000_v2.npz
src/catsim/data/racs_mid1/lookups/
  absolute_error_lookup_noise256_grid200x300_min10_bounds-noise100to1000_flux0p1to10000_v2.npz
src/catsim/data/racs_low2/lookups/
  absolute_error_lookup_noise256_grid200x300_min10_bounds-noise100to1000_flux0p1to10000_v2.npz
```

Each product directory also contains `noise_map_nside256_nested_v1.npz`.

Initialization validates and loads these caches first. When both are valid,
the original noise map, `noisemap_data_dir`, and training catalogue are not
needed for this feature. A missing/incompatible noise-map cache needs the
product `.hpx` source; a missing/incompatible conditional lookup needs the
catalogue. Changing nside invalidates both caches. Changing grid dimensions,
physical bounds, or minimum occupancy rebuilds only the conditional lookup.
Catalogue rows outside the inclusive bounds are excluded during training. At
runtime both NumPy and JAX clip finite-positive queries into boundary cells;
neither rejects them nor performs retries.

Use the explicit precompute entry point to build or replace them and retain a
JSON record of commit, configuration, timings, memory, paths, and benchmark
seed:

```bash
uv run python scripts/precompute_racs_noise_lookup.py \
  --product mid1 \
  --noisemap-data-dir /path/to/racs/noisemaps \
  --catalogue-path /path/to/racs-mid1.fits \
  --rebuild all \
  --benchmark-samples 1000000
```

`--rebuild noise` necessarily replaces the identity-dependent lookup too.
`--rebuild lookup` reuses the existing compatible noise cache. With
`--rebuild none`, valid caches are only loaded and can be benchmarked without
external data.

Cache construction emits, beside each `.npz`:

- `noise_map_....map.png`, `.coverage.png`, `.hist.png`, and `.summary.json`;
- `absolute_error_lookup_....diagnostics.png`, `.marginals.png`, and
  `.summary.json`.

The plots show the accepted in-bound training distribution. The summaries
record the configured grid range, robust in-bound noise/flux/error percentiles,
finite-positive candidates, disjoint invalid-row reasons, per-bound exclusion
counts and their union, accepted rows, eligible-cell fractions, and fallback
distances. Runtime cache loading does not regenerate diagnostics and does not
retain catalogue conditioning coordinates.

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

## Flux-error and invalid-noise semantics

Both NumPy and JAX now draw catalogue `E_Total_flux` as an **absolute** base
sigma in mJy. The lookup coordinates are `log10(noise / uJy beam^-1)` and
`log10(pre-noise flux / mJy)`. Sparse cells route to a precomputed nearest
eligible cell in the two coordinates measured in dex; there is no runtime tree
search or padded per-cell allocation.

The legacy public parameter name and scaling convention are preserved exactly:

```text
sigma_effective = sigma_lookup * sqrt(1 + fractional_error_eta)
```

The sampled sigma is not multiplied by flux. Stored fractional-error samples
and maps are secondary diagnostics derived by dividing the base/effective
absolute sigma by a safe pre-noise flux.

Noise is queried using aberrated equatorial coordinates. A source receiving
`NaN`, HEALPix `UNSEEN`, zero, or negative local noise is excluded before
error sampling, observed-flux thresholding, the density map, and all flux
summaries. This source-level exclusion does not alter the returned survey mask
or mask a whole output pixel.

## Pinned legacy-versus-new comparisons

Generate old and new ensembles in separate pinned commits/worktrees and save
each as an `.npz` containing:

```text
maps                         (n_simulations, 49152), NESTED nside 64
rejected_invalid_noise_maps  same shape (explicit zeros for legacy if applicable)
metadata_json                JSON with commit, config, seeds, cache identities,
                             and environment
dipole                       optional
angular_power                optional
```

The current JAX runner writes this artifact directly, including the
source-rejection map, cache identities, environment, initialization timing,
and simulation throughput:

```bash
uv run python scripts/run_racs_jax_batch.py \
  --n-sims 100 --batch-size 10 --log10-n 6 \
  --output artifacts/noise-conditioned.npz
```

After a NumPy realization, the corresponding diagnostics are available as
`sim.invalid_noise_rejection_map` and
`sim.invalid_noise_rejection_count`. Batched JAX runs expose
`sim.last_invalid_noise_rejection_maps` and counts.

Then compare the saved artifacts without checking out either branch:

```bash
uv run python scripts/compare_racs_noise_ensembles.py \
  --old artifacts/legacy.npz \
  --new artifacts/noise-conditioned.npz \
  --output-dir artifacts/comparison
```

The output includes nside-64 and summed-NESTED nside-4 ensemble means,
absolute/fractional differences, Monte Carlo uncertainty, standardized
differences, variance changes, total-count distributions, and maps/counts of
invalid-noise rejections. Embedded or separately supplied dipole/angular-power
arrays are compared as optional products. `comparison_metadata.json` records
both artifact hashes and their pinned metadata. Equal root seeds alone do not
prove pairing because RNG call sequences can change. Paired uncertainty is
used only when both artifacts set `pairing_verified: true` and provide the
same `paired_realization_ids`, obtained by explicitly reusing identical
pre-noise sources and Gaussian deviates. The independent-ensemble uncertainty
is always retained in the `.npz`. The comparison also rejects mismatches in
the complete effective science configuration (source counts, clustering,
observer, temperature/elevation, flux cuts, eta, and sampling controls).

## Shared clustering configuration

The shared NumPy-side clustering cap is configured through `RacsConfig`:

```python
max_cluster_children_per_parent: int = 16
```

This field is used by `RacsJax` for fixed-shape clustering. The existing
`Racs.generate_dipole` path does not use the cap and still performs dynamic
NumPy clustering as before.

JAX is now an optional dependency extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
jax = [
    "jax[cuda13]>=0.10.1,<0.11.0",
]
```

`from catsim import Racs` remains JAX-free. `RacsJax` is loaded lazily.
