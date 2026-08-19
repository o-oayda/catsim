## RACS LOW2/LOW3/MID1 noise-conditioned flux errors

RACS LOW2, its 25- and 45-arcsec patched variants, LOW3, and MID1 use an
empirical distribution of **absolute** total-flux errors conditioned on local
survey noise and deterministic pre-noise total flux. The LOW2 variants share
the LOW2 noise map but retain independent caches because they are trained on
different catalogues. The external map contract is:

| Product | Source file | Source map | Assigned unit |
| --- | --- | --- | --- |
| LOW2 | `RACS-low2.iqr.hpx` | nside 2048, RING, equatorial | uJy/beam |
| LOW3 | `RACS-low3.iqr.hpx` | nside 2048, RING, equatorial | uJy/beam |
| MID1 | `RACS-mid1.iqr.hpx` | nside 1024, RING, equatorial | uJy/beam |

The FITS headers do not supply the unit; `uJy/beam` is part of the CatSIM
scientific contract. The production defaults cache a NESTED nside-256 map and a
bounded 200 x 300 `(log10 noise, log10 flux)` grid whose directly sampled cells
contain at least 10 catalogue sources. All products use inclusive flux bounds
of 0.1--10,000 mJy. MID1 uses noise bounds of 100--1,000 uJy/beam; LOW2,
its patched variants, and LOW3 extend the lower noise edge down by 0.1 dex to
10^1.9 = 79.43 uJy/beam:

```python
from catsim import RACS_MID1, Racs, RacsConfig

cfg = RacsConfig(
    product=RACS_MID1,
    flux_min=15.0,
    noisemap_data_dir="/path/to/racs/noisemaps",  # build/rebuild only
    noise_map_nside=256,
    flux_error_noise_bins=200,
    flux_error_flux_bins=300,
    flux_error_min_cell_count=10,
    flux_error_noise_bounds_ujy_beam=(100.0, 1000.0),
    flux_error_flux_bounds_mjy=(0.1, 10_000.0),
)
sim = Racs(cfg)
sim.initialise_data()
```

Generated products live under
`src/catsim/data/racs_low3/lookups/` or
`src/catsim/data/racs_mid1/lookups/`, or `src/catsim/data/racs_low2/lookups/`.
The feature-specific files are named
`noise_map_nside<N>_nested_v1.npz` and
`absolute_error_lookup_noise<N>_grid<A>x<B>_min<C>_bounds-..._v2.npz`. The
bounds are included in the v2 filename and validated metadata, so the previous
unbounded 400 x 400 v1 cache cannot be mistaken for the production lookup. A
matching cache is
loaded before any source input is consulted, so these two caches can be used
without `noisemap_data_dir`, the original `.hpx` map, or the training
catalogue. A missing or incompatible cache needs those build inputs.

To explicitly build, rebuild, or benchmark the feature caches, use:

```bash
uv run python scripts/precompute_racs_noise_lookup.py \
  --product mid1 \
  --noisemap-data-dir /path/to/racs/noisemaps \
  --catalogue-path /path/to/mid1-catalogue.fits \
  --rebuild all \
  --benchmark-samples 1000000
```

For patched LOW2 catalogues, use `--product low2-25as` or `--product
low2-45as` with their respective FITS file. Their results are written to
`src/catsim/data/racs_low2_25as/lookups/` and
`src/catsim/data/racs_low2_45as/lookups/`.

`--rebuild noise` also rebuilds the identity-dependent error grid;
`--rebuild lookup` reuses a valid noise-map cache. Every build emits sidecar
diagnostics beside its cache: noise `.map.png`, `.coverage.png`, `.hist.png`,
and `.summary.json`; and error-grid `.diagnostics.png`, `.marginals.png`, and
`.summary.json`. The summaries record configured ranges, robust percentiles,
finite-positive candidates, invalid-row reasons, per-bound exclusion counts,
their union, occupancy, and sparse-cell routing distances.

Training rows outside either inclusive physical range are excluded; they are
not clipped into and do not contaminate a boundary cell. Runtime behavior is
deliberately different: every finite-positive query is clipped to the nearest
boundary grid cell, then follows the same precomputed sparse-cell routing and
makes one draw. There is no rejection, retry, or runtime nearest-neighbour
search. `ConditionalErrorLookup.query_range_counts()` reports the four
below/above counters and their union for a supplied NumPy query batch.

At runtime, catalogue `E_Total_flux` is sampled in mJy as a base absolute
sigma. The existing parameter name and convention remain:

```text
sigma_effective = sigma_lookup * sqrt(1 + fractional_error_eta)
```

It is not multiplied by flux. Fractional-error result fields are derived
diagnostics (`sigma / pre_noise_flux`). Sources whose aberrated coordinates
query `NaN`, HEALPix `UNSEEN`, zero, or negative noise are removed before flux
sampling, thresholding, maps, and summaries. This is source-level removal; the
returned survey mask is unchanged.

For rollout comparisons, save legacy and new nside-64 ensembles from pinned
commits/worktrees and run `scripts/compare_racs_noise_ensembles.py`. It consumes
the saved artifacts and produces nside-64 and nside-4 mean, difference,
fractional, uncertainty, standardized, variance, total-count, and
invalid-noise-rejection comparisons without checking out either branch.
Artifact metadata must include commit, configuration, seeds, cache identities,
and environment. Paired uncertainty additionally requires explicit verified
per-realization pairing IDs; equal seed lists alone are not treated as proof of
identical simulated sources.
