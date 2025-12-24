import time
from catsim import CatwiseConfig, Catwise
import numpy as np
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt


@dataclass
class HashGridLevel:
    step: np.ndarray                 # (4,) float
    shape: np.ndarray                # (4,) int  number of cells per dim
    unique_cell_ids: np.ndarray      # (U,) int64 sorted unique occupied cell ids
    offsets: np.ndarray              # (U+1,) int64 offsets into members
    members: np.ndarray              # (N,) int32/int64 training indices sorted by cell id


@dataclass
class SigmaHashGrid4D:
    """
    Multi-resolution sparse 4D hash grid for fast sampling of (sigma_w1, sigma_w2)
    conditioned on (w1_mag, w1_cov, w2_mag, w2_cov).
    """
    mins: np.ndarray                 # (4,) float
    maxs: np.ndarray                 # (4,) float
    base_step: np.ndarray            # (4,) float
    levels: List[HashGridLevel]
    Y: np.ndarray                    # (N, 2) float  training sigmas
    rng_seed: Optional[int] = None


def _compute_shape(mins: np.ndarray, maxs: np.ndarray, step: np.ndarray) -> np.ndarray:
    # inclusive-ish coverage; we clip coordinates so exact shape is not too precious
    span = np.maximum(maxs - mins, 0.0)
    shape = np.floor(span / step).astype(np.int64) + 1
    shape = np.maximum(shape, 1)
    return shape.astype(np.int64)


def _coords_to_cell_id(coords: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    Map a 4d integer coordinate (i_0, i_1, i_2, i_3) -> a single scalar index.

    coords: (M, 4) int64, clipped to [0, shape[d]-1]
    shape: (4,) int64
    returns: (M,) int64 linear cell id
    """
    # linearization: (((i0 * n1 + i1) * n2 + i2) * n3 + i3)
    n1, n2, n3 = shape[1], shape[2], shape[3]
    return (((coords[:, 0] * n1 + coords[:, 1]) * n2 + coords[:, 2]) * n3 + coords[:, 3]).astype(np.int64)


def _points_to_cell_id(X: np.ndarray, mins: np.ndarray, step: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    X: (M,4) float
    mins: (4,) float
    step: (4,) float
    shape: (4,) int64
    returns: (M,) int64 cell ids
    """
    # Convert to coordinates
    coords = np.floor((X - mins) / step).astype(np.int64)
    # Clip to grid bounds to avoid out-of-range ids
    coords = np.clip(coords, 0, shape - 1)
    return _coords_to_cell_id(coords, shape)


def _build_level(X_train: np.ndarray, shape: np.ndarray, cell_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CSR-like bucket structure:
      unique_cell_ids (sorted),
      offsets (len=U+1),
      members (training indices sorted by cell id).
    """
    N = X_train.shape[0]
    order = np.argsort(cell_ids, kind="mergesort")  # stable
    cell_sorted = cell_ids[order]
    members = order.astype(np.int32) if N < (1 << 31) else order.astype(np.int64)

    unique_ids, first_idx, counts = np.unique(cell_sorted, return_index=True, return_counts=True)
    offsets = np.empty(unique_ids.size + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts, dtype=np.int64)

    return unique_ids.astype(np.int64), offsets, members


def build_sigma_hashgrid_4d(
    w1_mag: np.ndarray,
    w1_cov: np.ndarray,
    w2_mag: np.ndarray,
    w2_cov: np.ndarray,
    sigma_w1: np.ndarray,
    sigma_w2: np.ndarray,
    *,
    base_step: Tuple[float, float, float, float] = (0.10, 0.10, 0.10, 0.10),
    n_levels: int = 3,
    coarsen_factor: float = 2.0,
    mins: Optional[Tuple[float, float, float, float]] = None,
    maxs: Optional[Tuple[float, float, float, float]] = None,
    drop_nonfinite: bool = True,
    rng_seed: Optional[int] = None,
) -> SigmaHashGrid4D:
    """
    Build a multi-resolution 4D hash grid for fast sampling of (sigma_w1, sigma_w2).

    Parameters
    ----------
    base_step:
        Per-dimension cell sizes at the finest level, in the *native units* you provide.
        You said mags ~ [9,17] and cov already log-space ~ [1,3], so these are meaningful.
        Tune these; smaller = more local but more empty buckets.
    n_levels:
        Number of levels. Each level l uses step = base_step * (coarsen_factor ** l).
    mins, maxs:
        Optional explicit bounding box. If None, computed from data min/max.
        Using explicit bounds is helpful for portability/reproducibility.
    """
    w1_mag = np.asarray(w1_mag)
    w1_cov = np.asarray(w1_cov)
    w2_mag = np.asarray(w2_mag)
    w2_cov = np.asarray(w2_cov)
    sigma_w1 = np.asarray(sigma_w1)
    sigma_w2 = np.asarray(sigma_w2)

    if not (w1_mag.shape == w1_cov.shape == w2_mag.shape == w2_cov.shape == sigma_w1.shape == sigma_w2.shape):
        raise ValueError("All inputs must have the same shape (N,).")

    X = np.column_stack([w1_mag, w1_cov, w2_mag, w2_cov]).astype(np.float32, copy=False)
    Y = np.column_stack([sigma_w1, sigma_w2]).astype(np.float32, copy=False)

    if drop_nonfinite:
        good = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X = X[good]
        Y = Y[good]

    if X.shape[0] == 0:
        raise ValueError("No valid training points after filtering.")

    if mins is None:
        mins_arr = X.min(axis=0).astype(np.float32)
    else:
        mins_arr = np.array(mins, dtype=np.float32)

    if maxs is None:
        maxs_arr = X.max(axis=0).astype(np.float32)
    else:
        maxs_arr = np.array(maxs, dtype=np.float32)

    base_step_arr = np.array(base_step, dtype=np.float32)
    if np.any(base_step_arr <= 0):
        raise ValueError("All base_step values must be > 0.")

    levels: List[HashGridLevel] = []
    for l in range(n_levels):
        step_l = base_step_arr * (coarsen_factor ** l) # coarsen cell size at each level
        shape_l = _compute_shape(mins_arr, maxs_arr, step_l) # input: min and max along each data dim, step size
        # we get out the number of grids along each dimension

        # convert data X to integer coordinates, return single cell id for each point
        cell_ids = _points_to_cell_id(X, mins_arr, step_l, shape_l)
        unique_ids, offsets, members = _build_level(X, shape_l, cell_ids)

        levels.append(HashGridLevel(
            step=step_l.astype(np.float32),
            shape=shape_l.astype(np.int64),
            unique_cell_ids=unique_ids,
            offsets=offsets,
            members=members,
        ))

    return SigmaHashGrid4D(
        mins=mins_arr,
        maxs=maxs_arr,
        base_step=base_step_arr,
        levels=levels,
        Y=Y,
        rng_seed=rng_seed,
    )


def sample_sigmas_from_hashgrid_4d(
    model: SigmaHashGrid4D,
    w1_mag_sim: np.ndarray,
    w1_cov_sim: np.ndarray,
    w2_mag_sim: np.ndarray,
    w2_cov_sim: np.ndarray,
    *,
    batch_size: int = 2_000_000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast sampling of (sigma_w1, sigma_w2) for simulated points.

    Uses multi-level fallback: try finest grid first; unresolved queries fall back
    to coarser grids. Remaining unresolved (rare) fall back to global sampling.

    Returns
    -------
    sigma_w1_sim, sigma_w2_sim : arrays shape (M,)
    """
    if rng is None:
        rng = np.random.default_rng(model.rng_seed)

    w1_mag_sim = np.asarray(w1_mag_sim)
    w1_cov_sim = np.asarray(w1_cov_sim)
    w2_mag_sim = np.asarray(w2_mag_sim)
    w2_cov_sim = np.asarray(w2_cov_sim)

    if not (w1_mag_sim.shape == w1_cov_sim.shape == w2_mag_sim.shape == w2_cov_sim.shape):
        raise ValueError("All simulated inputs must have the same shape (M,).")

    M = w1_mag_sim.size
    out = np.empty((M, 2), dtype=np.float32)

    Xq_all = np.column_stack([w1_mag_sim, w1_cov_sim, w2_mag_sim, w2_cov_sim]).astype(np.float32, copy=False)

    # Batch to keep peak memory bounded
    for start in range(0, M, batch_size):
        end = min(M, start + batch_size)
        Xq = Xq_all[start:end]
        B = Xq.shape[0]

        unresolved = np.ones(B, dtype=bool)
        out_block = np.empty((B, 2), dtype=np.float32)

        # multi-level lookup
        for lvl in model.levels:
            if not np.any(unresolved):
                break

            idx_u = np.where(unresolved)[0]
            X_u = Xq[idx_u]

            q_cell = _points_to_cell_id(X_u, model.mins, lvl.step, lvl.shape)

            # membership via searchsorted (unique_cell_ids sorted)
            pos = np.searchsorted(lvl.unique_cell_ids, q_cell)
            hit = (pos < lvl.unique_cell_ids.size) & (lvl.unique_cell_ids[pos] == q_cell)

            if not np.any(hit):
                continue

            idx_hit = idx_u[hit]
            bucket = pos[hit].astype(np.int64)

            # bucket ranges
            start_off = lvl.offsets[bucket]
            end_off = lvl.offsets[bucket + 1]
            lengths = end_off - start_off  # (H,)

            # sample index within each bucket: start + floor(u * len)
            u = rng.random(idx_hit.size, dtype=np.float64)
            pick = start_off + np.floor(u * lengths).astype(np.int64)

            train_idx = lvl.members[pick]  # indices into model.Y
            out_block[idx_hit, :] = model.Y[train_idx, :]

            unresolved[idx_hit] = False

        # final fallback (global) for anything still unresolved
        if np.any(unresolved):
            idx_u = np.where(unresolved)[0]
            train_idx = rng.integers(0, model.Y.shape[0], size=idx_u.size, dtype=np.int64)
            out_block[idx_u, :] = model.Y[train_idx, :]

        out[start:end, :] = out_block

    return out[:, 0], out[:, 1]

import numpy as np
from typing import Dict, Optional, Tuple


def hashgrid_diagnostics(
    model,
    w1_mag_sim: np.ndarray,
    w1_cov_sim: np.ndarray,
    w2_mag_sim: np.ndarray,
    w2_cov_sim: np.ndarray,
    *,
    sample_size: int = 1_000_000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """
    Compute per-level hit rates for the multi-resolution hash grid.

    The diagnostic simulates the lookup procedure (finest -> coarser -> global fallback)
    on a random subset of your simulated points.

    Returns a dict with:
      - n_total: number of query points tested
      - per_level: list of dicts with hit counts/rates at each level
      - unresolved_after_levels: fraction not found in any level (would go to global fallback)
      - cumulative_found_rate: fraction found by or before each level
    """
    if rng is None:
        rng = np.random.default_rng(getattr(model, "rng_seed", None))

    w1_mag_sim = np.asarray(w1_mag_sim)
    w1_cov_sim = np.asarray(w1_cov_sim)
    w2_mag_sim = np.asarray(w2_mag_sim)
    w2_cov_sim = np.asarray(w2_cov_sim)

    if not (w1_mag_sim.shape == w1_cov_sim.shape == w2_mag_sim.shape == w2_cov_sim.shape):
        raise ValueError("All simulated inputs must have the same shape (M,).")

    M = w1_mag_sim.size
    n = min(sample_size, M)
    idx = rng.choice(M, size=n, replace=False) if n < M else np.arange(M)

    Xq = np.column_stack([w1_mag_sim[idx], w1_cov_sim[idx], w2_mag_sim[idx], w2_cov_sim[idx]]).astype(np.float32)

    unresolved = np.ones(n, dtype=bool)
    per_level = []
    cumulative_found = np.zeros(n, dtype=bool)

    for li, lvl in enumerate(model.levels):
        if not np.any(unresolved):
            per_level.append({
                "level": li,
                "step": tuple(float(s) for s in lvl.step),
                "hit_count": 0,
                "hit_rate": 0.0,
                "cumulative_found_rate": 1.0,
            })
            continue

        idx_u = np.where(unresolved)[0]
        q_cell = _points_to_cell_id(Xq[idx_u], model.mins, lvl.step, lvl.shape)

        pos = np.searchsorted(lvl.unique_cell_ids, q_cell)
        hit = (pos < lvl.unique_cell_ids.size) & (lvl.unique_cell_ids[pos] == q_cell)

        hit_count = int(hit.sum())
        hit_rate = hit_count / n

        idx_hit = idx_u[hit]
        unresolved[idx_hit] = False
        cumulative_found[idx_hit] = True

        per_level.append({
            "level": li,
            "step": tuple(float(s) for s in lvl.step),
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "cumulative_found_rate": float(cumulative_found.mean()),
        })

    unresolved_rate = float(unresolved.mean())

    return {
        "n_total": n,
        "per_level": per_level,
        "unresolved_after_levels": unresolved_rate,
        "overall_found_in_levels": 1.0 - unresolved_rate,
    }


def print_hashgrid_diagnostics(diag: Dict[str, object]) -> None:
    """
    Pretty-print diagnostics from hashgrid_diagnostics().
    """
    n = diag["n_total"]
    print(f"HashGrid diagnostics over n={n:,} query points")
    print("-" * 72)
    print(f"{'Level':>5}  {'Step (W1mag,W1cov,W2mag,W2cov)':<40}  {'Hit%':>6}  {'CumFound%':>9}")
    print("-" * 72)

    for row in diag["per_level"]:
        li = row["level"]
        step = row["step"]
        hit_pct = 100.0 * row["hit_rate"]
        cum_pct = 100.0 * row["cumulative_found_rate"]
        step_str = f"({step[0]:.3g},{step[1]:.3g},{step[2]:.3g},{step[3]:.3g})"
        print(f"{li:5d}  {step_str:<40}  {hit_pct:6.2f}  {cum_pct:9.2f}")

    print("-" * 72)
    print(f"Unresolved after all levels (global fallback): {100.0*diag['unresolved_after_levels']:.2f}%")
    print(f"Found within some level:                    {100.0*diag['overall_found_in_levels']:.2f}%")



def _summarise_sizes(sizes: np.ndarray) -> Dict[str, float]:
    sizes = np.asarray(sizes)
    if sizes.size == 0:
        return dict(n_cells=0, mean=np.nan, median=np.nan, p10=np.nan, p25=np.nan,
                    p75=np.nan, p90=np.nan, p95=np.nan, p99=np.nan, max=np.nan)
    q = np.percentile(sizes, [10, 25, 50, 75, 90, 95, 99])
    return {
        "n_cells": int(sizes.size),
        "mean": float(sizes.mean()),
        "median": float(q[2]),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(sizes.max()),
    }


def hashgrid_bucket_occupancy_stats(model) -> Dict[str, object]:
    """
    Training-side occupancy stats: bucket sizes across all occupied cells per level.
    """
    per_level = []
    for li, lvl in enumerate(model.levels):
        sizes = (lvl.offsets[1:] - lvl.offsets[:-1]).astype(np.int64)
        per_level.append({
            "level": li,
            "step": tuple(float(s) for s in lvl.step),
            "bucket_sizes": _summarise_sizes(sizes),
        })
    return {"per_level": per_level}


def hashgrid_query_bucket_occupancy_stats(
    model,
    w1_mag_sim: np.ndarray,
    w1_cov_sim: np.ndarray,
    w2_mag_sim: np.ndarray,
    w2_cov_sim: np.ndarray,
    *,
    sample_size: int = 1_000_000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """
    Query-side occupancy stats: for a subset of simulated points, compute the
    bucket size of the cell that each query would hit at each level (for hits only).

    This answers: "when my simulation hits a cell, how many training points are in it?"
    """
    if rng is None:
        rng = np.random.default_rng(getattr(model, "rng_seed", None))

    w1_mag_sim = np.asarray(w1_mag_sim)
    w1_cov_sim = np.asarray(w1_cov_sim)
    w2_mag_sim = np.asarray(w2_mag_sim)
    w2_cov_sim = np.asarray(w2_cov_sim)

    if not (w1_mag_sim.shape == w1_cov_sim.shape == w2_mag_sim.shape == w2_cov_sim.shape):
        raise ValueError("All simulated inputs must have the same shape (M,).")

    M = w1_mag_sim.size
    n = min(sample_size, M)
    idx = rng.choice(M, size=n, replace=False) if n < M else np.arange(M)

    Xq = np.column_stack([
        w1_mag_sim[idx],
        w1_cov_sim[idx],
        w2_mag_sim[idx],
        w2_cov_sim[idx],
    ]).astype(np.float32)

    per_level = []

    for li, lvl in enumerate(model.levels):
        q_cell = _points_to_cell_id(Xq, model.mins, lvl.step, lvl.shape)

        pos = np.searchsorted(lvl.unique_cell_ids, q_cell)
        hit = (pos < lvl.unique_cell_ids.size) & (lvl.unique_cell_ids[pos] == q_cell)

        if not np.any(hit):
            per_level.append({
                "level": li,
                "step": tuple(float(s) for s in lvl.step),
                "hit_count": 0,
                "hit_rate": 0.0,
                "hit_bucket_sizes": _summarise_sizes(np.array([], dtype=np.int64)),
            })
            continue

        bucket = pos[hit].astype(np.int64)
        sizes = (lvl.offsets[bucket + 1] - lvl.offsets[bucket]).astype(np.int64)

        per_level.append({
            "level": li,
            "step": tuple(float(s) for s in lvl.step),
            "hit_count": int(hit.sum()),
            "hit_rate": float(hit.mean()),
            "hit_bucket_sizes": _summarise_sizes(sizes),
        })

    return {"n_total": n, "per_level": per_level}


def print_hashgrid_occupancy_stats(stats: Dict[str, object], *, title: str) -> None:
    print(title)
    print("-" * 88)
    hdr = (
        f"{'Lvl':>3}  {'Step (W1mag,W1cov,W2mag,W2cov)':<36}  "
        f"{'Cells':>7}  {'Mean':>7}  {'Med':>7}  {'P90':>7}  {'P99':>7}  {'Max':>7}"
    )
    print(hdr)
    print("-" * 88)

    for row in stats["per_level"]:
        li = row["level"]
        step = row["step"]
        step_str = f"({step[0]:.3g},{step[1]:.3g},{step[2]:.3g},{step[3]:.3g})"

        # training-side format
        if "bucket_sizes" in row:
            s = row["bucket_sizes"]
            print(
                f"{li:3d}  {step_str:<36}  {s['n_cells']:7d}  {s['mean']:7.2f}  {s['median']:7.2f}  "
                f"{s['p90']:7.2f}  {s['p99']:7.2f}  {s['max']:7.0f}"
            )

        # query-side format
        if "hit_bucket_sizes" in row:
            s = row["hit_bucket_sizes"]
            hit_rate = 100.0 * row["hit_rate"]
            print(
                f"{li:3d}  {step_str:<36}  hits={row['hit_count']:7d} ({hit_rate:5.1f}%)  "
                f"mean={s['mean']:.2f}  med={s['median']:.2f}  p90={s['p90']:.2f}  p99={s['p99']:.2f}  max={s['max']:.0f}"
            )

    print("-" * 88)

# --- Decode linear cell_id to 4D integer coords (i0,i1,i2,i3) ---
def decode_cell_ids(cell_ids: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    cell_ids: (U,) int64
    shape: (4,) int64
    returns coords: (U,4) int64
    """
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    shape = np.asarray(shape, dtype=np.int64)
    n1, n2, n3 = shape[1], shape[2], shape[3]
    coords = np.empty((cell_ids.size, 4), dtype=np.int64)

    # Reverse of: (((i0*n1 + i1)*n2 + i2)*n3 + i3)
    coords[:, 3] = cell_ids % n3
    tmp = cell_ids // n3
    coords[:, 2] = tmp % n2
    tmp = tmp // n2
    coords[:, 1] = tmp % n1
    coords[:, 0] = tmp // n1
    return coords

def cell_centers_from_coords(coords: np.ndarray, mins: np.ndarray, step: np.ndarray) -> np.ndarray:
    """
    coords: (U,4) int64, mins/step: (4,)
    returns centers: (U,4) float
    """
    return mins + (coords + 0.5) * step

def level_bucket_sizes(level) -> np.ndarray:
    """Return bucket sizes per occupied cell at this level."""
    return (level.offsets[1:] - level.offsets[:-1]).astype(np.int64)

def plot_hashgrid_projection(
    model,
    level_index: int,
    dims=(0, 2),
    *,
    slice_dims=None,
    slice_ranges=None,
    min_occupancy: int = 1,
    max_cells: int = 200_000,
    log_color: bool = True,
    ax=None,
    s=6,
    alpha=0.6,
):
    """
    Plot occupied cells of a given level projected to a 2D plane.

    Parameters
    ----------
    dims: tuple of two ints in [0..3]
        Which axes to plot, e.g. (0,2) for (W1mag, W2mag).
        Axis meaning depends on how you built the model.
        In your original 4D: 0=W1mag,1=W1cov,2=W2mag,3=W2cov.
        If you reparameterize, update labels accordingly.
    slice_dims: tuple of remaining dims to slice on (optional)
        e.g. slice_dims=(1,3) and slice_ranges=[(1.5,2.0),(1.5,2.0)]
    slice_ranges: list of (low, high) for each slice dim (same length as slice_dims)
    min_occupancy: filter out cells with fewer than this many occupants
    max_cells: cap number of plotted cells (randomly subsample) for performance
    """
    if ax is None:
        fig, ax = plt.subplots()

    lvl = model.levels[level_index]
    sizes = level_bucket_sizes(lvl)
    keep = sizes >= min_occupancy

    cell_ids = lvl.unique_cell_ids[keep]
    sizes = sizes[keep]

    coords = decode_cell_ids(cell_ids, lvl.shape)
    centers = cell_centers_from_coords(coords, model.mins, lvl.step)

    # Optional slicing on other dimensions
    if slice_dims is not None and slice_ranges is not None:
        slice_dims = tuple(slice_dims)
        if len(slice_dims) != len(slice_ranges):
            raise ValueError("slice_dims and slice_ranges must have the same length.")
        mask = np.ones(centers.shape[0], dtype=bool)
        for d, (lo, hi) in zip(slice_dims, slice_ranges):
            mask &= (centers[:, d] >= lo) & (centers[:, d] <= hi)
        centers = centers[mask]
        sizes = sizes[mask]

    # Subsample if needed
    if centers.shape[0] > max_cells:
        rng = np.random.default_rng(0)
        idx = rng.choice(centers.shape[0], size=max_cells, replace=False)
        centers = centers[idx]
        sizes = sizes[idx]

    x = centers[:, dims[0]]
    y = centers[:, dims[1]]

    c = np.log10(sizes) if log_color else sizes
    sc = ax.scatter(x, y, c=c, s=s, alpha=alpha)

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("log10(occupants)" if log_color else "occupants")

    ax.set_xlabel(f"dim {dims[0]}")
    ax.set_ylabel(f"dim {dims[1]}")
    ax.set_title(f"Level {level_index} occupied cells (projection {dims})")

    return ax


if __name__ == '__main__':
    config = CatwiseConfig(
        cat_w12_min=0.5, 
        cat_w1_max=17.0, 
        magnitude_error_dist='gaussian'
    )
    sim = Catwise(config)

    sim.load_catalogue()
    sim.determine_masked_pixels()
    sim.make_masked_catalogue()

    N_1D_BINS = 200
    magnitude_bins = np.linspace(
        np.min(sim.masked_catalogue['w1']), sim.cat_w1_max, N_1D_BINS
    )
    coverage_bins = np.linspace(1.5, 4., N_1D_BINS)

    cat = sim.masked_catalogue
    w1mag = cat['w1']; w2mag = cat['w2']
    w1cov = np.log10(cat['w1cov']); w2cov = np.log10(cat['w2cov'])
    w1sigma = cat['w1e']; w2sigma = cat['w2e']
    covmean = w1cov; covdelta = w2cov
    # covmean = 0.5 * (w1cov + w2cov)
    # covdelta = w1cov - w2cov
    
    # plt.figure()
    # plt.hist2d(w1cov, w2cov, bins=100, norm='log')
    # plt.figure()
    # plt.hist2d(covmean, covdelta, bins=100, norm='log')
    # plt.show()
    # sys.exit()
    print('Building data structure...')
    model = build_sigma_hashgrid_4d(
        w1mag, covmean,
        w2mag, covdelta,
        w1sigma, w2sigma,
        base_step=(0.1, 0.1, 0.1, 0.1),
        n_levels=3,
        coarsen_factor=2.0,
        rng_seed=123,
    )
    print('Finished building data structure.')

    N_SOURCES = len(w1mag)
    w1mag_sim = np.random.choice(w1mag) + np.random.normal(scale=0.001, size=(N_SOURCES,))
    w2mag_sim = np.random.choice(w2mag) + np.random.normal(scale=0.001, size=(N_SOURCES,))
    # w1cov_sim = w1cov
    # w2cov_sim = w2cov
    # w1mag_sim = 16.5 * np.ones(shape=(10_000,))
    # w2mag_sim = 16.3 * np.ones(shape=(10_000,))
    w1cov_sim = np.random.choice(covmean, size=(N_SOURCES,))
    w2cov_sim = np.random.choice(covdelta, size=(N_SOURCES,))
    # w1cov_sim = 1.9  * np.ones(shape=(10_000,))
    # w2cov_sim = 1.9  * np.ones(shape=(10_000,))

    t0 = time.time()
    sig_w1_sim, sig_w2_sim = sample_sigmas_from_hashgrid_4d(
        model,
        w1mag_sim, w1cov_sim,
        w2mag_sim, w2cov_sim,
        batch_size=5_000_000,
    )
    t1 = time.time()
    print(t1-t0)

    sig_w1_sim = sig_w1_sim # + np.random.normal(scale=0.001, size=(N_SOURCES,))
    sig_w2_sim = sig_w2_sim # + np.random.normal(scale=0.001, size=(N_SOURCES,))

    plt.scatter(w1sigma, w2sigma, s=0.1, alpha=0.3)
    plt.scatter(sig_w1_sim, sig_w2_sim)
    # plt.scatter(sig_w1_sim, sig_w2_sim, s=0.1, alpha=0.3)
    # plt.scatter(w1sigma, w2sigma, s=0.1, alpha=0.3)
    plt.show()
    
    plot_hashgrid_projection(model, level_index=0, dims=(0,2))  # W1mag vs W2mag
    plt.show()

    diagnostics = hashgrid_diagnostics(
        model,
        w1mag_sim, w1cov_sim,
        w2mag_sim, w2cov_sim,
    )
    print_hashgrid_diagnostics(diagnostics)

    train_stats = hashgrid_bucket_occupancy_stats(model)
    print_hashgrid_occupancy_stats(train_stats, title="Training-side bucket occupancy (all occupied cells)")
