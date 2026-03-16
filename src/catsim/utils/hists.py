import os
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class MultinomialSample2DHistogram:
    """
    Fast 2D histogram sampling using multinomial distribution.
    
    This approach treats the 2D histogram as a single multinomial distribution
    over all bins, allowing for very fast sampling by directly using np.random.choice.
    Expected to be ~10-20x faster than the conditional CDF approach.
    """
    
    def __init__(self) -> None:
        pass

    def build(self,
            x_data = None,
            y_data = None,
            **hist_kwargs
        ) -> None:
        """
        Build the multinomial sampler from data with jittering support.
        
        Parameters:
        -----------
        x_data : array-like
            X coordinates of data points
        y_data : array-like  
            Y coordinates of data points
        **hist_kwargs : dict
            Additional arguments passed to np.histogram2d
        """
        
        # Create 2D histogram
        counts_2d, self.x_edges, self.y_edges = np.histogram2d(
            x_data, y_data, **hist_kwargs # pyright: ignore[reportArgumentType, reportCallIssue]
        )
        
        # Calculate bin centers
        x_centres = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        y_centres = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        
        # Calculate bin widths for jittering
        self.x_bin_widths = np.diff(self.x_edges)
        self.y_bin_widths = np.diff(self.y_edges)
        
        # Create 2D coordinate grids for centers and widths
        self.x_centres_2d, self.y_centres_2d = np.meshgrid(
            x_centres, y_centres, indexing='ij'
        )
        x_widths_2d, y_widths_2d = np.meshgrid(
            self.x_bin_widths, self.y_bin_widths, indexing='ij'
        )
        
        # Flatten coordinate grids for multinomial sampling
        self.x_flat = self.x_centres_2d.flatten()
        self.y_flat = self.y_centres_2d.flatten()
        self.x_widths_flat = x_widths_2d.flatten()
        self.y_widths_flat = y_widths_2d.flatten()
        
        # Flatten counts and normalize to probabilities
        counts_flat = counts_2d.flatten()
        self.probs_flat = counts_flat / np.sum(counts_flat)
        
        # Store original shape for potential debugging
        self.original_shape = counts_2d.shape
        
        # Filter out zero-probability bins for efficiency (optional)
        nonzero_mask = self.probs_flat > 0
        if np.sum(nonzero_mask) < len(self.probs_flat):
            self.x_flat = self.x_flat[nonzero_mask]
            self.y_flat = self.y_flat[nonzero_mask]
            self.x_widths_flat = self.x_widths_flat[nonzero_mask]
            self.y_widths_flat = self.y_widths_flat[nonzero_mask]
            self.probs_flat = self.probs_flat[nonzero_mask]
            # Renormalize after filtering
            self.probs_flat = self.probs_flat / np.sum(self.probs_flat)
        
        print(f"MultinomialSample2DHistogram built with {len(self.probs_flat)} active bins")

    def save_data(
            self, 
            save_dir: os.PathLike[str] | str, 
            fname_append: Optional[str] = None
    ) -> None:
        """Save the multinomial sampler data."""
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        
        sampler_data = {
            'x_flat': self.x_flat,
            'y_flat': self.y_flat,
            'x_widths_flat': self.x_widths_flat,
            'y_widths_flat': self.y_widths_flat,
            'probs_flat': self.probs_flat,
            'x_edges': self.x_edges,
            'y_edges': self.y_edges,
            'original_shape': self.original_shape
        }

        if fname_append:
            sampler_file = save_dir_path / f'multinomial_sampler_data{fname_append}.pkl'
        else:
            sampler_file = save_dir_path / 'multinomial_sampler_data.pkl'

        with sampler_file.open('wb') as handle:
            pickle.dump(sampler_data, handle)

    def load_data(
            self, 
            save_dir: os.PathLike[str] | str,
            fname_append: Optional[str] = None
    ) -> None:
        """Load the multinomial sampler data."""
        if fname_append:
            sampler_file = (
                Path(save_dir) / f'multinomial_sampler_data{fname_append}.pkl'
            )
        else:
            sampler_file = Path(save_dir) / 'multinomial_sampler_data.pkl'

        with sampler_file.open('rb') as handle:
            sampler_data = pickle.load(handle)
        
        self.x_flat = sampler_data['x_flat']
        self.y_flat = sampler_data['y_flat']
        self.x_widths_flat = sampler_data['x_widths_flat']
        self.y_widths_flat = sampler_data['y_widths_flat']
        self.probs_flat = sampler_data['probs_flat']
        self.x_edges = sampler_data['x_edges']
        self.y_edges = sampler_data['y_edges']
        self.original_shape = sampler_data['original_shape']

    def sample(
            self,
            n_samples: int,
            rng: Optional[np.random.Generator] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Sample from the 2D distribution using multinomial sampling with uniform jittering.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        x_samples : NDArray[np.float64]
            X coordinates of samples with uniform jittering within bins
        y_samples : NDArray[np.float64]
            Y coordinates of samples with uniform jittering within bins
        """
        if rng is None:
            rng = np.random.default_rng()

        # Multinomial sampling to select bins
        indices = rng.choice(
            len(self.probs_flat), 
            size=n_samples, 
            p=self.probs_flat
        )
        
        # Get bin centers and widths for selected bins
        x_centers = self.x_flat[indices]
        y_centers = self.y_flat[indices]
        x_widths = self.x_widths_flat[indices]
        y_widths = self.y_widths_flat[indices]
        
        # Add uniform jitter within each bin
        # Jitter is uniform in [-width/2, +width/2] around bin center
        x_jitter = rng.uniform(-0.5, 0.5, n_samples) * x_widths
        y_jitter = rng.uniform(-0.5, 0.5, n_samples) * y_widths
        
        # Apply jittering to get continuous samples
        x_samples = x_centers + x_jitter
        y_samples = y_centers + y_jitter
        
        return x_samples, y_samples
    
    def get_bin_info(self) -> dict:
        """
        Get information about the binning for debugging/analysis.
        
        Returns:
        --------
        info : dict
            Dictionary containing bin information
        """
        return {
            'n_bins_total': len(self.x_flat),
            'x_range': (self.x_edges[0], self.x_edges[-1]),
            'y_range': (self.y_edges[0], self.y_edges[-1]),
            'original_shape': self.original_shape,
            'min_probability': np.min(self.probs_flat),
            'max_probability': np.max(self.probs_flat)
        }

def kdtree_binned_distribution_2d(
    x,
    y,
    values=None,
    target_count=200,
    max_factor=4.0,
    min_count=10,
    max_depth=20,
):
    """
    Build adaptive, roughly square 2D bins using a kd-tree style split.

    Each bin is an axis-aligned rectangle in (x, y) with:
      - ~target_count points (up to max_factor * target_count)
      - roughly balanced x/y side lengths.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Coordinates of points (e.g. magnitude and coverage).
    values : array_like, shape (N,), optional
        Values to store per bin (e.g. photometric uncertainties).
        If None, uses x.
    target_count : int
        Desired number of points per bin.
    max_factor : float
        Split nodes while n_points > max_factor * target_count.
        Larger values allow bigger bins.
    min_count : int
        Do not split a node if it would create children smaller than this.
    max_depth : int
        Safety limit on recursion depth.

    Returns
    -------
    bin_bounds : ndarray, shape (n_bins, 4)
        Each row is [xmin, xmax, ymin, ymax] for a bin.
    bin_values : list of 1D ndarrays
        bin_values[k] contains all `values` belonging to bin k.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if values is None:
        values = x
    values = np.asarray(values)

    if x.shape != y.shape:
        raise ValueError("x and y values must all have the same 1D shape")

    n = x.size
    if n == 0:
        raise ValueError("No points provided")

    # Normalised coordinates for deciding split axis (aspect ratio)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    dx = x_max - x_min if x_max > x_min else 1.0
    dy = y_max - y_min if y_max > y_min else 1.0

    bins_bounds = []
    bins_vals = []

    def build_node(idx, xmin, xmax, ymin, ymax, depth):
        nonlocal bins_bounds, bins_vals

        n_node = idx.size
        if n_node == 0:
            return

        # Stopping criteria
        if (
            n_node <= max(target_count, min_count)
            or n_node <= target_count * max_factor
            or depth >= max_depth
        ):
            bins_bounds.append([xmin, xmax, ymin, ymax])
            bins_vals.append(values[idx])
            return

        # Decide split axis based on *normalised* side lengths to aim for square-ish bins
        wx = (xmax - xmin) / dx
        wy = (ymax - ymin) / dy
        split_axis = 0 if wx >= wy else 1  # 0 -> split in x, 1 -> split in y

        if split_axis == 0:
            # split along x at median x of the node
            x_node = x[idx]
            x_med = np.median(x_node)
            # avoid degenerate split
            if x_med <= xmin or x_med >= xmax:
                # cannot split meaningfully; make a leaf
                bins_bounds.append([xmin, xmax, ymin, ymax])
                bins_vals.append(values[idx])
                return
            left_mask = x_node <= x_med
            right_mask = ~left_mask
            idx_left = idx[left_mask]
            idx_right = idx[right_mask]
            if idx_left.size < min_count or idx_right.size < min_count:
                # splitting would create tiny children; stop
                bins_bounds.append([xmin, xmax, ymin, ymax])
                bins_vals.append(values[idx])
                return
            build_node(idx_left, xmin, x_med, ymin, ymax, depth + 1)
            build_node(idx_right, x_med, xmax, ymin, ymax, depth + 1)

        else:
            # split along y at median y of the node
            y_node = y[idx]
            y_med = np.median(y_node)
            if y_med <= ymin or y_med >= ymax:
                bins_bounds.append([xmin, xmax, ymin, ymax])
                bins_vals.append(values[idx])
                return
            bottom_mask = y_node <= y_med
            top_mask = ~bottom_mask
            idx_bottom = idx[bottom_mask]
            idx_top = idx[top_mask]
            if idx_bottom.size < min_count or idx_top.size < min_count:
                bins_bounds.append([xmin, xmax, ymin, ymax])
                bins_vals.append(values[idx])
                return
            build_node(idx_bottom, xmin, xmax, ymin, y_med, depth + 1)
            build_node(idx_top, xmin, xmax, y_med, ymax, depth + 1)

    # Kick off recursion with all points in the full bounding box
    all_idx = np.arange(n, dtype=int)
    build_node(all_idx, x_min, x_max, y_min, y_max, depth=0)

    bin_bounds = np.asarray(bins_bounds, dtype=float)
    return bin_bounds, bins_vals

def save_sigma_bins(filename, bin_bounds, bin_values):
    """
    Save adaptive sigma bins (bin_bounds, bin_values) to an .npz file.

    Supports both:
      - scalar values per bin: bin_values[k].shape == (n_k,)
      - vector values per bin: bin_values[k].shape == (n_k, K)

    Parameters
    ----------
    filename : str
        Path to output file, e.g. "sigma_bins.npz".
    bin_bounds : ndarray, shape (n_bins, 4)
        [xmin, xmax, ymin, ymax] per bin.
    bin_values : list of ndarrays
        bin_values[k] is a 1D or 2D array of values for bin k.
    """
    bin_bounds = np.asarray(bin_bounds, dtype=float)

    # Determine feature dimension from first non-empty bin
    non_empty = [np.asarray(v) for v in bin_values if len(v) > 0]
    if len(non_empty) == 0:
        # No values at all; default to scalar
        feature_dim = 1
    else:
        v0 = non_empty[0]
        if v0.ndim == 1:
            feature_dim = 1
        elif v0.ndim == 2:
            feature_dim = v0.shape[1]
        else:
            raise ValueError(
                "bin_values entries must be 1D or 2D arrays; "
                f"got ndim={v0.ndim}"
            )
        # Sanity-check all other non-empty bins have same feature_dim
        for v in non_empty[1:]:
            if v.ndim == 1 and feature_dim != 1:
                raise ValueError("Mixed 1D and 2D bin_values not supported.")
            if v.ndim == 2 and v.shape[1] != feature_dim:
                raise ValueError("Inconsistent feature dimension across bins.")

    # Per-bin lengths and offsets
    lengths = np.array([len(v) for v in bin_values], dtype=int)
    offsets = np.zeros(len(lengths) + 1, dtype=int)
    offsets[1:] = np.cumsum(lengths)
    total = offsets[-1]

    # Concatenate into one big array
    if total == 0:
        if feature_dim == 1:
            all_values = np.empty((0,), dtype=float)
        else:
            all_values = np.empty((0, feature_dim), dtype=float)
    else:
        if feature_dim == 1:
            arrays = [np.asarray(v, dtype=float).reshape(-1)
                      for v in bin_values]
            all_values = np.concatenate(arrays, axis=0)
        else:
            arrays = [np.asarray(v, dtype=float).reshape(-1, feature_dim)
                      for v in bin_values]
            all_values = np.concatenate(arrays, axis=0)

    np.savez_compressed(
        filename,
        bin_bounds=bin_bounds,
        all_values=all_values,
        offsets=offsets,
        feature_dim=np.array(feature_dim, dtype=int),
    )

def load_sigma_bins(filename):
    """
    Load adaptive sigma bins previously saved with save_sigma_bins.

    Returns bin_bounds and bin_values in the same structure that went in:
      - If feature_dim == 1: each bin_values[k] is 1D (n_k,)
      - If feature_dim > 1: each bin_values[k] is 2D (n_k, feature_dim)

    Parameters
    ----------
    filename : str
        Path to the .npz file.

    Returns
    -------
    bin_bounds : ndarray, shape (n_bins, 4)
    bin_values : list of ndarrays
    """
    data = np.load(filename, allow_pickle=False)
    bin_bounds = data["bin_bounds"]
    all_values = data["all_values"]
    offsets = data["offsets"]

    # Backwards compatibility: if feature_dim missing, infer
    if "feature_dim" in data:
        feature_dim = int(np.asarray(data["feature_dim"]))
    else:
        # Older files: infer from all_values shape
        feature_dim = 1 if all_values.ndim == 1 else all_values.shape[1]

    n_bins = len(offsets) - 1
    bin_values = []

    for i in range(n_bins):
        start = offsets[i]
        end = offsets[i + 1]
        if feature_dim == 1:
            vals = all_values[start:end]
        else:
            vals = all_values[start:end, :]
        bin_values.append(vals)

    return bin_bounds, bin_values


def build_bin_lookup_grid(bin_bounds):
    """
    Build a fast lookup grid for bin indices.

    Parameters
    ----------
    bin_bounds : ndarray, shape (n_bins, 4)
        [xmin, xmax, ymin, ymax] for each bin.

    Returns
    -------
    x_grid : ndarray, shape (nx+1,)
        Sorted unique x boundaries (global).
    y_grid : ndarray, shape (ny+1,)
        Sorted unique y boundaries (global).
    bin_index_grid : ndarray, shape (nx, ny), dtype=int
        bin_index_grid[ix, iy] = bin index k for the cell
        [x_grid[ix], x_grid[ix+1]) × [y_grid[iy], y_grid[iy+1])
        or -1 if no bin covers that cell (should not happen if bins partition the space).
    """
    bin_bounds = np.asarray(bin_bounds)
    xmin = bin_bounds[:, 0]
    xmax = bin_bounds[:, 1]
    ymin = bin_bounds[:, 2]
    ymax = bin_bounds[:, 3]

    # Global unique boundaries
    x_grid = np.unique(np.concatenate([xmin, xmax]))
    y_grid = np.unique(np.concatenate([ymin, ymax]))

    nx = x_grid.size - 1
    ny = y_grid.size - 1

    bin_index_grid = -np.ones((nx, ny), dtype=int)

    # Map exact boundary values to indices to avoid O(n_bins * nx) search
    x_to_ix = {val: i for i, val in enumerate(x_grid)}
    y_to_iy = {val: i for i, val in enumerate(y_grid)}

    for k, (x0, x1, y0, y1) in enumerate(bin_bounds):
        ix0 = x_to_ix[x0]
        ix1 = x_to_ix[x1]
        iy0 = y_to_iy[y0]
        iy1 = y_to_iy[y1]

        # In our kd-style partition, each bin should correspond to a single cell:
        # i.e., ix1 = ix0 + 1, iy1 = iy0 + 1.
        # But to be robust:
        for ix in range(ix0, ix1):
            for iy in range(iy0, iy1):
                bin_index_grid[ix, iy] = k

    return x_grid, y_grid, bin_index_grid


def sample_sigma_w1w2_from_bins_vectorized_fast(
    x_vals,
    y_vals,
    bin_bounds,
    bin_values,
    x_grid,
    y_grid,
    bin_index_grid,
    rng=None,
):
    """
    Sample (sigma_W1, sigma_W2) for arrays of simulated (W1_mag, W1_cov),
    using adaptive bins built on W1 and joint (sigma_W1, sigma_W2) values.

    Parameters
    ----------
    x_vals, y_vals : array_like, shape (M,)
        Simulated W1 magnitudes and coverages.
    bin_bounds : ndarray, shape (n_bins, 4)
        [xmin, xmax, ymin, ymax] for each adaptive bin (from
        adaptive_square_binned_distribution_2d).
    bin_values : list of ndarrays
        bin_values[k] has shape (n_k, 2) with columns [sigma_W1, sigma_W2].
    x_grid, y_grid : ndarrays
        Boundary arrays from build_bin_lookup_grid(bin_bounds).
    bin_index_grid : ndarray, shape (nx, ny)
        Bin indices grid from build_bin_lookup_grid.
    rng : numpy.random.Generator, optional
        RNG to use. If None, uses np.random.default_rng().

    Returns
    -------
    sigma_w1, sigma_w2 : ndarrays, shape (M,)
        Sampled uncertainties in W1 and W2.
    """
    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)
    if x_vals.shape != y_vals.shape:
        raise ValueError("x_vals and y_vals must have the same shape")

    if rng is None:
        rng = np.random.default_rng()

    M = x_vals.size
    sigma_w1 = np.empty(M, dtype=float)
    sigma_w2 = np.empty(M, dtype=float)

    # Map all points into grid indices in one shot
    ix = np.searchsorted(x_grid, x_vals, side="right") - 1
    iy = np.searchsorted(y_grid, y_vals, side="right") - 1

    # Clip to grid; or you can choose to raise instead if out-of-range
    ix = np.clip(ix, 0, bin_index_grid.shape[0] - 1)
    iy = np.clip(iy, 0, bin_index_grid.shape[1] - 1)

    bin_idx = bin_index_grid[ix, iy]

    # Precompute bin centers and non-empty bins for any fallbacks
    bb = np.asarray(bin_bounds)
    cx = 0.5 * (bb[:, 0] + bb[:, 1])
    cy = 0.5 * (bb[:, 2] + bb[:, 3])

    non_empty_bins = [i for i, v in enumerate(bin_values) if v.size > 0]
    if not non_empty_bins:
        raise RuntimeError("All bins are empty; cannot sample sigmas.")

    # Handle any -1 (should not happen if bins fully tile the space)
    bad = np.where(bin_idx < 0)[0]
    for i in bad:
        dx = x_vals[i] - cx
        dy = y_vals[i] - cy
        j = int(np.argmin(dx * dx + dy * dy))
        bin_idx[i] = j

    # Now sample per bin
    n_bins = len(bin_values)
    for k in range(n_bins):
        mask = (bin_idx == k)
        count = mask.sum()
        if count == 0:
            continue

        vals = bin_values[k]      # shape (n_k, 2)
        n_k = vals.shape[0]
        if n_k == 0:
            # Fallback: send these points to nearest non-empty bin
            # (rare if adaptive binning is set up sensibly)
            # Choose one representative non-empty bin (could be smarter if needed)
            k_ne = non_empty_bins[0]
            vals = bin_values[k_ne]
            n_k = vals.shape[0]

        j = rng.integers(0, n_k, size=count)
        sigma_w1[mask] = vals[j, 0]
        sigma_w2[mask] = vals[j, 1]

    return sigma_w1, sigma_w2
