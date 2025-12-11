import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.patches import Rectangle


def binned_distribution_2d(x, y, values=None, bins=10, data_range=None,
                           expand_binnumbers=False):
    """
    Group values into 2D bins and return the full distribution in each bin.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Coordinates of the points.
    values : array_like, shape (N,), optional
        Values to be binned. If None, `values` defaults to `x`.
    bins : int or [int, int] or [array_like, array_like], optional
        - If int: the number of bins for both x and y, or
        - If [int, int]: the number of bins in each dimension, or
        - If [x_edges, y_edges]: the bin edges directly.
        Semantics match ``numpy.histogram2d`` / ``scipy.stats.binned_statistic_2d``.
    range : (2,2) array_like, optional
        Lower and upper range of the bins [[xmin, xmax], [ymin, ymax]].
        Used only when `bins` is an int or [int, int].
    expand_binnumbers : bool, optional
        If True, return `binnumber` with shape (2, N) giving the (x_bin, y_bin)
        for each point (1-based, 0 for out-of-range), similar to
        `binned_statistic_2d`. If False, return a flattened 1D binnumber.

    Returns
    -------
    distribution : ndarray of shape (nx, ny), dtype=object
        Each element distribution[i, j] is a 1D ndarray containing all
        `values` that fell into that 2D bin.
    xedges : ndarray, shape (nx+1,)
        The bin edges along x.
    yedges : ndarray, shape (ny+1,)
        The bin edges along y.
    binnumber : ndarray
        If expand_binnumbers is False: shape (N,), 1D bin index (1-based)
        or 0 for out-of-range.
        If True: shape (2, N), giving (x_bin, y_bin) per sample, each 1-based
        or 0 for out-of-range.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if values is None:
        values = x
    values = np.asarray(values)

    if x.shape != y.shape or x.shape != values.shape:
        raise ValueError("x, y, and values must all have the same shape 1D arrays.")

    # Determine bin edges
    if np.isscalar(bins):
        # Same number of bins in x and y
        if data_range is None:
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
        else:
            (xmin, xmax), (ymin, ymax) = data_range

        xedges = np.linspace(xmin, xmax, int(bins) + 1)
        yedges = np.linspace(ymin, ymax, int(bins) + 1)

    elif len(np.atleast_1d(bins)) == 2:
        bx, by = bins
        bx = np.atleast_1d(bx)
        by = np.atleast_1d(by)

        if np.isscalar(bx) and np.isscalar(by):
            # bins = [nx, ny]
            nx, ny = int(bx), int(by)
            if data_range is None:
                xmin, xmax = x.min(), x.max()
                ymin, ymax = y.min(), y.max()
            else:
                (xmin, xmax), (ymin, ymax) = data_range

            xedges = np.linspace(xmin, xmax, nx + 1)
            yedges = np.linspace(ymin, ymax, ny + 1)
        else:
            # bins = [xedges, yedges]
            xedges = np.asarray(bx)
            yedges = np.asarray(by)
            if xedges.ndim != 1 or yedges.ndim != 1:
                raise ValueError("Bin edges must be 1D arrays.")
    else:
        raise ValueError(
            "bins must be an int, [int, int], or [x_edges, y_edges]."
        )

    nx = len(xedges) - 1
    ny = len(yedges) - 1

    # Digitize to get 1-based bin indices (like np.histogram)
    xi = np.digitize(x, xedges)
    yi = np.digitize(y, yedges)

    # Points are in-range if their bin index is within [1, nx] and [1, ny]
    in_x = (xi >= 1) & (xi <= nx)
    in_y = (yi >= 1) & (yi <= ny)
    in_range = in_x & in_y

    # Prepare distribution array of lists
    distribution = np.empty((nx, ny), dtype=object)
    for i in range(nx):
        for j in range(ny):
            distribution[i, j] = []

    # Populate per-bin distributions
    for k in range(x.size):
        if not in_range[k]:
            continue
        i = xi[k] - 1  # convert to 0-based index
        j = yi[k] - 1
        distribution[i, j].append(values[k])

    # Convert inner lists to ndarrays
    for i in range(nx):
        for j in range(ny):
            distribution[i, j] = np.asarray(distribution[i, j])

    # Build binnumber output like scipy
    xbin = xi.copy()
    ybin = yi.copy()

    # Mark out-of-range as 0
    xbin[~in_x] = 0
    ybin[~in_y] = 0

    if expand_binnumbers:
        binnumber = np.vstack((xbin, ybin))
    else:
        # Flattened bin index (1..nx*ny), or 0 if any dimension is out-of-range
        flat = np.zeros_like(xbin, dtype=int)
        in_both = in_x & in_y
        # Only combine where both in range
        flat[in_both] = (xbin[in_both] - 1) * ny + (ybin[in_both] - 1) + 1
        binnumber = flat

    return distribution, xedges, yedges, binnumber


def plot_top_bin_histograms(dist, top_k=20, bins=30):
    """
    dist : ndarray (nx, ny) of object arrays (each entry is a 1D array of values)
    top_k : number of bins with largest counts to inspect
    bins : number of histogram bins for plotting
    """

    nx, ny = dist.shape
    flat = dist.ravel()

    # Compute sizes
    sizes = np.array([len(v) for v in flat])

    # Top nonempty bins
    top_idx = np.argsort(sizes)[::-1][:top_k]

    for rank, idx in enumerate(top_idx, 1):
        vals = flat[idx]
        if len(vals) == 0:
            continue

        # Recover 2D bin index
        i = idx // ny
        j = idx % ny

        print(f"Rank {rank}: bin (i={i}, j={j}), size={len(vals)}")

        # Plot
        plt.figure()
        plt.hist(np.log(vals), bins=bins,)
        plt.axvline(np.log(np.mean(vals)), linestyle='--', label='Mean', color='tab:orange')
        plt.axvline(np.log(np.median(vals)), linestyle='--', label='Median', color='tab:red')
        plt.title(f"Bin (i={i}, j={j}) — size={len(vals)}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


def adaptive_binned_distribution_2d(
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

    if x.shape != y.shape:# or x.shape != values.shape:
        raise ValueError("x, y, and values must all have the same 1D shape")

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


def plot_adaptive_bins(
    bin_bounds,
    x=None,
    y=None,
    ax=None,
    show_counts=False,
    bin_values=None,
    **rect_kwargs,
):
    """
    Overlay adaptive square-ish bins as rectangles in (x, y) space.

    Parameters
    ----------
    bin_bounds : ndarray, shape (n_bins, 4)
        [xmin, xmax, ymin, ymax] for each bin.
    x, y : array_like, optional
        Optional scatter of underlying data points for context.
    ax : matplotlib Axes, optional
        Axes to draw on; if None, a new figure+axes is created.
    show_counts : bool
        If True, annotate each bin with the number of points
        (requires bin_values).
    bin_values : list of ndarrays, optional
        Same length as bin_bounds; if show_counts is True, used to
        get counts.
    rect_kwargs :
        Extra keyword args forwarded to matplotlib.patches.Rectangle.

    Returns
    -------
    ax : matplotlib Axes
        The axes with the overlay.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Optional scatter
    if x is not None and y is not None:
        ax.scatter(x, y, s=2, alpha=0.3)

    rect_defaults = dict(fill=False, linewidth=0.7, alpha=0.9)
    rect_defaults.update(rect_kwargs)

    for i, (xmin, xmax, ymin, ymax) in enumerate(bin_bounds):
        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            **rect_defaults,
        )
        ax.add_patch(rect)

        if show_counts and bin_values is not None:
            count = len(bin_values[i]) if i < len(bin_values) else 0
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            ax.text(
                cx,
                cy,
                str(count),
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax

