from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray
from .physics import omega_to_theta
import healpy as hp
import matplotlib.pyplot as plt


def average_smooth_map(
        healpy_map: NDArray[np.floating],
        weights: NDArray[np.floating] | None = None, 
        angle_scale: float = 1.
    ) -> NDArray:
    '''
    Smooth a healpy map using a moving average.
    '''
    included_pixels = np.where(~np.isnan(healpy_map))[0]
    smoothed_map = np.nan * np.empty_like(healpy_map)
    nside = hp.get_nside(healpy_map)
    
    if weights is None:
        weights = np.ones_like(healpy_map)

    smoothing_radius = omega_to_theta(angle_scale)
    for p_index in included_pixels:
        vec = hp.pix2vec(nside, p_index, nest=True)
        disc = hp.query_disc(nside, vec, smoothing_radius, nest=True)
        smoothed_map[p_index] = np.nanmean(healpy_map[disc] * weights[disc])

    return smoothed_map

def smooth_map(
        healpy_map: NDArray,
        weights: NDArray | None = None,
        angle_scale: float = 1.,
        only_return_data: bool = False,
        fig = None, 
        **kwargs
    ) -> NDArray | None:
    smoothed_map_to_plot = average_smooth_map(
        healpy_map,
        weights=weights,
        angle_scale=angle_scale
    )

    if only_return_data:
        return smoothed_map_to_plot

    hp.projview(
        smoothed_map_to_plot,
        nest=True,
        fig=fig.number if fig is not None else None,
        **kwargs
    )
    return None

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
