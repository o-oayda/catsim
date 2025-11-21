import numpy as np
from numpy.typing import NDArray
from .physics import omega_to_theta
import healpy as hp


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
