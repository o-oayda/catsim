from collections import defaultdict
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray


def downgrade_ignore_nan(
    map_in: NDArray,
    mask_in: NDArray,
    nside_out: int
) -> tuple[NDArray, NDArray]:
    """Downgrade NEST-ordered maps/masks, ignoring masked pixels.

    Args:
        map_in: Array of shape ``(npix,)`` or ``(batch, npix)``.
        mask_in: Boolean array matching ``map_in``; ``False`` marks masked pixels.
        nside_out: Target HEALPix nside (must be a power-of-two divisor).
    Returns:
        Tuple ``(map_out, mask_out)`` matching the input batch shape.
    """

    map_arr = np.asarray(map_in)
    mask_arr = np.asarray(mask_in)

    squeezed_input = map_arr.ndim == 1

    if map_arr.ndim == 2 and mask_arr.ndim == 1:
        assert mask_arr.size == map_arr.shape[1]
        mask_arr = np.broadcast_to(mask_arr, map_arr.shape)

    if squeezed_input and mask_arr.ndim == 2:
        assert mask_arr.shape[0] == 1 and mask_arr.shape[1] == map_arr.size
        mask_arr = mask_arr[0]

    assert map_arr.shape == mask_arr.shape

    if squeezed_input:
        map_arr = map_arr[None, :]
        mask_arr = mask_arr[None, :]

    assert map_arr.ndim == 2, 'map_in must be 1D or 2D (batch, npix).'

    orig_dtype = map_arr.dtype
    work_map = map_arr.astype(np.float64, copy=False)
    work_mask = mask_arr.astype(bool, copy=False)

    work_map = np.where(work_mask, work_map, np.nan)

    npix = work_map.shape[-1]
    cur_nside = hp.npix2nside(npix)
    assert cur_nside >= nside_out, 'Output nside must be lower than input nside.'
    ratio = cur_nside // nside_out
    assert (cur_nside % nside_out) == 0 and (ratio & (ratio - 1)) == 0

    out_map = work_map
    out_mask = work_mask

    while cur_nside > nside_out:
        cur_npix = hp.nside2npix(cur_nside)
        cur_npix //= 4

        out_map = out_map.reshape(out_map.shape[0], cur_npix, 4)
        out_mask = out_mask.reshape(out_mask.shape[0], cur_npix, 4)

        valid_counts = out_mask.sum(axis=-1)
        summed = np.where(out_mask, out_map, 0.0).sum(axis=-1)

        out_mask = valid_counts > 0
        out_map = summed
        out_map[~out_mask] = np.nan

        cur_nside //= 2

    if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
        out_map = out_map.astype(orig_dtype, copy=False)

    if squeezed_input:
        out_map = out_map[0]
        out_mask = out_mask[0]

    return out_map, out_mask


class ParameterMap:
    def __init__(self,
            pixel_indices: NDArray[np.int_],
            parameter: NDArray[np.float64],
            nside: int
    ) -> None:
        self.parameter_dict = defaultdict(list)
        self.nside = nside
        for idx, pix in enumerate(pixel_indices):
            self.parameter_dict[int(pix)].append( parameter[idx] )
        self.median_map = None

    def get_map(self) -> NDArray[np.float64]:
        if self.median_map is None:
            n_pix = hp.nside2npix(self.nside)
            self.median_map = np.full(n_pix, np.nan)
            
            for pix, parameter_values in self.parameter_dict.items():
                if len(parameter_values) > 0:
                    self.median_map[pix] = np.median(parameter_values)
        
        return self.median_map


class Mask:
    def __init__(self, nside: int = 32):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.all_pixel_indices = set(np.arange(self.npix))

    def equator_mask(self, mask_angle: float) -> list:
        south_pole_vec = hp.ang2vec(0, -90, lonlat=True)
        north_pole_vec = hp.ang2vec(0, 90, lonlat=True)
        north_pole_indices = hp.query_disc(
            self.nside, north_pole_vec, radius=np.deg2rad(90 - mask_angle),
            nest=True
        )
        south_pole_indices = hp.query_disc(
            self.nside, south_pole_vec, radius=np.deg2rad(90 - mask_angle),
            nest=True
        )
        masked_pixel_indices = (
            self.all_pixel_indices
            - set([*north_pole_indices, *south_pole_indices])
        )
        return list(masked_pixel_indices)
    
    def catwise_mask(self) -> list:
        '''
        Return CatWISE2020 mask used in Secrest et al. (2021) in Galactic
        coordinates, with `nside=64` and in nest ordering.
        '''
        galactic_mask = hp.reorder(
            np.load('dipolesbi/catwise/CatWISE_Mask_nside64.npy'),
            r2n=True
        )
        masked_pixel_indices = list(np.where(galactic_mask == 0)[0])
        return masked_pixel_indices
    
    def north_ecliptic_mask(self) -> list:
        ecl_north_pole = SkyCoord(
            lon=0 * u.deg,  # type: ignore
            lat=90 * u.deg, # type: ignore
            frame='geocentrictrueecliptic'
        )
        gal_north_pole = ecl_north_pole.transform_to('galactic')
        pole_colatitude = np.deg2rad(90 - gal_north_pole.b.deg)  # type: ignore
        pole_longitude = np.deg2rad(gal_north_pole.l.deg)        # type: ignore
        vec_north_ecl_gal = hp.ang2vec(pole_colatitude, pole_longitude)

        north_pole_disc_pixels = hp.query_disc(
            nside=self.nside,
            vec=vec_north_ecl_gal,
            radius=np.deg2rad(5), # hardcoded 5 degrees, see error_sampling_gp.py
            nest=True
        )
        return list(north_pole_disc_pixels)
