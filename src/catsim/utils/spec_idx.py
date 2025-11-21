import os
from typing import Optional
from astropy.table import Table
import numpy as np
from scipy.constants import speed_of_light
from numpy.typing import NDArray


class AlphaLookup:
    '''
    Adapted from lookup_alpha_catwise.py from Secrest+21.
    '''
    def __init__(self, no_check: bool = False) -> None:
        self.no_check = no_check

        # load only smaller-size version of table if not needed
        if self.no_check:
            self.lookup_table_path = 'dipolesbi/catwise/alpha_w12_only.fits'
        else:
            self.lookup_table_path = 'dipolesbi/catwise/alpha_colors.fits'

        assert os.path.exists(self.lookup_table_path), 'Cannot find lookup tab.'
        self._load_lookup_data()
        self._extrapolate_colour_alpha_relation()

        self.AB_VEGA_OFFSET = 2.673
        self.SPEED_OF_LIGHT_ANGSTROMS_S = speed_of_light * 1e10

    def _extrapolate_colour_alpha_relation(self, order: int = 5) -> None:
        '''
        With a degree 5 polynomial fit, we are anticipating errors in alpha
        of order 10**-4.
        '''
        polynomial_fit = np.polyfit(self.lookup_W1_W2, self.lookup_alpha, deg=order)
        self.p_W12 = np.poly1d(polynomial_fit)

    def fit_alpha(
            self,
            w12_colour: NDArray,
            out: Optional[NDArray[np.float32]] = None
        ) -> NDArray[np.float32]:
        coeffs = self.p_W12.c.astype(np.float32)
        colour = np.asarray(w12_colour, dtype=np.float32)

        if out is None:
            result = np.empty_like(colour, dtype=np.float32)
        else:
            if out.shape != colour.shape:
                raise ValueError('Output buffer must match input shape.')
            result = out

        # Horner's method
        result[:] = coeffs[0] # highest order coefficient
        for coefficient in coeffs[1:]:
            np.multiply(result, colour, out=result)
            result += coefficient

        return result
    
    def _load_lookup_data(self):
        self.lookup_tab = Table.read(self.lookup_table_path)
        self.lookup_alpha = self.lookup_tab['alpha'].data.astype('float32')
        self.lookup_W1_W2 = self.lookup_tab['W1_W2'].data.astype('float32')

        if not self.no_check:
            self.lookup_k_W1 = self.lookup_tab['k_W1'].data.astype('float32')
            self.lookup_nu_W1_iso = self.lookup_tab['nu_W1_iso'].data.astype('float32')

        del self.lookup_tab
