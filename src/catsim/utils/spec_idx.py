from pathlib import Path
from typing import Optional
from astropy.table import Table
import numpy as np
from scipy.constants import speed_of_light
from numpy.typing import NDArray
from .package_data import data_path


class AlphaLookup:
    '''
    Adapted from lookup_alpha_catwise.py from Secrest+21.
    '''
    def __init__(self) -> None:
        with data_path('spec_idx', 'alpha_w12_only.fits') as lookup_path:
            self._load_lookup_data(lookup_path)
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
    
    def _load_lookup_data(self, lookup_path: Path) -> None:
        self.lookup_tab = Table.read(lookup_path)
        self.lookup_alpha = self.lookup_tab['alpha'].data.astype('float32')
        self.lookup_W1_W2 = self.lookup_tab['W1_W2'].data.astype('float32')

        del self.lookup_tab
