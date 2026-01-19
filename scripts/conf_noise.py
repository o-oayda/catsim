from catsim import Catwise, CatwiseConfig
from astropy.table import Table, join
import healpy as hp
from catsim.utils.healsphere import ParameterMap
import matplotlib.pyplot as plt
import numpy as np


config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian'
)
sim = Catwise(config)

sim.load_catalogue()

sim.determine_masked_pixels()
sim.make_masked_catalogue()

all_table = sim.masked_catalogue
conf_table = Table.read('src/catsim/data/catwise_w120p5_confdata.fits')


out = join(all_table, conf_table, keys=['source_id'])

real_idxs = hp.ang2pix(64,  out['l'],  out['b'], nest=True, lonlat=True)
real_w2 = ParameterMap(real_idxs, out['w2conf'] / out['w2flux'], nside=64)
real_w1 = ParameterMap(real_idxs, out['w1conf'] / out['w1flux'], nside=64)

# real_emap[real_emap == 0] = np.nan
