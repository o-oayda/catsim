from catsim import CatwiseConfig, Catwise
import healpy as hp
import matplotlib.pyplot as plt
from catsim.utils.plotting import smooth_map


config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian'
)
sim = Catwise(config)

sim.initialise_data()
dmap, mask = sim.generate_dipole(log10_n_initial_samples=7.5)

hp.projview(dmap, nest=True)
smooth_map(dmap)
plt.show()
