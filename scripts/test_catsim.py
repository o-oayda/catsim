from time import time
from catsim import CatwiseConfig, Catwise, smooth_map
import healpy as hp
import matplotlib.pyplot as plt


config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian',
    generate_correlated_points=False,
    use_common_extra_error=True,
    add_confusion_noise=True
)
sim = Catwise(config)

sim.initialise_data()
t0 = time()
dmap, mask = sim.generate_dipole(
    log10_n_initial_samples=7.5402,
    observer_speed=2.0,
    dipole_longitude=220,
    dipole_latitude=44,
    w1_extra_error=0.,
    w2_extra_error=0.,
    log10_w1conf_scale=1.3,
    log10_w2conf_scale=1.3
)
t1 = time()
time_taken = t1 - t0

print(f'Time to generate CatSIM: {time_taken:.3g}')

hp.projview(dmap, nest=True)
smooth_map(dmap)
plt.show()


