from catsim import CatwiseConfig, Catwise, smooth_map
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


VERSION = 'S21'

if VERSION == 'S22': # median free gauss extra err S22
    params = {
        'log10_n_initial_samples': 7.5418,
        'w1_extra_error': 3.39,
        'observer_speed': 1.94,
        'dipole_longitude': 215,
        'dipole_latitude': 45.7,
        'w1_max': 16.5
    }
    real_dmap = np.load('catwise_S22.npy')
elif VERSION == 'S21':
    params = {
        'log10_n_initial_samples': 7.5497,
        'w1_extra_error': 3.57,
        'observer_speed': 2.07,
        'dipole_longitude': 221,
        'dipole_latitude': 44.,
        'w1_max': 16.4
    }
    real_dmap = np.load('catwise_S21_probably.npy')
else:
    raise ValueError(f'Catwise version ({VERSION}) not recognised.')

config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian',
    use_common_extra_error=True,
    base_mask_version=VERSION
)
sim = Catwise(config)

sim.initialise_data()
dmap, mask = sim.generate_dipole(**params)

MIN = np.nanmin(dmap)
MAX = np.nanmax(dmap)
binary_mask = ~np.isnan(dmap)

dmap_seen = dmap[binary_mask]
bin_ints = np.arange(MIN, MAX)
bin_edges = np.arange(MIN - 0.5, MAX + 1.5, 1)
print(bin_ints)
print(bin_edges)
plt.hist(
    dmap_seen, bins=bin_edges, alpha=0.4, density=True, align='mid',
    label=f'CatSIM cell counts ({VERSION})'
)

# actual CatWISE (2022) counts
real_dmap[~binary_mask] = np.nan
plt.hist(
    real_dmap, bins=bin_edges, alpha=0.4, density=True, align='mid',
    label=f'CatWISE cell counts ({VERSION})'
)

# overlay poisson
mean_d = np.mean(dmap_seen)
pois_d = poisson.pmf(bin_ints, mu=mean_d)
plt.plot(bin_ints, pois_d, label=r'Poisson dist., $\lambda = \bar{N}$')

plt.yscale('log')
plt.legend()
plt.show()

smooth_map(dmap, sub=211)
smooth_map(real_dmap, sub=212)
plt.show()
