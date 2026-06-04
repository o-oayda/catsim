from time import time
import jax
import numpy as np
from catsim import CatwiseConfig, CatwiseJax, smooth_map
import matplotlib.pyplot as plt


n_sims = 100

config = CatwiseConfig(
    cat_w12_min=0.5,
    cat_w1_max=17.0,
    magnitude_error_dist="gaussian",
    generate_correlated_points=False,
    use_common_extra_error=True,
    add_confusion_noise=False,
)
sim = CatwiseJax(config)
sim.initialise_data()

theta = {
    "log10_n_initial_samples": np.full(n_sims, 7.5402, dtype=np.float32),
    "observer_speed": np.full(n_sims, 2.0, dtype=np.float32),
    "dipole_longitude": np.full(n_sims, 220.0, dtype=np.float32),
    "dipole_latitude": np.full(n_sims, 44.0, dtype=np.float32),
    "w1_extra_error": np.full(n_sims, 4., dtype=np.float32),
}

t0 = time()
maps, masks = sim.batch_generate_dipole(
    theta,
    key=jax.random.PRNGKey(0),
    batch_size=10,
    show_progress=True,
)
t1 = time()

print(f"Generated {n_sims} CatWISE JAX sims in {t1 - t0:.3g} s")
print(f"maps: {maps.shape} {maps.dtype}")
print(f"masks: {masks.shape} {masks.dtype}")

## count statistics
true_dmap = np.load('catwise_S21_probably.npy')
bins = np.arange(np.nanmin(true_dmap), np.nanmax(true_dmap))
# av_sim_dmap = np.mean(maps, axis=0)
plt.hist(true_dmap, alpha=0.3, bins=bins, label='CatWISE')
plt.hist(maps[2, :], alpha=0.3, bins=bins, label='CatSIM')
# plt.yscale('log')
plt.legend()
plt.show()
