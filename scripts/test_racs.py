from time import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from catsim import RacsLow3, RacsLow3Config, smooth_map
from catsim.utils.constants import CMB_B, CMB_L
from dipoleutils.utils.plotting import plot_log_log_histogram


config = RacsLow3Config(
    flux_min=15.0,
    chunk_size=100_000,
    store_final_samples=True,
    nside=32
)
sim = RacsLow3(config)

init_t0 = time()
sim.initialise_data()
init_t1 = time()
t0 = time()
A = -0.25
B = -A + 1
dmap, mask = sim.generate_dipole(
    log10_n_initial_samples=6.65,
    observer_speed=1.,
    dipole_longitude=CMB_L,
    dipole_latitude=CMB_B,
    temp_slope=A,
    temp_intercept=B,
    temp_pivot_c=30.
)
t1 = time()

print(f"Time to initialise RACS SIM: {init_t1 - init_t0:.3g} s")
print(f"Time to generate RACS SIM: {t1 - t0:.3g} s")
print(f"Map shape: {dmap.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Unmasked pixels: {mask.sum()}")
print(f"Simulated sources retained: {len(sim.final_pixel_indices)}")
print(f"Mean count in observed footprint: {dmap[mask].mean():.3f}")
print(f"Max count in observed footprint: {dmap[mask].max():.0f}")

sbid_map = sim.tile_lookup_map.astype(np.float32)
sbid_map[sbid_map < 0] = np.nan
print(f"SBID-covered pixels: {np.isfinite(sbid_map).sum()}")
temperature_map = sim.temperature_map
if temperature_map is not None:
    finite_temperatures = np.isfinite(temperature_map)
    print(f"Temperature-covered pixels: {finite_temperatures.sum()}")
    if np.any(finite_temperatures):
        print(f"Temperature range (C): {temperature_map[finite_temperatures].min():.2f} to {temperature_map[finite_temperatures].max():.2f}")

print(f"n sources: {np.nansum(dmap)}")
# hp.projview(dmap, nest=True, title="RACS-low3 Simulated Count Map")
# hp.projview(sbid_map, nest=True, title="RACS-low3 SBID Map")
# if temperature_map is not None:
#     hp.projview(temperature_map, nest=True, title="RACS-low3 Temperature Map (C)")
smooth_map(dmap, coord=['C'], graticule=True, graticule_labels=True)
plt.show()

# plot_log_log_histogram(sim.final_observed_flux_samples, bins=100)
# plt.show()
