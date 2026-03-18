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
    # temp_slope=A,
    # temp_intercept=B,
    temp_pivot_c=30.,
    fractional_error_eta=20.
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
fractional_error_map = sim.fractional_error_map
sampled_fractional_error_map = sim.sampled_fractional_error_map
if fractional_error_map is not None:
    finite_fractional_errors = np.isfinite(fractional_error_map)
    print(f"Fractional-error-covered pixels: {finite_fractional_errors.sum()}")
    if np.any(finite_fractional_errors):
        print(
            "Fractional error range: "
            f"{fractional_error_map[finite_fractional_errors].min():.4f} to "
            f"{fractional_error_map[finite_fractional_errors].max():.4f}"
        )
if sampled_fractional_error_map is not None:
    finite_sampled_fractional_errors = np.isfinite(sampled_fractional_error_map)
    print(f"Sampled fractional-error-covered pixels: {finite_sampled_fractional_errors.sum()}")
    if np.any(finite_sampled_fractional_errors):
        print(
            "Sampled fractional error range: "
            f"{sampled_fractional_error_map[finite_sampled_fractional_errors].min():.4f} to "
            f"{sampled_fractional_error_map[finite_sampled_fractional_errors].max():.4f}"
        )

print(f"n sources: {np.nansum(dmap)}")
# hp.projview(dmap, nest=True, title="RACS-low3 Simulated Count Map")
# hp.projview(sbid_map, nest=True, title="RACS-low3 SBID Map")
# if temperature_map is not None:
#     hp.projview(temperature_map, nest=True, title="RACS-low3 Temperature Map (C)")
if sampled_fractional_error_map is not None:
    hp.projview(
        sampled_fractional_error_map,
        nest=True,
        title="RACS-low3 Sampled Fractional Error Map",
    )
smooth_map(dmap, coord=['C'], graticule=True, graticule_labels=True)
plt.show()

# plot_log_log_histogram(sim.final_observed_flux_samples, bins=100)
# plt.show()
