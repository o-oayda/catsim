from pathlib import Path
from time import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from catsim import RACS_MID1, Racs, RacsConfig, smooth_map
from catsim.utils.constants import CMB_B, CMB_L
from dipoleutils.utils.plotting import plot_log_log_histogram


reference_mask_path = (
    Path.home()
    / "Documents/sbi/derived/observations/racs_mid1_flux15_ds4"
    / "reference_observation_native.npz"
)
with np.load(reference_mask_path, allow_pickle=False) as reference_observation:
    reference_mask = reference_observation["mask"].astype(np.bool_, copy=False)

config = RacsConfig(
    product=RACS_MID1,
    flux_min=15.0,
    mask_map=reference_mask,
    chunk_size=100_000,
    store_final_samples=True,
    paf_reference_temp_c=27.1,
    cluster_count_model='poisson'
)
sim = Racs(config)

init_t0 = time()
sim.initialise_data()
init_t1 = time()
t0 = time()
dmap, mask = sim.generate_dipole(
    log10_n_initial_samples=6.554,
    observer_speed=4.96,
    dipole_longitude=169,
    dipole_latitude=43,
    temp_beta=0.0067,
    lambda_clus=0.58,
    fractional_error_eta=4.
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
sampled_flux_error_map = sim.sampled_flux_error_map
sampled_fractional_error_map = sim.sampled_fractional_error_map
elevation_map = sim.elevation_map
if elevation_map is not None:
    finite_elevations = np.isfinite(elevation_map)
    print(f"Elevation-covered pixels: {finite_elevations.sum()}")
    if np.any(finite_elevations):
        print(
            "Elevation range (deg): "
            f"{elevation_map[finite_elevations].min():.2f} to "
            f"{elevation_map[finite_elevations].max():.2f}"
        )
if sampled_flux_error_map is not None:
    finite_flux_errors = np.isfinite(sampled_flux_error_map)
    print(f"Sampled flux-error-covered pixels: {finite_flux_errors.sum()}")
    if np.any(finite_flux_errors):
        print(
            "Sampled absolute flux error range (mJy): "
            f"{sampled_flux_error_map[finite_flux_errors].min():.4f} to "
            f"{sampled_flux_error_map[finite_flux_errors].max():.4f}"
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
# hp.projview(dmap, nest=True, title="RACS-mid1 Simulated Count Map")
# hp.projview(sbid_map, nest=True, title="RACS-mid1 SBID Map")
# if temperature_map is not None:
#     hp.projview(temperature_map, nest=True, title="RACS-mid1 Temperature Map (C)")
if sampled_fractional_error_map is not None:
    hp.projview(
        sampled_fractional_error_map,
        nest=True,
        title="RACS-mid1 Sampled Fractional Error Map",
    )
smooth_map(dmap, coord=['C'], graticule=True, graticule_labels=True)
plt.show()

# plot_log_log_histogram(sim.final_observed_flux_samples, bins=100)
# plt.show()
