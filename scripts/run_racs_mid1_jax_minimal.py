"""Minimal RACS MID1 JAX smoke script.

This opts into Open-Meteo ambient temperatures as a fallback when PAF
temperatures are unavailable for the RACS observations.
"""

from __future__ import annotations

import logging

import jax
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from catsim import RACS_MID1, RacsConfig, RacsJax


def main() -> None:
    NSIMS = 100
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    cfg = RacsConfig(
        product=RACS_MID1,
        flux_min=0.015,
        chunk_size=100_000,
        store_final_samples=False,
        fractional_error_flux_min_mjy=0.01,
        temperature_fallback="open_meteo",
    )
    sim = RacsJax(cfg)

    sim.initialise_data()
    density_map, mask = sim.batch_generate_dipole(
        theta={
            'log10_n_initial_samples': 6.5 * np.ones((NSIMS,)),
            'temp_beta': 0.02 * np.ones((NSIMS,))
        },
        batch_size=5,
        key=jax.random.PRNGKey(0),
        show_progress=True
    )

    print(f"density_map shape: {density_map.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"retained sources: {np.nansum(density_map):.0f}")

    return sim, density_map


if __name__ == "__main__":
    sim, density_map = main()
