"""Minimal RACS MID1 JAX smoke script.

This currently fails during initialisation until PAF temperature data are
available for the RACS observations.
"""

from __future__ import annotations

import jax
import numpy as np

from catsim import RACS_MID1, RacsConfig, RacsJax, Racs


def main() -> None:
    cfg = RacsConfig(
        product=RACS_MID1,
        flux_min=0.015,
        chunk_size=50_000,
        store_final_samples=False,
    )
    sim = Racs(cfg)

    sim.initialise_data()
    density_map, mask = sim.generate_dipole(
        log10_n_initial_samples=6.5,
        # key=jax.random.PRNGKey(0),
    )

    print(f"density_map shape: {density_map.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"retained sources: {np.nansum(density_map):.0f}")

    return sim


if __name__ == "__main__":
    sim = main()
