from catsim import CatwiseConfig, Catwise, batch_simulate, prng_key
import numpy as np


N_SIMS = 10
rng_key = prng_key(42)
config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian'
)
sim = Catwise(config)
sim.initialise_data()

simulator_function = sim.generate_dipole
theta = {'log10_n_initial_samples': np.linspace(7, 8, N_SIMS)}
dmap, mask = batch_simulate(
    theta=theta, 
    model_callable=simulator_function, 
    n_workers=32,
    rng_key=rng_key
)
