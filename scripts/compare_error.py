from catsim import CatwiseConfig, Catwise
import numpy as np
import matplotlib.pyplot as plt


config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian',
    generate_correlated_points=False,
    use_common_extra_error=True,
    store_final_samples=True
)
sim = Catwise(config)

sim.initialise_data()
dmap, mask = sim.generate_dipole(
    log10_n_initial_samples=7.5497,
    observer_speed=2.07,
    dipole_longitude=220,
    dipole_latitude=44,
    w1_extra_error=0.
)
sim.make_real_sample(mask_catalogue=True)

# compare real error scatter vs. simulated
plt.scatter(
    sim.real_catalogue['w1e'], 
    sim.real_catalogue['w2e'], 
    s=0.1, 
    alpha=0.4, 
    label='S+21 CatWISE'
)
plt.scatter(
    sim.final_w1e_samples, 
    sim.final_w2e_samples, 
    s=0.1, 
    alpha=0.4, 
    label='CatSIM'
)

print(len(sim.real_catalogue['w1e']))
print(len(sim.final_w1e_samples))
print(len(sim.real_catalogue[sim.real_catalogue['w1e'] > 0.04]))

plt.xlabel('W1 error (mag)')
plt.ylabel('W2 error (mag)')
plt.xlim(0, 0.25)
plt.ylim(0, 0.25)
plt.legend()
plt.show()
