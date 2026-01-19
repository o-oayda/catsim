from catsim.configs import CatwiseConfig
from catsim.simulator import Catwise
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


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
realmap, realmask = sim.make_real_sample(mask_catalogue=True)

w1 = sim.real_catalogue['w1']
w2 = sim.real_catalogue['w2']
log_w1cov = np.log10(sim.real_catalogue['w1cov'])
log_w2cov = np.log10(sim.real_catalogue['w2cov'])
logcov_stack = np.column_stack([log_w1cov, log_w2cov])
cov_delta = log_w1cov - log_w2cov
cov_mean = 0.5 * ( log_w1cov + log_w2cov )

plt.hist2d(log_w1cov, log_w2cov, bins=100, norm='log')
plt.colorbar()

plt.figure()
plt.hist2d(cov_delta, cov_mean, norm='log', bins=100)
xtick_spacing = 0.01
ytick_spacing = 0.02
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_spacing))
ax.grid(True, linestyle='--', linewidth=0.6, color='gray')
ax.set_axisbelow(True)
plt.colorbar()

plt.figure()
plt.hist2d(w1, cov_delta, norm='log', bins=100)
plt.colorbar()
plt.show()
