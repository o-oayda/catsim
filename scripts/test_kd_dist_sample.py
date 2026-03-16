from catsim import CatwiseConfig, Catwise
from catsim.utils.hists import build_bin_lookup_grid, sample_sigma_w1w2_from_bins_vectorized_fast
from tmp_helpers import adaptive_binned_distribution_2d
import numpy as np


config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian'
)
sim = Catwise(config)

sim.load_catalogue()
sim.determine_masked_pixels()
sim.make_masked_catalogue()

band = 'w1'
N_1D_BINS = 200
magnitude_bins = np.linspace(
    np.min(sim.masked_catalogue['w1']), sim.cat_w1_max, N_1D_BINS
)
coverage_bins = np.linspace(1.5, 4., N_1D_BINS)

cat = sim.masked_catalogue
x = cat[f'{band}']
y = np.log10(cat[f'{band}cov'])
z = np.concatenate([cat['w1e'][:, None], cat['w2e'][:, None]], axis=1)

bin_bounds, bin_values = adaptive_binned_distribution_2d(
    x, y, z, target_count=10000, max_factor=2, min_count=5000
)
x_grid, y_grid, bin_index_grid = build_bin_lookup_grid(bin_bounds)

# ax = plot_adaptive_bins(
#     bin_bounds,
#     x=x,
#     y=y,
#     show_counts=False,
#     bin_values=bin_values,
# )
# plt.show()

# check resampled dist agrees with true bin dist
BIN_IDX = 500
bounds = bin_bounds[BIN_IDX]
mags = np.random.uniform(low=bounds[0], high=bounds[1], size=10_000)
covs = np.random.uniform(low=bounds[2], high=bounds[3], size=10_000)

resampled_sigma = sample_sigma_w1w2_from_bins_vectorized_fast(
    mags, covs, bin_bounds, bin_values, x_grid, y_grid, bin_index_grid
)

# yes they agree, good
# plt.hist(bin_values[BIN_IDX], density=True, alpha=0.4, bins=100)
# plt.hist(resampled_sigma, density=True, alpha=0.4, bins=100)
# plt.show()

# plot_top_bin_histograms(dist, top_k=5, bins=100)
