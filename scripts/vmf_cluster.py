from catsim.utils.physics import sample_spherical_points, spherical_to_cart_deg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, vonmises_fisher
import healpy as hp
import argparse
from tqdm import tqdm
from catsim.utils.plotting import smooth_map


def scatter_3D(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z, s=0.1) # pyright: ignore[reportArgumentType]
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z))) 
    plt.show()

def make_density_map(nside, x, y, z):
    source_indices = hp.vec2pix(nside, x, y, z, nest=True)
    return np.bincount(source_indices, minlength=hp.nside2npix(nside))

def overlay_poisson(counts, bins):
    mean_count = np.mean(counts)
    p_bins = poisson.pmf(mu=mean_count, k=bins)
    plt.plot(bins, p_bins)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_3d_plots',
        action='store_true'
    )
    parser.add_argument(
        '--target_sources',
        type=int,
        default=2_500_000
    )
    parser.add_argument(
        '--cluster_rate_param',
        type=float,
        default=10
    )
    parser.add_argument( # kappa is probably very big to get what we want
        '--kappa',
        type=float,
        default=100.
    )
    args = parser.parse_args()

    TARGET_N_SOURCES = args.target_sources
    CLUSTER_RATE_PARAM = args.cluster_rate_param
    KAPPA = args.kappa
    DISABLE_3D = args.disable_3d_plots

    n_parents = int(TARGET_N_SOURCES / CLUSTER_RATE_PARAM)
    print(CLUSTER_RATE_PARAM, n_parents)

    long_deg, lat_deg = sample_spherical_points(n_parents)
    xyz = spherical_to_cart_deg(long_deg, lat_deg)

    x = xyz[:, 0]; y = xyz[:, 1]; z = xyz[:, 2]
    if not DISABLE_3D:
        scatter_3D(x, y, z)

    all_offspring = []
    n_offspring = poisson.rvs(CLUSTER_RATE_PARAM * np.ones(n_parents))
    for i in tqdm(range(n_parents)):
        parent_direction = xyz[i, :]
        vmf = vonmises_fisher(parent_direction, kappa=KAPPA)
        child_points = vmf.rvs(size=n_offspring[i])
        all_offspring.append(child_points)

    offspring_dirs = np.concatenate(all_offspring, axis=0)
    print(offspring_dirs.shape)

    x = offspring_dirs[:, 0]; y = offspring_dirs[:, 1]; z = offspring_dirs[:, 2]
    if not DISABLE_3D:
        scatter_3D(x, y, z)

    dmap = make_density_map(64, x, y, z)
    MIN = np.nanmin(dmap)
    MAX = np.nanmax(dmap)
    bin_ints = np.arange(MIN, MAX)
    bin_edges = np.arange(MIN - 0.5, MAX + 1.5, 1)

    plt.hist(dmap, bins=bin_edges, density=True, alpha=0.4)
    overlay_poisson(dmap, bin_ints)
    plt.yscale('log')
    plt.show()

    smooth_map(dmap)
    plt.show()
