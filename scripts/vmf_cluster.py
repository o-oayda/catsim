from time import time
from typing import Literal
from numpy.typing import NDArray
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

def make_tangent_basis(unit_point_vector: NDArray) -> tuple[NDArray, NDArray]:
    zhat = np.asarray([0., 0., 1.])
    yhat = np.asarray([0., 1., 0.])

    cos_to_z = np.dot(unit_point_vector, zhat) 
    use_z = np.abs(cos_to_z) < 0.9

    ref = np.empty_like(unit_point_vector)
    ref[use_z] = zhat
    ref[~use_z] = yhat

    e1 = np.cross(ref, unit_point_vector)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)

    e2 = np.cross(unit_point_vector, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)

    assert np.isclose(np.linalg.norm(e1, axis=1), 1.).all()
    assert np.isclose(np.linalg.norm(e2, axis=1), 1.).all()

    return e1, e2
    
def generate_clusters(
        parent_points, 
        cluster_rate_param, 
        method: Literal['vmf', 'gaussian']
):
    n_parents = parent_points.shape[0]
    per_cluster_n_offspring = poisson.rvs(cluster_rate_param * np.ones(n_parents))
    total_n_offspring = np.sum(per_cluster_n_offspring)

    if method == 'vmf':
        all_offspring = []

        for i in tqdm(range(n_parents)):
            parent_direction = parent_points[i, :]
            vmf = vonmises_fisher(parent_direction, kappa=KAPPA)
            child_points = vmf.rvs(size=per_cluster_n_offspring[i]) # pyright: ignore[reportIndexIssue]
            all_offspring.append(child_points)

        offspring_dirs = np.concatenate(all_offspring, axis=0)
        print(offspring_dirs.shape)

        return offspring_dirs

    elif method == 'gaussian':
        t0 = time()
        e1, e2 = make_tangent_basis(parent_points)

        uv = np.random.normal(scale=SIGMA, size=(total_n_offspring, 2))
        u = uv[:, 0]; v = uv[:, 1]

        parent_idxs = np.repeat(np.arange(n_parents), per_cluster_n_offspring)

        offspring_dirs = (
            parent_points[parent_idxs, :]
          + u[:, None] * e1[parent_idxs]
          + v[:, None] * e2[parent_idxs]
        )

        norms = np.linalg.norm(offspring_dirs, axis=1, keepdims=True)
        offspring_dirs /= norms
        print(offspring_dirs.shape)

        t1 = time()
        print(t1 - t0)

        return offspring_dirs
        
    else:
        raise ValueError(f'Method ({method}) not recognised.')


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
    parser.add_argument(
        '--method',
        choices=['vmf', 'gaussian'],
        default='vmf'
    )
    args = parser.parse_args()

    TARGET_N_SOURCES = args.target_sources
    CLUSTER_RATE_PARAM = args.cluster_rate_param
    KAPPA = args.kappa
    SIGMA = KAPPA ** (-1/2)
    DISABLE_3D = args.disable_3d_plots
    METHOD = args.method

    n_parents = int(TARGET_N_SOURCES / CLUSTER_RATE_PARAM)
    print(CLUSTER_RATE_PARAM, n_parents)

    long_deg, lat_deg = sample_spherical_points(n_parents)
    xyz = spherical_to_cart_deg(long_deg, lat_deg)

    offspring_dirs = generate_clusters(xyz, CLUSTER_RATE_PARAM, method=METHOD)
    assert offspring_dirs is not None

    x = offspring_dirs[:, 0]; y = offspring_dirs[:, 1]; z = offspring_dirs[:, 2]
    if not DISABLE_3D:
        scatter_3D(x, y, z)

    dmap = make_density_map(64, x, y, z)
    MIN = np.nanmin(dmap)
    MAX = np.nanmax(dmap)
    bin_ints = np.arange(MIN, MAX)
    bin_edges = np.arange(MIN - 0.5, MAX + 1.5, 1)

    plt.hist(dmap, bins=bin_edges, density=True, alpha=0.4) # pyright: ignore[reportArgumentType]
    overlay_poisson(dmap, bin_ints)
    plt.yscale('log')
    plt.show()

    smooth_map(dmap)
    hp.projview(dmap, nest=True)
    plt.show()
