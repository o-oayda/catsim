import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import healpy as hp
from dipoleutils import DataLoader, CatalogueToMap
from pathlib import Path


def load_catalogue(sample):
    return DataLoader('racs', sample).load()


if __name__ == "__main__":
    low3 = load_catalogue('low3')
    mid1 = load_catalogue('mid1-scaled')

    path_to_noisemaps = Path.home() / 'catalogue_data' / 'racs' / 'noisemaps'
    noisemap_low3_source = hp.read_map(path_to_noisemaps / 'RACS-low3.iqr.hpx')
    noisemap_mid1_source = hp.read_map(path_to_noisemaps / 'RACS-mid1.iqr.hpx')
    noisemap_low3_source[noisemap_low3_source == hp.UNSEEN] = np.nan
    noisemap_mid1_source[noisemap_mid1_source == hp.UNSEEN] = np.nan

    noisemap_low3 = CatalogueToMap(low3).make_parameter_map("Noise", "equatorial")
    noisemap_mid1 = CatalogueToMap(mid1).make_parameter_map("noise", "equatorial")
    hp.projview(noisemap_low3, norm='log', sub=221, rlabel='low3')
    hp.projview(noisemap_mid1, norm='log', sub=222, rlabel='mid1')
    hp.projview(noisemap_low3_source, norm='log', sub=223, rlabel='low3')
    hp.projview(noisemap_mid1_source, norm='log', sub=224, rlabel='mid1')
    plt.show()

    # do noise lookup
    print('starting lookup')
    map_nside = hp.npix2nside(len(noisemap_low3_source))
    ra, dec = low3['RA'], low3['Dec']
    low3['hpx_idx'] = hp.ang2pix(map_nside, ra, dec, lonlat=True)
    low3['noise_source'] = noisemap_low3_source[low3['hpx_idx']]
    print('finished lookup')

    noise = np.asarray(low3['noise_source'])
    flux = np.asarray(low3['Total_flux'])
    flux_error = np.asarray(low3['E_Total_flux'])

    valid = (
        np.isfinite(noise)
        & np.isfinite(flux)
        & np.isfinite(flux_error)
        & (noise > 0)
        & (flux > 0)
        & (flux_error > 0)
    )
    noise = noise[valid]
    flux = flux[valid]
    flux_error = flux_error[valid]

    flux_limits = np.percentile(flux, [0.01, 99.9])
    noise_limits = np.percentile(noise, [0.01, 99.9])
    hexbin_extent = (
        np.log10(flux_limits[0]),
        np.log10(flux_limits[1]),
        np.log10(noise_limits[0]),
        np.log10(noise_limits[1]),
    )

    fig, ax = plt.subplots()
    mesh = ax.hexbin(
        flux,
        noise,
        C=flux_error,
        reduce_C_function=np.mean,
        gridsize=400,
        xscale='log',
        yscale='log',
        extent=hexbin_extent,
        mincnt=10,
        cmap='viridis',
        norm=LogNorm(),
        linewidths=0,
        edgecolors='none',
        antialiased=False,
    )
    ax.set_xlabel('Total flux')
    ax.set_ylabel('Noise')
    ax.set_title('Mean flux error by total flux and noise')
    fig.colorbar(mesh, ax=ax, label='Mean flux error')
    fig.tight_layout()

    count_fig, count_ax = plt.subplots()
    count_mesh = count_ax.hexbin(
        flux,
        noise,
        gridsize=400,
        xscale='log',
        yscale='log',
        extent=hexbin_extent,
        mincnt=10,
        cmap='viridis',
        norm=LogNorm(),
        linewidths=0,
        edgecolors='none',
        antialiased=False,
    )
    count_ax.set_xlabel('Total flux')
    count_ax.set_ylabel('Noise')
    count_ax.set_title('Source count by total flux and noise')
    count_fig.colorbar(count_mesh, ax=count_ax, label='Source count')
    count_fig.tight_layout()

    plt.show()

    # plt.scatter(flux, noise, c=flux_error, s=1, alpha=0.2, norm='log')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()
