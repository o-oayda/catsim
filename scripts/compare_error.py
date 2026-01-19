from catsim import CatwiseConfig, Catwise
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from catsim.utils.healsphere import ParameterMap
import matplotlib.lines as mlines
import matplotlib as mpl


mpl.rcParams['text.usetex'] = True

config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian',
    generate_correlated_points=False,
    use_common_extra_error=True,
    store_final_samples=True,
    use_noecl_mask=False,
    add_confusion_noise=True
)
sim = Catwise(config)

sim.initialise_data()
dmap, mask = sim.generate_dipole(
    log10_n_initial_samples=7.5402,
    observer_speed=2.00,
    dipole_longitude=221,
    dipole_latitude=44,
    w1_extra_error=0,
    w2_extra_error=0,
    log10_w1conf_scale=0.,
    log10_w2conf_scale=0.
    # w1_extra_error=3.,
    # w2_extra_error=0.
)
realmap, realmask = sim.make_real_sample(mask_catalogue=True)
hp.projview(realmap, nest=True)
plt.show()

assert sim.final_w1e_samples is not None
assert sim.final_w2e_samples is not None
assert sim.final_pixel_indices is not None

# compare real error scatter vs. simulated
plt.scatter(
    sim.final_w1e_samples, 
    sim.final_w2e_samples, 
    s=0.1, 
    alpha=0.2, 
    label='CatSIM',
    color='tab:orange'
)
plt.scatter(
    sim.real_catalogue['w1e'], 
    sim.real_catalogue['w2e'], 
    s=0.1,
    alpha=0.2, 
    label='S+21 CatWISE',
    color='tab:blue'
)

print(len(sim.real_catalogue['w1e']))
print(len(sim.final_w1e_samples))
print(len(sim.real_catalogue[sim.real_catalogue['w1e'] > 0.04]))

plt.xlabel('W1 error (mag)')
plt.ylabel('W2 error (mag)')
plt.xlim(0, 0.25)
plt.ylim(0, 0.25)
orange_circle = mlines.Line2D(
    [], [], 
    color='tab:orange',
    marker='o',
    linestyle='None',
    markersize=3,
    label='CatSIM'
)
blue_circle = mlines.Line2D(
    [], [], 
    color='tab:blue',
    marker='o',
    linestyle='None',
    markersize=3,
    label='CatWISE'
)

# Add the legend using the custom handles and labels
plt.legend(handles=[orange_circle, blue_circle], loc='upper center')

W1_MIN = 0.035
W1_MAX = 0.055
W2_MIN = 0.065
plt.axvline(x=W1_MIN, linestyle='--', color='black')
plt.axvline(x=W1_MAX, linestyle='--', color='black')
plt.axhline(y=W2_MIN, linestyle='--', color='black')
plt.savefig(
    '/home/oliver/Documents/dipole_notes/ref_report_figs/w1e_w2e.png', 
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# bad points outside of real w1e-w2e space
cut = (
        ((sim.final_w1e_samples > W1_MIN) & (sim.final_w1e_samples < W1_MAX))
        &
        ((sim.final_w2e_samples > W2_MIN))
)
bad_pidx = sim.final_pixel_indices[cut]
bad_dmap = np.bincount(bad_pidx, minlength=49152).astype('float32')
bad_dmap[bad_dmap == 0] = np.nan
print('Bad points: ', int(np.nansum(bad_dmap)))
hp.projview(
    bad_dmap, 
    nest=True, 
    title="Dashed points density map", 
    unit='Sources per healpixel'
)
plt.savefig(
    '/home/oliver/Documents/dipole_notes/ref_report_figs/dashed_points.png', 
    dpi=300, 
    bbox_inches='tight'
)
plt.show()


real_idxs = hp.ang2pix(
    64, 
    sim.real_catalogue['l'], 
    sim.real_catalogue['b'], 
    nest=True, 
    lonlat=True
)
real_emap = ParameterMap(real_idxs, sim.real_catalogue['w1e'], nside=64).get_map()
hp.projview(real_emap, nest=True, sub=311, title='Real CatWISE', min=0.017, max=0.034)

sim_idxs = sim.final_pixel_indices
sim_emap = ParameterMap(sim_idxs, sim.final_w1e_samples, nside=64).get_map()
hp.projview(sim_emap, nest=True, sub=312, title='Sim. CatWISE', min=0.017, max=0.034)

delta_emap = sim_emap - real_emap
hp.projview(delta_emap, nest=True, sub=313, title='Delta (sim $-$ real)', cmap='coolwarm', min=-0.01, max=0.01)
plt.show()

bins = np.linspace(0.018, 0.035, 18)
plt.hist(real_emap, bins=bins, density=True, alpha=0.4, label='Real cell med. errors')
plt.hist(sim_emap, bins=bins, density=True, alpha=0.4, label='Sim. cell med. errors')
plt.legend()
plt.show()

plt.hist(sim.real_catalogue['w1e'], bins=100, alpha=0.4, density=True)
plt.hist(sim.final_w1e_samples, bins=100, alpha=0.4, density=True)
plt.show()

plt.hist(sim.real_catalogue['w2e'], bins=100, alpha=0.4, density=True)
plt.hist(sim.final_w2e_samples, bins=100, alpha=0.4, density=True)
plt.show()
