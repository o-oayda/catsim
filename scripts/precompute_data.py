from catsim import CatwiseConfig, Catwise


config = CatwiseConfig(
    cat_w12_min=0.5, 
    cat_w1_max=17.0, 
    magnitude_error_dist='gaussian'
)
sim = Catwise(config)


sim.load_catalogue()

sim.determine_masked_pixels()
sim.make_masked_catalogue()

sim.create_confusion_skylookup()

# sim.precompute_data(mask_north_ecliptic=False)

# if False:
# sim.load_catalogue()
# # sim.create_coverage_maps(use_mask=False)
# sim.determine_masked_pixels()
# sim.make_masked_catalogue()
# # sim.create_magnitude_coverage_cell_dist()
# sim.create_mag_cov_hashgrid(
#     grid_step=[0.1, 0.02, 0.1, 0.04], 
#     project_coverage=True
# )
