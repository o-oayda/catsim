from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.samples import CatalogueToMap
from dipoleutils.utils.mask import Masker
import healpy as hp
import matplotlib.pyplot as plt


data = DataLoader('racs', 'low3').load()
low3 = CatalogueToMap(data)
catalogue = low3.get_catalogue()

low3.make_cut('Total_flux', minimum=15, maximum=None)
sbidmap = low3.make_parameter_map('SBID', coordinate_system='equatorial')
psfmap = low3.make_parameter_map('PSF_Maj', coordinate_system='equatorial')

mask = Masker([sbidmap, psfmap], coordinate_system='equatorial')
mask.mask_galactic_plane(5)
mask.mask_a_team_sources(radius_deg=3, source_names=['Cygnus A'])
sbidmap, psfmap = mask.get_masked_density_map()

hp.projview(sbidmap)
plt.show()
