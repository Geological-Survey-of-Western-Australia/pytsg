"""
Convex Hull
===========

Calculate the convex hull of the first spectra in the GSNSW_testrocks dataset and plot it
along with the original spectra.
"""

import numpy as np
from matplotlib import pyplot as plt

from pytsg import feature, read_tsg

dataset = read_tsg("../example_data/GSNSW_testrocks")

points = np.column_stack((dataset.nir.wavelength, dataset.nir.spectra[0]))
chull = feature.chull(points)

plt.plot(dataset.nir.wavelength, dataset.nir.spectra[0], label="NIR spectrum")
plt.plot(dataset.nir.wavelength, chull, "--", label="Convex hull")
plt.legend()
plt.show()
