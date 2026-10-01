"""
Simple Quadratic Method
=======================

The following is an example of using :func:`pytsg.feature.sqm` which implements the simple quadratic method
(Rodger et al) for extracting the feature depth and position.

For this example we will look for absorbance features in the Short Wave InfraRed (SWIR) region between 2120
and 2380 nm as described in the original paper.
"""

# sphinx_gallery_start_ignore
import warnings

# sphinx_gallery_end_ignore
import numpy as np
from matplotlib import pyplot as plt

from pytsg import feature, read_tsg

# Read the dataset
dataset = read_tsg("../example_data/GSNSW_testrocks")

# Calculate the continuum removed spectra
convex_hulls = np.array(
    [feature.chull(np.column_stack((dataset.nir.wavelength, spectrum))) for spectrum in dataset.nir.spectra]
)
continuum_removed_spectra = dataset.nir.spectra / convex_hulls

# sphinx_gallery_start_ignore
warnings.filterwarnings(
    "ignore",
    message="Casting complex values to real discards the imaginary part",
)
# sphinx_gallery_end_ignore
# Run the Simple Quadratic Method between 2120 and 2380 nm
results, coeffs = feature.sqm(
    dataset.nir.wavelength,
    continuum_removed_spectra,
    start_wavelength=2120,
    end_wavelength=2380,
)

# Plot the results
plt.plot(dataset.nir.wavelength, dataset.nir.spectra[173], label="Original")
plt.plot(dataset.nir.wavelength, continuum_removed_spectra[173], label="Continuum removed")
plt.axvline(results[173][0], linestyle="--", label=f"{results[173][0]:.2f}nm (SQM)")
plt.axvspan(2120, 2380, color="yellow", alpha=0.3)
plt.legend()
plt.show()

# Print the full result of what is plotted
print(results[173])
