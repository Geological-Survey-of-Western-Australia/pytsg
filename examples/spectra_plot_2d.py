"""
Plot Spectra in 2D
==================

Here is a simple plot of the NIR/MIR/TIR spectra using :mod:`matplotlib`. We will merge the wavelength
ranges for the NIR, MIR, and TIR before plotting so that we get the same colour for the same sample
across the three spectral ranges.
"""

import numpy as np
from matplotlib import pyplot as plt

import pytsg

# Read the data
data = pytsg.read_tsg("../example_data/PE257D")

# Concatenate the wavelength indices for NIR, MIR, and TIR spectra
wvl_idx = np.concat([data.nir.wavelength, data.mir.wavelength, data.tir.wavelength])

# Concatenate the spectra for samples 10 through 20 across NIR, MIR, and TIR
spectra = np.concat(
    [data.nir.spectra[10:20, :].T, data.mir.spectra[10:20, :].T, data.tir.spectra[10:20, :].T]
)

# Create a 2D plot of the spectra
plt.plot(wvl_idx, spectra)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("NIR/MIR/TIR Spectra")
plt.show()
