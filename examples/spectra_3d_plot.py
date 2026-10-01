"""
Plot Spectra in 3D
==================

Here is a simple example of a 3D surface plot of the NIR/MIR/TIR spectra using :mod:`matplotlib`.
"""

import numpy as np
from matplotlib import pyplot as plt

import pytsg

# Read the data
data = pytsg.read_tsg("../example_data/PE257D")

# Concatenate the wavelength indices for NIR, MIR, and TIR spectra
wvl_idx = np.concat([data.nir.wavelength, data.mir.wavelength, data.tir.wavelength])

# Select a range of samples (skipped a few suprious samples at the start of the dataset)
samples_idx = np.arange(10, 100)

# Concatenate the spectra for the selected samples across NIR, MIR, and TIR
spectra = np.concat(
    [data.nir.spectra[samples_idx, :], data.mir.spectra[samples_idx, :], data.tir.spectra[samples_idx, :]],
    axis=1,
)

# Create a 3D surface plot of the spectra
W, S = np.meshgrid(wvl_idx, samples_idx)
fig, ax = plt.subplots(layout="constrained", subplot_kw={"projection": "3d"})
surf = ax.plot_surface(
    W, S, spectra, cmap="Spectral", edgecolor="none", linewidth=0, antialiased=False, shade=False
)
ax.set_xlabel("Wavelength (nm)", fontsize=9)
ax.set_ylabel("Sample Index", fontsize=9)
ax.set_zlabel("Intensity", fontsize=9)
ax.set_title("PE257D Raw Spectra (NIR/MIR/TIR)")
plt.tick_params(axis="both", labelsize=8)
plt.show()
