"""
Simple Read
===========

This example shows the easiest way to use pytsg
"""

from pathlib import Path
from typing import List

from matplotlib import pyplot as plt

from pytsg import parse_tsg

# %%
# Read and Plot
# -------------
# Define some data directories and plot data read from each of them.

dir_data_root = "../example_data"
data_dirs: List[Path] = [path for path in Path(dir_data_root).iterdir() if path.is_dir()]

for data_dir in data_dirs:
    data = parse_tsg.read_package(data_dir)

    hf = plt.figure()
    plt.plot(data.nir.wavelength, data.nir.spectra[0:10, :].T)
    plt.plot(data.tir.wavelength, data.tir.spectra[0:10, :].T)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"pytsg reads tsg files!\n{data_dir.name}")
    plt.show()
