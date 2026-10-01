[![PyPI](https://img.shields.io/pypi/v/pytsg.svg?style=flat)](https://pypi.python.org/pypi/pytsg)
[![pytsg downloads](https://img.shields.io/pypi/dm/pytsg.svg?style=flat)](https://pypistats.org/packages/pytsg)

# pytsg

**pytsg** is a lightweight, open-source utility that provides a simple way to load and access TSG file packages from Python. With a single function call, a TSG package can be imported into an easy-to-use Python object for further analysis and processing.

[The Spectral Geologist (TSG)](https://research.csiro.au/thespectralgeologist/) is an industry-standard application for hyperspectral data analysis developed by CSIRO.

## Installation

```
pip install pytsg

# To use zarr when loading large imagery/cras files
pip install pytsg[bigfile]
```

## Usage

The TSG dataset will typically have the following file structure:
```
HOLENAME/
├── HOLENAME_tsg.bip
├── HOLENAME_tsg.tsg
├── HOLENAME_tsg_cras.bip
├── HOLENAME_tsg_hires.dat
├── HOLENAME_tsg_mir.bip
├── HOLENAME_tsg_mir.tsg
├── HOLENAME_tsg_tir.bip
└── HOLENAME_tsg_tir.tsg
```

You can load the TSG dataset using `read_tsg` and then interact with the data:
```python
from matplotlib import pyplot as plt
import pytsg

data = pytsg.read_tsg("example_data/PE257D")

plt.plot(data.nir.wavelength, data.nir.spectra[0:10, :].T)
plt.plot(data.tir.wavelength, data.tir.spectra[0:10, :].T)
plt.xlabel("Wavelength nm")
plt.ylabel("Reflectance")
plt.title("pytsg reads tsg files")
plt.show()
```

If you would prefer to load just the individual files you can do that also:

```python
# read bip files containing the spectra
import pytsg

nir = pytsg.read_spectra("ETG0187_tsg.tsg", "ETG0187_tsg.bip", "nir")
tir = pytsg.read_spectra("ETG0187_tsg_tir.tsg", "ETG0187_tsg_tir.bip", "tir")

# read cras files containing the imagery
cras = pytsg.read_cras("ETG0187_tsg_cras.bip")

# read hires dat file containing the LiDAR/profilometer data
lidar = pytsg.read_lidar("ETG0187_tsg_hires.dat")
```

## Thanks
Thanks to CSIRO and in particular Andrew Rodger for his assistance in decoding the file structures.

## Supported by the Geoscience Open Maintainers Collective
This project is supported by the Geoscience Open Maintainers Collective, an initiative of Geoscience Data Integrations (GDI), the open data and software division of the Geological Survey of Western Australia (GSWA). The Collective provides targeted developer support to open geoscience software projects; working alongside maintainers to share the maintenance load and help keep community tools stable, usable, and accessible.