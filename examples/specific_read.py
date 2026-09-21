"""
Specific Read
=============

This example shows how to read specific files using pytsg
"""

from pathlib import Path

from pytsg import parse_tsg
from pytsg.parse_tsg import Cras, Spectra

# %%
# Read Specific Files
# -------------------
# Extract data using the appropriate methods.


dir_demo_1: Path = Path(r"../example_data/SWMB007s")

for fp in dir_demo_1.iterdir():
    print(fp.name)

# %%
# Read bip files

nir: Spectra = parse_tsg.read_tsg_bip_pair(
    dir_demo_1 / "SWMB007s_chips_tsg.tsg", dir_demo_1 / "SWMB007s_chips_tsg.bip", "nir"
)
print(nir)

# %%
# Read tir files

tir: Spectra = parse_tsg.read_tsg_bip_pair(
    dir_demo_1 / "SWMB007s_chips_tsg_tir.tsg", dir_demo_1 / "SWMB007s_chips_tsg_tir.bip", "tir"
)
print(tir)

# %%
# Read cras file

cras: Cras = parse_tsg.read_cras(dir_demo_1 / "SWMB007s_chips_tsg_cras.bip")
print(cras)

# %%
# Read hi-res dat file
#
# sample pending :-(
#
# .. code-block:: python
#
#    lidar: Cras = parse_tsg.read_cras('some_valid_tsg_hires.dat')
