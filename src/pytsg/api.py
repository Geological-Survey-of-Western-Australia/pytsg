"""Public eager readers for TSG packages and their components."""

from pathlib import Path
from typing import Union

from numpy.typing import NDArray

from . import parse_tsg as _parse_tsg
from .parse_tsg import TSG, Spectra, read_cras

__all__ = ["read_tsg", "read_spectra", "read_cras", "read_lidar"]


def read_tsg(path: Union[str, Path], *, include_cras: bool = False) -> TSG:
    """Read a complete TSG package eagerly into detached in-memory objects.

    Source files are read and closed before this function returns. The source package is
    never modified; mutate or export the returned NumPy arrays and Pandas DataFrames
    using their normal APIs.

    Args:
        path: Directory containing the TSG package files.
        include_cras: Read the optional CRAS image when true.

    Returns:
        A fully materialized :class:`~pytsg.parse_tsg.TSG` object.
    """
    return _parse_tsg._read_tsg_package(path, read_cras_file=include_cras)


def read_spectra(
    tsg_file: Union[str, Path],
    bip_file: Union[str, Path],
    *,
    spectrum_name: str = "nir",
) -> Spectra:
    """Read one TSG metadata/data spectral pair eagerly."""
    return _parse_tsg.read_tsg_bip_pair(tsg_file, bip_file, spectrum_name)


def read_lidar(path: Union[str, Path], *, per_spectra: bool = True) -> NDArray:
    """Read a high-resolution lidar/profilometer file into a NumPy array."""
    return _parse_tsg.read_hires_dat(path, per_spectra=per_spectra)
