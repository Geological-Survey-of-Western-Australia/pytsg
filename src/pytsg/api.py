"""Functions for reading TSG packages and their components."""

from pathlib import Path
from typing import Union

from .models import TSG
from .readers import package as _package
from .readers import read_cras, read_lidar, read_spectra

__all__ = ["read_tsg", "read_spectra", "read_cras", "read_lidar"]


def read_tsg(path: Union[str, Path], *, include_cras: bool = False) -> TSG:
    """Read a complete TSG package

    Source files are read and closed before this function returns. The source package is
    never modified; mutate or export the returned NumPy arrays and Pandas DataFrames
    using their normal APIs.

    Args:
        path: Directory containing the TSG package files.
        include_cras: Read the optional CRAS image when true.

    Returns:
        A fully materialised `~pytsg.models.TSG` object.
    """
    return _package._read_tsg_package(path, read_cras_file=include_cras)
