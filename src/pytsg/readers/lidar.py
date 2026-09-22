"""Readers for TSG high-resolution lidar/profilometer files."""

import warnings
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray


def read_lidar(path: Union[str, Path], *, per_spectra: bool = True) -> NDArray:
    """
    Read a high-resolution lidar/profilometer file into a NumPy array.

    Note:
        HyLogger data will usually have more LiDAR data points than samples so
        we return the mean value per sample when `per_sample` is `True`.

    Args:
        path (Union[str, Path]): path to `*_hires.dat` containing LiDAR data.
        per_spectra (bool, optional): Return one sample per spectrum when true, or all raw samples.
            Defaults to True.

    Returns:
        NDArray: LiDAR/profilometer data

    Example:
        >>> # Get LiDAR values per sample
        >>> from pytsg import read_lidar
        >>> profile = read_lidar("ETG0187_tsg_hires.dat")
        >>>
        >>> # Get the full resolution profile
        >>> full_profile = read_lidar("ETG0187_tsg_hires.dat", per_spectra=False)
    """
    with open(path, "rb") as file:
        _idchar = file.read(20)
        _nsclr, _nl, nsps = np.fromfile(file, np.int32, 3)
        minp, maxp = np.fromfile(file, np.float32, 2)
        _prof = file.read(12)
        _ = np.fromfile(file, np.ubyte, 4)
        lidar = np.fromfile(file, dtype=np.float32)
        lidar[lidar < minp] = np.nan

    if per_spectra:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            lidar = np.nanmean(lidar.reshape(-1, nsps), axis=1)
    return lidar


def read_hires_dat(filename: Union[str, Path], per_spectra: bool = True) -> NDArray:
    """
    Compatibility wrapper for :func:`read_lidar`.

    Args:
        filename (Union[str, Path]): path to `*_hires.dat` containing LiDAR data.
        per_spectra (bool, optional): Return one sample per spectrum when true, or all raw samples.
            Defaults to True.

    Returns:
        NDArray: LiDAR/profilometer data
    """
    return read_lidar(filename, per_spectra=per_spectra)


__all__ = ["read_lidar"]
