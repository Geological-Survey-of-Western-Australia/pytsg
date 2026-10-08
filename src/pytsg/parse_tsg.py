"""Compatibility facade for the historical :mod:`pytsg.parse_tsg` module.

New code should use the top-level readers or ``pytsg.readers``. The names in this
module remain available so existing applications can migrate without an import
break, including historically importable private parser helpers.
"""

from pathlib import Path
from typing import Union

from ._inventory import FilePairs
from .models import (
    TSG,
    BandHeaders,
    ClassHeaders,
    Cras,
    CrasHeader,
    NDArrayOrZarrArray,
    SectionInfo,
    Spectra,
    TrayInfo,
)
from .readers import cras as _cras
from .readers import lidar as _lidar
from .readers import package as _package
from .readers import spectra as _spectra

# Public legacy names are aliases to the focused implementations.
composite_spectra = _cras.composite_spectra
extract_chips = _cras.extract_chips
generate_chips = _cras.generate_chips
read_cras = _cras.read_cras
read_hires_dat = _lidar.read_hires_dat

# Historically importable private parser/package helpers remain compatibility aliases.
_calculate_wavelengths = _spectra._calculate_wavelengths
_find_header_sections = _spectra._find_header_sections
_parse_bandheaders = _spectra._parse_bandheaders
_parse_class_section = _spectra._parse_class_section
_parse_kvp = _spectra._parse_kvp
_parse_sample_header = _spectra._parse_sample_header
_parse_scalars = _spectra._parse_scalars
_parse_section = _spectra._parse_section
_parse_tsg = _spectra._parse_tsg
_parse_wavelength_specs = _spectra._parse_wavelength_specs
_read_bip = _spectra._read_bip
_read_tsg_file = _spectra._read_tsg_file
_read_tsg_package = _package._read_tsg_package

__all__ = [
    "BandHeaders",
    "ClassHeaders",
    "Cras",
    "CrasHeader",
    "FilePairs",
    "NDArrayOrZarrArray",
    "SectionInfo",
    "Spectra",
    "TSG",
    "TrayInfo",
    "composite_spectra",
    "extract_chips",
    "generate_chips",
    "read_cras",
    "read_hires_dat",
    "read_package",
    "read_tsg_bip_pair",
]


def read_tsg_bip_pair(
    tsg_file: Union[Path, str],
    bip_file: Union[Path, str],
    spectrum: str,
) -> Spectra:
    """
    Compatibility wrapper for the historical spectral reader name.

    .. deprecated:: 0.6.0
       Use :func:`pytsg.read_spectra` instead.
    """
    return _spectra.read_spectra(tsg_file, bip_file, spectrum_name=spectrum)


def read_package(
    foldername: Union[str, Path],
    read_cras_file: bool = False,
    extract_cras: bool = False,
    imageoutput: Union[str, Path, None] = None,
    backing_file: Union[Path, str, None] = None,
) -> TSG:
    """
    Read a TSG dataset.

    .. deprecated:: 0.6.0
       Use :func:`pytsg.read_tsg` instead.
    """
    if extract_cras or imageoutput is not None or backing_file is not None:
        return _read_tsg_package(
            foldername,
            read_cras_file=read_cras_file,
            extract_cras=extract_cras,
            imageoutput=imageoutput,
            backing_file=backing_file,
        )

    from pytsg import read_tsg

    return read_tsg(foldername, include_cras=read_cras_file)
