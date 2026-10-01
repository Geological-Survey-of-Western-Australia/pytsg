"""Data models used by :mod:`pytsg`."""

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

NDArrayOrZarrArray = NDArray | Any
"""An object of type ``np.ndarray`` or ``zarr.Array``."""


class ClassHeaders(NamedTuple):
    """Class definition."""

    class_number: int
    """Class number from the section. e.g. 3 from "[Class 3]"""
    name: str
    """Class name"""
    max: int
    """Number of classes"""
    classes: dict[int, str]
    """Mapping of class ids to class labels"""
    colors: list
    """Colours associated with each class as BGR colour integers"""

    def map_ints(self, index: NDArray) -> list[str]:
        """
        Returns the corresponding classes based on the class id.

        Args:
            index (NDArray): array of class ids

        Returns:
            list[str]: List of corresponding class labels
        """
        outindex: list[str] = []
        for i in index:
            if i >= 0:
                outindex.append(self.classes[i])
            else:
                outindex.append("")
        return outindex


class CrasHeader(NamedTuple):
    """CRAS imagery metadata."""

    id: str
    """starts with 'CoreLog Linescan'"""
    ns: int
    """image width in pixels"""
    nl: int
    """image height in lines"""
    nb: int
    """number of bands"""
    org: int
    """interleave (1=BIL, 2=BIP)"""
    dtype: int
    """datatype (unused; always byte)"""
    specny: int
    """number of linescan lines per dataset sample"""
    specnx: int
    """unused"""
    specpx: int
    """unused across-scan position"""
    ctype: int
    """compression type (0=uncompressed, 1=JPEG chunks)"""
    chunksize: int
    """number of image lines per JPEG chunk"""
    nchunks: int
    """number of compressed image chunks"""
    csize32_obs: int
    """obsolete compressed image size field"""
    ntrays: int
    """number of tray records after image data"""
    nsections: int
    """number of section records after image data"""
    finerep: int
    """chip-mode spectral measurements per chip bucket"""
    jpqual: int
    """JPEG quality factor"""


class TrayInfo(NamedTuple):
    """Linescan Tray Metadata."""

    utlengthmm: float
    """untrimmed length of tray imagery in mm"""
    baseheightmm: float
    """height of bottom of tray above table"""
    coreheightmm: float
    """height of the core above the table"""
    nsections: int
    """number of core sections"""
    nlines: int
    """number of image lines in this tray"""


class SectionInfo(NamedTuple):
    """Linescan Section metadata."""

    utlengthmm: float
    """untrimmed length of imagery in mm"""
    startmm: float
    """start position along scan in mm"""
    endmm: float
    """end position in mm"""
    trimwidthmm: float
    """active image width in mm"""
    startcol: int
    """first active image column"""
    endcol: int
    """final active image column"""
    nlines: int
    """number of image lines in this section"""


class BandHeaders(NamedTuple):
    """Metadata associated with each scalar column. e.g. associated classes."""

    band: int
    """Index of the band"""
    name: str
    """Name of the band"""
    class_number: Union[int, str, float]
    """id of the classes used in this scalar column"""
    flag: int
    """
    Scalar Type:

    * **2:** Imported or user-class scalar
    * **8:** Batch scalar
    * **9:** Core Logging Scalar
    * **10:** AuxMatch Scalar
    * **13:** PLS prediction scalar
    """


@dataclass
class Cras:
    """The CRAS file (compressed raster) contains core imagery."""

    image: NDArrayOrZarrArray
    """Typically an ``np.ndarray`` or a ``zarr.Array`` with a backing file."""
    tray: list[TrayInfo]
    """Linescan tray metadata"""
    section: list[SectionInfo]
    """Linescan section metadata"""

    def __repr__(self) -> str:
        """
        Return string representation of the Cras object.

        Returns:
            str: Representation of Cras object
        """
        return (
            f"Cras(image_shape={self.image.shape}, "
            f"image_dtype={self.image.dtype}, "
            f"tray_count={len(self.tray)}, "
            f"section_count={len(self.section)})"
        )

@dataclass
class Spectra:
    """The raw spectra, bandheaders, and scalar data for a given spectral range."""

    spectrum_name: str
    """Name of the spectral range (e.g. NIR)"""
    spectra: NDArray
    """The RAW spectra"""
    wavelength: NDArray
    """Wavelengths"""
    sampleheaders: pd.DataFrame
    """Index information associated with each spectrum sample"""
    classes: list[dict[str, Any]]
    """Class definitions. e.g. Mineral species/groups"""
    bandheaders: list[BandHeaders]
    """Metadata associated with each scalar column. e.g. associated classes"""
    scalars: pd.DataFrame
    """Scalar data associated with, or derived from, the spectra"""

    def __repr__(self) -> str:
        """
        Return string representation of the Spectra object.

        Returns:
            str: Representation of Spectra object
        """
        return (
            f"Spectra("
            f"spectrum_name={self.spectrum_name!r}, "
            f"spectra_shape={self.spectra.shape}, "
            f"spectra_dtype={self.spectra.dtype}, "
            f"wavelength_shape={self.wavelength.shape}, "
            f"sampleheaders_shape={self.sampleheaders.shape}, "
            f"scalars_shape={self.scalars.shape})"
        )

@dataclass
class TSG:
    """A collection of all the data in a TSG dataset."""

    nir: Spectra
    """Near-infrared spectra and scalar data"""
    tir: Optional[Spectra] = None
    """Thermal-infrared spectra and scalar data"""
    mir: Optional[Spectra] = None
    """Mid-infrared spectra and scalar data"""
    cras: Optional[Cras] = None
    """Imagery data"""
    lidar: Optional[NDArray] = None
    """LiDAR/profilemetry data"""

    @property
    def available(self) -> tuple[str, ...]:
        """Return the names of components currently present in this dataset."""
        component_names = ("nir", "tir", "mir", "cras", "lidar")
        return tuple(name for name in component_names if getattr(self, name) is not None)

    @staticmethod
    def _component_summary(name: str, component: object) -> str:
        if component is None:
            return f"{name}: absent"
        if isinstance(component, Spectra):
            return f"{name}: {component.__repr__()}"
        if isinstance(component, Cras):
            return f"{name}: {component.__repr__()}"
        if isinstance(component, np.ndarray):
            return f"{name}: ndarray(shape={component.shape}, dtype={component.dtype})"
        return f"{name}: {type(component).__name__}"

    def __repr__(self) -> str:
        """
        Return string representation of the TSG object.

        Returns:
            str: Representation of TSG object
        """
        lines = ["TSG(", f"  available={self.available!r},"]
        for name in ("nir", "tir", "mir", "cras", "lidar"):
            lines.append(f"  {self._component_summary(name, getattr(self, name))},")
        lines.append(")")
        return "\n".join(lines)


__all__ = [
    "BandHeaders",
    "ClassHeaders",
    "Cras",
    "CrasHeader",
    "NDArrayOrZarrArray",
    "SectionInfo",
    "Spectra",
    "TSG",
    "TrayInfo",
]
