"""Data models used by the public and focused TSG readers."""

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

NDArrayOrZarrArray = NDArray | Any
"""An ``np.ndarray`` or ``zarr.Array``."""


class ClassHeaders(NamedTuple):
    class_number: int
    name: str
    max: int
    classes: dict[int, str]
    colors: list

    def map_ints(self, index: NDArray) -> list[str]:
        outindex: list[str] = []
        for i in index:
            if i >= 0:
                outindex.append(self.classes[i])
            else:
                outindex.append("")
        return outindex


class CrasHeader(NamedTuple):
    id: str  # starts with "CoreLog Linescan ".
    ns: int  # image width in pixels
    nl: int  # image height in lines
    nb: int  # number of bands
    org: int  # interleave (1=BIL, 2=BIP)
    dtype: int  # datatype (unused; always byte)
    specny: int  # number of linescan lines per dataset sample
    specnx: int  # unused
    specpx: int  # unused across-scan position
    ctype: int  # compression type (0=uncompressed, 1=JPEG chunks)
    chunksize: int  # number of image lines per JPEG chunk
    nchunks: int  # number of compressed image chunks
    csize32_obs: int  # obsolete compressed image size field
    ntrays: int  # number of tray records after image data
    nsections: int  # number of section records after image data
    finerep: int  # chip-mode spectral measurements per chip bucket
    jpqual: int  # JPEG quality factor


class TrayInfo(NamedTuple):
    utlengthmm: float  # untrimmed length of tray imagery in mm
    baseheightmm: float  # height of bottom of tray above table
    coreheightmm: float  # height of the core above the table
    nsections: int  # number of core sections
    nlines: int  # number of image lines in this tray


class SectionInfo(NamedTuple):
    utlengthmm: float  # untrimmed length of imagery in mm
    startmm: float  # start position along scan in mm
    endmm: float  # end position in mm
    trimwidthmm: float  # active image width in mm
    startcol: int  # first active image column
    endcol: int  # final active image column
    nlines: int  # number of image lines in this section


class BandHeaders(NamedTuple):
    band: int
    name: str
    class_number: Union[int, str, float]
    flag: int


@dataclass
class Cras:
    """The CRAS file (compressed raster) contains core imagery."""

    image: NDArrayOrZarrArray
    """Typically an ``np.ndarray`` or a ``zarr.Array`` with a backing file."""

    tray: list[TrayInfo]
    section: list[SectionInfo]


@dataclass
class Spectra:
    spectrum_name: str
    spectra: NDArray
    wavelength: NDArray
    sampleheaders: pd.DataFrame
    classes: list[dict[str, Any]]
    bandheaders: list[BandHeaders]
    scalars: pd.DataFrame


@dataclass
class TSG:
    nir: Spectra
    tir: Optional[Spectra] = None
    mir: Optional[Spectra] = None
    cras: Optional[Cras] = None
    lidar: Optional[NDArray] = None

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
            return (
                f"{name}: Spectra("
                f"spectra_shape={component.spectra.shape}, "
                f"spectra_dtype={component.spectra.dtype}, "
                f"wavelength_shape={component.wavelength.shape}, "
                f"sampleheaders_shape={component.sampleheaders.shape}, "
                f"scalars_shape={component.scalars.shape})"
            )
        if isinstance(component, Cras):
            return (
                f"{name}: Cras("
                f"image_shape={component.image.shape}, "
                f"image_dtype={component.image.dtype}, "
                f"tray_count={len(component.tray)}, "
                f"section_count={len(component.section)})"
            )
        if isinstance(component, np.ndarray):
            return f"{name}: ndarray(shape={component.shape}, dtype={component.dtype})"
        return f"{name}: {type(component).__name__}"

    def __repr__(self) -> str:
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
