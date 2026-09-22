"""Complete TSG package assembly."""

from pathlib import Path
from typing import Union

from .._inventory import discover_package
from ..models import TSG, Cras
from .cras import extract_chips, read_cras
from .lidar import read_lidar
from .spectra import read_spectra


def _read_tsg_package(
    foldername: Union[str, Path],
    read_cras_file: bool = False,
    extract_cras: bool = False,
    imageoutput: Union[str, Path, None] = None,
    backing_file: Union[Path, str, None] = None,
) -> TSG:
    """
    Read all supported components in a TSG package.

    This is the internal package-assembly implementation. The public facade exposes
    the smaller ``read_tsg(path, *, include_cras=False)`` API, while the legacy
    compatibility facade may continue to pass the advanced CRAS options here.

    Args:
        foldername (Union[str, Path]): Path to folder containing the TSG dataset
        read_cras_file (bool, optional): Load raster imagery (can be large!). Defaults to False.
        extract_cras (bool, optional): Extract imagery to .jpg files on load. Defaults to False.
        imageoutput (Union[str, Path, None], optional): location to extract cras to. Defaults to None.
        backing_file (Union[Path, str, None], optional): path to store cras as .zarr if data
            is too large. Defaults to None.

    Raises:
        ValueError: when NIR .tsg and .bip files are missing.

    Returns:
        TSG: A :class:`TSG` object which contains the spectra/scalars/imagery from the TSG dataset
    """
    folder = Path(foldername)
    file_pairs = discover_package(folder)

    nir_pair = file_pairs._get_bip_tsg_pair("nir")
    if nir_pair is None:
        raise ValueError("Missing required NIR spectral pair: expected .tsg and .bip files.")
    nir = read_spectra(nir_pair[0], nir_pair[1], spectrum_name="nir")

    tir_pair = file_pairs._get_bip_tsg_pair("tir")
    tir = None if tir_pair is None else read_spectra(tir_pair[0], tir_pair[1], spectrum_name="tir")

    mir_pair = file_pairs._get_bip_tsg_pair("mir")
    mir = None if mir_pair is None else read_spectra(mir_pair[0], mir_pair[1], spectrum_name="mir")

    lidar_file = file_pairs._get_lidar()
    lidar = None if lidar_file is None else read_lidar(lidar_file)

    cras: Cras | None = None
    cras_file = file_pairs._get_cras()
    if cras_file is not None and read_cras_file:
        if extract_cras:
            output_folder = folder.joinpath("IMG") if imageoutput is None else Path(imageoutput)
            if not output_folder.exists():
                output_folder.mkdir()
            extract_chips(cras_file, output_folder, nir)
        cras = read_cras(cras_file, backing_file)

    return TSG(nir, tir, mir, cras, lidar)
