"""Package discovery and validation for files in a TSG package."""

from pathlib import Path
from typing import Optional, Union


class FilePairs:
    """Keep track of files belonging to one TSG package."""

    nir_tsg: Union[Path, None] = None
    nir_bip: Union[Path, None] = None
    tir_tsg: Union[Path, None] = None
    tir_bip: Union[Path, None] = None
    mir_tsg: Union[Path, None] = None
    mir_bip: Union[Path, None] = None
    lidar: Union[Path, None] = None
    cras: Union[Path, None] = None

    def add_file(self, attribute: str, filename: Path) -> None:
        """Add an inventory file and reject duplicate component files."""
        current = getattr(self, attribute)
        if current is not None:
            raise ValueError(f"Multiple {attribute} files found: {current.name!r} and {filename.name!r}.")
        setattr(self, attribute, filename)

    def validate(self) -> None:
        """Validate all discovered spectral metadata/data file pairs."""
        for spectrum in ("nir", "tir", "mir"):
            tsgfile: Union[Path, None] = getattr(self, f"{spectrum}_tsg")
            bipfile: Union[Path, None] = getattr(self, f"{spectrum}_bip")
            label = spectrum.upper()

            if tsgfile is None and bipfile is None:
                if spectrum == "nir":
                    raise ValueError("Missing required NIR spectral pair: expected .tsg and .bip files.")
                continue

            if tsgfile is None:
                if bipfile is None:
                    raise ValueError(f"Missing {label} spectral pair: expected .tsg and .bip files.")
                raise ValueError(
                    f"Incomplete {label} spectral pair: found {bipfile.name!r} without a matching .tsg file."
                )
            if bipfile is None:
                raise ValueError(
                    f"Incomplete {label} spectral pair: found {tsgfile.name!r} without a matching .bip file."
                )
            if tsgfile.stem != bipfile.stem:
                raise ValueError(
                    f"Mismatched {label} spectral pair: {tsgfile.name!r} and {bipfile.name!r} "
                    "have different stems."
                )

    def _get_bip_tsg_pair(self, spectrum: str) -> Optional[tuple[Path, Path]]:
        tsgfile: Union[Path, None] = getattr(self, f"{spectrum}_tsg")
        bipfile: Union[Path, None] = getattr(self, f"{spectrum}_bip")

        if isinstance(tsgfile, Path) and isinstance(bipfile, Path) and tsgfile.stem == bipfile.stem:
            return (tsgfile, bipfile)
        return None

    def _get_lidar(self) -> Union[Path, None]:
        if isinstance(self.lidar, Path):
            return self.lidar
        return None

    def _get_cras(self) -> Union[Path, None]:
        if isinstance(self.cras, Path):
            return self.cras
        return None

    def valid_nir(self) -> bool:
        return self._get_bip_tsg_pair("nir") is not None

    def valid_tir(self) -> bool:
        return self._get_bip_tsg_pair("tir") is not None

    def valid_mir(self) -> bool:
        return self._get_bip_tsg_pair("mir") is not None

    def valid_lidar(self) -> bool:
        return self._get_lidar() is not None

    def valid_cras(self) -> bool:
        return self._get_cras() is not None


def _classify_file(name: str) -> Optional[str]:
    """Return the inventory attribute for one supported package filename."""
    name = name.casefold()
    if name.endswith("tsg_tir.tsg"):
        return "tir_tsg"
    if name.endswith("tsg_tir.bip"):
        return "tir_bip"
    if name.endswith("tsg_mir.tsg"):
        return "mir_tsg"
    if name.endswith("tsg_mir.bip"):
        return "mir_bip"
    if name.endswith("tsg.tsg"):
        return "nir_tsg"
    if name.endswith("tsg.bip"):
        return "nir_bip"
    if name.endswith("tsg_cras.bip"):
        return "cras"
    if name.endswith("hires.dat"):
        return "lidar"
    return None


def discover_package(path: Union[str, Path]) -> FilePairs:
    """Discover and validate supported files in a TSG package directory.

    Discovery is deterministic and does not open or decode any data files. The returned
    :class:`FilePairs` contains only supported package components.
    """
    folder = Path(path)
    if not folder.exists():
        raise FileNotFoundError("The directory does not exist.")

    file_pairs = FilePairs()
    files = sorted(
        (item for item in folder.iterdir() if item.is_file()),
        key=lambda item: (item.name.casefold(), item.name),
    )
    for file in files:
        attribute = _classify_file(file.name)
        if attribute is not None:
            file_pairs.add_file(attribute, file)

    file_pairs.validate()
    return file_pairs


__all__ = ["FilePairs", "discover_package"]
