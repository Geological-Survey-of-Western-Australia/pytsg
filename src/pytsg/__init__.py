from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytsg")
except PackageNotFoundError:
    # Package is not installed (e.g. running locally during development)
    __version__ = "0.0.0"

from .api import read_cras, read_lidar, read_spectra, read_tsg
from .parse_tsg import TSG, Cras, Spectra

__all__ = [
    "__version__",
    "Cras",
    "Spectra",
    "TSG",
    "read_cras",
    "read_lidar",
    "read_spectra",
    "read_tsg",
]
