from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytsg")
except PackageNotFoundError:
    # Package is not installed (e.g. running locally during development)
    __version__ = "0.0.0"
