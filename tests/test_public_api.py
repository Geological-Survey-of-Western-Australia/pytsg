"""Tests for the public API."""

from pathlib import Path

import numpy as np
import pandas as pd

import pytsg
from pytsg import TSG, Cras, Spectra, parse_tsg, read_cras, read_lidar, read_spectra, read_tsg

EXAMPLE_DATA = Path(__file__).parent.parent / "example_data"
SWMB007S = EXAMPLE_DATA / "SWMB007s"


def _spectral_paths(package: Path, spectrum_name: str = "nir") -> tuple[Path, Path]:
    suffix = "" if spectrum_name == "nir" else f"_{spectrum_name}"
    tsg_file = next(package.glob(f"*tsg{suffix}.tsg"))
    bip_file = next(package.glob(f"*tsg{suffix}.bip"))
    return tsg_file, bip_file


def test_public_complete_reader_returns_tsg_for_path_and_string():
    path_result = read_tsg(SWMB007S)
    string_result = read_tsg(str(SWMB007S))

    assert isinstance(path_result, TSG)
    assert isinstance(path_result.nir, Spectra)
    assert path_result.tir is not None
    assert path_result.mir is None
    assert path_result.cras is None
    assert path_result.lidar is None
    np.testing.assert_array_equal(path_result.nir.spectra, string_result.nir.spectra)


def test_public_complete_reader_can_include_cras():
    dataset = read_tsg(SWMB007S, include_cras=True)

    assert isinstance(dataset.cras, Cras)
    assert dataset.cras.image.shape == (12360, 915, 3)


def test_public_component_readers_match_legacy_readers():
    tsg_file, bip_file = _spectral_paths(SWMB007S)
    public_spectra = read_spectra(
        str(tsg_file),
        str(bip_file),
        spectrum_name="nir",
    )
    legacy_spectra = parse_tsg.read_tsg_bip_pair(tsg_file, bip_file, "nir")

    assert isinstance(public_spectra, Spectra)
    np.testing.assert_array_equal(public_spectra.spectra, legacy_spectra.spectra)
    np.testing.assert_array_equal(public_spectra.wavelength, legacy_spectra.wavelength)
    pd.testing.assert_frame_equal(public_spectra.sampleheaders, legacy_spectra.sampleheaders)
    pd.testing.assert_frame_equal(public_spectra.scalars, legacy_spectra.scalars)

    cras_file = next(SWMB007S.glob("*tsg_cras.bip"))
    public_cras = read_cras(str(cras_file))
    assert isinstance(public_cras, Cras)
    assert public_cras.image.shape == (12360, 915, 3)

    lidar_file = next((EXAMPLE_DATA / "PE257D").glob("*tsg_hires.dat"))
    public_lidar = read_lidar(str(lidar_file))
    legacy_lidar = parse_tsg.read_hires_dat(lidar_file)
    assert isinstance(public_lidar, np.ndarray)
    np.testing.assert_allclose(public_lidar, legacy_lidar, equal_nan=True)


def test_legacy_read_package_delegates_to_public_reader(monkeypatch):
    sentinel = object()
    calls = {}

    def fake_read_tsg(path, *, include_cras=False):
        calls["path"] = path
        calls["include_cras"] = include_cras
        return sentinel

    monkeypatch.setattr(pytsg, "read_tsg", fake_read_tsg)

    result = parse_tsg.read_package(SWMB007S, read_cras_file=True)

    assert result is sentinel
    assert calls == {"path": SWMB007S, "include_cras": True}


def test_legacy_read_package_matches_public_reader():
    public_dataset = read_tsg(SWMB007S)
    legacy_dataset = parse_tsg.read_package(SWMB007S)

    np.testing.assert_array_equal(public_dataset.nir.spectra, legacy_dataset.nir.spectra)
    np.testing.assert_array_equal(public_dataset.nir.wavelength, legacy_dataset.nir.wavelength)
    pd.testing.assert_frame_equal(public_dataset.nir.sampleheaders, legacy_dataset.nir.sampleheaders)
    pd.testing.assert_frame_equal(public_dataset.nir.scalars, legacy_dataset.nir.scalars)
    assert public_dataset.tir is not None
    assert legacy_dataset.tir is not None
    np.testing.assert_array_equal(public_dataset.tir.spectra, legacy_dataset.tir.spectra)
    assert public_dataset.mir is None
    assert legacy_dataset.mir is None
    assert public_dataset.cras is None
    assert legacy_dataset.cras is None
    assert public_dataset.lidar is None
    assert legacy_dataset.lidar is None


def test_public_reader_mutation_does_not_write_back_to_source():
    tsg_file, bip_file = _spectral_paths(SWMB007S)
    source_tsg = tsg_file.read_bytes()
    source_bip = bip_file.read_bytes()

    dataset = read_tsg(SWMB007S)
    original_spectra = dataset.nir.spectra.copy()
    original_sampleheaders = dataset.nir.sampleheaders.copy(deep=True)

    dataset.nir.spectra[0, 0] += np.float32(1)
    dataset.nir.sampleheaders.iloc[0, 0] = "changed-in-memory"

    assert tsg_file.read_bytes() == source_tsg
    assert bip_file.read_bytes() == source_bip

    reread = read_tsg(SWMB007S)
    np.testing.assert_array_equal(reread.nir.spectra, original_spectra)
    pd.testing.assert_frame_equal(reread.nir.sampleheaders, original_sampleheaders)
