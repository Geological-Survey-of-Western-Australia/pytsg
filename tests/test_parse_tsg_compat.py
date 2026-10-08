"""Test that the deprecated parse_tsg API remains functional (until it's removed)."""

import shutil
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import pytest

from pytsg import parse_tsg
from pytsg.parse_tsg import TSG, BandHeaders, ClassHeaders, Cras, SectionInfo, Spectra, TrayInfo

EXAMPLE_DATA = Path(__file__).parent.parent / "example_data"


class SpectrumExpectation(TypedDict):
    spectra: tuple[int, int]
    sampleheaders: tuple[int, int]
    scalars: tuple[int, int]
    wavelength: tuple[float, float]
    classes: int


class PackageExpectation(TypedDict):
    nir: SpectrumExpectation
    tir: SpectrumExpectation | None
    mir: SpectrumExpectation | None
    lidar: tuple[int, ...] | None


PACKAGE_EXPECTATIONS: dict[str, PackageExpectation] = {
    "SWMB007s": {
        "nir": {
            "spectra": (30, 531),
            "sampleheaders": (30, 6),
            "scalars": (30, 148),
            "wavelength": (380.0, 2500.0),
            "classes": 11,
        },
        "tir": {
            "spectra": (30, 341),
            "sampleheaders": (30, 6),
            "scalars": (30, 133),
            "wavelength": (6000.0, 14500.0),
            "classes": 9,
        },
        "mir": None,
        "lidar": None,
    },
    "SWMB007d": {
        "nir": {
            "spectra": (72, 531),
            "sampleheaders": (72, 6),
            "scalars": (72, 148),
            "wavelength": (380.0, 2500.0),
            "classes": 11,
        },
        "tir": {
            "spectra": (72, 341),
            "sampleheaders": (72, 6),
            "scalars": (72, 133),
            "wavelength": (6000.0, 14500.0),
            "classes": 9,
        },
        "mir": None,
        "lidar": None,
    },
    "SWMB008d": {
        "nir": {
            "spectra": (54, 531),
            "sampleheaders": (54, 6),
            "scalars": (54, 148),
            "wavelength": (380.0, 2500.0),
            "classes": 11,
        },
        "tir": {
            "spectra": (54, 341),
            "sampleheaders": (54, 6),
            "scalars": (54, 133),
            "wavelength": (6000.0, 14500.0),
            "classes": 9,
        },
        "mir": None,
        "lidar": None,
    },
    "PE257D": {
        "nir": {
            "spectra": (625, 531),
            "sampleheaders": (625, 7),
            "scalars": (625, 123),
            "wavelength": (380.0, 2500.0),
            "classes": 11,
        },
        "tir": {
            "spectra": (625, 387),
            "sampleheaders": (625, 7),
            "scalars": (625, 123),
            "wavelength": (5350.0, 15000.0),
            "classes": 9,
        },
        "mir": {
            "spectra": (625, 559),
            "sampleheaders": (625, 7),
            "scalars": (625, 93),
            "wavelength": (2000.0, 5348.0),
            "classes": 5,
        },
        "lidar": (625,),
    },
    "GSNSW_testrocks": {
        "nir": {
            "spectra": (222, 526),
            "sampleheaders": (222, 7),
            "scalars": (222, 137),
            "wavelength": (400.0, 2500.0),
            "classes": 11,
        },
        "tir": {
            "spectra": (222, 341),
            "sampleheaders": (222, 7),
            "scalars": (222, 126),
            "wavelength": (6000.0, 14500.0),
            "classes": 7,
        },
        "mir": {
            "spectra": (222, 438),
            "sampleheaders": (222, 7),
            "scalars": (222, 108),
            "wavelength": (2000.0, 5496.0),
            "classes": 5,
        },
        "lidar": (222,),
    },
}


@pytest.mark.parametrize("package_name", sorted(PACKAGE_EXPECTATIONS))
def test_read_package_characterizes_supported_packages(package_name):
    """Record current package component types, dimensions, and relationships."""
    expected = PACKAGE_EXPECTATIONS[package_name]
    package = parse_tsg.read_package(EXAMPLE_DATA / package_name)

    assert isinstance(package, TSG)
    assert isinstance(package.nir, Spectra)

    for spectrum_name in ("nir", "tir", "mir"):
        expected_spectrum = expected[spectrum_name]
        component = getattr(package, spectrum_name)
        if expected_spectrum is None:
            assert component is None
            continue

        assert isinstance(component, Spectra)
        assert component.spectrum_name == spectrum_name
        assert component.spectra.shape == expected_spectrum["spectra"]
        assert component.spectra.dtype == np.float32
        assert component.sampleheaders.shape == expected_spectrum["sampleheaders"]
        assert isinstance(component.sampleheaders, pd.DataFrame)
        assert component.scalars.shape == expected_spectrum["scalars"]
        assert isinstance(component.scalars, pd.DataFrame)
        assert component.wavelength.shape == (expected_spectrum["spectra"][1],)
        assert component.wavelength.dtype == np.float64
        np.testing.assert_allclose(
            component.wavelength[[0, -1]],
            expected_spectrum["wavelength"],
        )
        assert np.all(np.diff(component.wavelength) > 0)

        assert component.spectra.shape == (
            component.sampleheaders.shape[0],
            component.wavelength.shape[0],
        )
        assert component.scalars.shape[0] == component.spectra.shape[0]
        assert len(component.bandheaders) == component.scalars.shape[1]
        assert len(component.classes) == expected_spectrum["classes"]
        assert isinstance(component.classes, dict)
        assert all(isinstance(value, ClassHeaders) for value in component.classes.values())
        assert all(isinstance(value, BandHeaders) for value in component.bandheaders)

    # read_cras_file defaults to False, so an unread optional image is None.
    assert package.cras is None

    if expected["lidar"] is None:
        assert package.lidar is None
    else:
        assert isinstance(package.lidar, np.ndarray)
        assert package.lidar.shape == expected["lidar"]
        assert package.lidar.dtype == np.float32


def test_read_package_nir_only_optional_components(tmp_path):
    """Characterize a valid NIR-only package assembled from repository fixtures."""
    source = EXAMPLE_DATA / "SWMB007s"
    for source_file in source.glob("*tsg.tsg"):
        shutil.copy2(source_file, tmp_path / source_file.name)
    for source_file in source.glob("*tsg.bip"):
        shutil.copy2(source_file, tmp_path / source_file.name)

    package = parse_tsg.read_package(tmp_path)

    assert isinstance(package.nir, Spectra)
    assert package.nir.spectra.shape == PACKAGE_EXPECTATIONS["SWMB007s"]["nir"]["spectra"]
    assert package.tir is None
    assert package.mir is None
    assert package.cras is None
    assert package.lidar is None


def test_read_package_with_cras_reads_decoded_image():
    """Record the current opt-in package CRAS image and table behavior."""
    package = parse_tsg.read_package(EXAMPLE_DATA / "SWMB007s", read_cras_file=True)

    assert isinstance(package.cras, Cras)
    assert package.cras.image.shape == (12360, 915, 3)
    assert package.cras.image.dtype == np.uint8
    assert len(package.cras.tray) == 2
    assert len(package.cras.section) == 2
    assert sum(item.nlines for item in package.cras.tray) == package.cras.image.shape[0]
    assert sum(item.nlines for item in package.cras.section) == package.cras.image.shape[0]
    assert all(isinstance(item, TrayInfo) for item in package.cras.tray)
    assert all(isinstance(item, SectionInfo) for item in package.cras.section)


@pytest.mark.parametrize(
    ("package_name", "spectrum_name"),
    [
        ("SWMB007s", "nir"),
        ("SWMB007d", "tir"),
        ("PE257D", "mir"),
    ],
)
def test_read_tsg_bip_pair_matches_package_reader(package_name, spectrum_name):
    """The legacy direct spectral reader matches its package-reader component."""
    folder = EXAMPLE_DATA / package_name
    suffix = "" if spectrum_name == "nir" else f"_{spectrum_name}"
    tsg_file = next(folder.glob(f"*tsg{suffix}.tsg"))
    bip_file = next(folder.glob(f"*tsg{suffix}.bip"))

    direct = parse_tsg.read_tsg_bip_pair(tsg_file, bip_file, spectrum_name)
    package_component = getattr(parse_tsg.read_package(folder), spectrum_name)

    assert isinstance(direct, Spectra)
    assert isinstance(package_component, Spectra)
    assert direct.spectrum_name == spectrum_name
    np.testing.assert_array_equal(direct.spectra, package_component.spectra)
    np.testing.assert_array_equal(direct.wavelength, package_component.wavelength)
    pd.testing.assert_frame_equal(direct.sampleheaders, package_component.sampleheaders)
    pd.testing.assert_frame_equal(direct.scalars, package_component.scalars)
    assert direct.classes == package_component.classes
    assert direct.bandheaders == package_component.bandheaders


def test_read_cras_decodes_image_and_metadata_tables():
    """Record direct CRAS reader output types, dimensions, and table relationships."""
    cras_file = next((EXAMPLE_DATA / "SWMB007s").glob("*tsg_cras.bip"))
    cras = parse_tsg.read_cras(cras_file)

    assert isinstance(cras, Cras)
    assert isinstance(cras.image, np.ndarray)
    assert cras.image.shape == (12360, 915, 3)
    assert cras.image.dtype == np.uint8
    assert isinstance(cras.tray, list)
    assert isinstance(cras.section, list)
    assert len(cras.tray) == 2
    assert len(cras.section) == 2
    assert sum(item.nlines for item in cras.tray) == cras.image.shape[0]
    assert sum(item.nlines for item in cras.section) == cras.image.shape[0]
    assert all(isinstance(item, TrayInfo) for item in cras.tray)
    assert all(isinstance(item, SectionInfo) for item in cras.section)


@pytest.mark.parametrize(
    ("package_name", "raw_shape", "per_spectra_shape"),
    [
        ("PE257D", (80000,), (625,)),
        ("GSNSW_testrocks", (17760,), (222,)),
    ],
)
def test_read_hires_dat_returns_expected_lidar_shapes(package_name, raw_shape, per_spectra_shape):
    """The legacy direct lidar reader supports both repository high-res files."""
    lidar_file = next((EXAMPLE_DATA / package_name).glob("*hires.dat"))

    raw = parse_tsg.read_hires_dat(lidar_file, per_spectra=False)
    per_spectra = parse_tsg.read_hires_dat(lidar_file, per_spectra=True)

    assert isinstance(raw, np.ndarray)
    assert isinstance(per_spectra, np.ndarray)
    assert raw.shape == raw_shape
    assert per_spectra.shape == per_spectra_shape
    assert raw.dtype == np.float32
    assert per_spectra.dtype == np.float32
    assert raw.size % per_spectra.size == 0


def test_read_package_rejects_incomplete_required_nir_pair(tmp_path):
    (tmp_path / "sample_tsg.tsg").touch()

    with pytest.raises(ValueError, match="Incomplete NIR spectral pair"):
        parse_tsg.read_package(tmp_path)


def test_read_package_rejects_incomplete_optional_pair(tmp_path):
    source = EXAMPLE_DATA / "SWMB007s"
    for source_file in source.glob("*tsg.tsg"):
        shutil.copy2(source_file, tmp_path / source_file.name)
    for source_file in source.glob("*tsg.bip"):
        shutil.copy2(source_file, tmp_path / source_file.name)
    (tmp_path / "sample_tsg_tir.tsg").touch()

    with pytest.raises(ValueError, match="Incomplete TIR spectral pair"):
        parse_tsg.read_package(tmp_path)


def test_read_package_rejects_mismatched_optional_pair(tmp_path):
    source = EXAMPLE_DATA / "SWMB007s"
    for source_file in source.glob("*tsg.tsg"):
        shutil.copy2(source_file, tmp_path / source_file.name)
    for source_file in source.glob("*tsg.bip"):
        shutil.copy2(source_file, tmp_path / source_file.name)
    (tmp_path / "first_tsg_tir.tsg").touch()
    (tmp_path / "second_tsg_tir.bip").touch()

    with pytest.raises(ValueError, match="Mismatched TIR spectral pair"):
        parse_tsg.read_package(tmp_path)


def test_read_package_result_remains_usable_after_return():
    package = parse_tsg.read_package(EXAMPLE_DATA / "SWMB007s")
    original_value = package.nir.spectra[0, 0]

    package.nir.spectra[0, 0] = original_value + 1
    package.nir.sampleheaders.iloc[0, 0] = "changed-in-memory"

    assert package.nir.spectra[0, 0] == original_value + 1
    assert package.nir.sampleheaders.iloc[0, 0] == "changed-in-memory"
