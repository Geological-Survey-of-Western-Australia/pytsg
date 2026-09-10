"""Tests for TSG inspection and named-component discovery."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytsg import read_tsg

EXAMPLE_DATA = Path(__file__).parent.parent / "example_data"


@pytest.mark.parametrize(
    ("package_name", "expected"),
    [
        ("SWMB007s", ("nir", "tir")),
        ("SWMB007d", ("nir", "tir")),
        ("SWMB008d", ("nir", "tir")),
        ("PE257D", ("nir", "tir", "mir", "lidar")),
        ("GSNSW_testrocks", ("nir", "tir", "mir", "lidar")),
    ],
)
def test_available_reports_present_components(package_name, expected):
    dataset = read_tsg(EXAMPLE_DATA / package_name)

    assert isinstance(dataset.available, tuple)
    assert dataset.available == expected


def test_repr_summarises_spectra_shapes_dtypes_and_tables():
    dataset = read_tsg(EXAMPLE_DATA / "SWMB007s")

    summary = repr(dataset)

    assert summary.startswith("TSG(\n")
    assert "available=('nir', 'tir')" in summary
    assert "nir: Spectra(" in summary
    assert "spectra_shape=(30, 531)" in summary
    assert "spectra_dtype=float32" in summary
    assert "wavelength_shape=(531,)" in summary
    assert "sampleheaders_shape=(30, 6)" in summary
    assert "scalars_shape=(30, 148)" in summary
    assert "tir: Spectra(" in summary
    assert "mir: absent" in summary
    assert "cras: absent" in summary
    assert "lidar: absent" in summary


def test_repr_summarises_cras_and_lidar_components():
    cras_dataset = read_tsg(EXAMPLE_DATA / "SWMB007s", include_cras=True)
    lidar_dataset = read_tsg(EXAMPLE_DATA / "PE257D")

    cras_summary = repr(cras_dataset)
    lidar_summary = repr(lidar_dataset)

    assert "available=('nir', 'tir', 'cras')" in cras_summary
    assert "cras: Cras(" in cras_summary
    assert "image_shape=(12360, 915, 3)" in cras_summary
    assert "image_dtype=uint8" in cras_summary
    assert "tray_count=2" in cras_summary
    assert "section_count=2" in cras_summary
    assert "lidar: ndarray(shape=(625,), dtype=float32)" in lidar_summary


def test_returned_arrays_and_dataframes_remain_mutable():
    dataset = read_tsg(EXAMPLE_DATA / "SWMB007s")
    original_spectrum_value = dataset.nir.spectra[0, 0]
    original_scalar_value = dataset.nir.scalars.iloc[0, 0]

    dataset.nir.spectra[0, 0] = original_spectrum_value + np.float32(1)
    dataset.nir.sampleheaders.iloc[0, 0] = "changed-in-memory"
    dataset.nir.scalars.iloc[0, 0] = -999

    assert dataset.nir.spectra[0, 0] == original_spectrum_value + 1
    assert dataset.nir.sampleheaders.iloc[0, 0] == "changed-in-memory"
    assert dataset.nir.scalars.iloc[0, 0] == -999
    assert original_scalar_value != dataset.nir.scalars.iloc[0, 0]
    assert isinstance(dataset.nir.sampleheaders, pd.DataFrame)
    assert isinstance(dataset.nir.scalars, pd.DataFrame)
