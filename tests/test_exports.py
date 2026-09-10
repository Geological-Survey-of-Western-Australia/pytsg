"""Tests for standard NumPy/Pandas interaction and export boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytsg import read_tsg

EXAMPLE_DATA = Path(__file__).parent.parent / "example_data"
PARQUET_ENGINE_AVAILABLE = any(
    importlib.util.find_spec(name) is not None for name in ("pyarrow", "fastparquet")
)


def _nir_paths() -> tuple[Path, Path]:
    package = EXAMPLE_DATA / "SWMB007s"
    return next(package.glob("*tsg.tsg")), next(package.glob("*tsg.bip"))


def test_standard_numpy_and_pandas_exports_round_trip(tmp_path):
    dataset = read_tsg(EXAMPLE_DATA / "SWMB007s")

    sampleheaders_path = tmp_path / "sampleheaders.csv"
    scalars_path = tmp_path / "scalars.csv"
    spectra_path = tmp_path / "spectra.npy"
    wavelength_path = tmp_path / "wavelength.npy"

    dataset.nir.sampleheaders.to_csv(sampleheaders_path, index=False)
    dataset.nir.scalars.to_csv(scalars_path, index=False)
    np.save(spectra_path, dataset.nir.spectra)
    np.save(wavelength_path, dataset.nir.wavelength)

    sampleheaders = pd.read_csv(sampleheaders_path)
    scalars = pd.read_csv(scalars_path)
    spectra = np.load(spectra_path)
    wavelength = np.load(wavelength_path)

    assert sampleheaders.shape == dataset.nir.sampleheaders.shape
    assert list(sampleheaders.columns) == list(dataset.nir.sampleheaders.columns)
    assert scalars.shape == dataset.nir.scalars.shape
    np.testing.assert_array_equal(spectra, dataset.nir.spectra)
    np.testing.assert_array_equal(wavelength, dataset.nir.wavelength)


@pytest.mark.skipif(
    not PARQUET_ENGINE_AVAILABLE,
    reason="Parquet export requires an optional pyarrow or fastparquet engine",
)
def test_standard_parquet_export_round_trip_when_engine_available(tmp_path):
    dataset = read_tsg(EXAMPLE_DATA / "SWMB007s")
    parquet_path = tmp_path / "sampleheaders.parquet"

    dataset.nir.sampleheaders.to_parquet(parquet_path)
    sampleheaders = pd.read_parquet(parquet_path)

    assert sampleheaders.shape == dataset.nir.sampleheaders.shape
    assert list(sampleheaders.columns) == list(dataset.nir.sampleheaders.columns)


def test_standard_exports_and_mutation_do_not_modify_source(tmp_path):
    tsg_file, bip_file = _nir_paths()
    source_tsg = tsg_file.read_bytes()
    source_bip = bip_file.read_bytes()

    dataset = read_tsg(EXAMPLE_DATA / "SWMB007s")
    original_spectra = dataset.nir.spectra.copy()
    original_sampleheaders = dataset.nir.sampleheaders.copy(deep=True)

    dataset.nir.spectra[0, 0] += np.float32(1)
    dataset.nir.sampleheaders.iloc[0, 0] = "changed-in-memory"
    dataset.nir.sampleheaders.to_csv(tmp_path / "mutated-sampleheaders.csv", index=False)
    np.save(tmp_path / "mutated-spectra.npy", dataset.nir.spectra)

    assert tsg_file.read_bytes() == source_tsg
    assert bip_file.read_bytes() == source_bip

    reread = read_tsg(EXAMPLE_DATA / "SWMB007s")
    np.testing.assert_array_equal(reread.nir.spectra, original_spectra)
    pd.testing.assert_frame_equal(reread.nir.sampleheaders, original_sampleheaders)
