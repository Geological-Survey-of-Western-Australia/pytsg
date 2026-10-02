import tempfile
from pathlib import Path

import pytest

from pytsg import parse_tsg


@pytest.fixture
def example_data_dir():
    return Path(__file__).parent.parent / "example_data" / "SWMB007d"


@pytest.fixture
def cras_fpath(example_data_dir):
    return example_data_dir / "SWMB007d_chips_tsg_cras.bip"


@pytest.fixture
def tsg_fpath(example_data_dir):
    return example_data_dir / "SWMB007d_chips_tsg.tsg"


@pytest.fixture
def bip_fpath(example_data_dir):
    return example_data_dir / "SWMB007d_chips_tsg.bip"


def test_read_package(example_data_dir):
    tmp_data = parse_tsg.read_package(example_data_dir, read_cras_file=True)
    assert hasattr(tmp_data, "nir")
    assert hasattr(tmp_data, "tir")
    assert hasattr(tmp_data, "cras")


def test_extract_chips(cras_fpath, tsg_fpath, bip_fpath):
    spectra = parse_tsg.read_tsg_bip_pair(tsg_fpath, bip_fpath, "nir")
    with tempfile.TemporaryDirectory() as tmpdirname:
        parse_tsg.extract_chips(cras_fpath, tmpdirname, spectra)


def test_generate_chips(cras_fpath, tsg_fpath, bip_fpath):
    spectra = parse_tsg.read_tsg_bip_pair(tsg_fpath, bip_fpath, "nir")
    chip_generator = parse_tsg.generate_chips(cras_fpath, spectra, batch_size=12)
    bsize: list[int] = [12, 12, 12, 12, 12, 12, 0]
    actual: list[int] = []
    for i in chip_generator:
        actual.append(len(i))
    assert bsize == actual
