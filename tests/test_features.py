import numpy as np
import pytest

from pytsg import feature


@pytest.fixture
def wavelength():
    return np.arange(-10, 10)


@pytest.fixture
def single_spectra(wavelength):
    return feature.gaussian(wavelength, -1, 0, 5).reshape(1, -1)


@pytest.fixture
def multi_spectra(single_spectra):
    return np.vstack([single_spectra] * 2)


def test_band_statistics_1d(single_spectra):
    br = feature.band_extractor(single_spectra)
    expectation = np.asarray([10, -1, -11.953864]).reshape(1, 3)
    np.testing.assert_allclose(br, expectation, atol=10e-3)


def test_band_statistics_2d(multi_spectra):
    br = feature.band_extractor(multi_spectra)
    expectation = np.asarray([10, -1, -11.953864]).reshape(1, 3)
    np.testing.assert_allclose(br, np.vstack([expectation] * 2), atol=10e-3)


def test_sqm_1d(wavelength, single_spectra):
    results, _ = feature.sqm(wavelength, single_spectra)
    expectation = np.asanyarray([-0.12984183, -0.9112505, 19.70058995]).reshape(1, 3)
    np.testing.assert_allclose(results, expectation, atol=10e-3)


def test_sqm_2d(wavelength, multi_spectra):
    results, _ = feature.sqm(wavelength, multi_spectra)
    expectation = np.asanyarray([-0.12984183, -0.9112505, 19.70058995]).reshape(1, 3)
    np.testing.assert_allclose(results, np.vstack([expectation] * 2), atol=10e-3)


def test_gaussian_1d(wavelength, single_spectra):
    results = feature.fit_gaussian(wavelength, single_spectra, [1, 9, -5])
    expectation = np.asarray([-1, 0, 5]).reshape(1, 3)
    np.testing.assert_allclose(results, expectation, atol=10e-3)


def test_gaussian_2d(wavelength, multi_spectra):
    results = feature.fit_gaussian(wavelength, multi_spectra, [1, 9, -5])
    expectation = np.asarray([-1, 0, 5]).reshape(1, 3)
    np.testing.assert_allclose(results, np.vstack([expectation] * 2), atol=10e-3)
