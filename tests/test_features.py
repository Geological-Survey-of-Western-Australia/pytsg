import unittest

import numpy as np

from src.pytsg import feature


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.wavelength = np.arange(-10, 10)
        self.signal = feature.gaussian(self.wavelength, -1, 0, 5).reshape(1, -1)
        self.signal2 = np.vstack([self.signal] * 2)

    def test_band_statistics_1d(self):
        br = feature.band_extractor(self.signal)
        expectation = np.asarray([10, -1, -11.953864]).reshape(1, 3)
        np.testing.assert_allclose(br, expectation, atol=10e-3)

    def test_band_statistics_2d(self):
        br = feature.band_extractor(self.signal2)
        expectation = np.asarray([10, -1, -11.953864]).reshape(1, 3)
        np.testing.assert_allclose(br, np.vstack([expectation] * 2), atol=10e-3)

    def test_sqm_1d(self):
        results, _ = feature.sqm(self.wavelength, self.signal)
        expectation = np.asanyarray([-0.12984183, -0.9112505, 19.70058995]).reshape(
            1, 3
        )
        np.testing.assert_allclose(results, expectation, atol=10e-3)

    def test_sqm_2d(self):
        results, _ = feature.sqm(self.wavelength, self.signal2)
        expectation = np.asanyarray([-0.12984183, -0.9112505, 19.70058995]).reshape(
            1, 3
        )
        np.testing.assert_allclose(results, np.vstack([expectation] * 2), atol=10e-3)

    def test_gaussian_1d(self):
        results = feature.fit_gaussian(self.wavelength, self.signal, [1, 9, -5])
        expectation = np.asarray([-1, 0, 5]).reshape(1, 3)
        np.testing.assert_allclose(results, expectation, atol=10e-3)

    def test_gaussian_2d(self):
        results = feature.fit_gaussian(self.wavelength, self.signal2, [1, 9, -5])
        expectation = np.asarray([-1, 0, 5]).reshape(1, 3)
        np.testing.assert_allclose(results, np.vstack([expectation] * 2), atol=10e-3)

    def tearDown(self):
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
