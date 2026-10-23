"""Regression intervals must include the full polynomial design matrix."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

_spec = importlib.util.spec_from_file_location(
    "bmc_polyfit", Path(__file__).resolve().parents[1] / "functions" / "polyfit.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
polyfit = _module.polyfit


class PolynomialIntervalTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-2, 3, 19)
        self.y = 0.3 * self.x**3 - 2 * self.x**2 + self.x + 1
        self.y += 0.15 * np.sin(np.arange(self.x.size))
        self.scales = np.linspace(0.5, 1.5, self.x.size)

    def test_weighted_quadratic_and_cubic_intervals_against_linear_algebra(self):
        for degree in (2, 3):
            with self.subTest(degree=degree):
                design = np.vander(self.x, degree + 1)
                weighted = design / self.scales[:, None]
                expected_coef = np.linalg.lstsq(
                    weighted, self.y / self.scales, rcond=None
                )[0]
                residual = self.y - design @ expected_coef
                dof = self.x.size - degree - 1
                scale = np.sum((residual / self.scales) ** 2) / dof
                covariance = scale * np.linalg.inv(weighted.T @ weighted)
                variance = np.diag(design @ covariance @ design.T)
                result = polyfit(self.x, self.y, degree, self.scales, plot=False)
                np.testing.assert_allclose(result[0], expected_coef, rtol=1e-10)
                np.testing.assert_allclose(
                    result[5], stats.t.ppf(0.975, dof) * np.sqrt(variance), rtol=1e-10
                )
                np.testing.assert_allclose(
                    result[6],
                    stats.t.ppf(0.8415, dof)
                    * np.sqrt(variance + scale * self.scales**2),
                    rtol=1e-10,
                )

    def test_relative_weight_rescaling_preserves_intervals_and_input(self):
        original = self.scales.copy()
        first = polyfit(self.x, self.y, 2, self.scales, plot=False)
        second = polyfit(self.x, self.y, 2, 3 * self.scales, plot=False)
        np.testing.assert_array_equal(self.scales, original)
        for index in (0, 1, 4, 5, 6):
            np.testing.assert_allclose(first[index], second[index], rtol=1e-10)
        self.assertAlmostEqual(first[3] / 9, second[3])

    def test_invalid_noise_scales_are_rejected_without_mutation(self):
        for bad in (0, -1, np.nan, np.inf):
            scales = self.scales.copy()
            scales[3] = bad
            original = scales.copy()
            with self.subTest(scale=bad), self.assertRaises(ValueError):
                polyfit(self.x, self.y, 2, scales, plot=False)
            np.testing.assert_array_equal(scales, original)
