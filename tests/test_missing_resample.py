"""Regression checks for data preparation used in the teaching notebooks."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "functions"))
from rep_missing import rep_missing
from resample import resample


class MissingDataTests(unittest.TestCase):
    def test_replacement_preserves_input_and_fractional_mean(self):
        original = np.array([1, 0, 4])
        actual = rep_missing(original, value=0, new_value="mean")
        np.testing.assert_array_equal(original, [1, 0, 4])
        np.testing.assert_allclose(actual, [1, 2.5, 4])

    def test_median_excludes_missing_sentinel(self):
        np.testing.assert_allclose(
            rep_missing([1, 0, 9], value=0, new_value="median"), [1, 5, 9]
        )

    def test_interpolation_fills_only_missing_positions(self):
        np.testing.assert_allclose(
            rep_missing([np.nan, 2, np.nan, 6, np.nan]), [2, 2, 4, 6, 6]
        )

    def test_all_missing_and_unknown_method_fail_clearly(self):
        for values, method in [([np.nan, np.nan], "mean"), ([1, np.nan], "typo")]:
            with self.subTest(method=method), self.assertRaises(ValueError):
                rep_missing(values, new_value=method)


class ResampleTests(unittest.TestCase):
    def test_frame_time_origin_and_effective_frequency_are_preserved(self):
        frame = pd.DataFrame({"x": np.arange(100.0)}, index=3 + np.arange(100) / 100)
        frame.index.name = "Time"
        result = resample(frame, freq_new=150, freq_old=100)
        self.assertEqual(result.shape, (150, 1))
        self.assertEqual(result.index[0], 3)
        self.assertEqual(result.index.name, "Time")
        np.testing.assert_allclose(np.diff(result.index), 1 / 150)

    def test_downsampling_attenuates_above_new_nyquist(self):
        t = np.arange(1000) / 1000
        result = resample(np.sin(2 * np.pi * 300 * t)[:, None], 100, 1000)
        self.assertLess(np.std(result[10:-10]), 0.01)

    def test_invalid_method_and_irregular_times_are_rejected(self):
        with self.assertRaises(ValueError):
            resample(np.ones((10, 1)), 100, 50, method="typo")
        with self.assertRaises(ValueError):
            resample(pd.DataFrame({"x": [1, 2, 3]}, index=[0, 0.01, 0.03]), 100)


if __name__ == "__main__":
    unittest.main()


class MissingInterpolationRegressionTests(unittest.TestCase):
    def test_custom_sentinel_absent_still_interpolates_nan(self):
        np.testing.assert_allclose(rep_missing([1, np.nan, 3], value=0), [1, 2, 3])
