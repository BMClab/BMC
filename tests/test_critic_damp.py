"""Physical-response regressions for the shared digital filter design."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy import signal

_spec = importlib.util.spec_from_file_location(
    "bmc_critic_damp",
    Path(__file__).resolve().parents[1] / "functions" / "critic_damp.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
critic_damp = _module.critic_damp


class CriticDampTests(unittest.TestCase):
    def test_requested_cutoff_has_combined_half_power(self):
        for kind in ("critic", "butter"):
            for cutoff in (2, 10, 35):
                for passes in (1, 2, 4):
                    with self.subTest(kind=kind, cutoff=cutoff, passes=passes):
                        b, a, design_frequency = critic_damp(
                            cutoff, 100, npass=passes, filt=kind
                        )
                        _, response = signal.freqz(b, a, worN=[cutoff], fs=100)
                        self.assertAlmostEqual(abs(response[0]) ** (2 * passes), 0.5)
                        self.assertLess(design_frequency, 50)
                        self.assertGreater(design_frequency, 0)
                        self.assertTrue(np.all(np.abs(np.roots(a)) < 1))
                        self.assertAlmostEqual(np.sum(b) / np.sum(a), 1)

    def test_butterworth_coefficients_match_scipy_at_returned_design_frequency(self):
        b, a, design_frequency = critic_damp(10, 100, filt="butter")
        self.assertAlmostEqual(design_frequency, 12.249240607627653)
        scipy_b, scipy_a = signal.butter(2, design_frequency, fs=100)
        np.testing.assert_allclose(b, scipy_b)
        np.testing.assert_allclose(a, scipy_a)

    def test_uncorrected_mode_keeps_the_requested_design_frequency(self):
        for kind, power in (("butter", 0.25), ("critic", 0.0625)):
            with self.subTest(kind=kind):
                b, a, frequency = critic_damp(10, 100, fcorr=False, filt=kind)
                self.assertAlmostEqual(frequency, 10)
                _, response = signal.freqz(b, a, worN=[10], fs=100)
                self.assertAlmostEqual(abs(response[0]) ** 4, power)

    def test_invalid_designs_raise_instead_of_falling_back(self):
        cases = [
            {"fcut": 0},
            {"fcut": -1},
            {"fcut": 50},
            {"fcut": 60},
            {"fcut": np.nan},
            {"freq": 0},
            {"freq": np.inf},
            {"npass": 0},
            {"npass": 1.5},
            {"npass": np.nan},
            {"filt": "bessel"},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                critic_damp(**({"fcut": 10, "freq": 100} | changes))


if __name__ == "__main__":
    unittest.main()
