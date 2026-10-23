import importlib.util
import unittest
from pathlib import Path

import numpy as np
from optcutfreq import optcutfreq
from psd2 import psd2
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]


def load_notebook(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "notebooks" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SignalBasicPropertiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook("signal_basic_properties")

    def test_quantize_uses_2_to_the_n_levels_and_half_step_error(self):
        x = np.linspace(-1, 1, 10001)
        for n_bits in (1, 4, 8):
            with self.subTest(n_bits=n_bits):
                xq = self.notebook.quantize(x, n_bits, 2)
                self.assertEqual(np.unique(xq).size, 2**n_bits)
                self.assertLessEqual(np.max(np.abs(xq - x)), 2 / 2**n_bits / 2 + 1e-12)

    def test_quantize_clips_values_outside_the_range(self):
        xq = self.notebook.quantize(np.array([-10.0, 10.0]), 2, 2)
        np.testing.assert_allclose(xq, [-0.75, 0.75])


class DataFilteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook("data_filtering")

    def test_critic_damp_butter_option_matches_scipy(self):
        b, a, fc = self.notebook.critic_damp(10, 100, npass=2, filt="butter")
        correction = (2**0.5 - 1) ** 0.25
        expected = 100 / np.pi * np.arctan(np.tan(np.pi * 10 / 100) / correction)
        self.assertAlmostEqual(fc, expected)
        b_sp, a_sp = signal.butter(2, fc / 50)
        np.testing.assert_allclose(b, b_sp)
        np.testing.assert_allclose(a, a_sp)

    def test_critic_damp_combined_response_is_half_power_at_requested_cutoff(self):
        for kind in ("butter", "critic"):
            for cutoff in (2, 10, 35):
                for passes in (1, 2, 4):
                    with self.subTest(kind=kind, cutoff=cutoff, passes=passes):
                        b, a, _ = self.notebook.critic_damp(
                            cutoff, 100, npass=passes, filt=kind
                        )
                        _, response = signal.freqz(b, a, worN=[cutoff], fs=100)
                        self.assertAlmostEqual(abs(response[0]) ** (2 * passes), 0.5)

    def test_critic_damp_rejects_invalid_cutoff(self):
        for cutoff in (0, -1, 50, 60):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                self.notebook.critic_damp(cutoff, 100)

    def test_critic_damp_has_unit_dc_gain_and_no_step_overshoot(self):
        b, a, _ = self.notebook.critic_damp(10, 100, npass=2, filt="critic")
        self.assertAlmostEqual(np.sum(b) - np.sum(a[1:]), 1)
        y = signal.filtfilt(b, a, np.r_[np.zeros(20), np.ones(20)])
        self.assertLessEqual(y.max(), 1 + 1e-12)
        self.assertGreaterEqual(y.min(), -1e-12)

    def test_critic_damp_rejects_unknown_filter(self):
        with self.assertRaises(ValueError):
            self.notebook.critic_damp(10, 100, filt="bessel")

    def test_moving_averages_agree(self):
        x = np.random.default_rng(0).standard_normal(200)
        m = 7
        y = self.notebook.moving_average(x, m)
        np.testing.assert_allclose(self.notebook.moving_average_cumsum(x, m), y)
        np.testing.assert_allclose(
            self.notebook.moving_average_lfilter(x, m)[m - 1 :], y
        )
        np.testing.assert_allclose(self.notebook.moving_average_convolve(x, m)[3:-3], y)

    def test_odd_extension_continues_a_line(self):
        x = np.arange(10.0)
        np.testing.assert_allclose(
            self.notebook.odd_extension(x, 3), np.arange(-3.0, 13.0)
        )


class ResidualAnalysisTests(unittest.TestCase):
    """The optcutfreq package, as the ResidualAnalysis notebook uses it."""

    @classmethod
    def setUpClass(cls):
        data = np.loadtxt(ROOT / "data" / "Pezzack.txt", skiprows=6)
        cls.disp = data[:, 1]
        cls.freq = 1 / np.mean(np.diff(data[:, 0]))

    def test_optcutfreq_on_pezzack_data(self):
        fc_opt = optcutfreq(self.disp, freq=self.freq)
        self.assertAlmostEqual(fc_opt, 5.59, places=2)

    def test_optcutfreq_accepts_fclim_as_a_list(self):
        fc_opt = optcutfreq(self.disp, freq=self.freq, fclim=[8, 18])
        self.assertGreater(fc_opt, 4)
        self.assertLess(fc_opt, 8)

    def test_optcutfreq_returns_none_for_white_noise(self):
        y = np.random.default_rng(seed=42).standard_normal(100)
        self.assertIsNone(optcutfreq(y, freq=100))


class FourierTransformTests(unittest.TestCase):
    """The psd2 package, as the FourierTransform notebook uses it."""

    def test_psd_of_a_sine_finds_its_frequency_and_power(self):
        fs = 1000
        t = np.arange(10 * fs) / fs
        x = 2 * np.sin(2 * np.pi * 50 * t)  # power = 2**2 / 2
        fpcntile, mpf, fmax, ptotal, f, _ = psd2(x, fs=fs, show=False)
        self.assertAlmostEqual(fmax, 50, delta=f[1])
        # the median is the first bin reaching 50% of the power: up to one bin above
        self.assertAlmostEqual(fpcntile[50], 50, delta=1.5 * f[1])
        self.assertAlmostEqual(mpf, 50, delta=1)
        self.assertAlmostEqual(ptotal, 2, delta=0.02)
        self.assertEqual(len(fpcntile), 101)


if __name__ == "__main__":
    unittest.main()
