import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]


def load_notebook(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "notebooks_marimo" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SignalBasicPropertiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook("SignalBasicProperties")

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
        cls.notebook = load_notebook("DataFiltering")

    def test_critic_damp_butter_option_matches_scipy(self):
        b, a, fc = self.notebook.critic_damp(10, 100, npass=2, filt="butter")
        self.assertAlmostEqual(fc, 10 / (2**0.5 - 1) ** 0.25)
        b_sp, a_sp = signal.butter(2, fc / 50)
        np.testing.assert_allclose(b, b_sp)
        np.testing.assert_allclose(a, a_sp)

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
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook("ResidualAnalysis")
        data = np.loadtxt(ROOT / "data" / "Pezzack.txt", skiprows=6)
        cls.disp = data[:, 1]
        cls.freq = 1 / np.mean(np.diff(data[:, 0]))

    def test_optcutfreq_on_pezzack_data(self):
        fc_opt = self.notebook.optcutfreq(self.disp, freq=self.freq)
        self.assertAlmostEqual(fc_opt, 5.59, places=2)

    def test_optcutfreq_accepts_fclim_as_a_list(self):
        fc_opt = self.notebook.optcutfreq(self.disp, freq=self.freq, fclim=[8, 18])
        self.assertGreater(fc_opt, 4)
        self.assertLess(fc_opt, 8)

    def test_optcutfreq_returns_none_for_white_noise(self):
        y = np.random.default_rng(seed=42).standard_normal(100)
        self.assertIsNone(self.notebook.optcutfreq(y, freq=100))


if __name__ == "__main__":
    unittest.main()
