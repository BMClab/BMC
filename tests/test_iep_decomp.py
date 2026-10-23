import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('iep_decomp', Path(__file__).resolve().parents[1] / 'functions' / 'iep_decomp.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IepTests(unittest.TestCase):
    def test_crossings_are_interpolated_and_components_reconstruct_cop(self):
        result = module.iep_decomp(np.arange(6.), [-1, 1, -1, 1, -1, 1], np.arange(6.) * 2, demean_force=False)
        np.testing.assert_allclose(result['knot_time'], [.5, 1.5, 2.5, 3.5, 4.5])
        np.testing.assert_allclose(result['knot_cop'], [1, 3, 5, 7, 9])
        np.testing.assert_allclose(result['rambling'] + result['trembling'], result['cop'])
        self.assertEqual(result['time'].tolist(), [1, 2, 3, 4])

    def test_constant_force_and_insufficient_crossings_are_explicit_errors(self):
        for force in ([0, 0, 0, 0, 0, 0], [-1, -1, -1, 1, 1, 1]):
            with self.subTest(force=force), self.assertRaises(ValueError):
                module.iep_decomp(np.arange(6.), force, np.arange(6.))
