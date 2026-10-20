import importlib.util
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import sympy as sym

MODULE_PATH = Path(__file__).resolve().parents[1] / "functions" / "euler_rotmat.py"


def load_euler_rotmat():
    spec = importlib.util.spec_from_file_location("euler_rotmat", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.euler_rotmat


def local_rotation(axis, angle):
    """Elemental rotation matrix of the local frame, as defined in euler_rotmat."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    if axis == "y":
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


class EulerRotmatTests(unittest.TestCase):
    def setUp(self):
        self.euler_rotmat = load_euler_rotmat()

    def test_algebraic_matrix_without_angles(self):
        R = self.euler_rotmat(showA=False, showN=False)

        self.assertIsInstance(R, sym.MatrixBase)
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual({str(s) for s in R.free_symbols}, {"alpha", "beta", "gamma"})

    def test_numeric_matrix_for_one_rotation(self):
        _, Rn = self.euler_rotmat(order="x", angles=[90], showA=False, showN=False)

        np.testing.assert_allclose(
            np.asarray(Rn, dtype=float),
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            atol=1e-12,
        )

    def test_numeric_matrix_for_xyz_sequence(self):
        angles = [30, 45, 60]
        _, Rn = self.euler_rotmat(order="xyz", angles=angles, showA=False, showN=False)

        a, b, g = np.radians(angles)
        expected = (
            local_rotation("z", g) @ local_rotation("y", b) @ local_rotation("x", a)
        )
        np.testing.assert_allclose(np.asarray(Rn, dtype=float), expected, atol=1e-12)

    def test_partial_angles_keep_the_unknown_symbols(self):
        _, Rn = self.euler_rotmat(
            order="zxz", angles=[None, 0, None], showA=False, showN=False
        )

        self.assertIsInstance(Rn, sym.MatrixBase)
        self.assertTrue({str(s) for s in Rn.free_symbols} <= {"alpha", "gamma"})
        self.assertTrue(Rn.free_symbols)

    def test_matrices_are_displayed_when_requested(self):
        with mock.patch("IPython.display.display") as display:
            self.euler_rotmat(angles=[90, 90, 90])

        self.assertEqual(display.call_count, 2)

    def test_source_has_no_invalid_escape_sequences(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            compile(source, str(MODULE_PATH), "exec")


if __name__ == "__main__":
    unittest.main()
