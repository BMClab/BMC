import importlib.util
import unittest
import warnings
from pathlib import Path

import numpy as np

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks_marimo" / "PathFrame.py"
)


class PathFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("path_frame", NOTEBOOK_PATH)
        cls.notebook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.notebook)

    def test_straight_paths_have_undefined_normals_and_zero_curvature(self):
        dt = 0.01
        t = np.arange(0, 10 + dt / 2, dt)
        for velocity in ([10.0, 5.0], [10.0, 5.0, -2.0]):
            with self.subTest(velocity=velocity):
                r = t[:, None] * velocity
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    speed, e_t, e_n, kappa, a_t, a_n = self.notebook.path_frame(r, dt)

                self.assertTrue(np.isnan(e_n).all())
                np.testing.assert_array_equal(kappa, 0)
                np.testing.assert_array_equal(a_n, 0)
                np.testing.assert_allclose(speed, np.linalg.norm(velocity))
                np.testing.assert_allclose(
                    e_t, np.broadcast_to(velocity / np.linalg.norm(velocity), r.shape)
                )
                np.testing.assert_allclose(a_t, 0, atol=1e-9)
                self.assertEqual(caught, [])

    def test_bend_retains_inward_normals_and_centripetal_acceleration(self):
        radius, speed, dt = 36.8, 10.0, 0.01
        t = np.arange(0, np.pi * radius / speed, dt)
        phi = speed * t / radius
        r = radius * np.column_stack((np.cos(phi), np.sin(phi)))

        _, e_t, e_n, kappa, _, a_n = self.notebook.path_frame(r, dt)

        np.testing.assert_allclose(e_n[2:-2], -r[2:-2] / radius, atol=1e-8)
        np.testing.assert_allclose(np.sum(e_t[2:-2] * e_n[2:-2], axis=1), 0, atol=1e-8)
        np.testing.assert_allclose(kappa, 1 / radius, rtol=2e-5)
        np.testing.assert_allclose(a_n, speed**2 / radius, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
