"""Regressions for the calibration and trial-selection teaching examples."""

import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import TestCase

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions.fpcalibra import fpcalibra
from functions.kistler_fp_cal import kistler_fp_cal
from functions.simila import mse, similarity


class MeasurementHelpersTests(TestCase):
    def test_mse_target_follows_sample_axis_for_each_trial(self):
        trials = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
        target = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            mse(trials, target=target, normalization=None), [0.0, 4.0]
        )
        np.testing.assert_allclose(
            mse(trials.T, target=target, axis1=1, axis2=0, normalization=None),
            [0.0, 4.0],
        )

    def test_kistler_accepts_array_calibration_and_origin(self):
        volts = np.array([[0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1]])
        force, cop = kistler_fp_cal(
            volts,
            S_matrix=np.ones((1, 8)),
            origin=np.array([0.12, 0.2, 0.0]),
            fp_type=0,
        )
        np.testing.assert_allclose(force, [[0.0, 0.0, 400.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(cop, [[0.0, 0.0]])

    def test_identical_trials_have_zero_score_and_are_kept(self):
        trials = np.tile(np.array([1.0, 2.0, 3.0])[:, None], (1, 5))
        with np.errstate(divide="raise", invalid="raise"):
            np.testing.assert_array_equal(mse(trials), np.zeros(5))
            selected, kept, removed, _ = similarity(trials, msg=False)
        np.testing.assert_array_equal(selected, trials)
        np.testing.assert_array_equal(kept, np.arange(5))
        self.assertEqual(removed.size, 0)

    def test_kistler_polynomials_use_measured_millimetre_coordinates(self):
        # Surface COP is (100, 50) mm before correction, with Fz = 1000 N.
        # §8.4.3, Kistler 9287C_002-462e-03.11, type-3 coefficients:
        # Dax = -5.1842329375 mm and Day = 6.095278625 mm.
        volts = np.array([[0.0, 0.0, 0.0, 0.0, 55 / 96, 5 / 96, 1 / 32, 11 / 32]])
        _, cop = kistler_fp_cal(
            volts,
            S_matrix=np.ones((1, 8)),
            origin=[0.12, 0.2, 0.0],
            fp_type=3,
        )
        np.testing.assert_allclose(cop, [[0.1051842329375, 0.043904721375]], atol=1e-12)

    def test_single_sensitivity_row_does_not_index_amplifier_ranges(self):
        volts = np.array([[0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1]])
        force, _ = kistler_fp_cal(
            volts,
            S_matrix=np.ones(8),
            fxfy_range=2,
            fz_range=3,
            origin=[0.12, 0.2, 0.0],
            fp_type=0,
        )
        np.testing.assert_allclose(force, [[0.0, 0.0, 400.0, 0.0, 0.0, 0.0]])

    def test_zero_median_score_does_not_hide_a_different_trial(self):
        trials = np.zeros((4, 5))
        trials[:, -1] = 2.0
        with np.errstate(divide="raise", invalid="raise"):
            scores = mse(trials)
            selected, kept, removed, _ = similarity(trials, msg=False)
        np.testing.assert_array_equal(scores[:4], np.zeros(4))
        self.assertTrue(np.isposinf(scores[-1]))
        np.testing.assert_array_equal(kept, [0, 1, 2, 3])
        np.testing.assert_array_equal(removed, [4])
        np.testing.assert_array_equal(selected, np.zeros((4, 4)))

    def test_calibration_continues_when_one_site_angle_is_already_stationary(self):
        # The centre site has vertical load only, so its angle is stationary.
        # Four off-centre sites still identify the full calibration matrix.
        cop = np.array(
            [
                [0.0, 0.112, 0.112, -0.112, -0.112],
                [0.0, 0.192, -0.192, 0.192, -0.192],
                [-0.124] * 5,
            ]
        )
        rng = np.random.default_rng(2)
        forces = rng.normal(0, 50, (3, 500))
        forces[2] += 300
        forces[:2, :100] = 0
        loads = np.concatenate(
            [
                np.vstack(
                    (
                        forces[:, k * 100 : (k + 1) * 100],
                        np.cross(cop[:, k], forces[:, k * 100 : (k + 1) * 100].T).T,
                    )
                )
                for k in range(5)
            ],
            axis=1,
        )
        angles = np.deg2rad([0, 20, -10, 15, -5])
        measured = forces.copy()
        for k, angle in enumerate(angles):
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            measured[:, k * 100 : (k + 1) * 100] = (
                rotation.T @ forces[:, k * 100 : (k + 1) * 100]
            )
        with redirect_stdout(StringIO()):
            calibration, estimated_angles = fpcalibra(
                loads,
                measured,
                cop,
                threshold=1e-7,
            )
        np.testing.assert_allclose(calibration, np.eye(6), atol=1e-4)
        np.testing.assert_allclose(estimated_angles, angles, atol=1e-4)
