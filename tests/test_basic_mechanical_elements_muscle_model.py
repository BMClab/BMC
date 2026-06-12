import importlib.util
from pathlib import Path
import re
import unittest

import numpy as np


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks_marimo"
    / "BasicMechanicalElementsMuscleModel.py"
)


def load_notebook_module():
    spec = importlib.util.spec_from_file_location(
        "basic_mechanical_elements_muscle_model", NOTEBOOK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BasicMechanicalElementsMuscleModelTests(unittest.TestCase):
    def test_notebook_has_requested_scope_and_tutorial_style(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertIn("# Basic Mechanical Elements of a Muscle Model", source)
        self.assertIn("Marcos Duarte, Renato Watanabe", source)
        self.assertIn("## How to use this tutorial", source)
        self.assertIn("Maxwell", source)
        self.assertIn("Voight", source)
        self.assertIn("Kelvin solid", source)
        self.assertIn("spring and dashpot", source)
        self.assertIn("same force passes through both elements", source)
        self.assertIn("same deformation", source)
        self.assertGreaterEqual(source.count("**Challenge"), 5)
        self.assertGreaterEqual(source.count("**Guiding questions"), 3)
        self.assertIn("Read the chapter first", source)

    def test_notebook_uses_marimo_interactive_controls(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertGreaterEqual(source.count("mo.ui.slider"), 10)
        self.assertIn("element_stiffness_slider.value", source)
        self.assertIn("step_force_slider.value", source)
        self.assertIn("kelvin_series_stiffness_slider.value", source)
        self.assertIn("sinusoid_frequency_slider.value", source)
        self.assertIn("Spring stiffness k [N/m]", source)
        self.assertIn("Constant force F0 [N]", source)
        self.assertIn("Input frequency [Hz]", source)
        self.assertGreaterEqual(source.count("mo.output.append"), 5)

    def test_guiding_question_numbers_are_unique(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        numbers = re.findall(r"\*\*Guiding questions? (\d+)", source)

        self.assertEqual(numbers, list(dict.fromkeys(numbers)))

    def test_notebook_reactive_graph_validates(self):
        module = load_notebook_module()

        module.app._maybe_initialize()

    def test_maxwell_step_responses_match_closed_form(self):
        module = load_notebook_module()
        time = np.array([0.0, 1.0, 2.0])

        extension = module.maxwell_constant_load_response(
            time,
            force=10.0,
            stiffness=100.0,
            damping=40.0,
        )
        force = module.maxwell_constant_deformation_response(
            time,
            deformation=0.08,
            stiffness=100.0,
            damping=40.0,
        )

        np.testing.assert_allclose(extension, [0.1, 0.35, 0.6])
        self.assertAlmostEqual(force[0], 8.0)
        self.assertLess(force[-1], force[0])

    def test_voight_step_responses_match_expected_limits(self):
        module = load_notebook_module()
        time = np.array([0.0, 1.0, 10.0])

        extension = module.voight_constant_load_response(
            time,
            force=10.0,
            stiffness=100.0,
            damping=40.0,
        )
        force = module.voight_constant_deformation_response(
            time,
            deformation=0.08,
            stiffness=100.0,
        )

        self.assertAlmostEqual(extension[0], 0.0)
        self.assertLess(extension[1], 0.1)
        self.assertAlmostEqual(extension[-1], 0.1, places=10)
        np.testing.assert_allclose(force, [8.0, 8.0, 8.0])

    def test_kelvin_step_responses_have_instant_and_equilibrium_values(self):
        module = load_notebook_module()
        time = np.array([0.0, 10.0])

        extension = module.kelvin_constant_load_response(
            time,
            force=10.0,
            series_stiffness=120.0,
            parallel_stiffness=80.0,
            damping=50.0,
        )
        force = module.kelvin_constant_deformation_response(
            time,
            deformation=0.08,
            series_stiffness=120.0,
            parallel_stiffness=80.0,
            damping=50.0,
        )

        self.assertAlmostEqual(extension[0], 10.0 / 120.0)
        self.assertAlmostEqual(extension[-1], 10.0 / 120.0 + 10.0 / 80.0, places=6)
        self.assertAlmostEqual(force[0], 120.0 * 0.08)
        self.assertAlmostEqual(force[-1], (120.0 * 80.0 / 200.0) * 0.08, places=6)

    def test_length_input_simulation_returns_all_model_forces(self):
        module = load_notebook_module()
        time = np.linspace(0, 1, 101)
        deformation = 0.05 + 0.01 * np.sin(2 * np.pi * time)

        responses = module.simulate_length_input_responses(
            time,
            deformation,
            stiffness=100.0,
            damping=40.0,
            series_stiffness=120.0,
            parallel_stiffness=80.0,
        )

        self.assertEqual(set(responses), {"velocity", "maxwell", "voight", "kelvin"})
        for response in responses.values():
            self.assertEqual(response.shape, time.shape)
            self.assertTrue(np.all(np.isfinite(response)))


if __name__ == "__main__":
    unittest.main()
