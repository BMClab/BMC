import importlib.util
from pathlib import Path
import re
import unittest

import numpy as np


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks_marimo"
    / "OrdinaryDifferentialEquation.py"
)


def load_notebook_module():
    spec = importlib.util.spec_from_file_location(
        "ordinary_differential_equation", NOTEBOOK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OrdinaryDifferentialEquationTests(unittest.TestCase):
    def test_ball_euler_preserves_nonzero_initial_time(self):
        module = load_notebook_module()

        t, y, v = module.ball_euler(t0=2.0, tend=2.3, y0=1.0, v0=3.0, h=0.1)

        np.testing.assert_allclose(t[:4], [2.0, 2.1, 2.2, 2.3])
        self.assertEqual(y.shape, t.shape)
        self.assertEqual(v.shape, t.shape)

    def test_notebook_uses_current_scipy_ivp_api(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertIn("from scipy.integrate import solve_ivp", source)
        self.assertIn("scipy.integrate.solve_ivp", source)
        self.assertNotIn("odeint", source)
        self.assertNotIn("from scipy.integrate import ode", source)
        self.assertNotIn("set_integrator", source)

    def test_notebook_text_and_references_are_updated(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertIn("Current IFAB Law 2", source)
        self.assertNotIn("FIFA (2015)", source)
        self.assertIn("SciPy recommends `solve_ivp` for new code", source)
        self.assertIn("$$", source)
        self.assertNotIn("Where$", source)

    def test_ivp_and_euler_variant_sections_are_present(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertIn("## Initial Value Problems (IVP)", source)
        self.assertIn("A solution to an IVP", source)
        self.assertIn("## Explicit and Semi-Implicit Euler Methods", source)
        self.assertIn("semi-implicit Euler", source)
        self.assertIn("fully implicit backward Euler", source)
        self.assertIn("### Example: Simple Pendulum under Gravity", source)
        self.assertLess(
            source.index("## Initial Value Problems (IVP)"),
            source.index("### Euler method"),
        )
        self.assertLess(
            source.index("## Explicit and Semi-Implicit Euler Methods"),
            source.index("## Examples"),
        )

    def test_notebook_has_conversational_tutorial_challenges_and_links(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "## An unassuming derivation of a numerical solution to an Ordinary Differential Equation (ODE)",
            source,
        )
        self.assertLess(
            source.index(
                "## An unassuming derivation of a numerical solution to an Ordinary Differential Equation (ODE)"
            ),
            source.index("## Ordinary Differential Equation"),
        )
        self.assertIn("Imagine you are in a car", source)
        self.assertIn("| \\(i\\) | Time \\(t_i\\) [s] |", source)
        self.assertIn("| 0 | 0 | -- | 100 |", source)
        self.assertIn("| 1 | 10 | 20 | 300 |", source)
        self.assertIn("| 2 | 20 | 25 | 550 |", source)
        self.assertIn("That is what makes this a numerical solution.", source)
        self.assertIn("initial value problem, or IVP", source)
        self.assertIn("## How to use this tutorial", source)
        self.assertIn("You will", source)
        self.assertGreaterEqual(source.count("**Challenge"), 6)
        self.assertGreaterEqual(source.count("**Guiding questions"), 4)
        self.assertIn("Before you run the next cell", source)
        self.assertIn("## Practice problems", source)
        self.assertIn("Design a mass-spring-damper system", source)
        self.assertIn("## Go deeper", source)
        self.assertIn(
            "https://openstax.org/books/university-physics-volume-1/pages/15-5-damped-oscillations",
            source,
        )
        self.assertIn(
            "https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-ii-second-order-constant-coefficient-linear-equations/damped-harmonic-oscillators/",
            source,
        )

    def test_notebook_proposes_course_aligned_questions_and_problems(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertIn("## Checkpoint questions", source)
        self.assertIn("terminal velocity", source)
        self.assertIn("one-mass spring problem", source)
        self.assertIn("two-mass spring-damper system", source)
        self.assertIn("simple muscle-tendon model", source)
        self.assertIn("force-platform record", source)
        self.assertIn("**IVP audit.**", source)
        self.assertIn("**Projectile with terminal velocity.**", source)
        self.assertIn("**Your own movement model.**", source)

    def test_notebook_reactive_graph_validates(self):
        module = load_notebook_module()

        module.app._maybe_initialize()

    def test_guiding_question_numbers_are_unique(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        numbers = re.findall(r"\*\*Guiding questions? (\d+)", source)

        self.assertEqual(numbers, list(dict.fromkeys(numbers)))

    def test_euler_method_supports_explicit_and_semi_implicit_updates(self):
        module = load_notebook_module()

        def acceleration(_t, y):
            return -y[0]

        t_forward, y_forward = module.euler_method(
            acceleration, T=0.1, y0=(1.0, 0.0), h=0.05, method=1
        )
        t_semi, y_semi = module.euler_method(
            acceleration, T=0.1, y0=(1.0, 0.0), h=0.05, method=2
        )

        self.assertEqual(t_forward.shape, t_semi.shape)
        self.assertEqual(y_forward.shape, y_semi.shape)
        self.assertEqual(y_forward.shape[0], 2)
        self.assertAlmostEqual(y_forward[0, 1], 1.0)
        self.assertLess(y_semi[0, 1], y_forward[0, 1])

        with self.assertRaises(ValueError):
            module.euler_method(acceleration, T=0.1, y0=(1.0, 0.0), h=0.05, method=99)

    def test_mass_spring_damper_helper_returns_expected_acceleration(self):
        module = load_notebook_module()

        acceleration = module.mass_spring_damper_acceleration(
            0.0,
            [0.1, -0.2],
            mass=2.0,
            stiffness=8.0,
            damping=0.5,
            external_force=1.0,
        )

        self.assertAlmostEqual(acceleration, (1.0 - 0.5 * -0.2 - 8.0 * 0.1) / 2.0)

    def test_helper_function_cells_follow_their_narrative_sections(self):
        source = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertLess(
            source.index(
                "# Introduction to numerical solution of Ordinary Differential Equation"
            ),
            source.index("@app.function\ndef ball_euler"),
        )
        self.assertLess(
            source.index("This function calculates the vertical trajectory"),
            source.index("@app.function\ndef ball_euler"),
        )
        self.assertLess(
            source.index("@app.function\ndef ball_euler"),
            source.index("Now integrate the same motion with two step sizes"),
        )
        self.assertLess(
            source.index("`solve_ivp` can call the LSODA method"),
            source.index("@app.function\ndef ball_constant_force"),
        )
        self.assertLess(
            source.index("@app.function\ndef solve_trajectory"),
            source.index('method="LSODA"'),
        )
        self.assertLess(
            source.index("Now include air resistance"),
            source.index("@app.function\ndef ball_with_drag"),
        )
        self.assertLess(
            source.index("## Explicit and Semi-Implicit Euler Methods"),
            source.index("@app.function\ndef euler_method"),
        )
        self.assertLess(
            source.index("### Example: Simple Pendulum under Gravity"),
            source.index("@app.function\ndef pendulum_acceleration"),
        )
        self.assertLess(
            source.index("### Example: Mass-Spring-Damper System"),
            source.index("@app.function\ndef mass_spring_damper_acceleration"),
        )
        self.assertLess(
            source.index("@app.function\ndef mass_spring_damper_acceleration"),
            source.index("mass_spring_time, mass_spring_state ="),
        )


if __name__ == "__main__":
    unittest.main()
