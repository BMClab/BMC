import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic Mechanical Elements of a Muscle Model

    > Marcos Duarte, Renato Watanabe,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this tutorial

    This notebook is a companion to the chapter section on basic mechanical elements of a muscle model. Read the chapter first. Here we will not repeat every derivation; instead, we will turn the main passive elements into small computational models and use simulations to check whether each arrangement behaves like the soft-tissue responses described in the book.

    Move through the notebook in order. Before each plot, predict the shape of the response on paper: initial jump or no initial jump, relaxation or creep, finite asymptote or continuing deformation. Then run the cell and compare the result with your prediction.

    **Challenge 0.** Before you begin, choose one passive tissue or muscle-tendon structure. If it is suddenly loaded, do you expect it to stretch all at once, slowly over time, or both?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From passive elements to a muscle model

    The chapter builds muscle models from idealized elements. For this notebook, keep the mental picture simple:

    - a spring stores elastic energy and relates force to deformation;
    - a dashpot dissipates energy and relates force to deformation velocity;
    - a muscle model adds an active tensile-force generator to passive elastic and viscous elements.

    We will stay with the passive part. The goal is to understand what changes when a spring and a dashpot are arranged in series, in parallel, or in a mixed arrangement. This is the mechanical vocabulary used later by Hill-type muscle models.

    **Guiding questions 1.**
    1. Which element can change force instantly when its length changes?
    2. Which element makes the response depend on time?
    3. Which model would you expect to be useful if a tissue shows both an immediate deformation and a slow creep?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic passive elements

    A linear spring is modeled as

    $$
    F_k = kx,
    $$

    where \(x\) is deformation from the relaxed length and \(k\) is stiffness.

    A linear dashpot is modeled as

    $$
    F_b = b\dot{x},
    $$

    where \(\dot{x}\) is deformation velocity and \(b\) is the viscous damping coefficient.

    In this notebook, positive force and positive deformation mean tension and elongation. Real biological tissues are often tension-only over the range of interest, but the linear equations are useful because they isolate the effect of arrangement before we add nonlinearities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Series and parallel rules

    The first modeling decision is not the value of \(k\) or \(b\). It is the arrangement.

    For two elements in **series**:

    $$
    F = F_1 = F_2,\qquad x = x_1 + x_2.
    $$

    The force is common and the deformations add.

    For two elements in **parallel**:

    $$
    x = x_1 = x_2,\qquad F = F_1 + F_2.
    $$

    The deformation is common and the forces add.

    This small change is enough to produce very different creep and relaxation behavior.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The three arrangements

    | Model | Arrangement | Useful mental model |
    |---|---|---|
    | Maxwell | spring and dashpot in series | same force passes through both elements |
    | Voight | spring and dashpot in parallel | both elements experience the same deformation |
    | Kelvin solid | spring in series with a Voight element | immediate elastic response plus delayed viscoelastic response |

    The chapter uses these arrangements to ask two practical questions:

    1. What happens when a constant force is applied?
    2. What happens when a constant deformation is imposed?

    We will answer both with plots.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The simulations below use only NumPy and Matplotlib. The equations are simple enough that closed-form solutions are available for the step responses, but later we will also integrate the model equations numerically for a time-varying deformation input.
    """)
    return


@app.cell
def _():
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["lines.linewidth"] = 3
    matplotlib.rcParams["font.size"] = 13
    matplotlib.rcParams["lines.markersize"] = 5
    matplotlib.rc("axes", grid=False, labelsize=14, titlesize=16, ymargin=0.05)
    matplotlib.rc("legend", numpoints=1, fontsize=11)
    return np, plt


@app.function
def ensure_positive(**values):
    """Raise ValueError if a model parameter is not strictly positive."""
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


@app.function
def linear_spring_force(extension, stiffness):
    """Return the force produced by a linear spring."""
    import numpy as np

    ensure_positive(stiffness=stiffness)
    return stiffness * np.asarray(extension, dtype=float)


@app.function
def dashpot_force(velocity, damping):
    """Return the force produced by a linear dashpot."""
    import numpy as np

    ensure_positive(damping=damping)
    return damping * np.asarray(velocity, dtype=float)


@app.function
def maxwell_constant_load_response(time, force, stiffness, damping):
    """Return Maxwell-model deformation under a suddenly applied constant force."""
    import numpy as np

    ensure_positive(stiffness=stiffness, damping=damping)
    time = np.asarray(time, dtype=float)
    return force / stiffness + force * time / damping


@app.function
def maxwell_constant_deformation_response(time, deformation, stiffness, damping):
    """Return Maxwell-model force under a suddenly imposed constant deformation."""
    import numpy as np

    ensure_positive(stiffness=stiffness, damping=damping)
    time = np.asarray(time, dtype=float)
    tau = damping / stiffness
    return stiffness * deformation * np.exp(-time / tau)


@app.function
def voight_constant_load_response(time, force, stiffness, damping):
    """Return Voight-model deformation under a suddenly applied constant force."""
    import numpy as np

    ensure_positive(stiffness=stiffness, damping=damping)
    time = np.asarray(time, dtype=float)
    tau = damping / stiffness
    return force / stiffness * (1 - np.exp(-time / tau))


@app.function
def voight_constant_deformation_response(time, deformation, stiffness):
    """Return post-step Voight-model force under a fixed deformation."""
    import numpy as np

    ensure_positive(stiffness=stiffness)
    time = np.asarray(time, dtype=float)
    return np.full_like(time, stiffness * deformation, dtype=float)


@app.function
def kelvin_constant_load_response(
    time,
    force,
    series_stiffness,
    parallel_stiffness,
    damping,
):
    """Return Kelvin-solid deformation under a suddenly applied constant force."""
    import numpy as np

    ensure_positive(
        series_stiffness=series_stiffness,
        parallel_stiffness=parallel_stiffness,
        damping=damping,
    )
    time = np.asarray(time, dtype=float)
    tau = damping / parallel_stiffness
    series_deformation = force / series_stiffness
    voight_deformation = force / parallel_stiffness * (1 - np.exp(-time / tau))
    return series_deformation + voight_deformation


@app.function
def kelvin_constant_deformation_response(
    time,
    deformation,
    series_stiffness,
    parallel_stiffness,
    damping,
):
    """Return Kelvin-solid force under a suddenly imposed constant deformation."""
    import numpy as np

    ensure_positive(
        series_stiffness=series_stiffness,
        parallel_stiffness=parallel_stiffness,
        damping=damping,
    )
    time = np.asarray(time, dtype=float)
    tau = damping / (series_stiffness + parallel_stiffness)
    initial_force = series_stiffness * deformation
    equilibrium_force = (
        series_stiffness
        * parallel_stiffness
        / (series_stiffness + parallel_stiffness)
        * deformation
    )
    return equilibrium_force + (initial_force - equilibrium_force) * np.exp(-time / tau)


@app.function
def simulate_length_input_responses(
    time,
    deformation,
    stiffness,
    damping,
    series_stiffness,
    parallel_stiffness,
):
    """Simulate Maxwell, Voight, and Kelvin forces for prescribed deformation."""
    import numpy as np

    ensure_positive(
        stiffness=stiffness,
        damping=damping,
        series_stiffness=series_stiffness,
        parallel_stiffness=parallel_stiffness,
    )
    time = np.asarray(time, dtype=float)
    deformation = np.asarray(deformation, dtype=float)
    if time.ndim != 1 or deformation.ndim != 1 or time.shape != deformation.shape:
        raise ValueError("time and deformation must be one-dimensional arrays.")
    if time.size < 2:
        raise ValueError("time must contain at least two samples.")

    velocity = np.gradient(deformation, time, edge_order=2)
    maxwell_force = np.zeros_like(time)
    kelvin_force = np.zeros_like(time)
    maxwell_force[0] = stiffness * deformation[0]
    kelvin_force[0] = series_stiffness * deformation[0]

    for index in range(time.size - 1):
        dt = time[index + 1] - time[index]
        if dt <= 0:
            raise ValueError("time must be strictly increasing.")

        maxwell_force[index + 1] = maxwell_force[index] + dt * (
            stiffness * velocity[index] - stiffness * maxwell_force[index] / damping
        )
        kelvin_force[index + 1] = kelvin_force[index] + dt * (
            series_stiffness * velocity[index]
            + series_stiffness * parallel_stiffness * deformation[index] / damping
            - (series_stiffness + parallel_stiffness) * kelvin_force[index] / damping
        )

    voight_force = stiffness * deformation + damping * velocity
    return {
        "velocity": velocity,
        "maxwell": maxwell_force,
        "voight": voight_force,
        "kelvin": kelvin_force,
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First look: one spring and one dashpot

    Before combining elements, check the two individual relationships. The spring-force plot is a force-deformation curve. The dashpot-force plot is a force-velocity curve.

    Use the controls below to change the spring stiffness, dashpot damping, and the plotted deformation and velocity ranges.

    **Before you run the next cell**, predict which plot crosses the origin with slope \(k\), and which one crosses the origin with slope \(b\).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    element_stiffness_slider = mo.ui.slider(
        20.0,
        300.0,
        10.0,
        value=120.0,
        label="Spring stiffness k [N/m]",
        show_value=True,
        include_input=True,
    )
    element_damping_slider = mo.ui.slider(
        5.0,
        100.0,
        5.0,
        value=35.0,
        label="Dashpot damping b [N s/m]",
        show_value=True,
        include_input=True,
    )
    element_extension_limit_slider = mo.ui.slider(
        0.05,
        0.40,
        0.01,
        value=0.20,
        label="Maximum plotted deformation [m]",
        show_value=True,
        include_input=True,
    )
    element_velocity_limit_slider = mo.ui.slider(
        0.05,
        0.60,
        0.01,
        value=0.25,
        label="Maximum plotted speed [m/s]",
        show_value=True,
        include_input=True,
    )
    return (
        element_damping_slider,
        element_extension_limit_slider,
        element_stiffness_slider,
        element_velocity_limit_slider,
    )


@app.cell
def _(
    element_damping_slider,
    element_extension_limit_slider,
    element_stiffness_slider,
    element_velocity_limit_slider,
    np,
):
    element_demo_extension = np.linspace(0, element_extension_limit_slider.value, 201)
    element_demo_velocity = np.linspace(
        -element_velocity_limit_slider.value,
        element_velocity_limit_slider.value,
        201,
    )
    spring_demo_force = linear_spring_force(
        element_demo_extension,
        stiffness=element_stiffness_slider.value,
    )
    dashpot_demo_force = dashpot_force(
        element_demo_velocity,
        damping=element_damping_slider.value,
    )
    return (
        dashpot_demo_force,
        element_demo_extension,
        element_demo_velocity,
        spring_demo_force,
    )


@app.cell
def _(
    dashpot_demo_force,
    element_damping_slider,
    element_demo_extension,
    element_demo_velocity,
    element_extension_limit_slider,
    element_stiffness_slider,
    element_velocity_limit_slider,
    mo,
    plt,
    spring_demo_force,
):
    mo.output.append(
        mo.vstack(
            [
                mo.hstack(
                    [element_stiffness_slider, element_damping_slider],
                    widths="equal",
                ),
                mo.hstack(
                    [element_extension_limit_slider, element_velocity_limit_slider],
                    widths="equal",
                ),
            ],
            align="stretch",
        )
    )
    _, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(element_demo_extension, spring_demo_force, color="tab:blue")
    _axes[0].set_title("Linear spring")
    _axes[0].set_xlabel("Deformation [m]")
    _axes[0].set_ylabel("Force [N]")

    _axes[1].plot(element_demo_velocity, dashpot_demo_force, color="tab:red")
    _axes[1].axhline(0, color="0.6", linewidth=1)
    _axes[1].axvline(0, color="0.6", linewidth=1)
    _axes[1].set_title("Linear dashpot")
    _axes[1].set_xlabel("Velocity [m/s]")
    _axes[1].set_ylabel("Force [N]")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Maxwell model

    In the Maxwell model, the spring and dashpot are in series:

    $$
    F = F_k = F_b,\qquad x = x_k + x_b.
    $$

    Combining the spring and dashpot equations gives

    $$
    \dot{F} + \frac{k}{b}F = k\dot{x}.
    $$

    Two common simulations are:

    $$
    x(t) = \frac{F_0}{k} + \frac{F_0}{b}t
    $$

    for a constant load \(F_0\), and

    $$
    F(t) = kx_0 e^{-t/\tau},\qquad \tau=\frac{b}{k}
    $$

    for a constant deformation \(x_0\).

    **Guiding questions 2.**
    1. Why does the constant-load response have an immediate deformation?
    2. Why does the same response continue to deform without reaching a finite limit?
    3. What does the force approach during a fixed-deformation relaxation test?
    """)
    return


@app.cell
def _(mo):
    step_force_slider = mo.ui.slider(
        1.0,
        30.0,
        1.0,
        value=10.0,
        label="Constant force F0 [N]",
        show_value=True,
        include_input=True,
    )
    step_deformation_slider = mo.ui.slider(
        0.01,
        0.20,
        0.01,
        value=0.08,
        label="Constant deformation x0 [m]",
        show_value=True,
        include_input=True,
    )
    step_stiffness_slider = mo.ui.slider(
        20.0,
        300.0,
        10.0,
        value=100.0,
        label="Spring stiffness k [N/m]",
        show_value=True,
        include_input=True,
    )
    step_damping_slider = mo.ui.slider(
        5.0,
        150.0,
        5.0,
        value=40.0,
        label="Dashpot damping b [N s/m]",
        show_value=True,
        include_input=True,
    )
    step_duration_slider = mo.ui.slider(
        2.0,
        12.0,
        0.5,
        value=8.0,
        label="Simulation duration [s]",
        show_value=True,
        include_input=True,
    )
    return (
        step_damping_slider,
        step_deformation_slider,
        step_duration_slider,
        step_force_slider,
        step_stiffness_slider,
    )


@app.cell
def _(
    np,
    step_damping_slider,
    step_deformation_slider,
    step_duration_slider,
    step_force_slider,
    step_stiffness_slider,
):
    step_time = np.linspace(
        0,
        step_duration_slider.value,
        int(step_duration_slider.value * 100) + 1,
    )
    step_force = float(step_force_slider.value)
    step_deformation = float(step_deformation_slider.value)
    step_stiffness = float(step_stiffness_slider.value)
    step_damping = float(step_damping_slider.value)
    return (
        step_damping,
        step_deformation,
        step_force,
        step_stiffness,
        step_time,
    )


@app.cell
def _(step_damping, step_deformation, step_force, step_stiffness, step_time):
    maxwell_load_extension = maxwell_constant_load_response(
        step_time,
        force=step_force,
        stiffness=step_stiffness,
        damping=step_damping,
    )
    maxwell_relaxation_force = maxwell_constant_deformation_response(
        step_time,
        deformation=step_deformation,
        stiffness=step_stiffness,
        damping=step_damping,
    )
    return maxwell_load_extension, maxwell_relaxation_force


@app.cell
def _(
    maxwell_load_extension,
    maxwell_relaxation_force,
    mo,
    plt,
    step_damping_slider,
    step_deformation_slider,
    step_duration_slider,
    step_force_slider,
    step_stiffness_slider,
    step_time,
):
    mo.output.append(
        mo.vstack(
            [
                mo.hstack([step_force_slider, step_deformation_slider], widths="equal"),
                mo.hstack([step_stiffness_slider, step_damping_slider], widths="equal"),
                step_duration_slider,
            ],
            align="stretch",
        )
    )
    _, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(step_time, maxwell_load_extension, color="tab:blue")
    _axes[0].set_title("Maxwell: constant load")
    _axes[0].set_xlabel("Time [s]")
    _axes[0].set_ylabel("Deformation [m]")

    _axes[1].plot(step_time, maxwell_relaxation_force, color="tab:red")
    _axes[1].set_title("Maxwell: constant deformation")
    _axes[1].set_xlabel("Time [s]")
    _axes[1].set_ylabel("Force [N]")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Maxwell model captures two transient ideas: an immediate elastic deformation under load and a relaxing force under fixed deformation. It is less convincing as a long-term passive tissue model because the constant-load deformation keeps increasing and the relaxation force goes to zero.

    **Challenge 1.** In the constant-load plot, estimate the deformation at \(t=0\) and at the final plotted time. Which part comes from the spring, and which part comes from the dashpot? Use the controls to double \(b\) and then double \(k\); explain which change alters the slope.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Voight model

    In the Voight model, the spring and dashpot are in parallel:

    $$
    x = x_k = x_b,\qquad F = F_k + F_b.
    $$

    Combining the element equations gives

    $$
    F = kx + b\dot{x}.
    $$

    For a constant load \(F_0\),

    $$
    x(t)=\frac{F_0}{k}\left(1-e^{-t/\tau}\right),\qquad \tau=\frac{b}{k}.
    $$

    For a constant deformation \(x_0\), the post-step response is

    $$
    F(t)=kx_0.
    $$

    The ideal model would require an infinite force impulse to impose the deformation instantaneously, so the plot below shows only the finite force after the instant of the step.
    """)
    return


@app.cell
def _(step_damping, step_deformation, step_force, step_stiffness, step_time):
    voight_load_extension = voight_constant_load_response(
        step_time,
        force=step_force,
        stiffness=step_stiffness,
        damping=step_damping,
    )
    voight_deformation_force = voight_constant_deformation_response(
        step_time,
        deformation=step_deformation,
        stiffness=step_stiffness,
    )
    return voight_deformation_force, voight_load_extension


@app.cell
def _(
    mo,
    plt,
    step_damping_slider,
    step_deformation_slider,
    step_duration_slider,
    step_force_slider,
    step_stiffness_slider,
    step_time,
    voight_deformation_force,
    voight_load_extension,
):
    mo.output.append(
        mo.vstack(
            [
                mo.hstack([step_force_slider, step_deformation_slider], widths="equal"),
                mo.hstack([step_stiffness_slider, step_damping_slider], widths="equal"),
                step_duration_slider,
            ],
            align="stretch",
        )
    )
    _, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(step_time, voight_load_extension, color="tab:blue")
    _axes[0].set_title("Voight: constant load")
    _axes[0].set_xlabel("Time [s]")
    _axes[0].set_ylabel("Deformation [m]")

    _axes[1].plot(step_time, voight_deformation_force, color="tab:red")
    _axes[1].set_title("Voight: constant deformation")
    _axes[1].set_xlabel("Time [s]")
    _axes[1].set_ylabel("Force [N]")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Voight model captures the finite long-term deformation under constant load. But because the dashpot is in parallel with the spring, it prevents an instantaneous deformation under a finite force. This is why the model starts from zero deformation in the constant-load simulation.

    **Challenge 2.** Compare the Maxwell and Voight constant-load plots. Which one matches the long-term creep shape better? Which one matches the immediate jump better?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kelvin solid model

    The Kelvin solid used in the chapter places a spring in series with a Voight element. Let \(k_s\) be the series spring stiffness, \(k_p\) the parallel spring stiffness, and \(b\) the damping coefficient in the Voight element.

    Under constant load \(F_0\), the total deformation is

    $$
    x(t)=\frac{F_0}{k_s}
    + \frac{F_0}{k_p}\left(1-e^{-t/\tau}\right),
    \qquad \tau=\frac{b}{k_p}.
    $$

    Under constant deformation \(x_0\), the force relaxes from the series spring's instantaneous force to a nonzero equilibrium force:

    $$
    F(t)=F_{\infty} + \left(F_0-F_{\infty}\right)e^{-t/\tau_r},
    $$

    where

    $$
    F_0=k_sx_0,\qquad
    F_{\infty}=\frac{k_sk_p}{k_s+k_p}x_0,\qquad
    \tau_r=\frac{b}{k_s+k_p}.
    $$

    This model is the first one in this notebook that has both a finite immediate response and a finite long-term response in both tests.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    kelvin_series_stiffness_slider = mo.ui.slider(
        20.0,
        300.0,
        10.0,
        value=120.0,
        label="Kelvin series stiffness ks [N/m]",
        show_value=True,
        include_input=True,
    )
    kelvin_parallel_stiffness_slider = mo.ui.slider(
        20.0,
        300.0,
        10.0,
        value=80.0,
        label="Kelvin parallel stiffness kp [N/m]",
        show_value=True,
        include_input=True,
    )
    kelvin_damping_slider = mo.ui.slider(
        5.0,
        150.0,
        5.0,
        value=50.0,
        label="Kelvin damping b [N s/m]",
        show_value=True,
        include_input=True,
    )
    return (
        kelvin_damping_slider,
        kelvin_parallel_stiffness_slider,
        kelvin_series_stiffness_slider,
    )


@app.cell
def _(
    kelvin_damping_slider,
    kelvin_parallel_stiffness_slider,
    kelvin_series_stiffness_slider,
):
    kelvin_series_stiffness = float(kelvin_series_stiffness_slider.value)
    kelvin_parallel_stiffness = float(kelvin_parallel_stiffness_slider.value)
    kelvin_damping = float(kelvin_damping_slider.value)
    return kelvin_damping, kelvin_parallel_stiffness, kelvin_series_stiffness


@app.cell
def _(
    kelvin_damping,
    kelvin_parallel_stiffness,
    kelvin_series_stiffness,
    step_deformation,
    step_force,
    step_time,
):
    kelvin_load_extension = kelvin_constant_load_response(
        step_time,
        force=step_force,
        series_stiffness=kelvin_series_stiffness,
        parallel_stiffness=kelvin_parallel_stiffness,
        damping=kelvin_damping,
    )
    kelvin_relaxation_force = kelvin_constant_deformation_response(
        step_time,
        deformation=step_deformation,
        series_stiffness=kelvin_series_stiffness,
        parallel_stiffness=kelvin_parallel_stiffness,
        damping=kelvin_damping,
    )
    return kelvin_load_extension, kelvin_relaxation_force


@app.cell
def _(
    kelvin_damping_slider,
    kelvin_load_extension,
    kelvin_parallel_stiffness_slider,
    kelvin_relaxation_force,
    kelvin_series_stiffness_slider,
    mo,
    plt,
    step_damping_slider,
    step_deformation_slider,
    step_duration_slider,
    step_force_slider,
    step_stiffness_slider,
    step_time,
):
    mo.output.append(
        mo.vstack(
            [
                mo.hstack([step_force_slider, step_deformation_slider], widths="equal"),
                mo.hstack([step_stiffness_slider, step_damping_slider], widths="equal"),
                step_duration_slider,
                mo.hstack(
                    [
                        kelvin_series_stiffness_slider,
                        kelvin_parallel_stiffness_slider,
                        kelvin_damping_slider,
                    ],
                    widths="equal",
                ),
            ],
            align="stretch",
        )
    )
    _, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(step_time, kelvin_load_extension, color="tab:blue")
    _axes[0].set_title("Kelvin solid: constant load")
    _axes[0].set_xlabel("Time [s]")
    _axes[0].set_ylabel("Deformation [m]")

    _axes[1].plot(step_time, kelvin_relaxation_force, color="tab:red")
    _axes[1].set_title("Kelvin solid: constant deformation")
    _axes[1].set_xlabel("Time [s]")
    _axes[1].set_ylabel("Force [N]")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Guiding questions 3.**
    1. Which part of the Kelvin constant-load response is instantaneous?
    2. Which part changes slowly with time?
    3. Why does the force relaxation stop at a nonzero value instead of decaying to zero?

    **Challenge 3.** Increase \(k_s\) with the slider above. What happens to the size of the immediate deformation under constant load?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare the step responses

    The same inputs now go through all three arrangements. Focus on shape, not only on magnitude. A model can be useful over one time range and fail over another.
    """)
    return


@app.cell
def _(
    kelvin_load_extension,
    kelvin_relaxation_force,
    maxwell_load_extension,
    maxwell_relaxation_force,
    voight_deformation_force,
    voight_load_extension,
):
    load_responses = {
        "Maxwell": maxwell_load_extension,
        "Voight": voight_load_extension,
        "Kelvin solid": kelvin_load_extension,
    }
    deformation_responses = {
        "Maxwell": maxwell_relaxation_force,
        "Voight": voight_deformation_force,
        "Kelvin solid": kelvin_relaxation_force,
    }
    return deformation_responses, load_responses


@app.cell
def _(
    deformation_responses,
    kelvin_damping_slider,
    kelvin_parallel_stiffness_slider,
    kelvin_series_stiffness_slider,
    load_responses,
    mo,
    plt,
    step_damping_slider,
    step_deformation_slider,
    step_duration_slider,
    step_force_slider,
    step_stiffness_slider,
    step_time,
):
    mo.output.append(
        mo.vstack(
            [
                mo.hstack([step_force_slider, step_deformation_slider], widths="equal"),
                mo.hstack([step_stiffness_slider, step_damping_slider], widths="equal"),
                step_duration_slider,
                mo.hstack(
                    [
                        kelvin_series_stiffness_slider,
                        kelvin_parallel_stiffness_slider,
                        kelvin_damping_slider,
                    ],
                    widths="equal",
                ),
            ],
            align="stretch",
        )
    )
    _, _axes = plt.subplots(1, 2, figsize=(10, 4))

    for _label, _response in load_responses.items():
        _axes[0].plot(step_time, _response, label=_label)
    _axes[0].set_title("Constant load comparison")
    _axes[0].set_xlabel("Time [s]")
    _axes[0].set_ylabel("Deformation [m]")
    _axes[0].legend()

    for _label, _response in deformation_responses.items():
        _axes[1].plot(step_time, _response, label=_label)
    _axes[1].set_title("Constant deformation comparison")
    _axes[1].set_xlabel("Time [s]")
    _axes[1].set_ylabel("Force [N]")
    _axes[1].legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the comparison plots as a model-selection exercise:

    - Maxwell is useful for short-term transient behavior but not for finite long-term creep.
    - Voight is useful for finite long-term creep but misses the immediate elastic deformation.
    - Kelvin solid combines the two ideas and is therefore closer to the qualitative response expected from many passive soft tissues.

    **Challenge 4.** Pick one curve from the comparison plot and explain which physical element is responsible for its initial value, its slope, and its final value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A computational model for a prescribed deformation

    Step inputs are useful because they reveal the character of a model quickly. Real movements are rarely perfect steps. Now prescribe a smooth deformation \(x(t)\) and compute the force generated by each arrangement.

    For the Maxwell model:

    $$
    \dot{F} = k\dot{x} - \frac{k}{b}F.
    $$

    For the Voight model:

    $$
    F = kx + b\dot{x}.
    $$

    For the Kelvin solid:

    $$
    \dot{F}
    =
    k_s\dot{x}
    + \frac{k_sk_p}{b}x
    - \frac{k_s+k_p}{b}F.
    $$

    The code below uses a simple forward Euler update for the two force ODEs. The point is not to advocate Euler for every simulation. The point is to expose the state update: current force plus a small time increment times the current rate of change.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sinusoid_offset_slider = mo.ui.slider(
        0.00,
        0.12,
        0.005,
        value=0.05,
        label="Mean deformation [m]",
        show_value=True,
        include_input=True,
    )
    sinusoid_amplitude_slider = mo.ui.slider(
        0.005,
        0.05,
        0.005,
        value=0.02,
        label="Deformation amplitude [m]",
        show_value=True,
        include_input=True,
    )
    sinusoid_frequency_slider = mo.ui.slider(
        0.10,
        3.00,
        0.05,
        value=0.75,
        label="Input frequency [Hz]",
        show_value=True,
        include_input=True,
    )
    sinusoid_stiffness_slider = mo.ui.slider(
        20.0,
        300.0,
        10.0,
        value=100.0,
        label="Maxwell/Voight stiffness k [N/m]",
        show_value=True,
        include_input=True,
    )
    sinusoid_damping_slider = mo.ui.slider(
        5.0,
        150.0,
        5.0,
        value=40.0,
        label="Maxwell/Voight damping b [N s/m]",
        show_value=True,
        include_input=True,
    )
    sinusoid_series_stiffness_slider = mo.ui.slider(
        20.0,
        300.0,
        10.0,
        value=120.0,
        label="Kelvin series stiffness ks [N/m]",
        show_value=True,
        include_input=True,
    )
    sinusoid_parallel_stiffness_slider = mo.ui.slider(
        20.0,
        300.0,
        10.0,
        value=80.0,
        label="Kelvin parallel stiffness kp [N/m]",
        show_value=True,
        include_input=True,
    )
    return (
        sinusoid_amplitude_slider,
        sinusoid_damping_slider,
        sinusoid_frequency_slider,
        sinusoid_offset_slider,
        sinusoid_parallel_stiffness_slider,
        sinusoid_series_stiffness_slider,
        sinusoid_stiffness_slider,
    )


@app.cell
def _(
    np,
    sinusoid_amplitude_slider,
    sinusoid_damping_slider,
    sinusoid_frequency_slider,
    sinusoid_offset_slider,
    sinusoid_parallel_stiffness_slider,
    sinusoid_series_stiffness_slider,
    sinusoid_stiffness_slider,
):
    length_input_time = np.linspace(0, 6, 1201)
    length_input_deformation = (
        sinusoid_offset_slider.value
        + sinusoid_amplitude_slider.value
        * np.sin(2 * np.pi * sinusoid_frequency_slider.value * length_input_time)
    )
    length_input_responses = simulate_length_input_responses(
        length_input_time,
        length_input_deformation,
        stiffness=sinusoid_stiffness_slider.value,
        damping=sinusoid_damping_slider.value,
        series_stiffness=sinusoid_series_stiffness_slider.value,
        parallel_stiffness=sinusoid_parallel_stiffness_slider.value,
    )
    return length_input_deformation, length_input_responses, length_input_time


@app.cell
def _(
    length_input_deformation,
    length_input_responses,
    length_input_time,
    mo,
    plt,
    sinusoid_amplitude_slider,
    sinusoid_damping_slider,
    sinusoid_frequency_slider,
    sinusoid_offset_slider,
    sinusoid_parallel_stiffness_slider,
    sinusoid_series_stiffness_slider,
    sinusoid_stiffness_slider,
):
    mo.output.append(
        mo.vstack(
            [
                mo.hstack(
                    [
                        sinusoid_offset_slider,
                        sinusoid_amplitude_slider,
                        sinusoid_frequency_slider,
                    ],
                    widths="equal",
                ),
                mo.hstack(
                    [sinusoid_stiffness_slider, sinusoid_damping_slider],
                    widths="equal",
                ),
                mo.hstack(
                    [
                        sinusoid_series_stiffness_slider,
                        sinusoid_parallel_stiffness_slider,
                    ],
                    widths="equal",
                ),
            ],
            align="stretch",
        )
    )
    _, _axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    _axes[0].plot(length_input_time, length_input_deformation, color="0.2")
    _axes[0].set_title("Prescribed deformation")
    _axes[0].set_ylabel("Deformation [m]")

    for _label, _color in [
        ("maxwell", "tab:blue"),
        ("voight", "tab:orange"),
        ("kelvin", "tab:green"),
    ]:
        _axes[1].plot(
            length_input_time,
            length_input_responses[_label],
            color=_color,
            label=_label.capitalize(),
        )
    _axes[1].set_title("Force response to the same deformation")
    _axes[1].set_xlabel("Time [s]")
    _axes[1].set_ylabel("Force [N]")
    _axes[1].legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The sinusoidal example is where the dashpot becomes easy to see: force is no longer determined by deformation alone. A viscous element responds to velocity, so force can lead or lag deformation depending on the arrangement and parameters.

    **Challenge 5.** Change the sinusoidal frequency from \(0.75\) Hz to \(0.25\) Hz and then to \(2.0\) Hz. Which model changes the most? Explain your answer using the dashpot term. Then reduce the deformation amplitude and check whether the same model still changes the most.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause before the practice problems and answer these without looking back at the code.

    1. In a series arrangement, which quantity is shared by the elements?
    2. In a parallel arrangement, which quantity is shared by the elements?
    3. Why does a Maxwell model creep indefinitely under constant load?
    4. Why does a Voight model lack an instantaneous finite deformation under constant load?
    5. Why can the Kelvin solid model both jump immediately and still creep toward an asymptote?
    6. In a muscle-tendon model, where would you place the active force generator relative to passive elastic and viscous elements?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practice problems

    Work through these after reading the chapter. Use the code above as a starting point, but first write the equations on paper.

    1. **Parameter audit.** For the Maxwell constant-deformation response, double \(b\) and then double \(k\). Which change increases the relaxation time constant?

    2. **Creep comparison.** Choose \(k\) and \(b\) values so the Voight model reaches approximately 95 percent of its final deformation by \(t=3\) s under a constant load.

    3. **Kelvin design.** Choose \(k_s\), \(k_p\), and \(b\) so the Kelvin solid has a large immediate deformation but only a small additional creep.

    4. **Force relaxation.** For the Kelvin solid, calculate \(F_0\) and \(F_\infty\) by hand for the parameters used in this notebook. Check your calculation against the plot.

    5. **Sinusoidal input.** In the prescribed-deformation simulation, increase the frequency and describe how peak force changes in each model.

    6. **Muscle-model bridge.** Sketch a simple Hill-type muscle model and label which parts are passive elastic, passive viscous, and active force-generating elements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Yamaguchi GT (2001) *Dynamic Modeling of Musculoskeletal Motion: A Vectorized Approach for Biomechanical Analysis in Three Dimensions*. Kluwer Academic Publishers.
    - Zajac FE (1989) Muscle and tendon: properties, models, scaling, and application to biomechanics and motor control. *Critical Reviews in Biomedical Engineering*, 17, 359-411.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
