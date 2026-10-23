# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Velocity and acceleration of a point of a rigid body

    > Renato Naville Watanabe, Marcos Duarte<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A body segment is not a particle. When the shank swings forward, the knee and the ankle move with different velocities. When a wheel rolls, its top races ahead while the point touching the ground stands still. Each point of a moving rigid body has a velocity and an acceleration of its own, and yet they are not independent: the body is rigid, so they are tied together.

    This notebook derives the two equations that tie them together. Given the motion of a single point of the body and the body's angular velocity, they give the velocity and the acceleration of every other point. Along the way they reveal a point that, at each instant, does not move at all, and show why a point that does not move can still be accelerating.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it. Make the prediction; a wrong guess that the code then corrects teaches more than a right one. The **Exercises** come with worked solutions, folded away under each exercise: work the exercise out first, and open the solution only to check your answer.

    You will need the cross product from [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py), the angular velocity from [Angular kinematics in a plane](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py), and the normal (centripetal) acceleration from [Path frame](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/path_frame.py).

    **Challenge 0.** Before you begin, picture a bicycle wheel rolling along the road at 3 m/s. Write down the velocity (magnitude and direction) of three of its points: the hub, the top of the tyre, and the point of the tyre touching the road. Keep your answers nearby; the checkpoint questions ask you to come back to them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Catching the push rim

    A wheelchair user rolls along at a steady 1.5 m/s. To push, the user grabs the push rim, a ring fixed to the wheel and slightly smaller than it, and drives it forward. But before the hand can push, it has to *catch* the rim. At the instant of contact the hand should be moving with the point of the rim it grabs; any difference between the two velocities is absorbed by friction against the palm, and a hand that arrives too slowly brakes the chair instead of propelling it.

    So, how fast is the rim moving where the hand grabs it?

    The hub moves at 1.5 m/s, like the chair. But the wheel also rotates, so every point of the rim moves with a different velocity. And if the wheel does not slip, the point of the tyre touching the ground does not move at all, at least at that instant. Somewhere between "not at all" and something faster than the chair is the speed the hand must match.

    **Guiding questions 0.**

    1. Where on the push rim would the hand need the lowest speed to catch it? And the highest?
    2. The point of the tyre in contact with the ground has zero velocity. Does it also have zero acceleration?
    3. At the instant shown in any photograph of a rolling wheel, is the wheel rotating about its hub, or about the point touching the ground?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the numerical work and Matplotlib draws the figures. Two of the figures are driven by sliders, marimo's interactive controls, to move the body through time.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return FancyArrowPatch, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A frame of reference attached to the body

    The concept of reference frame is central to the understanding of human motion, in biomechanics and in motor control. For example, do we see, plan and control the movement of our hand with respect to reference frames within our body, or in the environment we move in? Or a combination of both? The figure below, although drawn for a robot, illustrates well that we may have to deal with several coordinate systems at once.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/coordinatesystems.png?raw=1" width=450 alt="multiple coordinate systems of a robot"/></center><figcaption><center><i>Figure. Multiple coordinate systems for use in robots (figure from Corke (2017)).</i></center></figcaption></figure>

    In three-dimensional motion analysis we use several frames of reference for convenience, and call them global, laboratory, local, anatomical or technical reference frames or coordinate systems. Different standardizations have been proposed for how to define the frames of reference of the main segments and joints of the human body; the International Society of Biomechanics keeps a [page listing standardization proposals](https://isbweb.org/activities/standards) by its standardization committee and subcommittees.

    In this notebook we need one frame in particular: a frame attached to the moving body, with axes that turn and travel with it. In a motion analysis, such a frame is built from markers placed on the segment, as shown in [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py) and [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py). Here we work in a plane, where the body's frame has two versors, $\hat{\mathbf{i}}'$ and $\hat{\mathbf{j}}'$, and the laboratory's frame has $\hat{\mathbf{i}}$ and $\hat{\mathbf{j}}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Position of a point on a rigid body

    The position of a point P of a rigid body rotating about the origin O is

    $$
    \vec{\mathbf{r}}_{P/O} = x_{P/O}^*\,\hat{\mathbf{i}}' + y_{P/O}^*\,\hat{\mathbf{j}}'
    $$

    where $x_{P/O}^*$ and $y_{P/O}^*$ are the coordinates of the point P in a reference state, when the body's frame coincides with the laboratory's, and the versors of the body's frame, turned by the angle $\theta$, are

    $$
    \hat{\mathbf{i}}' = \cos(\theta)\,\hat{\mathbf{i}} + \sin(\theta)\,\hat{\mathbf{j}}
    $$

    $$
    \hat{\mathbf{j}}' = -\sin(\theta)\,\hat{\mathbf{i}} + \cos(\theta)\,\hat{\mathbf{j}}
    $$

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotBody.png?raw=1" width=500 alt="point of a rotating rigid body"/></center><figcaption><center><i>Figure. A point P of a rigid body rotating about the origin O.</i></center></figcaption></figure>

    This is what rigidity means, written as an equation. As the body rotates, the versors $\hat{\mathbf{i}}'$ and $\hat{\mathbf{j}}'$ turn with it, but the coefficients $x_{P/O}^*$ and $y_{P/O}^*$ of a given point never change. Every bit of the motion of P is carried by the versors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Translation of a rigid body

    Let's consider now the case in which, besides rotating, the body also translates, as in the figure below. Describing P from another point of the body, A, the position of P is

    $$
    \vec{\mathbf{r}}_{P/O} = \vec{\mathbf{r}}_{A/O} + \vec{\mathbf{r}}_{P/A} = \vec{\mathbf{r}}_{A/O} + x_{P/A}^*\,\hat{\mathbf{i}}' + y_{P/A}^*\,\hat{\mathbf{j}}'
    $$

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotTrBody.png?raw=1" width=500 alt="point of a rotating and translating rigid body"/></center><figcaption><center><i>Figure. A point P of a rigid body that rotates and translates, described from the point A of the same body.</i></center></figcaption></figure>

    The first term carries the translation, the motion of A. The second carries the rotation, and has exactly the form of the previous section, with A in place of O.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Angular velocity of a body

    The magnitude of the angular velocity of a rigid body rotating in a plane is defined as

    $$
    \omega = \frac{d\theta}{dt}
    $$

    It is usual to define an angular velocity vector perpendicular to the plane of the rotation (here the $x$-$y$ plane), with magnitude $\omega$:

    $$
    \vec{\boldsymbol{\omega}} = \omega\,\hat{\mathbf{k}}
    $$

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/angvel.png?raw=1" width=400 alt="angular velocity vector"/></center><figcaption><center><i>Figure. The angular velocity vector is perpendicular to the plane of rotation.</i></center></figcaption></figure>

    By the right-hand rule, a positive $\omega$ is a counterclockwise rotation seen from above the plane. Note that the angular velocity belongs to the body as a whole: every line drawn on a rigid body turns through the same angle in the same time, so there is one $\omega$ for the whole body, not one per point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Velocity of a point with no translation

    First, the situation with no translation. The velocity of the point P is

    $$
    \vec{\mathbf{v}}_{P/O} = \frac{d\vec{\mathbf{r}}_{P/O}}{dt} = \frac{d\left(x_{P/O}^*\,\hat{\mathbf{i}}' + y_{P/O}^*\,\hat{\mathbf{j}}'\right)}{dt}
    $$

    The coefficients are constant, so what we need are the derivatives of the versors $\hat{\mathbf{i}}'$ and $\hat{\mathbf{j}}'$. They are found exactly like the derivatives of $\hat{\mathbf{e}}_R$ and $\hat{\mathbf{e}}_\theta$ of the [polar basis](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/polar_basis.py):

    $$
    \frac{d\hat{\mathbf{i}}'}{dt} = -\dot{\theta}\sin(\theta)\,\hat{\mathbf{i}} + \dot{\theta}\cos(\theta)\,\hat{\mathbf{j}} = \dot{\theta}\,\hat{\mathbf{j}}'
    $$

    $$
    \frac{d\hat{\mathbf{j}}'}{dt} = -\dot{\theta}\cos(\theta)\,\hat{\mathbf{i}} - \dot{\theta}\sin(\theta)\,\hat{\mathbf{j}} = -\dot{\theta}\,\hat{\mathbf{i}}'
    $$

    Each versor changes in the direction perpendicular to itself, at a rate equal to the angular velocity. That is a job for the cross product. With the angular velocity as the vector $\vec{\boldsymbol{\omega}} = \dot{\theta}\,\hat{\mathbf{k}} = \omega\,\hat{\mathbf{k}}$, the two expressions become

    $$
    \frac{d\hat{\mathbf{i}}'}{dt} = \dot{\theta}\,\hat{\mathbf{j}}' = \dot{\theta}\,\hat{\mathbf{k}} \times \hat{\mathbf{i}}' = \vec{\boldsymbol{\omega}} \times \hat{\mathbf{i}}'
    $$

    $$
    \frac{d\hat{\mathbf{j}}'}{dt} = -\dot{\theta}\,\hat{\mathbf{i}}' = \dot{\theta}\,\hat{\mathbf{k}} \times \hat{\mathbf{j}}' = \vec{\boldsymbol{\omega}} \times \hat{\mathbf{j}}'
    $$

    So the velocity of the point P, with no translation, is

    $$
    \vec{\mathbf{v}}_{P/O} = x_{P/O}^*\frac{d\hat{\mathbf{i}}'}{dt} + y_{P/O}^*\frac{d\hat{\mathbf{j}}'}{dt} = x_{P/O}^*\,\vec{\boldsymbol{\omega}} \times \hat{\mathbf{i}}' + y_{P/O}^*\,\vec{\boldsymbol{\omega}} \times \hat{\mathbf{j}}' = \vec{\boldsymbol{\omega}} \times \left(x_{P/O}^*\,\hat{\mathbf{i}}' + y_{P/O}^*\,\hat{\mathbf{j}}'\right)
    $$

    $$
    \vec{\mathbf{v}}_{P/O} = \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{P/O}
    $$

    This expression shows that the velocity of any point of a body rotating about O is perpendicular to the vector from O to that point, with magnitude $\omega\,\Vert\vec{\mathbf{r}}_{P/O}\Vert$: the $v = r\omega$ of circular motion, now with a direction attached.

    Although it was derived here for planar motion, the expression is general and holds for three-dimensional rotations as well (see [Angular velocity in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/angular_velocity_3d.py)).

    **Guiding questions 1.**

    1. A point on the axis of rotation has $\vec{\mathbf{r}}_{P/O}$ parallel to $\vec{\boldsymbol{\omega}}$ (in 3D) or equal to zero (in the plane). What is its velocity?
    2. Two points of the same body are at the same distance from O, on opposite sides. Compare their velocities.
    3. Why does $\vec{\boldsymbol{\omega}}$ have to point out of the plane for $\vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}$ to lie in the plane?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A forearm rotates about a fixed elbow, at the origin, with $\omega = 4$ rad/s counterclockwise. At this instant the wrist is 0.25 m from the elbow, along the direction $30^\circ$ above the $+x$ axis. Calculate the velocity of the wrist, as a vector and as a speed.

    /// details | Solution to Exercise 1

    The position of the wrist relative to the elbow is

    $$
    \vec{\mathbf{r}} = 0.25\,(\cos 30^\circ\,\hat{\mathbf{i}} + \sin 30^\circ\,\hat{\mathbf{j}}) = 0.2165\,\hat{\mathbf{i}} + 0.125\,\hat{\mathbf{j}}\;\mathrm{m}
    $$

    With $\hat{\mathbf{k}} \times \hat{\mathbf{i}} = \hat{\mathbf{j}}$ and $\hat{\mathbf{k}} \times \hat{\mathbf{j}} = -\hat{\mathbf{i}}$,

    $$
    \vec{\mathbf{v}} = \omega\,\hat{\mathbf{k}} \times \vec{\mathbf{r}} = 4\,(0.2165\,\hat{\mathbf{j}} - 0.125\,\hat{\mathbf{i}}) = -0.5\,\hat{\mathbf{i}} + 0.866\,\hat{\mathbf{j}}\;\mathrm{m/s}
    $$

    The speed is $\omega\,\Vert\vec{\mathbf{r}}\Vert = 4 \times 0.25 = 1.0$ m/s, and the velocity points $120^\circ$ from the $+x$ axis: perpendicular to the forearm, turned $90^\circ$ counterclockwise from it, as $\hat{\mathbf{k}} \times$ always does in the plane.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relative velocity of a point on a rigid body to another point

    To find the velocity of a point of a body that also translates, we first need the velocity of a point P relative to another point A of the same body:

    $$
    \vec{\mathbf{v}}_{P/A} = \vec{\mathbf{v}}_{P/O} - \vec{\mathbf{v}}_{A/O} = \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{P/O} - \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{A/O} = \vec{\boldsymbol{\omega}} \times \left(\vec{\mathbf{r}}_{P/O} - \vec{\mathbf{r}}_{A/O}\right) = \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{P/A}
    $$

    Seen from any point of a rigid body, every other point moves on a circle around it.

    There is a test of rigidity hidden here. Because $\vec{\mathbf{v}}_{P/A}$ is perpendicular to $\vec{\mathbf{r}}_{P/A}$, the dot product $(\vec{\mathbf{v}}_P - \vec{\mathbf{v}}_A)\cdot\vec{\mathbf{r}}_{P/A}$ is zero: **two points of a rigid body have the same velocity component along the line that joins them**. If they did not, the distance between them would be changing. In [Angular kinematics in a plane](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py), the markers on the leg failed this test by up to 0.4 m/s, a sign of skin movement and measurement error, not of a stretching bone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Two markers are on the thigh: A at $(0, 0.9)$ m with velocity $(1.2, 0.1)$ m/s, and B at $(0.1, 0.5)$ m with a horizontal velocity component of 1.5 m/s. What must the vertical velocity component of B be for the two markers to pass the rigidity test?

    /// details | Solution to Exercise 2

    The test requires $(\vec{\mathbf{v}}_B - \vec{\mathbf{v}}_A)\cdot\vec{\mathbf{r}}_{B/A} = 0$, with $\vec{\mathbf{r}}_{B/A} = (0.1, -0.4)$ m and $\vec{\mathbf{v}}_B - \vec{\mathbf{v}}_A = (0.3,\; v_{By} - 0.1)$ m/s:

    $$
    0.3 \times 0.1 + (v_{By} - 0.1)(-0.4) = 0
    \quad\Longrightarrow\quad
    v_{By} = 0.1 + \frac{0.03}{0.4} = 0.175\;\mathrm{m/s}
    $$

    Any other value would mean that the distance between the markers is changing. For instance, with $v_{By} = 0.2$ m/s, the dot product is $-0.01\;\mathrm{m^2/s}$, and dividing by the distance between the markers, 0.412 m, the markers approach each other at 2.4 cm/s.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Velocity of a point on a rigid body that translates

    Differentiating the position of P when the body rotates and translates, and using the previous result for the rotating part:

    $$
    \vec{\mathbf{v}}_{P/O} = \frac{d\vec{\mathbf{r}}_{P/O}}{dt} = \frac{d\left(\vec{\mathbf{r}}_{A/O} + x_{P/A}^*\,\hat{\mathbf{i}}' + y_{P/A}^*\,\hat{\mathbf{j}}'\right)}{dt} = \frac{d\vec{\mathbf{r}}_{A/O}}{dt} + \frac{d\left(x_{P/A}^*\,\hat{\mathbf{i}}' + y_{P/A}^*\,\hat{\mathbf{j}}'\right)}{dt} = \vec{\mathbf{v}}_{A/O} + \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{P/A}
    $$

    This is the first of the two equations of this notebook. The velocity of any point P is the velocity of one reference point A, which carries the translation, plus a rotational part that is perpendicular to the line from A to P and proportional to its length. Know one point and $\omega$, and you know them all.

    Let's write three small functions for the computations ahead. NumPy's cross product needs three-dimensional vectors, so `omega_cross` adds a zero third component to the planar vector and removes it from the result:
    """)
    return


@app.function
def rotation_matrix(theta):
    """Matrix that rotates planar vectors by the angle theta [rad], counterclockwise."""
    import numpy as np

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@app.function
def omega_cross(omega, r):
    """Cross product omega k x r, for a planar vector r = [x, y] or one vector per row."""
    import numpy as np

    r = np.asarray(r, dtype=float)
    r3 = np.concatenate((r, np.zeros(r.shape[:-1] + (1,))), axis=-1)
    return np.cross([0.0, 0.0, omega], r3)[..., :2]


@app.function
def ellipse_body(n=100):
    """Outline of the example body, an ellipse with semi-axes of 2 m and 6 m,
    in the body's own frame: x in the first row, y in the second."""
    import numpy as np

    phi = np.linspace(0, 2 * np.pi, n)
    return np.vstack((2 * np.cos(phi), 6 * np.sin(phi)))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: a body that rotates and translates

    Below is a body, an ellipse with semi-axes of 2 m and 6 m, rotating with the angular velocity $\omega = \pi/10$ rad/s while its geometric centre, the point A, translates with the velocity $\vec{\mathbf{v}}_A = 0.7\,\hat{\mathbf{i}} + 0.5\,\hat{\mathbf{j}}$ m/s. We follow the point P of the outline that is at the bottom of the body at $t = 0$:
    """)
    return


@app.cell
def _(np):
    ex1_omega = np.pi / 10  # rad/s
    ex1_v_A = np.array([0.7, 0.5])  # m/s, velocity of the geometric centre

    body = ellipse_body()
    tip = body[:, 75]  # coordinates of P in the body's frame

    print("Coordinates of P in the body's frame:", tip.round(4), "m")
    return body, ex1_omega, ex1_v_A, tip


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Drag the slider to move through time. The figure shows the body at the chosen instant, in green, and where it was at every earlier multiple of 2 s, in grey. The red arrow is the velocity of the centre and the blue arrow the velocity of P, both drawn 5 times their size in metres per second.

    **Before you move the slider**, predict: is the blue arrow always longer than the red one? When is it shortest?
    """)
    return


@app.cell
def _(mo):
    t_slider_1 = mo.ui.slider(start=0, stop=13, step=0.5, value=8, label="Time [s]", show_value=True)
    t_slider_1
    return (t_slider_1,)


@app.cell
def _(FancyArrowPatch, body, ex1_omega, ex1_v_A, np, plt, t_slider_1, tip):
    _t = t_slider_1.value
    _fig, _ax = plt.subplots(figsize=(6.5, 6.5))
    for _tg in np.arange(0, _t, 2):
        _outline = rotation_matrix(ex1_omega * _tg) @ body + (ex1_v_A * _tg)[:, np.newaxis]
        _ax.plot(_outline[0], _outline[1], color="0.75", linewidth=1)

    _R = rotation_matrix(ex1_omega * _t)
    _r_A = ex1_v_A * _t
    _outline = _R @ body + _r_A[:, np.newaxis]
    _ax.fill(_outline[0], _outline[1], color="tab:green", alpha=0.6)

    _r_PA = _R @ tip
    _v_P = ex1_v_A + omega_cross(ex1_omega, _r_PA)
    for _start, _vel, _color in [(_r_A, ex1_v_A, "tab:red"), (_r_A + _r_PA, _v_P, "tab:blue")]:
        _ax.add_patch(
            FancyArrowPatch(_start, _start + 5 * _vel, mutation_scale=20, lw=2, arrowstyle="->", color=_color)
        )
    _ax.plot(*_r_A, "k.")
    _ax.set_xlim(-10, 20)
    _ax.set_ylim(-10, 20)
    _ax.set_aspect("equal")
    _ax.grid(True)
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title(f"t = {_t:.1f} s")
    plt.tight_layout()
    plt.show()

    print(f"v_A = {ex1_v_A} m/s, |v_A| = {np.linalg.norm(ex1_v_A):.3f} m/s")
    print(f"v_P = {_v_P.round(3)} m/s, |v_P| = {np.linalg.norm(_v_P):.3f} m/s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rotational part, $\vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{P/A}$, has the same magnitude at every instant, $\omega\,\Vert\vec{\mathbf{r}}_{P/A}\Vert = 1.88$ m/s, but its direction turns with the body. Added to the constant $\vec{\mathbf{v}}_A$, of magnitude 0.86 m/s, it makes the speed of P swing between $1.88 + 0.86 = 2.74$ m/s, when the two parts point the same way, and $1.88 - 0.86 = 1.02$ m/s, when they point in nearly opposite directions, as they do at $t$ = 12 s. So the blue arrow is always the longer one here, but only because the rotational part is larger than the translation. A point closer to the centre, with $\omega\,\Vert\vec{\mathbf{r}}_{P/A}\Vert$ below 0.86 m/s, would at times move more slowly than the centre.

    Velocities from a formula should be checked against velocities from positions. Let's sample the position of P every millisecond for 13 s, differentiate it numerically as in [Kinematics of a particle](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_particle.py), and compare. We also check the rigidity test of the previous section, $(\vec{\mathbf{v}}_P - \vec{\mathbf{v}}_A)\cdot\vec{\mathbf{r}}_{P/A} = 0$:
    """)
    return


@app.cell
def _(ex1_omega, ex1_v_A, np, tip):
    _dt = 0.001
    _t = np.arange(0, 13 + _dt / 2, _dt)
    _r_A = ex1_v_A * _t[:, np.newaxis]
    _r_PA = np.array([rotation_matrix(ex1_omega * _ti) @ tip for _ti in _t])

    _v_numerical = np.gradient(_r_A + _r_PA, _dt, axis=0, edge_order=2)
    _v_formula = ex1_v_A + omega_cross(ex1_omega, _r_PA)

    print(f"Largest |v_numerical - v_formula|: {np.abs(_v_numerical - _v_formula).max():.1e} m/s")
    print(f"Largest |(v_P - v_A) . r_P/A|:     {np.abs(np.sum((_v_numerical - ex1_v_A) * _r_PA, axis=1)).max():.1e} m2/s")
    _speed = np.linalg.norm(_v_formula, axis=1)
    print(f"Speed of P: from {_speed.min():.2f} to {_speed.max():.2f} m/s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The formula and the numerical derivative agree to within $10^{-7}$ m/s, and the relative velocity is perpendicular to $\vec{\mathbf{r}}_{P/A}$ to within rounding error.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: the leg in the swing phase

    The velocity equation links two points of the *same* body. A limb is a chain of bodies, but the equation can be applied link by link: the knee belongs to both the thigh and the shank, so the velocity of the knee, computed from the hip with the thigh's angular velocity, is the starting point for the shank.

    Take a plausible instant of mid-swing in walking, in the sagittal plane, with $x$ forward and $y$ up. The hip, at $(0, 0.90)$ m, moves forward at the walking speed, 1.4 m/s. The thigh (0.42 m long) is flexed $20^\circ$ forward of the vertical and swings forward at $\omega_{thigh}$ = 3 rad/s. The shank (0.43 m long) still hangs $15^\circ$ behind the vertical and swings forward at $\omega_{shank}$ = 4 rad/s. Swinging forward is counterclockwise here, so both angular velocities are positive. Then

    $$
    \vec{\mathbf{v}}_{knee} = \vec{\mathbf{v}}_{hip} + \omega_{thigh}\,\hat{\mathbf{k}} \times \vec{\mathbf{r}}_{knee/hip},
    \qquad
    \vec{\mathbf{v}}_{ankle} = \vec{\mathbf{v}}_{knee} + \omega_{shank}\,\hat{\mathbf{k}} \times \vec{\mathbf{r}}_{ankle/knee}
    $$

    **Before you run the next cell**, predict: how many times faster than the hip does the ankle move?
    """)
    return


@app.cell
def _(np):
    swing_hip = np.array([0.0, 0.90])  # m
    swing_v_hip = np.array([1.4, 0.0])  # m/s, the walking speed
    swing_omega_thigh, swing_omega_shank = 3.0, 4.0  # rad/s, counterclockwise = swinging forward

    # segment vectors from the angles to the downward vertical (positive = forward)
    _r_knee_hip = 0.42 * np.array([np.sin(np.radians(20)), -np.cos(np.radians(20))])
    _r_ankle_knee = 0.43 * np.array([np.sin(np.radians(-15)), -np.cos(np.radians(-15))])
    swing_knee = swing_hip + _r_knee_hip
    swing_ankle = swing_knee + _r_ankle_knee

    swing_v_knee = swing_v_hip + omega_cross(swing_omega_thigh, _r_knee_hip)
    swing_v_ankle = swing_v_knee + omega_cross(swing_omega_shank, _r_ankle_knee)

    print("            position [m]      velocity [m/s]    speed [m/s]")
    for _name, _r, _v in [
        ("hip  ", swing_hip, swing_v_hip),
        ("knee ", swing_knee, swing_v_knee),
        ("ankle", swing_ankle, swing_v_ankle),
    ]:
        print(f"{_name}   {_r.round(3)}   {_v.round(3)}   {np.linalg.norm(_v):6.2f}")
    print(f"\nThe ankle moves {np.linalg.norm(swing_v_ankle) / np.linalg.norm(swing_v_hip):.1f} times faster than the hip")
    return swing_ankle, swing_hip, swing_knee, swing_omega_shank, swing_v_ankle, swing_v_hip, swing_v_knee


@app.cell
def _(plt, swing_ankle, swing_hip, swing_knee, swing_v_ankle, swing_v_hip, swing_v_knee):
    _fig, _ax = plt.subplots(figsize=(6, 5.5))
    _ax.plot(*zip(swing_hip, swing_knee), color="tab:green", linewidth=6, solid_capstyle="round", label="thigh")
    _ax.plot(*zip(swing_knee, swing_ankle), color="tab:olive", linewidth=6, solid_capstyle="round", label="shank")
    for _r, _v in [(swing_hip, swing_v_hip), (swing_knee, swing_v_knee), (swing_ankle, swing_v_ankle)]:
        _ax.quiver(*_r, *(0.1 * _v), angles="xy", scale_units="xy", scale=1, width=0.008, color="tab:blue")
        _ax.plot(*_r, "ko")
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.1, 0.6)
    _ax.set_ylim(0, 1.0)
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title("Velocities in mid-swing (arrows drawn 0.1 s long)")
    _ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ankle moves at 4.25 m/s, three times the walking speed, and almost horizontally. Each link adds its own rotational part: 1.26 m/s from the thigh and 1.72 m/s from the shank, on top of the 1.4 m/s of the hip. The vertical components, upward at the knee and downward from the shank's rotation, nearly cancel at the ankle, which is why the foot skims forward close to the ground instead of rising. This is the same chaining of rigid-body equations that [Kinematic chain](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematic_chain.py) builds up with joint angles.

    **Exercise 3.** Where is the instantaneous centre of rotation of the shank at this instant? Is it on the shank? Which point of the shank moves most slowly?
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, swing_ankle, swing_hip, swing_knee, swing_omega_shank, swing_v_knee):
    _ic = instant_centre(swing_knee, swing_v_knee, swing_omega_shank)
    _from_knee = _ic - swing_knee
    # speed of points along the shank, from the knee (s = 0) to the ankle (s = 1)
    _s = np.linspace(0, 1, 101)
    _points = swing_knee + _s[:, np.newaxis] * (swing_ankle - swing_knee)
    _speeds = np.linalg.norm(swing_v_knee + omega_cross(swing_omega_shank, _points - swing_knee), axis=1)
    mo.accordion(
        {
            "Solution to Exercise 3": mo.md(
                r"""
    Use the knee as the reference point, with the velocity of the knee found above and the shank's angular velocity:

    $$
    \vec{\mathbf{r}}_{IC} = \vec{\mathbf{r}}_{knee} + \frac{\hat{\mathbf{k}} \times \vec{\mathbf{v}}_{knee}}{\omega_{shank}}
    $$
    """
                + f"""
    The instantaneous centre is at ({_ic[0]:.3f}, {_ic[1]:.3f}) m: {np.linalg.norm(_from_knee):.2f} m from the knee, {_from_knee[1]:.2f} m above it and {abs(_from_knee[0]):.2f} m behind it. That is {_ic[1] - swing_hip[1]:.2f} m above the hip, well off the shank: at this instant the shank moves mostly forward, and turns about a point high above it.

    Along the shank, the speed grows with the distance from the instantaneous centre, so the slowest point of the shank is the one closest to it: the knee, at {_speeds[0]:.2f} m/s (the speed rises steadily to {_speeds[-1]:.2f} m/s at the ankle). Of all its points, only the instantaneous centre itself, which is not part of the shank, has zero velocity.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back to the push rim: rolling without slipping

    Now the wheelchair. Take a wheel of radius $R$ = 0.30 m, with a push rim of radius $r$ = 0.27 m, rolling to the right ($+x$) at $v$ = 1.5 m/s. Use the hub as the reference point A.

    If the wheel does not slip, the point C of the tyre in contact with the ground is momentarily at rest, $\vec{\mathbf{v}}_C = 0$. The velocity equation then gives the angular velocity:

    $$
    \vec{\mathbf{v}}_C = \vec{\mathbf{v}}_A + \omega\,\hat{\mathbf{k}} \times \vec{\mathbf{r}}_{C/A} = v\,\hat{\mathbf{i}} + \omega\,\hat{\mathbf{k}} \times (-R\,\hat{\mathbf{j}}) = (v + \omega R)\,\hat{\mathbf{i}} = 0
    \quad\Longrightarrow\quad
    \omega = -\frac{v}{R}
    $$

    The minus sign says the wheel turns clockwise as it rolls to the right, as it should. Here that is $-5$ rad/s, about 0.8 turns per second.

    **Before you run the next cell**, return to Guiding question 0.1 and put numbers on it. How fast does the top of the push rim move? And its lowest point?
    """)
    return


@app.cell
def _(np):
    v_chair = 1.5  # m/s
    R_wheel = 0.30  # m
    r_rim = 0.27  # m
    omega_wheel = -v_chair / R_wheel  # rad/s, rolling without slipping

    _v_A = np.array([v_chair, 0.0])
    print(f"omega = {omega_wheel:.1f} rad/s\n")
    print("Point of the push rim         velocity [m/s]      speed [m/s]")
    for _beta in [0, 30, 60, 90, 120, 150, 180]:
        _r_PA = r_rim * np.array([np.sin(np.radians(_beta)), np.cos(np.radians(_beta))])
        _v_P = _v_A + omega_cross(omega_wheel, _r_PA)
        print(f"{_beta:3d} degrees ahead of the top   {_v_P.round(3) + 0.0}   {np.linalg.norm(_v_P):6.3f}")
    return R_wheel, omega_wheel, r_rim, v_chair


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The angle of each point is measured from the top of the push rim, positive towards the front of the chair. The top of the push rim moves at 2.85 m/s, almost twice the speed of the chair; its lowest point at only 0.15 m/s. The rotational part, $\omega r$ = 1.35 m/s, is added to the chair's speed at the top and subtracted at the bottom. A user who grabs the rim on its upper part, where the hand can push forward, must therefore swing the hand forward at well over 2 m/s just to catch it, faster than the chair itself.

    Let's draw the velocities of points all around the tyre and the push rim, together with the point C:
    """)
    return


@app.cell
def _(R_wheel, np, omega_wheel, plt, r_rim, v_chair):
    _beta = np.radians(np.arange(0, 360, 30))
    _fig, _ax = plt.subplots(figsize=(7, 6))
    _phi = np.linspace(0, 2 * np.pi, 200)
    _ax.axhline(0, color="k", linewidth=2)
    for _radius, _color, _label in [(R_wheel, "tab:blue", "tyre"), (r_rim, "tab:red", "push rim")]:
        _ax.plot(_radius * np.cos(_phi), R_wheel + _radius * np.sin(_phi), color=_color, linewidth=1)
        _r_PA = _radius * np.column_stack((np.sin(_beta), np.cos(_beta)))
        _v_P = np.array([v_chair, 0.0]) + omega_cross(omega_wheel, _r_PA)
        _ax.quiver(
            _r_PA[:, 0], R_wheel + _r_PA[:, 1], 0.1 * _v_P[:, 0], 0.1 * _v_P[:, 1],
            angles="xy", scale_units="xy", scale=1, width=0.005, color=_color, label=_label,
        )
    _ax.quiver(0, R_wheel, 0.1 * v_chair, 0, angles="xy", scale_units="xy", scale=1, width=0.005, color="k", label="hub")
    _ax.plot(0, 0, "ko", markersize=8)
    _ax.annotate("C", (0, 0), xytext=(-0.05, -0.06), fontsize=14)
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.45, 0.75)
    _ax.set_ylim(-0.12, 0.75)
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title("Velocities of a rolling wheel (arrows drawn 0.1 s long)")
    _ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the arrows as a whole. Every one of them is perpendicular to the line joining its point to C, and its length is proportional to the distance from C: the top of the tyre, twice as far from C as the hub, moves twice as fast, at 3 m/s. At this instant the wheel moves exactly as if it were rotating about C. That answers Guiding question 0.3: about both, in a sense, but only C is at rest.

    ### The instantaneous centre of rotation

    Any rigid body moving in a plane with $\omega \neq 0$ has, at each instant, such a point of zero velocity: the **instantaneous centre of rotation**. Setting $\vec{\mathbf{v}}_{IC} = \vec{\mathbf{v}}_A + \omega\,\hat{\mathbf{k}} \times \vec{\mathbf{r}}_{IC/A} = 0$ and solving for $\vec{\mathbf{r}}_{IC/A}$ gives

    $$
    \vec{\mathbf{r}}_{IC/A} = \frac{\hat{\mathbf{k}} \times \vec{\mathbf{v}}_A}{\omega}
    $$

    as you can verify by substituting it back: $\omega\,\hat{\mathbf{k}} \times (\hat{\mathbf{k}} \times \vec{\mathbf{v}}_A)/\omega = -\vec{\mathbf{v}}_A$ for any $\vec{\mathbf{v}}_A$ in the plane. The instantaneous centre lies on the perpendicular to $\vec{\mathbf{v}}_A$, at a distance $\Vert\vec{\mathbf{v}}_A\Vert/|\omega|$: the faster the translation compared with the rotation, the further away it is. A body that only translates ($\omega = 0$) has its instantaneous centre at infinity.
    """)
    return


@app.function
def instant_centre(r_A, v_A, omega):
    """Instantaneous centre of rotation of a rigid body moving in a plane, from
    the position r_A and velocity v_A of one of its points and its angular
    velocity omega (nonzero)."""
    import numpy as np

    return np.asarray(r_A, dtype=float) + omega_cross(1.0, v_A) / omega


@app.cell
def _(R_wheel, ex1_omega, ex1_v_A, np, omega_wheel, v_chair):
    print("Wheel, instantaneous centre:", instant_centre([0, R_wheel], [v_chair, 0], omega_wheel).round(4) + 0.0, "m")
    _ic = instant_centre([0, 0], ex1_v_A, ex1_omega)
    print(f"Ellipse, instantaneous centre relative to its centre: {_ic.round(2)} m, {np.linalg.norm(_ic):.2f} m away")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the wheel the formula lands exactly on the contact point. For the ellipse of the previous example, the instantaneous centre is 2.74 m from its geometric centre, and it stays at that position relative to the centre throughout the motion, because $\vec{\mathbf{v}}_A$ and $\omega$ are both constant. The body turns around it: at some instants the instantaneous centre lies inside the ellipse, at others outside, and it is not attached to any particular point of the body.

    The instantaneous centre has a long history in joint biomechanics. A hinge would keep its centre of rotation fixed in both bones; the knee does not. Its instantaneous centre moves as the knee flexes, because the femoral condyles roll and slide on the tibia, and Frankel et al. (1971) used the path of the instantaneous centres of the knee to study its internal derangements.

    **Guiding questions 2.**

    1. Where is the instantaneous centre of a body that rotates about a fixed pivot? Of a body that only translates?
    2. During the stance phase of walking, roughly where is the instantaneous centre of the shank? (Hint: which point of the shank moves least while the foot is flat on the ground?)
    3. The instantaneous centre of the wheel is the point C of the tyre. A tenth of a second later, is it still the same point of the tyre?

    **Challenge 1.** Add the instantaneous centre to the slider figure of the ellipse, and draw dashed lines from it to the centre and to P. Check by eye that each velocity arrow is perpendicular to its dashed line.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: when the wheel slips

    Rolling without slipping is a condition, not a law: on a wet floor or a loose surface, a wheel can spin or skid. The instantaneous centre then leaves the contact point, and where it goes tells you which way the wheel is slipping.

    Take the wheelchair wheel ($R$ = 0.30 m) with its hub moving forward at 1.0 m/s. Rolling without slipping would need $\omega = -v/R = -3.33$ rad/s. Consider two cases: the user pushes too hard on a slippery floor and the wheel *spins* at $-5$ rad/s, or the user brakes and the wheel turns too slowly, at $-3$ rad/s, so that it *skids*.

    **Before you run the next cell**, predict for each case: does the contact point slide forward or backward? Is the instantaneous centre above or below the ground?
    """)
    return


@app.cell
def _(R_wheel, np):
    _v_hub = np.array([1.0, 0.0])  # m/s
    _r_hub = np.array([0.0, R_wheel])  # m, the contact point is at the origin
    for _omega, _case in [(-5.0, "spinning"), (-1.0 / R_wheel, "rolling"), (-3.0, "skidding")]:
        _v_contact = _v_hub + omega_cross(_omega, -_r_hub)
        _ic = instant_centre(_r_hub, _v_hub, _omega)
        print(
            f"{_case:9s} omega = {_omega:5.2f} rad/s:  velocity of the contact point {_v_contact.round(3) + 0.0} m/s,"
            f"  instantaneous centre at height {_ic[1]:+.3f} m"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A spinning wheel slides backward at the contact point, at 0.5 m/s, and its instantaneous centre rises inside the wheel, 10 cm above the ground: the wheel turns as if around a point between the floor and the hub. A skidding wheel slides forward, at 0.1 m/s, and its instantaneous centre drops 3.3 cm *below* the ground. Only rolling without slipping puts it exactly at the contact point.

    The same reasoning applies to a foot. During stance the instantaneous centre of the foot should stay near the part of the foot in contact with the ground; when it drifts away from the contact, the foot is sliding, as in a slip on ice.

    **Exercise 4.** Seen from above, a wheelchair has its two wheels 0.6 m apart, on a common axle. The left wheel moves forward at 1.0 m/s and the right wheel at 0.5 m/s. Treating the chair as a rigid body moving in the horizontal plane, with $x$ forward and $y$ to the left, find its angular velocity and its instantaneous centre. Which way is the chair turning?

    /// details | Solution to Exercise 4

    Put the left wheel at $(0, 0.3)$ m and the right wheel at $(0, -0.3)$ m. The velocity equation from the left wheel to the right wheel gives

    $$
    0.5\,\hat{\mathbf{i}} = 1.0\,\hat{\mathbf{i}} + \omega\,\hat{\mathbf{k}} \times (-0.6\,\hat{\mathbf{j}}) = (1.0 + 0.6\,\omega)\,\hat{\mathbf{i}}
    \quad\Longrightarrow\quad
    \omega = -0.833\;\mathrm{rad/s}
    $$

    The minus sign means clockwise seen from above: the chair turns to the right, towards the slower wheel. The instantaneous centre, from the left wheel, is

    $$
    \vec{\mathbf{r}}_{IC} = \vec{\mathbf{r}}_{L} + \frac{\hat{\mathbf{k}} \times \vec{\mathbf{v}}_{L}}{\omega} = 0.3\,\hat{\mathbf{j}} + \frac{1.0\,\hat{\mathbf{j}}}{-0.833} = -0.9\,\hat{\mathbf{j}}\;\mathrm{m}
    $$

    It lies on the line of the axle, 0.9 m to the right of the centre of the chair and 0.6 m beyond the right wheel: the chair is driving round a circle of radius 0.9 m, at the speed of its centre, 0.75 m/s.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Acceleration of a point on a rigid body

    The acceleration of a point on a rigid body is obtained by differentiating the velocity equation. The derivative of $\vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{P/A}$ follows the product rule, and the derivative of $\vec{\mathbf{r}}_{P/A}$ is $\vec{\mathbf{v}}_{P/A} = \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{P/A}$:

    $$
    \begin{aligned}
    \vec{\mathbf{a}}_{P/O} &= \vec{\mathbf{a}}_{A/O} + \dot{\vec{\boldsymbol{\omega}}} \times \vec{\mathbf{r}}_{P/A} + \vec{\boldsymbol{\omega}} \times \vec{\mathbf{v}}_{P/A} \\
    &= \vec{\mathbf{a}}_{A/O} + \dot{\vec{\boldsymbol{\omega}}} \times \vec{\mathbf{r}}_{P/A} + \vec{\boldsymbol{\omega}} \times (\vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}_{P/A}) \\
    &= \vec{\mathbf{a}}_{A/O} + \ddot{\theta}\,\hat{\mathbf{k}} \times \vec{\mathbf{r}}_{P/A} - \dot{\theta}^2\,\vec{\mathbf{r}}_{P/A}
    \end{aligned}
    $$

    The last line holds for planar motion, where $\vec{\boldsymbol{\omega}}$ is perpendicular to $\vec{\mathbf{r}}_{P/A}$. (In general, $\vec{\boldsymbol{\omega}} \times (\vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}) = (\vec{\boldsymbol{\omega}}\cdot\vec{\mathbf{r}})\,\vec{\boldsymbol{\omega}} - \omega^2\,\vec{\mathbf{r}}$.) This is the second equation of this notebook, and the acceleration has three terms:

    - $\vec{\mathbf{a}}_{A/O}$: the acceleration of the reference point A, which carries the translation.
    - $\ddot{\theta}\,\hat{\mathbf{k}} \times \vec{\mathbf{r}}_{P/A}$: the acceleration of P due to the angular acceleration of the body. It is perpendicular to $\vec{\mathbf{r}}_{P/A}$, the tangential acceleration $r\alpha$ of circular motion.
    - $-\dot{\theta}^2\,\vec{\mathbf{r}}_{P/A}$: the acceleration of P due to the angular velocity of the body. It points from P to A, and is known as the centripetal acceleration, $\omega^2 r$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: a body that speeds up

    Below is the same body, now with an angular acceleration of $\alpha = \pi/150$ rad/s² and an initial angular velocity of $\omega_0 = \pi/100$ rad/s. Its centre accelerates with $\vec{\mathbf{a}}_A = 0.01\,\hat{\mathbf{i}} + 0.05\,\hat{\mathbf{j}}$ m/s², starting from rest. The angle, the angular velocity and the motion of the centre follow from constant accelerations:

    $$
    \theta = \omega_0 t + \frac{\alpha}{2}t^2, \qquad \omega = \omega_0 + \alpha t, \qquad \vec{\mathbf{r}}_A = \frac{\vec{\mathbf{a}}_A}{2}t^2
    $$

    The blue arrow is the acceleration of P, drawn 5 times its size in m/s².

    **Before you move the slider**, predict: which of the three terms dominates the acceleration of P at the start, and which at the end?
    """)
    return


@app.cell
def _(np):
    ex2_alpha = np.pi / 150  # rad/s2, angular acceleration
    ex2_omega0 = np.pi / 100  # rad/s, initial angular velocity
    ex2_a_A = np.array([0.01, 0.05])  # m/s2, acceleration of the centre
    return ex2_a_A, ex2_alpha, ex2_omega0


@app.cell
def _(mo):
    t_slider_2 = mo.ui.slider(start=0, stop=20, step=0.5, value=20, label="Time [s]", show_value=True)
    t_slider_2
    return (t_slider_2,)


@app.cell
def _(FancyArrowPatch, body, ex2_a_A, ex2_alpha, ex2_omega0, np, plt, t_slider_2, tip):
    _t = t_slider_2.value
    _fig, _ax = plt.subplots(figsize=(6.5, 6.5))
    for _tg in np.arange(0, _t, 4):
        _outline = rotation_matrix(ex2_omega0 * _tg + ex2_alpha / 2 * _tg**2) @ body + (ex2_a_A / 2 * _tg**2)[:, np.newaxis]
        _ax.plot(_outline[0], _outline[1], color="0.75", linewidth=1)

    _theta = ex2_omega0 * _t + ex2_alpha / 2 * _t**2
    _omega = ex2_omega0 + ex2_alpha * _t
    _r_A = ex2_a_A / 2 * _t**2
    _R = rotation_matrix(_theta)
    _outline = _R @ body + _r_A[:, np.newaxis]
    _ax.fill(_outline[0], _outline[1], color="tab:green", alpha=0.6)

    _r_PA = _R @ tip
    _tangential = omega_cross(ex2_alpha, _r_PA)
    _centripetal = -(_omega**2) * _r_PA
    _a_P = ex2_a_A + _tangential + _centripetal
    _ax.add_patch(
        FancyArrowPatch(_r_A + _r_PA, _r_A + _r_PA + 5 * _a_P, mutation_scale=20, lw=2, arrowstyle="->", color="tab:blue")
    )
    _ax.plot(*_r_A, "k.")
    _ax.set_xlim(-10, 20)
    _ax.set_ylim(-10, 20)
    _ax.set_aspect("equal")
    _ax.grid(True)
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title(f"t = {_t:.1f} s, omega = {_omega:.3f} rad/s")
    plt.tight_layout()
    plt.show()

    print(f"|a_A|         = {np.linalg.norm(ex2_a_A):.3f} m/s2")
    print(f"|alpha x r|   = {np.linalg.norm(_tangential):.3f} m/s2")
    print(f"|omega^2 r|   = {np.linalg.norm(_centripetal):.3f} m/s2")
    print(f"|a_P|         = {np.linalg.norm(_a_P):.3f} m/s2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how the three terms evolve, and check the formula against the numerical second derivative of the position of P:
    """)
    return


@app.cell
def _(ex2_a_A, ex2_alpha, ex2_omega0, np, tip):
    _dt = 0.001
    _t = np.arange(0, 20 + _dt / 2, _dt)
    _theta = ex2_omega0 * _t + ex2_alpha / 2 * _t**2
    _omega = ex2_omega0 + ex2_alpha * _t
    _r_A = ex2_a_A * _t[:, np.newaxis] ** 2 / 2
    _r_PA = np.array([rotation_matrix(_th) @ tip for _th in _theta])

    _a_numerical = np.gradient(np.gradient(_r_A + _r_PA, _dt, axis=0, edge_order=2), _dt, axis=0, edge_order=2)
    _a_formula = ex2_a_A + omega_cross(ex2_alpha, _r_PA) - _omega[:, np.newaxis] ** 2 * _r_PA
    _inner = slice(5, -5)  # the first and last samples of a second difference are less accurate
    print(f"Largest |a_numerical - a_formula|: {np.abs(_a_numerical[_inner] - _a_formula[_inner]).max():.1e} m/s2\n")

    print("   t     omega    |a_A|  |alpha x r|  |omega^2 r|   |a_P|")
    for _time in [0, 5, 10, 15, 20]:
        _i = np.argmin(np.abs(_t - _time))
        print(
            f"{_t[_i]:4.0f} s  {_omega[_i]:.3f}  {np.linalg.norm(ex2_a_A):7.3f}"
            f"  {np.linalg.norm(omega_cross(ex2_alpha, _r_PA[_i])):9.3f}"
            f"  {_omega[_i] ** 2 * np.linalg.norm(_r_PA[_i]):11.3f}  {np.linalg.norm(_a_formula[_i]):7.3f}  m/s2"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The acceleration of the centre is small and constant, 0.051 m/s². The tangential term is constant too, 0.126 m/s², because both $\alpha$ and the distance from A to P are constant. The centripetal term grows with the *square* of the angular velocity: negligible at the start, 0.006 m/s², it is the largest of the three by $t$ = 10 s, and almost ten times the tangential term by $t$ = 20 s, when the body turns at 0.45 rad/s. A steady angular acceleration produces an acceleration of P that is anything but steady.

    Notice also that at $t$ = 15 s the magnitude of the total, 0.678 m/s², is smaller than that of the centripetal term alone: the terms are vectors, and they partly cancel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: the giant swing

    In a giant swing on the high bar, a gymnast circles the bar with the body extended, hands fixed on the bar. The hands are a fixed pivot, so $\vec{\mathbf{a}}_A = 0$ and only the two rotational terms are left. Treat the body as rigid, hanging from the bar, with the shoulders 0.6 m, the hips 1.25 m and the feet 2.2 m from the hands, and consider two instants of the downswing:

    - at the bottom of the swing, where the body turns fastest, $\omega$ = 5 rad/s, and so is momentarily not speeding up, $\alpha = 0$;
    - $45^\circ$ before the bottom, with $\omega$ = 4 rad/s and $\alpha$ = 8 rad/s².

    These values are plausible for a giant swing, not measurements. **Before you run the next cell**, predict: at the bottom, which way do the accelerations point, and how large is the acceleration of the feet compared with $g$? And $45^\circ$ before the bottom, do the accelerations still point at the bar?
    """)
    return


@app.cell
def _(np):
    giant_distances = {"shoulders": 0.6, "hips": 1.25, "feet": 2.2}  # m from the hands
    giant_instants = [  # (label, body angle from the downward vertical [deg], omega [rad/s], alpha [rad/s2])
        ("bottom of the swing", 0, 5.0, 0.0),
        ("45 degrees before the bottom", -45, 4.0, 8.0),
    ]

    for _label, _angle, _omega, _alpha in giant_instants:
        _direction = np.array([np.sin(np.radians(_angle)), -np.cos(np.radians(_angle))])  # from the bar along the body
        print(f"{_label}: omega = {_omega} rad/s, alpha = {_alpha} rad/s2")
        for _name, _d in giant_distances.items():
            _r = _d * _direction
            _a = omega_cross(_alpha, _r) - _omega**2 * _r
            _to_bar = np.degrees(np.arccos(np.dot(_a, -_r) / (np.linalg.norm(_a) * np.linalg.norm(_r))))
            print(
                f"  {_name:9s} |a| = {np.linalg.norm(_a):5.1f} m/s2 = {np.linalg.norm(_a) / 9.81:.1f} g,"
                f"  {_to_bar:4.1f} degrees from the line to the bar"
            )
    return giant_distances, giant_instants


@app.cell
def _(giant_distances, giant_instants, np, plt):
    _fig, _axs = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    for _ax, (_label, _angle, _omega, _alpha) in zip(_axs, giant_instants):
        _direction = np.array([np.sin(np.radians(_angle)), -np.cos(np.radians(_angle))])
        _feet = giant_distances["feet"] * _direction
        _ax.plot([0, _feet[0]], [0, _feet[1]], color="tab:green", alpha=0.35, linewidth=4, solid_capstyle="round")
        _ax.plot(0, 0, "ks", markersize=10)
        for _d in giant_distances.values():
            _r = _d * _direction
            _a = omega_cross(_alpha, _r) - _omega**2 * _r
            _ax.plot(*_r, "o", color="tab:green", markersize=7)
            _ax.quiver(*_r, *(0.01 * _a), angles="xy", scale_units="xy", scale=1, width=0.012, color="tab:blue", zorder=3)
        _ax.set_aspect("equal")
        _ax.set_xlim(-2.0, 1.2)
        _ax.set_ylim(-2.5, 0.4)
        _ax.set_xlabel("x [m]")
        _ax.set_title(_label)
    _axs[0].set_ylabel("y [m]")
    _fig.suptitle("Accelerations of the shoulders, hips and feet (arrows: 0.01 m per m/s²)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At the bottom of the swing the accelerations point straight at the bar, and they grow in proportion to the distance from it: 1.5 $g$ at the shoulders, 3.2 $g$ at the hips, 5.6 $g$ at the feet. With $\alpha = 0$ only the centripetal term is left, $\omega^2 r$. The arms, which hold the whole body, must supply the force for these accelerations and support the body's weight as well, which is why the pull on the hands at the bottom of a giant swing reaches several times body weight.

    Before the bottom the accelerations are smaller, but they no longer point at the bar: every one of them is tilted by $26.6^\circ$ from the line to the bar, towards the direction of motion. The tilt is the same for every point, because both terms are proportional to the distance $r$: the tangential term is $\alpha r$, the centripetal term $\omega^2 r$, and the angle between their sum and the body satisfies

    $$
    \tan\varphi = \frac{\alpha}{\omega^2} = \frac{8}{16} = 0.5
    $$

    So for any body rotating about a fixed pivot, the acceleration of every point makes the same angle with the line to the pivot, and its magnitude is proportional to the distance, $r\sqrt{\alpha^2 + \omega^4}$.

    **Exercise 5.** The forearm of Exercise 1 is now slowing down: about the fixed elbow, $\omega$ = 4 rad/s counterclockwise with $\alpha = -20$ rad/s². The wrist is 0.25 m from the elbow, $30^\circ$ above the $+x$ axis. Calculate the tangential and centripetal accelerations of the wrist, the total acceleration and its magnitude, and the angle between the acceleration and the line from the wrist to the elbow.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    _omega, _alpha = 4.0, -20.0
    _r = 0.25 * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    _tangential = omega_cross(_alpha, _r)
    _centripetal = -(_omega**2) * _r
    _a = _tangential + _centripetal
    _phi = np.degrees(np.arccos(np.dot(_a, -_r) / (np.linalg.norm(_a) * np.linalg.norm(_r))))

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.plot([0, _r[0]], [0, _r[1]], color="tab:green", alpha=0.35, linewidth=3, solid_capstyle="round")
    _ax.plot(0, 0, "ko", markersize=8)
    for _vec, _color, _name in [(_tangential, "tab:orange", "tangential"), (_centripetal, "tab:purple", "centripetal"), (_a, "tab:blue", "total")]:
        _ax.quiver(*_r, *(0.04 * _vec), angles="xy", scale_units="xy", scale=1, width=0.012, color=_color, label=_name, zorder=3)
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.15, 0.4)
    _ax.set_ylim(-0.2, 0.3)
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title("Wrist acceleration (0.04 m per m/s²)")
    _ax.legend(loc="upper left")
    plt.tight_layout()
    _figure = mo.as_html(_fig)
    plt.close(_fig)

    mo.accordion(
        {
            "Solution to Exercise 5": mo.vstack(
                [
                    mo.md(
                        r"""
    With $\vec{\mathbf{r}} = (0.2165, 0.125)$ m from Exercise 1, the two rotational terms are

    $$
    \vec{\mathbf{a}}_t = \alpha\,\hat{\mathbf{k}} \times \vec{\mathbf{r}}, \qquad \vec{\mathbf{a}}_n = -\omega^2\,\vec{\mathbf{r}}
    $$
    """
                        + f"""
    - tangential: ({_tangential[0]:.3f}, {_tangential[1]:.3f}) m/s², of magnitude |α| r = {np.linalg.norm(_tangential):.1f} m/s². Because α is negative, it points opposite to the velocity of Exercise 1: the forearm is braking.
    - centripetal: ({_centripetal[0]:.3f}, {_centripetal[1]:.3f}) m/s², of magnitude ω² r = {np.linalg.norm(_centripetal):.1f} m/s², pointing from the wrist to the elbow.
    - total: ({_a[0]:.3f}, {_a[1]:.3f}) m/s², of magnitude {np.linalg.norm(_a):.2f} m/s².

    The total makes an angle of {_phi:.1f}° with the line from the wrist to the elbow, the angle whose tangent is |α|/ω² = 20/16. It is tilted away from the elbow on the side opposite to the motion, because the forearm is slowing down.
    """
                    ),
                    _figure,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zero velocity is not zero acceleration

    Back to Guiding question 0.2. The contact point C of the rolling wheel has zero velocity. What about its acceleration?

    **Before you run the next cell**, make a prediction. Then apply the acceleration equation to C with the chair rolling at constant speed: $\vec{\mathbf{a}}_A = 0$ and $\dot{\omega} = 0$, so only the centripetal term is left,

    $$
    \vec{\mathbf{a}}_C = -\omega^2\,\vec{\mathbf{r}}_{C/A} = -\omega^2(-R\,\hat{\mathbf{j}}) = \omega^2 R\,\hat{\mathbf{j}} = \frac{v^2}{R}\,\hat{\mathbf{j}}
    $$

    Let's check it numerically, following a point of the tyre through the instant it touches the ground, sampled at 1 kHz:
    """)
    return


@app.cell
def _(R_wheel, np, omega_wheel, v_chair):
    _dt = 0.001
    _t = np.arange(-0.3, 0.3 + _dt / 2, _dt)  # the point touches the ground at t = 0
    _hub = np.column_stack((v_chair * _t, np.full_like(_t, R_wheel)))
    _point = _hub + np.array([rotation_matrix(omega_wheel * _ti) @ [0, -R_wheel] for _ti in _t])

    _v = np.gradient(_point, _dt, axis=0, edge_order=2)
    _a = np.gradient(_v, _dt, axis=0, edge_order=2)
    _i = np.argmin(np.abs(_t))
    print(f"At contact: position {_point[_i].round(4) + 0.0} m")
    print(f"            velocity {_v[_i].round(4) + 0.0} m/s")
    print(f"            acceleration {_a[_i].round(3) + 0.0} m/s2")
    print(f"v^2 / R = {v_chair**2 / R_wheel:.2f} m/s2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Zero velocity, and an acceleration of 7.5 m/s² straight up, three quarters of $g$. The contact point stops for an instant and immediately leaves the ground again, upwards. Its path shows why. Here are two turns of the wheel, centred on the instant $t = 0$ when the point of the tyre touches the ground, with the path of that point and of the point of the push rim just above it, at the bottom of the rim:
    """)
    return


@app.cell
def _(R_wheel, np, omega_wheel, plt, r_rim, v_chair):
    _period = 2 * np.pi * R_wheel / v_chair  # s, time for one turn
    _t = np.linspace(-_period, _period, 800)
    _hub = np.column_stack((v_chair * _t, np.full_like(_t, R_wheel)))
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.axhline(0, color="k", linewidth=2)
    _ax.plot(_hub[:, 0], _hub[:, 1], color="k", linestyle="--", linewidth=1, label="hub")
    for _radius, _color, _label in [(R_wheel, "tab:blue", "point of the tyre"), (r_rim, "tab:red", "point of the push rim")]:
        _path = _hub + np.array([rotation_matrix(omega_wheel * _ti) @ [0, -_radius] for _ti in _t])
        _ax.plot(_path[:, 0], _path[:, 1], color=_color, linewidth=2, label=_label)
    _ax.set_aspect("equal")
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title("Paths during two turns of the wheel")
    _ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The point of the tyre traces a [cycloid](https://en.wikipedia.org/wiki/Cycloid), with a sharp cusp each time it touches the ground: it arrives moving straight down, stops, and leaves straight up. Its velocity passes through zero there, but the velocity is changing direction completely, and that takes an acceleration. The point of the push rim never stops: at the bottom of the rim it still moves forward, at 0.15 m/s.

    The lesson is about the instantaneous centre. It is the point of zero *velocity*, so velocities can be computed as if the body rotated about it. But it is not a fixed pivot, and it generally has an acceleration, so accelerations cannot be computed that way. For accelerations, use the full equation with a reference point whose acceleration you know, such as the hub.

    **Guiding questions 3.**

    1. With the chair rolling at constant speed, what is the acceleration of the hub? Of the top of the push rim? (Use the acceleration equation.)
    2. If the chair speeds up, which terms of the acceleration equation appear for the contact point C? Do they change its acceleration?
    3. A sprinter's foot is flat on the track during stance. Is the instantaneous centre of the shank at rest in the laboratory? Is it accelerating?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 6.** A cyclist rides at a constant 8 m/s, pedalling at a steady 90 revolutions per minute with cranks 0.17 m long. Seen from the right side, with the bicycle moving in the $+x$ direction, the cranks turn clockwise. Calculate the velocity, relative to the ground, of a pedal at the lowest point of its circle and at the highest point, and the acceleration of the pedal at the lowest point. Does the cyclist's foot ever move backward relative to the ground?

    /// details | Solution to Exercise 6

    The angular velocity of the cranks is $\omega = -90 \times 2\pi/60 = -9.42$ rad/s (clockwise), so the rotational part of the pedal's velocity has magnitude $|\omega|\,r = 9.42 \times 0.17 = 1.60$ m/s. Take the crank axle, which moves with the bicycle at 8 m/s, as the reference point. At the lowest point, $\vec{\mathbf{r}} = -0.17\,\hat{\mathbf{j}}$ m and

    $$
    \vec{\mathbf{v}} = 8\,\hat{\mathbf{i}} + \omega\,\hat{\mathbf{k}} \times (-0.17\,\hat{\mathbf{j}}) = (8 + 0.17\,\omega)\,\hat{\mathbf{i}} = 6.40\,\hat{\mathbf{i}}\;\mathrm{m/s}
    $$

    At the highest point the rotational part adds instead, $8 + 1.60 = 9.60$ m/s. The rotational part never exceeds 1.60 m/s, far below the 8 m/s of the bicycle, so the foot always moves forward relative to the ground, between 6.40 and 9.60 m/s. (Compare it with the contact point of the bicycle's wheel, whose rotational part exactly cancels the speed of the bicycle.)

    The axle moves at constant velocity and the cranks turn at constant speed, so only the centripetal term is left. At the lowest point it points up, towards the axle:

    $$
    \vec{\mathbf{a}} = -\omega^2\,\vec{\mathbf{r}} = 9.42^2 \times 0.17\,\hat{\mathbf{j}} = 15.1\,\hat{\mathbf{j}}\;\mathrm{m/s^2}
    $$

    That is 1.5 $g$, at every point of the circle, always pointing at the axle.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your answers to Challenge 0. Was the point of the tyre touching the road at rest in your answer? Was the top moving at 6 m/s?
    2. Two markers on the same rigid segment have velocities whose components along the line between them differ by 0.2 m/s. What does that tell you?
    3. Why is one angular velocity enough for the whole body, while each point needs its own linear velocity?
    4. Give an example, other than a wheel, of a point of the body that has zero velocity and nonzero acceleration at some instant of a movement.
    5. Of the three terms of the acceleration, which one grows fastest when a limb rotates faster, and why does that matter for the forces at the joints?
    6. In three dimensions, $\vec{\boldsymbol{\omega}} \times (\vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}})$ is not always $-\omega^2\,\vec{\mathbf{r}}$. For which vectors $\vec{\mathbf{r}}$ is it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. A wheelchair user rolls at 1.2 m/s on wheels of radius 0.30 m with push rims of radius 0.27 m.<br>
       a. Calculate the angular velocity of the wheels.<br>
       b. The hand grabs the push rim 20° behind its top. With what velocity (magnitude and direction) must the hand be moving at that instant so as not to slip on the rim?<br>
       c. The user now accelerates the chair at 0.8 m/s², still rolling without slipping. Calculate the angular acceleration of the wheels and the acceleration of the point of the tyre touching the ground.

    2. At one instant of a kick, a marker on the ankle is at (0, 0.05) m with a velocity of (0.5, 0.1) m/s, and a marker on the knee is at (0.05, 0.45) m with a velocity of (−0.7, 0.25) m/s, both in the sagittal plane.<br>
       a. Check that the two velocities are consistent with a rigid shank.<br>
       b. Calculate the angular velocity of the shank.<br>
       c. Find the instantaneous centre of rotation of the shank.

    3. With the data of Table A.1 of Winter (2009), used in [Angular kinematics in a plane](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py), calculate the velocity of the head of the fibula from the velocity of the ankle and the angular velocity of the leg, with the equation of this notebook, and compare it with the velocity obtained by differentiating the position of the head of the fibula. Where do they disagree the most, and why?

    4. In late swing, the hip of a walker is at (0, 0.90) m and moves forward at 1.4 m/s. The thigh (0.42 m) is flexed 25° forward of the vertical and turns forward at 0.5 rad/s; the shank (0.43 m) is 10° forward of the vertical and turns forward at 7 rad/s.<br>
       a. Calculate the velocities of the knee and the ankle.<br>
       b. Find the instantaneous centre of the shank. Compare it with the one in mid-swing (Exercise 3), and explain the difference.

    5. For the giant swing of this notebook, at the bottom of the swing:<br>
       a. At what angular velocity would the acceleration of the feet reach 10 $g$? And that of the hips?<br>
       b. The gymnast flexes the arms, bringing every point of the body 0.2 m closer to the bar. By what percentage does the acceleration of each point change at $\omega$ = 5 rad/s? Why is the change not the same for all of them?

    6. To turn on the spot, a wheelchair user pushes the left wheel forward and pulls the right wheel backward, so that the wheels move at 0.4 m/s in opposite directions. The wheels are 0.6 m apart.<br>
       a. Calculate the angular velocity of the chair and find its instantaneous centre.<br>
       b. Calculate the velocity of the front of the footrest, 0.5 m ahead of the midpoint of the axle.

    7. Write a function that receives the positions and velocities of two markers on a segment moving in a plane and returns the angular velocity of the segment and its instantaneous centre. Use the component of the relative velocity perpendicular to the line between the markers to find $\omega$, and report the component along the line as a measure of how far the markers are from rigid. Test it on two points of the ellipse of this notebook, sampled from the body's outline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers to the problems

    1. (a) $\omega=-4$ rad/s for forward motion to the right. (b) At 20° behind the top, the hand velocity is $(2.21487,0.36938)$ m/s: magnitude 2.24546 m/s, 9.468° upward from forward. (c) $\alpha=-2.66667$ rad/s²; the instantaneous ground-contact material point has acceleration $(0,4.8)$ m/s², upward.
    2. (a) The relative velocity is perpendicular to the marker separation, so the observations are rigid-compatible. (b) $\omega=3$ rad/s counterclockwise. (c) Instantaneous centre $(-0.033333,0.216667)$ m.
    3. The predicted fibular velocity is $\mathbf v_F=\mathbf v_A+\omega(-r_y,r_x)$. Differentiated measured positions additionally include the measured radial length-change component and differentiation error; discrepancies grow where noise, soft-tissue motion and end-of-record differentiation dominate. Equality is exact only for a rigid segment with consistently differentiated motion.
    4. With forward positive $x$, upward positive $y$ and increasing forward flexion, knee velocity $(1.59032,0.08875)$ m/s and ankle velocity $(4.55460,0.61143)$ m/s. Shank instantaneous centre $(0.16482,0.74654)$ m; it is near the knee because the shank rotates rapidly relative to the translating thigh.
    5. (a) Feet: 6.678 rad/s; hips: 8.859 rad/s. (b) At fixed angular speed, acceleration falls by 33.33% for shoulders, 16.00% for hips and 9.09% for feet. Each radius shortens by the same 0.2 m but by a different fraction.
    6. Magnitude of chair angular velocity 1.3333 rad/s, with centre at the axle midpoint. Footrest speed 0.6667 m/s perpendicular to the chair’s forward direction, toward the turning side.
    7. With $r=p_2-p_1$ and $u=v_2-v_1$, final expressions are $\omega=(r_xu_y-r_yu_x)/(r\cdot r)$, radial residual $u\cdot r/\|r\|$, and $c=p_1+(-v_{1y},v_{1x})/\omega$. Coincident markers are invalid; at $\omega=0$ there is no finite instantaneous centre for nonzero translation.

    ///
    """)
    return








@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read chapter 16 (circular motion of a rigid object), sections 17.1 (rigid object kinematics) and 17.3 (kinematics of rolling and sliding) of chapter 17, and sections 18.2 (rotating frames and their base vectors) and 18.3 (general formulas for velocity and acceleration) of chapter 18 of [Ruina and Pratap's book](https://ruina.tam.cornell.edu/Book/index.html).
    - [Angular kinematics in a plane](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py): angular velocity and acceleration from marker data, and $v = r\omega$ on a real leg.
    - [Polar basis](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/polar_basis.py): the rotating basis whose derivatives were used here.
    - [Rigid-body transformations in a plane](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py): the rotation matrix used to move the body.
    - [Angular velocity in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/angular_velocity_3d.py): $\vec{\mathbf{v}} = \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}$ when the rotation is not confined to a plane.

    ### Video lectures on the Internet

    - [Kinematics Of Rigid Bodies - General Plane Motion - Solved Problems](https://www.youtube.com/watch?v=4LsLy9iJKFA)
    - [Kinematics of Rigid Bodies -Translation And Rotation About Fixed Axis - Rectilinear and Rotational](https://www.youtube.com/watch?v=VnzsQmP6eMQ)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional assignments from the original sources

    These original reading assignments remain available for further study. Their full statements and source-specific answer keys are external to this guide.

    - Solve the problems 16.2.5, 16.2.10, 16.2.11 and 16.2.20 from [Ruina and Pratap's book](https://ruina.tam.cornell.edu/Book/index.html).

    - Solve the problems 17.1.2, 17.1.8, 17.1.9, 17.1.10, 17.1.11 and 17.1.12 from [Ruina and Pratap's book](https://ruina.tam.cornell.edu/Book/index.html).

    - Solve the problems 2.1, 2.2, 2.7, 2.8, 2.17
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Corke P (2017) [Robotics, Vision and Control: Fundamental Algorithms in MATLAB](https://petercorke.com/RVC/). 2nd ed. Springer-Verlag Berlin.

    - Frankel VH, Burstein AH, Brooks DB (1971) [Biomechanics of internal derangement of the knee: pathomechanics as determined by analysis of the instant centers of motion](https://pubmed.ncbi.nlm.nih.gov/5109122/). Journal of Bone and Joint Surgery, American Volume, 53(5), 945–962.

    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](https://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.

    - Winter DA (2009) [Biomechanics and Motor Control of Human Movement](https://books.google.com.br/books?id=_bFHL08IWfwC). 4th edition. Hoboken, USA: Wiley.
    """)
    return


if __name__ == "__main__":
    app.run()
