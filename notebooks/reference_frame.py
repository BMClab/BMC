# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Frame of reference

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    "Where is the hand?" has no answer until someone says *relative to what*. Relative to the shoulder, a sprinter's hand swings back and forth over the same short arc; relative to the track, it travels a hundred metres in ten seconds. Every number in motion analysis is measured from somewhere, along some axes, and many confusions in kinematics turn out to be confusions about which "somewhere" was meant.

    This notebook is about that "somewhere". It starts with the physics: why motion is relative, and why some frames of reference are better than others for writing Newton's laws. It ends with the practical recipe that turns three markers on a body segment into a coordinate system, and with an anatomical convention that says which way the axes of that coordinate system must point.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a result *before* the code prints it. Make the prediction; a wrong guess that the code then corrects teaches more than a right one.

    The **Exercises** are short practice problems placed where the material they need has just been covered. Each one has a collapsed solution right below it: try the exercise first, and open the solution only to check your answer or when you are truly stuck.

    You will need the vector operations from [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py): the norm, the dot product and the cross product.

    **Challenge 0.** Before you begin, describe the motion of your hand as you scroll down this page, three times: relative to your shoulder, relative to the room, and relative to the screen. Which of the three descriptions is the simplest? Keep your answers nearby; the checkpoint questions ask you to come back to them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running on a treadmill

    Treadmills are everywhere in biomechanics laboratories. They keep the runner in front of the cameras, at a speed the experimenter controls, for as long as the experiment needs. But a nagging question comes with them: **is running on a treadmill the same as running on the ground?**

    Seen from the laboratory, the two look nothing alike. On the ground, the runner's body travels forward at, say, 4 m/s, and each foot stays put while it is in contact with the ground. On the treadmill, the body stays where it is, and each foot is carried backward by the belt at 4 m/s while it is in contact. Different motions, it seems; so perhaps different forces, different muscle actions, a different energy cost.

    Now watch the treadmill through the eyes of an ant standing on the belt. To the ant, each foot is at rest while it touches the belt, and the runner's body travels forward at 4 m/s, exactly as on the ground.

    Which observer is right? The question was settled by van Ingen Schenau (1980), and by the end of the next section you will be able to reproduce his argument in a few lines of code.

    **Guiding questions 0.**

    1. The laboratory observer and the ant disagree about the velocity of the runner's body. Do they also disagree about its acceleration?
    2. Which of the two observers sees the runner the way a coach standing beside an athletics track would?
    3. A real treadmill belt slows down slightly every time a foot lands on it. Would that matter for the comparison? Why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the numerical work and Matplotlib draws the figures. The function `CCS`, from the `functions` folder of this repository, draws coordinate systems in three dimensions; it is loaded later, where it is first needed.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Motion is relative

    <a href="https://en.wikipedia.org/wiki/Motion_(physics)">Motion</a>, a change of position in space with respect to time, is not an absolute concept. A reference is needed to describe the motion of an object, and the state of that reference cannot be absolute either: all motion is relative.

    A [frame of reference](https://en.wikipedia.org/wiki/Frame_of_reference) is whatever we choose to describe the motion with respect to: the laboratory floor, the treadmill belt, the pelvis, the Earth. In a frame of reference we define a [coordinate system](https://en.wikipedia.org/wiki/Coordinate_system), a set of axes along which positions are measured. The two terms are often used interchangeably, but they are not the same thing. The frame is the physical reference; the coordinate system is the bookkeeping attached to it. The same laboratory floor can carry a coordinate system with X pointing forward or with X pointing to the right, with its origin in a corner or at the centre of a force plate. Choosing between them changes the numbers, not the motion.

    How much can the choice of frame change the description? Let's look at one hand in two frames.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One hand, two frames

    A sprinter runs at 9 m/s. Seen from the shoulder, the hand swings back and forth like a pendulum: we model the arm as a rod of 0.6 m from shoulder to hand, swinging 50° either side of the vertical at 2.2 cycles per second, one cycle per stride. Seen from the track, the same hand also travels with the body.

    **Before you run the next cell**, predict: is the hand ever moving faster than the runner, relative to the track? Is it ever almost at rest?
    """)
    return


@app.cell
def _(np, plt):
    _V, _L, _f = 9.0, 0.6, 2.2  # running speed [m/s], arm length [m], arm swing frequency [Hz]
    _amp = np.radians(50)  # amplitude of the arm swing
    _dt = 0.001
    _t = np.arange(0, 1 + _dt / 2, _dt)

    _theta = _amp * np.sin(2 * np.pi * _f * _t)  # arm angle from the vertical
    _hand_shoulder = np.column_stack((_L * np.sin(_theta), -_L * np.cos(_theta)))
    _hand_track = _hand_shoulder + np.column_stack((_V * _t, np.zeros_like(_t)))  # Galilean transformation

    _speed_shoulder = np.linalg.norm(np.gradient(_hand_shoulder, _dt, axis=0, edge_order=2), axis=1)
    _speed_track = np.linalg.norm(np.gradient(_hand_track, _dt, axis=0, edge_order=2), axis=1)
    print(f"Hand speed relative to the shoulder: up to {_speed_shoulder.max():.1f} m/s")
    print(f"Hand speed relative to the track:    from {_speed_track.min():.1f} to {_speed_track.max():.1f} m/s")

    _, _axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 2.4]})
    _axs[0].plot(_hand_shoulder[:, 0], _hand_shoulder[:, 1], color="tab:blue", linewidth=3)
    _axs[0].plot(0, 0, "ko", markersize=8)
    _axs[0].set_aspect("equal")
    _axs[0].set_xlim(-0.7, 0.7)
    _axs[0].set_ylim(-0.75, 0.1)
    _axs[0].set_xlabel("x [m]")
    _axs[0].set_ylabel("y [m]")
    _axs[0].set_title("Relative to the shoulder")
    _axs[1].plot(_V * _t, np.zeros_like(_t), "k--", linewidth=1, label="shoulder")
    _axs[1].plot(_hand_track[:, 0], _hand_track[:, 1], color="tab:blue", linewidth=2, label="hand")
    _axs[1].set_ylim(-0.75, 0.1)
    _axs[1].set_xlabel("x [m]")
    _axs[1].set_title("Relative to the track, during 1 s")
    _axs[1].legend(loc="center right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Relative to the shoulder, the hand moves back and forth along the same short arc, never faster than 7.2 m/s. Relative to the track, it never goes backwards at all. On the forward swing it races ahead at 16.2 m/s, nearly twice the speed of the runner; at the bottom of the backswing, where it moves backwards relative to the shoulder almost as fast as the body carries it forwards, it slows down to 1.8 m/s. (The vertical scales of the two panels are the same; the horizontal scale of the second one is squeezed.)

    Both descriptions are correct. Which one is useful depends on the question: the shoulder's frame for the technique of the arm swing, the track's for whether the hand will hit a hurdle.

    **Exercise 1.** A 400 m runner moves at 7 m/s, and at the lowest point of the arm swing the hand moves at 5 m/s relative to the shoulder. What is the speed of the hand relative to the track at that point during the forward swing? And during the backswing?

    /// details | Solution to Exercise 1
    At the lowest point of the swing the velocity of the hand relative to the shoulder is horizontal, parallel to the velocity of the body, so the two velocities add as numbers along the direction of running (a Galilean transformation):

    - forward swing: $7 + 5 = 12$ m/s forward;
    - backswing: $7 - 5 = 2$ m/s, still forward.

    Relative to the track, the hand of a runner moves backwards only if it swings backwards faster than the body moves forwards.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inertial and non-inertial frames

    Often, the choice of reference frame and coordinate system is made for convenience, as with the hand above. There is one distinction between frames, though, that is not a matter of taste.

    When we deal with the dynamics of motion, where we want to understand the forces related to the motion, frames of reference split into two kinds. In an [inertial frame of reference](https://en.wikipedia.org/wiki/Inertial_frame_of_reference) (also called a Galilean reference frame), Newton's laws of motion are valid in their simple form. In a non-inertial frame they are not: the accelerations measured there cannot be explained by the forces acting, and to make the accounts balance we have to add *fictitious* forces, forces that no body exerts.

    An inertial frame is at rest or moves at constant velocity with respect to another inertial frame. There is no absolute rest to measure from, so "at rest" always means at rest relative to some other frame. A non-inertial frame is one that accelerates with respect to an inertial frame.

    The concept of frame of reference has changed drastically from Aristotle to Galileo, Newton and Einstein. To read more about that and its philosophical implications, see [Space and Time: Inertial Frames](https://plato.stanford.edu/entries/spacetime-iframes/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two observers of the same runner

    Let's put numbers on the treadmill question. We describe the horizontal motion of the runner's centre of mass with a deliberately simple model: an average speed $V$ = 4 m/s, plus a small fluctuation in every step, because the body is braked slightly in the first half of each ground contact and pushed forward in the second half. With a fluctuation of amplitude $\Delta V$ = 0.15 m/s at a step frequency $f$ = 2.8 steps per second, the velocity relative to the ground is

    $$
    v(t) = V + \Delta V \sin(2\pi f t)
    $$

    and the position is its integral,

    $$
    x(t) = V t - \frac{\Delta V}{2\pi f}\cos(2\pi f t)
    $$

    A sinusoid is a caricature of a real stride, but it is enough for the argument. If running on a treadmill is like running on the ground, this is also the motion of the treadmill runner *relative to the belt*. As in a motion capture session, we sample it at 100 Hz:
    """)
    return


@app.cell
def _(np):
    V, dV, f = 4.0, 0.15, 2.8  # m/s, m/s, Hz
    dt = 0.01  # s, 100 Hz
    t = np.arange(0, 2 + dt / 2, dt)

    x_belt = V * t - dV / (2 * np.pi * f) * np.cos(2 * np.pi * f * t)
    return V, dt, t, x_belt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The laboratory sees the belt move backward at the constant speed $V$. The position of the runner relative to the laboratory is therefore the position relative to the belt minus the distance the belt has carried it:

    $$
    x_{lab}(t) = x_{belt}(t) - V t
    $$

    This change of frame, subtracting the displacement of one frame relative to the other, is a **Galilean transformation**.

    **Before you run the next cell**, predict three things. How far does the runner's centre of mass move in the laboratory in two seconds? How do the velocities seen by the two observers compare? And the accelerations?
    """)
    return


@app.cell
def _(V, dt, np, t, x_belt):
    x_lab = x_belt - V * t

    v_belt = np.gradient(x_belt, dt, edge_order=2)
    a_belt = np.gradient(v_belt, dt, edge_order=2)
    v_lab = np.gradient(x_lab, dt, edge_order=2)
    a_lab = np.gradient(v_lab, dt, edge_order=2)

    print(f"x_lab stays between {1000 * x_lab.min():.1f} and {1000 * x_lab.max():.1f} mm")
    print(f"v_belt - v_lab: from {(v_belt - v_lab).min():.6f} to {(v_belt - v_lab).max():.6f} m/s")
    print(f"Largest |a_belt - a_lab|: {np.abs(a_belt - a_lab).max():.1e} m/s2")
    print(
        f"Peak horizontal acceleration: {np.abs(a_belt).max():.2f} m/s2,"
        f" a force of {70 * np.abs(a_belt).max():.0f} N for a 70 kg runner"
    )
    return a_belt, a_lab, v_belt, v_lab, x_lab


@app.cell
def _(a_belt, a_lab, plt, t, v_belt, v_lab):
    _, _axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    _axs[0].plot(t, v_belt, color="tab:blue", linewidth=2, label="relative to the belt")
    _axs[0].plot(t, v_lab, color="tab:orange", linewidth=2, label="relative to the laboratory")
    _axs[0].set_ylabel("Velocity [m/s]")
    _axs[0].set_title("Horizontal motion of the runner's centre of mass")
    _axs[0].legend(loc="center right")
    _axs[1].plot(t, a_belt, color="tab:blue", linewidth=5, alpha=0.5, label="relative to the belt")
    _axs[1].plot(t, a_lab, color="tab:orange", linewidth=1.5, linestyle="--", label="relative to the laboratory")
    _axs[1].set_ylabel("Acceleration [m/s$^2$]")
    _axs[1].set_xlabel("Time [s]")
    _axs[1].set_ylim(-3.2, 4.6)
    _axs[1].legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the laboratory, the centre of mass barely moves: it rocks back and forth by less than a centimetre either way, which is what makes a treadmill so convenient for measurement. The two velocities differ by exactly 4 m/s at every instant, the speed of the belt. And the two accelerations are the same, to within $10^{-12}\;\mathrm{m/s^2}$, which is rounding error. In the lower panel the dashed curve lies exactly on top of the solid one.

    That last result is the whole argument. Differentiate the Galilean transformation twice:

    $$
    x_{lab} = x_{belt} - V t
    \quad\Longrightarrow\quad
    v_{lab} = v_{belt} - V
    \quad\Longrightarrow\quad
    a_{lab} = a_{belt}
    $$

    Because $V$ is constant, it disappears from the acceleration. Newton's second law links force to acceleration, so both observers infer the same horizontal force on the runner: a peak of about 180 N for a 70 kg runner, braking and then propelling the body in each step. A frame moving at constant velocity relative to an inertial frame is itself inertial, and in all inertial frames the forces come out the same.

    That is van Ingen Schenau's conclusion: if the belt speed is constant and air resistance is negligible, running on a treadmill is mechanically equivalent to running on the ground. The differences that remain in practice come from breaking those conditions, not from the change of frame: there is no air resistance on a treadmill, and a real belt slows down slightly each time a foot lands on it.

    **Guiding questions 1.**

    1. Go back to Guiding question 0.1. Which kinematic quantities depend on the frame of reference, and which do not, as long as both frames are inertial?
    2. The laboratory sees the foot move backward at 4 m/s during contact; the ant on the belt sees it at rest. Do the two observers agree on the kinetic energy of the foot?
    3. A runner on a treadmill slowly drifts backward and, after a minute, steps off the back of the belt. Does the drift break the equivalence argument? What would you have to add to the model to describe it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When the frame accelerates

    Now suppose the belt speeds up, at $A$ = 0.5 m/s², and the runner keeps up, staying in the same place in the laboratory. The runner's motion relative to the laboratory is the same as before. Relative to the belt, the runner now goes faster and faster: to the $V t$ of before we must add the $A t^2/2$ by which the speeding belt carries its surface further back.
    """)
    return


@app.cell
def _(V, dt, np, t, x_lab):
    _A = 0.5  # m/s2, the belt speeds up
    _x_belt_accel = x_lab + V * t + _A * t**2 / 2

    _a_lab = np.gradient(np.gradient(x_lab, dt, edge_order=2), dt, edge_order=2)
    _a_belt_accel = np.gradient(np.gradient(_x_belt_accel, dt, edge_order=2), dt, edge_order=2)
    _diff = _a_belt_accel - _a_lab
    print(f"a_belt - a_lab: from {_diff.min():.4f} to {_diff.max():.4f} m/s2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two observers now disagree about the acceleration, by exactly the acceleration of the belt, at every instant. The laboratory is (very nearly) inertial, so its account is the one in which forces and accelerations match: the runner does not accelerate on average, and the average net horizontal force is zero. The observer riding the belt sees the runner accelerate forward at an extra 0.5 m/s², and no force to account for it. In that frame Newton's second law fails, unless we add a fictitious force of $m A$, 35 N for a 70 kg runner, pointing forward.

    In general, if a frame accelerates at $\overrightarrow{\mathbf{a}}_{frame}$ relative to an inertial frame, accelerations measured in it are $\overrightarrow{\mathbf{a}}' = \overrightarrow{\mathbf{a}} - \overrightarrow{\mathbf{a}}_{frame}$, and Newton's second law written in that frame picks up an extra term:

    $$
    m\,\overrightarrow{\mathbf{a}}' = \overrightarrow{\mathbf{F}} \; \underbrace{-\, m\,\overrightarrow{\mathbf{a}}_{frame}}_{\text{fictitious force}}
    $$

    It is the same push you feel when a bus brakes: no one pushes you forward; the bus, your frame of reference, is decelerating.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A force plate in a lift

    A force plate measures the force between the feet and the ground. Put one in a lift, with a 70 kg person standing still on it, and take the lift up one floor: it speeds up at 1.5 m/s² for 1.5 s, travels at a constant speed for 3 s, and slows down at 1.5 m/s² for 1.5 s.

    **Before you run the next cell**, predict what the force plate reads in each of the three phases. Is the reading during the constant-speed part the person's weight, or more?
    """)
    return


@app.cell
def _(np, plt):
    _m, _g = 70.0, 9.81  # kg, m/s2
    _dt = 0.01
    _t = np.arange(0, 8 + _dt / 2, _dt)
    _a = np.where((_t >= 1) & (_t < 2.5), 1.5, 0.0) + np.where((_t >= 5.5) & (_t < 7), -1.5, 0.0)
    _v = np.concatenate(([0], np.cumsum((_a[1:] + _a[:-1]) / 2 * _dt)))  # velocity of the lift
    _F = _m * (_g + _a)  # force of the plate on the person, from the laboratory: F - m g = m a

    for _label, _time in [("at rest", 0.5), ("speeding up", 2), ("constant speed", 4), ("slowing down", 6)]:
        _i = np.argmin(np.abs(_t - _time))
        print(f"{_label:15s} lift speed {_v[_i]:4.2f} m/s   plate reading {_F[_i]:6.1f} N")
    print(f"\nBody weight: {_m * _g:.1f} N")

    _, _axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    _axs[0].plot(_t, _v, color="tab:blue", linewidth=2)
    _axs[0].set_ylabel("Lift speed [m/s]")
    _axs[1].plot(_t, _F, color="tab:red", linewidth=2, label="plate reading")
    _axs[1].axhline(_m * _g, color="0.5", linestyle="--", label="body weight")
    _axs[1].set_ylabel("Force [N]")
    _axs[1].set_xlabel("Time [s]")
    _axs[1].legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plate reads 791.7 N while the lift speeds up, 15% more than the person's weight, and 581.7 N while it slows down. During the constant-speed part it reads exactly the weight, 686.7 N, although the person is moving upwards at 2.25 m/s.

    Seen from the laboratory, an inertial frame, the person accelerates with the lift, and Newton's second law needs a net upward force: $F - m g = m a$, so $F = m(g + a)$. Seen from the lift, the person is at rest, and the accounts balance only with a fictitious force $-m\,a_{lift}$, pointing down while the lift accelerates upwards: the "heavy" feeling at the start of the ride. At constant speed the lift is an inertial frame again, and the fictitious force vanishes.

    The lesson for measurement: a force plate tells you about the motion of the body relative to an inertial frame only if the plate itself is in one.

    **Exercise 2.** Imagine a force plate on the deck of a ship. The deck heaves up and down with a vertical acceleration $a(t) = 0.8\sin(2\pi\,0.1\,t)$ m/s². A 75 kg person stands still on the plate. Between which values does the reading oscillate, and how large is the error relative to the person's weight?
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _m, _g = 75.0, 9.81
    _t = np.arange(0, 20, 0.01)
    _F = _m * (_g + 0.8 * np.sin(2 * np.pi * 0.1 * _t))
    _W = _m * _g
    mo.accordion(
        {
            "Solution to Exercise 2": mo.md(
                rf"""
    As in the lift, the plate reads $F = m\,(g + a)$. Over one heave cycle of 10 s, the reading oscillates between **{_F.min():.1f} N** and **{_F.max():.1f} N**, around the weight of {_W:.1f} N: an error of up to {_F.max() - _W:.1f} N, or {100 * (_F.max() - _W) / _W:.1f}% of the weight, in each direction.

    The error is simply $m\,a_{{max}} = 75 \times 0.8 = 60$ N. It is slow (a 10 s period) compared with a jump, but far from negligible: any force plate used on a moving platform needs the platform's own acceleration measured and subtracted.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the laboratory an inertial frame?

    Strictly, no. The laboratory sits on the surface of a rotating Earth. Every point of that surface, except at the poles, travels in a circle once a day, and moving in a circle means accelerating towards its centre, with $a = \omega^2 r$ (see [Path frame](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/path_frame.py)). How large is that acceleration?
    """)
    return


@app.cell
def _(np):
    _omega = 2 * np.pi / 86164  # rad/s, one turn per sidereal day
    _R = 6.378e6  # m, equatorial radius of the Earth

    _a_c = _omega**2 * _R
    print(f"Centripetal acceleration at the equator: {_a_c:.4f} m/s2, {100 * _a_c / 9.81:.2f}% of g")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At most 0.034 m/s², about a third of a percent of the gravitational acceleration, and far smaller than the accelerations of human movement. That is why biomechanics treats the laboratory as an inertial frame without a second thought. The Earth's rotation matters for long-range ballistics, for weather systems and for Foucault's pendulum, not for a jump on a force plate.

    **Exercise 3.** In some tests of the vestibular system, a person sits on a chair that rotates at a constant rate about a vertical axis. At half a turn per second, what is the centripetal acceleration of a point of the head 10 cm from the axis of rotation? Is the chair an inertial frame?

    /// details | Solution to Exercise 3
    Half a turn per second is $\omega = \pi$ rad/s, so

    $$
    a = \omega^2 r = \pi^2 \times 0.1 = 0.99\;\mathrm{m/s^2}
    $$

    about 10% of $g$, and some thirty times the acceleration due to the Earth's rotation. The chair is not an inertial frame: every point of it, except those on the axis, accelerates towards the axis. In the chair's frame, a fictitious force $m\,\omega^2 r$ pointing away from the axis (the centrifugal force) has to be added, and a second one (the Coriolis force) for anything that moves relative to the chair.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frame of reference for human motion analysis

    In anatomy, we use a simplified reference frame made of perpendicular planes to provide a standard reference for qualitatively describing the structures and movements of the human body, as shown in the next figure.

    <figure><center><img src="https://upload.wikimedia.org/wikipedia/commons/3/34/BodyPlanes.jpg" width=300 alt="Anatomical body position and body planes"/></center><figcaption><center><i>Figure. Anatomical body position and body planes (<a href="https://en.wikipedia.org/wiki/Human_anatomical_terms">image from Wikipedia</a>).</i></center></figcaption></figure>

    The planes are qualitative. "The knee flexes in the sagittal plane" is a description, not a measurement. To measure, we need numbers, and numbers need a coordinate system.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cartesian coordinate system

    As we perceive the surrounding space as three-dimensional, a convenient coordinate system is the [Cartesian coordinate system](https://en.wikipedia.org/wiki/Cartesian_coordinate_system) in [Euclidean space](https://en.wikipedia.org/wiki/Euclidean_space), with three orthogonal axes as shown below. The directions of the axes are commonly defined by the [right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule) and labelled X, Y and Z. The orthogonality of the Cartesian coordinate system is what makes it so convenient in classical mechanics: space is assumed to have [Euclidean geometry](https://en.wikipedia.org/wiki/Euclidean_geometry), and as a consequence, the motions along different axes are independent of each other.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/CCS.png?raw=1" width=350 alt="Cartesian coordinate system"/></center><figcaption><center><i>Figure. A point in three-dimensional Euclidean space described in a Cartesian coordinate system.</i></center></figcaption></figure>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Many coordinate systems at once

    The concept of reference frame is central to understanding human motion, and not only in mechanics. Do we see, plan and control the movement of our hand with respect to reference frames within our body, or in the environment we move in? Or a combination of both? Questions like these are at the heart of motor control research.

    The figure below, although drawn for a robot, illustrates well that we may have to deal with several coordinate systems at once.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/coordinatesystems.png?raw=1" width=450 alt="multiple coordinate systems of a robot"/></center><figcaption><center><i>Figure. Multiple coordinate systems for use in robots (figure from Corke (2017)).</i></center></figcaption></figure>

    In three-dimensional motion analysis we routinely use several frames of reference, and call them global, laboratory, local, anatomical or technical, depending on what they are attached to. In a standard gait analysis there is one global, or laboratory, coordinate system fixed to the floor, and a coordinate system for each segment of the body, so that the motion of a segment can be described in relation to the anatomical axes of another one. The knee angle, for example, is the orientation of the shank's coordinate system relative to the thigh's.

    The anatomical coordinate system of a segment is defined by anatomical landmarks. Other markers, placed where they are easier to see or less disturbed by the skin, may be used to improve the motion analysis, and they define a technical coordinate system for the segment. We come back to the difference at the end of this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standardizations in movement analysis

    Two laboratories that define the thigh and shank coordinate systems differently will report different knee angles for the same knee. To make results comparable, several standardizations have been proposed for how to define the frames of reference of the main segments and joints of the human body. The International Society of Biomechanics keeps a [page listing standardization proposals](https://isbweb.org/activities/standards) by its standardization committee and subcommittees:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """<iframe src="https://isbweb.org/activities/standards"
        width="100%" height="400" style="border:none"></iframe>"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another initiative for the standardization of reference frames comes from the [Virtual Animation of the Kinematics of the Human for Industrial, Educational and Research Purposes (VAKHUM)](https://github.com/BMClab/BMC/blob/master/courses/refs/VAKHUM.pdf) project. We will use its definition of the femur's coordinate system later in this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Determination of a coordinate system

    From [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra), a set of linearly independent unit vectors that can represent any vector via [linear combination](https://en.wikipedia.org/wiki/Linear_combination) is called a <a href="https://en.wikipedia.org/wiki/Basis_(linear_algebra)">basis</a>. When its vectors are also mutually orthogonal, as in the Cartesian coordinate system, it is an orthonormal basis. The figure below shows a point, its position vector, and the versors (unit vectors) of the basis of a Cartesian coordinate system. See [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py) for a description of vectors.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/vector3Dijk.png?raw=1" width=350 alt="versors of the Cartesian basis"/></center><figcaption><center><i>Figure. Representation of a point $\mathbf{P}$ and its position vector $\overrightarrow{\mathbf{r}}$ in a Cartesian coordinate system. The versors $\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}$ form a basis for this coordinate system and are usually drawn in the colour sequence RGB (red, green, blue) for easier visualization.</i></center></figcaption></figure>

    In the Cartesian coordinate system itself, the versors of this basis have the coordinates

    $$
    \hat{\mathbf{i}} = \begin{bmatrix}1\\0\\0 \end{bmatrix}, \quad \hat{\mathbf{j}} = \begin{bmatrix}0\\1\\0 \end{bmatrix}, \quad \hat{\mathbf{k}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
    $$

    and, in the notation of the figure, the position vector $\overrightarrow{\mathbf{r}}$ of the point $\mathbf{P}$ is

    $$
    \overrightarrow{\mathbf{r}} = x\hat{\mathbf{i}} + y\hat{\mathbf{j}} + z\hat{\mathbf{k}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Definition of a basis

    The mathematical problem of determining a coordinate system is to find a basis and an origin for it. A basis on its own is only a set of directions, with no origin. There are different methods to calculate a basis from a set of points; one can use, for example, the cross product or the scalar product.

    The positions of the points used to construct a coordinate system have, by definition, to be specified in relation to an already existing coordinate system. In motion analysis, this is the coordinate system of the motion capture system, established in the calibration phase. In that phase, the positions of markers placed on an object with perpendicular axes and known distances between the markers are captured and used as the reference (laboratory) coordinate system.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using the cross product

    Here is a common method in motion analysis. Given the coordinates of three noncollinear points in 3D space (points that do not all lie on the same line), $\overrightarrow{\mathbf{m}}_1, \overrightarrow{\mathbf{m}}_2, \overrightarrow{\mathbf{m}}_3$, which would be the positions of markers captured in a motion analysis session, a basis is found in four steps:

    1. First axis, $\overrightarrow{\mathbf{v}}_1$: the vector $\overrightarrow{\mathbf{m}}_2-\overrightarrow{\mathbf{m}}_1$ (or any other difference between two of the points).
    2. Second axis, $\overrightarrow{\mathbf{v}}_2$: the cross product between $\overrightarrow{\mathbf{v}}_1$ and $\overrightarrow{\mathbf{m}}_3-\overrightarrow{\mathbf{m}}_1$ (or $\overrightarrow{\mathbf{m}}_3-\overrightarrow{\mathbf{m}}_2$).
    3. Third axis, $\overrightarrow{\mathbf{v}}_3$: the cross product between $\overrightarrow{\mathbf{v}}_1$ and $\overrightarrow{\mathbf{v}}_2$.
    4. Make every axis a versor by dividing it by its norm.

    The second axis is perpendicular to the plane of the three points, and the third lies in that plane, perpendicular to the first.

    For example, given the positions $\overrightarrow{\mathbf{m}}_1 = [1,2,5],\; \overrightarrow{\mathbf{m}}_2 = [2,3,3],\; \overrightarrow{\mathbf{m}}_3 = [4,0,2]$, a basis can be found with:
    """)
    return


@app.cell
def _(np):
    m1 = np.array([1, 2, 5])
    m2 = np.array([2, 3, 3])
    m3 = np.array([4, 0, 2])

    v1 = m2 - m1  # first axis
    v2 = np.cross(v1, m3 - m1)  # second axis
    v3 = np.cross(v1, v2)  # third axis

    # Vector normalization
    e1 = v1 / np.linalg.norm(v1)
    e2 = v2 / np.linalg.norm(v2)
    e3 = v3 / np.linalg.norm(v3)

    print("Versors:", "\ne1 =", e1, "\ne2 =", e2, "\ne3 =", e3)
    print(
        "\nNorm of each versor:",
        "\n||e1|| =", np.linalg.norm(e1),
        "\n||e2|| =", np.linalg.norm(e2),
        "\n||e3|| =", np.linalg.norm(e3),
    )
    print(
        "\nTest of orthogonality (cross product between versors):",
        "\ne1 x e2:", np.linalg.norm(np.cross(e1, e2)),
        "\ne1 x e3:", np.linalg.norm(np.cross(e1, e3)),
        "\ne2 x e3:", np.linalg.norm(np.cross(e2, e3)),
    )
    return e1, e2, e3, m1, m2, m3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Due to [rounding errors](https://en.wikipedia.org/wiki/Round-off_error) and to the [limited precision of floating-point arithmetic](https://docs.python.org/3/tutorial/floatingpoint.html), some of the results that should be exactly 1 are not. We could round them if desired:
    """)
    return


@app.cell
def _(e1, e2, e3, np):
    print(
        "Test of orthogonality (cross product between versors):",
        "\ne1 x e2:", np.linalg.norm(np.cross(e1, e2)).round(8),
        "\ne1 x e3:", np.linalg.norm(np.cross(e1, e3)).round(8),
        "\ne2 x e3:", np.linalg.norm(np.cross(e2, e3)).round(8),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or format the text that represents the result:
    """)
    return


@app.cell
def _(e1, e2, e3, np):
    print(
        "Test of orthogonality (cross product between versors):",
        f"\ne1 x e2: {np.linalg.norm(np.cross(e1, e2)):g}",
        f"\ne1 x e3: {np.linalg.norm(np.cross(e1, e3)):g}",
        f"\ne2 x e3: {np.linalg.norm(np.cross(e2, e3)):g}",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here `f"{x:g}"` formats the number as a string in the "general" format, which drops trailing zeros and shows six significant digits by default (see the [format specification](https://docs.python.org/3/library/string.html#format-specification-mini-language) and a [Python f-strings number formatting cheat sheet](https://cheatography.com/brianallan/cheat-sheets/python-f-strings-number-formatting/)). For a permanent change, NumPy's [`np.set_printoptions`](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html) sets the precision with which arrays are printed; it affects NumPy arrays only, not plain Python numbers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** Before computing anything, predict the basis that the four steps give for markers placed on the laboratory's axes: $\overrightarrow{\mathbf{m}}_1 = [0,0,0],\; \overrightarrow{\mathbf{m}}_2 = [1,0,0],\; \overrightarrow{\mathbf{m}}_3 = [0,1,0]$. Which laboratory axis does each versor coincide with? Then check your answer with the code above.

    /// details | Solution to Exercise 4
    Following the steps:

    - $\overrightarrow{\mathbf{v}}_1 = \overrightarrow{\mathbf{m}}_2 - \overrightarrow{\mathbf{m}}_1 = [1, 0, 0]$, so $\hat{\mathbf{e}}_1 = \hat{\mathbf{i}}$.
    - $\overrightarrow{\mathbf{v}}_2 = \overrightarrow{\mathbf{v}}_1 \times (\overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_1) = \hat{\mathbf{i}} \times \hat{\mathbf{j}} = \hat{\mathbf{k}}$, so $\hat{\mathbf{e}}_2 = \hat{\mathbf{k}}$.
    - $\overrightarrow{\mathbf{v}}_3 = \overrightarrow{\mathbf{v}}_1 \times \overrightarrow{\mathbf{v}}_2 = \hat{\mathbf{i}} \times \hat{\mathbf{k}} = -\hat{\mathbf{j}}$, so $\hat{\mathbf{e}}_3 = -\hat{\mathbf{j}}$.

    The basis is not $\hat{\mathbf{i}}, \hat{\mathbf{j}}, \hat{\mathbf{k}}$, as one might guess, but $\hat{\mathbf{i}}, \hat{\mathbf{k}}, -\hat{\mathbf{j}}$: the recipe always makes the second versor perpendicular to the plane of the markers and puts the third one in that plane. It is still right-handed, since $\hat{\mathbf{i}} \times \hat{\mathbf{k}} = -\hat{\mathbf{j}}$.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is that a good test of orthogonality?

    The test above relies on a true fact: for two versors, $\Vert\hat{\mathbf{a}} \times \hat{\mathbf{b}}\Vert = \sin\theta$, which equals 1 when they are perpendicular. But a test is only as good as its ability to fail when something is wrong.

    **Before you run the next cell**, predict: if two versors were 89° apart instead of 90°, what would the norm of their cross product be? And their dot product?

    The function below computes the angle between two vectors from the dot product (see [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py)). We will use it again later.
    """)
    return


@app.function
def angle_between(u, w):
    """Angle in degrees between the vectors `u` and `w` (along the last axis)."""
    import numpy as np

    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    cos = np.sum(u * w, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


@app.cell
def _(np):
    for _deg in [90, 89, 85]:
        _theta = np.radians(_deg)
        _a = np.array([1, 0, 0])
        _b = np.array([np.cos(_theta), np.sin(_theta), 0])
        print(
            f"{angle_between(_a, _b):4.0f} degrees apart:"
            f"  |a x b| = {np.linalg.norm(np.cross(_a, _b)):.5f},  a . b = {np.dot(_a, _b):.5f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An error of one degree changes the norm of the cross product in the fourth decimal place, and an error of five degrees only in the third. The sine is flat near 90°, so the cross product test is nearly blind exactly where we use it. The cosine is steep there: the dot product moves from 0 to 0.017 for one degree. **Test orthogonality with the dot product**, which should be zero.

    There is a second property that the cross product test cannot check at all: handedness. A basis whose third versor points the wrong way passes every test of length and orthogonality. For a right-handed basis, $\hat{\mathbf{e}}_1 \times \hat{\mathbf{e}}_2 = \hat{\mathbf{e}}_3$; equivalently, the determinant of the matrix with the versors as its rows is $+1$ (and $-1$ for a left-handed basis):
    """)
    return


@app.cell
def _(e1, e2, e3, np):
    print(
        "Test of orthogonality (dot product between versors):",
        f"\ne1 . e2: {np.dot(e1, e2):.1e}",
        f"\ne1 . e3: {np.dot(e1, e3):.1e}",
        f"\ne2 . e3: {np.dot(e2, e3):.1e}",
    )
    print("\nTest of handedness:")
    print("e1 x e2 - e3 =", np.cross(e1, e2) - e3)
    print(f"det([e1, e2, e3]) = {np.linalg.det(np.vstack((e1, e2, e3))):g}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dot products are zero to within $10^{-16}$ and the basis is right-handed, as the construction guarantees: $\overrightarrow{\mathbf{v}}_3 = \overrightarrow{\mathbf{v}}_1 \times \overrightarrow{\mathbf{v}}_2$ is, by definition, the third axis of a right-handed set.

    **Exercise 5.** A colleague hands you three versors, $\hat{\mathbf{u}}_1 = [0.6, 0.8, 0]$, $\hat{\mathbf{u}}_2 = [-0.8, 0.6, 0]$ and $\hat{\mathbf{u}}_3 = [0, 0, -1]$, as the basis of a segment. Is it orthonormal? Is it right-handed? If it is not, what is the smallest change that makes it right-handed?
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _u1 = np.array([0.6, 0.8, 0])
    _u2 = np.array([-0.8, 0.6, 0])
    _u3 = np.array([0, 0, -1])
    _dots = [round(float(np.dot(_p, _q)), 12) + 0.0 for _p, _q in [(_u1, _u2), (_u1, _u3), (_u2, _u3)]]
    _norms = [np.linalg.norm(_u) for _u in (_u1, _u2, _u3)]
    _det = np.linalg.det(np.vstack((_u1, _u2, _u3)))
    _cross = np.cross(_u1, _u2) + 0.0
    mo.accordion(
        {
            "Solution to Exercise 5": mo.md(
                rf"""
    The dot products are {", ".join(f"{_d:g}" for _d in _dots)} and the norms are {", ".join(f"{_n:g}" for _n in _norms)}: the basis is **orthonormal**.

    But $\hat{{\mathbf{{u}}}}_1 \times \hat{{\mathbf{{u}}}}_2 = {_cross.tolist()}$, which is $-\hat{{\mathbf{{u}}}}_3$, and the determinant of the matrix with the versors as rows is **{_det:g}**: the basis is **left-handed**.

    Reversing any one of the versors makes it right-handed. The natural choice here is $\hat{{\mathbf{{u}}}}_3 = [0, 0, 1]$, the cross product of the other two. Swapping two of the versors would also do it, but it changes which axis is which, which is usually worse.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coordinate system: origin and basis

    To define a coordinate system from the basis we also need an origin. In principle any point would do, but if the coordinate system should follow anatomical conventions, with its origin at a joint centre, for example, then both the basis and the origin must be calculated according to the standards discussed before.

    If the coordinate system is a technical one and not anatomically based, a common procedure in motion analysis is to place its origin at the centroid (average position) of the markers. Averaging the positions of several markers reduces the effect of noise, for example from soft tissue artifacts, on the location of the origin.

    For the markers in the example above, the origin of the coordinate system is:
    """)
    return


@app.cell
def _(m1, m2, m3, np):
    origin = np.mean((m1, m2, m3), axis=0)

    print("Origin:", origin)
    return (origin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visualization of the coordinate system

    Let's plot the coordinate system with the function `CCS`, from the file `CCS.py` in the `functions` folder of this repository. The first cell adds that folder to Python's search path.
    """)
    return


@app.function
def repo_file(path, *, base_url="https://raw.githubusercontent.com/BMClab/BMC/master/"):
    """Find a repository file beside this notebook, or fetch a temporary copy.

    Local files take precedence. Only file acquisition falls back to GitHub;
    parsing errors remain visible to the calling notebook. urllib3 selects its
    browser transport automatically without platform-specific imports.
    """
    import hashlib
    import tempfile
    from pathlib import Path, PurePosixPath
    from urllib.parse import quote

    import marimo as mo
    import urllib3

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise ValueError("Use a repository-relative path such as data/walk.trc")
    location = mo.notebook_dir()
    if location is not None:
        directory = Path(location).resolve()
        for base in (directory, *directory.parents):
            local = base.joinpath(*relative.parts)
            if local.is_file():
                return local
    url = base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")
    key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]
    cached = (Path(tempfile.gettempdir()) / "bmc-notebooks-v2" / key).joinpath(*relative.parts)
    if not cached.is_file():
        response = urllib3.request("GET", url, timeout=30.0, retries=2)
        if response.status != 200:
            raise OSError(f"Cannot load {path}: HTTP {response.status} from {url}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as download:
            temporary = Path(download.name)
            download.write(response.data)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    return cached


@app.cell
def _():
    import sys

    sys.path.insert(0, str(repo_file("functions/CCS.py").parent))  # add to pythonpath
    from CCS import CCS

    return (CCS,)


@app.cell
def _(e1, e2, e3, m1, m2, m3, np):
    markers = np.vstack((m1, m2, m3))
    basis = np.vstack((e1, e2, e3))
    return basis, markers


@app.cell
def _(CCS, basis, markers, origin):
    _ccs = CCS(xyz=[], Oijk=origin, ijk=basis, point=markers, vector=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The black arrows are the axes of the laboratory; the yellow arrows are the position vectors of the three markers; and the red, green and blue arrows, which `CCS` labels i, j and k, are the versors $\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{e}}_3$, drawn from the new origin. The green versor is perpendicular to the triangle formed by the three markers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gram–Schmidt process

    Another classical procedure, this one built on the scalar product, is the [Gram–Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process). It orthonormalizes a set of vectors, turning them into mutually orthogonal versors, and it works for any number of vectors. See [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py) for another demonstration.

    Given three vectors in 3D space, $\overrightarrow{\mathbf{a}}, \overrightarrow{\mathbf{b}}, \overrightarrow{\mathbf{c}}$, a basis $\{\hat{e}_a, \hat{e}_b, \hat{e}_c\}$ is found as follows. The first versor is in the direction of $\overrightarrow{\mathbf{a}}$ (or of any of the other vectors):

    $$
    \hat{e}_a = \frac{\overrightarrow{\mathbf{a}}}{||\overrightarrow{\mathbf{a}}||}
    $$

    For the second versor, orthogonal to $\hat{e}_a$, write $\overrightarrow{\mathbf{b}}$ as a part along $\hat{e}_a$ plus a part perpendicular to it:

    $$
    \overrightarrow{\mathbf{b}} = \overrightarrow{\mathbf{b}}^\| + \overrightarrow{\mathbf{b}}^\bot
    $$

    The parallel part is what the dot product gives us, so

    $$
    \overrightarrow{\mathbf{b}}^\bot = \overrightarrow{\mathbf{b}} - \overrightarrow{\mathbf{b}}^\| = \overrightarrow{\mathbf{b}} - (\overrightarrow{\mathbf{b}} \cdot \hat{e}_a ) \hat{e}_a
    $$

    and finally

    $$
    \hat{e}_b = \frac{\overrightarrow{\mathbf{b}}^\bot}{||\overrightarrow{\mathbf{b}}^\bot||}
    $$

    The third versor, orthogonal to both $\hat{e}_a$ and $\hat{e}_b$, is found the same way, by removing from $\overrightarrow{\mathbf{c}}$ its parts along the two directions already fixed:

    $$
    \overrightarrow{\mathbf{c}} = \overrightarrow{\mathbf{c}}^\| + \overrightarrow{\mathbf{c}}^\bot,
    \qquad
    \overrightarrow{\mathbf{c}}^\| = (\overrightarrow{\mathbf{c}} \cdot \hat{e}_a ) \hat{e}_a + (\overrightarrow{\mathbf{c}} \cdot \hat{e}_b ) \hat{e}_b
    $$

    $$
    \overrightarrow{\mathbf{c}}^\bot = \overrightarrow{\mathbf{c}} - \overrightarrow{\mathbf{c}}^\|,
    \qquad
    \hat{e}_c = \frac{\overrightarrow{\mathbf{c}}^\bot}{||\overrightarrow{\mathbf{c}}^\bot||}
    $$

    Here is an animation of the process at work:

    <figure><center><img src="https://upload.wikimedia.org/wikipedia/commons/e/ee/Gram-Schmidt_orthonormalization_process.gif" width=450 alt="animation of the Gram-Schmidt process"/></center><figcaption><center><i>Figure. The Gram–Schmidt process executed on three vectors of a basis for R<sup>3</sup> (<a href="https://commons.wikimedia.org/w/index.php?curid=24396471">image from Wikipedia</a>).</i></center></figcaption></figure>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's implement the Gram–Schmidt process in Python, using the positions we saw before, $\overrightarrow{\mathbf{m}}_1 = [1,2,5],\; \overrightarrow{\mathbf{m}}_2 = [2,3,3],\; \overrightarrow{\mathbf{m}}_3 = [4,0,2]$, as the three vectors. The first versor is:
    """)
    return


@app.cell
def _(m1, np):
    ea = m1 / np.linalg.norm(m1)

    print(ea)
    return (ea,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The second versor:
    """)
    return


@app.cell
def _(ea, m2, np):
    eb = m2 - np.dot(m2, ea) * ea
    eb = eb / np.linalg.norm(eb)

    print(eb)
    return (eb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And the third versor:
    """)
    return


@app.cell
def _(ea, eb, m3, np):
    ec = m3 - np.dot(m3, ea) * ea - np.dot(m3, eb) * eb
    ec = ec / np.linalg.norm(ec)

    print(ec)
    return (ec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check the orthonormality of these versors:
    """)
    return


@app.cell
def _(ea, eb, ec, np):
    print("Versors:", "\nea =", ea, "\neb =", eb, "\nec =", ec)
    print(
        "\nNorm of each versor:",
        "\n||ea|| =", np.linalg.norm(ea),
        "\n||eb|| =", np.linalg.norm(eb),
        "\n||ec|| =", np.linalg.norm(ec),
    )
    print(
        "\nTest of orthogonality (scalar product between versors):",
        "\nea . eb:", np.dot(ea, eb),
        "\neb . ec:", np.dot(eb, ec),
        "\nec . ea:", np.dot(ec, ea),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A perfectly good orthonormal basis. But is it a good coordinate system for the segment that carries the three markers?

    ### A basis that belongs to the laboratory

    Let's collect both methods in functions, so we can apply them to other markers:
    """)
    return


@app.function
def gram_schmidt(a, b, c):
    """Orthonormal basis from the vectors `a`, `b`, `c` by the Gram–Schmidt process.

    The first versor is along `a`, the second is the part of `b` perpendicular
    to the first, and the third is the part of `c` perpendicular to both.
    """
    import numpy as np

    ea = a / np.linalg.norm(a)
    eb = b - np.dot(b, ea) * ea
    eb = eb / np.linalg.norm(eb)
    ec = c - np.dot(c, ea) * ea - np.dot(c, eb) * eb
    ec = ec / np.linalg.norm(ec)
    return ea, eb, ec


@app.function
def cross_product_basis(m1, m2, m3):
    """Orthonormal basis from the positions of three noncollinear markers.

    The first versor is along m2 - m1, the second is perpendicular to the plane
    of the markers, and the third completes a right-handed basis.
    """
    import numpy as np

    v1 = m2 - m1
    v2 = np.cross(v1, m3 - m1)
    v3 = np.cross(v1, v2)
    return tuple(v / np.linalg.norm(v) for v in (v1, v2, v3))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now imagine that the motion capture system had been calibrated with the origin of the laboratory one unit further along $-X$. The markers have not moved; only the numbers that describe them have, all of them shifted by $[1, 0, 0]$.

    **Before you run the next cell**, predict: does the basis from the Gram–Schmidt process change? Does the basis from the cross product?
    """)
    return


@app.cell
def _(m1, m2, m3, np):
    _shift = np.array([1, 0, 0])
    _gs = gram_schmidt(m1, m2, m3)
    _gs_shifted = gram_schmidt(m1 + _shift, m2 + _shift, m3 + _shift)
    _cp = cross_product_basis(m1, m2, m3)
    _cp_shifted = cross_product_basis(m1 + _shift, m2 + _shift, m3 + _shift)

    print("How much each versor turned when the laboratory origin moved [degrees]:")
    print("  Gram-Schmidt on the positions: ", [round(float(angle_between(u, w)), 2) for u, w in zip(_gs, _gs_shifted)])
    print("  cross product of differences:  ", [round(float(angle_between(u, w)), 2) for u, w in zip(_cp, _cp_shifted)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The markers did not move, and yet the Gram–Schmidt basis turned by 7 to 10 degrees. Shift the origin further and it turns further. The cross-product basis did not change at all.

    The difference is in what each method was fed. The Gram–Schmidt process above was applied to the *position vectors* of the markers, the arrows from the laboratory's origin to each marker, and those arrows depend on where that origin happens to be. The cross-product recipe uses only *differences* between markers, $\overrightarrow{\mathbf{m}}_2 - \overrightarrow{\mathbf{m}}_1$ and $\overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_1$, which are the same wherever the origin is. A coordinate system for a body segment must depend on the segment alone, so it must be built from differences.

    The Gram–Schmidt process is not to blame; it was simply given the wrong vectors. Fed with differences, it works. With three markers there are only two independent differences, so the process gives two versors, both in the plane of the markers, and the cross product supplies the third:
    """)
    return


@app.cell
def _(e1, e2, e3, m1, m2, m3, np):
    _a, _b = m2 - m1, m3 - m1  # differences between markers
    _fa = _a / np.linalg.norm(_a)
    _fb = _b - np.dot(_b, _fa) * _fa
    _fb = _fb / np.linalg.norm(_fb)
    _fc = np.cross(_fa, _fb)

    print("Gram-Schmidt on differences:", "\nfa =", _fa, "\nfb =", _fb, "\nfc =", _fc)
    print("\nCompared with the cross-product basis:")
    print("fa equals  e1:", np.allclose(_fa, e1))
    print("fb equals -e3:", np.allclose(_fb, -e3))
    print("fc equals  e2:", np.allclose(_fc, e2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two methods give the same three axes, with different names and one of them pointing the opposite way. Neither is wrong. Which axis is called the first one, and which way each one points, are *conventions*, and conventions are exactly what an anatomical coordinate system has to fix.

    **Challenge 1.** In a new cell, plot the Gram–Schmidt basis built from the differences with `CCS`, using the same origin and markers as before, and compare the figure with the one for the cross-product basis. Which arrows coincide? Which one is reversed?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From markers to anatomy

    ### An anatomical frame: the femur

    The VAKHUM project defines the anatomical coordinate system of the femur from three anatomical landmarks: the centre of the femoral head ($fh$), the lateral epicondyle ($le$) and the medial epicondyle ($me$). The laboratory axes follow the same convention as the segment axes: $x$ points anteriorly, $y$ upwards and $z$ to the right. In VAKHUM's words, paraphrased:

    - The origin, $O_t$, is the midpoint between $le$ and $me$.
    - The $y_t$ axis is along the line through $O_t$ and $fh$, positive upwards.
    - The $z_t$ axis lies in the plane defined by $fh$, $le$ and $me$, is perpendicular to $y_t$, and points to the right.
    - The $x_t$ axis is perpendicular to both, and points anteriorly.

    Each sentence translates into one of the operations of this notebook:

    $$
    O_t = \frac{le + me}{2},
    \qquad
    \hat{y}_t = \frac{fh - O_t}{\Vert fh - O_t \Vert}
    $$

    The plane of the three landmarks contains both $\hat{y}_t$ and the line between the epicondyles. Removing from $le - me$ its component along $\hat{y}_t$, which is one step of the Gram–Schmidt process, leaves a vector in that plane, perpendicular to $\hat{y}_t$. For a right femur, $le - me$ points to the right, as the convention requires:

    $$
    \hat{z}_t = \frac{(le - me) - \left[(le - me)\cdot\hat{y}_t\right]\hat{y}_t}{\Vert (le - me) - \left[(le - me)\cdot\hat{y}_t\right]\hat{y}_t \Vert}
    $$

    And for $x_t$, $y_t$, $z_t$ to be right-handed, $\hat{x}_t = \hat{y}_t \times \hat{z}_t$.

    Here are hypothetical positions of the three landmarks of the right femur of a person standing upright, facing $+x$, in metres.

    **Before you run the next cell**, predict: which laboratory axis should each axis of the femur be close to?
    """)
    return


@app.cell
def _(np):
    fh = np.array([0.02, 0.92, 0.09])  # centre of the femoral head [m]
    le = np.array([0.03, 0.50, 0.14])  # lateral epicondyle [m]
    me = np.array([0.02, 0.49, 0.05])  # medial epicondyle [m]

    Ot = (le + me) / 2  # origin, midpoint of the epicondyles

    yt = (fh - Ot) / np.linalg.norm(fh - Ot)  # long axis, upwards

    zt = (le - me) - np.dot(le - me, yt) * yt  # epicondylar line, made perpendicular to yt
    zt = zt / np.linalg.norm(zt)

    xt = np.cross(yt, zt)  # anterior

    print("Ot =", Ot)
    print("xt =", xt.round(4), "\nyt =", yt.round(4), "\nzt =", zt.round(4))
    return Ot, fh, le, me, xt, yt, zt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before interpreting anything, test the basis:
    """)
    return


@app.cell
def _(le, me, np, xt, yt, zt):
    print(f"xt . yt = {np.dot(xt, yt):.1e},  yt . zt = {np.dot(yt, zt):.1e},  zt . xt = {np.dot(zt, xt):.1e}")
    print(f"det([xt, yt, zt]) = {np.linalg.det(np.vstack((xt, yt, zt))):g}")
    print("\nAngle between each femur axis and the laboratory axis of the same name:")
    for _name, _axis, _lab in [("x", xt, [1, 0, 0]), ("y", yt, [0, 1, 0]), ("z", zt, [0, 0, 1])]:
        print(f"  {_name}: {angle_between(_axis, _lab):.1f} degrees")
    print(f"\nAngle between the epicondylar line and yt: {angle_between(le - me, yt):.1f} degrees")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An orthonormal, right-handed basis. Its long axis, $y_t$, is within one degree of the vertical, as it should be for a person standing upright. But $x_t$ and $z_t$ are both turned by about 6° about the vertical: the lateral epicondyle is 1 cm anterior to the medial one, so the knee points slightly inwards. That is about 6° of internal rotation of the femur, a number that was hidden in the landmark coordinates until the coordinate system brought it out.

    Notice also the order of the definitions. The line between the epicondyles is 84.4° from the long axis, not 90°, so the two cannot both be axes of an orthogonal basis. The convention settles which one gives way: $y_t$ is defined first and kept exactly, and $z_t$ gets only what is left of the epicondylar line once its component along $y_t$ is removed.

    **Guiding questions 2.**

    1. For a left femur, $le - me$ points to the left. What would you change in the code so that $z_t$ still points to the right, as the convention requires? What would happen to $x_t$ if you forgot?
    2. The convention defines $y_t$ first, from the two landmarks that are farthest apart. Why is that a sensible choice when every landmark is located with some error?
    3. Why is the origin at the midpoint of the epicondyles, and not at the centroid of the three landmarks, as for the technical coordinate system earlier?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using the femur's coordinate system

    A coordinate system earns its keep when we use it to describe points. Any point P can be described by its **local coordinates**: the projections of $\overrightarrow{\mathbf{r}}_P - O_t$ on the three versors, found with the dot product. From the local coordinates we get back to the laboratory by adding up the versors:

    $$
    x^* = (\overrightarrow{\mathbf{r}}_P - O_t)\cdot\hat{x}_t, \quad y^* = (\overrightarrow{\mathbf{r}}_P - O_t)\cdot\hat{y}_t, \quad z^* = (\overrightarrow{\mathbf{r}}_P - O_t)\cdot\hat{z}_t
    $$

    $$
    \overrightarrow{\mathbf{r}}_P = O_t + x^*\,\hat{x}_t + y^*\,\hat{y}_t + z^*\,\hat{z}_t
    $$

    Take a marker on the side of the thigh, 15 cm above the origin of the femur along its long axis and 6 cm lateral to it: local coordinates $(0,\, 0.15,\, 0.06)$ m, since lateral is $+z_t$ for the right femur. Where is it in the laboratory? Then flex the hip by 30°, turning the three landmarks about the centre of the femoral head (the turn is done with a rotation matrix; see [Rigid-body transformations in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py)), and rebuild the femur's coordinate system from the new landmarks. The function below collects the recipe of the previous section.

    **Before you run the next cell**, predict: after the hip flexes, are the marker's local coordinates still $(0,\, 0.15,\, 0.06)$? Does the marker move more or less than the knee?
    """)
    return


@app.function
def femur_frame(fh, le, me):
    """Origin and basis (versors as rows) of the right femur, VAKHUM convention.

    `fh`, `le` and `me` are the positions of the centre of the femoral head and
    of the lateral and medial epicondyles, in a laboratory with x anterior,
    y upwards and z to the right.
    """
    import numpy as np

    origin = (le + me) / 2
    yt = (fh - origin) / np.linalg.norm(fh - origin)
    zt = (le - me) - np.dot(le - me, yt) * yt
    zt = zt / np.linalg.norm(zt)
    xt = np.cross(yt, zt)
    return origin, np.vstack((xt, yt, zt))


@app.cell
def _(Ot, fh, le, me, np, xt, yt, zt):
    _local = np.array([0.0, 0.15, 0.06])  # marker on the thigh, in the femur's coordinates [m]
    _basis = np.vstack((xt, yt, zt))
    _p_lab = Ot + _local @ _basis  # local to laboratory

    print("Marker in the laboratory, standing:   ", _p_lab.round(4), "m")
    print("Back to local coordinates:            ", ((_p_lab - Ot) @ _basis.T).round(4) + 0.0, "m")

    # Flex the hip by 30 degrees: turn the landmarks about the femoral head, about the z axis
    _theta = np.radians(30)
    _Rz = np.array([[np.cos(_theta), -np.sin(_theta), 0], [np.sin(_theta), np.cos(_theta), 0], [0, 0, 1]])

    def _turn(p):
        return fh + _Rz @ (p - fh)

    _Ot_flex, _basis_flex = femur_frame(_turn(fh), _turn(le), _turn(me))
    _p_flex = _Ot_flex + _local @ _basis_flex  # the same local coordinates, in the new frame

    print("Marker in the laboratory, hip flexed: ", _p_flex.round(4), "m")
    print(f"Difference from turning the marker itself: {np.abs(_p_flex - _turn(_p_lab)).max():.1e} m")
    print(
        f"The marker moved {100 * np.linalg.norm(_p_flex - _p_lab):.1f} cm,"
        f" the origin of the femur (the knee) {100 * np.linalg.norm(_Ot_flex - Ot):.1f} cm"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The local coordinates do not change. Rebuilt from the flexed landmarks, the femur's coordinate system puts the marker exactly where turning the marker itself would, to within $10^{-16}$ m: the frame carries every point of the segment with it, which is the whole reason for attaching one to the segment. The marker moved 14.2 cm and the knee 22.0 cm, because points nearer the hip, the centre of the rotation, travel along shorter arcs.

    This is also the principle behind the technical and anatomical coordinate systems of the next section: local coordinates measured once, in a calibration trial, give the position of a point at every later instant.

    **Exercise 6.** The centre of the femoral head cannot be palpated; it has to be estimated, and every estimate carries an error. Suppose the estimate is 1 cm too medial (for the right femur, 1 cm towards $-z$). Recalculate the femur's coordinate system with `femur_frame`. Which of its axes change, and by how much? Which joint angle would this error disturb the most?
    """)
    return


@app.cell(hide_code=True)
def _(fh, le, me, mo, np):
    _, _basis = femur_frame(fh, le, me)
    _, _basis_err = femur_frame(fh + np.array([0, 0, -0.01]), le, me)
    _angles = angle_between(_basis_err, _basis)
    _tilt = np.degrees(np.arctan(0.01 / np.linalg.norm(fh - (le + me) / 2)))
    mo.accordion(
        {
            "Solution to Exercise 6": mo.md(
                rf"""
    With the femoral head 1 cm too medial, the axes turn by **{_angles[0]:.2f}°** ($x_t$), **{_angles[1]:.2f}°** ($y_t$) and **{_angles[2]:.2f}°** ($z_t$).

    The long axis tilts towards the midline by about the error divided by the length of the femur, $\arctan(0.01/0.425) = {_tilt:.2f}°$, and $z_t$, which is kept perpendicular to it, tilts with it. Both turn about $x_t$, which barely moves. The error is a rotation in the frontal plane of the femur, so it disturbs the abduction–adduction angles of the hip and the knee the most, and their flexion angles hardly at all.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Technical and anatomical coordinate systems

    In practice, anatomical landmarks make poor markers. The centre of the femoral head is inside the body, and its position has to be estimated. The epicondyles are under skin that slides over them as the knee moves, and a marker on the medial side of the knee gets knocked off by the other leg.

    The usual solution (Cappozzo et al., 1995) separates the two jobs. Markers placed where they are easy to see and stay put define a technical coordinate system, which is tracked throughout the movement. In a static calibration trial, the anatomical landmarks are located, by palpation or with a pointer, and their positions are expressed in the technical coordinate system. As long as the segment behaves as a rigid body, those positions do not change, so the anatomical coordinate system can be rebuilt from the technical one at every instant of the movement.

    The weak point of the method, and of all skin-marker methods, is that the markers move relative to the bone, by as much as a few centimetres in some places: the soft tissue artifact. Three practical defences follow from this notebook and the ones it links to:

    - Place the origin of a technical coordinate system at the centroid of its markers, so that the errors of individual markers partly average out.
    - Spread the markers far apart compared with the size of their errors (Cappozzo et al., 1997). How far, and what goes wrong when the markers are nearly in a line, is worked out in [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py).
    - Use more than three markers and find the best-fitting basis by least squares, usually with the singular value decomposition (see [SVD algorithm](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/svd_algorithm.py)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your three descriptions from Challenge 0. Which frame of reference made the motion of your hand simplest? Which of the three frames are inertial, approximately?
    2. Two observers in different inertial frames measure the same runner. List what they agree on and what they disagree on: position, velocity, acceleration, force, kinetic energy.
    3. You film a sprinter with a camera on a trolley that runs alongside on a rail. What do you need to know about the trolley's motion before you can use the video to estimate forces?
    4. A colleague builds a thigh coordinate system by applying the Gram–Schmidt process to the raw marker positions, and the results look reasonable. Describe a test that would reveal the problem.
    5. A coordinate system needs a basis and an origin. Which of the two does a joint angle depend on? And the position of a joint centre?
    6. Two laboratories report knee angles that differ by 5° for the same person walking. Name two reasons, related to this notebook, why that could happen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Right now, how fast are you moving? In your answer, consider your motion in relation to the Earth and in relation to the Sun.

    2. Given the points in 3D space m1 = [2, 2, 0], m2 = [0, 1, 1], m3 = [1, 2, 0], find an orthonormal basis.

    3. Determine whether the following points form a basis in 3D space: m1 = [2, 2, 0], m2 = [1, 1, 1], m3 = [1, 1, 0].

    4. Derive expressions for the three axes of the pelvic basis according to the convention of the [Virtual Animation of the Kinematics of the Human for Industrial, Educational and Research Purposes (VAKHUM)](https://github.com/BMClab/BMC/blob/master/courses/refs/VAKHUM.pdf) project (use RASIS, LASIS, RPSIS and LPSIS as the names of the pelvic anatomical landmarks, and give the expression for each axis).

    5. Determine the basis for the pelvis following the convention of the [VAKHUM](https://github.com/BMClab/BMC/blob/master/courses/refs/VAKHUM.pdf) project for the following anatomical landmark positions (in metres): RASIS = [0.5, 0.8, 0.4], LASIS = [0.55, 0.78, 0.1], RPSIS = [0.3, 0.85, 0.2], LPSIS = [0.29, 0.78, 0.3].

    6. A treadmill's belt speeds up steadily from 2 to 4 m/s in 4 s, while a 60 kg runner stays in the same place in the laboratory.<br>
       a. What is the acceleration of the belt?<br>
       b. In the frame of reference of the belt, what fictitious force must be added for Newton's second law to hold? In which direction does it point?<br>
       c. Is the frame of the belt inertial before, during and after the change of speed?

    7. The landmarks of the left femur of the person in this notebook are the mirror images of those of the right femur: fh = [0.02, 0.92, −0.09], le = [0.03, 0.50, −0.14] and me = [0.02, 0.49, −0.05] (in metres). Calculate the anatomical coordinate system of the left femur according to the VAKHUM convention, check that it is right-handed, and compare it with that of the right femur. Is the left knee also rotated inwards?

    8. A person stands still on a force plate in a lift. While the lift moves at constant speed, the plate reads 600 N; when the lift starts or stops, the reading changes to 660 N.<br>
        a. What is the person's mass?<br>
        b. What is the acceleration of the lift, and in which direction does it point?<br>
        c. Is the lift starting to go up, or stopping on the way down? Can the force plate tell the two apart?

    9. In the model of the sprinter's hand ("One hand, two frames"), keep the running speed, the arm length and the frequency of the arm swing, and find the amplitude of the swing for which the hand comes to a complete stop relative to the track once per stride. Is that amplitude realistic for sprinting?

    10. In the standing posture of the femur example, the tibial tuberosity is at [0.06, 0.44, 0.10] m in the laboratory. Calculate its local coordinates in the femur's coordinate system and interpret their signs. Would they stay the same when the knee flexes?

    11. Write a function that receives three vectors and returns whether they form an orthonormal, right-handed basis, within a tolerance you choose. Test it with the bases of this notebook, including the one in Exercise 5, and with a basis that is orthogonal but not normalized.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers to the problems

    1. At rest relative to the ground: approximately 0 m/s relative to that surface. Relative to the Sun: approximately 29.8 km/s from Earth’s orbit, with a smaller latitude- and time-dependent contribution from Earth’s spin. The velocity vector, reference and instant must be specified.
    2. One right-handed orthonormal basis, using the notebook’s cross-product recipe, is $e_1=(-2,-1,1)/\sqrt6$, $e_2=(0,-1,-1)/\sqrt2$, $e_3=(1,-1,1)/\sqrt3$.
    3. The position vectors do not form a 3D basis: the first is twice the third. The three *points* are noncollinear and can still define a technical frame using differences.
    4. Let $A=(RASIS+LASIS)/2$, $P=(RPSIS+LPSIS)/2$ and $\operatorname{unit}(v)=v/\|v\|$. Then $e_z=\operatorname{unit}(RASIS-LASIS)$, $e_y=\operatorname{unit}[e_z\times(A-P)]$, $e_x=e_y\times e_z$; origin $A$.
    5. Origin $(0.525,0.790,0.250)$ m. Basis vectors as rows: $[[0.980530,-0.098327,0.169977],[0.107934,0.992988,-0.048210],[-0.164045,0.065618,0.984268]]$.
    6. With forward positive, the upper belt surface has $a=-0.5$ m/s²; fictitious force on the runner is +30 N, forward. Its uniformly translating frame is inertial before and after, but not during the speed change; this refers to the straight upper section, not a belt material point travelling around a roller.
    7. Origin $(0.025,0.495,-0.095)$ m. Rows: $[[0.993703,0.010378,0.111565],[-0.011763,0.999862,0.011763],[-0.111427,-0.013001,0.993688]]$; determinant +1. The left knee is internally rotated by about 6.4°, mirrored anatomically from the right.
    8. Mass $600/9.81=61.16$ kg; acceleration +0.981 m/s² upward. Either upward starting or downward stopping is compatible; the force plate alone cannot distinguish them.
    9. Swing amplitude $1.0851$ rad, or $62.17^\circ$ on either side of vertical. This is a large but possible idealized arm swing; the rigid straight-arm sinusoid should not be mistaken for a measured sprint.
    10. Local coordinates $(0.033651,-0.055463,0.008153)$ m: anterior, below the knee origin, and to the right. They change when the tibia moves relative to the femur.
    11. A valid final predicate for vectors stored as rows in $B$ is `np.allclose(B @ B.T, np.eye(3), atol=1e-8, rtol=0) and np.isclose(np.linalg.det(B), 1, atol=1e-8, rtol=0)`. It accepts right-handed orthonormal frames and rejects the reflected Exercise 5 frame and non-unit orthogonal vectors.

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

    - Klein et al. (2015), [The right frame of reference makes it simple](https://doi.org/10.1088/0143-0807/36/1/015004): an example of introductory mechanics supported by video analysis of motion.
    - [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py): how errors in marker positions become errors in the axes, and how to place markers to keep them small.
    - [Kinematics of a particle](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_particle.py): position, velocity and acceleration, described in an inertial frame.
    - [Rigid-body transformations in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py): how to go from one coordinate system to another.
    - [SVD algorithm](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/svd_algorithm.py): the best-fitting basis for more than three markers.

    ### Video lectures on the Internet

    - [Introduction to reference frames - Khan Academy](https://www.khanacademy.org/science/physics/one-dimensional-motion/displacement-velocity-time/v/introduction-to-reference-frames)
    - [Introduction to orthonormal bases - Khan Academy](https://www.khanacademy.org/math/linear-algebra/alternate-bases/orthonormal-basis/v/linear-algebra-introduction-to-orthonormal-bases)
    - [The Gram-Schmidt process - Khan Academy](https://www.khanacademy.org/math/linear-algebra/alternate-bases/orthonormal-basis/v/linear-algebra-the-gram-schmidt-process)
    - [Biomechanics of Movement | Demo: Motion Capture Placement and Reference Frames](https://youtu.be/ctwoH59Obew)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional assignments from the original sources

    These original reading assignments remain available for further study. Their full statements and source-specific answer keys are external to this guide.

    - Complete the Wisc-Online interactive lesson [Anatomical terminology: relative position](https://www.wistechopen.org/wisc-online-learning-object/anatomical-terminology-relative-position) to learn about the anatomical terminology that describes relative position in the human body.

    - To learn more about Cartesian coordinate systems, study the material at [Math is Fun: Cartesian coordinates](https://www.mathsisfun.com/data/cartesian-coordinates.html) and answer the 10 questions at the end.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Cappozzo A, Catani F, Della Croce U, Leardini A (1995) [Position and orientation in space of bones during movement: anatomical frame definition and determination](https://doi.org/10.1016/0268-0033(95)91394-T). Clinical Biomechanics, 10(4), 171–178.

    - Cappozzo A, Cappello A, Della Croce U, Pensalfini F (1997) [Surface-marker cluster design criteria for 3-D bone movement reconstruction](https://doi.org/10.1109/10.649988). IEEE Transactions on Biomedical Engineering, 44(12), 1165–1174.

    - Corke P (2017) [Robotics, Vision and Control: Fundamental Algorithms in MATLAB](https://petercorke.com/RVC/). 2nd ed. Springer-Verlag Berlin.

    - Klein P, Gröber S, Kuhn J, Fleischhauer A, Müller A (2015) [The right frame of reference makes it simple: an example of introductory mechanics supported by video analysis of motion](https://doi.org/10.1088/0143-0807/36/1/015004). European Journal of Physics, 36(1), 015004.

    - [Standards - International Society of Biomechanics](https://isbweb.org/activities/standards).

    - Stanford Encyclopedia of Philosophy. [Space and Time: Inertial Frames](https://plato.stanford.edu/entries/spacetime-iframes/).

    - van Ingen Schenau GJ (1980) [Some fundamental aspects of the biomechanics of overground versus treadmill locomotion](https://pubmed.ncbi.nlm.nih.gov/7421475/). Medicine and Science in Sports and Exercise, 12(4), 257–261.

    - [Virtual Animation of the Kinematics of the Human for Industrial, Educational and Research Purposes (VAKHUM)](https://github.com/BMClab/BMC/blob/master/courses/refs/VAKHUM.pdf).
    """)
    return


if __name__ == "__main__":
    app.run()
