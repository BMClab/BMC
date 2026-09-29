import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Angular kinematics in a plane (2D)

    > Marcos Duarte,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Human motion is a combination of linear and angular movement, and it happens in three-dimensional space. For many movements, depending on the detail the analysis needs, it is enough to study the motion in its main plane: walking seen from the side, a squat, the swing of an arm. The simplification is welcome because a planar (2D) analysis is far cheaper than a 3D one. One camera can replace several, and trigonometry can replace rotation matrices.

    This notebook is about angles in that plane. It covers how to compute them from marker coordinates without falling into the classic traps, how to turn two segment angles into a joint angle, and how to differentiate an angle into an angular velocity and an angular acceleration. It ends with real data, the leg of a woman walking, and with the link between angular and linear kinematics.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and being wrong is the most useful thing that can happen to you here.

    You will need the dot and cross products from [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb) and the finite differences from [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb).

    **Challenge 0.** Stand up and straighten one knee. Write down the knee angle. Now write down a second number, just as defensible, for exactly the same posture. Keep both; the next section is about why you needed two.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is a knee angle?

    Two gait laboratories measure the same person standing upright. One reports a knee angle of $0^o$, the other $180^o$. Both are right.

    An angle is always measured *between two specific things*, *from* one of them *to* the other, in a chosen direction. Change any of those choices and the number changes, while the knee stays exactly where it was. The figure below shows the convention used in Winter's textbook, which is also the one used for the data at the end of this notebook:

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/jointangles.png?raw=1" width=350 alt="Joint angle convention"/></center><figcaption><center><i>Figure. Convention for the sagittal joint angles of the lower limb (from Winter, 2009).</i></center></figcaption></figure>

    In this convention, each *segment* angle is measured counterclockwise from the forward horizontal, with the segment pointing from its distal marker to its proximal one; for instance, the leg (shank) angle $\theta_{43}$ goes from the ankle, marker 4, to the head of the fibula, marker 3. A *joint* angle is then a difference of segment angles. The knee angle is $\theta_{21}-\theta_{43}$, positive for flexion, so a straight knee is at $0^o$. Many clinicians report instead the included angle between thigh and leg, which is $180^o$ for the same straight knee.

    **Guiding questions 0.**

    1. In the convention of the figure, roughly what is the knee angle in a deep squat? And the included angle?
    2. Why does the ankle formula in the figure add $90^o$?
    3. A report says "peak knee flexion of $60^o$" and nothing more. What do you need to ask before comparing that number with your own data?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the computation, pandas reads the data file at the end, and Matplotlib draws the plots.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return FancyArrowPatch, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Angles in a plane

    In the planar case, computing an angle reduces to trigonometry on the kinematic data. Given the coordinates in the plane of two markers on a segment, as in the figure below, the angle of the segment can be found with the inverse of the sine, the cosine or the tangent.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/segment.png?raw=1" width=250 alt="segment"/></center><figcaption><center><i>Figure. A segment in a plane and its coordinates.</i></center></figcaption></figure>

    The inverse of the tangent is preferred, because it uses both coordinates and so keeps its accuracy at every orientation. For the segment in the figure:

    $$
    \theta = \arctan\left(\frac{y_2-y_1}{x_2-x_1}\right)
    $$

    In NumPy this is `numpy.arctan((y2 - y1) / (x2 - x1))`. But the division throws information away. A segment at $45^o$ and the same segment pointing the opposite way, at $225^o$, have the same ratio $\Delta y/\Delta x$, so `arctan` cannot tell them apart.

    **Before you run the next cell**, predict what `arctan` returns for a segment at $225^o$.
    """)
    return


@app.cell
def _(np):
    print("Segment at 45 degrees, from (0, 0) to (1, 1):")
    print(f"  arctan:  {np.rad2deg(np.arctan((1 - 0) / (1 - 0))):7.1f}")
    print(f"  arctan2: {np.rad2deg(np.arctan2(1 - 0, 1 - 0)):7.1f}")
    print("Segment at 225 degrees, from (0, 0) to (-1, -1):")
    print(f"  arctan:  {np.rad2deg(np.arctan((-1 - 0) / (-1 - 0))):7.1f}")
    print(f"  arctan2: {np.rad2deg(np.arctan2(-1 - 0, -1 - 0)):7.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `arctan` says $45^o$ for both. The function `numpy.arctan2(y, x)` receives the two differences separately, so it knows their signs and therefore the quadrant, and it gets the second segment right: $-135^o$ is the same direction as $225^o$. Be aware of that convention. `arctan2` returns angles in the interval $[-\pi, \pi]$, that is, between $-180^o$ and $180^o$.

    NumPy also has functions to convert an angle from radians to degrees and back:
    """)
    return


@app.cell
def _(np):
    print("np.rad2deg(np.pi) =", np.rad2deg(np.pi))
    print("np.deg2rad(180) =", np.deg2rad(180))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two turns of the arm

    Let's simulate the planar motion of an arm performing two complete turns around the shoulder, at one turn per second, sampled at 100 Hz, and compute its angle with `arctan2`.

    **Before you run the next cell**, sketch what you expect the angle to look like over the two seconds.
    """)
    return


@app.cell
def _(np, plt):
    t = np.arange(0, 2, 0.01)
    x = np.cos(2 * np.pi * t)
    y = np.sin(2 * np.pi * t)
    _angle = np.rad2deg(np.arctan2(y, x))

    plt.figure(figsize=(12, 4))
    _ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    _ax1.plot(x, y, "go", markersize=4)
    _ax1.plot(0, 0, "ko", markersize=8)
    _ax1.set_xlabel("x [m]")
    _ax1.set_ylabel("y [m]")
    _ax1.set_xlim([-1.1, 1.1])
    _ax1.set_ylim([-1.1, 1.1])
    _ax1.set_aspect("equal")
    _ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    _ax2.plot(t, x, "bo", markersize=4, label="x")
    _ax2.plot(t, y, "ro", markersize=4, label="y")
    _ax2.legend(numpoints=1, frameon=True, framealpha=0.8, loc="upper right")
    _ax2.set_ylabel("Position [m]")
    _ax2.set_ylim([-1.1, 1.1])
    _ax3 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    _ax3.plot(t, _angle, "go", markersize=4)
    _ax3.set_yticks(np.arange(-180, 181, 90))
    _ax3.set_xlabel("Time [s]")
    _ax3.set_ylabel("Angle [$^o$]")
    plt.tight_layout()
    plt.show()
    return t, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because the output of `arctan2` is bounded to $[-180^o, 180^o]$, the angle looks chopped: every time the arm passes $180^o$ it appears to jump back by a full turn, which it obviously did not do. Differentiate that curve and the jump becomes a huge, completely fictitious angular velocity.

    The function `numpy.unwrap` solves the problem. It looks for jumps larger than $180^o$ ($\pi$ rad) between consecutive samples and removes them by adding or subtracting whole turns:
    """)
    return


@app.cell
def _(np, plt, t, x, y):
    _angle = np.rad2deg(np.unwrap(np.arctan2(y, x)))

    _, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(t, _angle, "go", markersize=4)
    _ax.set_yticks(np.arange(0, 721, 90))
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Angle [$^o$]")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two turns, $720^o$, as a smooth ramp. Note that `unwrap` expects and returns radians; convert to degrees afterwards, as above.

    `unwrap` rests on an assumption that is worth knowing: that the true change between two consecutive samples is less than half a turn. At 100 Hz this arm turns $3.6^o$ per sample, so the assumption is safe. It is not safe for any sampling rate.

    **Challenge 1.** Sample the same two turns every 0.4 s and then every 0.6 s (`np.arange(0, 2, 0.4)`), and unwrap each. How many degrees does `unwrap` report in each case, and in which direction does the arm seem to turn in the second one? Where have you seen the same illusion in a film of a spinning wheel?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Joint angles

    To measure the angle of a joint, that is, the angle of one segment relative to another, we can simply subtract the two segment angles. This is correct only if both angles are in the same plane.
    """)
    return


@app.cell
def _(FancyArrowPatch, np, plt):
    x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0  # segment 1
    x3, y3, x4, y4 = 1.1, 1.0, 2.1, 0.0  # segment 2
    _arc = {
        "arrowstyle": "->,head_length=10,head_width=5",
        "connectionstyle": "arc3,rad=0.3",
    }

    _, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot((x1, x2), (y1, y2), "b-", (x3, x4), (y3, y4), "b-", linewidth=3)
    _ax.plot((x1, x2, x3, x4), (y1, y2, y3, y4), "ro", markersize=12)
    _ax.add_patch(
        FancyArrowPatch(posA=(x1 + np.sqrt(2) / 3, y1), posB=(x2 / 3, y2 / 3), **_arc)
    )
    _ax.text(1 / 2, 1 / 5, r"$\theta_1$", fontsize=24)
    _ax.add_patch(
        FancyArrowPatch(
            posA=(x4 + np.sqrt(2) / 3, y4), posB=(x4 - 1 / 3, y4 + 1 / 3), **_arc
        )
    )
    _ax.text(x4 + 0.2, y4 + 0.3, r"$\theta_2$", fontsize=24)
    _ax.add_patch(
        FancyArrowPatch(
            posA=(x2 - 1 / 3, y2 - 1 / 3), posB=(x3 + 1 / 3, y3 - 1 / 3), **_arc
        )
    )
    _ax.text(x1 + 0.8, y1 + 0.35, r"$\theta_J=\theta_2-\theta_1$", fontsize=24)
    _ax.set_xticks((x1, x2, x3, x4))
    _ax.set_xticklabels(("x1", "x2", "x3", "x4"), fontsize=13)
    _ax.set_yticks((y1, y2))
    _ax.set_xlim(min(x1, x2, x3, x4) - 0.1, max(x1, x2, x3, x4) + 0.5)
    _ax.set_ylim(min(y1, y2, y3, y4) - 0.1, max(y1, y2, y3, y4) + 0.1)
    _ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()
    return x1, x2, x3, x4, y1, y2, y3, y4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The joint angle is the difference between the adjacent segment angles:
    """)
    return


@app.cell
def _(np, x1, x2, x3, x4, y1, y2, y3, y4):
    ang1 = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    ang2 = np.rad2deg(np.arctan2(y3 - y4, x3 - x4))

    print(f"theta_1 = {ang1:.1f} degrees")
    print(f"theta_2 = {ang2:.1f} degrees")
    print(f"theta_J = {ang2 - ang1:.1f} degrees")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look at the order of the markers in the two calls. Segment 1 points from marker 1 to marker 2, and segment 2 from marker 4 to marker 3: in both cases from the lower end to the upper end, as Winter's convention points every segment from distal to proximal. Point either segment the other way and its angle changes by $180^o$, and so does the joint angle. That choice, and which angle is subtracted from which, *is* the convention from the section on knee angles.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Angle between two vectors

    Sometimes we have the 3D coordinates of the markers but only care about the angle between two segments in the plane the two segments define. (If the segments also move considerably out of that plane, this simple angle can give unexpected results.) Say `p1` and `p2` are the coordinates of the markers on segment 1, and `p3` and `p4` those on segment 2; the segments are then the vectors $\vec{\mathbf{a}} = p_1 - p_2$ and $\vec{\mathbf{b}} = p_4 - p_3$.

    The angle between them follows from the definition of the dot product:

    $$
    \vec{\mathbf{a}} \cdot \vec{\mathbf{b}} = \Vert\vec{\mathbf{a}}\Vert\,\Vert\vec{\mathbf{b}}\Vert\cos(\theta)
    \quad \Longrightarrow \quad
    \theta = \arccos\left(\frac{\vec{\mathbf{a}} \cdot \vec{\mathbf{b}}}{\Vert\vec{\mathbf{a}}\Vert\,\Vert\vec{\mathbf{b}}\Vert}\right)
    $$

    or from the magnitude of the cross product:

    $$
    \Vert\vec{\mathbf{a}} \times \vec{\mathbf{b}}\Vert = \Vert\vec{\mathbf{a}}\Vert\,\Vert\vec{\mathbf{b}}\Vert\sin(\theta)
    \quad \Longrightarrow \quad
    \theta = \arcsin\left(\frac{\Vert\vec{\mathbf{a}} \times \vec{\mathbf{b}}\Vert}{\Vert\vec{\mathbf{a}}\Vert\,\Vert\vec{\mathbf{b}}\Vert}\right)
    $$

    Each has a weakness. The arccosine loses accuracy for nearly parallel vectors, where the cosine barely changes with the angle. The arcsine loses it near $90^o$, and it cannot tell $\theta$ from $180^o-\theta$ at all. Dividing the second equation by the first gives $\tan(\theta)$, and `arctan2` combines the strengths of both with no division by the norms:

    ```python
    angle = np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
    ```

    See [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb) for a review of the dot and cross products. Let's write it as a function:
    """)
    return


@app.function
def angle_between(a, b):
    """Angle [deg] between the vectors `a` and `b`, in the interval [0, 180].

    Works for 2D and 3D vectors; 2D vectors get a zero third component,
    because NumPy's cross product only accepts 3D vectors.
    """
    import numpy as np

    a = np.pad(np.asarray(a, dtype=float), (0, 3 - len(a)))
    b = np.pad(np.asarray(b, dtype=float), (0, 3 - len(b)))

    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the padding. Older versions of NumPy quietly accepted 2D vectors in `np.cross` and assumed a zero third component; since NumPy 2.0 that is deprecated, and current versions raise an error. Our function adds the zero explicitly.

    We can use it to calculate the joint angle of the two segments above, which are 2D:
    """)
    return


@app.cell
def _(np):
    p1, p2 = np.array([0, 0]), np.array([1, 1])  # segment 1
    p3, p4 = np.array([1.1, 1]), np.array([2.1, 0])  # segment 2

    print(f"Joint angle: {angle_between(p1 - p2, p4 - p3):.1f} degrees")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same $90^o$ as the difference of the segment angles.

    ### The missing sign

    There is a price for that generality. **Before you run the next cell**, predict what the function returns for a joint bent $30^o$ one way and for the same joint bent $30^o$ the other way, as a knee in flexion and a knee in hyperextension.
    """)
    return


@app.function
def planar_angle(a, b):
    """Signed angle [deg] from the planar vector `a` to the planar vector `b`.

    Positive when `b` is counterclockwise from `a`, in the interval (-180, 180].
    """
    import numpy as np

    return float(np.rad2deg(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])))


@app.cell
def _(np):
    _a = [1.0, 0.0]
    for _bend in [30, -30]:
        _b = [np.cos(np.deg2rad(_bend)), np.sin(np.deg2rad(_bend))]
        print(
            f"bent {_bend:+d} degrees:  angle_between -> {angle_between(_a, _b):5.1f}"
            f"   planar_angle -> {planar_angle(_a, _b):6.1f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `angle_between` returns $30^o$ both times. It uses the *magnitude* of the cross product, and a magnitude has no sign. That is not a flaw in the code: in 3D there is no preferred side from which to look at the plane of two vectors, so "clockwise" means nothing until you choose one.

    In a plane there is a preferred side, the one we look from, along $+\hat{\mathbf{k}}$. The $z$ component of the cross product, $a_x b_y - a_y b_x$, keeps its sign, and `planar_angle` uses it to tell the two cases apart. For a knee, that sign is the difference between $5^o$ of flexion and $5^o$ of hyperextension, a distinction a clinician will care about a great deal.

    **Guiding questions 1.**

    1. What fixed direction did `planar_angle` use implicitly to decide the sign? What happens to the sign if you look at the same movement from the other side of the plane, as a camera on the left side of a walkway would, compared with one on the right?
    2. Compute the angle between $(1, 0, 0)$ and $(\cos 10^{-8}, \sin 10^{-8}, 0)$ with `np.arccos` of the normalized dot product and with `angle_between`. Which one is right?
    3. For 3D markers of a knee, how would you recover the sign of the angle? What extra information do you need?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Angular position, velocity, and acceleration

    In a plane, the angular position of a segment is described by one angle, $\theta$, measured about the axis perpendicular to the plane; the motion, if there is any, is said to occur around that axis. The angular position can be treated as a vector along that axis, with its sense given by the right-hand rule, and because the axis never changes, it behaves like a number with a sign: counterclockwise positive. In 3D this is no longer true for finite angles, which do not add like vectors; see [Angular velocity in 3D](https://github.com/BMClab/BMC/blob/master/notebooks/AngularVelocity3D.ipynb).

    The **angular velocity** is the rate of change of the angular position with respect to time. Over a finite interval, the average angular velocity is

    $$
    \bar{\omega} = \frac{\theta(t_2)-\theta(t_1)}{t_2-t_1} = \frac{\Delta \theta}{\Delta t}
    $$

    and the instantaneous angular velocity is its limit as the interval shrinks to zero:

    $$
    \omega(t) = \frac{d\theta(t)}{dt}
    $$

    The **angular acceleration** is the rate of change of the angular velocity, which is also the second-order rate of change of the angular position:

    $$
    \bar{\alpha} = \frac{\omega(t_2)-\omega(t_1)}{t_2-t_1} = \frac{\Delta \omega}{\Delta t}
    \qquad \text{and} \qquad
    \alpha(t) = \frac{d\omega(t)}{dt} = \frac{d^2\theta(t)}{dt^2}
    $$

    The angular velocity and acceleration vectors have the same direction as the angular position, perpendicular to the plane of rotation, with the sense given by the right-hand rule.

    ### The antiderivative

    As the angular acceleration is the derivative of the angular velocity, which is the derivative of the angular position, the inverse operation is the [antiderivative](http://en.wikipedia.org/wiki/Antiderivative), or integral:

    $$
    \theta(t) = \theta_0 + \int \omega(t)\, dt
    \qquad \text{and} \qquad
    \omega(t) = \omega_0 + \int \alpha(t)\, dt
    $$

    These are the same relations as between linear position, velocity and acceleration in [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb), with $\theta$ in place of $x$. Everything learned there about finite differences carries over, including what they do to noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The leg during walking

    Time for real data. Table A.1 of Winter (2009) gives the coordinates of eight markers on the right side of a woman walking at a fast cadence: 22 years old, 55.7 kg, 156 cm tall, 115 steps per minute. The coordinates are in centimetres, in the sagittal plane, with $x$ in the direction of walking and $y$ vertical, and there are 106 frames, sampled at about 70 Hz. The file is in this repository's [data folder](https://github.com/BMClab/BMC/blob/master/data/WinterTableA1.txt), originally from [Winter's book student site](http://bcs.wiley.com/he-bcs/Books?action=index&bcsId=5453&itemId=0470398183).

    Let's load the data and plot the markers' positions:
    """)
    return


@app.cell
def _(pd):
    data = pd.read_csv(
        "https://raw.githubusercontent.com/BMClab/BMC/master/data/WinterTableA1.txt",
        sep=" ",
        header=None,
        skiprows=2,
    ).to_numpy()
    markers = ["RIB CAGE", "HIP", "KNEE", "FIBULA", "ANKLE", "HEEL", "MT5", "TOE"]

    print("Columns: frame, time, then x and y for each marker")
    print("Shape of the data:", data.shape)
    return data, markers


@app.cell
def _(data, markers, plt):
    plt.figure(figsize=(10, 6))

    _ax = plt.subplot2grid((2, 2), (0, 0))
    _ax.plot(data[:, 1], data[:, 2::2])
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Horizontal [cm]")

    _ax = plt.subplot2grid((2, 2), (0, 1))
    _ax.plot(data[:, 1], data[:, 3::2])
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Vertical [cm]")

    _ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    _ax.plot(data[:, 2::2], data[:, 3::2])
    _ax.set_xlabel("Horizontal [cm]")
    _ax.set_ylabel("Vertical [cm]")
    _ax.legend(markers, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Markers")
    plt.suptitle(
        "Table A.1 (Winter, 2009): female, 22 yrs, 55.7 kg, 156 cm, "
        "fast cadence (115 steps/min)",
        y=1.0,
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will follow one segment, the leg, with the same definition as Winter's $\theta_{43}$ in the figure at the top: the vector from the ankle to the head of the fibula, measured counterclockwise from the forward horizontal. A vertical leg is then at $90^o$; a leg leaning forward, with the knee ahead of the ankle, is below $90^o$.

    One detail about time. The time column is rounded to the millisecond, so the intervals between frames alternate between 0.014 and 0.015 s. The actual sampling interval is their mean, about 1/70 s, and that is what we use.

    **Before you run the next cell**, predict:

    1. The range of the leg angle over the stride. Does the leg ever lean backwards past the vertical?
    2. The largest angular velocity of the leg, in degrees per second. Is it during stance, with the foot on the ground, or during swing?
    """)
    return


@app.cell
def _(data, np):
    time = data[:, 1]
    dt = np.mean(np.diff(time))  # s

    fibula = data[:, 8:10] / 100  # m
    ankle = data[:, 10:12] / 100  # m
    leg = fibula - ankle

    leg_angle = np.unwrap(np.arctan2(leg[:, 1], leg[:, 0]))  # rad
    leg_omega = np.gradient(leg_angle, dt, edge_order=2)  # rad/s
    leg_alpha = np.gradient(leg_omega, dt, edge_order=2)  # rad/s2

    print(f"Sampling interval: {dt:.4f} s ({1 / dt:.1f} Hz)")
    print(
        f"Leg angle: from {np.rad2deg(leg_angle.min()):.1f} to "
        f"{np.rad2deg(leg_angle.max()):.1f} degrees"
    )
    _i = np.argmax(leg_omega)
    print(
        f"Largest angular velocity: {leg_omega[_i]:.2f} rad/s "
        f"({np.rad2deg(leg_omega[_i]):.0f} degrees/s) at t = {time[_i]:.2f} s"
    )
    print(
        f"Angular acceleration: from {leg_alpha.min():.0f} to "
        f"{leg_alpha.max():.0f} rad/s2"
    )
    return dt, leg, leg_alpha, leg_angle, leg_omega, time


@app.cell
def _(leg_alpha, leg_angle, leg_omega, np, plt, time):
    _fig, _axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    _axs[0].plot(time, np.rad2deg(leg_angle), color="tab:blue", linewidth=2)
    _axs[0].axhline(90, color="0.5", linestyle=":")
    _axs[0].set_ylabel("Angle [$^o$]")
    _axs[1].plot(time, leg_omega, color="tab:green", linewidth=2)
    _axs[1].set_ylabel(r"$\omega$ [rad/s]")
    _axs[2].plot(time, leg_alpha, color="tab:red", linewidth=2)
    _axs[2].set_ylabel(r"$\alpha$ [rad/s$^2$]")
    _axs[2].set_xlabel("Time [s]")
    for _ax in _axs:
        _ax.axhline(0, color="k", linewidth=0.5)
        for _toe_off in (time[0], time[69]):
            _ax.axvline(_toe_off, color="0.5", linestyle="--", linewidth=1)
    _axs[0].text(time[0] + 0.02, 100, "toe-off", fontsize=11)
    _axs[0].text(time[69] + 0.02, 100, "toe-off", fontsize=11)
    _axs[0].set_title("Leg (shank) angle during walking, Winter's Table A.1")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dashed lines mark two consecutive toe-offs of the right foot, frames 1 and 70, so the plot shows one full stride and the beginning of the next.

    Right after toe-off the leg swings forward fast. It rotates counterclockwise from about $30^o$, leaning far forward, to about $115^o$, past the vertical, with the ankle ahead of the knee as the foot reaches for the ground. Its angular velocity peaks at about 7.5 rad/s, over $400^o$ per second. Then the foot lands, and during stance the body passes over it: the leg rotates back slowly, clockwise, until the next toe-off.

    Now look at the bottom panel. The angle is smooth, the angular velocity is still reasonably smooth, and the angular acceleration is rough, with spikes beyond 100 rad/s². The leg is not really jerking back and forth like that. Each differentiation amplifies whatever small errors the coordinates carry, as in [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb), and the second one amplifies them twice. Before angular accelerations from marker data are used for anything, such as the joint moments of an inverse dynamics analysis, the coordinates are low-pass filtered; see [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb).

    **Guiding questions 2.**

    1. The dotted line marks the vertical leg. How many times per stride does the leg pass through it, and what is happening in the gait cycle each time?
    2. Why is the angular velocity so much larger in swing than in stance?
    3. The leg angle stays between about $30^o$ and $116^o$, far from $\pm 180^o$. Did `unwrap` change anything here? For which segment of the lower limb, and with which choice of direction for its vector, would it be needed?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relationship between linear and angular kinematics

    Consider a particle rotating around a point at a fixed distance $r$, that is, in circular motion. As the particle moves along the circle it travels an arc of length $s$, and its angular position is

    $$
    \theta = \frac{s}{r}
    $$

    which is in fact the definition of the radian:

    <figure><center><img src="http://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Radian_cropped_color.svg/220px-Radian_cropped_color.svg.png" width=200 alt="radian"/></center><figcaption><center><i>Figure. An arc of a circle with the same length as the radius of that circle corresponds to an angle of 1 radian (<a href="https://en.wikipedia.org/wiki/Radian">image from Wikipedia</a>).</i></center></figcaption></figure>

    The distance travelled by the particle is therefore the arc length $s = r\theta$, and because the radius is constant, the speed and the rate of change of the speed follow directly:

    $$
    v = \frac{ds}{dt} = r\frac{d\theta}{dt} = r\omega
    \qquad \text{and} \qquad
    a_t = \frac{dv}{dt} = r\frac{d\omega}{dt} = r\alpha
    $$

    The subscript on $a_t$ matters. $r\alpha$ is the **tangential** acceleration, the part that changes the particle's speed, and it is not the whole acceleration. A particle moving on a circle at constant speed has $\alpha = 0$ and is still accelerating, because the direction of its velocity keeps turning. That is the **normal**, or centripetal, acceleration, pointing to the centre of the circle:

    $$
    a_n = \frac{v^2}{r} = \omega^2 r
    $$

    The general form of this split, for any curved path, is the subject of [Path frame](https://github.com/BMClab/BMC/blob/master/notebooks/PathFrame.ipynb). Note also that all these relations need $\theta$, $\omega$ and $\alpha$ in radians: $r\omega$ with $\omega$ in degrees per second is not a speed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Checking $v = r\omega$ on the leg

    The fibula head and the ankle are two points of the same rigid bone, so relative to the ankle, the fibula head moves on a circle of radius equal to the length of the leg, $L$. Its speed relative to the ankle should then be $L\,|\omega|$.

    **Before you run the next cell**, predict whether the two will agree, and whether $L$ will be constant.
    """)
    return


@app.cell
def _(dt, leg, leg_omega, np, plt, time):
    leg_length = np.linalg.norm(leg, axis=1)  # m
    _speed_rel = np.linalg.norm(np.gradient(leg, dt, axis=0, edge_order=2), axis=1)

    _fig, _axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    _axs[0].plot(time, _speed_rel, color="tab:blue", linewidth=3, label="measured relative speed")
    _axs[0].plot(time, leg_length * np.abs(leg_omega), color="tab:orange", linewidth=2, label=r"$L\,|\omega|$")
    _axs[0].set_ylabel("Speed [m/s]")
    _axs[0].legend(loc="upper center")
    _axs[1].plot(time, 100 * leg_length, color="k", linewidth=2)
    _axs[1].set_ylabel("Leg length L [cm]")
    _axs[1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    _i = np.argmax(leg_omega)
    print(
        f"Leg length: from {100 * leg_length.min():.1f} to "
        f"{100 * leg_length.max():.1f} cm"
    )
    print(
        f"At the peak angular velocity: measured {_speed_rel[_i]:.2f} m/s, "
        f"L|omega| = {leg_length[_i] * leg_omega[_i]:.2f} m/s, "
        f"centripetal acceleration omega^2 L = {leg_length[_i] * leg_omega[_i] ** 2:.1f} m/s2"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two curves agree almost everywhere, with a median difference of a few millimetres per second: the relative speed of the fibula head really is $L\omega$. At the peak, while the leg rotates at 7.5 rad/s, the fibula head also has a centripetal acceleration towards the ankle of $\omega^2 L \approx 19\;\mathrm{m/s^2}$, about twice $g$, just from the leg's rotation.

    Yet the bottom panel says the "rigid" leg changes length by almost 3 cm. The bone does not stretch. Markers on the skin slide over the bone, the digitized coordinates carry errors, and the leg is not perfectly in the plane of the camera. All of these show up as motion *along* the segment, which $v = r\omega$ cannot produce, at up to 0.4 m/s. Why does it barely show in the top panel? Because motion along the segment is perpendicular to the rotational motion, so the two add in quadrature: the measured speed is $\sqrt{v_r^2 + (L\omega)^2}$. Next to a fast rotation a small $v_r$ disappears. Only where the rotation is slow, during stance, as near 0.58 s, does the measured speed rise visibly above $L|\omega|$.

    **Challenge 2.** Split the relative velocity of the fibula head into its component along the leg and its component perpendicular to it, using the dot product with the leg's versor and with that versor rotated by $90^o$. Which component does $v = r\omega$ predict? Plot the other one and decide: does it look like a physical movement or like noise?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to the two knee angles you wrote down in Challenge 0. State the convention behind each one precisely enough that someone else would compute the same numbers.
    2. Give a segment orientation for which `arctan` of the ratio of the differences returns the wrong angle, and say what information it threw away.
    3. What is the lowest sampling rate at which `unwrap` could correctly follow the leg's swing, at its peak of about $430^o$ per second? Would you ever sample that slowly?
    4. When does it matter that `angle_between` is unsigned? Name one joint and one movement where losing the sign would lead to a wrong clinical conclusion.
    5. From marker data you computed $\theta$, $\omega$ and $\alpha$. Which do you trust the least, and why?
    6. A gymnast lets go of the bar at the bottom of a giant circle. In which direction is the velocity of the gymnast's centre of mass at the instant of release, and what is its magnitude in terms of $\omega$ and $r$?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. A gymnast performs giant circles around the horizontal bar (with the official dimensions for Artistic Gymnastics) at a constant rate of one circle every 2 s, and the gymnast's centre of mass is 1 m from the bar. At the lowest point, exactly beneath the bar, the gymnast releases it, moves forward, and lands standing on the ground.<br>
       a. Calculate the angular and linear velocity of the gymnast's centre of mass at the point of release.<br>
       b. Calculate the horizontal distance travelled by the gymnast's centre of mass.

    2. With the data from Table A.1 of Winter (2009), already loaded above in `data`, and the convention for the sagittal joint angles of the lower limb in the figure at the top:<br>
       a. Calculate and plot the angles of the foot and thigh segments (the leg was done above).<br>
       b. Calculate and plot the angles of the ankle, knee and hip joints.<br>
       c. Calculate and plot the angular velocities and accelerations of the joint angles calculated in b.<br>
       d. Compare the ankle angle using the two different conventions described by Winter (2009), that is, defining the foot segment with the MT5 or with the TOE marker.<br>
       e. Knowing that a stride corresponds to the data between frames 1 and 70 (two subsequent toe-offs of the right foot), can you suggest a way to determine a stride automatically? Hint: look at the vertical displacement and acceleration of the heel marker.

    3. Recompute the angular velocity and acceleration of the leg keeping only every second frame of the data (about 35 Hz). Compare with the results at 70 Hz. Which looks smoother? Which is more accurate, and how could you tell?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read pages 718–742 of chapter 15 of [Ruina and Pratap's book](http://ruina.tam.cornell.edu/Book/index.html) about the circular motion of a particle.
    - [Path frame](https://github.com/BMClab/BMC/blob/master/notebooks/PathFrame.ipynb) — the tangential and normal accelerations for any curved path.
    - [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb) — the dot and cross products behind `angle_between`.
    - [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) — what to do before differentiating measured angles.
    - [Angular velocity in 3D](https://github.com/BMClab/BMC/blob/master/notebooks/AngularVelocity3D.ipynb) — what changes when the plane is not fixed.

    ### Video lectures on the Internet

    - Khan Academy: [Uniform Circular Motion Introduction](https://www.khanacademy.org/science/ap-physics-1/ap-centripetal-force-and-gravitation)
    - [Angular Motion and Torque](https://www.youtube.com/watch?v=jNc2SflUl9U)
    - [Rotational Motion Physics, Basic Introduction, Angular Velocity & Tangential Acceleration](https://www.youtube.com/watch?v=WQ9AH2S8B6Y)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.
    - Winter DA (2009) [Biomechanics and Motor Control of Human Movement](http://books.google.com.br/books?id=_bFHL08IWfwC). 4th edition. Hoboken, USA: Wiley.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
