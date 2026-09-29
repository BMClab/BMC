import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Path frame

    > Renato Naville Watanabe, Marcos Duarte,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Every notebook before this one describes motion with the same three axes, fixed to the laboratory floor. That is a fine choice for bookkeeping and a poor one for understanding. Fixed axes do not care where the body is going, so they report "how fast" and "which way" mixed together in every component.

    This notebook introduces a basis that travels *with* the particle, with its first axis always pointing where the particle is heading. In that basis the acceleration splits into two pieces, each with a plain physical meaning. One changes how fast you go. The other changes where you go. The split explains why a sprinter running the bend at constant speed is still accelerating, and it tells you where a projectile's path is tightest without drawing it.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and being wrong is the most useful thing that can happen to you here.

    You will need the vector operations from [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb), mainly the norm and the cross product, and the definitions of velocity and acceleration from [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb).

    **Challenge 0.** Before you begin, think of a movement where the direction changes a lot but the speed hardly does: running a bend, cycling round a roundabout, the last turns of a hammer throw, a skater's arc. Write down whether you think the body is accelerating and, if so, in which direction. Keep your answer nearby; the checkpoint questions ask you to come back to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running the bend

    In the 200 m the athletes run the first half of the race on a bend. On a standard 400 m track the inner edge of lane 1 has a radius of 36.5 m and each lane is 1.22 m wide, so the line an athlete runs has a radius of about 37 m in lane 1 and about 45 m in lane 8.

    Suppose an athlete holds a perfectly steady 10 m/s all the way round the bend. The speedometer never moves.

    **Is the athlete accelerating?**

    The speed is constant, but the velocity is not: its direction turns through $180^o$ in about 11.5 s. And in the laboratory's fixed axes the answer is surprisingly hard to see. With the centre of the bend at the origin and $\phi = v\,t/R$ the angle travelled, the velocity is

    $$
    \vec{\mathbf{v}}(t) = v\left[-\sin(\phi)\,\hat{\mathbf{i}} + \cos(\phi)\,\hat{\mathbf{j}}\right]
    $$

    Both components swing up and down continuously. Nothing in either curve on its own says "constant speed, turning at a steady rate"; that information is smeared across the two of them. By the end of this notebook the same velocity will be written with a single term, and the acceleration with two terms that each say one thing.

    **Guiding questions 0.**

    1. If the athlete is accelerating, something must be pushing. What is it, and in which direction does it push?
    2. At the same speed, who needs the larger push: the athlete in lane 1 or the one in lane 8?
    3. The track is flat. Why do the athletes lean into the bend?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the numerical work and Matplotlib the plots. SymPy appears in the last section, for the symbolic version of the same computation.
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
    ## A fixed basis

    We perceive the space around us as three-dimensional, and the usual way to describe it is the [Cartesian coordinate system](http://en.wikipedia.org/wiki/Cartesian_coordinate_system) in [Euclidean space](http://en.wikipedia.org/wiki/Euclidean_space): three orthogonal axes, their directions fixed by the [right-hand rule](http://en.wikipedia.org/wiki/Right-hand_rule) and labelled X, Y and Z. Because the axes are orthogonal, motion along one of them is independent of motion along the others, which is what makes the coordinate system so convenient in classical mechanics.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/CCS.png?raw=1" width=350 alt="Cartesian coordinate system"/></center><figcaption><center><i>Figure. Representation of a point and its position vector in a Cartesian coordinate system.</i></center></figcaption></figure>

    In biomechanics we use several coordinate systems at once, and call them global, laboratory, local, anatomical or technical reference frames, depending on what they are attached to. Whatever the name, each is defined by a <a href="http://en.wikipedia.org/wiki/Basis_(linear_algebra)">basis</a>: a set of mutually orthogonal unit vectors (**versors**) from which any vector can be built by [linear combination](http://en.wikipedia.org/wiki/Linear_combination). Such a set is called an **orthonormal basis**.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/vector3Dijk.png?raw=1" width=350 alt="versors of the Cartesian basis"/></center><figcaption><center><i>Figure. Representation of a point $\mathbf{P}$ and its position vector $\vec{\mathbf{r}}$ in a Cartesian coordinate system. The versors $\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}$ form a basis for this coordinate system and are usually drawn in the colour sequence RGB (red, green, blue).</i></center></figcaption></figure>

    In the Cartesian coordinate system itself, the versors of this basis have the coordinates

    $$
    \hat{\mathbf{i}} = \begin{bmatrix}1\\0\\0 \end{bmatrix}, \quad \hat{\mathbf{j}} = \begin{bmatrix}0\\1\\0 \end{bmatrix}, \quad \hat{\mathbf{k}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
    $$

    and the position vector of the point is

    $$
    \vec{\mathbf{r}} = x\hat{\mathbf{i}} + y\hat{\mathbf{j}} + z\hat{\mathbf{k}}
    $$

    This basis is fixed: it does not know or care where the particle is going. That indifference is exactly what made the velocity on the bend look complicated. For many problems a fixed basis leads to needlessly complex expressions, and the fix is to choose a basis that moves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The path basis

    Consider a particle moving along a path, with its position described in the fixed reference frame by

    $$
    \vec{\mathbf{r}}(t) = x(t)\,\hat{\mathbf{i}} + y(t)\,\hat{\mathbf{j}} + z(t)\,\hat{\mathbf{k}}
    $$

    As it moves, the particle covers a distance $s(t)$ measured along the path: the **arc length**. The rate at which the arc length grows is the speed, $ds/dt = \Vert\vec{\mathbf{v}}\Vert$.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/velRefFrame.png?raw=1" width=500 alt="position vector of a moving particle"/></center><figcaption><center><i>Figure. Position vector of a moving particle in relation to a coordinate system.</i></center></figcaption></figure>

    Instead of describing every kinematic variable in the fixed frame, we attach a basis to the particle itself and let it travel along the path. Its three versors are built one at a time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tangential versor

    The natural first choice is a unit vector in the direction of the velocity, which is to say the direction the particle is going:

    $$
    \hat{\mathbf{e}}_t = \frac{\vec{\mathbf{v}}}{\Vert\vec{\mathbf{v}}\Vert}
    $$

    It is tangent to the path at every instant. Note the division: $\hat{\mathbf{e}}_t$ exists only while the particle moves. At an instant of zero speed it is undefined.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normal versor

    For the second versor we first define the **curvature vector** of the path, the rate at which the tangential versor changes per metre travelled:

    $$
    \vec{\mathbf{C}} = \frac{d\hat{\mathbf{e}}_t}{ds}
    $$

    Because $\hat{\mathbf{e}}_t$ depends on time only through the arc length $s(t)$, the chain rule converts this into something we can compute from a trajectory sampled in time:

    $$
    \frac{d\hat{\mathbf{e}}_t}{dt} = \frac{d\hat{\mathbf{e}}_t}{ds}\frac{ds}{dt}
    \quad \Longrightarrow \quad
    \vec{\mathbf{C}} = \frac{d\hat{\mathbf{e}}_t/dt}{ds/dt} = \frac{1}{\Vert\vec{\mathbf{v}}\Vert}\frac{d\hat{\mathbf{e}}_t}{dt}
    $$

    The curvature vector has two properties worth understanding rather than memorizing.

    - **It is perpendicular to the path.** A unit vector can change its direction but never its length. Differentiating $\hat{\mathbf{e}}_t \cdot \hat{\mathbf{e}}_t = 1$ gives $2\,\hat{\mathbf{e}}_t \cdot d\hat{\mathbf{e}}_t/ds = 0$, so $\vec{\mathbf{C}}$ has no component along $\hat{\mathbf{e}}_t$. It points to the side the path is bending towards.
    - **Its magnitude measures how sharply the path bends.** $\kappa = \Vert\vec{\mathbf{C}}\Vert$ is the **curvature**: the angle the direction of motion turns through per metre travelled, in rad/m. Its inverse, $\rho = 1/\kappa$, is the **radius of curvature**, the radius of the circle that best fits the path at that point. A straight line has $\kappa = 0$ and $\rho = \infty$; a circle of radius $R$ has $\kappa = 1/R$ everywhere.

    The second versor of the basis is the direction of the curvature vector:

    $$
    \hat{\mathbf{e}}_n = \frac{\vec{\mathbf{C}}}{\Vert\vec{\mathbf{C}}\Vert}
    $$

    Like $\hat{\mathbf{e}}_t$, it comes with a condition. Where the path is straight, $\vec{\mathbf{C}} = 0$ and there is no direction to normalize: $\hat{\mathbf{e}}_n$ is undefined.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/velRefFrameeten.png?raw=1" width=500 alt="path basis of a moving particle"/></center><figcaption><center><i>Figure. A moving particle and a corresponding path basis.</i></center></figcaption></figure>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Binormal versor

    The third versor completes a right-handed basis. It is the cross product of $\hat{\mathbf{e}}_t$ and $\hat{\mathbf{e}}_n$, in that order:

    $$
    \hat{\mathbf{e}}_b = \hat{\mathbf{e}}_t \times \hat{\mathbf{e}}_n
    $$

    The three versors $\hat{\mathbf{e}}_t$, $\hat{\mathbf{e}}_n$ and $\hat{\mathbf{e}}_b$ travel and turn together with the particle. The basis is known as the **Frenet–Serret frame**.

    For a path that stays in one plane, $\hat{\mathbf{e}}_t$ and $\hat{\mathbf{e}}_n$ both lie in that plane, so $\hat{\mathbf{e}}_b$ is perpendicular to it and can only be $+\hat{\mathbf{k}}$ or $-\hat{\mathbf{k}}$. Its sign tells you which way the path turns: counterclockwise for $+\hat{\mathbf{k}}$, clockwise for $-\hat{\mathbf{k}}$.

    **Challenge 1.** A particle moves along a straight line, $\vec{\mathbf{r}}(t) = 10t\,\hat{\mathbf{i}} + 5t\,\hat{\mathbf{j}}$. Work out $\hat{\mathbf{e}}_t$, $\vec{\mathbf{C}}$ and $\hat{\mathbf{e}}_n$ by hand. Which of the three versors exists? Then predict what the numerical code later in this notebook will print when you feed it this path, and try it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Velocity and acceleration in the path frame

    ### Velocity

    In the fixed frame, the velocity is the time derivative of each coordinate (see [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb)):

    $$
    \vec{\mathbf{v}}(t) = \dot{x}\,\hat{\mathbf{i}} + \dot{y}\,\hat{\mathbf{j}} + \dot{z}\,\hat{\mathbf{k}}
    $$

    In the path frame it takes one term. By the definition of $\hat{\mathbf{e}}_t$, all of the velocity lies along it:

    $$
    \vec{\mathbf{v}} = \Vert\vec{\mathbf{v}}\Vert\,\hat{\mathbf{e}}_t
    $$

    This is not a result; it is the definition of $\hat{\mathbf{e}}_t$ read backwards. The payoff comes when we differentiate it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Acceleration

    In the fixed frame, the acceleration is

    $$
    \vec{\mathbf{a}}(t) = \ddot{x}\,\hat{\mathbf{i}} + \ddot{y}\,\hat{\mathbf{j}} + \ddot{z}\,\hat{\mathbf{k}}
    $$

    In the path frame, differentiate $\vec{\mathbf{v}} = \Vert\vec{\mathbf{v}}\Vert\,\hat{\mathbf{e}}_t$ with the product rule. The versor $\hat{\mathbf{e}}_t$ is not constant, so it gets differentiated too, and the chain rule turns its derivative into the curvature vector:

    $$
    \begin{aligned}
    \vec{\mathbf{a}} &= \frac{d\vec{\mathbf{v}}}{dt} = \frac{d\left(\Vert\vec{\mathbf{v}}\Vert\,\hat{\mathbf{e}}_t\right)}{dt} \\
    &= \frac{d\Vert\vec{\mathbf{v}}\Vert}{dt}\,\hat{\mathbf{e}}_t + \Vert\vec{\mathbf{v}}\Vert\,\frac{d\hat{\mathbf{e}}_t}{dt} \\
    &= \frac{d\Vert\vec{\mathbf{v}}\Vert}{dt}\,\hat{\mathbf{e}}_t + \Vert\vec{\mathbf{v}}\Vert\,\frac{d\hat{\mathbf{e}}_t}{ds}\frac{ds}{dt} \\
    &= \frac{d\Vert\vec{\mathbf{v}}\Vert}{dt}\,\hat{\mathbf{e}}_t + \Vert\vec{\mathbf{v}}\Vert^2\,\Vert\vec{\mathbf{C}}\Vert\,\hat{\mathbf{e}}_n
    \end{aligned}
    $$

    Replacing $\Vert\vec{\mathbf{C}}\Vert$ by the radius of curvature, $\rho = 1/\Vert\vec{\mathbf{C}}\Vert$:

    $$
    \vec{\mathbf{a}} = \underbrace{\frac{d\Vert\vec{\mathbf{v}}\Vert}{dt}}_{a_t}\,\hat{\mathbf{e}}_t + \underbrace{\frac{\Vert\vec{\mathbf{v}}\Vert^2}{\rho}}_{a_n}\,\hat{\mathbf{e}}_n
    $$

    This is the equation the notebook has been building to, and each term says exactly one thing:

    - The **tangential acceleration**, $a_t$, is the rate of change of the speed. It makes the particle go faster or slower and has no effect on the direction.
    - The **normal** (or centripetal) **acceleration**, $a_n$, turns the velocity. It changes the direction and has no effect on the speed. It always points to the inside of the curve, and it grows with the *square* of the speed.
    - There is no third term. The acceleration always lies in the plane of $\hat{\mathbf{e}}_t$ and $\hat{\mathbf{e}}_n$; its component along $\hat{\mathbf{e}}_b$ is zero by construction.

    Back to the bend. The speed is constant, so $a_t = 0$. The radius is about 37 m, so

    $$
    a_n = \frac{(10\;\mathrm{m/s})^2}{37\;\mathrm{m}} \approx 2.7\;\mathrm{m/s^2}
    $$

    pointing to the centre of the bend. That is more than a quarter of the gravitational acceleration, sustained for the whole bend, and the only thing that can supply the force behind it is the ground, through the athlete's feet.

    For motion on a circle of radius $r$, $\rho = r$ and the speed is $\Vert\vec{\mathbf{v}}\Vert = r\omega$, so $a_n = \omega^2 r$: the centripetal acceleration of circular motion, which reappears in [Angular kinematics in a plane](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb).

    **Guiding questions 1.**

    1. A cyclist brakes while riding round a roundabout. Sketch $\hat{\mathbf{e}}_t$, $\hat{\mathbf{e}}_n$ and the acceleration vector. Is the acceleration pointing to the centre of the roundabout?
    2. At the top of a projectile's flight the speed is momentarily not changing. Which of the two terms is zero there, and which one carries all of $g$?
    3. A ball thrown straight up stops for an instant at the top. What happens to $\hat{\mathbf{e}}_t$ and $\hat{\mathbf{e}}_n$ at that instant? Is the acceleration zero?
    4. Double the speed on the same bend. By what factor does the normal acceleration change? And the force the athlete's legs must supply sideways?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: the path of a projectile

    Consider a particle following the path

    $$
    \vec{\mathbf{r}}(t) = (10t+100)\,\hat{\mathbf{i}} + \left(-\frac{9.81}{2}t^2+50t+100\right)\hat{\mathbf{j}}
    $$

    This is the path of a projectile (see [Projectile motion](https://github.com/BMClab/BMC/blob/master/notebooks/ProjectileMotion.ipynb)) launched from the point $(100, 100)$ m with a horizontal velocity of 10 m/s and a vertical velocity of 50 m/s. It is a good first example precisely because we already know the answer: the acceleration is $(0, -9.81)\;\mathrm{m/s^2}$ at every instant, and at the top of the flight, reached at $t = 50/9.81 = 5.10$ s, the velocity is horizontal and equal to 10 m/s. Whatever the path frame says has to be consistent with that.

    **Before you run the next cells**, predict two things:

    1. Where along the path is the curvature largest?
    2. What is the radius of curvature there? (Guiding question 1.2 is the hint.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Solving numerically

    We start numerically, because that is the method you need when there is no formula for the path, which is the usual situation in biomechanics, where the path comes from measured data.

    To mimic that situation, we sample the expression above at 100 Hz, a typical rate for motion capture, and from then on pretend we do not know where the numbers came from:
    """)
    return


@app.cell
def _(np):
    g = 9.81  # m/s2
    dt = 0.01  # s, 100 Hz
    t = np.arange(0, 10 + dt / 2, dt)
    r = np.column_stack((10 * t + 100, -g / 2 * t**2 + 50 * t + 100))

    print(f"{len(t)} samples, from t = {t[0]} s to t = {t[-1]:.2f} s")
    return dt, g, r, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every derivative now becomes a finite difference. We use `numpy.gradient`, which takes *central* differences, $(\vec{\mathbf{r}}_{i+1} - \vec{\mathbf{r}}_{i-1})/(2\Delta t)$. Unlike `numpy.diff`, it returns an array of the same length as its input and aligned with it in time. A forward difference is half a sample late and one sample short, and those offsets pile up when you differentiate twice, as we are about to. The option `edge_order=2` keeps the first and last samples accurate as well.

    The tangential versor is the velocity divided by the speed:
    """)
    return


@app.cell
def _(dt, np, r):
    v = np.gradient(r, dt, axis=0, edge_order=2)
    speed = np.linalg.norm(v, axis=1)

    e_t = v / speed[:, np.newaxis]
    return e_t, speed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The curvature vector is the time derivative of $\hat{\mathbf{e}}_t$ divided by the speed, and the normal versor is its direction:
    """)
    return


@app.cell
def _(dt, e_t, np, speed):
    C = np.gradient(e_t, dt, axis=0, edge_order=2) / speed[:, np.newaxis]
    kappa = np.linalg.norm(C, axis=1)

    e_n = C / kappa[:, np.newaxis]
    return e_n, kappa


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's draw the two versors along the path. The function below plots any set of vectors at regularly spaced points of a planar path. It keeps the two axes at the same scale: without that, perpendicular vectors are drawn at the wrong angle to each other, and the whole point of the picture is lost.
    """)
    return


@app.function
def plot_path_vectors(r, vectors, labels, step=50, scale=1.0, title=""):
    """Plot a planar path with vectors drawn at every `step`-th sample.

    `r` holds the positions [m] in rows; each array in `vectors` holds one
    vector per row of `r`. The vectors are multiplied by `scale` before being
    drawn, and are coloured red, green and blue, in that order.
    """
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(r[:, 0], r[:, 1], color="0.6", linewidth=2, zorder=0)
    idx = slice(0, None, step)
    for vec, label, color in zip(vectors, labels, ["tab:red", "tab:green", "tab:blue"]):
        ax.quiver(
            r[idx, 0],
            r[idx, 1],
            scale * vec[idx, 0],
            scale * vec[idx, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.005,
            color=color,
            label=label,
        )
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="lower center")
    plt.tight_layout()
    plt.show()


@app.cell
def _(e_n, e_t, r):
    plot_path_vectors(
        r,
        [e_t, e_n],
        [r"$\hat{\mathbf{e}}_t$", r"$\hat{\mathbf{e}}_n$"],
        scale=10,
        title="Path basis along the trajectory (versors drawn 10 m long)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The red versor follows the path, and the green one always points to the inside of the curve, which for a projectile is always somewhere downwards. The pair rotates clockwise from launch to landing.

    Now the components of the acceleration. The tangential component is the time derivative of the speed; the normal component is the curvature times the squared speed. Here are a few instants along the flight, including the top:
    """)
    return


@app.cell
def _(dt, g, kappa, np, speed, t):
    a_t = np.gradient(speed, dt, edge_order=2)
    a_n = kappa * speed**2

    print("   t      speed      a_t      a_n    radius of curvature   |a|")
    for _time in [0, 2, 50 / g, 8, 10]:
        _i = np.argmin(np.abs(t - _time))
        print(
            f"{t[_i]:5.2f} s {speed[_i]:7.2f} m/s {a_t[_i]:7.2f} {a_n[_i]:7.2f} m/s2"
            f" {1 / kappa[_i]:10.1f} m {np.hypot(a_t[_i], a_n[_i]):12.2f} m/s2"
        )
    return a_n, a_t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At the top of the flight the tangential acceleration is zero (to the precision of the sampling), all of $g$ is normal, and the radius of curvature is 10.2 m. That matches what the formula predicts: $\rho = \Vert\vec{\mathbf{v}}\Vert^2/a_n = 10^2/9.81 = 10.19$ m. At launch, the same path has a radius of curvature of 1351 m. The path bends a hundred times more sharply at the top than at the bottom, although nothing about the force changed.

    And yet the last column is the same at every instant. The two components change continuously, trading $g$ between them, but together they always make up the same $9.81\;\mathrm{m/s^2}$.

    **Guiding questions 2.**

    1. Why is the path nearly straight at launch and tightest at the top? Answer in terms of the angle between the velocity and gravity.
    2. $a_t$ changes sign at the top. What is it telling you before and after that instant?
    3. Call $\beta$ the angle between the velocity and the horizontal. Show that $a_t = -g\sin\beta$ and $a_n = g\cos\beta$, and check two rows of the table with it.
    4. Where along the path would you expect the numerical estimate of $\hat{\mathbf{e}}_n$ to be least reliable? Keep your answer for the section on measured data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Putting the acceleration back together

    If the decomposition is right, reassembling it must give back the acceleration we know, $\vec{\mathbf{a}} = a_t\,\hat{\mathbf{e}}_t + a_n\,\hat{\mathbf{e}}_n$. Let's draw it once every second together with the velocity, $\vec{\mathbf{v}} = \Vert\vec{\mathbf{v}}\Vert\,\hat{\mathbf{e}}_t$, both at the same scale of half a metre of arrow per unit:
    """)
    return


@app.cell
def _(a_n, a_t, e_n, e_t, g, np, r, speed):
    a = a_t[:, np.newaxis] * e_t + a_n[:, np.newaxis] * e_n

    print(
        f"Largest deviation of the reassembled acceleration from (0, -{g}): "
        f"{np.abs(a - [0, -g]).max():.4f} m/s2"
    )
    plot_path_vectors(
        r,
        [speed[:, np.newaxis] * e_t, a],
        [r"$\vec{\mathbf{v}}$ [m/s]", r"$\vec{\mathbf{a}}$ [m/s$^2$]"],
        step=100,
        scale=0.5,
        title="Velocity and acceleration rebuilt from the path frame",
    )
    return (a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every green arrow is the same: $9.81\;\mathrm{m/s^2}$, straight down. The pieces $a_t\,\hat{\mathbf{e}}_t$ and $a_n\,\hat{\mathbf{e}}_n$ change all along the path; their sum does not.

    This is the thing to take away from the example. **The path frame does not change the acceleration; it changes how we describe it.** In the fixed frame the projectile's acceleration is the simplest thing in the problem. In the path frame it is split into a part that is slowing the projectile down and a part that is bending its path, and it is that split, not the vector, that tells you about speed and turning.

    Finally, the binormal versor. The path is planar, so it should be $\pm\hat{\mathbf{k}}$ everywhere, and since the path turns clockwise, it should be $-\hat{\mathbf{k}}$. NumPy's cross product needs three-dimensional vectors, so we add a zero third component to the two versors:
    """)
    return


@app.cell
def _(e_n, e_t, np, t):
    _zeros = np.zeros((len(t), 1))
    e_b = np.cross(np.hstack((e_t, _zeros)), np.hstack((e_n, _zeros)))

    print("Distinct values of e_b along the path:", np.unique(e_b.round(6), axis=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A function for any path

    The steps above work for any planar or spatial trajectory sampled at a constant rate, so let's collect them in a function we can reuse:

    There is one numerical trap in Challenge 1's straight path: rounding errors can leave a tiny nonzero curvature. Normalizing that tiny vector invents a normal where none exists. The function treats curvature at or below `curvature_tol` (by default $10^{-10}\;\mathrm{m^{-1}}$) as zero and returns `NaN` for the undefined normal versor. The normal acceleration is still zero. When reconstructing acceleration there, omit the normal term rather than multiplying zero by `NaN`. This tolerance is adjustable for the scale of the trajectory; it is not a substitute for filtering noisy measurements. As in the derivation, the computation assumes nonzero speed.
    """)
    return


@app.function
def path_frame(r, dt, curvature_tol=1e-10):
    """Path-frame description of a trajectory sampled at a constant rate.

    `r` holds the positions [m] in rows, one row every `dt` [s]. Returns the
    speed [m/s], the tangential and normal versors, the curvature [1/m], and
    the tangential and normal components of the acceleration [m/s2].

    Assumes nonzero speed. Curvatures at or below `curvature_tol` [1/m]
    are treated as zero, with a NaN normal versor and zero normal
    acceleration. Omit that undefined normal term when rebuilding acceleration.
    """
    import numpy as np

    v = np.gradient(r, dt, axis=0, edge_order=2)
    speed = np.linalg.norm(v, axis=1)
    e_t = v / speed[:, np.newaxis]

    C = np.gradient(e_t, dt, axis=0, edge_order=2) / speed[:, np.newaxis]
    kappa = np.linalg.norm(C, axis=1)
    kappa = np.where(kappa <= curvature_tol, 0.0, kappa)
    e_n = np.divide(
        C,
        kappa[:, np.newaxis],
        out=np.full_like(C, np.nan),
        where=kappa[:, np.newaxis] > 0,
    )

    a_t = np.gradient(speed, dt, edge_order=2)
    a_n = kappa * speed**2

    return speed, e_t, e_n, kappa, a_t, a_n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Back to the bend

    Let's give it the athlete in lane 1: a half-circle of radius 36.8 m, run at a constant 10 m/s and sampled at 100 Hz, with the centre of the bend at the origin.
    """)
    return


@app.cell
def _(np):
    _radius, _v = 36.8, 10.0  # m, m/s
    _t = np.arange(0, np.pi * _radius / _v, 0.01)
    _phi = _v * _t / _radius
    _r = np.column_stack((_radius * np.cos(_phi), _radius * np.sin(_phi)))

    _speed, _e_t, _e_n, _kappa, _a_t, _a_n = path_frame(_r, 0.01)

    print(f"Speed:                   {_speed.mean():.2f} m/s")
    print(f"Largest |a_t|:           {np.abs(_a_t).max():.3f} m/s2")
    print(f"a_n:                     {_a_n.mean():.2f} m/s2")
    print(f"Radius of curvature:     {1 / _kappa.mean():.1f} m")
    print(f"e_n at the start:        {_e_n[0].round(3) + 0.0}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A tangential acceleration that is zero to within a few thousandths of a m/s², a normal acceleration of 2.72 m/s², a radius of curvature equal to the radius of the bend, and at the start, where the athlete is at $(36.8, 0)$, a normal versor of $(-1, 0)$: pointing straight at the centre. The fixed-frame velocity components we started with swing up and down the whole way; the path frame reduces the bend to two numbers.

    Those two numbers matter for performance. Usherwood and Wilson (2006) used exactly this requirement to explain why, on the tighter bends of indoor tracks, the athletes drawn in the inside lanes run the 200 m measurably slower. The ground has to supply the sideways force for the centripetal acceleration on top of the vertical force that supports the body, and a leg can only push so hard during the brief time it is on the ground.

    **Challenge 2.** Compute $a_n$ for lane 8 (a radius of about 45.2 m) at 10 m/s, with the formula and with `path_frame`. Then find the speed in lane 1 that has the same normal acceleration. Before you conclude that lane 1 costs that much speed, ask whether equal normal acceleration is really the right thing to hold constant. What limits a sprinter on a bend?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From clean curves to measured data

    Everything so far used positions computed from a formula, exact to about sixteen digits. Measured positions are nothing like that. A good optoelectronic motion capture system locates a marker with an error of the order of a millimetre.

    Let's add random noise with a standard deviation of 1 mm to the projectile's positions, a hundred-thousandth of the size of the path, and compute the path frame again.

    **Before you run the next cell**, predict: will the reconstructed acceleration still look like $g$, straight down? Will the radius of curvature at the top still be about 10 m?
    """)
    return


@app.cell
def _(a, dt, g, np, plt, r, t):
    _rng = np.random.default_rng(0)
    _r_noisy = r + _rng.normal(0, 0.001, r.shape)  # 1 mm of noise

    _speed, _e_t, _e_n, _kappa, _a_t, _a_n = path_frame(_r_noisy, dt)
    _a_noisy = _a_t[:, np.newaxis] * _e_t + _a_n[:, np.newaxis] * _e_n
    _mag = np.linalg.norm(_a_noisy, axis=1)
    _top = np.argmin(np.abs(t - 50 / g))

    print(f"Radius of curvature at the top: {1 / _kappa[_top]:.1f} m (true: 10.2 m)")
    print(f"|a|: median {np.median(_mag):.1f} m/s2, largest {_mag.max():.1f} m/s2 (true: {g})")

    _, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(t, _mag, color="tab:red", linewidth=1, label="1 mm of noise")
    _ax.plot(t, np.linalg.norm(a, axis=1), color="k", linewidth=2, label="no noise")
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("|a| [m/s$^2$]")
    _ax.set_title("Magnitude of the acceleration rebuilt from the path frame")
    _ax.legend(loc="upper center")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A millimetre of noise is enough to wreck it. The typical magnitude of the acceleration is off by a fifth, the worst samples are off by a factor of nearly seven, and the radius of curvature at the top is wrong by a third.

    The reason is the same one met in [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb) and [Projectile motion](https://github.com/BMClab/BMC/blob/master/notebooks/ProjectileMotion.ipynb): every finite difference divides a small difference of noisy numbers by a small time interval. The tangential versor needs one derivative of the position, and survives. The curvature and the normal versor need two, and the noise is amplified twice. Near launch and landing, where the path is almost straight, the true curvature is so small that the noise overwhelms it, and $\hat{\mathbf{e}}_n$ occasionally points the wrong way altogether.

    In practice, positions are low-pass filtered before they are differentiated; see the notebooks on [data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) and [residual analysis](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb).

    **Challenge 3.** Repeat the computation with the same 1 mm of noise but with the path sampled at 25 Hz ($\Delta t$ = 0.04 s). Does the result get better or worse? Explain why a *lower* sampling rate can help here, when a higher one was what reduced the error of the take-off angle in the Projectile motion notebook. Is there a sampling rate that is best?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Symbolic solution (extra reading)

    When the path *is* given by a formula, we can do the whole computation exactly with [SymPy](http://www.sympy.org), the symbolic mathematics package for Python. Below we define a Cartesian coordinate system, `N`, and the symbols for time and gravity. We keep $g$ as a symbol instead of 9.81, so the results hold for any gravity and SymPy's simplifications have exact expressions to work with.
    """)
    return


@app.cell
def _():
    import sympy as sym
    from sympy.vector import CoordSys3D

    N = CoordSys3D("N")
    t_s, g_s = sym.symbols("t g", positive=True)
    return N, g_s, sym, t_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The position vector of the projectile:
    """)
    return


@app.cell
def _(N, g_s, t_s):
    r_s = (10 * t_s + 100) * N.i + (-g_s / 2 * t_s**2 + 50 * t_s + 100) * N.j
    r_s
    return (r_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Its velocity and speed:
    """)
    return


@app.cell
def _(r_s, sym, t_s):
    v_s = sym.diff(r_s, t_s)
    speed_s = sym.sqrt(v_s.dot(v_s))
    v_s
    return speed_s, v_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The tangential versor:
    """)
    return


@app.cell
def _(speed_s, v_s):
    et_s = v_s / speed_s
    et_s
    return (et_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The curvature vector, the curvature and the normal versor:
    """)
    return


@app.cell
def _(et_s, speed_s, sym, t_s):
    C_s = sym.simplify(sym.diff(et_s, t_s) / speed_s)
    kappa_s = sym.simplify(sym.sqrt(C_s.dot(C_s)))
    en_s = sym.simplify(C_s / kappa_s)
    en_s
    return en_s, kappa_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The radius of curvature at the top of the flight, $t = 50/g$:
    """)
    return


@app.cell
def _(g_s, kappa_s, sym, t_s):
    sym.simplify(1 / kappa_s.subs(t_s, 50 / g_s))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Exactly $100/g$, the $10.19$ m found numerically. Now the acceleration, assembled from its tangential and normal components:
    """)
    return


@app.cell
def _(en_s, et_s, kappa_s, speed_s, sym, t_s):
    a_s = sym.diff(speed_s, t_s) * et_s + speed_s**2 * kappa_s * en_s
    sym.simplify(a_s)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two long, time-varying expressions simplify to $-g\,\hat{\mathbf{j}}$. That is the symbolic version of the plot of green arrows above, and this time it is exact.

    Finally, let's check the numerical versors against the exact ones, evaluating the symbolic expressions at the sampled instants with `sympy.lambdify`:
    """)
    return


@app.cell
def _(N, e_n, e_t, en_s, et_s, g, g_s, np, r, sym, t, t_s):
    def _evaluate(vec):
        """Evaluate the x and y components of a symbolic vector at the instants t."""
        _f = sym.lambdify(t_s, [vec.dot(N.i).subs(g_s, g), vec.dot(N.j).subs(g_s, g)])
        return np.column_stack([np.broadcast_to(c, t.shape) for c in _f(t)])

    _et_exact, _en_exact = _evaluate(et_s), _evaluate(en_s)

    print(f"Largest difference in e_t: {np.abs(_et_exact - e_t).max():.1e}")
    print(f"Largest difference in e_n: {np.abs(_en_exact - e_n).max():.1e}")
    plot_path_vectors(
        r,
        [_et_exact, _en_exact],
        [r"$\hat{\mathbf{e}}_t$", r"$\hat{\mathbf{e}}_n$"],
        step=33,
        scale=10,
        title="Path basis from the symbolic solution",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Agreement to several decimal places, as it should be with clean data sampled at 100 Hz. The symbolic route is exact but needs a formula; the numerical one works on any data but inherits every error in it. In biomechanics you will nearly always be on the numerical side, which is why the previous section matters more than this one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to the movement you chose in Challenge 0. Is the body accelerating? Which component of the acceleration, $a_t$ or $a_n$, is doing the work, and in which direction does it point?
    2. Why can the acceleration never have a component along $\hat{\mathbf{e}}_b$?
    3. A sprinter accelerates out of the blocks on the straight. Which of the three path versors can you define? What changes when they reach the bend?
    4. $a_n = \kappa\,\Vert\vec{\mathbf{v}}\Vert^2$ can never be negative. Where did the information about which way the path turns go?
    5. You have marker data at 200 Hz for a hand during a reaching movement. Rank the speed, $\hat{\mathbf{e}}_t$, $\hat{\mathbf{e}}_n$ and the radius of curvature from most to least trustworthy, and explain the ranking.
    6. The [polar basis](https://github.com/BMClab/BMC/blob/master/notebooks/PolarBasis.ipynb) is another basis that moves with the particle. What is each of the two bases attached to? For a particle on a circle centred at the origin, how are they related?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Obtain the vectors $\hat{\mathbf{e}}_n$ and $\hat{\mathbf{e}}_t$ for problem 18.1.1 of Ruina and Pratap's book.

    2. Solve problem 18.1.9 of Ruina and Pratap's book.

    3. Write a Python program to solve problem 18.1.10 of Ruina and Pratap's book (only the part about $\hat{\mathbf{e}}_n$ and $\hat{\mathbf{e}}_t$).

    4. Solve problems 1.13, 1.15 and 1.16 of Rade's book.

    5. A track cyclist rides a velodrome bend of radius 25 m at 15 m/s while slowing down at 1 m/s².<br>
       a. Calculate the tangential and normal components of the acceleration.<br>
       b. Calculate the magnitude of the acceleration and the angle between it and $\hat{\mathbf{e}}_n$.<br>
       c. Velodrome bends are steeply banked. Explain what the banking does for the cyclist, using the direction of the acceleration you found.

    6. Return to the movement you chose in Challenge 0. Estimate its speed and the radius of curvature of its path, and calculate its normal acceleration. Compare it with $g$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read pages 932–971 of chapter 18 of [Ruina and Pratap's book](http://ruina.tam.cornell.edu/Book/index.html) about path basis vectors.
    - [Polar basis](https://github.com/BMClab/BMC/blob/master/notebooks/PolarBasis.ipynb) — another basis that moves with the particle.
    - [Angular kinematics in a plane](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb) — circular motion, where $\rho$ is constant and $a_n = \omega^2 r$.
    - [Projectile motion](https://github.com/BMClab/BMC/blob/master/notebooks/ProjectileMotion.ipynb) — the path used in the example.
    - [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) — what to do before differentiating measured positions.

    ### Video lectures on the Internet

    - [Path Vectors](https://eaulas.usp.br/portal/video?idItem=7800)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Rade D (2017) [Cinemática e Dinâmica para Engenharia](https://www.grupogen.com.br/e-book-cinematica-e-dinamica-para-engenharia). Grupo GEN.
    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.
    - Usherwood JR, Wilson AM (2006) [Accounting for elite indoor 200 m sprint results](https://royalsocietypublishing.org/rsbl/article-abstract/2/1/47/63751/Accounting-for-elite-indoor-200-m-sprint-results). Biology Letters, 2(1), 47–50.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
