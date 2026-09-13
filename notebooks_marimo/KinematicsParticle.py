import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kinematics of a particle

    > Marcos Duarte, Renato Naville Watanabe,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    This notebook builds the vocabulary of kinematics — position, displacement, velocity, acceleration — and then spends that vocabulary on a real measurement: the fastest 100 m ever run by a human.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and getting it wrong is the most useful thing that can happen to you here.

    The mathematics is deliberately plain. Everything in the first half is one idea applied twice: velocity is how fast position changes, and acceleration is how fast velocity changes. Almost all the difficulty in real biomechanics comes not from those definitions but from measuring them well.

    **Challenge 0.** Before you begin, name one movement whose *speed* you would like to know: a sprint, a bicycle kick, a reaching movement, a car, an animal. Write down how you would measure it with the equipment you actually have. Keep it nearby; the last section asks you to come back to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A measurement worth understanding

    Here is a plot of Usain Bolt's world-record 100 m final in Berlin, 2009, measured with a laser radar. The horizontal axis is where he was on the track; the vertical axis is how fast he was moving at that place.

    <div style="background-color:#F1F1F1;border:1px solid black;padding:10px;">
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/usain_bolt_berlin2009.png?raw=1" width=700 alt="Momentary velocity vs location for Usain Bolt"/><figcaption><center><br><i><b>Figure. Momentary velocity vs location for Usain Bolt in the men's 100 m final at the IAAF World Championships in Athletics, Berlin 2009. This measurement represents the velocity of the body (considered as a particle) and was measured with a laser radar. From <a href="http://www.meathathletics.ie/devathletes/pdf/Biomechanics%20of%20Sprints.pdf">Graubner and Nixdorf (2011)</a>.</b></i></center></figcaption></figure>
     </div>

    Look at it for a moment before reading on. A few things in that curve are worth noticing, and each one is a kinematic statement:

    - The velocity is not constant. He is still accelerating well past the halfway mark.
    - The velocity peaks somewhere around 60–70 m and then *falls*. The fastest human on record was slowing down when he crossed the line.
    - The curve is smooth, and its steepness at each point is itself a meaningful quantity.

    By the end of this notebook you will have reconstructed a coarse version of this plot yourself, from nothing but the ten split times published for that race.

    **Guiding questions 0.**

    1. The vertical axis is velocity and the horizontal axis is position — not time. Is this a graph you could read off a stopwatch?
    2. Roughly what is his top speed, in m/s? Convert it to km/h. Does the number surprise you?
    3. If he is slowing down over the last 30 m, why does nobody coach sprinters to "just keep accelerating"?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    Two libraries carry most of this notebook: NumPy for the numbers and Matplotlib for the plots. We will add SymPy later, when we let the computer do the calculus for us.
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
    ## Why mechanics, and why kinematics first

    **A good knowledge of mechanics is a necessary condition, although not a sufficient one, to master biomechanics.** For that reason we will review the principles of classical mechanics in the context of biomechanics rather than assume them.

    As argued in the notebook [Introduction to Biomechanics](https://github.com/BMClab/BMC/blob/master/notebooks/Biomechanics.ipynb), we begin with the branch of classical mechanics whose quantities are easiest to measure on a living system: kinematics. You can film someone sprinting with a phone. Measuring the force in their Achilles tendon while they do it is an entirely different undertaking. That asymmetry is why nearly every biomechanics curriculum, and this collection, starts here.

    There are relevant cases in the study of human movement where modeling the whole body, or one of its segments, as a *particle* is all you need. Performance in the 100 m race is one. The spatial and temporal description of a movement pattern is another. So is the minimum-jerk conjecture about how voluntary movements are planned.

    The book [*Introduction to Statics and Dynamics*](http://ruina.tam.cornell.edu/Book/index.html), by Andy Ruina and Rudra Pratap, is an excellent reference — rigorous and yet didactic — and freely available. Most of the content of this notebook is covered in its chapter 12, and in chapter 1 of Rade's book. Its preface and first chapter are a genuinely good read on *how* to study mechanics; do read them.

    **Guiding questions 1.**

    1. For the movement you chose in Challenge 0, which is easier to obtain: its motion, or the forces that caused the motion?
    2. Would treating the moving body as a single point throw away something you care about?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kinematics

    **Kinematics** is the branch of classical mechanics that describes the motion of objects without consideration of the causes of motion ([Wikipedia](http://en.wikipedia.org/wiki/Kinematics)). Kinematics of a particle is that description when the object is treated as a particle.

    A particle, as a physical object, does not exist in nature. It is either a simplification adopted to understand the motion of a body, or a conceptual definition such as the center of mass of a system of objects. Bolt is not a particle: his arms swing, his trunk rotates, his foot strikes the ground. But the laser radar in the figure above tracked essentially one point on his body, and for the question *how fast did he run* that was enough.

    This is the first modeling decision of biomechanics, and it is made silently far too often: **deciding what you are willing to ignore**.

    **Challenge 1.** Give one question about sprinting that the particle model can answer, and one it cannot. For the second, what would you have to add to the model?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scalars and vectors

    Some mechanical quantities in kinematics — position and its derivatives — are vectors. Others, such as time and distance, are scalars.

    A **vector** in mechanics is a physical quantity with magnitude *and* direction, which also satisfies some elementary vector arithmetic. A **scalar** is fully expressed by a magnitude, a number, alone.

    The distinction is not pedantry. "He moved 100 m" and "he moved 100 m north" are different claims, and only one of them lets you say where he ended up.

    For a review of scalars and vectors, see chapter 1 of [Ruina and Pratap's book](http://ruina.tam.cornell.edu/Book/index.html). For how to work with them in Python, see the notebook [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Position

    Consider a point in the three-dimensional Euclidean space, described in a Cartesian coordinate system. See the notebook [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb) for an introduction to coordinate systems in mechanics and biomechanics.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/vector3Dijk.png?raw=1" width=350 alt="Position vector in a Cartesian coordinate system"/></center><figcaption><center><i>Figure. Representation of a point $\mathbf{P}$ and its position vector $\overrightarrow{\mathbf{r}}$ in a Cartesian coordinate system. The versors $\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}$ form a basis for this coordinate system and are usually represented in the color sequence RGB (red, green, blue) for easier visualization.</i></center></figcaption></figure>

    The position of this point can be written as a triple of values, each the coordinate along one axis, following the $\mathbf{X, Y, Z}$ convention order, which is then omitted:

    $$
    (x,\, y,\, z).
    $$

    It can also be represented by a **vector** with its origin at the origin of the coordinate system and its tip at the point:

    $$
    \overrightarrow{\mathbf{r}} = x\,\hat{\mathbf{i}} + y\,\hat{\mathbf{j}} + z\,\hat{\mathbf{k}},
    $$

    where $\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}$ are unit vectors along the axes $\mathbf{X, Y, Z}$.

    A particle that moves has coordinates that change with time, so its position vector is a function of time:

    $$
    \overrightarrow{\mathbf{r}}(t) = x(t)\,\hat{\mathbf{i}} + y(t)\,\hat{\mathbf{j}} + z(t)\,\hat{\mathbf{k}}.
    $$

    The same vector in matrix form:

    $$
    \overrightarrow{\mathbf{r}}(t) =
    \begin{bmatrix} x(t) \\ y(t) \\ z(t) \end{bmatrix},
    $$

    and the unit vectors of each Cartesian coordinate, likewise:

    $$
    \hat{\mathbf{i}} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad
    \hat{\mathbf{j}} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad
    \hat{\mathbf{k}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}.
    $$

    **Guiding questions 2.**

    1. A position is always measured *relative to something*. What is the position of Bolt's center of mass relative to the starting blocks, and what is it relative to the Earth's center?
    2. Which of those two would you use to describe the race, and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A basis

    In [linear algebra](http://en.wikipedia.org/wiki/Linear_algebra), a set of linearly independent unit vectors such as the three above — orthogonal, in the Euclidean space — able to represent any vector through a [linear combination](http://en.wikipedia.org/wiki/Linear_combination), is called a **basis**.

    A basis is the foundation of a frame of reference, and building frames of reference well is most of the practical work in three-dimensional motion analysis. We will come back to it in its own notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Displacement

    **Displacement** is the shortest distance between two positions of a particle. Being the difference between two vectors, it is itself a vector:

    $$
    \overrightarrow{\mathbf{d}} = \overrightarrow{\mathbf{r}}_2 - \overrightarrow{\mathbf{r}}_1.
    $$

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/displacement.png?raw=1" width=450 alt="Displacement between two positions"/></center><figcaption><center><i>Figure. Representation of the displacement vector $\overrightarrow{\mathbf{d}}$ between two positions $\overrightarrow{\mathbf{r}}_1$ and $\overrightarrow{\mathbf{r}}_2$.</i></center></figcaption></figure>

    Displacement is not the same thing as the distance traveled. Distance is a scalar and accumulates along the whole path; displacement only cares about the endpoints.

    **Challenge 2.** A 100 m sprinter and a 400 m runner both finish their race on the same track.

    1. What distance did each one travel?
    2. What was the magnitude of each one's displacement? (An Olympic track is 400 m around.)
    3. What was the *average velocity* of the 400 m runner over the whole race, using the definition in the next section? Does that number describe the race usefully?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Velocity

    **Velocity** is the rate of change, with respect to time, of the position of a particle.

    The average velocity between two instants is the displacement divided by the elapsed time:

    $$
    \overrightarrow{\mathbf{v}} =
    \frac{\overrightarrow{\mathbf{r}}(t_2)-\overrightarrow{\mathbf{r}}(t_1)}{t_2-t_1} =
    \frac{\Delta \overrightarrow{\mathbf{r}}}{\Delta t}.
    $$

    The instantaneous velocity is what this becomes as $\Delta t$ approaches zero, which from calculus is the first-order [derivative](http://en.wikipedia.org/wiki/Derivative) of the position vector:

    $$
    \overrightarrow{\mathbf{v}}(t) =
    \lim_{\Delta t \to 0} \frac{\Delta \overrightarrow{\mathbf{r}}}{\Delta t} =
    \lim_{\Delta t \to 0} \frac{\overrightarrow{\mathbf{r}}(t+\Delta t)-\overrightarrow{\mathbf{r}}(t)}{\Delta t} =
    \frac{\mathrm{d}\overrightarrow{\mathbf{r}}}{\mathrm{d}t}.
    $$

    For motion described with respect to an [inertial frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb), the derivative of a vector is obtained by differentiating each of its Cartesian components, because the base versors $\hat{\mathbf{i}}, \hat{\mathbf{j}}, \hat{\mathbf{k}}$ are constant:

    $$
    \overrightarrow{\mathbf{v}}(t) = \frac{\mathrm{d}\overrightarrow{\mathbf{r}}(t)}{\mathrm{d}t} =
    \frac{\mathrm{d}x(t)}{\mathrm{d}t}\hat{\mathbf{i}} +
    \frac{\mathrm{d}y(t)}{\mathrm{d}t}\hat{\mathbf{j}} +
    \frac{\mathrm{d}z(t)}{\mathrm{d}t}\hat{\mathbf{k}}.
    $$

    Or in matrix form, using Newton's dot notation for differentiation:

    $$
    \overrightarrow{\mathbf{v}}(t) =
    \begin{bmatrix} \dot x(t) \\ \dot y(t) \\ \dot z(t) \end{bmatrix}.
    $$

    Note the word *average* is doing real work in the first equation. In a measured sprint you never get the instantaneous velocity directly; you get positions at instants, and every velocity you compute is an average over some interval. The whole art is choosing that interval.

    **Guiding questions 3.**

    1. Bolt covered 100 m in 9.58 s. What was his average velocity? Look back at the figure at the top: at how many places on the track was he *actually* moving at that speed?
    2. Speed is the magnitude of velocity — a scalar. Can a particle have a constant speed and a changing velocity?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Acceleration

    **Acceleration** is the rate of change, with respect to time, of the velocity of a particle — equivalently, the second-order rate of change of its position.

    The average acceleration between two instants is:

    $$
    \overrightarrow{\mathbf{a}} =
    \frac{\overrightarrow{\mathbf{v}}(t_2)-\overrightarrow{\mathbf{v}}(t_1)}{t_2-t_1} =
    \frac{\Delta \overrightarrow{\mathbf{v}}}{\Delta t}.
    $$

    And the instantaneous acceleration is the first-order derivative of velocity, or the second-order derivative of position:

    $$
    \overrightarrow{\mathbf{a}}(t) =
    \frac{\mathrm{d}\overrightarrow{\mathbf{v}}(t)}{\mathrm{d}t} =
    \frac{\mathrm{d}^2\overrightarrow{\mathbf{r}}(t)}{\mathrm{d}t^2} =
    \frac{\mathrm{d}^2x(t)}{\mathrm{d}t^2}\hat{\mathbf{i}} +
    \frac{\mathrm{d}^2y(t)}{\mathrm{d}t^2}\hat{\mathbf{j}} +
    \frac{\mathrm{d}^2z(t)}{\mathrm{d}t^2}\hat{\mathbf{k}},
    $$

    in matrix form:

    $$
    \overrightarrow{\mathbf{a}}(t) =
    \begin{bmatrix} \ddot x(t) \\ \ddot y(t) \\ \ddot z(t) \end{bmatrix}.
    $$

    Two remarks worth carrying with you.

    Out of curiosity, see [Notation for differentiation](https://en.wikipedia.org/wiki/Notation_for_differentiation) on where the different notations came from; you will meet all of them in the literature.

    More consequentially: when the base versors themselves change in time — for instance when the basis is attached to a rotating frame of reference — the components of a vector's derivative are **not** simply the derivatives of its components. The derivative of the basis has to be accounted for as well. Everything in this notebook assumes a fixed basis; the rotating case gets its own treatment later.

    **Challenge 3.** In the figure at the top of this notebook, Bolt's velocity falls over the last 30 m. What is the sign of his acceleration there? Is he "decelerating", and is that a different thing?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Going backwards: the antiderivative

    Acceleration is the derivative of velocity, which is the derivative of position. Going the other way — from acceleration to velocity to position — is the inverse operation, the [antiderivative](http://en.wikipedia.org/wiki/Antiderivative), or integral:

    $$
    \begin{array}{l}
    \overrightarrow{\mathbf{v}}(t) = \overrightarrow{\mathbf{v}}_0 + \int \overrightarrow{\mathbf{a}}(t)\,\mathrm{d}t, \\[4pt]
    \overrightarrow{\mathbf{r}}(t) = \overrightarrow{\mathbf{r}}_0 + \int \overrightarrow{\mathbf{v}}(t)\,\mathrm{d}t.
    \end{array}
    $$

    Notice the constants $\overrightarrow{\mathbf{v}}_0$ and $\overrightarrow{\mathbf{r}}_0$. Differentiation destroys information — where the particle started, and how fast — and integration cannot invent it back. You have to supply it. Those constants are the *initial conditions*, and they are why an accelerometer alone can never tell you where something is.

    This is exactly the structure of an initial value problem; see the notebook [Ordinary differential equations](https://github.com/BMClab/BMC/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb), where the same two integrations are done numerically, step by step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three cases you should be able to write from memory

    Three trivial motions cover an enormous amount of introductory mechanics: a particle at rest, at constant speed, and at constant acceleration.

    There are two ways to derive them. Start from the position and differentiate twice, or start from the acceleration and integrate twice. Both are valid; here integrating is the more natural direction, because in each case it is the acceleration that is specified.

    **Particle at rest**

    $$
    \begin{array}{l}
    \overrightarrow{\mathbf{a}}(t) = 0, \\
    \overrightarrow{\mathbf{v}}(t) = 0, \\
    \overrightarrow{\mathbf{r}}(t) = \overrightarrow{\mathbf{r}}_0.
    \end{array}
    $$

    **Particle at constant speed**

    $$
    \begin{array}{l}
    \overrightarrow{\mathbf{a}}(t) = 0, \\
    \overrightarrow{\mathbf{v}}(t) = \overrightarrow{\mathbf{v}}_0, \\
    \overrightarrow{\mathbf{r}}(t) = \overrightarrow{\mathbf{r}}_0 + \overrightarrow{\mathbf{v}}_0 t.
    \end{array}
    $$

    **Particle at constant acceleration**

    $$
    \begin{array}{l}
    \overrightarrow{\mathbf{a}}(t) = \overrightarrow{\mathbf{a}}_0, \\
    \overrightarrow{\mathbf{v}}(t) = \overrightarrow{\mathbf{v}}_0 + \overrightarrow{\mathbf{a}}_0 t, \\
    \overrightarrow{\mathbf{r}}(t) = \overrightarrow{\mathbf{r}}_0 + \overrightarrow{\mathbf{v}}_0 t + \frac{1}{2}\overrightarrow{\mathbf{a}}_0 t^2.
    \end{array}
    $$

    Note that the first case is a special case of the second, which is a special case of the third. Only one formula is really being memorized here.

    **Before you run the next cells**, take the third case with $r_0 = 1\,\mathrm{m}$, $v_0 = 2\,\mathrm{m/s}$ and $a_0 = 4\,\mathrm{m/s^2}$, and predict where the particle is at $t = 2\,\mathrm{s}$. Write the number down.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Putting numbers in

    We will work in one dimension from here on, so the vectors become plain numbers and the arrows can be dropped.
    """)
    return


@app.cell
def _(np):
    t = np.linspace(0, 2, 101)  # 101 instants between 0 and 2 s
    r0, v0, a0 = 1, 2, 4  # [m], [m/s], [m/s^2]

    r = r0 + v0 * t + 1 / 2 * a0 * t**2

    print(f"First instants [s]:  {np.round(t[:5], 2)} ...")
    print(f"First positions [m]: {np.round(r[:5], 3)} ...")
    print(f"Position at t = {t[-1]:.0f} s: {r[-1]:.1f} m")
    return a0, r, r0, t, v0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Seeing them

    The simplest possible plot of that position first:
    """)
    return


@app.cell
def _(plt, r, t):
    plt.figure(figsize=(6, 3))
    plt.plot(t, r)
    plt.xlabel("Time [s]")
    plt.ylabel("r(t) [m]")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And now the three cases side by side, one column each, with position, velocity, and acceleration stacked in rows. This single figure is worth more than the three sets of equations above, because it makes the relationship between the rows visible: each row is the slope of the row above it.
    """)
    return


@app.function
def plot_kinematic_cases(t, r0, v0, a0):
    """Plot position, velocity and acceleration for the three cases of motion.

    Columns are a particle at rest, at constant speed, and at constant
    acceleration; rows are position, velocity and acceleration.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    tones = np.ones(np.size(t))

    _, axarr = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 7))
    plt.suptitle("Scalar kinematics of a particle", fontsize=20)

    axarr[0, 0].set_title("at rest", fontsize=14)
    axarr[0, 0].plot(t, r0 * tones, "g", linewidth=4, label="$r(t)=1$")
    axarr[1, 0].plot(t, 0 * tones, "b", linewidth=4, label="$v(t)=0$")
    axarr[2, 0].plot(t, 0 * tones, "r", linewidth=4, label="$a(t)=0$")
    axarr[0, 0].set_ylabel("r(t) [m]")
    axarr[1, 0].set_ylabel("v(t) [m/s]")
    axarr[2, 0].set_ylabel("a(t) [m/s$^2$]")

    axarr[0, 1].set_title("at constant speed", fontsize=14)
    axarr[0, 1].plot(t, r0 * tones + v0 * t, "g", linewidth=4, label="$r(t)=1+2t$")
    axarr[1, 1].plot(t, v0 * tones, "b", linewidth=4, label="$v(t)=2$")
    axarr[2, 1].plot(t, 0 * tones, "r", linewidth=4, label="$a(t)=0$")

    axarr[0, 2].set_title("at constant acceleration", fontsize=14)
    axarr[0, 2].plot(
        t,
        r0 * tones + v0 * t + 1 / 2 * a0 * t**2,
        "g",
        linewidth=4,
        label="$r(t)=1+2t+\\frac{1}{2}4t^2$",
    )
    axarr[1, 2].plot(t, v0 * tones + a0 * t, "b", linewidth=4, label="$v(t)=2+4t$")
    axarr[2, 2].plot(t, a0 * tones, "r", linewidth=4, label="$a(t)=4$")

    for i in range(3):
        axarr[2, i].set_xlabel("Time [s]")
        for j in range(3):
            axarr[i, j].set_ylim((-0.2, 10))
            axarr[i, j].legend(
                loc="upper left", frameon=True, framealpha=0.9, fontsize=14
            )

    plt.subplots_adjust(hspace=0.09, wspace=0.07)
    plt.show()


@app.cell
def _(a0, r0, t, v0):
    plot_kinematic_cases(t, r0, v0, a0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Guiding questions 4.**

    1. In the middle column, the position line is straight and the velocity line is flat. State the relationship between those two facts in one sentence.
    2. In the right column, the position is a parabola. What shape is its velocity, and why?
    3. Cover the top row with your hand. Could you reconstruct it from the two rows below? What would you still be missing?

    **Challenge 4.** Change $a_0$ to a negative value and re-run the two cells above. Which panels change, and which do not? At what instant does the particle turn around, and can you read that instant off the *velocity* panel without computing anything?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Letting the computer do the algebra

    We integrated those cases by hand. [SymPy](http://www.sympy.org/en/index.html), a Python library for symbolic mathematics, can do it for us — and unlike NumPy it returns *expressions*, not arrays of numbers.

    Let's integrate the constant-acceleration case. First, declare the symbols:
    """)
    return


@app.cell
def _():
    import sympy as sym

    sym.init_printing(use_latex="mathjax")  # print pretty symbols

    t_s = sym.symbols("t", real=True)
    r0_s, v0_s, a0_s = sym.symbols("r0, v0, a0", real=True, constant=True)
    return a0_s, r0_s, sym, t_s, v0_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Integrating the constant acceleration $a_0$ gives the velocity. Note that the constant of integration — the initial velocity — has to be added by hand; SymPy will not guess your initial conditions any more than nature will.
    """)
    return


@app.cell
def _(a0_s, sym, t_s, v0_s):
    v_s = sym.integrate(a0_s, t_s) + v0_s
    v_s
    return (v_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And integrating that velocity, again adding the constant, gives the position:
    """)
    return


@app.cell
def _(r0_s, sym, t_s, v_s):
    r_s = sym.integrate(v_s, t_s) + r0_s
    r_s
    return (r_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That is the third case from memory, derived rather than recalled.

    A symbolic expression can also be plotted over a range, once the symbols have been substituted by numbers:
    """)
    return


@app.cell
def _(a0_s, r0_s, r_s, t_s, v0_s):
    from sympy.plotting import plot

    r_num = r_s.subs({r0_s: 1, v0_s: 2, a0_s: 4})

    plot(
        r_num,
        (t_s, 0, 2),
        xlim=(0, 2),
        ylim=(0, 10),
        axis_center=(0, 0),
        line_color="g",
        xlabel="Time [s]",
        ylabel="r(t) [m]",
        legend=True,
        title="Scalar kinematics of a particle at constant acceleration",
        backend="matplotlib",
        size=(5, 3),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 5.** Use `sym.diff` to go the other way: differentiate `r_s` twice with respect to `t_s` and confirm you recover $a_0$. Then try the same starting from a position that is *not* polynomial, such as $r(t) = A\sin(\omega t)$, and inspect the velocity and acceleration you get. What is the relationship between the position and the acceleration in that case?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kinematics of human movement: the 100 m race

    Now we spend all of this on a real measurement.

    The study of the biomechanics of the 100 m dash is a case where the analysis of the human body reduces honestly to the analysis of a particle. A technical report with the kinematic data for Usain Bolt's world record can be downloaded from the [website for research projects](http://www.iaaf.org/development/research) of the International Association of Athletics Federations; [here is a direct link to the report](http://www.iaaf.org/download/download?filename=76ade5f9-75a0-4fda-b9bf-1b30be6f60d2.pdf&urlSlug=1-biomechanics-report-wc-berlin-2009-sprint). The table below shows the data for the three medalists.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/Berlin2009_100m.png?raw=1" width=700 alt="Partial times of the 100 m race at Berlin 2009"/></center><figcaption><center><i>Figure. Data from the three medalists of the 100 m dash in Berlin, 2009 (<a href="http://www.iaaf.org/download/download?filename=76ade5f9-75a0-4fda-b9bf-1b30be6f60d2.pdf&urlSlug=1-biomechanics-report-wc-berlin-2009-sprint">IAAF report</a>).</i></center></figcaption></figure>

    The column **RT** is the reaction time of each athlete. The IAAF rule on it is strict: any athlete with a reaction time below 100 ms is disqualified, on the grounds that no one can respond to the gun that fast, so they must have anticipated it. See [Reaction Times and Sprint False Starts](http://condellpark.com/kd/reactiontime.htm) for a discussion of the rule, and you can measure your own at [humanbenchmark.com](http://www.humanbenchmark.com/tests/reactiontime).

    Note what the reaction time does to the kinematics: the clock starts at the gun, but the athlete does not. Roughly 0.15 s of Bolt's 9.58 s were spent motionless.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reconstructing the race from ten numbers

    Graubner and Nixdorf (2011) published the 10 m split times for that final:

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/Berlin2009_100m_10.png?raw=1" width=600 alt="10-m split times of the 100 m race at Berlin 2009"/></center><figcaption><center><i>Figure. Split times every 10 m for the three medalists of the 100 m dash in Berlin, 2009 (Graubner and Nixdorf, 2011).</i></center></figcaption></figure>

    That is all we have: eleven positions, eleven instants. No velocity was measured, and no acceleration. Everything else has to be computed with the two definitions from earlier in this notebook — each velocity as $\Delta r / \Delta t$ over one 10 m segment, and each acceleration as $\Delta v / \Delta t$ between consecutive velocities.

    **Before you run the next cells**, predict three numbers and write them down:

    1. Bolt's average velocity over the whole race, in m/s.
    2. His *peak* velocity — how much higher than the average?
    3. His peak acceleration, in m/s². For scale, gravity is $9.8\,\mathrm{m/s^2}$.
    """)
    return


@app.function
def bolt_splits():
    """Return the 10-m split times of Usain Bolt's 100-m world record.

    Berlin, 2009, from Graubner and Nixdorf (2011). Returns the instants [s]
    at which he crossed each 10-m mark and the corresponding positions [m],
    both starting at the gun (t = 0 s, r = 0 m).
    """
    import numpy as np

    time = np.array(
        [0.00, 1.88, 2.88, 3.78, 4.64, 5.47, 6.29, 7.10, 7.92, 8.74, 9.58]
    )
    position = np.arange(0, 101, 10, dtype=float)
    return time, position


@app.function
def split_kinematics(time, position):
    """Average velocity and acceleration from split times, by finite differences.

    Each velocity is the average over one segment, so it is attributed to the
    midpoint in time of that segment; each acceleration likewise sits at the
    midpoint between two velocities.

    Returns (t_v, velocity, t_a, acceleration).
    """
    import numpy as np

    velocity = np.diff(position) / np.diff(time)
    t_v = (time[:-1] + time[1:]) / 2

    acceleration = np.diff(velocity) / np.diff(t_v)
    t_a = (t_v[:-1] + t_v[1:]) / 2

    return t_v, velocity, t_a, acceleration


@app.cell
def _():
    bolt_time, bolt_position = bolt_splits()
    bolt_tv, bolt_velocity, bolt_ta, bolt_acceleration = split_kinematics(
        bolt_time, bolt_position
    )

    print("Segment      Δt [s]   v [m/s]   v [km/h]")
    for _i, _v in enumerate(bolt_velocity):
        _dt = bolt_time[_i + 1] - bolt_time[_i]
        print(
            f"{bolt_position[_i]:3.0f}-{bolt_position[_i + 1]:3.0f} m"
            f"{_dt:9.2f}{_v:10.2f}{3.6 * _v:11.1f}"
        )

    print()
    print(f"Average velocity: {bolt_position[-1] / bolt_time[-1]:.2f} m/s "
          f"({3.6 * bolt_position[-1] / bolt_time[-1]:.1f} km/h)")
    print(f"Peak velocity:    {bolt_velocity.max():.2f} m/s "
          f"({3.6 * bolt_velocity.max():.1f} km/h), "
          f"around t = {bolt_tv[bolt_velocity.argmax()]:.1f} s")
    print(f"Peak acceleration:{bolt_acceleration.max():7.2f} m/s², "
          f"around t = {bolt_ta[bolt_acceleration.argmax()]:.1f} s")
    return (
        bolt_acceleration,
        bolt_position,
        bolt_ta,
        bolt_time,
        bolt_tv,
        bolt_velocity,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A few things in that table deserve to be read slowly.

    The first segment is by far the slowest, and it is not close: about 5.3 m/s against a peak near 12.3 m/s. Part of that is the reaction time sitting inside it, but most of it is simply that he started from rest. Sprinting is, for the first 30 m, mostly a problem of acceleration.

    The peak velocity of roughly 12.3 m/s is about 44 km/h, and it is reached somewhere past 60 m — remarkably late. Then it falls. Bolt was *slowing down* over the last 20 m of the fastest 100 m ever run, and so was everyone else in that final.

    His average over the whole race, about 10.4 m/s, describes no particular moment of it. It is the number that gets printed, and it is the least informative one in the table.

    Now the same numbers as plots, including the velocity-against-position panel that reproduces, coarsely, the figure that opened this notebook.
    """)
    return


@app.function
def plot_sprint(time, position, t_v, velocity, t_a, acceleration, name="Bolt"):
    """Plot position, velocity and acceleration of a sprint from split times."""
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(2, 2, figsize=(11, 6))

    axs[0, 0].plot(time, position, "go-", linewidth=2)
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Position [m]")
    axs[0, 0].set_title("Position")

    axs[0, 1].plot(t_v, velocity, "bo-", linewidth=2)
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Velocity [m/s]")
    axs[0, 1].set_title("Velocity")

    axs[1, 0].plot(t_a, acceleration, "ro-", linewidth=2)
    axs[1, 0].axhline(0, color="k", linewidth=1, linestyle=":")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Acceleration [m/s$^2$]")
    axs[1, 0].set_title("Acceleration")

    position_v = (position[:-1] + position[1:]) / 2
    axs[1, 1].plot(position_v, velocity, "bo-", linewidth=2)
    axs[1, 1].set_xlabel("Position [m]")
    axs[1, 1].set_ylabel("Velocity [m/s]")
    axs[1, 1].set_title("Velocity vs position")

    plt.suptitle(
        f"Kinematics of the 100-m dash, {name}, Berlin 2009", y=1.0, fontsize=16
    )
    plt.tight_layout()
    plt.show()


@app.cell
def _(
    bolt_acceleration,
    bolt_position,
    bolt_ta,
    bolt_time,
    bolt_tv,
    bolt_velocity,
):
    plot_sprint(
        bolt_time,
        bolt_position,
        bolt_tv,
        bolt_velocity,
        bolt_ta,
        bolt_acceleration,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the bottom-right panel with the laser-radar figure at the top of this notebook. The shape is right — a fast rise, a flat maximum past halfway, a decline to the finish — but ours is built from ten averages and theirs from a continuous measurement.

    That comparison is the honest lesson of this section, and it is worth stating plainly. **Every derivative we computed is a finite difference over a coarse interval, and each one is noisier than the one before it.** The position data are excellent. The velocities, averaged over roughly 0.8 s each, are already smoothed versions of the truth — they cannot show the within-stride fluctuation that the laser radar records. The accelerations, being differences of those differences, are the roughest estimate in the figure: the acceleration panel jumps around in a way that no human body actually did.

    The error is not only random, either. Look at the peak acceleration the code printed: a little over 3 m/s². A sprinter leaving the blocks actually reaches something closer to 8-10 m/s², and our estimate misses it by a factor of three — not because the arithmetic is wrong, but because the first split spans 1.88 s and the whole explosive start is averaged away inside it. **A finite difference cannot resolve anything faster than its own interval.** If you had predicted a number near gravity before running the cell, you were closer to the truth than the data were.

    Neither problem is a flaw in our arithmetic. Differentiating measured data amplifies whatever error is in it, every time you do it, and smooths away whatever is briefer than the sampling interval. It is the reason the notebooks on [data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) and [residual analysis](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb) exist, and the reason experienced people are suspicious of any acceleration curve computed from position data.

    **Challenge 6.** Work these with the functions above.

    1. Was Bolt's acceleration constant over any part of the race? Look at the acceleration panel, then at the velocity panel, and decide which one gives you a more trustworthy answer.
    2. Fit a straight line to the first three velocity points and use its slope as an estimate of the initial acceleration. How does it compare to the peak in the acceleration panel? Which would you report?
    3. The first segment includes about 0.15 s of reaction time. Recompute the first velocity using the actual movement time instead of the split time. How much does it change?
    4. Add Gay (9.71 s) and Powell (9.84 s) from the split table above and plot all three velocity curves together. Where in the race was the race actually decided?
    5. Bolt's peak velocity is around 12.3 m/s. If he could have held exactly that from the gun, what would his time have been? What does the gap between that and 9.58 s tell you about where sprint performance is won?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems. These connect this notebook to the rest of the collection.

    1. You are given position data and you need acceleration. You are given acceleration data and you need position. Which of the two situations is more dangerous, and why?
    2. A marker on a walking person's heel is tracked at 100 Hz. What is the shortest event you could hope to resolve in its velocity?
    3. Why is the average velocity of a 100 m sprinter a poor description of the race, while the average velocity of a marathon runner is a reasonable one?
    4. An accelerometer on a phone records a person jumping. What two pieces of information, not in the accelerometer signal, do you need before you can say how high they jumped?
    5. In which of these is the particle model adequate, and in which does it break down: the flight of a long jumper, the rotation of a diver, the path of a wheelchair, the swing of a golf club?
    6. Displacement and distance coincide for the 100 m dash. Name a movement in your own field where they differ substantially, and say which one you would report.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Answer the 12 questions of the [Khan Academy's test on one-dimensional motion](https://www.khanacademy.org/science/ap-physics-1/ap-one-dimensional-motion/test/ap-one-dimensional-motion-unit-test?modal=1).

    2. Consider the data for the three medalists of the 100 m dash in Berlin, 2009, shown in the 20 m split table above.<br>
       a. Calculate the average velocity and acceleration.<br>
       b. Plot the graphs for the displacement, velocity, and acceleration versus time.<br>
       c. Plot the graphs of velocity and acceleration versus partial distance (every 20 m).<br>
       d. Calculate the average velocity and average acceleration, and the instants and values of the peak velocity and peak acceleration.

    3. Repeat the calculations of problem 2 with the 10 m split times of Graubner and Nixdorf, reusing the functions written above, and compare the results. Which quantities change most when the sampling interval is halved — position, velocity, or acceleration? Relate your answer to the discussion of finite differences above.

    4. On an Olympic running track, runners A and B start running on the first lane (the inside lane) from the same position on the track, but in opposite directions. If $\lVert v_A \rVert = 4$ m/s and $\lVert v_B \rVert = 6$ m/s, how far from the starting line will the runners meet?

    5. A body attached to a spring has its position (in cm) described by the equation $x(t) = 2\sin(4\pi t + \pi/4)$.<br>
       a. Calculate the equations for the body's velocity and acceleration.<br>
       b. Plot the position, velocity, and acceleration in the interval $[0, 1]$ s.

    6. The rectilinear motion of a particle is given by $x(t) = -12t^3 + 15t^2 + 5t + 2$ [s; m]. Calculate:<br>
       a. Velocity and acceleration of the particle as functions of time. Solution: $v(t) = -36t^2 + 30t + 5$ [s; m/s], $a(t) = -72t + 30$ [s; m/s²].<br>
       b. Total distance traveled by the particle in the interval $0 \leq t \leq 4$ s. Solution: $\Delta_{0-4} = 524.02$ m.<br>
       c. Maximum value of the velocity magnitude reached by the particle in the interval $0 \leq t \leq 4$ s. Solution: $\lVert v_{max} \rVert = 451.0$ m/s.<br>
       d. Plots of the position, velocity and acceleration of the particle in the interval $0 \leq t \leq 4$ s.

    7. A stone is released from the opening of a well, and the noise of its fall reaching the bottom is heard 4 seconds later. Knowing that the speed of sound in air is 340 m/s, determine the depth of the well. Solution: $h = 70.55$ m.

    8. The position of a particle is given by $\overrightarrow{\mathbf{r}}(t) = t^2\,\hat{\mathbf{i}} + e^{t}\,\hat{\mathbf{j}}$.<br>
       a. Calculate the velocity and acceleration of the particle as functions of time.<br>
       b. Draw the path of the particle and show the vectors $\overrightarrow{\mathbf{v}}(t)$ and $\overrightarrow{\mathbf{a}}(t)$ at $t = 1$ s.

    9. Sometimes all you have is an image of the data you want. The first figure in this notebook contains velocity-versus-position data for Bolt's sprint at a much higher resolution than the split times we used. Software exists to recover numbers from such images by identifying points of interest, automatically where possible and manually where not; [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) is one such tool. Use it to extract the data from that figure and compare the result with the velocity curve you computed. It will not be easy — that is part of the lesson.

    10. There is good free software for the kinematic analysis of human motion, among them [Kinovea](http://www.kinovea.org/), [Tracker](http://www.cabrillo.edu/~dbrown/tracker/), and [SkillSpector](http://video4coach.com/index.php?option=com_content&task=view&id=13&Itemid=45). Visit their websites and work out which biomechanical applications each is suited to.

    11. Return to the movement you chose in Challenge 0. Write one paragraph: which kinematic quantity would answer your question, how you would measure it, how many times per second you would have to sample it, and whether you would have to differentiate or integrate to get there.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More examples

    - From Ruina and Pratap's book, study samples 12.2 and 12.3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read the preface and first chapter of [Ruina and Pratap's book](http://ruina.tam.cornell.edu/Book/index.html) on how someone should study mechanics.
    - [Spatial and temporal characteristics](https://github.com/BMClab/BMC/blob/master/notebooks/SpatialTemporalCharacteristcs.ipynb) — how simple measurements of spatial and temporal kinematic variables describe human gait.
    - [The minimum jerk hypothesis](https://github.com/BMClab/BMC/blob/master/notebooks/MinimumJerkHypothesis.ipynb) — the conjecture that voluntary movements are organized along the smoothest trajectory possible, another case where the particle model earns its keep.
    - [Ordinary differential equations](https://github.com/BMClab/BMC/blob/master/notebooks/OrdinaryDifferentialEquation.ipynb) — the numerical version of the integrations we did symbolically here.
    - [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) and [residual analysis](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb) — what to do about the noise amplification we ran into when differentiating the split times.
    - [Projectile motion](https://github.com/BMClab/BMC/blob/master/notebooks/ProjectileMotion.ipynb) — constant acceleration in two dimensions.

    ### Video lectures on the Internet

    - Khan Academy: [One-dimensional motion](https://www.khanacademy.org/science/ap-physics-1/ap-one-dimensional-motion)
    - [Powers of 10, Units, Dimensions, Uncertainties, Scaling Arguments](https://youtu.be/GtOGurrUPmQ)
    - [1D Kinematics — Speed, Velocity, Acceleration](https://youtu.be/q9IWoQ199_o)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Graubner R, Nixdorf E (2011) [Biomechanical Analysis of the Sprint and Hurdles Events at the 2009 IAAF World Championships in Athletics](http://www.meathathletics.ie/devathletes/pdf/Biomechanics%20of%20Sprints.pdf). [New Studies in Athletics](http://www.iaaf.org/development/new-studies-in-athletics), 1/2, 19-53.
    - Krzysztof M, Mero A (2013) [A Kinematics Analysis Of Three Best 100 M Performances Ever](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3661886/). Journal of Human Kinetics, 36, 149-160.
    - [Research Projects](http://www.iaaf.org/development/research) from the International Association of Athletics Federations.
    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
