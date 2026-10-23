# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "sympy>=1.13"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kinematic chain in a plane (2D)

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Your muscles do not move your hand. They rotate joints: the shoulder, the elbow, the wrist. The hand goes wherever the combination of those rotations happens to put it. And yet almost every task you perform is defined by where the hand must go, not by what the joints must do. Reaching for a cup, the nervous system has to translate one description into the other, continuously, in both directions.

    The mathematical model behind that translation is the **kinematic chain**: rigid links connected by joints. This notebook builds it up from one link to two, in a plane, using only trigonometry and calculus, with [SymPy](http://sympy.org/) doing the algebra. It covers how to go from joint angles to the position of the endpoint and back, how joint velocities become endpoint velocities through a matrix called the Jacobian, and why the endpoint accelerates in ways that no single joint does.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a result *before* the code prints it. Make the prediction; a wrong guess that the code then corrects teaches more than a right one.

    Along the way there are also **Exercises**, each followed by a collapsed solution. Try the exercise first, and open the solution only to check your answer.

    You will need vectors from [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py), derivatives from [Kinematics of a particle](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_particle.py), and the tangential and normal accelerations from [Path frame](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/path_frame.py).

    **Challenge 0.** Put your hand flat on a table and slide your fingertip along a straight line, from close to your chest to almost arm's length, without moving your trunk. Pay attention to your shoulder and your elbow. Does each of them rotate in one direction only? At a constant rate? Write down your answers; the checkpoint questions ask you to come back to them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reaching in a straight line

    When people move the hand from one point to another on a table, the hand travels along a nearly straight path, and its speed rises and falls in a smooth, bell-shaped curve (Morasso, 1981). Flash and Hogan (1985) showed that these paths are well described by the *minimum-jerk* trajectory, the smoothest possible movement of the hand between the two points (see [The minimum-jerk hypothesis](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/minimum_jerk_hypothesis.py)).

    A straight, smooth path is simple to describe for the hand. It is not obviously simple for the joints. The shoulder and the elbow are hinges; each of them can only swing the parts beyond it around a circle. Somehow, two circular motions must add up to a straight line.

    By the end of this notebook you will compute exactly what the shoulder and elbow have to do for such a reach, and see that the simplicity of the hand's movement is not shared by the joints.

    **Guiding questions 0.**

    1. For the hand to move in a straight line, must both joints rotate in the same direction? Must they rotate at constant angular velocities?
    2. An arm with two joints moving in a plane has two angles to set, and the hand has two coordinates. For a given point on the table, how many different arm configurations put the hand there?
    3. Suppose the elbow is locked and the shoulder rotates at a constant angular velocity. The joints do not accelerate. Does the hand?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    SymPy derives the expressions, NumPy evaluates them and Matplotlib draws the plots. SymPy prints derivatives with respect to time as $\frac{d}{dt}\theta(t)$; the function `show`, defined below, prints them instead in [Newton's notation](https://en.wikipedia.org/wiki/Notation_for_differentiation), $\dot{\theta}$, which is much easier to read.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import sympy as sym
    from sympy.vector import CoordSys3D
    from sympy.physics.mechanics import dynamicsymbols

    matplotlib.rc("axes", labelsize=12, titlesize=13)
    matplotlib.rc("xtick", labelsize=10)
    matplotlib.rc("ytick", labelsize=10)
    matplotlib.rc("legend", fontsize=10)
    return CoordSys3D, dynamicsymbols, np, plt, sym


@app.function
def show(expr, lhs=""):
    """Display a SymPy expression, with time derivatives in Newton's notation."""
    import marimo as mo
    from sympy.physics.vector.printing import vlatex

    return mo.md(rf"$$ {lhs} {vlatex(expr)} $$")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What a kinematic chain is

    <figure><a href="https://en.wikipedia.org/wiki/Kinematic_chain"><img align="right" src="https://upload.wikimedia.org/wikipedia/commons/2/2c/Modele_cinematique_corps_humain.svg" width="240" style="margin: 0 10px 0 10px" alt="human body modelled as a kinematic chain"/></a></figure>

    **A kinematic chain is an assembly of rigid bodies (links) connected by joints; it is the mathematical model for a mechanical system, which in turn can represent a biological system such as the human body** ([Wikipedia](https://en.wikipedia.org/wiki/Kinematic_chain)).

    The term chain refers to the fact that the links are constrained by their connections to other links, typically by a hinge joint, also called a pin joint or revolute joint. As a consequence of this constraint, each link of a kinematic chain in a plane rotates about its joint: the motion of a planar chain is built out of circular motions of rigid objects.

    Chapter 16 of Ruina and Pratap's book and chapter 2 of Rade's book are a good formal introduction to the circular motion of a rigid object, and sections 2.7 and 2.8 of Rade's book cover kinematic chains specifically. In this notebook, however, we do not use the formalism introduced in those chapters, the rotating reference frame and the rotation matrix; those are covered in [Rigid-body transformations in 2D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py). Here, we describe the kinematics of a chain in a Cartesian coordinate system using only trigonometry and calculus. This approach is simpler and more intuitive, but it gets too complicated for a chain with many links or in three dimensions; for such problems, rigid-body transformations are the recommended tool (see, for example, Siciliano et al. (2009)).

    We derive the kinematic properties of the chains algebraically with [SymPy](http://sympy.org/), a Python library for symbolic mathematics. SymPy has a [mechanics module](https://docs.sympy.org/latest/modules/physics/mechanics/index.html) built for deriving the equations of motion of multibody systems, but we will derive most of it ourselves, to understand the details.

    <p style="text-align: right;"><i>Figure. The human body modelled as a kinematic chain (<a href="https://en.wikipedia.org/wiki/Kinematic_chain">image from Wikipedia</a>).</i></p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Properties of kinematic chains

    The **base** of a kinematic chain is the extremity (origin) of the chain that is typically considered attached to the ground, to the body, or otherwise fixed. The **endpoint** is the other extremity, which typically can move. In robotics, the term **end-effector** is used, and it usually refers to the last link of the chain.

    In topological terms, a kinematic chain is called **open** when there is only one sequence of links connecting its two ends. Otherwise it is called **closed**, and a sequence of links forms a loop. A kinematic chain can be **serial**, **parallel**, or a **mix** of both. In a serial chain the links are connected one after the other; a serial chain is an open chain. Otherwise, the chain is parallel, or branched (for example, the hand and fingers).

    Although this definition is clear and classic in mechanics, it is not the one used by health professionals (clinicians and athletic trainers) to describe human movement. They call a set of joints and segments a closed or open kinematic (or kinetic) chain simply according to whether the distal segment, typically the foot or the hand, is fixed (closed chain) or not (open chain). The two definitions sometimes classify the same movement differently. A person standing on one leg, for example, is an open kinematic chain in mechanics, but a closed one in the clinical sense. In this text we stay consistent with mechanics, but keep the difference in mind when talking with clinicians and athletic trainers.

    Another important property of a kinematic chain is its number of **<a href="https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics)">degrees of freedom (DOF)</a>**. In mechanics, the degrees of freedom of a system are the number of independent parameters needed to define its configuration. A particle in 3D space has three DOFs, because we need three coordinates to specify its position. A rigid body in 3D space has six DOFs: three coordinates of one of its points to specify its position, and three angles to specify its orientation. A link attached to a fixed body by a hinge joint in a plane needs only one angle to define its configuration, so it has one DOF. A kinematic chain with two links in a plane has two DOFs, and so on.

    The **mobility** of a kinematic chain is its total number of degrees of freedom. The **redundancy** of a kinematic chain is its mobility minus the number of degrees of freedom of the endpoint.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The kinematics of a one-link system

    Let's start with a system made of one planar hinge joint and one link. Technically it is not a chain, but it introduces, one at a time, all the concepts we will need.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/onelink.gif?raw=1" width=350 alt="one link attached to a fixed body by a hinge joint"/></center><figcaption><center><i>Figure. One link attached to a fixed body by a hinge joint in a plane.</i></center></figcaption></figure>

    We need a Cartesian coordinate system and the symbols for time, $t$, the length of the link, $\ell$, and the joint angle, $\theta$, which is a function of time:
    """)
    return


@app.cell
def _(CoordSys3D, dynamicsymbols, sym):
    G = CoordSys3D("")
    t = sym.Symbol("t")
    ell = sym.Symbol("ell", real=True, positive=True)
    theta = dynamicsymbols("theta", real=True)  # or sym.Function("theta")(t)
    return G, ell, t, theta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using trigonometry, the position of the endpoint in terms of the joint angle and the length of the link is:
    """)
    return


@app.cell
def _(G, ell, sym, theta):
    r_p = ell * sym.cos(theta) * G.i + ell * sym.sin(theta) * G.j + 0 * G.k

    show(r_p, r"\overrightarrow{\mathbf{r}}_P =")
    return (r_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the components:
    """)
    return


@app.cell
def _(G, mo, r_p):
    _components = r_p.components

    mo.vstack([show(_components[G.i], "x_P ="), show(_components[G.j], "y_P =")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forward and inverse kinematics

    Computing the configuration of a link or a chain, including the location of the endpoint, from the joint parameters (joint angles and link lengths), as we have just done, is called [**forward or direct kinematics**](https://en.wikipedia.org/wiki/Forward_kinematics).

    If the coordinates of the endpoint are known, for example because they were measured with a motion capture system, and we want the joint angle(s), the process is called [**inverse kinematics**](https://en.wikipedia.org/wiki/Inverse_kinematics). For the one-link system:

    $$
    \theta = \arctan\left(\frac{y_P}{x_P}\right)
    $$

    with one caveat that matters in code. The ratio $y_P/x_P$ is the same for a point and for its mirror image through the joint, so the plain arctangent cannot tell $45^o$ from $-135^o$. The two-argument function `numpy.arctan2(y, x)` uses the signs of both coordinates and returns the correct angle in all four quadrants. We will meet this again, with consequences, in the two-link chain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matrix representation of the kinematics

    The algebra is easier in matrix form (and let's drop the explicit dependence on $t$ when printing):
    """)
    return


@app.cell
def _(G, r_p, sym):
    r = sym.Matrix((r_p.dot(G.i), r_p.dot(G.j)))

    show(r, r"\overrightarrow{\mathbf{r}}_P =")
    return (r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The matrix form simplifies things, but we lose some of SymPy's methods for vectors; for instance, the vector `r_p` has a method `magnitude`, and the matrix `r` does not. If you prefer, you can keep the vector representation and switch to the matrix representation only to display a variable:
    """)
    return


@app.cell
def _(G, r_p):
    show(r_p.to_matrix(G), r"\overrightarrow{\mathbf{r}}_P =")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The third element of this matrix is the $\hat{\mathbf{k}}$ component, which is zero because the movement is planar.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differential kinematics

    Differential kinematics gives the relationship between the angular velocities of the joints and the linear velocity of the endpoint. This mapping is described by a matrix, the [**Jacobian matrix**](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant), which depends on the configuration of the chain and is of great use in the study of kinematic chains. First, let's derive the velocity of the endpoint without the Jacobian; then we will see how to calculate it with the Jacobian.

    The velocity of the endpoint is the first-order derivative of its position vector, and the derivative of a vector is obtained by differentiating each of its components:

    $$
    \frac{\mathrm{d}\overrightarrow{\mathbf{r}}}{\mathrm{d}t} =
    \begin{bmatrix}
    \dfrac{\mathrm{d}x_P}{\mathrm{d}t} \\[2ex]
    \dfrac{\mathrm{d}y_P}{\mathrm{d}t}
    \end{bmatrix}
    $$

    The derivative is with respect to time, but $x_P$ and $y_P$ depend explicitly on $\theta$, and it is $\theta$ that depends on $t$: $x_P$ and $y_P$ depend on $t$ only implicitly. To calculate this type of derivative we use the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).

    > **[Chain rule](https://en.wikipedia.org/wiki/Chain_rule).** For a variable $f$ that is a function of a variable $g$, which in turn is a function of $t$, $f(g(t))$ or $(f\circ g)(t)$, the derivative of $f$ with respect to $t$ is, in [Lagrange's notation](https://en.wikipedia.org/wiki/Notation_for_differentiation):
    >
    > $$(f\circ g)^{'}(t) = f'(g(t)) \cdot g'(t)$$
    >
    > or, in [Leibniz's notation](https://en.wikipedia.org/wiki/Notation_for_differentiation):
    >
    > $$\frac{\mathrm{d}f}{\mathrm{d}t} = \frac{\mathrm{d}f}{\mathrm{d}g} \cdot \frac{\mathrm{d}g}{\mathrm{d}t}$$
    >
    > If $f$ is a function of two variables that are both functions of $t$, $f(x(t),y(t))$, the chain rule becomes:
    >
    > $$\frac{\mathrm{d}f}{\mathrm{d}t} = \frac{\partial f}{\partial x} \cdot \frac{\mathrm{d}x}{\mathrm{d}t} + \frac{\partial f}{\partial y} \cdot \frac{\mathrm{d}y}{\mathrm{d}t}$$
    >
    > where $\mathrm{d}f/\mathrm{d}t$ is the [total derivative](https://en.wikipedia.org/wiki/Total_derivative) and $\partial f / \partial x$ is a [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative).
    >
    > **[Product rule](https://en.wikipedia.org/wiki/Product_rule).** The derivative of the product of two functions is:
    >
    > $$ (f \cdot g)' = f' \cdot g + f \cdot g' $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear velocity of the endpoint

    For the planar one-link system, the linear velocity of the endpoint is:
    """)
    return


@app.cell
def _(r, t):
    v = r.diff(t)

    show(v, r"\overrightarrow{\mathbf{v}} =")
    return (v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\dot{\theta}$ is the angular velocity of the joint. It is an unknown function of time, which is why SymPy leaves it as a derivative instead of evaluating it. The magnitude, or [Euclidean norm](https://en.wikipedia.org/wiki/Vector_norm), of the vector $\overrightarrow{\mathbf{v}}$ is:

    $$
    ||\overrightarrow{\mathbf{v}}||=\sqrt{v_x^2+v_y^2}
    $$
    """)
    return


@app.cell
def _(sym, v):
    show(sym.simplify(sym.sqrt(v[0] ** 2 + v[1] ** 2)), r"||\overrightarrow{\mathbf{v}}|| =")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That is $\ell|\dot{\theta}|$: the speed of the endpoint is the length of the link times the magnitude of the angular velocity (SymPy does not know the sign of $\dot{\theta}$, so it keeps the square root of its square). We could have used SymPy's `norm` method, but the output does not simplify as nicely:
    """)
    return


@app.cell
def _(sym, v):
    show(sym.simplify(v.norm()), r"||\overrightarrow{\mathbf{v}}|| =")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The direction of $\overrightarrow{\mathbf{v}}$ is tangent to the circular trajectory of the endpoint, as the figure below shows, together with the components of the velocity.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/onelink_vel.gif?raw=1" width=350 alt="endpoint velocity of one link"/></center><figcaption><center><i>Figure. Endpoint velocity of one link attached to a fixed body by a hinge joint in a plane.</i></center></figcaption></figure>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear acceleration of the endpoint

    The acceleration of the endpoint is the second-order derivative of its position, or the first-order derivative of its velocity. Using the chain and product rules, the linear acceleration of the endpoint is:
    """)
    return


@app.cell
def _(t, v):
    acc = v.diff(t, 1)

    show(acc, r"\overrightarrow{\mathbf{a}} =")
    return (acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The expression has two types of terms in each direction: one proportional to the angular acceleration, $\ddot{\theta}$, and one proportional to the square of the angular velocity, $\dot{\theta}^{2}$.

    #### Tangential acceleration

    The term proportional to the angular acceleration, $a_t$, is always tangent to the trajectory of the endpoint (see the figure below), and its magnitude is:
    """)
    return


@app.cell
def _(acc, sym, t, theta):
    _A = theta.diff(t, 2)

    show(sym.simplify(sym.sqrt(sym.expand(acc[0]).coeff(_A) ** 2 + sym.expand(acc[1]).coeff(_A) ** 2)) * _A, "a_t =")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Centripetal acceleration

    The term proportional to the square of the angular velocity, $a_c$, always points to the joint, the centre of the circular motion (see the figure below), which is why it is called the [centripetal acceleration](https://en.wikipedia.org/wiki/Centripetal_acceleration#Tangential_and_centripetal_acceleration). Its magnitude is:
    """)
    return


@app.cell
def _(acc, sym, t, theta):
    _A = theta.diff(t) ** 2

    show(sym.simplify(sym.sqrt(sym.expand(acc[0]).coeff(_A) ** 2 + sym.expand(acc[1]).coeff(_A) ** 2)) * _A, "a_c =")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So the endpoint accelerates even when the angular acceleration is zero: the magnitude of its velocity is then constant, but its direction keeps changing, and the centripetal acceleration is what changes it. These are the tangential and normal accelerations of [Path frame](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/path_frame.py), for a path whose radius of curvature is always $\ell$.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/onelink_acc.gif?raw=1" width=350 alt="tangential and centripetal acceleration of one link"/></center><figcaption><center><i>Figure. Endpoint tangential and centripetal acceleration terms of one link attached to a fixed body by a hinge joint in a plane.</i></center></figcaption></figure>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulation

    Let's put numbers on these expressions. Consider a link of length $\ell = 1\:m$ that rotates from $\theta_i = 0^o$ to $\theta_f = 90^o$ in $1\:s$, following a [minimum-jerk trajectory](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/minimum_jerk_hypothesis.py), the smooth movement we met in the opening section, here applied to the joint angle.

    **Before you run the next cells**, predict: at the middle of the movement, the angular velocity is at its peak and the angular acceleration is zero. What is the acceleration of the endpoint at that instant?

    First, the angle and its derivatives, and the endpoint expressions turned into numerical functions with `lambdify`:
    """)
    return


@app.cell
def _(acc, ell, np, r, sym, t, theta, v):
    theta_i, theta_f, duration = 0, np.pi / 2, 1
    ts = np.linspace(0, 1, 101)
    mjt = theta_i + (theta_f - theta_i) * (10 * (t / duration) ** 3 - 15 * (t / duration) ** 4 + 6 * (t / duration) ** 5)

    ang = sym.lambdify(t, mjt, "numpy")(ts)
    vang = sym.lambdify(t, mjt.diff(t, 1), "numpy")(ts)
    aang = sym.lambdify(t, mjt.diff(t, 2), "numpy")(ts)
    jang = sym.lambdify(t, mjt.diff(t, 3), "numpy")(ts)

    _b, _c, _d, _e = sym.symbols("b c d e")
    _subs = {ell: 1, theta: _b, theta.diff(t, 1): _c, theta.diff(t, 2): _d, theta.diff(t, 3): _e}

    _r = r.subs(_subs)
    rxfu = sym.lambdify(_b, _r[0], modules="numpy")
    ryfu = sym.lambdify(_b, _r[1], modules="numpy")

    _v = v.subs(_subs)
    vxfu = sym.lambdify((_b, _c), _v[0], modules="numpy")
    vyfu = sym.lambdify((_b, _c), _v[1], modules="numpy")

    _acc = acc.subs(_subs)
    axfu = sym.lambdify((_b, _c, _d), _acc[0], modules="numpy")
    ayfu = sym.lambdify((_b, _c, _d), _acc[1], modules="numpy")

    _jerk = r.diff(t, 3).subs(_subs)
    jxfu = sym.lambdify((_b, _c, _d, _e), _jerk[0], modules="numpy")
    jyfu = sym.lambdify((_b, _c, _d, _e), _jerk[1], modules="numpy")
    return aang, ang, axfu, ayfu, jang, jxfu, jyfu, rxfu, ryfu, ts, vang, vxfu, vyfu


@app.cell
def _(aang, ang, axfu, ayfu, jang, jxfu, jyfu, np, plt, rxfu, ryfu, ts, vang, vxfu, vyfu):
    _fig, _hax = plt.subplots(2, 4, sharex=True, figsize=(14, 7), layout="constrained")

    _hax[0, 0].plot(ts, np.degrees(ang), linewidth=3)
    _hax[0, 0].set_title("Angular displacement [$^o$]")
    _hax[0, 0].set_ylabel("Joint")
    _hax[0, 1].plot(ts, np.degrees(vang), linewidth=3)
    _hax[0, 1].set_title("Angular velocity [$^o/s$]")
    _hax[0, 2].plot(ts, np.degrees(aang), linewidth=3)
    _hax[0, 2].set_title("Angular acceleration [$^o/s^2$]")
    _hax[0, 3].plot(ts, np.degrees(jang), linewidth=3)
    _hax[0, 3].set_title("Angular jerk [$^o/s^3$]")

    _hax[1, 0].plot(ts, rxfu(ang), "r", linewidth=3, label="x")
    _hax[1, 0].plot(ts, ryfu(ang), "k", linewidth=3, label="y")
    _hax[1, 0].set_title("Linear displacement [$m$]")
    _hax[1, 0].legend(loc="best").get_frame().set_alpha(0.8)
    _hax[1, 0].set_ylabel("Endpoint")
    _hax[1, 1].plot(ts, vxfu(ang, vang), "r", linewidth=3)
    _hax[1, 1].plot(ts, vyfu(ang, vang), "k", linewidth=3)
    _hax[1, 1].set_title("Linear velocity [$m/s$]")
    _hax[1, 2].plot(ts, axfu(ang, vang, aang), "r", linewidth=3)
    _hax[1, 2].plot(ts, ayfu(ang, vang, aang), "k", linewidth=3)
    _hax[1, 2].set_title("Linear acceleration [$m/s^2$]")
    _hax[1, 3].plot(ts, jxfu(ang, vang, aang, jang), "r", linewidth=3)
    _hax[1, 3].plot(ts, jyfu(ang, vang, aang, jang), "k", linewidth=3)
    _hax[1, 3].set_title("Linear jerk [$m/s^3$]")

    _fig.suptitle("Minimum-jerk trajectory kinematics of a one-link system", fontsize=18)
    for _i, _ax in enumerate(_hax.flat):
        _ax.locator_params(nbins=5)
        _ax.grid(True)
        if _i > 3:
            _ax.set_xlabel("Time [s]")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The joint angle follows the familiar S-shaped curve, and its velocity the bell-shaped one. The endpoint, though, has an acceleration that does not look like the angular acceleration at all. Using the two expressions we derived, $a_t = \ell\ddot{\theta}$ and $a_c = \ell\dot{\theta}^2$, let's look at a few instants:
    """)
    return


@app.cell
def _(aang, np, ts, vang):
    _ell = 1.0  # m
    print("    t    angular vel.   angular acc.     a_t      a_c")
    for _time in [0, 0.21, 0.5, 0.79, 1]:
        _i = np.argmin(np.abs(ts - _time))
        print(
            f"{ts[_i]:5.2f} s {np.degrees(vang[_i]):8.1f} deg/s {np.degrees(aang[_i]):8.1f} deg/s2"
            f" {_ell * aang[_i]:7.2f} {_ell * vang[_i] ** 2:7.2f} m/s2"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At the middle of the movement the angular acceleration is zero, but the endpoint is accelerating at 8.67 m/s², all of it centripetal, pointing at the joint. That is nearly as much as the peak tangential acceleration, 9.07 m/s², which occurs about 0.21 s into the movement. Even a perfectly smooth joint rotation produces an endpoint acceleration with two parts that peak at different instants and point in different directions.

    **Challenge 1.** Halve the amplitude of the movement, to $45^o$ in the same 1 s. Before running it, predict by what factor the peak tangential acceleration changes, and by what factor the peak centripetal acceleration changes. Then check.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** With the elbow resting on a table and the upper arm still, the forearm and hand form a one-link system about the elbow, 0.35 m long from the elbow to the centre of the hand. The forearm rotates at a constant $180^o/s$. What are the speed of the hand and its tangential and centripetal accelerations? Which way does the acceleration point?

    /// details | Solution to Exercise 1
    With $\ell = 0.35$ m and $\dot{\theta} = 180^o/s = \pi$ rad/s (the angles in these formulas must be in radians):

    $$
    ||\overrightarrow{\mathbf{v}}|| = \ell|\dot{\theta}| = 0.35\,\pi = 1.10\;\mathrm{m/s}
    $$

    The angular velocity is constant, so $a_t = \ell\ddot{\theta} = 0$, and

    $$
    a_c = \ell\dot{\theta}^2 = 0.35\,\pi^2 = 3.45\;\mathrm{m/s^2}
    $$

    All of the acceleration is centripetal: it points from the hand to the elbow, and it is about a third of the gravitational acceleration, although the forearm turns at a perfectly steady rate.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Jacobian matrix

    > The **[Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)** is the matrix of all first-order partial derivatives of a vector-valued function $F$:
    >
    > $$
    > F(q_1,...q_n) = \begin{bmatrix}F_{1}(q_1,...q_n)\\
    > \vdots\\
    > F_{m}(q_1,...q_n)\\
    > \end{bmatrix}
    > $$
    >
    > In general form, the Jacobian matrix of the function $F$ is:
    >
    > $$
    > \mathbf{J}=
    > \begin{bmatrix}
    > \dfrac{\partial F_{1}}{\partial q_{1}} & ... & \dfrac{\partial F_{1}}{\partial q_{n}} \\
    > \vdots  & \ddots  & \vdots \\
    > \dfrac{\partial F_{m}}{\partial q_{1}} & ... & \dfrac{\partial F_{m}}{\partial q_{n}} \\
    > \end{bmatrix}
    > $$

    ### Derivative of a vector-valued function using the Jacobian matrix

    > The time derivative of a vector-valued function $F$ can be computed with the Jacobian matrix:
    >
    > $$
    > \frac{dF}{dt} = \mathbf{J} \cdot \begin{bmatrix}\dfrac{d q_1}{dt}\\
    > \vdots\\
    > \dfrac{d q_n}{dt}\\
    > \end{bmatrix}
    > $$

    ### Jacobian matrix in the context of kinematic chains

    For a kinematic chain, the Jacobian is the matrix of all first-order partial derivatives of the linear position vector of the endpoint with respect to the vector of joint angles. It relates differential changes in the joint angles to the resulting differential changes in the position of the endpoint. The function $F_{i}$ is the expression for the endpoint position in each of its $m$ coordinates, and the variable $q_{i}$ is the angle of each of the $n$ joints.

    #### Jacobian matrix of a one-link chain

    For the planar one-link system, the Jacobian of the position vector of the endpoint, $\overrightarrow{\mathbf{r}}_P$, with respect to the joint angle, $q_1=\theta$, is

    $$
    \mathbf{J}=
    \begin{bmatrix}
    \dfrac{\partial x_P}{\partial \theta} \\[2ex]
    \dfrac{\partial y_P}{\partial \theta}
    \end{bmatrix}
    $$

    which evaluates to:
    """)
    return


@app.cell
def _(r, theta):
    show(r.diff(theta), r"\mathbf{J} =")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SymPy also has a function to calculate the Jacobian:
    """)
    return


@app.cell
def _(r, theta):
    J = r.jacobian([theta])

    show(J, r"\mathbf{J} =")
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can recalculate the kinematic expressions with the Jacobian, which can simplify the derivation. The linear velocity of the endpoint is the product of the Jacobian and the angular velocity,

    $$
    \overrightarrow{\mathbf{v}} = \mathbf{J} \cdot \dot{\theta}
    $$

    where:
    """)
    return


@app.cell
def _(t, theta):
    omega = theta.diff(t)

    show(omega, r"\omega =")
    return (omega,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The angular velocity is also a vector. Its direction is perpendicular to the plane of rotation, and by the [right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule) it has the direction of the versor $\hat{\mathbf{k}}$, coming out of the screen (or paper), when $\dot{\theta} > 0$. Then:
    """)
    return


@app.cell
def _(J, omega):
    show(J * omega, r"\overrightarrow{\mathbf{v}} =")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And the linear acceleration of the endpoint is the derivative of this product, with the product rule:

    $$
    \overrightarrow{\mathbf{a}} = \dot{\mathbf{J}} \cdot \overrightarrow{\mathbf{\omega}} + \mathbf{J} \cdot \dot{\overrightarrow{\mathbf{\omega}}}
    $$

    Let's calculate this derivative, and check that it is the same as the second derivative of the position we calculated before:
    """)
    return


@app.cell
def _(J, acc, mo, omega, sym, t):
    accJ = J.diff(t) * omega + J * omega.diff(t)

    mo.vstack(
        [
            show(accJ, r"\overrightarrow{\mathbf{a}} ="),
            mo.md(f"Same as the direct derivative: `{sym.simplify(accJ - acc) == sym.zeros(2, 1)}`"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For one link, the Jacobian is not much of a shortcut. Its value appears with two or more links, where it organizes the bookkeeping of many joints at once, and where its properties start to say something about the movement itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Show that, for the one-link system, the Jacobian $\mathbf{J}$ is always perpendicular to the position vector $\overrightarrow{\mathbf{r}}_P$ and has the same length as the link. What does that say about the direction and the magnitude of the velocity of the endpoint?

    /// details | Solution to Exercise 2
    With $\mathbf{J} = [-\ell\sin\theta,\; \ell\cos\theta]^T$ and $\overrightarrow{\mathbf{r}}_P = [\ell\cos\theta,\; \ell\sin\theta]^T$:

    $$
    \mathbf{J}\cdot\overrightarrow{\mathbf{r}}_P = -\ell^2\sin\theta\cos\theta + \ell^2\cos\theta\sin\theta = 0
    $$

    $$
    ||\mathbf{J}|| = \sqrt{\ell^2\sin^2\theta + \ell^2\cos^2\theta} = \ell
    $$

    Since $\overrightarrow{\mathbf{v}} = \mathbf{J}\,\dot{\theta}$, the velocity is perpendicular to the link, that is, tangent to the circle the endpoint travels on, and its magnitude is $\ell|\dot{\theta}|$, as we found by differentiating. The Jacobian is the velocity of the endpoint for an angular velocity of 1 rad/s.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The kinematics of a two-link chain

    Now let's look at a planar kinematic chain with two links, as shown below. The derivation follows the same steps as for one link.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/twolinks.gif?raw=1" width=400 alt="two-link chain"/></center><figcaption><center><i>Figure. Kinematics of a two-link chain with hinge joints in a plane.</i></center></figcaption></figure>

    We reuse the coordinate system and the time symbol, and define the link lengths $\ell_1, \ell_2$ and the joint angles $\theta_1, \theta_2$, functions of time:
    """)
    return


@app.cell
def _(dynamicsymbols, sym):
    ell1, ell2 = sym.symbols("ell_1 ell_2", positive=True)
    theta1, theta2 = dynamicsymbols("theta1 theta2")
    return ell1, ell2, theta1, theta2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The position of the endpoint in terms of the joint angles and link lengths is:
    """)
    return


@app.cell
def _(G, ell1, ell2, sym, theta1, theta2):
    r2_p = (ell1 * sym.cos(theta1) + ell2 * sym.cos(theta1 + theta2)) * G.i + (
        ell1 * sym.sin(theta1) + ell2 * sym.sin(theta1 + theta2)
    ) * G.j

    show(r2_p, r"\overrightarrow{\mathbf{r}}_P =")
    return (r2_p,)




@app.cell
def _(G, mo, r2_p):
    _components = r2_p.components

    mo.vstack([show(_components[G.i], "x_P ="), show(_components[G.j], "y_P =")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And in matrix form:
    """)
    return


@app.cell
def _(G, r2_p, sym):
    r2 = sym.Matrix((r2_p.dot(G.i), r2_p.dot(G.j)))

    show(r2, r"\overrightarrow{\mathbf{r}}_P =")
    return (r2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Joint and segment angles

    Note that $\theta_2$ is a joint angle, measured in what is called the **joint space**: it is the angle of the second link relative to the first. The angle of the second link with respect to the horizontal is $\theta_1+\theta_2$, an angle in the **segment space**. Joint and segment angles are also called relative and absolute angles, respectively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inverse kinematics

    Using the [law of cosines](https://en.wikipedia.org/wiki/Law_of_cosines) in the triangle formed by the two links and the line from the base to the endpoint, the angle $\theta_2$ in terms of the endpoint position is given by

    $$
    x_P^2 + y_P^2 = \ell_1^2+\ell_2^2 - 2\ell_1 \ell_2 \cos(\pi-\theta_2)
    $$

    $$
    \theta_2 = \arccos\left(\frac{x_P^2 + y_P^2 - \ell_1^2 - \ell_2^2}{2\ell_1 \ell_2}\right)
    $$

    To find the angle $\theta_1$, look at the triangle in red in the figure below. Its angle $\phi$ is

    $$
    \phi = \arctan\left(\frac{\ell_2 \sin(\theta_2)}{\ell_1 + \ell_2 \cos(\theta_2)}\right)
    $$

    and the angle of its hypotenuse with the horizontal is

    $$
    \theta_1 + \phi = \arctan\left(\frac{y_P}{x_P}\right)
    $$

    So the angle $\theta_1$ is:

    $$
    \theta_1 = \arctan\left(\frac{y_P}{x_P}\right) - \arctan\left(\frac{\ell_2 \sin(\theta_2)}{\ell_1+\ell_2 \cos(\theta_2)}\right)
    $$

    There are two possible sets of angles $(\theta_1, \theta_2)$ for the same endpoint coordinates $(x_P, y_P)$. The figure below shows in orange the other configuration of the chain with the same endpoint. The other solution is $\theta_2'=2\pi - \theta_2$ (equivalently, $-\theta_2$); since $\sin(\theta_2')=-\sin(\theta_{2})$, the second arctangent in the last equation changes sign. Even a simple two-link chain already has more than one joint configuration for the same endpoint position, and the problem gets much worse for chains with more links (more degrees of freedom).

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/twolinks_ik.gif?raw=1" width=350 alt="two solutions of the inverse kinematics"/></center><figcaption><center><i>Figure. Indetermination in the inverse kinematics of a two-link chain with hinge joints in a plane: two configurations reach the same endpoint.</i></center></figcaption></figure>

    Let's write the inverse kinematics as a function, together with the forward kinematics to check it. As announced in the one-link section, the code uses `arctan2` for both arctangents, so that the angles come out right in every quadrant:
    """)
    return


@app.function
def forward_kinematics(theta1, theta2, l1, l2):
    """Endpoint position (x, y) of a planar two-link chain with joint angles in radians."""
    import numpy as np

    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y


@app.function
def inverse_kinematics(x, y, l1, l2, elbow=1):
    """Joint angles [rad] of a planar two-link chain with its endpoint at (x, y).

    `elbow=1` returns the solution with theta2 >= 0 and `elbow=-1` the mirror
    solution with theta2 <= 0. Points out of reach give NaN.
    """
    import numpy as np

    cos_theta2 = (np.square(x) + np.square(y) - l1**2 - l2**2) / (2 * l1 * l2)
    with np.errstate(invalid="ignore"):
        theta2 = elbow * np.arccos(cos_theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    return theta1, theta2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a chain with $\ell_1 = \ell_2 = 0.5$ m, let's find both solutions for three endpoint positions, the first two of which will be the start and the end of the reach simulated later, and check each of them with the forward kinematics:
    """)
    return


@app.cell
def _(np):
    _l1 = _l2 = 0.5  # m
    for _x, _y in [(-0.5, 0.5), (0.0, 0.5 * np.sqrt(2)), (0.6, 0.3)]:
        print(f"Endpoint ({_x:.3f}, {_y:.3f}) m:")
        for _elbow in [1, -1]:
            _th1, _th2 = inverse_kinematics(_x, _y, _l1, _l2, elbow=_elbow)
            _xf, _yf = forward_kinematics(_th1, _th2, _l1, _l2)
            print(
                f"   theta1 = {np.degrees(_th1):7.2f}, theta2 = {np.degrees(_th2):7.2f} degrees"
                f"  -> forward kinematics: ({round(_xf, 3) + 0.0:.3f}, {round(_yf, 3) + 0.0:.3f}) m"
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every pair of angles puts the endpoint back where it should be. The two solutions for each point are mirror images of each other across the line from the base to the endpoint, with $\theta_2$ of opposite signs.

    **Before you run the next cell**, predict what goes wrong with the first point, $(-0.5, 0.5)$, if $\theta_1$ is computed with the plain arctangent, $\arctan(y_P/x_P)$, as the equations above are written. And what does `inverse_kinematics` return for the point $(1.2, 0)$?
    """)
    return


@app.cell
def _(np):
    _l1 = _l2 = 0.5  # m
    _x, _y = -0.5, 0.5
    _th2 = np.arccos((_x**2 + _y**2 - _l1**2 - _l2**2) / (2 * _l1 * _l2))
    _th1 = np.arctan(_y / _x) - np.arctan(_l2 * np.sin(_th2) / (_l1 + _l2 * np.cos(_th2)))
    _xf, _yf = forward_kinematics(_th1, _th2, _l1, _l2)

    print(f"arctan(y/x) = {np.degrees(np.arctan(_y / _x)):.1f} degrees, arctan2(y, x) = {np.degrees(np.arctan2(_y, _x)):.1f} degrees")
    print(f"With arctan: theta1 = {np.degrees(_th1):.1f} degrees, and the endpoint lands at ({_xf:.2f}, {_yf:.2f}) m")
    print("\nOut of reach:", inverse_kinematics(1.2, 0, _l1, _l2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plain arctangent sees $y/x = -1$ and answers $-45^o$, the direction of the opposite quadrant, and the chain ends up with its endpoint at $(0.5, -0.5)$, the mirror image of the target through the base. And a point farther from the base than $\ell_1 + \ell_2$ has no solution at all: the argument of the arccosine is larger than 1, and the function returns `NaN` (not a number). The **workspace** of the chain, the set of points the endpoint can reach, is the ring between the radii $|\ell_1 - \ell_2|$ and $\ell_1 + \ell_2$.

    Use the sliders below to set the two joint angles of the chain. The other configuration that reaches the same endpoint is drawn dashed, in orange.
    """)
    return


@app.cell
def _(mo):
    theta1_slider = mo.ui.slider(-180, 180, value=40, step=1, label=r"$\theta_1$ [degrees]")
    theta2_slider = mo.ui.slider(-180, 180, value=70, step=1, label=r"$\theta_2$ [degrees]")

    mo.hstack([theta1_slider, theta2_slider], justify="start")
    return theta1_slider, theta2_slider


@app.cell
def _(np, plt, theta1_slider, theta2_slider):
    _l1 = _l2 = 0.5  # m
    _th1, _th2 = np.radians(theta1_slider.value), np.radians(theta2_slider.value)
    _x, _y = forward_kinematics(_th1, _th2, _l1, _l2)
    _alt1, _alt2 = inverse_kinematics(_x, _y, _l1, _l2, elbow=-1 if _th2 > 0 else 1)

    _, _ax = plt.subplots(figsize=(6, 6))
    _circle = np.linspace(0, 2 * np.pi, 200)
    _ax.plot((_l1 + _l2) * np.cos(_circle), (_l1 + _l2) * np.sin(_circle), ":", color="0.6", label="edge of the workspace")
    for _a1, _a2, _style, _label in [(_alt1, _alt2, "--", "other solution"), (_th1, _th2, "-", "chain")]:
        _elbow_xy = (_l1 * np.cos(_a1), _l1 * np.sin(_a1))
        _color = "tab:orange" if _style == "--" else "tab:blue"
        _ax.plot([0, _elbow_xy[0], _x], [0, _elbow_xy[1], _y], _style, color=_color, linewidth=4, marker="o", markersize=8, label=_label)
    _ax.plot(0, 0, "ks", markersize=12)
    _ax.set_aspect("equal")
    _ax.set_xlim(-1.1, 1.1)
    _ax.set_ylim(-1.1, 1.1)
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title(f"Endpoint at ({_x:.2f}, {_y:.2f}) m;  det(J) = {_l1 * _l2 * np.sin(_th2):.3f} m$^2$")
    _ax.legend(loc="lower left")
    _ax.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (The determinant of the Jacobian, shown in the title, will make sense in a few sections.)

    **Challenge 2.** Find joint angles for which the two configurations coincide. Where in the workspace is the endpoint then? Explain, with the law of cosines, why there is only one solution there.

    **Exercise 3.** For the chain with $\ell_1 = \ell_2 = 0.5$ m, find by hand, with the law of cosines and a sketch, the two pairs of joint angles that put the endpoint at $(0.5, 0.5)$ m. Then check your answer with `inverse_kinematics`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _rows = []
    for _elbow in [1, -1]:
        _th1, _th2 = inverse_kinematics(0.5, 0.5, 0.5, 0.5, elbow=_elbow)
        _x, _y = forward_kinematics(_th1, _th2, 0.5, 0.5)
        _rows.append(
            f"| {_elbow:+d} | {round(float(np.degrees(_th1)), 1) + 0.0} | {round(float(np.degrees(_th2)), 1) + 0.0}"
            f" | ({_x:.3f}, {_y:.3f}) |"
        )

    mo.accordion(
        {
            "Solution to Exercise 3": mo.md(
                r"""
    The endpoint is at a distance $\sqrt{0.5^2 + 0.5^2} = \sqrt{0.5}$ m from the base, so $x_P^2 + y_P^2 = 0.5$ and

    $$
    \cos\theta_2 = \frac{0.5 - 0.25 - 0.25}{2 \cdot 0.5 \cdot 0.5} = 0 \quad\Longrightarrow\quad \theta_2 = \pm 90^o
    $$

    The line from the base to the endpoint is at $45^o$. With $\theta_2 = 90^o$, the triangle formed by the two links is right-angled and isosceles, so $\phi = 45^o$ and $\theta_1 = 45^o - 45^o = 0^o$. With $\theta_2 = -90^o$, $\phi = -45^o$ and $\theta_1 = 90^o$. In words: the first link along $x$ and the second link pointing straight up, or the first link pointing straight up and the second along $x$.

    `inverse_kinematics` agrees, and the forward kinematics puts both back on the target:

    | elbow | $\theta_1$ [degrees] | $\theta_2$ [degrees] | forward kinematics [m] |
    |:-----:|:--------------------:|:--------------------:|:----------------------:|
    """
                + "\n".join("    " + _row for _row in _rows)
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: an arm reaching across a table

    Let's use the two-link chain for a real arm. Seen from above, with the shoulder at the origin, $x$ pointing to the right and $y$ pointing forward, a right arm reaching over a table is a planar two-link chain: the upper arm, from the shoulder to the elbow, about $\ell_1 = 0.30$ m long in an adult, and the forearm with the hand, from the elbow to the centre of the hand, about $\ell_2 = 0.33$ m. The hand is to reach a cup at $(0.15, 0.40)$ m.

    **Before you run the next cell**, predict: the inverse kinematics has two solutions. Can your right arm adopt both of them? Try it.
    """)
    return


@app.cell
def _(np):
    L1_arm, L2_arm = 0.30, 0.33  # m, upper arm, and forearm with hand
    cup = (0.15, 0.40)  # m

    print(f"Distance from the shoulder to the cup: {np.hypot(*cup):.3f} m (maximum reach {L1_arm + L2_arm:.2f} m)")
    for _elbow in [1, -1]:
        _th1, _th2 = inverse_kinematics(*cup, L1_arm, L2_arm, elbow=_elbow)
        print(
            f"theta1 = {np.degrees(_th1):6.1f}, theta2 = {np.degrees(_th2):6.1f} degrees,"
            f" elbow at ({L1_arm * np.cos(_th1):.3f}, {L1_arm * np.sin(_th1):.3f}) m"
        )
    return L1_arm, L2_arm, cup


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both solutions put the hand on the cup, 0.427 m from the shoulder. In the first, the elbow is flexed by $94.8^o$ and sits out to the right of the shoulder, at $(0.283, 0.098)$ m: the posture you would adopt. In the second, $\theta_2 = -94.8^o$, and the forearm would have to bend the other way by the same amount, with the elbow in front of the chest.

    Seen from above, flexing the right elbow turns the forearm counterclockwise, so $\theta_2 > 0$ is flexion and $\theta_2 < 0$ is hyperextension, which the elbow allows by a few degrees at most in most adults. The mathematics has two answers; the anatomy has one. Joint limits are how a model of the body discards the solutions the body cannot use.

    **Exercise 4.** For this arm, with no joint limits, what are the inner and outer radii of the workspace? Can the hand reach the points $(0.45, 0.45)$ m and $(0.40, 0.40)$ m? For the point it can reach, by how much is the elbow flexed?
    """)
    return


@app.cell(hide_code=True)
def _(L1_arm, L2_arm, mo, np):
    _inner, _outer = abs(L1_arm - L2_arm), L1_arm + L2_arm
    _far, _near = (0.45, 0.45), (0.40, 0.40)
    _th1_far, _th2_far = inverse_kinematics(*_far, L1_arm, L2_arm)
    _th1, _th2 = inverse_kinematics(*_near, L1_arm, L2_arm)

    mo.accordion(
        {
            "Solution to Exercise 4": mo.md(
                rf"""
    The workspace is the ring between $|\ell_1 - \ell_2| = {_inner:.2f}$ m and $\ell_1 + \ell_2 = {_outer:.2f}$ m from the shoulder.

    - The point $(0.45, 0.45)$ m is {np.hypot(*_far):.3f} m from the shoulder, just out of reach; `inverse_kinematics` returns `{_th2_far}` for it.
    - The point $(0.40, 0.40)$ m is {np.hypot(*_near):.3f} m away, inside the ring. The elbow is flexed by $\theta_2 = {np.degrees(_th2):.1f}^o$, with $\theta_1 = {np.degrees(_th1):.1f}^o$.

    The inner radius is tiny because the two links have nearly the same length: with the elbow folded all the way, the hand would come within {100 * _inner:.0f} cm of the shoulder. The real elbow stops long before that; the example on the workspace of a real arm, later in this notebook, adds the joint limits.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differential kinematics of the two-link chain

    The linear velocity of the endpoint is:
    """)
    return


@app.cell
def _(r2, t):
    vel2 = r2.diff(t)

    show(vel2, r"\overrightarrow{\mathbf{v}} =")
    return (vel2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The linear velocity of the endpoint is the sum of two velocities: the velocity of the endpoint relative to joint 2, for instance $\ell_2\cos(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2)$ in the $y$ direction, plus the velocity of joint 2 relative to joint 1, for instance $\ell_1\cos(\theta_1)\dot{\theta}_1$, the term we already saw for the one-link system. In classical mechanics this is known as [relative velocity](https://en.wikipedia.org/wiki/Relative_velocity). Note that the second link turns at $\dot{\theta}_1 + \dot{\theta}_2$, the rate of change of its segment angle, not at $\dot{\theta}_2$.

    The linear acceleration of the endpoint is:
    """)
    return


@app.cell
def _(r2, t):
    acc2 = r2.diff(t, 2)

    show(acc2, r"\overrightarrow{\mathbf{a}} =")
    return (acc2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can separate this expression into three types of terms: those proportional to $\ddot{\theta}$ and to $\dot{\theta}^2$, as for the one-link system, and a new term, proportional to $\dot{\theta}_1\dot{\theta}_2$:
    """)
    return


@app.cell
def _(acc2, mo, sym, t, theta1, theta2):
    _acc = acc2.expand()

    _A, _B = theta1.diff(t, 2), theta2.diff(t, 2)
    tg = _A * sym.Matrix((_acc[0].coeff(_A), _acc[1].coeff(_A))) + _B * sym.Matrix((_acc[0].coeff(_B), _acc[1].coeff(_B)))

    _A, _B = theta1.diff(t) ** 2, theta2.diff(t) ** 2
    ct = _A * sym.Matrix((_acc[0].coeff(_A), _acc[1].coeff(_A))) + _B * sym.Matrix((_acc[0].coeff(_B), _acc[1].coeff(_B)))

    _A = theta1.diff(t) * theta2.diff(t)
    co = _A * sym.Matrix((_acc[0].coeff(_A), _acc[1].coeff(_A)))

    mo.vstack(
        [
            show(tg, r"\text{Tangential:} \quad"),
            show(ct, r"\text{Centripetal:} \quad"),
            show(co, r"\text{Coriolis:} \quad"),
            mo.md(f"The three terms add up to the acceleration: `{sym.simplify(tg + ct + co - _acc) == sym.zeros(2, 1)}`"),
        ]
    )
    return co, ct, tg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The new term is the [Coriolis acceleration](https://en.wikipedia.org/wiki/Coriolis_effect). It appears only when both joints rotate at the same time: the endpoint is then moving, at the speed $\ell_2\dot{\theta}_2$, relative to the first link, while the first link itself is rotating, at $\dot{\theta}_1$. Its magnitude is $2\ell_2\dot{\theta}_1\dot{\theta}_2$, and it points along the second link, perpendicular to the velocity of the endpoint relative to joint 2, $\overrightarrow{\mathbf{v}} - \overrightarrow{\mathbf{v}}_{joint2}$: it deflects the endpoint's motion sideways to that relative velocity.

    **Guiding questions 1.**

    1. Go back to Guiding question 0.3. With the elbow locked ($\dot{\theta}_2 = \ddot{\theta}_2 = 0$) and the shoulder rotating at a constant rate, which of the three terms survive? Where do they point?
    2. With the shoulder fixed and only the elbow moving, the chain is a one-link system. Check that the three terms reduce to the ones we found for one link.
    3. Why is the Coriolis term zero when either joint is still, however fast the other one moves?

    **Exercise 5.** Both joints of the chain with $\ell_1 = \ell_2 = 0.5$ m rotate at constant rates, $\dot{\theta}_1 = 2$ rad/s and $\dot{\theta}_2 = 3$ rad/s. At the instant when $\theta_1 = 0^o$ and $\theta_2 = 90^o$, calculate the tangential, centripetal and Coriolis terms and the total acceleration of the endpoint. Then explain the total without using the three terms.
    """)
    return


@app.cell(hide_code=True)
def _(co, ct, ell1, ell2, mo, np, sym, t, tg, theta1, theta2):
    _w1, _w2, _l1, _l2 = 2, 3, 0.5, 0.5
    _values = [  # higher derivatives first, so they are not caught by the lower ones
        {theta1.diff(t, 2): 0, theta2.diff(t, 2): 0},
        {theta1.diff(t): _w1, theta2.diff(t): _w2},
        {theta1: 0, theta2: sym.pi / 2, ell1: _l1, ell2: _l2},
    ]

    def _evaluate(term):
        """Evaluate an acceleration term at the instant of Exercise 5."""
        for _v in _values:
            term = term.subs(_v)
        return np.round(np.array(term, dtype=float).ravel(), 6) + 0.0

    def _fmt(vector):
        return f"({vector[0]:.1f}, {vector[1]:.1f})"

    _tg, _ct, _co = _evaluate(tg), _evaluate(ct), _evaluate(co)
    _total = _tg + _ct + _co
    _text = r"""
    Nothing accelerates angularly, so the tangential term is @TG@ m/s². With $\theta_1 = 0^o$ the first link lies along $x$, and with $\theta_1 + \theta_2 = 90^o$ the second link points along $y$, so the expressions above give:

    - centripetal: $-\ell_1\dot{\theta}_1^2\,\hat{\mathbf{i}} - \ell_2(\dot{\theta}_1^2 + \dot{\theta}_2^2)\,\hat{\mathbf{j}}$ = @CT@ m/s²;
    - Coriolis: $-2\ell_2\dot{\theta}_1\dot{\theta}_2\,\hat{\mathbf{j}}$ = @CO@ m/s²;
    - total: @TOT@ m/s², with a magnitude of @MAG@ m/s².

    Without the three terms: the elbow goes round the shoulder at 2 rad/s, with a centripetal acceleration $\ell_1\dot{\theta}_1^2$ = @A1@ m/s² towards the shoulder, and the second link turns at $\dot{\theta}_1 + \dot{\theta}_2 = 5$ rad/s, which gives the endpoint $\ell_2(\dot{\theta}_1 + \dot{\theta}_2)^2$ = @A2@ m/s² towards the elbow. Expanding $(\dot{\theta}_1 + \dot{\theta}_2)^2$ gives $\dot{\theta}_1^2 + \dot{\theta}_2^2$, which this notebook counts as centripetal, plus $2\dot{\theta}_1\dot{\theta}_2$, the Coriolis term. The split into three terms is bookkeeping; the total is not.
    """
    for _key, _value in {
        "@TG@": _fmt(_tg),
        "@CT@": _fmt(_ct),
        "@CO@": _fmt(_co),
        "@TOT@": _fmt(_total),
        "@MAG@": f"{np.linalg.norm(_total):.2f}",
        "@A1@": f"{_l1 * _w1**2:.1f}",
        "@A2@": f"{_l2 * (_w1 + _w2) ** 2:.1f}",
    }.items():
        _text = _text.replace(_key, _value)

    mo.accordion({"Solution to Exercise 5": mo.md(_text)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Jacobian of the two-link chain

    Let's now derive the Jacobian of the planar two-link chain:

    $$
    \mathbf{J} =
    \begin{bmatrix}
    \dfrac{\partial x_P}{\partial \theta_{1}} & \dfrac{\partial x_P}{\partial \theta_{2}} \\[2ex]
    \dfrac{\partial y_P}{\partial \theta_{1}} & \dfrac{\partial y_P}{\partial \theta_{2}}
    \end{bmatrix}
    $$

    We could build it by hand:

    ```python
    J2 = sym.Matrix([[r2[0].diff(theta1), r2[0].diff(theta2)], [r2[1].diff(theta1), r2[1].diff(theta2)]])
    ```

    but SymPy's `jacobian` method is shorter:
    """)
    return


@app.cell
def _(r2, theta1, theta2):
    J2 = r2.jacobian([theta1, theta2])

    show(J2, r"\mathbf{J} =")
    return (J2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each column of the Jacobian is the velocity the endpoint would have if only one joint rotated, at 1 rad/s. Using the Jacobian, the linear velocity of the endpoint is

    $$
    \overrightarrow{\mathbf{v}}_J = \mathbf{J} \cdot \begin{bmatrix}\dot{\theta}_1 \\
    \dot{\theta}_2
    \end{bmatrix}
    $$

    where:
    """)
    return


@app.cell
def _(sym, t, theta1, theta2):
    omega2 = sym.Matrix((theta1, theta2)).diff(t)

    show(omega2, r"\overrightarrow{\mathbf{\omega}} =")
    return (omega2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then:
    """)
    return


@app.cell
def _(J2, omega2):
    vel2J = J2 * omega2

    show(vel2J, r"\overrightarrow{\mathbf{v}}_J =")
    return (vel2J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This expression is the same as the first derivative of the endpoint position, which we can check by comparing the two expressions with SymPy:
    """)
    return


@app.cell
def _(vel2, vel2J):
    vel2.expand() == vel2J.expand()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once again, the linear acceleration of the endpoint is the derivative of the product of the Jacobian and the angular velocity:

    $$
    \overrightarrow{\mathbf{a}} = \dot{\mathbf{J}} \cdot \overrightarrow{\mathbf{\omega}} + \mathbf{J} \cdot \dot{\overrightarrow{\mathbf{\omega}}}
    $$

    Let's calculate this derivative:
    """)
    return


@app.cell
def _(J2, omega2, t):
    acc2J = J2.diff(t) * omega2 + J2 * omega2.diff(t)

    show(acc2J, r"\overrightarrow{\mathbf{a}} =")
    return (acc2J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And once again, it is the same as the second derivative of the endpoint position:
    """)
    return


@app.cell
def _(acc2, acc2J):
    acc2.expand() == acc2J.expand()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: what each joint contributes to the velocity of the hand

    Each column of the Jacobian is the velocity the endpoint would have if only that joint rotated, at 1 rad/s. The velocity of the endpoint is the sum of the two columns, each multiplied by the angular velocity of its joint, and since the columns are vectors, the two contributions add as vectors.

    Take the chain with $\ell_1=\ell_2=0.5$ m at $\theta_1 = 45^o$ and $\theta_2 = 90^o$, with both joints rotating at 1 rad/s.

    **Before you run the next cell**, predict: is the endpoint faster than it would be with only the shoulder rotating? Faster than the sum of the two speeds?
    """)
    return


@app.cell
def _(J2, ell1, ell2, np, plt, sym, theta1, theta2):
    _l1 = _l2 = 0.5  # m
    _th1, _th2 = np.radians(45), np.radians(90)
    _w = np.array([1.0, 1.0])  # rad/s
    _J = np.array(sym.lambdify((theta1, theta2, ell1, ell2), J2, "numpy")(_th1, _th2, _l1, _l2), dtype=float)

    _shoulder, _elbow = _J[:, 0] * _w[0], _J[:, 1] * _w[1]  # contribution of each joint
    _v = _shoulder + _elbow
    _p = np.array(forward_kinematics(_th1, _th2, _l1, _l2))
    _e = np.array([_l1 * np.cos(_th1), _l1 * np.sin(_th1)])
    _angle = np.degrees(np.arccos(_shoulder @ _elbow / (np.linalg.norm(_shoulder) * np.linalg.norm(_elbow))))

    print(f"Shoulder alone: {np.linalg.norm(_shoulder):.3f} m/s, elbow alone: {np.linalg.norm(_elbow):.3f} m/s, both: {np.linalg.norm(_v):.3f} m/s")
    print(f"Angle between the two contributions: {_angle:.1f} degrees")
    print(f"Distance from the shoulder to the endpoint: {np.linalg.norm(_p):.3f} m")

    _, _ax = plt.subplots(figsize=(6.5, 5))
    _ax.plot([0, _e[0], _p[0]], [0, _e[1], _p[1]], "-o", color="tab:blue", linewidth=4, markersize=8, label="chain")
    _ax.plot([0, _p[0]], [0, _p[1]], ":", color="0.5", label="shoulder to endpoint")
    _ax.plot(0, 0, "ks", markersize=12)
    for _vec, _color, _label in [(_shoulder, "tab:red", "shoulder only"), (_elbow, "tab:green", "elbow only"), (_v, "k", "both joints")]:
        _ax.quiver(*_p, *_vec, angles="xy", scale_units="xy", scale=2, color=_color, width=0.012, label=_label)
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.75, 0.55)
    _ax.set_ylim(-0.1, 0.95)
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title("Endpoint velocity, 1 rad/s at each joint\n(arrows drawn at 0.5 m per m/s)")
    _ax.legend(loc="upper right", fontsize=9)
    _ax.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alone, the shoulder would move the endpoint at 0.707 m/s and the elbow at 0.5 m/s; together they move it at 1.118 m/s, more than either alone and less than their sum, because the two contributions are $45^o$ apart. Look at the directions of the arrows. The contribution of the shoulder is perpendicular to the dotted line from the shoulder to the endpoint, and its magnitude for 1 rad/s, 0.707 m/s, is the length of that line. The contribution of the elbow is perpendicular to the second link, with magnitude $\ell_2$. Each joint swings everything beyond it on a circle around itself: the rule $\overrightarrow{\mathbf{v}} = \overrightarrow{\boldsymbol{\omega}} \times \overrightarrow{\mathbf{r}}$ of [Kinematics of a rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py), applied once per joint.

    **Exercise 6.** Same chain, now at $\theta_1 = 30^o$ and $\theta_2 = 60^o$, with $\dot{\theta}_1 = 1$ rad/s and $\dot{\theta}_2 = -1$ rad/s. Calculate the velocity of the endpoint with the Jacobian. Its magnitude comes out as a round number. Why? (Hint: what happens to the segment angle of the second link?)
    """)
    return


@app.cell(hide_code=True)
def _(J2, ell1, ell2, mo, np, sym, theta1, theta2):
    _J = np.array(
        sym.lambdify((theta1, theta2, ell1, ell2), J2, "numpy")(np.radians(30), np.radians(60), 0.5, 0.5), dtype=float
    )
    _J = np.round(_J, 6) + 0.0
    _v = _J @ np.array([1.0, -1.0])
    _text = r"""
    At $\theta_1 = 30^o$ and $\theta_2 = 60^o$ the Jacobian is (in metres per radian)

    $$
    \mathbf{J} = \begin{bmatrix} @J00@ & @J01@ \\ @J10@ & @J11@ \end{bmatrix}
    $$

    and the velocity of the endpoint is $\mathbf{J}\,[1,\; -1]^T$ = @V@ m/s, with a magnitude of @SPEED@ m/s, pointing at @DIR@° from the $x$ axis.

    The round number has a simple reason. With $\dot{\theta}_2 = -\dot{\theta}_1$, the segment angle of the second link, $\theta_1 + \theta_2$, does not change: the second link translates without rotating, and all its points, the endpoint included, move with the elbow. The elbow goes round the shoulder at $\ell_1\dot{\theta}_1 = 0.5$ m/s, perpendicular to the first link, which is at $30^o$; hence the direction $30^o + 90^o = 120^o$.
    """
    for _key, _value in {
        "@J00@": f"{_J[0, 0]:.3f}",
        "@J01@": f"{_J[0, 1]:.3f}",
        "@J10@": f"{_J[1, 0]:.3f}",
        "@J11@": f"{_J[1, 1]:.3f}",
        "@V@": f"({_v[0]:.3f}, {_v[1]:.3f})",
        "@SPEED@": f"{np.linalg.norm(_v):.3f}",
        "@DIR@": f"{np.degrees(np.arctan2(_v[1], _v[0])):.1f}",
    }.items():
        _text = _text.replace(_key, _value)

    mo.accordion({"Solution to Exercise 6": mo.md(_text)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### When the Jacobian cannot be inverted

    The Jacobian turns joint velocities into an endpoint velocity. Going the other way, from a desired endpoint velocity to the joint velocities that produce it, means solving $\mathbf{J}\,\overrightarrow{\mathbf{\omega}} = \overrightarrow{\mathbf{v}}$ for $\overrightarrow{\mathbf{\omega}}$, which is possible only where the matrix can be inverted, that is, where its determinant is not zero:
    """)
    return


@app.cell
def _(J2, sym):
    show(sym.simplify(J2.det()), r"\det(\mathbf{J}) =")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The determinant depends only on the elbow angle, and it is zero when $\sin\theta_2 = 0$: with the chain fully stretched ($\theta_2 = 0$) or fully folded ($\theta_2 = \pm 180^o$). Such configurations are called **singular**.

    **Before you run the next cell**, think of your own arm almost fully stretched. Which is harder: to move your hand a little further away from your shoulder, or sideways? Then let's compute the joint velocities needed to move the endpoint at 1 m/s, either radially (away from the base) or tangentially (perpendicular to that), for decreasing elbow angles:
    """)
    return


@app.cell
def _(J2, ell1, ell2, np, sym, theta1, theta2):
    _l1 = _l2 = 0.5  # m
    _J = sym.lambdify((theta1, theta2, ell1, ell2), J2, "numpy")

    print("Elbow     reach     joint velocities [deg/s] for 1 m/s of endpoint velocity")
    print("angle                 radially                 tangentially")
    for _deg in [90, 30, 10, 2]:
        _th1, _th2 = np.radians(30), np.radians(_deg)
        _Jn = np.array(_J(_th1, _th2, _l1, _l2), dtype=float)
        _p = np.array(forward_kinematics(_th1, _th2, _l1, _l2))
        _radial = _p / np.linalg.norm(_p)
        _tangential = np.array([-_radial[1], _radial[0]])
        _w_radial = np.degrees(np.linalg.solve(_Jn, _radial)).round(1) + 0.0
        _w_tangential = np.degrees(np.linalg.solve(_Jn, _tangential)).round(1) + 0.0
        print(
            f"{_deg:4d} deg  {100 * np.linalg.norm(_p) / (_l1 + _l2):6.2f}%"
            f"   ({_w_radial[0]:7.1f}, {_w_radial[1]:7.1f})      ({_w_tangential[0]:5.1f}, {_w_tangential[1]:5.1f})"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Moving the endpoint sideways always costs about the same: between 57 and 81 °/s at the shoulder, and nothing at the elbow, because rotating the shoulder alone swings the endpoint perpendicular to the line from the base. Moving it radially costs more and more as the chain straightens: 162 °/s at the elbow with the elbow at 90°, 1315 °/s at 10°, and 6566 °/s at 2°, when the endpoint is at 99.98% of its maximum reach. At $\theta_2 = 0$ no finite joint velocity will do; the endpoint simply cannot move radially. Close to a singular configuration, small endpoint movements in some directions require enormous joint movements, and small errors in the endpoint position become large errors in the joint angles calculated by inverse kinematics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: the workspace of a real arm

    The ring of Exercise 4 is the workspace of an idealized chain whose joints turn all the way round. Real joints do not. For illustration, let's give the arm of the cup example approximate joint limits: shoulder angles between $\theta_1 = -30^o$ (the upper arm pointing out to the right and slightly backwards) and $135^o$ (across the front of the chest), and elbow angles between $0^o$ (straight) and $145^o$, roughly the normal range of elbow flexion.

    The region the hand can reach can be mapped forwards, by sweeping the joint angles through their ranges, or backwards, by asking `inverse_kinematics` whether each point of the plane has a solution within the limits. The code below does it backwards, for 400 000 random points, which also gives the area of the region, and tests four targets: the cup, a point on the left, a point behind the shoulder, and a point just in front of the shoulder.

    **Before you run the next cell**, predict: what fraction of the ideal ring can the real arm reach? A quarter? A half? Most of it?
    """)
    return


@app.cell
def _(L1_arm, L2_arm, cup, np, plt):
    joint_limits = np.radians([[-30, 135], [0, 145]])  # shoulder and elbow [rad]

    def _reachable(x, y):
        """True where the hand reaches (x, y) with the joints within their limits."""
        _th1, _th2 = inverse_kinematics(x, y, L1_arm, L2_arm, elbow=1)  # elbow flexion: theta2 >= 0
        _th1 = (_th1 + np.pi) % (2 * np.pi) - np.pi  # shoulder angle in [-180, 180) degrees
        with np.errstate(invalid="ignore"):
            return (_th2 <= joint_limits[1, 1]) & (_th1 >= joint_limits[0, 0]) & (_th1 <= joint_limits[0, 1])

    _R = L1_arm + L2_arm
    _pts = np.random.default_rng(0).uniform(-_R, _R, (400_000, 2))
    _ok = _reachable(_pts[:, 0], _pts[:, 1])
    workspace_area = _ok.mean() * (2 * _R) ** 2
    _ring = np.pi * ((L1_arm + L2_arm) ** 2 - (L1_arm - L2_arm) ** 2)
    print(
        f"Area reached within the joint limits: {workspace_area:.3f} m2,"
        f" of an ideal ring of {_ring:.3f} m2 ({100 * workspace_area / _ring:.0f}%)"
    )

    _targets = {"cup": cup, "left": (-0.30, 0.20), "behind": (0.0, -0.40), "close": (0.10, 0.05)}
    for _name, (_x, _y) in _targets.items():
        print(f"{_name:7s} ({_x:5.2f}, {_y:5.2f}) m: {'reachable' if _reachable(_x, _y) else 'out of reach'}")

    _, _ax = plt.subplots(figsize=(6.5, 6))
    _sample = _pts[_ok][:30_000]
    _ax.plot(_sample[:, 0], _sample[:, 1], ".", color="tab:blue", markersize=1, alpha=0.3)
    _ax.plot([], [], "s", color="tab:blue", alpha=0.4, label="reachable within the limits")
    _circle = np.linspace(0, 2 * np.pi, 200)
    for _radius in [L1_arm + L2_arm, abs(L1_arm - L2_arm)]:
        _ax.plot(_radius * np.cos(_circle), _radius * np.sin(_circle), "--", color="0.5")
    _ax.plot([], [], "--", color="0.5", label="ideal ring")
    _ax.plot(0, 0, "ks", markersize=10, label="shoulder")
    for _name, (_x, _y) in _targets.items():
        _in = bool(_reachable(_x, _y))
        _ax.plot(_x, _y, "o" if _in else "X", color="tab:green" if _in else "tab:red", markersize=11)
        _ax.annotate(_name, (_x, _y), textcoords="offset points", xytext=(8, 6), fontsize=11)
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.7, 0.7)
    _ax.set_ylim(-0.7, 0.7)
    _ax.set_xlabel("x [m] (to the right)")
    _ax.set_ylabel("y [m] (forward)")
    _ax.set_title("Workspace of a right arm seen from above")
    _ax.legend(loc="lower left", fontsize=9)
    _ax.grid(True)
    plt.tight_layout()
    plt.show()
    return joint_limits, workspace_area


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The real arm reaches 0.519 m², 42% of the ideal ring. The cup and the point on the left are within reach. The point behind the shoulder would need a shoulder angle of $-144^o$, far beyond the limit, and the point just in front of the shoulder would need the elbow flexed by $160^o$. What the limits remove is the region behind the body and a hole around the shoulder much larger than the 3 cm of Exercise 4: the elbow cannot fold the hand back onto the shoulder.

    There is a neat way to check the area, and it gives the determinant of the Jacobian a second meaning. The forward kinematics maps a small patch of joint space, $d\theta_1\,d\theta_2$, to a small patch of the plane with area $|\det\mathbf{J}|\,d\theta_1\,d\theta_2$: the determinant is the factor by which the chain stretches areas. Within these limits each point is reached by a single posture, so the area of the workspace is the integral of $|\det\mathbf{J}| = \ell_1\ell_2|\sin\theta_2|$ over the joint ranges:

    $$
    A = \ell_1\ell_2 \int_{-30^o}^{135^o} d\theta_1 \int_{0}^{145^o} \sin\theta_2\, d\theta_2 = \ell_1\ell_2\,\Delta\theta_1\,(1 - \cos 145^o)
    $$

    with $\Delta\theta_1$ in radians:
    """)
    return


@app.cell
def _(L1_arm, L2_arm, joint_limits, np, workspace_area):
    _formula = L1_arm * L2_arm * np.ptp(joint_limits[0]) * (np.cos(joint_limits[1, 0]) - np.cos(joint_limits[1, 1]))

    print(f"Integral of |det J|: {_formula:.4f} m2;  random points: {workspace_area:.4f} m2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two agree to within the random error of the 400 000 points. The formula also explains where the area comes from: elbow angles near $0^o$ or $180^o$, where $\sin\theta_2$ is small, add little area. It is the singularity of the previous section seen from another side: where the chain cannot move the endpoint in every direction, it also covers little ground.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulation: reaching in a straight line

    Back to the question of the opening section. Consider a chain with $\ell_1=\ell_2=0.5\:m$ whose endpoint moves in a straight line, from $(-0.5, 0.5)$ m to $(0, 0.5\sqrt{2})$ m in 1 s, following a [minimum-jerk trajectory](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/minimum_jerk_hypothesis.py). We found the joint angles at the two ends of this path above: the chain starts at $\theta_1 = \theta_2 = 90^o$ and ends at $\theta_1 = 45^o$, $\theta_2 = 90^o$.

    **Before you run the next cells**, predict: the elbow angle is the same at the start and at the end. Does it stay constant during the movement?

    First, the minimum-jerk trajectory of the endpoint, as functions of time:
    """)
    return


@app.cell
def _(np, sym, t):
    _p0, _pf, _d = sym.symbols("p0 pf d")

    # minimum-jerk kinematics
    _mjt = _p0 + (_pf - _p0) * (10 * (t / _d) ** 3 - 15 * (t / _d) ** 4 + 6 * (t / _d) ** 5)
    _rfu = sym.lambdify((t, _p0, _pf, _d), _mjt, "numpy")
    _vfu = sym.lambdify((t, _p0, _pf, _d), sym.diff(_mjt, t, 1), "numpy")
    _afu = sym.lambdify((t, _p0, _pf, _d), sym.diff(_mjt, t, 2), "numpy")
    _jfu = sym.lambdify((t, _p0, _pf, _d), sym.diff(_mjt, t, 3), "numpy")

    # values
    D, L1, L2 = 1, 0.5, 0.5
    p0, pf = [-0.5, 0.5], [0, 0.5 * np.sqrt(2)]  # initial and final endpoint positions
    ts2 = np.linspace(0, 1, 101)

    # endpoint kinematics
    x = _rfu(ts2, p0[0], pf[0], D)
    y = _rfu(ts2, p0[1], pf[1], D)
    vx = _vfu(ts2, p0[0], pf[0], D)
    vy = _vfu(ts2, p0[1], pf[1], D)
    ax = _afu(ts2, p0[0], pf[0], D)
    ay = _afu(ts2, p0[1], pf[1], D)
    jx = _jfu(ts2, p0[0], pf[0], D)
    jy = _jfu(ts2, p0[1], pf[1], D)

    print(f"Path length: {np.hypot(pf[0] - p0[0], pf[1] - p0[1]):.3f} m, peak speed: {np.hypot(vx, vy).max():.2f} m/s")
    return L1, L2, ax, ay, jx, jy, ts2, vx, vy, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the inverse kinematics. The joint angles are written in terms of the endpoint coordinates $r_x(t)$ and $r_y(t)$, and their derivatives with respect to time are found symbolically, with the chain rule doing the work. Each derivative is then evaluated at every instant with the endpoint position, velocity, acceleration and jerk computed above:
    """)
    return


@app.cell
def _(L1, L2, ax, ay, dynamicsymbols, ell1, ell2, jx, jy, np, sym, t, ts2, vx, vy, x, y):
    rx, ry = dynamicsymbols("rx ry", real=True)  # or sym.Function("rx")(t)

    # inverse kinematics
    theta2_ik = sym.acos((rx**2 + ry**2 - ell1**2 - ell2**2) / (2 * ell1 * ell2))
    theta1_ik = sym.atan2(ry, rx) - sym.atan2(ell2 * sym.sin(theta2_ik), ell1 + ell2 * sym.cos(theta2_ik))

    _X, _Y, _Xd, _Yd, _Xdd, _Ydd, _Xddd, _Yddd = sym.symbols("X Y Xd Yd Xdd Ydd Xddd Yddd")
    _subs = {
        rx: _X, ry: _Y,
        rx.diff(t, 1): _Xd, ry.diff(t, 1): _Yd,
        rx.diff(t, 2): _Xdd, ry.diff(t, 2): _Ydd,
        rx.diff(t, 3): _Xddd, ry.diff(t, 3): _Yddd,
    }
    _args = (_X, _Y, _Xd, _Yd, _Xdd, _Ydd, _Xddd, _Yddd, ell1, ell2)
    _vals = (x, y, vx, vy, ax, ay, jx, jy, L1, L2)

    def _evaluate(expr):
        """Evaluate a symbolic expression of the endpoint kinematics at every instant."""
        _f = sym.lambdify(_args, expr.subs(_subs), "numpy")
        return np.broadcast_to(_f(*_vals), ts2.shape).astype(float)

    ang1, ang2 = _evaluate(theta1_ik), _evaluate(theta2_ik)
    vang1, vang2 = _evaluate(theta1_ik.diff(t, 1)), _evaluate(theta2_ik.diff(t, 1))
    aang1, aang2 = _evaluate(theta1_ik.diff(t, 2)), _evaluate(theta2_ik.diff(t, 2))
    jang1, jang2 = _evaluate(theta1_ik.diff(t, 3)), _evaluate(theta2_ik.diff(t, 3))
    return aang1, aang2, ang1, ang2, jang1, jang2, vang1, vang2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And the plots of the trajectories:
    """)
    return


@app.cell
def _(aang1, aang2, ang1, ang2, ax, ay, jang1, jang2, jx, jy, np, plt, ts2, vang1, vang2, vx, vy, x, y):
    _fig, _hax = plt.subplots(2, 4, sharex=True, figsize=(14, 7), layout="constrained")

    _hax[0, 0].plot(ts2, x, "r", linewidth=3, label="x")
    _hax[0, 0].plot(ts2, y, "k", linewidth=3, label="y")
    _hax[0, 0].set_title("Linear displacement [$m$]")
    _hax[0, 0].legend(loc="best").get_frame().set_alpha(0.8)
    _hax[0, 0].set_ylabel("Endpoint")
    _hax[0, 1].plot(ts2, vx, "r", linewidth=3)
    _hax[0, 1].plot(ts2, vy, "k", linewidth=3)
    _hax[0, 1].set_title("Linear velocity [$m/s$]")
    _hax[0, 2].plot(ts2, ax, "r", linewidth=3)
    _hax[0, 2].plot(ts2, ay, "k", linewidth=3)
    _hax[0, 2].set_title("Linear acceleration [$m/s^2$]")
    _hax[0, 3].plot(ts2, jx, "r", linewidth=3)
    _hax[0, 3].plot(ts2, jy, "k", linewidth=3)
    _hax[0, 3].set_title("Linear jerk [$m/s^3$]")

    _hax[1, 0].plot(ts2, np.degrees(ang1), "b", linewidth=3, label="joint 1")
    _hax[1, 0].plot(ts2, np.degrees(ang2), "g", linewidth=3, label="joint 2")
    _hax[1, 0].set_title("Angular displacement [$^o$]")
    _hax[1, 0].legend(loc="best").get_frame().set_alpha(0.8)
    _hax[1, 0].set_ylabel("Joint")
    _hax[1, 1].plot(ts2, np.degrees(vang1), "b", linewidth=3)
    _hax[1, 1].plot(ts2, np.degrees(vang2), "g", linewidth=3)
    _hax[1, 1].set_title("Angular velocity [$^o/s$]")
    _hax[1, 2].plot(ts2, np.degrees(aang1), "b", linewidth=3)
    _hax[1, 2].plot(ts2, np.degrees(aang2), "g", linewidth=3)
    _hax[1, 2].set_title("Angular acceleration [$^o/s^2$]")
    _hax[1, 3].plot(ts2, np.degrees(jang1), "b", linewidth=3)
    _hax[1, 3].plot(ts2, np.degrees(jang2), "g", linewidth=3)
    _hax[1, 3].set_title("Angular jerk [$^o/s^3$]")

    _fig.suptitle("Minimum-jerk trajectory kinematics of a two-link chain", fontsize=18)
    for _i, _ax in enumerate(_hax.flat):
        _ax.locator_params(nbins=5)
        _ax.grid(True)
        if _i > 3:
            _ax.set_xlabel("Time [$s$]")
    plt.show()
    return


@app.cell
def _(ang1, ang2, np, plt, x, y):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")

    _ax1.plot(x, y, "r", linewidth=3)
    _ax1.set_xlabel("Displacement in x [$m$]")
    _ax1.set_ylabel("Displacement in y [$m$]")
    _ax1.set_title("Endpoint space")
    _ax1.axis("equal")
    _ax1.grid(True)

    _ax2.plot(np.degrees(ang1), np.degrees(ang2), "b", linewidth=3)
    _ax2.set_xlabel("Displacement in joint 1 [$^o$]")
    _ax2.set_ylabel("Displacement in joint 2 [$^o$]")
    _ax2.set_title("Joint space")
    _ax2.axis("equal")
    _ax2.grid(True)
    plt.show()
    return


@app.cell
def _(ang1, ang2, np, ts2, vang1, vang2):
    print(f"Joint 1: from {np.degrees(ang1[0]):.1f} to {np.degrees(ang1[-1]):.1f} degrees, peak velocity {np.degrees(np.abs(vang1).max()):.1f} deg/s")
    print(
        f"Joint 2: from {np.degrees(ang2[0]):.1f} to {np.degrees(ang2[-1]):.1f} degrees,"
        f" maximum {np.degrees(ang2.max()):.1f} degrees at t = {ts2[ang2.argmax()]:.2f} s,"
        f" peak velocity {np.degrees(np.abs(vang2).max()):.1f} deg/s"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The hand moves in a straight line with a single, symmetric, bell-shaped speed profile. The joints do not. The shoulder rotates by $45^o$ in one direction, with a peak velocity of about 93 °/s. The elbow starts and ends at $90^o$, yet it does not stay there: it flexes by more than 8° and then extends back, with a peak velocity of about 30 °/s. In joint space, the straight line of the hand becomes a curve. Simple at the hand, complicated at the joints; this is the observation that led Morasso (1981) and Flash and Hogan (1985) to propose that the nervous system plans reaching movements in terms of the hand, not the joints.

    The joint velocities above came from differentiating the inverse kinematics symbolically. The Jacobian offers an independent check: at every instant, the joint velocities must solve $\mathbf{J}\,\overrightarrow{\mathbf{\omega}} = \overrightarrow{\mathbf{v}}$:
    """)
    return


@app.cell
def _(J2, L1, L2, ang1, ang2, ell1, ell2, np, sym, theta1, theta2, vang1, vang2, vx, vy):
    _J = sym.lambdify((theta1, theta2, ell1, ell2), J2, "numpy")
    _omega = np.array(
        [np.linalg.solve(np.array(_J(_a1, _a2, L1, L2), dtype=float), [_vx, _vy]) for _a1, _a2, _vx, _vy in zip(ang1, ang2, vx, vy)]
    )

    print(f"Largest difference: {np.degrees(np.abs(_omega - np.column_stack((vang1, vang2))).max()):.1e} deg/s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two methods agree to rounding error. The Jacobian route is also much cheaper: one small linear system per instant, instead of differentiating the inverse kinematics expressions.

    **Challenge 3.** Repeat the simulation with the other solution of the inverse kinematics, the one with $\theta_2 < 0$. Plot the joint angles and joint velocities, and compare them with the solution above. Does the hand path change?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Each type of acceleration of the endpoint in the simulated reach

    Let's evaluate the tangential, centripetal and Coriolis terms with the joint angles and their derivatives from the simulation:
    """)
    return


@app.cell
def _(L1, L2, aang1, aang2, ang1, ang2, co, ct, ell1, ell2, np, sym, t, tg, theta1, theta2, ts2, vang1, vang2):
    _A1, _A2, _A1d, _A2d, _A1dd, _A2dd = sym.symbols("A1 A2 A1d A2d A1dd A2dd")
    _subs = {
        theta1: _A1, theta2: _A2,
        theta1.diff(t, 1): _A1d, theta2.diff(t, 1): _A2d,
        theta1.diff(t, 2): _A1dd, theta2.diff(t, 2): _A2dd,
        ell1: L1, ell2: L2,
    }

    def _evaluate(term):
        """Evaluate the x and y components of an acceleration term at every instant."""
        _comps = [sym.lambdify((_A1, _A2, _A1d, _A2d, _A1dd, _A2dd), _c.subs(_subs), "numpy") for _c in term]
        return np.column_stack([np.broadcast_to(_f(ang1, ang2, vang1, vang2, aang1, aang2), ts2.shape) for _f in _comps])

    tg2n = _evaluate(tg)  # tangential acceleration
    ct2n = _evaluate(ct)  # centripetal acceleration
    co2n = _evaluate(co)  # Coriolis acceleration
    acc_tot = tg2n + ct2n + co2n  # total acceleration
    return acc_tot, co2n, ct2n, tg2n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The total has to be the acceleration of the minimum-jerk trajectory we started from. Let's check it, and compare the sizes of the three terms:
    """)
    return


@app.cell
def _(acc_tot, ax, ay, co2n, ct2n, np, tg2n, ts2):
    print(f"Largest difference between the sum of the terms and the minimum-jerk acceleration: {np.abs(acc_tot - np.column_stack((ax, ay))).max():.1e} m/s2\n")
    for _name, _term in [("Tangential", tg2n), ("Centripetal", ct2n), ("Coriolis", co2n), ("Total", acc_tot)]:
        _mag = np.linalg.norm(_term, axis=1)
        print(f"{_name:12s} peak magnitude {_mag.max():.2f} m/s2 at t = {ts2[_mag.argmax()]:.2f} s")
    _i = np.linalg.norm(tg2n, axis=1).argmax()
    print(f"\nAt t = {ts2[_i]:.2f} s: tangential {np.linalg.norm(tg2n[_i]):.2f} m/s2, total {np.linalg.norm(acc_tot[_i]):.2f} m/s2")
    return


@app.cell
def _(acc_tot, co2n, ct2n, plt, tg2n, ts2):
    _fig, _hax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 5), layout="constrained")
    for _ax, _term, _title in zip(_hax, [tg2n, ct2n, co2n], ["Tangential", "Centripetal", "Coriolis"]):
        _ax.plot(ts2, acc_tot[:, 0], color=(1, 0, 0, 0.3), linewidth=5, label="x total")
        _ax.plot(ts2, acc_tot[:, 1], color=(0, 0, 0, 0.3), linewidth=5, label="y total")
        _ax.plot(ts2, _term[:, 0], "r", linewidth=2, label="x")
        _ax.plot(ts2, _term[:, 1], "k", linewidth=2, label="y")
        _ax.set_title(_title)
        _ax.set_xlabel("Time [$s$]")
        _ax.locator_params(nbins=5)
        _ax.grid(True)
    _hax[0].set_ylabel("Endpoint acceleration [$m/s^2$]")
    _hax[1].legend(loc="best").get_frame().set_alpha(0.8)
    _fig.suptitle("Acceleration terms for the minimum-jerk trajectory of a two-link chain", fontsize=16)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The three terms add up to the minimum-jerk acceleration to within $10^{-15}\;\mathrm{m/s^2}$. The tangential term, the one produced by the angular accelerations of the joints, is the largest, but it is not the whole story. The centripetal term reaches more than half the peak of the total, and the Coriolis term almost a quarter. And at $t = 0.77$ s the tangential term alone is larger than the total acceleration: the velocity-dependent terms partly cancel it. To produce a simple hand acceleration, the joints must produce angular accelerations that compensate for the centripetal and Coriolis accelerations the movement itself generates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your answers to Challenge 0. Did your shoulder and elbow each rotate in one direction only, at a constant rate? Compare with the simulated reach.
    2. A person stands on one leg. Is the supporting leg an open or a closed kinematic chain in mechanics? And in the clinical sense?
    3. Why can the endpoint of a chain accelerate when no joint has an angular acceleration?
    4. Why does the Coriolis term need both joints to rotate, and what is it proportional to?
    5. Near full extension, why is it hard to move the hand radially, away from the shoulder? What does the determinant of the Jacobian say about it?
    6. A planar arm with a shoulder, an elbow and a wrist moves a fingertip in a plane. What is its mobility, and its redundancy? What does that mean for the inverse kinematics?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Calculate the Jacobian matrix of the following function:

        $$
        f(x, y) = \begin{bmatrix}
        x^2 y \\
        5 x + \sin y \end{bmatrix}
        $$

    2. For the two-link chain, calculate and interpret the Jacobian and the expressions for the position, velocity and acceleration of the endpoint in the following cases:<br>
       a) the first joint (the joint at the base) is fixed at $0^o$;<br>
       b) the second joint is fixed at $0^o$.

    3. Derive the equations for the kinematics of a two-link pendulum with the angles measured relative to the vertical.

    4. Derive the equations for the kinematics of a two-link system using segment angles, and compare them with the derivation using joint angles.

    5. For the two-link chain, a special case of movement occurs when the endpoint moves along a line passing through the first joint (the joint at the base). A system with this behaviour is known as a polar manipulator (Mussa-Ivaldi, 1986). For simplicity, consider that the two links have the same length, $\ell$. In this case, the two joint angles are related by $2\theta_1+\theta_2=\pi$.<br>
       a) Calculate the Jacobian of this polar manipulator and compare it with the Jacobian of the standard two-link chain. Note the difference between the off-diagonal terms.<br>
       b) Calculate the expressions for the endpoint position, velocity and acceleration.<br>
       c) For the endpoint acceleration of the polar manipulator, identify the tangential, centripetal and Coriolis components, and compare them with the expressions for the standard two-link chain.

    6. [Ex. 2.22, Rade's book] In an internal combustion engine, crank $AB$, shown in the figure below, rotates at a constant 1000 rpm, counterclockwise. For the position $\theta=30^o$, determine: a) the angular velocity of the connecting rod $BD$; b) the speed of the piston $P$.
       <figure><img src="https://github.com/BMClab/BMC/blob/master/images/motor_2rods.png?raw=1" width=450 alt="crank, connecting rod and piston"/></figure>

    7. [Ex. 2.30, Rade's book] The robot shown in the figure below moves in a plane, with velocity $\overrightarrow{v}_0$ and acceleration $\overrightarrow{a}_0$ relative to the floor. Arm $O_1O_2$ rotates with angular velocity $\omega_1=\dot{\theta}_1$ and angular acceleration $\alpha_1=\ddot{\theta}_1$ relative to the robot's body, and arm $O_2P$ rotates with angular velocity $\omega_2=\dot{\theta}_2$ and angular acceleration $\alpha_2=\ddot{\theta}_2$ relative to arm $O_1O_2$, in the directions indicated. Designating by $\ell_1$ the length of arm $O_1O_2$ and by $\ell_2$ the length of arm $O_2P$, obtain the expressions for the velocity $\overrightarrow{v}_P$ and the acceleration $\overrightarrow{a}_P$ of the endpoint $P$ relative to the floor, as functions of the indicated parameters.
       <figure><img src="https://github.com/BMClab/BMC/blob/master/images/robot_arm.png?raw=1" width=450 alt="planar robot with two arms"/></figure>

    8. For the one-link minimum-jerk movement of this notebook, find the instant at which the tangential and centripetal accelerations of the endpoint have the same magnitude. What is the magnitude of the total acceleration then?

    9. Modify the simulated reach so that the endpoint moves in a straight line from $(0.3, 0.3)$ m to $(0.65, 0.65)$ m, ending close to the edge of the workspace. Plot the joint angles and velocities, and explain what happens to the elbow near the end of the movement using the determinant of the Jacobian.

    10. After an elbow injury, a person can flex the elbow only up to $100^o$. Using the arm and the shoulder limits of the example on the workspace of a real arm, calculate how much of the workspace is lost, and draw the region that remains. Which everyday tasks, done with the hand close to the body, would become difficult?

    11. Add a third link to the chain, so that it models the upper arm, the forearm and the hand, with a wrist joint: $\ell_1 = 0.30$ m, $\ell_2 = 0.25$ m and $\ell_3 = 0.08$ m (from the wrist to the fingertip).<br>
        a) Write the forward kinematics of the fingertip and its $2\times3$ Jacobian.<br>
        b) At $\theta_1 = 30^o$, $\theta_2 = 60^o$ and $\theta_3 = 20^o$, find a combination of joint velocities that leaves the fingertip still (hint: `scipy.linalg.null_space`).<br>
        c) Relate the result to the redundancy of the chain.

    12. The quantity $|\det\mathbf{J}|$ is sometimes used as a measure of how easily the endpoint can move in every direction, the manipulability of the chain. Plot it as a colour map over the workspace of the chain with $\ell_1 = \ell_2 = 0.5$ m. Where is it largest, and what is the elbow angle there? Is that where you would choose to hold an object you are working on?

    13. Inverse kinematics amplifies measurement errors near singular configurations. For the chain with $\ell_1 = \ell_2 = 0.5$ m, put the endpoint at 0.99 m from the base, add random noise with a standard deviation of 1 mm to its coordinates 1000 times, and calculate the standard deviation of the elbow angle given by `inverse_kinematics`. Repeat with the endpoint at 0.70 m from the base, and explain the difference with the determinant of the Jacobian.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers to the problems

    1. $J=[[2xy,x^2],[5,\cos y]]$.
    2. Let $u(q)=(\cos q,\sin q)$ and $u_\perp(q)=(-\sin q,\cos q)$. (a) $r=(\ell_1,0)+\ell_2u(\theta_2)$, $v=\ell_2\dot\theta_2u_\perp$, $a=\ell_2\ddot\theta_2u_\perp-\ell_2\dot\theta_2^2u$; reduced Jacobian $\ell_2u_\perp$. (b) Replace the chain by one link of length $\ell_1+\ell_2$, angle $\theta_1$; reduced Jacobian $(\ell_1+\ell_2)u_\perp(\theta_1)$.
    3. For angles $q_1,q_2$ measured from downward vertical with $q_2$ relative to link 1: $r=(\ell_1\sin q_1+\ell_2\sin(q_1+q_2),-\ell_1\cos q_1-\ell_2\cos(q_1+q_2))$. The velocity and acceleration have the same form as item 4 with $u(q)=(\sin q,-\cos q)$ and $u_\perp(q)=(\cos q,\sin q)$.
    4. For absolute segment angles $\phi_1,\phi_2$: $r=\sum_i\ell_i u(\phi_i)$, $v=\sum_i\ell_i\dot\phi_i u_\perp(\phi_i)$, $a=\sum_i\ell_i[\ddot\phi_i u_\perp(\phi_i)-\dot\phi_i^2u(\phi_i)]$. The Jacobian columns are $\ell_i u_\perp(\phi_i)$; joint angles obey $\phi_1=\theta_1$, $\phi_2=\theta_1+\theta_2$.
    5. Under the stated constraint there is one independent coordinate $q=\theta_1$, with $\theta_2=\pi-2q$. Then $r=(0,2\ell\sin q)$, reduced $J=(0,2\ell\cos q)^T$, $v=(0,2\ell\dot q\cos q)$ and $a=(0,2\ell[\ddot q\cos q-\dot q^2\sin q])$. The net transverse component is zero; the full unconstrained Jacobian must be multiplied by $(1,-2)^T$ before comparison. Separate tangential, centripetal and Coriolis contributions depend on the selected unconstrained coordinates, although their sum is the expression above.
    6. $\omega_{BD}=-44.88$ rad/s; piston velocity $v_P=-8.976$ m/s, with the signs and dimensions in the retained figure.
    7. Let $q=\theta_1+\theta_2$, $\dot q=\omega_1+\omega_2$ and $\ddot q=\alpha_1+\alpha_2$. The final velocity is $(v_0+\ell_1\omega_1\cos\theta_1+\ell_2\dot q\cos q,\ell_1\omega_1\sin\theta_1+\ell_2\dot q\sin q)$. Acceleration is $(a_0+\ell_1[\alpha_1\cos\theta_1-\omega_1^2\sin\theta_1]+\ell_2[\ddot q\cos q-\dot q^2\sin q],\ell_1[\alpha_1\sin\theta_1+\omega_1^2\cos\theta_1]+\ell_2[\ddot q\sin q+\dot q^2\cos q])$.
    8. Nontrivial interior instants 0.357325 and 0.642675 s; acceleration magnitude 8.73409 m/s² at each. At the endpoints both components are also zero.
    9. End elbow flexion is about 46.37° for the positive-elbow solution, compared with about 129.79° initially. The Jacobian determinant decreases near the outer boundary, increasing sensitivity to endpoint errors; minimum-jerk endpoint speed still goes to zero at the end.
    10. Remaining area 0.334607 m²; area lost 0.184033 m², about 35.5% of the original workspace. The lost region is mainly close to the shoulder; bringing the hand to the mouth or chest becomes more difficult.
    11. With $\phi_i=\sum_{k\le i}\theta_k$, $r=\sum_i\ell_i u(\phi_i)$ and $J_{:,j}=\sum_{i\ge j}\ell_i u_\perp(\phi_i)$. A unit null-space velocity is approximately $(0.072823,-0.324444,0.943097)$ rad/s, or any scalar multiple. The fingertip has two position constraints and three joint rates, leaving one generic redundant direction.
    12. Maximum $|\det J|=0.25$ m² at elbow angles ±90°, on radius $\sqrt{0.5}=0.707107$ m. This is best for the stated kinematic area measure; comfort, strength and load direction also affect a useful working posture.
    13. With NumPy generator seed 0 and 1000 independent coordinate-noise samples, elbow-angle standard deviations are about 0.828° at 0.99 m and 0.164° at 0.70 m. The near-extended posture has the smaller determinant and greater amplification.

    ///
    """)
    return




@app.cell(hide_code=True)
def _(L1_arm, L2_arm, mo, np, plt):
    _t=np.linspace(0,1,501);_s=10*_t**3-15*_t**4+6*_t**5
    _x=.3+.35*_s;_y=_x.copy()
    _q1,_q2=inverse_kinematics(_x,_y,.5,.5,elbow=1)
    _fig,_ax=plt.subplots(1,3,figsize=(12,3.5))
    _ax[0].plot(_t,np.degrees(np.c_[_q1,_q2]));_ax[0].set(xlabel="Time [s]",ylabel="Joint angle [deg]")
    _ax[1].plot(_t,np.degrees(np.gradient(np.c_[_q1,_q2],_t,axis=0)));_ax[1].set(xlabel="Time [s]",ylabel="Joint rate [deg/s]")
    _a,_b=np.meshgrid(np.linspace(-30,135,160),np.linspace(0,100,120));_a=np.radians(_a);_b=np.radians(_b)
    _xx=L1_arm*np.cos(_a)+L2_arm*np.cos(_a+_b);_yy=L1_arm*np.sin(_a)+L2_arm*np.sin(_a+_b)
    _ax[2].scatter(_xx.ravel(),_yy.ravel(),s=1,alpha=.25);_ax[2].set(xlabel="X [m]",ylabel="Y [m]",aspect="equal",title="Elbow limit 100°")
    _ax[0].legend(["shoulder","elbow"]);_ax[1].legend(["shoulder","elbow"]);_fig.tight_layout()
    mo.accordion({"Final plots — Problems 9 and 10":mo.as_html(_fig)},expanded=False)
    plt.close(_fig)
    return





@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read pages 477–494 of chapter 10 of [Ruina and Pratap's book](https://ruina.tam.cornell.edu/Book/index.html) for a review of differential equations and kinematics.
    - Read section 2.8 of Rade's book about the rotation matrix applied to kinematic chains.
    - Read [Computational Motor Control: Kinematics](https://web.archive.org/web/20221202221546/https://gribblelab.org/teaching/compneuro2012/4_Computational_Motor_Control_Kinematics.html) (archived copy) about the kinematic chain as a model of the human upper limb.
    - For more about the Jacobian matrix, read [Jacobian matrix and determinant](https://www.algebrapracticeproblems.com/jacobian-matrix-determinant/).
    - [Rigid-body transformations in 2D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py): rotation matrices, the formalism that scales to many links and to three dimensions.
    - [The minimum-jerk hypothesis](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/minimum_jerk_hypothesis.py): the trajectory used in the simulations.

    ### Video lectures on the Internet

    - Khan Academy: [Differential Calculus Review](https://www.khanacademy.org/math/differential-calculus)
    - Khan Academy: [Chain Rule Review](https://www.khanacademy.org/math/differential-calculus/dc-chain)
    - [Multivariate Calculus – Jacobian applied](https://www.youtube.com/watch?v=57q-2YxIZss)
    - [Kinematics of the Two Link Arm](https://youtu.be/3WAk8ABj-kg)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Flash T, Hogan N (1985) [The coordination of arm movements: an experimentally confirmed mathematical model](https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985). Journal of Neuroscience, 5(7), 1688–1703.

    - Morasso P (1981) [Spatial control of arm movements](https://doi.org/10.1007/BF00236911). Experimental Brain Research, 42(2), 223–227.

    - Mussa-Ivaldi FA (1986) Compliance. In: Morasso P, Tagliasco V (eds), [Human Movement Understanding: from computational geometry to artificial intelligence](https://books.google.com.br/books?id=ZlZyLKNoAtEC). North-Holland, Amsterdam.

    - Rade D (2017) [Cinemática e Dinâmica para Engenharia](https://www.grupogen.com.br/e-book-cinematica-e-dinamica-para-engenharia). Grupo GEN.

    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](https://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.

    - Siciliano B et al. (2009) [Robotics - Modelling, Planning and Control](https://books.google.com.br/books/about/Robotics.html?hl=pt-BR&id=jPCAFmE-logC). Springer-Verlag London.

    - Zatsiorsky VM (1998) [Kinematics of Human Motion](https://books.google.com.br/books/about/Kinematics_of_Human_Motion.html?id=Pql_xXdbrMcC&redir_esc=y). Champaign, Human Kinetics.
    """)
    return


if __name__ == "__main__":
    app.run()
