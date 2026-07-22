import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Musculoskeletal modeling and simulation

    > Marcos Duarte, Renato Watanabe
    > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab)
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this tutorial

    This notebook is a guided study session, and it is the third step of a sequence:

    1. [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py) built a Hill-type muscle out of its force-length, force-velocity, and activation relationships.
    2. [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py) put that muscle in motion, but the muscle-tendon length $L_{MT}$ was *prescribed*: at every instant we told the muscle how long to be.
    3. This notebook removes that crutch. The muscle now pulls on a real limb: its force accelerates the leg, the leg rotates, the rotation changes $L_{MT}$, and the new length changes the force. Nothing about the movement is dictated in advance.

    That feedback loop is what turns muscle simulation into *musculoskeletal* simulation. The muscle and the skeleton are now two dynamical systems wired into each other, and we have to integrate them together.

    Read a short explanation, predict what the leg and the muscle should do, run the cell, and then use the plot to answer the challenge questions. Keep a scratchpad nearby and write your prediction *before* you reveal each plot.

    You do not need to finish every challenge on the first pass. Run the cells in order first; then come back and change one parameter at a time. The goal is not to make the code run — it already runs. The goal is to connect the free-body diagram, the coupled differential equations, and the movement you see in the plots.

    **Challenge 0.** Before you begin, picture the task we are about to simulate: a seated person with the knee flexed at $90^o$ suddenly activates the quadriceps fully, and the leg swings up into extension. Predict three things and write them down:

    1. As the knee extends, does the quadriceps muscle-tendon unit shorten or lengthen?
    2. Gravity pulls the leg back down. Early in the movement (knee near $90^o$), does gravity help or oppose the extension?
    3. The muscle shortens *while* producing force. From the force-velocity relationship you met in [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py), should the force rise or fall as the shortening speeds up?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1>Contents<span class="tocSkip"></span></h1>
    <div class="toc"><ul class="toc-item"><li><span><a href="#Forward-and-inverse-dynamics" data-toc-modified-id="Forward-and-inverse-dynamics-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Forward and inverse dynamics</a></span></li><li><span><a href="#One-link-system-with-one-DoF-and-one-muscle" data-toc-modified-id="One-link-system-with-one-DoF-and-one-muscle-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>One-link system with one DoF and one muscle</a></span><ul class="toc-item"><li><span><a href="#Moment-arm-is-not-constant" data-toc-modified-id="Moment-arm-is-not-constant-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Moment arm is not constant</a></span></li></ul></li><li><span><a href="#Checkpoint-questions" data-toc-modified-id="Checkpoint-questions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Checkpoint questions</a></span></li><li><span><a href="#Exercises" data-toc-modified-id="Exercises-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exercises</a></span></li><li><span><a href="#Go-deeper" data-toc-modified-id="Go-deeper-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Go deeper</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Module-muscles.py" data-toc-modified-id="Module-muscles.py-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Module muscles.py</a></span></li></ul></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will now simulate the dynamics of the musculoskeletal system with muscle dynamics. You should have read [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py) and [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py) first. In addition, chapter 4 of Nigg and Herzog (2006) is a good introduction to this topic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward and inverse dynamics

    In a typical movement analysis using inverse dynamics, we would measure the leg and thigh positions, calculate the leg and knee linear and angular position and acceleration, and then estimate the net moment of force at the knee joint and the force of the quadriceps muscle. Now we want to do the opposite: perform *forward* dynamics, where muscle force is the input and the movement of the leg is the output. The figure below compares the two approaches.

    <img src="https://raw.githubusercontent.com/BMClab/BMC/master/images/InvDirDyn.png" width="600" alt="Forward and inverse dynamics." />

    *Figure. Inverse dynamics and Forward (or Direct) dynamics approaches for movement analysis (adapted from Zajac and Gordon, 1989).*

    **Guiding questions 1.**

    1. Inverse dynamics starts from measured movement and works backwards to forces. Forward dynamics starts from forces and works forwards to movement. Which one can predict a movement that has *never been measured*, and why does that matter for answering questions such as "what would happen if this muscle were weaker?"
    2. Inverse dynamics gives you the *net* joint moment. Why can it not tell you, on its own, how much of that moment came from the quadriceps and how much from a co-contracting hamstring?
    3. Forward dynamics needs an initial state (position and velocity) and it needs to be integrated in time. Inverse dynamics needs neither. What does that tell you about which of the two accumulates numerical error over a long trial?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One-link system with one DoF and one muscle

    Let's start with the simplest case, a one-link rigid body with one degree of freedom and one muscle.

    Let's simulate the extension movement of the knee. This problem is based on an example from chapter 4 of Nigg and Herzog (2006). We will model the task as a planar movement, the leg and foot as one rigid segment and the rest of the body as fixed, the knee as a revolute joint with a fixed axis of rotation and one degree of freedom, and only one muscle as knee extensor, as illustrated in the figure below.

    <img src="https://raw.githubusercontent.com/BMClab/BMC/master/images/knee.png" width="500" alt="Planar knee model." />

    *Figure. Planar model of the knee with one degree of freedom, one extensor muscle, and the corresponding free-body diagram for the leg. $\theta$: knee angle with the horizontal; $F_M$: force of the quadriceps muscle; $F_J$: force at the knee joint; $r_M$: moment arm of the quadriceps; $r_{cm}$: distance from the knee joint to the center of mass of the leg+foot segment; $mg$: weight of the leg+foot; $L_{MT0}$: muscle-tendon unit length at $\theta = 90^o$.*

    From the figure above, the Newton-Euler equation for the sum of moments of force around the knee joint is:

    $r_M F_M + r_{cm}\cos(\theta)mg = I\dfrac{\mathrm{d}^2 \theta}{\mathrm{d}t^2}$

    where $I$ is the rotational inertia of the leg+foot segment around the knee joint.

    In our convention, the first term on the left-hand side is a positive (extensor) moment of force, and the second term becomes a negative (flexor) moment of force when $\theta$ is greater than $90^o$. The Newton-Euler equation above is a second-order nonlinear differential equation.

    **Guiding questions 2.**

    1. The gravitational term is $r_{cm}\cos(\theta)mg$. Evaluate $\cos\theta$ at $\theta = 90^o$, at $\theta = 135^o$, and at $\theta = 180^o$. At which of these knee angles does gravity resist the extension most strongly, and where does it do nothing at all?
    2. The muscle term is $r_M F_M$. Two muscles could produce the same joint moment with very different forces. What would have to differ between them?
    3. The equation is called *nonlinear*. Point to the term that makes it so.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The length of the quadriceps muscle-tendon unit depends on the knee angle:

    $L_{MT} = L_{MT0} - r_M (\theta - \pi/2)$

    Look carefully at what these equations imply. If we excite the muscle, it generates force; that force is transmitted to the leg segment and accelerates it; the rotation changes the muscle-tendon length and hence the muscle velocity; and those in turn change the muscle force. We already saw in [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py) that muscle force is related to muscle length and velocity through a first-order ODE. So muscle force and joint angle are described by two *coupled* ODEs, and neither can be solved without the other.

    Mathematically, the musculoskeletal system we modeled has as state variables $L_M$, $\theta$, and $\omega = \dot{\theta}$. (Muscle activation $a$ could also be a state variable in other situations, but not in this problem, where we hold it fixed as an input.)

    Let's rewrite the Newton-Euler equation in a clearer form:

    $\dfrac{\mathrm{d}^2 \theta}{\mathrm{d}t^2} = I^{-1} \left[ r_M F_M + r_{cm}\cos(\theta)mg \right]$

    $F_M$ is determined once $L_M$ is known, which according to the muscle model we used in [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py) is the solution of the following first-order ODE:

    $\dfrac{\mathrm{d} L_M}{\mathrm{d}t} = f_v^{-1}\left(\dfrac{F_{SE}(L_{MT}-L_M\cos\alpha)/\cos\alpha - F_{PE}(L_M)}{a f_l(L_M)}\right)$

    and:

    $F_M = F_{SE}(L_{MT}-L_M\cos\alpha)/\cos\alpha$

    To apply numerical methods to these coupled equations, we have to express them as a system of *first-order* differential equations (see [Ordinary differential equation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2FOrdinaryDifferentialEquation.py)). The trick is to promote the angular velocity $\omega$ to a state variable of its own, which turns one second-order equation into two first-order ones:

    $\left\{
    \begin{array}{l l}
    \dfrac{\mathrm{d} L_M}{\mathrm{d}t} = f^{-1}_{v}(L_M, L_{MT}(\theta)), \quad & L_M(t_0) = L_{M0}
    \\
    \dfrac{\mathrm{d} \theta}{\mathrm{d}t} = \omega, \quad & \theta(t_0) = \theta_0
    \\
    \dfrac{\mathrm{d} \omega}{\mathrm{d}t} = I^{-1} \left[ r_M F_M(L_M, \theta) + r_{cm}\cos(\theta)mg \right], \quad & \omega(t_0) = \omega_0
    \end{array}
    \right.$

    **Guiding questions 3.**

    1. In $L_{MT} = L_{MT0} - r_M(\theta - \pi/2)$, the sign in front of $r_M$ is negative. Confirm that this means the quadriceps *shortens* as the knee extends ($\theta$ grows past $\pi/2$). Does that agree with the prediction you wrote for Challenge 0?
    2. The three equations are coupled. Find the term in the third equation that depends on the first equation's state, and the term in the first equation that depends on the second equation's state. That pair of arrows is the feedback loop.
    3. Why is $\mathrm{d}\theta/\mathrm{d}t = \omega$ not a physical statement about muscles at all? What is it doing in the system?

    **Challenge 1.** In [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py), $L_{MT}$ was a function you wrote by hand, `lmt_eq(t, lmt0)` — a function of *time*. Here, $L_{MT}$ is a function of $\theta$, and $\theta$ is a state the solver is computing as it goes. Explain in one sentence why this makes the problem qualitatively harder than the previous notebook, even though the muscle model is identical.

    Let's solve this problem in Python. First, let's import the necessary Python libraries and customize the environment:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy.integrate import odeint
    from scipy import interpolate

    matplotlib.rcParams["lines.linewidth"] = 2
    matplotlib.rcParams["font.size"] = 12
    matplotlib.rcParams["lines.markersize"] = 4
    matplotlib.rc("axes", grid=True, labelsize=12, titlesize=13, ymargin=0.01)
    matplotlib.rc("legend", numpoints=1, fontsize=10)
    return interpolate, np, odeint, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The muscle model itself is the class `Thelen2003`, the same one used in [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py). Instead of importing it from an external `muscles.py` file, we define it at the end of this notebook, so the notebook is self-contained and runs anywhere, including in the browser at [marimo.app](https://marimo.app/).

    Same for its parameters and initial states, which we write directly as Python dictionaries. These are the values from the exercise in chapter 4 of Nigg and Herzog (2006):
    """)
    return


@app.cell
def _():
    parameters = {
        "id": "",
        "name": "",
        "u_max": 1.0,  # maximum value for muscle excitation
        "u_min": 0.01,  # minimum value for muscle excitation
        "t_act": 0.015,  # activation time constant [s]
        "t_deact": 0.050,  # deactivation time constant [s]
        "lmopt": 0.093,  # optimal CE length
        "alpha0": 0.0,  # pennation angle at rest
        "fm0": 7400.0,  # maximum isometric muscle force
        "gammal": 0.45,  # CE force-length shape factor
        "kpe": 5.0,  # PE exponential shape factor
        "epsm0": 0.6,  # PE passive muscle strain due to maximum isometric force
        "vmmax": 10.0,  # CE force-velocity maximum velocity (concentric)
        "fmlen": 1.4,  # CE force-velocity maximum force (lengthening)
        "af": 0.25,  # CE force-velocity shape factor
        "ltslack": 0.223,  # tendon slack length
        "epst0": 0.04,  # tendon strain at the maximal isometric muscle force
        "kttoe": 3.0,  # tendon linear scale factor
    }
    return (parameters,)


@app.cell
def _(np):
    states = {
        "id": "",
        "name": "",
        "lmt0": 0.310,  # muscle-tendon length at the initial knee angle of 90 deg
        "lm0": 0.087,  # initial muscle length
        "lt0": np.nan,  # initial tendon length (computed from the others if NaN)
    }
    return (states,)


@app.cell
def _(Thelen2003, parameters, states):
    ms = Thelen2003(parameters.copy(), states.copy())
    return (ms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that $L_{T0} = L_{MT0} - L_{M0} = 0.310 - 0.087 = 0.223$ m, exactly the tendon slack length. The simulation therefore starts with the tendon slack and the muscle carrying no force at all.

    **Before you run the next cell**, predict the three curves: the initial fiber length is $0.087$ m and the optimal length is $0.093$ m, so will the muscle start on the ascending or the descending limb of the active force-length curve? And with the tendon exactly at slack, where on the tendon curve should the red marker sit?
    """)
    return


@app.cell
def _(ms):
    ms.muscle_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the skeletal side of the model. These are the properties of the leg+foot segment and of the knee joint, also from Nigg and Herzog (2006):
    """)
    return


@app.cell
def _():
    LMT0 = 0.310  # muscle-tendon length at theta = 90 deg [m]
    RCM = 0.264  # knee joint to the center of mass of the leg+foot [m]
    MASS = 10.0  # mass of the leg+foot segment [kg]
    G = 9.8  # acceleration of gravity [m/s2]
    INERTIA = 0.1832  # rotational inertia of the leg+foot about the knee [kg.m2]
    return G, INERTIA, LMT0, MASS, RCM


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, the moment arm of the quadriceps. We start with the simplest possible assumption — that it does not change with the knee angle — and we will question that assumption later in the notebook:
    """)
    return


@app.cell
def _():
    def rm_const(theta):
        """Vastus intermedius moment arm, assumed constant [m]."""

        return 0.033

    return (rm_const,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now two helper functions for the muscle side of the problem. They are the same equations used in [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py), specialized to this problem by fixing the activation at $a = 1$ (the muscle is fully excited throughout) and the pennation angle at $\alpha = 0$ (a fusiform muscle, so $\cos\alpha = 1$):

    - `muscle_velocity` inverts the force-velocity relationship: given the current fiber length and the current muscle-tendon length, it returns how fast the fiber is shortening.
    - `muscle_force` evaluates the force-length and force-velocity relationships to return the muscle force in newtons.

    Note the guard `lm = max(lm, 0.1 * lmopt)`. That is the $f_l(L_M) \rightarrow 0$ singularity you met in the previous notebook, clamped so that the solver cannot divide by zero.
    """)
    return


@app.cell
def _(ms, np):
    def muscle_velocity(lm, lmt):
        """Muscle fiber velocity from the muscle-tendon force equilibrium."""

        lmopt = ms.P["lmopt"]
        ltslack = ms.P["ltslack"]
        a = 1  # the muscle is fully activated
        alpha = 0  # fusiform muscle, cos(alpha) = 1
        lm = max(lm, 0.1 * lmopt)  # avoid the fl -> 0 singularity

        lt = lmt - lm * np.cos(alpha)
        fse = ms.force_se(lt=lt, ltslack=ltslack)
        fpe = ms.force_pe(lm=lm / lmopt)
        fl = ms.force_l(lm=lm / lmopt)
        fce_t = fse / np.cos(alpha) - fpe
        vm = ms.velo_fm(fm=fce_t, a=a, fl=fl)

        return vm

    def muscle_force(lm, vm):
        """Muscle force [N] from the force-length and force-velocity relationships."""

        fm0 = ms.P["fm0"]
        lmopt = ms.P["lmopt"]
        a = 1
        lm = max(lm, 0.1 * lmopt)

        fl = ms.force_l(lm=lm / lmopt)
        fpe = ms.force_pe(lm=lm / lmopt)
        fm = ms.force_vm(vm=vm, fl=fl, lmopt=lmopt, a=a) + fpe

        return fm * fm0

    return muscle_force, muscle_velocity


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And here is the system of first-order ODEs itself — the three equations we wrote above, in code. Read it next to the mathematics: `state` holds $[L_M, \theta, \omega]$ and the function returns $[\dot{L}_M, \dot{\theta}, \dot{\omega}]$, in the same order.

    The moment arm is passed in as an argument, `rm_fun`, rather than hard-coded. That way the *same* equations can be solved with a constant moment arm or with a measured, angle-dependent one, and we can compare the two fairly.
    """)
    return


@app.cell
def _(G, INERTIA, LMT0, MASS, RCM, muscle_force, muscle_velocity, np):
    def onelink_eq(state, t, rm_fun):
        """System of first-order ODEs for the one-link system.

        Parameters
        state : array_like, [lm, theta, omega]
            muscle length [m], knee angle [rad], knee angular velocity [rad/s]
        t : float
            time instant [s]
        rm_fun : callable
            moment arm of the knee extensor [m] as a function of theta [rad]
        Returns
        derivatives : list, [lmd, omega, thetadd]
        """

        lm, theta, omega = state

        rm = rm_fun(theta)
        lmt = LMT0 - rm * (theta - np.pi / 2)
        lmd = muscle_velocity(lm, lmt)
        thetadd = (
            rm * muscle_force(lm, lmd) + RCM * np.cos(theta) * MASS * G
        ) / INERTIA

        return [lmd, omega, thetadd]

    return (onelink_eq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's set the initial states and use `scipy.integrate.odeint` to solve the ODEs. Unlike the fixed-step forward Euler solver we wrote by hand in the previous notebook, `odeint` chooses its own internal step size adaptively; the `time` array only says where we want the answer reported.

    The movement is fast, so we only simulate $0.12$ s.

    **Before you run the next two cells**, sketch the knee angle you expect over those $0.12$ s: starting at $90^o$, fully activated quadriceps. Does the angle rise linearly, or does it accelerate? Roughly how far does the knee get?
    """)
    return


@app.cell
def _(np, odeint, onelink_eq, rm_const):
    state0 = [0.087, np.pi / 2, 0]  # [lm (m), theta (rad), omega (rad/s)]
    time = np.arange(0, 0.12, 0.001)
    data = odeint(onelink_eq, state0, time, args=(rm_const,))
    return data, state0, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `odeint` returns only the three state variables. The function below replays the solution to recover every muscle-tendon variable at each instant (lengths, velocity, and the separate force contributions), so that we can reuse the plotting routine of the muscle model. It also plots the simulated knee angle and angular velocity.
    """)
    return


@app.cell
def _(LMT0, np, plt):
    def sim_plot(time, data, ms, rm_fun):
        """Plot the knee kinematics and rebuild the muscle-tendon variables."""

        fm0 = ms.P["fm0"]
        lmopt = ms.P["lmopt"]
        ltslack = ms.P["ltslack"]
        a = 1
        alpha = 0

        datam = []
        for i, t in enumerate(time):
            lm = data[i, 0]
            theta = data[i, 1]
            lmt = LMT0 - rm_fun(theta) * (theta - np.pi / 2)
            lt = lmt - lm * np.cos(alpha)
            fl = ms.force_l(lm=lm / lmopt)
            fpe = ms.force_pe(lm=lm / lmopt)
            fse = ms.force_se(lt=lt, ltslack=ltslack)
            fce_t = fse / np.cos(alpha) - fpe
            vm = ms.velo_fm(fm=fce_t, a=a, fl=fl, lmopt=lmopt)
            fm = ms.force_vm(vm=vm, fl=fl, lmopt=lmopt, a=a) + fpe
            datam.append(
                [t, lmt, lm, lt, vm, fm * fm0, fse * fm0, a * fl * fm0, fpe * fm0, alpha]
            )

        datam = np.array(datam)

        _fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(9, 4))
        axs[0].plot(time, data[:, 1] * 180 / np.pi)
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel(r"Knee angle $(^o)$")
        axs[1].plot(time, data[:, 2] * 180 / np.pi)
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel(r"Knee velocity $(^o/s)$")
        plt.suptitle("Simulation of the knee extension")
        plt.tight_layout()
        plt.show()

        return datam

    return (sim_plot,)


@app.cell
def _(data, ms, rm_const, sim_plot, time):
    data2 = sim_plot(time, data, ms, rm_const)
    return (data2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And here are the plots for the simulated muscle mechanics, using the same six-panel layout as in the previous notebook.

    **Before you run it**, predict the $L_{MT}$ panel. In the previous notebook $L_{MT}$ was a flat line during the isometric runs because we prescribed it. Here nobody prescribed it. What should it do now?
    """)
    return


@app.cell
def _(data2, ms):
    axs_lm = ms.lm_plot(data2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 2.** Look at the knee angle plot together with the muscle force plot. The force does *not* stay at its maximum even though activation is pinned at $a = 1$ the whole time. Name the two effects from the muscle model that pull the force down as the movement proceeds. (Hint: one is about how fast the fiber is shortening, the other about how short it has become.)

    **Challenge 3.** The knee velocity keeps increasing throughout the simulation. Using the Newton-Euler equation, explain what would eventually have to happen to the angular acceleration for the leg to stop accelerating, and identify which term would be responsible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here and answer these before moving on. They tie together the pieces you have simulated.

    1. What exactly makes this a *musculoskeletal* simulation rather than the muscle simulation of the previous notebook? Point to the single variable that changed role.
    2. The state vector is $[L_M, \theta, \omega]$. Why is $\omega$ in there, when the physical question we asked was only about the knee angle?
    3. Muscle activation was held at $a = 1$ and was *not* a state variable here. Under what circumstance would you have to promote it to a fourth state?
    4. The muscle force falls during the movement even though activation never changes. Which two properties of the contractile element are responsible?
    5. A moment arm does two jobs in these equations at once. Name both, and explain why making a moment arm larger is not automatically an advantage.
    6. In the previous notebook we integrated with a hand-written forward Euler loop; here we used `odeint`. What does an adaptive solver buy you in a problem where the muscle force can change very quickly?

    ## Exercises

    Work through these in order if you are studying independently. For each one, predict the outcome first, then modify the code and compare.

    1. **Activation level.** The helper functions fix $a = 1$. Change `a` to $0.5$ and then $0.1$ in both `muscle_velocity` and `muscle_force`, and re-run. Does the knee still reach full extension? Is the relationship between activation and final angle linear?
    2. **Gravity.** Set `G = 0` and re-run. How much of the movement was being resisted by the weight of the leg? Then set `MASS = 20` and explain the result using both terms of the Newton-Euler equation.
    3. **Starting posture.** Start the simulation from a more extended knee, `state0 = [0.087, 2*np.pi/3, 0]`. Careful: the initial muscle length is no longer consistent with that angle. Work out the correct $L_{M0}$ from $L_{MT} = L_{MT0} - r_M(\theta - \pi/2)$, assuming the tendon starts at slack length.
    4. **Simulation duration.** Extend `time` to $0.3$ s and re-run with the constant moment arm. What happens, and why is the result no longer physically meaningful? Relate your answer to Challenge 5.
    5. **A stronger muscle.** Double `fm0` in the parameters. Does the knee extend twice as fast? Explain any nonlinearity using the force-velocity relationship.
    6. **Tendon compliance.** Make the tendon more compliant by increasing `epst0` to $0.08$. How does the extra tendon stretch change the fiber velocity early in the movement?
    7. **Moment arm sensitivity.** Multiply the output of `rm_const` by $1.5$ and re-run. Compare against the measured, angle-dependent moment arm. Does a uniformly larger moment arm reproduce the measured result? What does this tell you about the value of measured musculoskeletal geometry?
    8. **Adding an antagonist.** Sketch (you do not have to code it) how you would extend `onelink_eq` to include a hamstring acting as a knee flexor. What new parameters would you need, and how many state variables would the system have?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py) — where the force-length, force-velocity, and activation functions used here were derived.
    - [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py) — the same muscle model, but with the muscle-tendon length prescribed instead of produced by the movement.
    - [Introduction to numerical solution of Ordinary Differential Equations](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2FOrdinaryDifferentialEquation.py) — how systems of first-order ODEs are set up and solved.
    - [OpenSim](https://simtk.org/home/opensim) — the musculoskeletal modeling software the moment-arm data came from, and the natural next tool once the models grow beyond one link and one muscle.

    ## References

    - Nigg BM and Herzog W (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678). 3rd Edition. Wiley.
    - Thelen DG (2003) [Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older adults](http://homepages.cae.wisc.edu/~thelen/pubs/jbme03.pdf). Journal of Biomechanical Engineering, 125(1):70-77.
    - Zajac FE, Gordon ME (1989) [Determining muscle's force and action in multi-articular movement](https://drive.google.com/open?id=0BxbW72zV7WmUcC1zSGpEOUxhWXM&authuser=0). Exercise and Sport Sciences Reviews, 17, 187-230.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Module muscles.py

    This is the same `Thelen2003` class used in [Muscle simulation](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_simulation.py), defined here rather than imported so that the notebook is self-contained.
    """)
    return


@app.cell
def _(np):
    """Muscle modeling and simulation."""

    import configparser

    __author__ = "Marcos Duarte, https://github.com/BMClab/BMC"
    __version__ = "muscles.py v.1.03 2026/07/07"

    class Thelen2003:
        """Thelen (2003) muscle model."""

        def __init__(self, parameters=None, states=None):
            if parameters is not None:
                self.set_parameters(parameters)
            if states is not None:
                self.set_states(states)

            self.lm_data = []
            self.act_data = []

        def set_parameters(self, var=None):
            """Load and set parameters for the muscle model."""
            if var is None:
                var = "./../data/muscle_parameter.txt"
            if isinstance(var, str):
                self.P = self.config_parser(var, "parameters")
            elif isinstance(var, dict):
                self.P = var
            else:
                raise ValueError("Wrong parameters!")

            print(
                "The parameters were successfully loaded "
                + "and are stored in the variable P."
            )

        def set_states(self, var=None):
            """Load and set states for the muscle model."""
            if var is None:
                var = "./../data/muscle_state.txt"
            if isinstance(var, str):
                self.S = self.config_parser(var, "states")
            elif isinstance(var, dict):
                self.S = var
            else:
                raise ValueError("Wrong states!")

            print(
                "The states were successfully loaded "
                + "and are stored in the variable S."
            )

        def config_parser(self, filename, var):
            """ """
            parser = configparser.ConfigParser()
            parser.optionxform = str  # make option names case sensitive
            read_files = parser.read(filename)
            if not read_files or not parser.sections():
                raise ValueError(f"Could not read {var} from '{filename}'.")
            var = {}
            for key, value in parser.items(parser.sections()[0]):
                if key.lower() in ["name", "id"]:
                    var.update({key: value})
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        print(f"{key} value {value} was replaced by NaN.")
                        value = np.nan
                    var.update({key: value})

            return var

        def force_l(self, lm, gammal=None):
            """Thelen (2003) force of the contractile element vs. muscle length.

            Parameters
            lm : float
                normalized muscle fiber length
            gammal : float, optional (default from parameter file)
                shape factor
            Returns
            fl : float
                normalized force of the muscle contractile element
            """

            if gammal is None:
                gammal = self.P["gammal"]
            fl = np.exp(-((lm - 1) ** 2) / gammal)
            return fl

        def force_pe(self, lm, kpe=None, epsm0=None):
            """Thelen (2003) force of the muscle parallel element vs. muscle length.

            Parameters
            lm : float
                normalized muscle fiber length
            kpe : float, optional (default from parameter file)
                exponential shape factor
            epsm0 : float, optional (default from parameter file)
                passive muscle strain due to maximum isometric force
            Returns
            fpe : float
                normalized force of the muscle parallel (passive) element
            """

            if kpe is None:
                kpe = self.P["kpe"]
            if epsm0 is None:
                epsm0 = self.P["epsm0"]

            if lm <= 1:
                fpe = 0
            else:
                fpe = (np.exp(kpe * (lm - 1) / epsm0) - 1) / (np.exp(kpe) - 1)

            return fpe

        def force_se(self, lt, ltslack=None, epst0=None, kttoe=None):
            """Thelen (2003) force-length relationship of tendon vs. tendon length.

            Parameters
            lt : float
                tendon length (normalized or not)
            ltslack : float, optional (default from parameter file)
                tendon slack length (normalized or not)
            epst0 : float, optional (default from parameter file)
                tendon strain at the maximal isometric muscle force
            kttoe : float, optional (default from parameter file)
                linear scale factor
            Returns
            fse : float
                normalized force of the tendon series element
            """

            if ltslack is None:
                ltslack = self.P["ltslack"]
            if epst0 is None:
                epst0 = self.P["epst0"]
            if kttoe is None:
                kttoe = self.P["kttoe"]

            epst = (lt - ltslack) / ltslack
            fttoe = 0.33
            # values from OpenSim Thelen2003Muscle
            epsttoe = 0.99 * epst0 * np.e**3 / (1.66 * np.e**3 - 0.67)
            ktlin = 0.67 / (epst0 - epsttoe)
            #
            if epst <= 0:
                fse = 0
            elif epst <= epsttoe:
                fse = fttoe / (np.exp(kttoe) - 1) * (np.exp(kttoe * epst / epsttoe) - 1)
            else:
                fse = ktlin * (epst - epsttoe) + fttoe

            return fse

        def velo_fm(self, fm, a, fl, lmopt=None, vmmax=None, fmlen=None, af=None):
            """Thelen (2003) velocity of the force-velocity relationship vs. CE force.

            Parameters
            fm : float
                normalized muscle force
            a : float
                muscle activation level
            fl : float
                normalized muscle force due to the force-length relationship
            lmopt : float, optional (default from parameter file)
                optimal muscle fiber length
            vmmax : float, optional (default from parameter file)
                normalized maximum muscle velocity for concentric activation
            fmlen : float, optional (default from parameter file)
                normalized maximum force generated at the lengthening phase
            af : float, optional (default from parameter file)
                shape factor
            Returns
            vm : float
                velocity of the muscle
            """

            if lmopt is None:
                lmopt = self.P["lmopt"]
            if vmmax is None:
                vmmax = self.P["vmmax"]
            if fmlen is None:
                fmlen = self.P["fmlen"]
            if af is None:
                af = self.P["af"]

            if fm <= a * fl:  # isometric and concentric activation
                if fm > 0:
                    b = a * fl + fm / af
                else:
                    b = a * fl
            else:  # eccentric activation
                asyE_thresh = 0.95  # from OpenSim Thelen2003Muscle
                if fm < a * fl * fmlen * asyE_thresh:
                    b = (2 + 2 / af) * (a * fl * fmlen - fm) / (fmlen - 1)
                else:
                    fm_thresh = a * fl * fmlen * asyE_thresh
                    b = (2 + 2 / af) * (a * fl * fmlen - fm_thresh) / (fmlen - 1)

            vm = (0.25 + 0.75 * a) * (fm - a * fl) / b
            vm = vm * vmmax * lmopt

            return vm

        def force_vm(self, vm, a, fl, lmopt=None, vmmax=None, fmlen=None, af=None):
            """Thelen (2003) force of the contractile element vs. muscle velocity.

            Parameters
            vm : float
                muscle velocity
            a : float
                muscle activation level
            fl : float
                normalized muscle force due to the force-length relationship
            lmopt : float, optional (default from parameter file)
                optimal muscle fiber length
            vmmax : float, optional (default from parameter file)
                normalized maximum muscle velocity for concentric activation
            fmlen : float, optional (default from parameter file)
                normalized normalized maximum force generated at the lengthening phase
            af : float, optional (default from parameter file)
                shape factor
            Returns
            fvm : float
                normalized force of the muscle contractile element
            """

            if lmopt is None:
                lmopt = self.P["lmopt"]
            if vmmax is None:
                vmmax = self.P["vmmax"]
            if fmlen is None:
                fmlen = self.P["fmlen"]
            if af is None:
                af = self.P["af"]

            vmmax = vmmax * lmopt
            if vm <= 0:  # isometric and concentric activation
                fvm = (
                    af
                    * a
                    * fl
                    * (4 * vm + vmmax * (3 * a + 1))
                    / (-4 * vm + vmmax * af * (3 * a + 1))
                )
            else:  # eccentric activation
                fvm = (
                    a
                    * fl
                    * (
                        af * vmmax * (3 * a * fmlen - 3 * a + fmlen - 1)
                        + 8 * vm * fmlen * (af + 1)
                    )
                    / (
                        af * vmmax * (3 * a * fmlen - 3 * a + fmlen - 1)
                        + 8 * vm * (af + 1)
                    )
                )

            return fvm

        def lmt_eq(self, t, lmt0=None):
            """Equation for muscle-tendon length."""

            if lmt0 is None:
                lmt0 = self.S["lmt0"]

            return lmt0

        def vm_eq(self, t, lm, lm0, lmt0, lmopt, ltslack, alpha0, vmmax, fm0):
            """Equation for muscle velocity."""

            if lm < 0.1 * lmopt:
                lm = 0.1 * lmopt
            # lt0 = lmt0 - lm0*np.cos(alpha0)
            a = self.activation(t)
            lmt = self.lmt_eq(t, lmt0)
            alpha = self.penn_ang(lmt=lmt, lm=lm, lm0=lm0, alpha0=alpha0)
            lt = lmt - lm * np.cos(alpha)
            fse = self.force_se(lt=lt, ltslack=ltslack)
            fpe = self.force_pe(lm=lm / lmopt)
            fl = self.force_l(lm=lm / lmopt)
            fce_t = fse / np.cos(alpha) - fpe
            # if fce_t < 0: fce_t=0
            vm = self.velo_fm(fm=fce_t, a=a, fl=fl)

            return vm

        def lm_sol(
            self,
            fun=None,
            t0=0,
            t1=3,
            dt=0.001,
            lm0=None,
            lmt0=None,
            ltslack=None,
            lmopt=None,
            alpha0=None,
            vmmax=None,
            fm0=None,
            show=True,
            axs=None,
        ):
            """Forward Euler ODE solver for muscle length."""

            if lm0 is None:
                lm0 = self.S["lm0"]
            if lmt0 is None:
                lmt0 = self.S["lmt0"]
            if ltslack is None:
                ltslack = self.P["ltslack"]
            if alpha0 is None:
                alpha0 = self.P["alpha0"]
            if lmopt is None:
                lmopt = self.P["lmopt"]
            if vmmax is None:
                vmmax = self.P["vmmax"]
            if fm0 is None:
                fm0 = self.P["fm0"]

            if fun is None:
                fun = self.vm_eq

            nsteps = int(round((t1 - t0) / dt))
            lm = lm0
            data = []
            for i in range(nsteps):
                t = t0 + i * dt
                lm = max(lm, 0.1 * lmopt)
                data.append(
                    self.calc_data(t, lm, lm0, lmt0, ltslack, lmopt, alpha0, fm0)
                )
                vm = fun(t, lm, lm0, lmt0, lmopt, ltslack, alpha0, vmmax, fm0)
                lm = lm + dt * vm

            data = np.array(data)
            self.lm_data = data
            if show:
                self.lm_plot(data, axs)

            return data

        def calc_data(self, t, lm, lm0, lmt0, ltslack, lmopt, alpha0, fm0):
            """Calculus of muscle-tendon variables."""

            a = self.activation(t)
            lmt = self.lmt_eq(t, lmt0=lmt0)
            alpha = self.penn_ang(lmt=lmt, lm=lm, lm0=lm0, alpha0=alpha0)
            lt = lmt - lm * np.cos(alpha)
            fl = self.force_l(lm=lm / lmopt)
            fpe = self.force_pe(lm=lm / lmopt)
            fse = self.force_se(lt=lt, ltslack=ltslack)
            fce_t = fse / np.cos(alpha) - fpe
            vm = self.velo_fm(fm=fce_t, a=a, fl=fl, lmopt=lmopt)
            fm = self.force_vm(vm=vm, fl=fl, lmopt=lmopt, a=a) + fpe
            data = [
                t,
                lmt,
                lm,
                lt,
                vm,
                fm * fm0,
                fse * fm0,
                a * fl * fm0,
                fpe * fm0,
                alpha,
            ]

            return data

        def muscle_plot(self, a=1, axs=None):
            """Plot muscle-tendon relationships with length and velocity."""

            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib is not available.")
                return

            if axs is None:
                fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))

            lmopt = self.P["lmopt"]
            ltslack = self.P["ltslack"]
            vmmax = self.P["vmmax"]
            alpha0 = self.P["alpha0"]
            fm0 = self.P["fm0"]
            lm0 = self.S["lm0"]
            lmt0 = self.S["lmt0"]
            lt0 = self.S["lt0"]
            if np.isnan(lt0):
                lt0 = lmt0 - lm0 * np.cos(alpha0)

            lm = np.linspace(0, 2, 101)
            lt = np.linspace(0, 1, 101) * 0.05 + 1
            vm = np.linspace(-1, 1, 101) * vmmax * lmopt
            fl = np.zeros(lm.size)
            fpe = np.zeros(lm.size)
            fse = np.zeros(lt.size)
            fvm = np.zeros(vm.size)

            fl_lm0 = self.force_l(lm0 / lmopt)
            fpe_lm0 = self.force_pe(lm0 / lmopt)
            fm_lm0 = fl_lm0 + fpe_lm0
            ft_lt0 = self.force_se(lt0, ltslack) * fm0

            for i in range(101):
                fl[i] = self.force_l(lm[i])
                fpe[i] = self.force_pe(lm[i])
                fse[i] = self.force_se(lt[i], ltslack=1)
                fvm[i] = self.force_vm(vm[i], a=a, fl=fl_lm0)

            lm = lm * lmopt
            lt = lt * ltslack
            fse = fse * fm0
            fvm = fvm * fm0

            xlim = self.margins(lm, margin=0.05, minmargin=False)
            axs[0].set_xlim(xlim)
            ylim = self.margins([0, 2], margin=0.05)
            axs[0].set_ylim(ylim)
            axs[0].plot(lm, fl, "b", label="Active")
            axs[0].plot(lm, fpe, "b--", label="Passive")
            axs[0].plot(lm, fl + fpe, "b:", label="")
            axs[0].plot([lm0, lm0], [ylim[0], fm_lm0], "k:", lw=2, label="")
            axs[0].plot([xlim[0], lm0], [fm_lm0, fm_lm0], "k:", lw=2, label="")
            axs[0].plot(lm0, fm_lm0, "o", ms=6, mfc="r", mec="r", mew=2, label="fl(LM0)")
            axs[0].legend(loc="best", frameon=True, framealpha=0.5)
            axs[0].set_xlabel("Length [m]")
            axs[0].set_ylabel("Scale factor")
            axs[0].xaxis.set_major_locator(plt.MaxNLocator(4))
            axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))
            axs[0].set_title("Muscle F-L (a=1)")

            xlim = self.margins(
                [0, np.min(vm), np.max(vm)], margin=0.05, minmargin=False
            )
            axs[1].set_xlim(xlim)
            ylim = self.margins([0, fm0 * 1.2, np.max(fvm) * 1.5], margin=0.025)
            axs[1].set_ylim(ylim)
            axs[1].plot(vm, fvm, label="")
            axs[1].set_xlabel(r"$\mathbf{^{CON}}\;$ Velocity [m/s] $\;\mathbf{^{EXC}}$")
            axs[1].plot([0, 0], [ylim[0], fvm[50]], "k:", lw=2, label="")
            axs[1].plot([xlim[0], 0], [fvm[50], fvm[50]], "k:", lw=2, label="")
            axs[1].plot(0, fvm[50], "o", ms=6, mfc="r", mec="r", mew=2, label="FM0(LM0)")
            axs[1].plot(xlim[0], fm0, "+", ms=10, mfc="r", mec="r", mew=2, label="")
            axs[1].text(vm[0], fm0, "FM0")
            axs[1].legend(loc="upper right", frameon=True, framealpha=0.5)
            axs[1].set_ylabel("Force [N]")
            axs[1].xaxis.set_major_locator(plt.MaxNLocator(4))
            axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))
            axs[1].set_title("Muscle F-V (a=1)")

            xlim = self.margins(
                [lt0, ltslack, np.min(lt), np.max(lt)], margin=0.05, minmargin=False
            )
            axs[2].set_xlim(xlim)
            ylim = self.margins([ft_lt0, 0, np.max(fse)], margin=0.05)
            axs[2].set_ylim(ylim)
            axs[2].plot(lt, fse, label="")
            axs[2].set_xlabel("Length [m]")
            axs[2].plot([lt0, lt0], [ylim[0], ft_lt0], "k:", lw=2, label="")
            axs[2].plot([xlim[0], lt0], [ft_lt0, ft_lt0], "k:", lw=2, label="")
            axs[2].plot(lt0, ft_lt0, "o", ms=6, mfc="r", mec="r", mew=2, label="FT(LT0)")
            axs[2].legend(loc="upper left", frameon=True, framealpha=0.5)
            axs[2].set_ylabel("Force [N]")
            axs[2].xaxis.set_major_locator(plt.MaxNLocator(4))
            axs[2].yaxis.set_major_locator(plt.MaxNLocator(4))
            axs[2].set_title("Tendon")
            plt.suptitle("Muscle-tendon mechanics")
            plt.tight_layout(w_pad=0.1)
            plt.show()

            return axs

        def lm_plot(self, x, axs=None):
            """Plot the muscle-tendon simulation results.
            data = [t, lmt, lm, lt, vm, fm*fm0, fse*fm0, fl*fm0, fpe*fm0, alpha]
            """

            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib is not available.")
                return

            if axs is None:
                fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(9, 6))

            axs[0, 0].plot(x[:, 0], x[:, 1], "b", label="LMT")
            lmt = x[:, 2] * np.cos(x[:, 9]) + x[:, 3]
            if np.sum(x[:, 9]) > 0:
                axs[0, 0].plot(x[:, 0], lmt, "g--", label=r"$LM \cos \alpha + LT$")
            else:
                axs[0, 0].plot(x[:, 0], lmt, "g--", label=r"LM+LT")
            ylim = self.margins(x[:, 1], margin=0.1)
            axs[0, 0].set_ylim(ylim)
            axs[0, 0].legend(framealpha=0.5, loc="best")

            axs[0, 1].plot(x[:, 0], x[:, 3], "b")
            # axs[0, 1].plot(x[:, 0], lt0*np.ones(len(x)), 'r')
            ylim = self.margins(x[:, 3], margin=0.1)
            axs[0, 1].set_ylim(ylim)

            axs[1, 0].plot(x[:, 0], x[:, 2], "b")
            # axs[1, 0].plot(x[:, 0], lmopt*np.ones(len(x)), 'r')
            ylim = self.margins(x[:, 2], margin=0.1)
            axs[1, 0].set_ylim(ylim)

            axs[1, 1].plot(x[:, 0], x[:, 4], "b")
            ylim = self.margins(x[:, 4], margin=0.1)
            axs[1, 1].set_ylim(ylim)

            axs[2, 0].plot(x[:, 0], x[:, 5], "b", label="Muscle")
            axs[2, 0].plot(x[:, 0], x[:, 6], "g--", label="Tendon")
            ylim = self.margins(x[:, [5, 6]], margin=0.1)
            axs[2, 0].set_ylim(ylim)
            axs[2, 0].set_xlabel("Time (s)")
            axs[2, 0].legend(framealpha=0.5, loc="best")

            axs[2, 1].plot(x[:, 0], x[:, 8], "b", label="PE")
            ylim = self.margins(x[:, 8], margin=0.1)
            axs[2, 1].set_ylim(ylim)
            axs[2, 1].set_xlabel("Time (s)")
            axs[2, 1].legend(framealpha=0.5, loc="best")

            ylabel = [
                r"$L_{MT}\,(m)$",
                r"$L_{T}\,(m)$",
                r"$L_{M}\,(m)$",
                r"$V_{CE}\,(m/s)$",
                r"$Force\,(N)$",
                r"$Force\,(N)$",
            ]
            for i, axi in enumerate(axs.flat):
                axi.set_ylabel(ylabel[i])
                axi.yaxis.set_major_locator(plt.MaxNLocator(4))
                fig.align_ylabels(axs)
                # axi.yaxis.set_label_coords(-.2, 0.5)

            plt.suptitle("Simulation of muscle-tendon mechanics")
            plt.tight_layout()
            plt.show()

            return axs

        def penn_ang(self, lmt, lm, lt=None, lm0=None, alpha0=None):
            """Pennation angle.

            Parameters
            lmt : float
                muscle-tendon length
            lt : float, optional (default=None)
                tendon length
            lm : float, optional (default=None)
                muscle fiber length
            lm0 : float, optional (default from states file)
                initial muscle fiber length
            alpha0 : float, optional (default from parameter file)
                initial pennation angle
            Returns
            alpha : float
                pennation angle
            """

            if lm0 is None:
                lm0 = self.S["lm0"]
            if alpha0 is None:
                alpha0 = self.P["alpha0"]

            alpha = alpha0
            if alpha0 != 0:
                w = lm0 * np.sin(alpha0)
                if lm is not None:
                    cosalpha = np.sqrt(1 - (w / lm) ** 2)
                elif lmt is not None and lt is not None:
                    cosalpha = 1 / (np.sqrt(1 + (w / (lmt - lt)) ** 2))
                alpha = np.arccos(cosalpha)

            if alpha > 1.4706289:  # np.arccos(0.1), 84.2608 degrees
                alpha = 1.4706289

            return alpha

        def excitation(self, t, u_max=None, u_min=None, t0=0, t1=5):
            """Excitation signal, a square wave.

            Parameters
            t : float
                time instant [s]
            u_max : float (0 < u_max <= 1), optional (default from parameter file)
                maximum value for muscle excitation
            u_min : float (0 < u_min < 1), optional (default from parameter file)
                minimum value for muscle excitation
            t0 : float, optional (default=0)
                initial time instant for muscle excitation equals to u_max [s]
            t1 : float, optional (default=5)
                final time instant for muscle excitation equals to u_max [s]
            Returns
            u : float (0 < u <= 1)
                excitation signal
            """

            if u_max is None:
                u_max = self.P["u_max"]
            if u_min is None:
                u_min = self.P["u_min"]

            u = u_min
            if t >= t0 and t <= t1:
                u = u_max

            return u

        def activation_dyn(self, t, a, t_act=None, t_deact=None):
            """Thelen (2003) activation dynamics, the derivative of `a` at `t`.

            Parameters
            t : float
                time instant [s]
            a : float (0 <= a <= 1)
                muscle activation
            t_act : float, optional (default from parameter file)
                activation time constant [s]
            t_deact : float, optional (default from parameter file)
                deactivation time constant [s]
            Returns
            adot : float
                derivative of `a` at `t`
            """

            if t_act is None:
                t_act = self.P["t_act"]
            if t_deact is None:
                t_deact = self.P["t_deact"]

            u = self.excitation(t)
            if u > a:
                adot = (u - a) / (t_act * (0.5 + 1.5 * a))
            else:
                adot = (u - a) / (t_deact / (0.5 + 1.5 * a))

            return adot

        def activation_sol(
            self,
            fun=None,
            t0=0,
            t1=3,
            dt=0.001,
            a0=0,
            u_min=None,
            t_act=None,
            t_deact=None,
            show=True,
            axs=None,
        ):
            """Forward Euler ODE solver for activation dynamics.

            Parameters
            ----------
            fun : function object, optional (default is None and `activation_dyn` is used)
                function with ODE to be solved
            t0 : float, optional (default=0)
                initial time instant for the simulation [s]
            t1 : float, optional (default=3)
                final time instant for the simulation [s]
            dt : float, optional (default=0.001)
                fixed integration time step [s]
            a0 : float, optional (default=0)
                initial muscle activation
            u_min : float (0 < u_min < 1), optional (default from parameter file)
                minimum value for muscle excitation
            t_act : float, optional (default from parameter file)
                activation time constant [s]
            t_deact : float, optional (default from parameter file)
                deactivation time constant [s]
            show : bool, optional (default = True)
                if True (1), plot data in matplotlib figure
            axs : a matplotlib.axes.Axes instance, optional (default = None)

            Returns
            -------
            data : 2-d array
                array with columns [time, excitation, activation]

            """

            if u_min is None:
                u_min = self.P["u_min"]
            if t_act is None:
                t_act = self.P["t_act"]
            if t_deact is None:
                t_deact = self.P["t_deact"]

            if fun is None:
                fun = self.activation_dyn

            nsteps = int(round((t1 - t0) / dt))
            a = a0
            data = []
            for i in range(nsteps):
                t = t0 + i * dt
                data.append([t, self.excitation(t), max(a, u_min)])
                adot = fun(t, a, t_act, t_deact)
                a = a + dt * adot

            data = np.array(data)
            if show:
                self.activation_plot(data, axs)

            self.act_data = data

            return data

        def activation(self, t=None):
            """Activation signal."""

            data = self.act_data
            if t is not None and len(data):
                if t <= self.act_data[0, 0]:
                    a = self.act_data[0, 2]
                elif t >= self.act_data[-1, 0]:
                    a = self.act_data[-1, 2]
                else:
                    a = np.interp(t, self.act_data[:, 0], self.act_data[:, 2])
            else:
                a = 1

            return a

        def activation_plot(self, data, axs=None):
            """Plot the activation dynamics results."""

            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib is not available.")
                return

            if axs is None:
                _, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

            axs.plot(data[:, 0], data[:, 1], color=[1, 0, 0, 0.6], label="Excitation")
            axs.plot(data[:, 0], data[:, 2], color=[0, 0, 1, 0.6], label="Activation")
            axs.set_xlabel("Time [s]")
            axs.set_ylabel("Level")
            axs.legend()
            plt.title("Activation dynamics")
            plt.tight_layout()
            plt.show()

            return axs

        def margins(self, x, margin=0.01, minmargin=True):
            """Calculate plot limits with extra margins."""
            rang = np.nanmax(x) - np.nanmin(x)
            if rang < 0.001 and minmargin:
                rang = 0.001 * np.nanmean(x) / margin
                if rang < 1:
                    rang = 1
            lim = [np.nanmin(x) - rang * margin, np.nanmax(x) + rang * margin]

            return lim
    return (Thelen2003,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
