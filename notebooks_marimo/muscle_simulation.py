import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Muscle simulation

    > Marcos Duarte, Renato Watanabe
    > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab)
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this tutorial

    This notebook is a guided study session. It puts the Hill-type muscle model from the companion notebook [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py) into motion: given an excitation signal and a muscle–tendon length, we integrate the muscle dynamics and watch the forces and lengths evolve in time.

    Read a short explanation, predict what the muscle should do, run the cell, and then use the plot to answer the challenge questions. Keep a scratchpad nearby and write your prediction *before* you reveal each plot.

    You do not need to finish every challenge on the first pass. Run the cells in order first; then come back and change one parameter at a time. The goal is not to make the code run — it already runs. The goal is to connect three things: the equations of the model, the step-by-step numerical integration, and the muscle behavior you see in the plots.

    **Challenge 0.** Before you begin, picture an *isometric* contraction: the whole muscle–tendon unit is held at a fixed length while the muscle activates. Predict what happens to the muscle fiber length $L_M$ and to the tendon length $L_T$ as activation rises. Can either one change even though their sum $L_{MT}$ is fixed? Write your answer down; you will test it in the first simulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1>Contents<span class="tocSkip"></span></h1>
    <div class="toc"><ul class="toc-item"><li><span><a href="#Pennation-angle" data-toc-modified-id="Pennation-angle-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Pennation angle</a></span></li><li><span><a href="#Muscle-force" data-toc-modified-id="Muscle-force-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Muscle force</a></span></li><li><span><a href="#Simulation" data-toc-modified-id="Simulation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Simulation</a></span></li><li><span><a href="#References" data-toc-modified-id="References-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Module-muscles.py" data-toc-modified-id="Module-muscles.py-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Module muscles.py</a></span></li></ul></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's simulate the 3-component Hill-type muscle model we described in [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py) and illustrated below:

    <img src="https://raw.githubusercontent.com/BMClab/BMC/master/images/muscle_hill.png" width="400" alt="Hill-type muscle model." />

    *Figure. A Hill-type muscle model with three components: two for the muscle, an active contractile element, $\mathsf{CE}$, and a passive elastic element in parallel, $\mathsf{PE}$, with the $\mathsf{CE}$, and one component for the tendon, an elastic element in series, $\mathsf{SE}$, with the muscle. $\mathsf{L_{MT}}$: muscle–tendon length, $\mathsf{L_T}$: tendon length, $\mathsf{L_M}$: muscle fiber length, $\mathsf{F_T}$: tendon force, $\mathsf{F_M}$: muscle force, and $\alpha$: pennation angle.*

    The following relationships are true for the model:

    $\begin{array}{l}
    L_{MT} = L_{T} + L_M\cos\alpha  \\
    \\
    L_M = L_{CE} = L_{PE} \\
    \\
    \dot{L}_M = \dot{L}_{CE} = \dot{L}_{PE} \\
    \\
    F_{M} = F_{CE} + F_{PE}
    \end{array}$

    If we assume that the muscle–tendon system is at equilibrium, that is, muscle,$F_{M}$, and tendon,$F_{T}$, forces are in equilibrium at all times, the following equation holds (and that a muscle can only pull):$F_{T} = F_{SE} = F_{M}\cos\alpha$

    **Guiding questions 1.**

    1. In the relationship $L_{MT} = L_T + L_M\cos\alpha$, which lengths change over the course of a simulation, and which one do *we* prescribe as an input?
    2. The equilibrium assumption sets $F_T = F_M\cos\alpha$ at every instant. What property of the tendon makes treating it as always in equilibrium with the muscle a reasonable simplification?
    3. If the tendon stretches while $L_{MT}$ is held fixed, what must happen to the fiber length $L_M$?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pennation angle

    The pennation angle will vary during muscle activation; for instance, Kawakami et al. (1998) showed that the pennation angle of the medial gastrocnemius muscle can vary from 22$^o$to 67$^o$during activation. The most used approach is to assume that the muscle width (defined as the length of the perpendicular line between the lines of the muscle origin and insertion) remains constant (Scott & Winter, 1991):

    $w = L_{M,0} \sin\alpha_0$

    The pennation angle as a function of time will be given by:

    $\alpha = \sin^{-1} \left(\dfrac{w}{L_M}\right)$

    The cosine of the pennation angle can be given by (if$L_M$is known):

    $\cos \alpha = \dfrac{\sqrt{L_M^2-w^2}}{L_M} = \sqrt{1-\left(\dfrac{w}{L_M}\right)^2}$

    or (if $L_M$ is not known):

    $\cos \alpha = \dfrac{L_{MT}-L_T}{L_M} = \dfrac{1}{\sqrt{1 + \left(\dfrac{w}{L_{MT}-L_T}\right)^2}}$

    **Guiding questions 2.**

    1. The tendon feels the force $F_T = F_M\cos\alpha$. As the pennation angle $\alpha$ grows during a contraction, does the tendon receive a larger or smaller fraction of the muscle fiber force?
    2. The constant-width assumption ($w = L_{M,0}\sin\alpha_0$) forces $\alpha$ to change as the fiber shortens. Using $\alpha = \sin^{-1}(w/L_M)$, explain whether the angle grows or shrinks when $L_M$ decreases.

    **Challenge 1.** A fusiform muscle has $\alpha_0 = 0$. Show from $w = L_{M,0}\sin\alpha_0$ that its width $w$ is zero, so $\alpha = 0$ and $\cos\alpha = 1$ for any fiber length. We will use $\alpha_0 = 0$ for most simulations and revisit a pennated ($30^o$) case later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Muscle force

    In general, the dependence of the force of the contractile element with its length and velocity and with the activation level are assumed independent of each other:

    $F_{CE}(a, L_{CE}, \dot{L}_{CE}) = a \: f_l(L_{CE}) \: f_v(\dot{L}_{CE}) \: F_{M0}$

    where $f_l(L_M)$ and $f_v(\dot{L}_M)$ are mathematical functions describing the force-length and force-velocity relationships of the contractile element (typically these functions are normalized by $F_{M0}$, the maximum isometric (at zero velocity) muscle force, so we have to multiply the right side of the equation by$F_{M0}$).

    And for the muscle force:

    $F_{M}(a, L_M, \dot{L}_M) = \left[a \: f_l(L_M)f_v(\dot{L}_M) + F_{PE}(L_M)\right]F_{M0}$

    This equation for the muscle force, with $a$, $L_{M}$, and $\dot{L}_{M}$ as state variables, can be used to simulate the dynamics of a muscle given an excitation and determine the muscle force and length. We can rearrange the equation, invert the expression for$f_v$, and integrate the resulting first-order ordinary differential equation (ODE) to obatin

    $L_M$:$\dot{L}_M = f_v^{-1}\left(\dfrac{F_{SE}(L_{MT}-L_M\cos\alpha)/\cos\alpha - F_{PE}(L_M)}{a f_l(L_M)}\right)$

    This approach is the most commonly employed in the literature (see for example, [OpenSim](http://simtk-confluence.stanford.edu:8080/display/OpenSim/Muscle+Model+Theory+and+Publications); McLean, Su, van den Bogert, 2003; Thelen, 2003; Nigg and Herzog, 2007).

    Although the equation for the muscle force doesn't have numerical singularities, the differential equation for muscle velocity has four ([OpenSim Millard 2012 Muscle Models](http://simtk-confluence.stanford.edu:8080/display/OpenSim/Millard+2012+Muscle+Models)):
    When $a \rightarrow 0$; when $f_l(L_M) \rightarrow 0$; when $\alpha \rightarrow \pi/2$; and when $\partial f_v/\partial v \rightarrow 0$.
    The following solutions can be employed to avoid the numerical singularities ([OpenSim Millard 2012 Muscle Models](http://simtk-confluence.stanford.edu:8080/display/OpenSim/Millard+2012+Muscle+Models)):
     - A minimum value for$a$; e.g.,$a_{min}=0.01$;
     - A minimum value for$f_l(L_M)$; e.g.,$f_l(0.1)$;
     - A maximum value for pennation angle; e.g., constrain$\alpha$to$\cos\alpha > 0.1; (\alpha < 84.26^o)$;
     - Make the slope of$f_V$at and beyond maximum velocity different than zero (for both concentric and eccentric activations).

    We will adopt these solutions to avoid singularities in the simulation of muscle mechanics. A problem of imposing values to variables as described above is that we can make the ordinary differential equation numerically stiff, which will increase the computational cost of the numerical integration. A better solution would be to modify the model to not have these singularities (see [OpenSim Millard 2012 Muscle Models](http://simtk-confluence.stanford.edu:8080/display/OpenSim/Millard+2012+Muscle+Models)).

    **Guiding questions 3.**

    1. The muscle force $F_{M} = \left[a \: f_l(L_M)f_v(\dot{L}_M) + F_{PE}(L_M)\right]F_{M0}$ has three factors that scale the *active* force. Which factor carries the neural command, which the length dependence, and which the velocity dependence?
    2. The velocity equation is obtained by inverting $f_v$, which divides by $a\,f_l(L_M)$. Why does this create a numerical singularity when the muscle is nearly inactive ($a\to 0$) or far from its optimal length ($f_l\to 0$)?

    **Challenge 2.** For each of the four singularities listed above ($a\to 0$, $f_l\to 0$, $\alpha\to\pi/2$, $\partial f_v/\partial v\to 0$), find the matching clamp in the code when you reach the module at the end of the notebook. (Hint: look for `0.1 * lmopt` in `vm_eq` and the pennation-angle cap in `penn_ang`.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation

    Let's simulate muscle dynamics using the Thelen2003Muscle model we defined in [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py). For the simulation of the Thelen2003Muscle, we simply have to integrate the equation:

    $V_M = (0.25+0.75a)\,V_{Mmax}\frac{\bar{F}_M-a\bar{f}_{l,CE}}{b}$

    where

    $b = \left\{
      \begin{array}{l l l}
        a\bar{f}_{l,CE} + \bar{F}_M/A_f \quad & \text{if} \quad \bar{F}_M \leq a\bar{f}_{l,CE} & \text{(shortening)} \\
        \\
        \dfrac{(2+2/A_f)(a\bar{f}_{l,CE}\bar{f}_{CEmax} - \bar{F}_M)}{\bar{f}_{CEmax}-1} \quad & \text{if} \quad \bar{F}_M > a\bar{f}_{l,CE} & \text{(lengthening)}
    \end{array} \right.$

    The equation above already contains the terms for actvation,$a$, and force-length dependence, $\bar{f}_{l,CE}$. The equation is too complicated for solving analytically, we will solve it by numerical integration with a simple fixed-step forward [Euler method](https://en.wikipedia.org/wiki/Euler_method). We could run a simulation using [OpenSim](https://simtk.org/home/opensim); it would be faster, but for fun, let's program in Python. All the necessary functions for the Thelen2003Muscle model described in [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py) were grouped in one file (module), `muscles.py`. Besides these functions, the module `muscles.py` contains a function for the muscle velocity, `vm_eq`, which is evaluated at each integration step by the solver `lm_sol`:

    ```python
    def vm_eq(self, t, lm, lm0, lmt0, lmopt, ltslack, alpha0, vmmax, fm0):
        "\"\"Equation for muscle velocity."\"\"
        if lm < 0.1*lmopt:
            lm = 0.1*lmopt
        a     = self.activation(t)
        lmt   = self.lmt_eq(t, lmt0)
        alpha = self.penn_ang(lmt=lmt, lm=lm, lm0=lm0, alpha0=alpha0)
        lt    = lmt - lm*np.cos(alpha)
        fse   = self.force_se(lt=lt, ltslack=ltslack)
        fpe   = self.force_pe(lm=lm/lmopt)
        fl    = self.force_l(lm=lm/lmopt)
        fce_t = fse/np.cos(alpha) - fpe
        vm    = self.velo_fm(fm=fce_t, a=a, fl=fl)
        return vm

    def lm_sol(self, fun, t0, t1, dt, lm0, lmt0, ltslack, lmopt, alpha0, vmmax, fm0, show, axs):
        "\"\"Forward Euler ODE solver for muscle length."\"\"
        if fun is None:
            fun = self.vm_eq
        nsteps = int(round((t1 - t0) / dt))
        lm = lm0
        data = []
        for i in range(nsteps):
            t = t0 + i*dt
            lm = max(lm, 0.1*lmopt)
            data.append(self.calc_data(t, lm, lm0, lmt0, ltslack, lmopt, alpha0, fm0))
            vm = fun(t, lm, lm0, lmt0, lmopt, ltslack, alpha0, vmmax, fm0)
            lm = lm + dt*vm
        data = np.array(data)
        self.lm_data = data
        if show:
            self.lm_plot(data, axs)
        return data
    ```

    **Challenge 3.** Read the `lm_sol` loop above. It is the same forward Euler update you met for a first-order ODE: `lm = lm + dt*vm`. Identify the three ingredients of one Euler step here — the current state, the rate of change, and the time step. Why does the solver take many small steps of size `dt` instead of one big jump?

    `muscles.py` also contains some auxiliary functions for entering data and for plotting the results. Let's import the necessary Python libraries and customize the environment in order to run some simulations using `muscles.py`:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["lines.linewidth"] = 2
    matplotlib.rcParams["font.size"] = 12
    matplotlib.rcParams["lines.markersize"] = 4
    matplotlib.rc("axes", grid=True, labelsize=12, titlesize=13, ymargin=0.01)
    matplotlib.rc("legend", numpoints=1, fontsize=10)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `muscles.py` module contains the class `Thelen2003()` which has the functions we want to use. To run a simulation, this class needs a set of muscle parameters and initial states.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of reading them from external files, we define the parameters and states directly in the notebook as Python dictionaries, so the notebook is self-contained and runs anywhere, including in the browser at [marimo.app](https://marimo.app/). Let's use the parameters and states from an exercise of the chapter 4 of Nigg and Herzog (2006):
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
        "lmt0": 0.313,  # initial muscle-tendon length
        "lm0": 0.09,  # initial muscle length
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
    We can see the parameters and states:
    """)
    return


@app.cell
def _(ms):
    print("Parameters:\n", ms.P)
    print("States:\n", ms.S)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot the muscle-tendon forces considering these parameters and initial states. **Before you run the next cell**, predict the shape of each curve: where along the fiber length is the active force–length curve maximal, and does the passive (PE) force appear before or after the optimal length? Where does the tendon (SE) force become non-zero?
    """)
    return


@app.cell
def _(ms):
    ms.muscle_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's simulate an isometric activation. Since we have not run the activation dynamics yet, the solver uses $a=1$ (full activation) throughout.

    **Before you run the next cell**, return to your answer for **Challenge 0**. The muscle–tendon length is held fixed, but the muscle is fully active. Predict whether the fiber length $L_M$ stays constant or changes, and what the tendon does. Then read the $L_M$ and $L_T$ panels of the plot to check.
    """)
    return


@app.cell
def _(ms):
    def lmt_eq(t, lmt0):
        # isometric activation

        lmt = lmt0

        return lmt

    ms.lmt_eq = lmt_eq
    return


@app.cell
def _(ms):
    data = ms.lm_sol()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Challenge 4.** In the isometric run above, the muscle–tendon length never changed, yet the fiber shortened a little while the tendon stretched. Explain this using tendon elasticity and the equilibrium condition $F_T = F_M\cos\alpha$. (This is the effect you predicted in Challenge 0.)

    Now let's prescribe a *changing* muscle–tendon length. The next cell redefines the length function `ms.lmt_eq`: it holds the length constant, ramps it down by 4 cm between $t=1$ and $t=2$ s (a shortening), then holds it again. **Before you run it**, predict what the muscle fiber velocity $V_{CE}$ will do during the ramp.
    """)
    return


@app.cell
def _(ms):
    def lmt_eq_1(t, lmt0):
        if t < 1:
            lmt = lmt0
        if 1 <= t < 2:
            lmt = lmt0 - 0.04 * (t - 1)
        if t >= 2:
            lmt = lmt0 - 0.04
        return lmt

    ms.lmt_eq = lmt_eq_1
    return


@app.cell
def _(ms):
    data_1 = ms.lm_sol()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's simulate a pennated muscle with an angle of$30^o$. We don't need to enter all parameters again, we can change only the parameter `alpha0`.

    **Before you run**, recall Guiding questions 2: with $\alpha_0 = 30^o$, only $\cos(30^o)\approx 0.87$ of the fiber force reaches the tendon. Predict whether the tendon force will be higher or lower than in the fusiform ($\alpha_0=0$) case.
    """)
    return


@app.cell
def _(ms, np):
    ms.P["alpha0"] = 30 * np.pi / 180
    print("New initial pennation angle:", ms.P["alpha0"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because the muscle length is now shortened by $\cos(30^o)$, we will also have to change the initial muscle-tendon length if we want to start with the tendon at its slack length:
    """)
    return


@app.cell
def _(ms, np):
    ms.S["lmt0"] = ms.S["lmt0"] - ms.S["lm0"] + ms.S["lm0"] * np.cos(ms.P["alpha0"])
    print("New initial muscle-tendon length:", ms.S["lmt0"])
    return


@app.cell
def _(ms):
    data_2 = ms.lm_sol()
    return (data_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is a plot of the simulated pennation angle:

    **Challenge 5.** The pennation angle is *not* constant during the contraction. From the constant-width assumption, does the angle grow or shrink as the fiber shortens? Confirm the direction of change against the plot.
    """)
    return


@app.cell
def _(data_2, np, plt):
    plt.figure(figsize=(7, 4))
    plt.plot(data_2[:, 0], data_2[:, 9] * 180 / np.pi)
    plt.xlabel("Time (s)")
    plt.ylabel("Pennation angle$(^o)$")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Change back to the old values:
    """)
    return


@app.cell
def _(ms):
    ms.P["alpha0"] = 0
    ms.S["lmt0"] = 0.313
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can change the initial states to show the role of the passive parallel element. The next cell sets a *longer* initial fiber length ($L_M = 0.10$ m, above the optimal $0.093$ m) so the passive element is engaged.

    **Before you run** the plot, predict: at a fiber length beyond the optimum, does the total muscle force come only from the active element, only from the passive (PE) element, or from both?
    """)
    return


@app.cell
def _(ms, np):
    ms.S = {"id": "", "lt0": np.nan, "lmt0": 0.323, "lm0": 0.10, "name": ""}
    return


@app.cell
def _(ms):
    ms.muscle_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far the activation was fixed at $a=1$. Now let's drive the muscle with a time-varying *excitation* and let the activation dynamics respond. The next cell defines a "hat" excitation: $u=1$ between $t=1$ and $t=2$ s, and $u=0.01$ otherwise.

    **Before you run** the activation plot, predict: will the activation $a(t)$ follow the excitation instantly, or will it lag on the way up and on the way down? Which should be slower — activation or deactivation — given $t_{act}=0.015$ s and $t_{deact}=0.05$ s?
    """)
    return


@app.cell
def _(ms):
    def excitation(t, u_max=1, u_min=0.01, t0=1, t1=2):
        """Excitation signal, a hat signal."""
        u = u_min
        if t >= t0 and t <= t1:
            u = u_max
        return u

    ms.excitation = excitation
    return


@app.cell
def _(ms):
    act = ms.activation_sol()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's simulate an isometric contraction:
    """)
    return


@app.cell
def _(ms):
    def lmt_eq_2(t, lmt0):
        lmt = lmt0
        return lmt

    ms.lmt_eq = lmt_eq_2
    return


@app.cell
def _(ms):
    data_3 = ms.lm_sol()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's use as excitation a train of short pulses (a crude model of repetitive motor-unit firing).

    **Before you run** the activation and force plots, predict what happens between pulses: does activation drop all the way back to rest, or does it stay partly elevated so that successive pulses *summate* into a smoother, sustained force?

    **Challenge 6.** Compare the muscle force from this pulse train with the single "hat" excitation before it. If you wanted a smoother force, would you increase the pulse *rate* or the pulse *amplitude*? Change the `0.1` spacing in `np.arange(1, 2.0, 0.1)` and re-run to test your answer.
    """)
    return


@app.cell
def _(ms, np):
    def excitation_1(t, u_max=0.5, u_min=0.01, t0=0.2, t1=2):
        """Excitation signal, a train of square pulses."""
        u = u_min
        ts = np.arange(1, 2.0, 0.1)
        if t >= ts[0] and t <= ts[1]:
            u = u_max
        elif t >= ts[2] and t <= ts[3]:
            u = u_max
        elif t >= ts[4] and t <= ts[5]:
            u = u_max
        elif t >= ts[6] and t <= ts[7]:
            u = u_max
        elif t >= ts[8] and t <= ts[9]:
            u = u_max
        return u

    ms.excitation = excitation_1
    return


@app.cell
def _(ms):
    act_1 = ms.activation_sol()
    return


@app.cell
def _(ms):
    data_4 = ms.lm_sol()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here and answer these before moving on. They tie together the pieces you have simulated.

    1. In a fixed-length (isometric) muscle–tendon unit, why can the muscle fiber length still change after activation changes?
    2. Which two force–length curves add together to give the total muscle force, and which one dominates at long fiber lengths?
    3. Why does the tendon force equal $F_M\cos\alpha$ rather than $F_M$?
    4. Activation lags behind excitation. Which state variable stores this "memory", and which two parameters set how fast it rises and falls?
    5. The solver clamps the fiber length at $0.1\,L_{M,opt}$ and the pennation angle below $\sim 84^o$. What numerical problem is each clamp preventing?
    6. When would a coarse forward-Euler solution be good enough for teaching, and when would you switch to an adaptive solver such as `solve_ivp`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    Work through these in order if you are studying independently. For each one, predict the outcome first, then modify the code and compare.

    1. **Activation level.** Run the activation dynamics with a constant excitation of $u=1$, $u=0.5$, and $u=0.1$. How does the steady-state muscle force scale with activation? Is it linear?
    2. **Tendon compliance.** Change the tendon slack length `ltslack` and/or `epst0` to make the tendon stiffer, then softer. In the isometric contraction, how does tendon compliance change how much the muscle fiber shortens?
    3. **Optimal length.** Set the initial fiber length `lm0` below, at, and above the optimal length `lmopt`. Predict from the force–length curve where the active force is largest, then confirm with `muscle_plot`.
    4. **Pennation sweep.** Sweep `alpha0` over $0^o$, $15^o$, $30^o$, and $45^o$ (remember to adjust `lmt0` as done above). Report the tendon force at each and explain the trend using $F_T = F_M\cos\alpha$.
    5. **Step size.** Re-run a simulation with `ms.lm_sol(dt=0.01)` and `ms.lm_sol(dt=0.0001)`. Where do the two solutions differ, and where do they agree? What does this tell you about choosing a step size?
    6. **Force–velocity.** During the prescribed shortening ramp, read the muscle force while the fiber is shortening and compare it with the isometric force at the same length. Does shortening raise or lower the active force, and why?
    7. **Your own contraction.** Design an excitation signal for a movement you care about (a quick tap, a sustained hold, a ramp). Predict the force, then simulate it by replacing `ms.excitation`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Muscle modeling](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2Fmuscle_modeling.py) — the companion notebook that derives the force–length, force–velocity, and activation functions used here.
    - [Introduction to numerical solution of Ordinary Differential Equations](https://marimo.app/?src=https%3A%2F%2Fgithub.com%2FBMClab%2FBMC%2Fblob%2Fmaster%2Fnotebooks_marimo%2FOrdinaryDifferentialEquation.py) — the forward Euler method and alternatives such as `solve_ivp`.
    - [OpenSim Millard 2012 Muscle Models](http://simtk-confluence.stanford.edu:8080/display/OpenSim/Millard+2012+Muscle+Models) — how modern implementations remove the numerical singularities discussed above.
    - [Thelen (2003)](http://homepages.cae.wisc.edu/~thelen/pubs/jbme03.pdf) — the muscle model formulation simulated in this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Kawakami Y, Ichinose Y, Fukunaga T (1998) [Architectural and functional features of human triceps surae muscles during contraction](http://www.ncbi.nlm.nih.gov/pubmed/9688711). Journal of Applied Physiology, 85, 398–404.
    - McLean SG, Su A, van den Bogert AJ (2003) [Development and validation of a 3-D model to predict knee joint loading during dynamic movement](http://www.ncbi.nlm.nih.gov/pubmed/14986412). Journal of Biomechanical Engineering, 125, 864-74.
    - Nigg BM and Herzog W (2006) [Biomechanics of the Musculo-skeletal System](https://books.google.com.br/books?id=hOIeAQAAIAAJ&dq=editions:ISBN0470017678). 3rd Edition. Wiley.
    - Scott SH, Winter DA (1991) [A comparison of three muscle pennation assumptions and their effect on isometric and isotonic force](http://www.ncbi.nlm.nih.gov/pubmed/2037616). Journal of Biomechanics, 24, 163–167.
    - Thelen DG (2003) [Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older adults](http://homepages.cae.wisc.edu/~thelen/pubs/jbme03.pdf). Journal of Biomechanical Engineering, 125(1):70–77.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Module muscles.py
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
