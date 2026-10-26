# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Principle of least action

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    This short guide introduces action as a number assigned to a whole candidate path. Keep both endpoint positions and endpoint times fixed while comparing paths. We will connect stationarity to an equation of motion and distinguish it from a guaranteed minimum.

    Try the distributed **Exercises** before opening their collapsed worked solutions. Then use the **Checkpoint questions** and attempt the **Problems** independently; their panels contain final answers only. Predict the effect of a parameter change before running it, and check units and limiting cases.

    **Opening challenge.** A free particle travels a fixed distance in a fixed time. Would alternating fast and slow motion give a smaller action than moving at constant velocity?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The dependency declaration at the top of this file lists the packages used by this guide. Marimo runs cells in dependency order. Run the notebook in an editor to use its displays and controls; the computational examples below explain their inputs before using them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The [principle of least action](https://en.wikipedia.org/wiki/Principle_of_least_action) applied to the movement of a mechanical system states that "the average kinetic energy less the average potential energy is as little as possible for the path of an object going from one point to another" (Prof. Richard Feynman in [The Feynman Lectures on Physics](https://www.feynmanlectures.caltech.edu/II_19.html)).

    This principle is so fundamental that it can be used to derive the equations of motion of a system, for example, independent of the Newton's laws of motion. Let's now see the principle of least action in mathematical terms.

    The difference between the kinetic and potential energy in a system is known as the [Lagrange or Lagrangian function](https://en.wikipedia.org/wiki/Lagrangian_mechanics):

    $$
    \mathcal{L} = T - V
    $$

    The [principle of least action](https://en.wikipedia.org/wiki/Principle_of_least_action) states that the actual path which a system follows between two points in the time interval $t_1$ and $t_2$ is such that the integral

    $$
    \mathcal{S}\; =\; \int _{t_1}^{t_2} \mathcal{L} \; dt
    $$

    is stationary, meaning that $\delta \mathcal{S}=0$ for variations with fixed endpoint positions and times. Stationarity is a first-variation condition: depending on the system and interval, the path may be a minimum or a saddle. It is not necessarily a global minimum. The integral above is known as the action integral and $\mathcal{S}$ is known as the action.

    A formal demonstration that the the value of this integral above is stationary and that one can derive the Euler-Lagrange equation in the present form we use today was first formulated by the mathematician [William Hamilton](https://en.wikipedia.org/wiki/William_Rowan_Hamilton) in the XIXth century; because that the principle of least action as presented above is also known as [Hamilton's principle](https://en.wikipedia.org/wiki/Hamilton%27s_principle).

    For a simple and didactic example where the integral above is stationary, see [The Feynman Lectures on Physics](https://www.feynmanlectures.caltech.edu/II_19.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — Compare two paths.** A 1 kg free particle moves 2 m in 2 s. Compare constant speed with 2 m/s for the first second and zero for the second.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 1

    Constant speed is 1 m/s and has action $S=\int Tdt=(1/2)(1)(1^2)(2)=1$ J s. The uneven path has $S=(1/2)(1)(2^2)(1)=2$ J s. Both reach the same endpoint at the same time; the uneven speed increases this free-particle action.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — Vary a whole trajectory.** Let $q_\epsilon(t)=q(t)+\epsilon\eta(t)$ with $\eta(t_1)=\eta(t_2)=0$. What does stationarity require of $dS[q_\epsilon]/d\epsilon$ at zero?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 2

    It must vanish for every admissible $\eta$. Integrating the velocity-variation term by parts gives $\delta S=\int[\partial L/\partial q-d(\partial L/\partial\dot q)/dt]\eta\,dt$. The endpoint term vanishes, yielding the Euler–Lagrange equation.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — Test the word least.** For a harmonic oscillator, examine the change in action along $q(t)=\epsilon\sin(\pi t/T)$ around the stationary path $q=0$, with endpoints zero. What is its sign?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 3

    For $L=m\dot q^2/2-kq^2/2$, direct integration gives $\Delta S=\epsilon^2[m\pi^2/T-kT]/4$. It is negative when $T>\pi\sqrt{m/k}$. Other higher-frequency variations can give positive changes, so the stationary path is then a saddle, not a minimum.
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
    ## Checkpoint questions

    1. What are the units of action?
    2. What is held fixed in Hamilton’s endpoint principle?
    3. Does stationary mean the particle is at rest?
    4. Why does the traditional title “least action” require care?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Checkpoint answers

    1. J s.
    2. Endpoint configurations and endpoint times.
    3. No; the first variation of action is zero.
    4. A stationary path need not minimize action over all admissible paths.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. A 2 kg free particle travels 3 m in 1.5 s at constant velocity. Find its action.

    2. For $L=m\dot x^2/2-mgx$, give the equation of motion.

    3. For the oscillator in Exercise 3, give the interval length at which the displayed variation changes sign.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers to Problems

    1. 6 J s.

    2. $\ddot x=-g$.

    3. $T=\pi\sqrt{m/k}$; this variation has zero quadratic action change at that value.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Use [KineticsFundamentalConcepts](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinetics_fundamental_concepts.py) to connect this guide with the surrounding mechanics and modeling material. Return to the assumptions behind each equation when applying it to a new experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Video lectures on the Internet

    [MIT OpenCourseWare](https://ocw.mit.edu/courses/2-003sc-engineering-dynamics-fall-2011/pages/lagrange-equations/) — Lagrange equations: follow the energy-to-equation method in the lecture and worked examples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Feynman R, Leighton R, Sands M (2013) [The Feynman Lectures on Physics - HTML edition](https://www.feynmanlectures.caltech.edu/).

    - Taylor JR (2005) [Classical Mechanics](https://books.google.com.br/books?id=P1kCtNr-pJsC). University Science Books.
    """)
    return


if __name__ == "__main__":
    app.run()
