# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Python for scientific computing

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A scientific notebook is an argument that the reader can run. It connects a question, a model, numerical inputs, and an interpretation. This short orientation explains where Python and Marimo fit; [Python tutorial](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_tutorial.py) supplies the basic language practice.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** A graph looks convincing, but its axes have no units and its input file is unavailable. What would you need before trusting or reproducing it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    This overview uses only Marimo and ordinary Python. For local setup use [Python installation](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_installation.py); hosted notebooks still need their declared packages and any required data.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment, theory, and computation

    An experiment observes movement; a model states what you think governs it; computation makes the model’s consequences explicit. None of the three replaces the others. A simulation can reproduce an equation exactly and still describe the wrong physical system.

    ![Theory, experiment, and computation](https://raw.githubusercontent.com/jrjohansson/scientific-python-lectures/master/images/theory-experiment-computation.png)

    The original illustration is from J. R. Johansson’s scientific Python lectures. Consider a walking experiment: distance and time are measurements, constant average speed is a summary model, and division computes its prediction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A notebook calculates 6 m / 4 s = 1.5 m/s. Which part is measurement, which part is computation, and what does the result leave out?

    /// details | Solution to Exercise 1
    Distance and time are measurements. Division is the computation. The result is an average speed; it does not describe accelerations or changes of speed within the trial. A correct calculation can be an incomplete answer to a movement question.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python and its scientific tools

    Python is a language; CPython is its most widely used implementation. Names refer to objects such as numbers, lists, and functions. Modules group reusable definitions, and `import` makes them available. You can learn these ideas gradually while running real examples.

    NumPy works with arrays, SciPy supplies numerical methods, Matplotlib draws figures, SymPy handles symbolic expressions, and pandas organizes tables. You do not need all of them for a first calculation. The later notebooks introduce the relevant library where a scientific task requires it.

    Python often coordinates efficient compiled numerical routines. That does not make every Python loop fast, and no single language is best for every problem. Choose tools that let you check both the calculation and its assumptions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** You want to integrate a mechanical model and draw its trajectory. Which two tools from the list fit those tasks? What do they not decide for you?

    /// details | Solution to Exercise 2
    SciPy provides numerical integration and Matplotlib draws the trajectory. They do not choose valid physical assumptions, initial conditions, or units. Those decisions remain part of the scientific explanation.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading a Marimo notebook

    This collection uses Marimo: prose cells explain the reasoning and code cells expose the calculation. Edit an input, run its cell, and dependent cells update. Keep one public definition of each variable, and place a control’s `.value` in a downstream cell. Use meaningful names and record units.

    Jupyter and IPython are related interactive tools you may encounter in older material. Their shell escapes and magic commands are not ordinary Python and are not required here. Keep terminal setup commands in a terminal; keep this lesson focused on computations you can inspect.

    A reproducible result needs the notebook, the input data, the package environment, and enough explanation to interpret them. Version control records source changes, while a suitable data archive preserves observations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** You changed an input but copied its old value as a literal into a later cell. Why can the downstream result stay unchanged?

    /// details | Solution to Exercise 3
    The later expression depends on the literal, not on the input variable. Reference the variable itself so the notebook can follow the dependency. A notebook is reproducible when the source of each result is explicit.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Is Python a synonym for NumPy?
    2. What does reactive execution track?
    3. Can a reproducible calculation still be scientifically wrong?

    /// details | Checkpoint answers
    1. No; NumPy is a library used from Python.
    2. Dependencies among named values and cells.
    3. Yes; assumptions, measurements, or interpretation can be wrong.
    ///

    ## Problems

    1. List four things needed to reproduce a notebook result.
    2. Name the tool in this collection that joins prose, code, and reactive outputs.
    3. A model ignores air resistance. Does changing the plot style improve that assumption?

    /// details | Final answers
    1. Source, data, environment specification, and explanatory context.
    2. Marimo.
    3. No.
    ///

    ## Go deeper

    [Python tutorial](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_tutorial.py) — begin with arithmetic and variables. [Version control](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/version_control_git_github.py) — keep a reviewable history. [Scientific Python lectures](https://lectures.scientific-python.org/) — optional later material.

    ### Video lectures on the Internet

    [MIT introductory Python lectures](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/video_galleries/lecture-videos/) — use the first lectures for expressions, variables, branching, and loops.

    ## References

    - [Python documentation](https://docs.python.org/3/); [Marimo key concepts](https://docs.marimo.io/getting_started/key_concepts/); [J. R. Johansson’s scientific Python lectures](https://github.com/jrjohansson/scientific-python-lectures).
    """)
    return


if __name__ == "__main__":
    app.run()
