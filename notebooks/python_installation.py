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
    # How to install Python for scientific computing

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    You should be able to reopen a scientific analysis months later and know which software produced it. This guide gets you from a terminal to a reproducible notebook environment, then distinguishes local and hosted execution. No previous Python knowledge is needed.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** Two computers both report “Python 3” but only one can import NumPy. What is missing from that description?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    This reading guide needs only Marimo. Commands shown in fenced blocks are instructions for your terminal; reading the notebook does not install software or change your system.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install an environment for this collection

    The first question is not “where is Python installed?” but “which Python runs this notebook, and which packages belong to it?” An environment keeps those answers together. Do not rely on a system Python being present on macOS, Linux, or Windows; installations differ.

    For this repository, install [uv using its official instructions](https://docs.astral.sh/uv/getting-started/installation/), obtain the BMC repository, and open a terminal in its root folder. The checked-in `pyproject.toml` and `uv.lock` specify the environment. Run:

    ```sh
    uv sync --locked
    uv run python --version
    uv run marimo edit notebooks/python_installation.py
    ```

    The project requires Python 3.14 or newer. `uv` can obtain a compatible interpreter; allow network access on the first setup. Later runs use the project environment in `.venv`. A shell prompt is not Python's `>>>` prompt: the commands above belong in the terminal.

    NumPy handles arrays; SciPy supplies numerical algorithms; Matplotlib draws figures; SymPy works symbolically; pandas organizes tables. Statsmodels and scikit-learn cover statistical modeling and machine learning. IPython, Jupyter, Spyder, and Qt-based interfaces are useful alternatives in the broader ecosystem, but you do not need to install every tool to read one guide. The notebook's inline dependency block describes its standalone needs.

    ### Historical terminal examples

    These original screenshots show older macOS, Ubuntu, and Windows 7 environments. Keep them as illustrations of a terminal and interpreter prompt, not as instructions for a current operating system.

    ![Historical macOS terminal](https://raw.githubusercontent.com/BMClab/BMC/master/images/terminal_mac_osx.png)
    ![Historical Ubuntu terminal](https://raw.githubusercontent.com/BMClab/BMC/master/images/terminal_linux_ubuntu.png)
    ![Historical Windows command prompt](https://raw.githubusercontent.com/BMClab/BMC/master/images/command_prompt.png)
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** You typed `uv sync --locked` at the Python `>>>` prompt and received a syntax error. What happened?

    /// details | Solution to Exercise 1
    The Python interpreter expects Python expressions, while `uv` is a terminal command. Exit the interpreter and run the command from your shell in the repository root. The error does not mean the project is broken.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python 3 and reproducibility

    Use Python 3. Python 2 reached end of life in 2020; compatibility shims and Python-2 installers are no longer part of this guide. A notebook that runs under one Python version can still depend on package versions, so save both the source and its environment specification.

    `uv run` selects the project environment when you run the commands from the repository root. To diagnose an unexpected interpreter, run `uv run python -c "import sys; print(sys.executable)"`. If an import fails, first check which environment is running; installing the package into an unrelated interpreter will not solve it.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** A package imports in one terminal but not in your notebook. What two facts would you compare before reinstalling anything?

    /// details | Solution to Exercise 2
    Compare `sys.executable` and the active environment/package set. A notebook server and a terminal can run different Python installations. Launching the editor through `uv run` from the project root connects it to the declared project environment.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other ways to obtain scientific Python

    [Anaconda Distribution and Miniconda](https://www.anaconda.com/docs/getting-started/installation) remain alternatives for managing Python environments. Follow the current instructions for your operating system and processor architecture instead of copying an old installer filename. Check the provider's terms for your use. Miniconda starts with a smaller environment; the distribution bundles more tools. These are alternatives to the repository's uv workflow, not additional requirements.

    For a notebook without a local installation, use [marimo.app](https://marimo.app/) or [molab](https://molab.marimo.io/notebooks). The first runs Python in the browser through WebAssembly; molab runs a hosted Python environment. Browser-supported packages and available files still matter. Uploading a notebook does not upload every file on your computer. Consult [Marimo's WebAssembly guide](https://docs.marimo.io/guides/wasm/) when a native dependency or file access behaves differently.
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** Will uploading only a `.py` notebook also upload a CSV stored next to it? Explain what dependency metadata does and does not cover.

    /// details | Solution to Exercise 3
    No. Inline metadata declares Python packages, not your private local data. Upload required files separately or use the notebook’s explicit public-data loader. A successful package installation cannot repair a missing dataset.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Launching and checking your tools

    Start the editor with `uv run marimo edit notebooks/python_installation.py`. Run the cells and inspect their outputs. A command that prints a version proves that a program can start; it does not prove that every notebook dependency and dataset is available. Open a computational guide next and run all its cells.

    The original material compared IPython notebooks, the Spyder IDE, and a hosted service called Wakari. The screenshots below preserve that history. Wakari is no longer an active recommendation; use the Marimo services linked above for this collection. Jupyter and Spyder remain separate tools with their own workflows.

    ![Historical Anaconda launcher](https://raw.githubusercontent.com/BMClab/BMC/master/images/anaconda_launcher_win7.png)
    ![Historical IPython notebook](https://raw.githubusercontent.com/BMClab/BMC/master/images/ipython_notebook_win7.png)
    ![Historical Spyder IDE](https://raw.githubusercontent.com/BMClab/BMC/master/images/spyder_win7.png)
    ![Historical Wakari service](https://raw.githubusercontent.com/BMClab/BMC/master/images/wakari.png)
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

    1. What does the lockfile add to a list of package names?
    2. Must your operating system provide Python?
    3. Are a terminal command and Python code interchangeable?

    /// details | Checkpoint answers
    1. A resolved dependency specification for reproducible project setup.
    2. No.
    3. No; run each in its appropriate interpreter.
    ///

    ## Problems

    1. State the command that opens this guide in the repository environment.
    2. Name the two project files to inspect when dependency requirements disagree.
    3. A browser notebook needs a native-only package. What environment is the practical next choice?

    /// details | Final answers
    1. `uv run marimo edit notebooks/python_installation.py`.
    2. `pyproject.toml` and `uv.lock`.
    3. A compatible local or hosted CPython environment, such as molab, with that package installed.
    ///

    ## Go deeper

    [uv project guide](https://docs.astral.sh/uv/guides/projects/) — environment and lockfile workflow. Continue with [Python tutorial](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/python_tutorial.py) to run real calculations.

    ### Video lectures on the Internet

    [MIT: Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/video_galleries/lecture-videos/) — start with computation, branching, and iteration; the videos explain how to reason about code before running it.

    ## References

    - [uv installation](https://docs.astral.sh/uv/getting-started/installation/); [Python 2 sunset](https://www.python.org/doc/sunset-python-2/); [Marimo WebAssembly](https://docs.marimo.io/guides/wasm/).
    """)
    return


if __name__ == "__main__":
    app.run()
