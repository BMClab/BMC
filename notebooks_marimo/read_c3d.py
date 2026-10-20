import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read c3D file using EZC3D library

    > Marcos Duarte<br>
    > [Laboratory of Biomechanics and Motor Control](http://demotu.org/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > EZC3D is an easy to use reader, modifier and writer for C3D format files.<br>
    > https://github.com/pyomeca/ezc3d
    """)
    return


@app.function
def repo_file(path):
    """Local path of a file of the BMC repository, e.g. "data/walk.trc".

    Where only the notebook is available (e.g., on molab), the file is
    downloaded from GitHub to a temporary folder.
    """
    import tempfile
    import urllib.request
    from pathlib import Path

    import marimo as mo

    local = mo.notebook_dir() / ".." / path
    if local.exists():
        return local
    cached = Path(tempfile.gettempdir()) / "BMC" / path
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/BMClab/BMC/master/{path}", cached)
    return cached


@app.cell
def _():
    import numpy as np
    import pandas as pd
    #%matplotlib notebook
    # '%matplotlib widget' command supported automatically in marimo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import sys, os
    import ezc3d

    _functions = [repo_file(f'functions/{_m}.py') for _m in ('read_c3d', 'dfmlevel')]
    sys.path.insert(0, str(_functions[0].parent))  # add to pythonpath
    from dfmlevel import dfmlevel
    from read_c3d import read_c3d

    return os, read_c3d


@app.cell
def _(os):
    path2 = '/mnt/A/BMClab/Projects/FapespRunAge/Data/Cadence/s20/'
    fname = 'run100c.c3d'
    fname = os.path.join(path2, fname)
    return (fname,)


@app.cell
def _(fname, read_c3d):
    an, pt = read_c3d(fname, analog='all', point='all', short_label=True)
    return an, pt


@app.cell
def _(an):
    an
    return


@app.cell
def _(pt):
    pt
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
