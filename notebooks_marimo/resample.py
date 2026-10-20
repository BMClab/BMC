import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Resample data
    > Marcos Duarte<br>
    > [Laboratory of Biomechanics and Motor Control](http://demotu.org/)<br>
    > Federal University of ABC, Brazil
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
    # '%matplotlib notebook' command supported automatically in marimo
    # tk qt notebook inline ipympl
    import matplotlib
    import matplotlib.pyplot as plt

    import sys, os
    sys.path.insert(0, str(repo_file('functions/resample.py').parent))  # add to pythonpath

    from resample import resample

    return np, pd, resample


@app.cell
def _(np, pd):
    y = np.random.randn(100, 3)
    freq_new = 100
    y = pd.DataFrame(data=y, columns=None)
    y.index = y.index/freq_new
    y.index.name = y.index.name
    y.head()
    return (y,)


@app.cell
def _(resample, y):
    y2 = resample(y, freq_old=100, freq_new=1000, limit=1000, method='resample_poly')
    y2
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
