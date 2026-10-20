import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kistler force plate calculation

    Marcos Duarte
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to get proper signals of a Kistler force plate, we have to calibrate the acquired data according to the factory calibration (i.e., transform the readed values in volts of the 8 channels to values in newtons for fx12, fx34, fy14, fy23, fz1, fz2, fz3, fz4) and then calculate the resultant forces (Fx, Fy, Fz), resultant moments of force (Mx, My, Mz), and center of pressure (COPx, COPy) quantities (see figure below for the convention used).

    <div class='center-align'><figure><img src="https://github.com/BMClab/BMC/blob/master/images/KistlerForcePlate.png?raw=1"alt="Kistler force plate and its coordinate system convention"/><figcaption><center><i>Figure. Kistler force plate and its coordinate system convention (from [Kistler Force Plate Formulae](http://isbweb.org/software/movanal/vaughan/kistler.pdf)).</i></center></figcaption></figure></div>

    The function `kistler_fp_cal.py` performs such calculations for the Kistler force plates of the BMClab, but it will work with other Kistler force plates. The function signature is:<br>
    ```python
    grf, cop = kistler_fp_cal(data, S_matrix=None, fxfy_range=0, fz_range=0, origin=None,
                              fp_type=3, show=False, axs=None):
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example
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
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    import sys
    sys.path.insert(0, str(repo_file('functions/kistler_fp_cal.py').parent))  # add to pythonpath
    return (np,)


@app.cell
def _(np):
    # load data file
    data = np.loadtxt(repo_file('data/kistler_fp_data.txt'), delimiter='\t', skiprows=1)
    data.shape
    return (data,)


@app.cell
def _():
    from kistler_fp_cal import kistler_fp_cal

    return (kistler_fp_cal,)


@app.cell
def _(data, kistler_fp_cal):
    grf, cop = kistler_fp_cal(data[:, 1:], fxfy_range=0, fz_range=1, show=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Kistler Force Plate Formulae](http://isbweb.org/software/movanal/vaughan/kistler.pdf)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function kistler_fp_cal.py
    """)
    return


@app.cell
def _(mo):
    mo.ui.code_editor(repo_file("functions/kistler_fp_cal.py").read_text(), language="python", disabled=True)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
