# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14", "seaborn>=0.13", "urllib3>=2.2", "detecta>=0.0.5"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read Cortex Motion Analysis Corporation .trc and .forces files (example)

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Marker trajectories and force plates share a trial but have different sample rates. How can we compare them on one time axis?

    We will read two public walking trials, inspect sampling rates, and resample marker trajectories before plotting heel position and force.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The repository files `walk1.trc/.forces` and `walk2.trc/.forces` replace the old private day/night filenames.
    """)
    return




@app.function
def repo_file(path, *, base_url="https://raw.githubusercontent.com/BMClab/BMC/master/"):
    """Find a repository file beside this notebook, or fetch a temporary copy.

    Local files take precedence. Only file acquisition falls back to GitHub;
    parsing errors remain visible to the calling notebook. urllib3 selects its
    browser transport automatically without platform-specific imports.
    """
    import hashlib
    import tempfile
    from pathlib import Path, PurePosixPath
    from urllib.parse import quote

    import marimo as mo
    import urllib3

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise ValueError("Use a repository-relative path such as data/walk.trc")
    location = mo.notebook_dir()
    if location is not None:
        directory = Path(location).resolve()
        for base in (directory, *directory.parents):
            local = base.joinpath(*relative.parts)
            if local.is_file():
                return local
    url = base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")
    key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]
    cached = (Path(tempfile.gettempdir()) / "bmc-notebooks-v2" / key).joinpath(*relative.parts)
    if not cached.is_file():
        response = urllib3.request("GET", url, timeout=30.0, retries=2)
        if response.status != 200:
            raise OSError(f"Cannot load {path}: HTTP {response.status} from {url}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as download:
            temporary = Path(download.name)
            download.write(response.data)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    return cached


@app.cell
def _():
    import numpy as np
    import pandas as pd
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('notebook', font_scale=1.2, rc={"lines.linewidth": 2})
    from scipy import signal
    import sys, os
    _functions = [repo_file(f'functions/{_m}.py') for _m in ('io_cortexmac', 'critic_damp', 'linear_envelope')]
    sys.path.insert(0, str(_functions[0].parent))  # add to pythonpath
    import io_cortexmac as io  # from https://github.com/BMClab/BMC/tree/master/functions

    return io, np, os, pd, plt, signal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The repository reader keeps marker and force channels in separate tables. NumPy and SciPy handle the numerical resampling; pandas preserves labels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use function `io_cortexmac.py` from BMClab's repo
    """)
    return


@app.cell
def _():
    path2 = repo_file("data/walk1.trc").parent
    return (path2,)


@app.cell
def _(io, os, path2):
    _fname = str(repo_file("data/walk1.trc"))
    (_h_trc, trc) = io.read_trc(_fname, fname2='', units='m', dropna=False, na=0.0, df_multi=False)
    trc.set_index('Time', drop=True, inplace=True)
    trc.drop('Frame#', axis=1, inplace=True)
    return (trc,)


@app.cell
def _(io, os, path2):
    _fname = str(repo_file("data/walk1.forces"))
    (_h_grf, grf) = io.read_forces(_fname, time=True, forcepla=[], mm2m=True, show_msg=True)
    return (grf,)


@app.cell
def _(trc):
    trc
    return


@app.cell
def _(grf):
    grf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### trc and forces data have different sampling rates
    """)
    return


@app.cell
def _(np, trc):
    freq_trc = 1/np.mean(np.diff(trc.index))
    freq_trc
    return (freq_trc,)


@app.cell
def _(grf, np):
    freq_grf = 1/np.mean(np.diff(grf.index))
    freq_grf
    return (freq_grf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** If marker data are sampled at 150 Hz and force data at 450 Hz, what is the upsampling ratio?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 1': mo.md(r"""
    The ratio is $450/150=3$. Each marker sampling bin corresponds to three force sampling bins. This interpolates marker values; it adds no measured kinematic detail.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Resample trc to the force sampling rate (150 Hz to 450 Hz)

    Linear boundary extension avoids forcing the last marker coordinate toward zero.
    Boundary values remain estimates: inspect the ends, and retain the original timestamps
    when deciding which part of the resampled signal to analyse.
    """)
    return


@app.cell
def _(freq_grf, freq_trc, np, pd, signal, trc):
    _nrows = int(trc.shape[0] * np.round(freq_grf / freq_trc))
    _ncols = trc.shape[1]
    _data = np.nan * np.zeros((_nrows, 1 + _ncols), dtype='float64')
    _data[:, 0] = np.linspace(start=0, stop=_nrows / freq_grf, num=_nrows, endpoint=False)
    for _i in range(_ncols):
        _data[:, _i + 1] = signal.resample_poly(trc.iloc[:, _i], int(round(freq_grf)), int(round(freq_trc)), window='blackman', padtype='line')
    trc_1 = pd.DataFrame(data=_data[:, 1:], index=_data[:, 0], columns=trc.columns)
    trc_1.index.name = trc_1.index.name
    return (trc_1,)


@app.cell
def _(trc_1):
    trc_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot of some data
    """)
    return


@app.cell
def _(trc_1):
    trc_1.columns
    return


@app.cell
def _(grf):
    grf.columns
    return


@app.cell
def _(grf, plt, trc_1):
    (_fig, _axs) = plt.subplots(2, 1, sharex=True, squeeze=True, figsize=(10, 5))
    trc_1.plot(y=['R.Heely', 'L.Heely'], ax=_axs[0], title='Walking trial 1')
    grf.plot(y=['FY5', 'FY6'], ax=_axs[1], colormap='viridis')
    _axs[0].set_ylabel('Position [m]')
    _axs[1].set_ylabel('Force [N]')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the synchronized heel and force traces to assess foot contacts. A force-plate number alone does not identify left versus right foot. Confirm the assignment against the marker positions and acquisition layout for each trial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Does matching the two sampling rates prove that the devices were synchronized?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 2': mo.md(r"""
    No. Matching rates gives compatible spacing, but an offset or clock drift can remain. The paired files must share a known acquisition time origin; verify that assumption before assigning force to a foot.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare a second trial
    """)
    return


@app.cell
def _(io, np, os, path2, pd, plt, signal):
    _fname = str(repo_file("data/walk2.trc"))
    (_h_trc, trc_2) = io.read_trc(_fname, fname2='', units='m', dropna=False, na=0.0, df_multi=False)
    trc_2.set_index('Time', drop=True, inplace=True)
    trc_2.drop('Frame#', axis=1, inplace=True)
    _fname = str(repo_file("data/walk2.forces"))
    (_h_grf, grf_1) = io.read_forces(_fname, time=True, forcepla=[], mm2m=True, show_msg=True)
    freq_trc_1 = 1 / np.mean(np.diff(trc_2.index))
    freq_grf_1 = 1 / np.mean(np.diff(grf_1.index))
    _nrows = int(trc_2.shape[0] * np.round(freq_grf_1 / freq_trc_1))
    _ncols = trc_2.shape[1]
    _data = np.nan * np.zeros((_nrows, 1 + _ncols), dtype='float64')
    _data[:, 0] = np.linspace(start=0, stop=_nrows / freq_grf_1, num=_nrows, endpoint=False)
    for _i in range(_ncols):
        _data[:, _i + 1] = signal.resample_poly(trc_2.iloc[:, _i], int(round(freq_grf_1)), int(round(freq_trc_1)), window='blackman', padtype='line')
    trc_2 = pd.DataFrame(data=_data[:, 1:], index=_data[:, 0], columns=trc_2.columns)
    trc_2.index.name = trc_2.index.name
    (_fig, _axs) = plt.subplots(2, 1, sharex=True, squeeze=True, figsize=(10, 5))
    trc_2.plot(y=['R.Heely', 'L.Heely'], ax=_axs[0], title='Walking trial 2')
    grf_1.plot(y=['FY5', 'FY6'], ax=_axs[1], colormap='viridis')
    _axs[0].set_ylabel('Position [m]')
    _axs[1].set_ylabel('Force [N]')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This means that the subject stepped on force plate **#6** with her/his **right** foot and then stepped on force plate **#5** with her/his **left** foot.
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

    1. Does resampling create new marker observations?

    2. Why plot both signals against time rather than row number?

    3. Can force-plate labels alone identify the contacting foot?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Checkpoint answers': mo.md(r"""
    1. No.
    2. Original rows correspond to different sampling intervals.
    3. No.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    Try these independently; open a panel only to check your final answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 1.** There are 364 marker frames at 150 Hz. How many output rows result at 450 Hz?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 1': mo.md(r"""
    1,092.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** What time interval separates force samples at 450 Hz?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 2': mo.md(r"""
    $1/450$ s, approximately 2.22 ms.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Inspect alignment at several contacts, and compare original versus resampled marker values away from filter boundaries.

    ### Video lectures on the Internet

    [OpenSim video tutorials](https://www.youtube.com/@OpenSimVideos) provide gait-analysis context.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [SciPy polyphase resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).

    - [Motion Analysis](https://www.motionanalysis.com/).
    """)
    return


if __name__ == "__main__":
    app.run()
