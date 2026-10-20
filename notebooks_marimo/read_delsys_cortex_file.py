import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read Delsys file from Cortex

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
    _functions = [repo_file(f'functions/{_m}.py') for _m in ('io_cortexmac', 'critic_damp', 'linear_envelope', 'tnorm')]
    sys.path.insert(0, str(_functions[0].parent))  # add to pythonpath

    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    from io_cortexmac import read_delsys

    return np, os, read_delsys


@app.cell
def _(os):
    freq_trc = 150 # Cortex sampling rate
    muscles = ['TA', 'Sol', 'VL', 'BF', 'GMax', 'GL', 'RF', 'GMed', 'VM']
    path2 = '/mnt/A/BMClab/Projects/FapespRunAge/Data/Cadence/s20/'
    fname = 'run100s.csv'
    fname = os.path.join(path2, fname)
    return fname, muscles


@app.cell
def _(fname, muscles, read_delsys):
    df_emg, df_imu = read_delsys(fname, fname2='=', sensors=muscles, freq_trc=150, emg=True, imu=True,
                                 resample=[150, 150], show_msg=True, show=True, suptitle=fname)
    return df_emg, df_imu


@app.cell
def _(df_emg):
    df_emg
    return


@app.cell
def _(df_imu):
    df_imu
    return


@app.cell
def _(df_emg, df_imu, np):
    print('Sampling rates for EMG and IMU data:')
    print(np.mean(1/np.diff(df_emg.index)), np.mean(1/np.diff(df_imu.index)))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
