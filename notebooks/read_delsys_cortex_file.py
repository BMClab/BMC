import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Read Delsys file from Cortex

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://demotu.org/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    # '%matplotlib notebook' command supported automatically in marimo
    # tk qt notebook inline ipympl
    import matplotlib
    import matplotlib.pyplot as plt

    import sys, os
    sys.path.insert(1, r'./../functions')

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
