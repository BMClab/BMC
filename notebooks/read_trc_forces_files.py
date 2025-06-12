import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Read Cortex Motion Analysis Corporation .trc and .forces files (example)

        > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab)  
        > Federal University of ABC, Brazil
        """
    )
    return


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
    import pyversions  # https://pypi.org/project/pyversions/
    sys.path.insert(1, r'./../functions')
    import io_cortexmac as io  # from https://github.com/BMClab/BMC/tree/master/functions
    return io, np, os, pd, plt, pyversions, signal


@app.cell
def _(pyversions):
    pyversions.versions();
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Use function `io_cortexmac.py` from BMClab's repo
        """
    )
    return


@app.cell
def _():
    path2 = r'/mnt/B/Dropbox/BMClab/stuff/Biomecanica/2020/'
    return (path2,)


@app.cell
def _(io, os, path2):
    _fname = os.path.join(path2, 'walk_diurno.trc')
    (_h_trc, trc) = io.read_trc(_fname, fname2='', units='m', dropna=False, na=0.0, df_multi=False)
    trc.set_index('Time', drop=True, inplace=True)
    trc.drop('Frame#', axis=1, inplace=True)
    return (trc,)


@app.cell
def _(io, os, path2):
    _fname = os.path.join(path2, 'walk_diurno.forces')
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
    mo.md(
        r"""
        ### trc and forces data have different sampling rates
        """
    )
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
    mo.md(
        r"""
        #### Resample trc to the force sampling rate (150 Hz to 450 Hz)
        """
    )
    return


@app.cell
def _(freq_grf, freq_trc, np, pd, signal, trc):
    _nrows = int(trc.shape[0] * np.round(freq_grf / freq_trc))
    _ncols = trc.shape[1]
    _data = np.nan * np.zeros((_nrows, 1 + _ncols), dtype='float64')
    _data[:, 0] = np.linspace(start=0, stop=_nrows / freq_grf, num=_nrows, endpoint=False)
    for _i in range(_ncols):
        _data[:, _i + 1] = signal.resample_poly(trc.iloc[:, _i], np.round(freq_grf), np.round(freq_trc), window='blackman')
    trc_1 = pd.DataFrame(data=_data[:, 1:], index=_data[:, 0], columns=trc.columns)
    trc_1.index.name = trc_1.index.name
    return (trc_1,)


@app.cell
def _(trc_1):
    trc_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Plot of some data
        """
    )
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
    trc_1.plot(y=['R.Heely', 'L.Heely'], ax=_axs[0], title='Data diurno')
    grf.plot(y=['FY5', 'FY6'], ax=_axs[1], colormap='viridis')
    _axs[0].set_ylabel('Position [m]')
    _axs[1].set_ylabel('Force [N]')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This means that the subject stepped on force plate **#6** with her/his **left** foot and then stepped on force plate **#5** with her/his **right** foot.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## For data noturno
        """
    )
    return


@app.cell
def _(io, np, os, path2, pd, plt, signal):
    _fname = os.path.join(path2, 'walk_noturno.trc')
    (_h_trc, trc_2) = io.read_trc(_fname, fname2='', units='m', dropna=False, na=0.0, df_multi=False)
    trc_2.set_index('Time', drop=True, inplace=True)
    trc_2.drop('Frame#', axis=1, inplace=True)
    _fname = os.path.join(path2, 'walk_noturno.forces')
    (_h_grf, grf_1) = io.read_forces(_fname, time=True, forcepla=[], mm2m=True, show_msg=True)
    freq_trc_1 = 1 / np.mean(np.diff(trc_2.index))
    freq_grf_1 = 1 / np.mean(np.diff(grf_1.index))
    _nrows = int(trc_2.shape[0] * np.round(freq_grf_1 / freq_trc_1))
    _ncols = trc_2.shape[1]
    _data = np.nan * np.zeros((_nrows, 1 + _ncols), dtype='float64')
    _data[:, 0] = np.linspace(start=0, stop=_nrows / freq_grf_1, num=_nrows, endpoint=False)
    for _i in range(_ncols):
        _data[:, _i + 1] = signal.resample_poly(trc_2.iloc[:, _i], np.round(freq_grf_1), np.round(freq_trc_1), window='blackman')
    trc_2 = pd.DataFrame(data=_data[:, 1:], index=_data[:, 0], columns=trc_2.columns)
    trc_2.index.name = trc_2.index.name
    (_fig, _axs) = plt.subplots(2, 1, sharex=True, squeeze=True, figsize=(10, 5))
    trc_2.plot(y=['R.Heely', 'L.Heely'], ax=_axs[0], title='Data noturno')
    grf_1.plot(y=['FY5', 'FY6'], ax=_axs[1], colormap='viridis')
    _axs[0].set_ylabel('Position [m]')
    _axs[1].set_ylabel('Force [N]')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This means that the subject stepped on force plate **#6** with her/his **right** foot and then stepped on force plate **#5** with her/his **left** foot.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
