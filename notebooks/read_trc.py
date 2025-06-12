import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Read Cortex Motion Analysis Corporation .trc file

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://demotu.org/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Motion Analysis Corporation (MAC, http://www.motionanalysis.com/) builds motion capture systems and their software (e.g., Cortex) generates files in ASCII and binary formats for the different signals (kinematics, analog data, force plate data, etc.). Here are functions for reading most of the files saved in ASCII format. These files have headers with few lines with meta data and the signals are stored in columns and the rows for the different frames (instants of time).

        The ".trc" (Track Row Column) file in ASCII contains X-Y-Z position data for the reflective markers from a motion capture trial. The position data for each marker is organized into 3 columns per marker (X, Y and Z position) with each row being a new frame. The position data is relative to the global coordinate system of the capture volume and the position values are in the units used for calibration.

        The ".anc" (Analog ASCII Row Column) file contains ASCII analog data in row-column format. The data is derived from ".anb" analog binary files. These binary ".anb" files are generated simultaneously with video ".vc" files if an optional analog input board is used in conjunction with video data capture.

        The ".cal" file contains force plate calibration parameters. 

        The ".forces" file contains force plate data. The data is saved based on the "forcepla.cal" file of the trial and converts the raw force plate data into calibrated forces. The units used are Newtons and Newton-meters and each line in the file equates to one analog sample.

        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy import signal
    import sys
    _sys.path.insert(1, './../functions')
    from read_trc import read_trc
    return (read_trc,)


@app.cell
def _(read_trc):
    print(read_trc.__doc__)
    return


@app.cell
def _():
    import sys, os

    path2 = r'./../data/'
    fname = os.path.join(path2, 'arm26_elbow_flex.trc')
    return fname, os, path2


@app.cell
def _(fname, read_trc):
    (h, _df) = read_trc(fname, fname2='', dropna=True, na=0.0, fmt='uni')
    _df.head()
    return (h,)


@app.cell
def _(h):
    h
    return


@app.cell
def _(fname, read_trc):
    (h_1, _df) = read_trc(fname, fname2='', dropna=True, na=0.0, fmt='multi')
    _df.head()
    return


@app.cell
def _(fname, read_trc):
    _da = read_trc(fname, fname2='', dropna=True, na=0.0, fmt='xarray')
    _da
    return


@app.cell
def _(os, path2):
    fname_1 = os.path.join(path2, 'arm26_elbow_flex_e.trc')
    return (fname_1,)


@app.cell
def _(fname_1, read_trc):
    (h_2, _df) = read_trc(fname_1, fname2='', dropna=False, na=0.0, fmt='multi')
    _df.head()
    return


@app.cell
def _(fname_1, read_trc):
    (h_3, _df) = read_trc(fname_1, fname2='', dropna=True, na=0.0, fmt='multi')
    _df.head()
    return


@app.cell
def _(fname_1, read_trc):
    _da = read_trc(fname_1, fname2='', dropna=True, na=0.0, fmt='xarray')
    _da
    return


@app.cell
def _(read_trc):
    (h_4, _data) = read_trc('./../data/walk.trc', fname2='', dropna=False, na=0.0, fmt='uni')
    _data
    return


@app.cell
def _(read_trc):
    (h_5, _data) = read_trc('./../data/walk.trc', fname2='', dropna=False, na=0.0, fmt='multi')
    _data
    return


@app.cell
def _(read_trc):
    (h_6, _data) = read_trc('./../data/walk.trc', fname2='', dropna=True, na=0.0, fmt='multi')
    _data
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
