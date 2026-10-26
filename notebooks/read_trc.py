# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "pandas>=2.2", "xarray", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read Cortex Motion Analysis Corporation .trc file

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A marker file opens successfully, but several trajectories are entirely empty. Should they be removed, preserved as NaN, or replaced?

    We will read the same TRC data as a flat table, a hierarchical table, and an xarray object, while checking units and empty-marker handling.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). Public `walk1.trc` and `walk.trc` files replace the unavailable arm-model files in the original notebook.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Motion Analysis Corporation (MAC, https://www.motionanalysis.com/) builds motion capture systems and their software (e.g., Cortex) generates files in ASCII and binary formats for the different signals (kinematics, analog data, force plate data, etc.). Here are functions for reading most of the files saved in ASCII format. These files have headers with few lines with meta data and the signals are stored in columns and the rows for the different frames (instants of time).

    The ".trc" (Track Row Column) file in ASCII contains X-Y-Z position data for the reflective markers from a motion capture trial. The position data for each marker is organized into 3 columns per marker (X, Y and Z position) with each row being a new frame. The position data is relative to the global coordinate system of the capture volume and the position values are in the units used for calibration.

    The ".anc" (Analog ASCII Row Column) file contains ASCII analog data in row-column format. The data is derived from ".anb" analog binary files. These binary ".anb" files are generated simultaneously with video ".vc" files if an optional analog input board is used in conjunction with video data capture.

    The ".cal" file contains force plate calibration parameters.

    The ".forces" file contains force plate data. The data is saved based on the "forcepla.cal" file of the trial and converts the raw force plate data into calibrated forces. The units used are Newtons and Newton-meters and each line in the file equates to one analog sample.
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        from notebook_resources import repo_file
    except ModuleNotFoundError as _error:
        if _error.name != "notebook_resources":
            raise
        raise ModuleNotFoundError(
            "This notebook needs notebook_resources.py. Download it from:\n"
            "https://raw.githubusercontent.com/BMClab/BMC/master/functions/notebook_resources.py\n"
            "Save it in the same folder as this notebook, then rerun this cell.",
            name="notebook_resources",
        ) from None
    return (repo_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The reader below is included in this notebook. Pandas labels the tables and xarray labels the time, marker, and component dimensions.
    """)
    return


@app.function
def read_trc(fname, fname2='', units='', dropna=False, na=0.0, fmt='multi', show_msg=True):
    """Read .trc file format from Cortex MAC.

    This function: 1. Read trc file; 2. Can delete or replace markers (columns)
    with empty data; 3. Correct number of markers in the header according to
    the actual number of non-empty markers; 4. Can save a '.trc' file with
    updated information and data; 5. Return header information (optional) and
    data (marker position) as dataframe or dataarray.

    The .trc (track row column) file in ASCII contains X-Y-Z position
    data for the reflective markers from a motion capture trial. The
    position data for each marker is organized into 3 columns per marker
    (X, Y and Z position) with each row being a new frame. The position
    data is relative to the global coordinate system of the capture volume
    and the position values are in the units used for calibration.

    Parameters
    ----------
    fname : string
        Full file name of the .trc file to be opened.
    fname2 : string (default = '')
        Full file name of the .trc file to be saved with updated information
        and data if desired.
        If fname2 is '', no file is saved.
        If fname2 is '=', the original file name will be used.
        If fname2 is a string with length between 1 and 3 (other than '='),
        e.g., '_2', this string is appended to the original file name.
    units : string (default = '')
        Change the units of the data if desired.
        Accepted output units are 'm' or 'mm'.
    dropna : bool (default = False)
        True: Delete a marker only if all three XYZ columns contain no data.
        Partially missing markers keep their complete XYZ group and NaNs.
        False: preserve column and replace column values by parameter `na`
        (see below) if inputed, otherwise maintain default pandas value (NaN).
    na : float or None (default = 0.0)
        Value to replace (if `dropna` is False) column values if this column
        has only missing or NaN values. Input None to maintain default pandas
        value for this case (NaN).
    fmt : string (default = 'multi')
        Format of the output: 'uni', 'multi', 'xarray'
        'uni': returns variable with trc header info plus pandas
        dataframe with markerxyz as labels and "Frame#" and "Time" as columns;
        'multi': returns variable with trc header info plus multilabel pandas
        dataframe  with "Marker", "Coordinate" and "XYZ", as labels and "Time"
        as index;
        'xarray': returns variable as dataarray xarray and trc header info as
        attributes of this dataarray.
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).

    Returns
    -------
    h : Python dictionary with .trc header info (if `fmt` = 'uni' or 'multi')
        keys: header (the .trc full header), data_rate (Hz), camera_rate (Hz),
        nframes, nmarkers, markers (names), xyz (X1,Y1,Z1...), units.
    data : pandas dataframe or xarray dataarray
        Three possible output formats according to the `fmt` option:
        'uni': dataframe with shape (nframes, 2+3*nmarkers) with markerxyz as
        labels and columns: Frame#, time and position data, or
        'multi': fataframe with shape (nframes, 3*nmarkers) with "Marker",
        "Coordinate" and "XYZ" as labels, "Time" as index, and data position
        as columns, or
        'xarray': dataarray with dims=['time', 'marker', 'component'] and trc
        header info as attributes of this dataarray.

    Examples
    --------

    Notes
    -----

    """
    import numpy as np
    import pandas as pd
    import csv
    import os
    with open(file=fname, mode='rt', encoding='utf-8', newline='') as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end='')
        read = csv.reader(f, delimiter='\t')
        header = [next(read) for x in range(5)]
        nmarkers = int((len(header[3]) - 2) / 3)
        markers = np.asarray(header[3])[np.arange(2, 2 + 3 * nmarkers, 3)].tolist()
        if len(markers) != len(set(markers)):
            from collections import Counter
            d = {a: [''] + list(range(2, b + 1)) if b > 1 else '' for a, b in Counter(markers).items()}
            markers = [i + str(d[i].pop(0)) if len(d[i]) else i for i in markers]
        markers3 = [m for m in markers for i in range(3)]
        XYZ = [i[0] for i in header[4][2:5]]
        xyz = [i[0].lower() for i in header[4][2:5]]
        markersxyz = [a + b for a, b in zip(markers3, xyz * nmarkers)]
        df = pd.read_csv(f, sep='\t', names=['Frame#', 'Time'] + markersxyz, index_col=False, encoding='utf-8', engine='c', skip_blank_lines=True)
        if dropna:
            keep = [0, 1]
            for start in range(2, df.shape[1], 3):
                if not df.iloc[:, start:start + 3].isna().to_numpy().all():
                    keep.extend(range(start, start + 3))
            df = df.iloc[:, keep].copy()
        elif na is not None:
            for col in df:
                if df.loc[:, col].isnull().sum() == df.shape[0]:
                    df.loc[:, col] = na
        nmarkers = int((df.shape[1] - 2) / 3)
        if header[2][3] != str(nmarkers):
            if show_msg:
                print(f' Number of markers changed from {header[2][3]} to {nmarkers}.')
            header[2][3] = str(nmarkers)
        header[3] = ['' if c[-1] in ['y', 'z'] else c[:-1] if c[-1] in ['x'] else c for c in df.columns.values.tolist()] + ['']
        markers = np.asarray(header[3])[np.arange(2, 2 + 3 * nmarkers, 3)].tolist()
        n3 = np.repeat(range(1, nmarkers + 1), 3).tolist()
        XYZs = [a + str(b) for a, b in zip(XYZ * nmarkers, n3)]
        header[4] = ['', ''] + XYZs
        if units in ("m", "mm") and units != header[2][4]:
            df = df.astype({column: float for column in df.columns[2:]})
        if units == 'm':
            if header[2][4] == 'mm':
                df.iloc[:, 2:] = df.iloc[:, 2:] / 1000
                header[2][4] = 'm'
                if show_msg:
                    print(' Units changed from {} to {}'.format('"mm"', '"m"'))
        elif units == 'mm':
            if header[2][4] == 'm':
                df.iloc[:, 2:] = df.iloc[:, 2:] * 1000
                header[2][4] = 'mm'
                if show_msg:
                    print(' Units changed from {} to {}'.format('"m"', '"mm"'))
        if show_msg:
            print('done.')
    if fname2:
        if fname2 == '=':
            fname2 = fname
        elif len(fname2) <= 3:
            name, extension = os.path.splitext(fname)
            fname2 = name + fname2 + extension
        write_trc(fname2, header, df, show_msg)
    h = {'header': header, 'data_rate': float(header[2][0]), 'camera_rate': float(header[2][1]), 'nframes': int(header[2][2]), 'nmarkers': int(header[2][3]), 'markers': markers, 'xyz': XYZs, 'units': header[2][4], 'fname': fname, 'fname2': fname2}
    if fmt.lower() == 'uni':
        return (h, df)
    elif fmt.lower() == 'multi':
        df.drop(labels='Frame#', axis=1, inplace=True)
        df.set_index('Time', inplace=True)
        df.index.name = 'Time'
        cols = [s[:-1] for s in df.columns.str.replace('.', '_', regex=False)]
        df.columns = [cols, XYZ * int(df.shape[1] / 3), XYZs]
        df.columns.set_names(names=['Marker', 'Coordinate', 'XYZ'], level=[0, 1, 2], inplace=True)
        return (h, df)
    else:
        import xarray as xr
        name = 'Position data'
        dims = ['time', 'marker', 'component']
        time = df.values[:, 1]
        component = XYZ
        coords = [time, markers, component]
        da = xr.DataArray(data=df.values[:, 2:].reshape(-1, nmarkers, 3, order='C'), dims=dims, coords=coords, name=name, attrs=h)
        da['time'].attrs['units'] = 's'
        da['component'].attrs['units'] = h['units']
        return da


@app.function
def write_trc(fname, header, df, show_msg=True):
    """Write .trc file format from Cortex MAC.

    See the read_forces.py function.

    Parameters
    ----------
    fname : string
        Full file name of the .trc file to be saved.
    header : list of lists
        header for the .trc file
    df : pandas dataframe
        dataframe with data for the .trc file (with frame and time columns)
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).
    """
    import numpy as np
    import pandas as pd
    import csv
    import os
    with open(file=fname, mode='wt', encoding='utf-8', newline='') as f:
        if show_msg:
            print(f'Saving file "{fname}" ... ', end='')
        f.writelines(('\t'.join(line) + '\n' for line in header))
        f.write('\n')
        data = df.to_csv(header=None, index=None, sep='\t', lineterminator='\n', float_format='%.6f')
        f.write(data.replace('\n', '\t\n'))
        if show_msg:
            print('done.')


@app.cell
def _(read_trc):
    print(read_trc.__doc__)
    return


@app.cell
def _(repo_file):
    fname = str(repo_file("data/walk1.trc"))
    return (fname,)


@app.cell
def _(fname, read_trc):
    (h, _df) = read_trc(fname, fname2='', dropna=True, na=0.0, fmt='uni')
    _df.head()
    return (h,)


@app.cell
def _(h):
    h
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A TRC file declares 28 markers. How many numeric columns does a flat table have when frame number and time are included?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 1

    There are $28\times3+2=86$ columns. A table indexed by time and containing only marker coordinates has 84 data columns.
    ///
    """)
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
def _(repo_file):
    fname_1 = str(repo_file("data/walk.trc"))
    return (fname_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Empty markers are a data decision

    The second file contains extra empty trajectories. `dropna=True` removes a marker only when all three XYZ columns are empty. Partially missing XYZ groups stay intact. `dropna=False, na=None` preserves NaNs, while `na=0` fills entirely empty columns with zeros. A filled zero is not a newly measured position; partial gaps remain NaN. Keep this distinction when plotting or fitting a model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Which option best preserves an entirely missing trajectory as missing data?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 2

    Use `dropna=False, na=None`. Dropping removes the trajectory; `na=0` assigns a numerical coordinate that can be confused with a real measurement.
    ///
    """)
    return


@app.cell
def _(fname_1, read_trc):
    (h_2, _df) = read_trc(fname_1, fname2='', dropna=False, na=None, fmt='multi')
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
def _(read_trc, repo_file):
    (h_4, _data) = read_trc(str(repo_file('data/walk.trc')), fname2='', dropna=False, na=0.0, fmt='uni')
    _data
    return


@app.cell
def _(read_trc, repo_file):
    (h_5, _data) = read_trc(str(repo_file('data/walk.trc')), fname2='', dropna=False, na=0.0, fmt='multi')
    _data
    return


@app.cell
def _(read_trc, repo_file):
    (h_6, _data) = read_trc(str(repo_file('data/walk.trc')), fname2='', dropna=True, na=0.0, fmt='multi')
    _data
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Does reading a file confirm that its header is correct?

    2. What is the advantage of named xarray dimensions?

    3. Does this guide overwrite the source file?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Checkpoint answers

    1. No; compare counts, units, and time intervals with the data.
    2. Selections can refer explicitly to time, marker, and component.
    3. No; all example calls use `fname2=""`.
    ///
    """)
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
    **Problem 1.** Convert a marker coordinate of 1,250 mm into metres.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 1

    1.25 m.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** How many coordinates per frame are stored for 12 markers?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 2

    36.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Compare outputs of `fmt="uni"`, `fmt="multi"`, and `fmt="xarray"` without changing the recording.

    ### Video lectures on the Internet

    [OpenSim video tutorials](https://www.youtube.com/@OpenSimVideos) show how motion-capture files enter biomechanics workflows.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [OpenSim: marker data in TRC files](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089972/Marker+.trc+Files).

    - [xarray data structures](https://docs.xarray.dev/en/stable/user-guide/data-structures.html).
    """)
    return


if __name__ == "__main__":
    app.run()
