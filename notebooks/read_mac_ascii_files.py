# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "pandas>=2.2", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read and write Cortex ASCII files

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A text file looks readable, yet its columns are meaningless until you know the units, sampling rate, and layout. What must you check before plotting it?

    We will inspect a TRC file, make a temporary round trip through the writer, and relate marker, analog, calibration, and force files.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The examples use the public `walk1.trc` and `walk1.forces` files. A temporary output and a download button leave the source recording intact.
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
def _(mo):
    mo.md(r"""
    ## Python setup

    The current reader and writer functions are included below. Expand the code cells to inspect their handling of headers and columns.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import tempfile
    return mo, np, pd, Path, tempfile


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


@app.function
def read_trc(fname, fname2='', units='', dropna=False, na=0.0, df_multi=True, show_msg=True):
    """Read .trc file format from Cortex MAC.

    This function: 1. Delete markers (columns) of empty data; 2. Correct
    number of markers in the header according to the actual number of
    non-empty markers; 3. Save a '.trc' file with updated information and
    data; 4. Returns header information and data.

    The .trc (Track Row Column) file in ASCII contains X-Y-Z position
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
    df_multi : bool (default = True)
        Whether to output data as pandas multilabel dataframe with "Marker",
        "Coordinate" and "XYZ", as labels and "Time" as index (True) or simple
        pandas dataframe with markerxyz as labels and "Frame#" and "Time" as
        columns (False).
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).

    Returns
    -------
    h : Python dictionary with .trc header information
        keys: header (the .trc full header), data_rate (Hz), camera_rate (Hz),
        nframes, nmarkers, markers (names), xyz (X1,Y1,Z1...), units.
    data : pandas dataframe
        Two possible output formats according to the `df_multi` option:
        Dataframe with shape (nframes, 2+3*nmarkers) with markerxyz as labels
        and columns: Frame#, time and position data.
        Dataframe with shape (nframes, 3*nmarkers) with "Marker", "Coordinate"
        and "XYZ" as labels, "Time" as index, and data position as columns.
    """
    import csv
    import os
    import numpy as np
    import pandas as pd
    with open(file=fname, mode='rt', encoding='utf-8', newline='') as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end='')
        read = csv.reader(f, delimiter='\t')
        header = [next(read) for x in range(6)]
        nmarkers = int((len(header[3]) - 2) / 3)
        markers = np.asarray(header[3])[np.arange(2, 2 + 3 * nmarkers, 3)].tolist()
        x = markers
        markers = [v + str(x[:i].count(v) + 1) if x.count(v) > 1 and x[:i].count(v) > 0 else v for i, v in enumerate(x)]
        markers3 = [m for m in markers for i in range(3)]
        markersxyz = [a + b for a, b in zip(markers3, ['x', 'y', 'z'] * nmarkers)]
        df = pd.read_csv(f, sep='\t', names=['Frame#', 'Time'] + markersxyz, index_col=False, encoding='utf-8', engine='c')
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
        xyz = [a + str(b) for a, b in zip(['X', 'Y', 'Z'] * nmarkers, n3)]
        header[4] = ['', ''] + xyz
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
    if len(fname2):
        if fname2 == '=':
            fname2 = fname
        elif len(fname2) <= 3:
            name, extension = os.path.splitext(fname)
            fname2 = name + fname2 + extension
        write_trc(fname2, header, df, show_msg)
    h = {'header': header, 'data_rate': float(header[2][0]), 'camera_rate': float(header[2][1]), 'nframes': int(header[2][2]), 'nmarkers': int(header[2][3]), 'markers': markers, 'xyz': xyz, 'units': header[2][4], 'fname': fname, 'fname2': fname2}
    if df_multi:
        df.drop(labels='Frame#', axis=1, inplace=True)
        df.set_index('Time', inplace=True)
        df.index.name = 'Time'
        cols = [s[:-1] for s in df.columns.str.replace('.', '_')]
        df.columns = [cols, list('XYZ') * int(df.shape[1] / 3), xyz]
        df.columns.set_names(names=['Marker', 'Coordinate', 'XYZ'], level=[0, 1, 2], inplace=True)
    return (h, df)


@app.function
def read_anc(fname, show_msg=True):
    """Read .anc file format from Cortex MAC.

    The .anc (Analog ASCII Row Column) file contain ASCII analog data
    in row-column format. The data is derived from *.anb analog binary
    files. These binary *.anb files are generated simultaneously with
    video *.vc files if an optional analog input board is used in
    conjunction with video data capture.

    Parameters
    ----------
    fname : string
        full file name of the .anc file to be opened

    Returns
    -------
    h : Python dictionary
        .anc header information
        keys: nbits, polarity, nchannels, data_rate, ch_names, ch_ranges
    data : pandas dataframe
        analog data with shape (nframes, nchannels)
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).

    """
    import csv
    import os
    import numpy as np
    import pandas as pd
    with open(file=fname, mode='rt', encoding='utf-8', newline='') as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end='')
        read = csv.reader(f, delimiter='\t')
        header = [next(read) for x in range(11)]
        h = {'nbits': int(header[3][1]), 'polarity': header[1][3], 'nchannels': int(header[2][7]), 'data_rate': float(header[3][3]), 'ch_names': header[8], 'ch_ranges': header[10]}
        h['ch_names'] = h['ch_names'][1:-1]
        h['ch_ranges'] = np.asarray(h['ch_ranges'][1:-1], dtype=float)
        data = pd.read_csv(f, sep='\t', names=h['ch_names'], engine='c', usecols=np.arange(1, 1 + h['nchannels']))
        data *= h['ch_ranges'] / (2 ** h['nbits'] / 2 - 2)
        if show_msg:
            print('done.')
    return (h, data)


@app.function
def read_cal(fname, show_msg=True):
    """Read .cal file format from Cortex MAC.

    The .cal (force plate calibration parameters) file in ASCII contains:

    <forceplate number> {1}
    <scale> <length (cm)> <width (cm)> {2}
    <N x N calibration matrix (the inverse sensitivity matrix)> {3}
    <true origin in relation to the geometric center (cm)>
    <geometric center in relation to LCS origin (cm)>
    <3 x 3 orientation matrix>
    ...repeat for next force plate...

    {1}: for a Kistler force plate, there is a 'K' after the number
    {2}: the scale is the inverse of the gain
    {3}: N equal 8 for Kistler and equal 6 for all AMTI and Bertec

    Parameters
    ----------
    fname : string
        full file name of the .trc file to be opened
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).

    Returns
    -------
    forcepla : Python dictionary
        parameter from the froce plate calibration file
        keys: 'fp', 'scale', 'size', 'cal_matrix', 'origin', 'center', 'orientation'
    """
    import csv
    import numpy as np
    import pandas as pd
    fp, scale, size, cal_matrix, origin, center, orientation = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with open(file=fname, mode="rt", encoding="utf-8", newline="") as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end="")
        reader = (line.split() for line in f if line.strip())
        for row in reader:
            # force plate number
            fp.append(int(row[0].removesuffix("K")))
            # number of rows for Kistler or AMTI/Bertec force plate
            n = 8 if row[0][-1] == "K" else 6
            # scale (inverse of the gain)
            scale_size = np.array(next(reader)).astype(float)
            scale.append(scale_size[0])
            # force plate length (cm) and width (cm)
            size.append(scale_size[1:])
            # calibration matrix (the inverse sensitivity matrix)
            matrix = [next(reader) for x in range(n)]
            cal_matrix.append(np.array(matrix).astype(float))
            # true origin in relation to the geometric center (cm)
            origin.append(np.array(next(reader)).astype(float))
            # geometric center in relation to LCS origin (cm)
            center.append(np.array(next(reader)).astype(float))
            # 3 x 3 orientation matrix
            orienta = [next(reader) for x in range(3)]
            orientation.append(np.array(orienta).astype(float))

        forcepla = {
            "fp": fp,
            "scale": scale,
            "size": size,
            "cal_matrix": cal_matrix,
            "origin": origin,
            "center": center,
            "orientation": orientation,
        }
        if show_msg:
            print("done.")

    return forcepla


@app.function
def read_forces(fname, time=True, forcepla=[], mm2m=True, show_msg=True):
    """Read .forces file format from Cortex MAC.

    The .forces file in ASCII contains force plate data. The data is saved
    based on the forcepla.cal file of the trial and converts the raw force
    plate data into calibrated forces. The source units used are N, mm (COP), and N mm (free moment) and each line in the file equates to one analog sample.

    Example of .forces file structure:

    [Force Data]
    NumberOfForcePlates=7
    SampleRate=150.000000
    NumberOfSamples=150
    #Sample FX1 FY1 FZ1 X1 Y1 Z1 MZ1 FX2 ...
    ...

    Parameters
    ----------
    fname : string
        full file name of the .forces file to be opened
    time : bool (default = True)
        Whether the data index is in units of time (True) or not (False).
    forcepla : list of integers (default = [])
        List of force plates to read. An empty list reads all force plates.
        Enter a list of force plate numbers to read.
    mm2m : bool (default = True)
        Whether to convert COP from mm to m and free moment from N mm to N m.
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).

    Returns
    -------
    h : Python dictionary
        .forces header information
        keys: name, nforceplates, data_rate, nsamples, ch_names
    df : pandas dataframe
        force plate data with shape (nsamples, 7*nforceplates)

    """
    import csv
    import numpy as np
    import pandas as pd
    with open(file=fname, mode="rt", encoding="utf-8", newline="") as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end="")
        # get header information
        read = csv.reader(f, delimiter="\t")
        header = [next(read) for x in range(5)]
        h = {
            "name": header[0][0],
            "NumberOfForcePlates": int(header[1][0].split("=")[1]),
            "SampleRate": float(header[2][0].split("=")[1]),
            "NumberOfSamples": int(header[3][0].split("=")[1]),
            "ch_names": header[4][1:],
        }
        # Match the entire numeric suffix, never "1" inside plate "10".
        import re
        channel_pattern = re.compile(r"^(FX|FY|FZ|X|Y|Z|MZ)(\d+)$")
        requested = ([int(forcepla)] if np.isscalar(forcepla)
                     else [int(plate) for plate in forcepla])
        all_names = h["ch_names"]
        df = pd.read_csv(f, sep="\t", header=None, usecols=range(1, len(all_names) + 1),
                         dtype=float, engine="c")
        df.columns = all_names
        if requested:
            selected = []
            for plate in dict.fromkeys(requested):
                names = [name for name in all_names
                         if (match := channel_pattern.fullmatch(name))
                         and int(match.group(2)) == plate]
                if len(names) != 7:
                    raise ValueError(f"Force plate {plate} does not have seven channels")
                selected.extend(names)
            df = df.loc[:, selected].copy()
            h["NumberOfForcePlates"] = len(set(requested))
            h["ch_names"] = selected
        if mm2m:
            cols = [name for name in df.columns
                    if (match := channel_pattern.fullmatch(name))
                    and match.group(1) in ("X", "Y", "Z", "MZ")]
            df.loc[:, cols] = df.loc[:, cols] / 1000
        if time:
            df.index = df.index / h["SampleRate"]
            df.index.name = "Time"
        if show_msg:
            print("done.")

    return h, df


@app.function
def write_trc(fname, header, df, show_msg=True):
    """Write .trc file format from Cortex MAC.

    See the read_forces.py function.

    Parameters
    ----------
    fname : string
        Full file name of the .forces file to be saved.
    header : list of lists
        header for the .forces file
    df : pandas dataframe
        dataframe with data for the .forces file (with frame and time columns)
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).
    """
    import csv
    import os
    import numpy as np
    import pandas as pd
    with open(file=fname, mode='wt', encoding='utf-8', newline='') as f:
        if show_msg:
            print(f'Saving file "{fname}" ... ', end='')
        f.writelines(('\t'.join(line) + '\n' for line in header))
        data = df.to_csv(header=None, index=None, sep='\t', lineterminator='\n', float_format='%.6f')
        f.write(data.replace('\n', '\t\n'))
        if show_msg:
            print('done.')


@app.function
def write_v3dtxt(fname, trc, forces, freq=0, show_msg=True):
    """Write synchronized marker/force tables in Visual3D ASCII format.

    TRC input is a filename or a flat DataFrame with Frame#, Time and complete
    XYZ marker groups. Force input is a filename or a DataFrame whose index is
    time in seconds. Empty strings omit either stream. Filename TRC data are
    converted to meters; DataFrame inputs must already use m, N and N m.

    Timestamps must increase, share their start, and describe the same recording
    interval (within timestamp rounding). The output uses the highest native
    or requested ``freq`` in Hz. Linear interpolation uses actual timestamps
    only over the common measured time span, never beyond either last sample.
    Thus final analog samples after the last marker sample are omitted.
    Missing coordinate values remain missing; interpolation does not bridge
    gaps. Frame numbers restart at 1, but the original time origin is retained.

    A force DataFrame with a RangeIndex has no time information; in that case
    ``freq`` must explicitly specify its sampling rate. For a time-indexed
    force DataFrame, ``freq`` specifies only a requested output rate.
    """
    import csv
    import os
    import numpy as np
    import pandas as pd

    if not np.isfinite(freq) or freq < 0:
        raise ValueError("freq must be a finite, nonnegative sampling rate")
    trc_rate = None
    force_rate = None
    if isinstance(trc, (str, os.PathLike)):
        if str(trc):
            header, trc = read_trc(trc, fname2="", units="m", na=None,
                                   df_multi=False, show_msg=show_msg)
            trc_rate = header["data_rate"]
        else:
            trc = pd.DataFrame()
    if isinstance(forces, (str, os.PathLike)):
        if str(forces):
            header, forces = read_forces(forces, show_msg=show_msg)
            force_rate = header["SampleRate"]
        else:
            forces = pd.DataFrame()
    if trc.empty and forces.empty:
        raise ValueError("Provide marker or force data with timestamps")

    def timeline(values, rate, name):
        times = np.asarray(values, dtype=float)
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{name} time must contain at least two increasing finite values")
        rate = float(rate) if rate is not None else (len(times) - 1) / (times[-1] - times[0])
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"{name} sampling rate must be positive")
        return times, rate

    streams = []
    if not trc.empty:
        if trc.shape[1] < 5 or (trc.shape[1] - 2) % 3:
            raise ValueError("TRC data must contain Frame#, Time and complete XYZ marker groups")
        trc_time, trc_rate = timeline(trc.iloc[:, 1], trc_rate, "TRC")
        streams.append((trc_time, trc_rate))
    if not forces.empty:
        if isinstance(forces.index, pd.RangeIndex):
            if not freq:
                raise ValueError("Force data need a time index or an explicit sampling frequency")
            force_values = np.arange(len(forces)) / freq
            force_rate = freq
        else:
            force_values = forces.index
        force_time, force_rate = timeline(force_values, force_rate, "Force")
        streams.append((force_time, force_rate))
    if len(streams) == 2:
        # Native sample bins may end at different last sample timestamps.
        # Compare their exclusive ends, allowing the precision of TRC text.
        tolerance = max(1e-6, 0.1 * max(1 / rate for _, rate in streams))
        if not np.isclose(trc_time[0], force_time[0], rtol=0, atol=1e-6):
            raise ValueError("TRC and force time origins differ; synchronize them before export")
        if abs((trc_time[-1] + 1 / trc_rate) - (force_time[-1] + 1 / force_rate)) > tolerance:
            raise ValueError("TRC and force data describe different recording intervals")

    start = max(times[0] for times, _ in streams)
    end = min(times[-1] for times, _ in streams)
    output_rate = max([float(freq)] + [rate for _, rate in streams])
    count = int(np.floor((end - start) * output_rate + 1e-8)) + 1
    output_time = np.minimum(start + np.arange(count) / output_rate, end)

    def interpolate(table, times):
        return np.column_stack([
            np.interp(output_time, times, column, left=np.nan, right=np.nan)
            for column in table.to_numpy(dtype=float).T
        ])

    values = [np.arange(1, count + 1), output_time]
    signal_names = ["TIME"]
    signal_types = ["FRAME_NUMBERS"]
    components = ["0"]
    if not trc.empty:
        values.extend(interpolate(trc.iloc[:, 2:], trc_time).T)
        marker_columns = list(trc.columns[2:])
        for start_column in range(0, len(marker_columns), 3):
            group = marker_columns[start_column:start_column + 3]
            names = [str(column)[:-1] for column in group]
            if len(set(names)) != 1:
                raise ValueError("Marker columns must be named as matching x/y/z triplets")
            signal_names.extend(names)
        signal_types.extend(["TARGET"] * len(marker_columns))
        components.extend(list("XYZ") * (len(marker_columns) // 3))
    if not forces.empty:
        values.extend(interpolate(forces, force_time).T)
        signal_names.extend(forces.columns)
        signal_types.extend(["ANALOG"] * forces.shape[1])
        components.extend(["0"] * forces.shape[1])
    data = np.column_stack(values)
    rows = [
        [""] + ["default"] * len(signal_names),
        [""] + signal_names,
        [""] + signal_types,
        [""] + ["ORIGINAL"] * len(signal_names),
        ["ITEM"] + components,
    ]
    with open(file=fname, mode="wt", encoding="utf-8", newline="") as stream:
        if show_msg:
            print(f'Saving file "{fname}" ... ', end="")
        writer = csv.writer(stream, delimiter="\t")
        writer.writerows(rows)
        writer.writerows(data)
        if show_msg:
            print("done.")


@app.function
def grf_moments(data, O, show_msg=True):
    """Calculate force plate moments around its origin given
    3 forces, 2 COPs, 1 free moment, and its geometric position.

    Parameters
    ----------
    data : Numpy array (n, 7)
        array with [Fx, Fy, Fz, COPx, COPy, COPz, Tz].
    O : Numpy array-like or list
        origin [x,y,z] of the force plate in the motion capture coordinate system [in meters].
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).

    Returns
    -------
    grf : Numpy array (n, 6)
        array with [Fx, Fy, Fz, Mx, My, Mz]
    """
    import csv
    import os
    import numpy as np
    import pandas as pd
    values = np.asarray(data, dtype=float)
    origin = np.asarray(O, dtype=float)
    force = values[:, :3]
    lever = values[:, 3:6] - origin
    moment = np.cross(lever, force)
    moment[:, 2] += values[:, 6]
    return np.column_stack((force, moment))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect a marker recording

    Read a flat table first so frame number and time remain explicit columns. The header reports positions in millimetres; requesting metres converts the numeric positions and updates the unit field together.
    """)
    return


@app.cell
def _():
    h, df = read_trc(repo_file("data/walk1.trc"), fname2="", units="m", df_multi=False)
    df.head()
    return df, h


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Why is editing a header label from mm to m without changing the numbers incorrect?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 1': mo.md(r"""
    A coordinate of 1,000 mm represents 1 m. Relabelling 1,000 as metres changes the physical position by a factor of 1,000. Convert the values and the metadata together.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write a copy and read it back

    A round trip is a practical check that writing preserves the recorded values and labels. The generated file is kept in a temporary directory and is offered as a download.
    """)
    return


@app.cell
def _(Path, df, h, mo, np, tempfile):
    _folder = Path(tempfile.mkdtemp(prefix="bmc-trc-"))
    _copy = _folder / "walk1_metres.trc"
    write_trc(str(_copy), h["header"], df, show_msg=False)
    _roundtrip_h, _roundtrip = read_trc(_copy, fname2="", df_multi=False, show_msg=False)
    np.testing.assert_allclose(_roundtrip.to_numpy(), df.to_numpy(), atol=1e-6)
    trc_bytes = _copy.read_bytes()
    mo.download(trc_bytes, filename="walk1_metres.trc", label="Download the converted copy")
    return (trc_bytes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Why compare values with a tolerance after saving a decimal text file?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 2': mo.md(r"""
    The writer rounds to a finite number of decimal places. Exact binary equality is too strict after that formatting step; a tolerance tied to the written precision checks the intended numerical accuracy.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect the force table

    Each force plate contributes force components, a centre-of-pressure position, and a free moment. The supplied Cortex export uses N, mm, and N·mm before conversion. `mm2m=True` converts the three COP coordinates and the free moment to m and N·m. Match the convention to your acquisition software before using a different file.
    """)
    return


@app.cell
def _():
    force_header, forces = read_forces(repo_file("data/walk1.forces"), mm2m=True, show_msg=False)
    forces.head()
    return force_header, forces


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transfer the moment to another origin

    For a point of application P and target origin O, the moment is $(P-O)\times F$ plus the free moment. Coordinate origins and force directions must share one frame. The included correction uses this cross product explicitly, including nonzero COP height.
    """)
    return


@app.cell
def _(np):
    grf_moments(np.array([[5., 6., 7., 2., 3., 4., 8.]]), [1., 1., 1.], show_msg=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** For a vertical force `(0,0,100)` N at `(0.2,0,0)` m, with origin `(0,0,0)` and no free moment, find the moment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 3': mo.md(r"""
    Using the right-hand cross product, $(0.2,0,0)\times(0,0,100)=(0,-20,0)$ N·m. The negative Y component is fixed by the chosen right-handed axes.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other Cortex text formats

    The retained `read_anc` reader interprets analog-converter counts using header ranges. `read_cal` reads force-plate geometry and calibration matrices. `write_v3dtxt` combines marker and force data for Visual3D; use it only after checking acquisition time alignment and resampling assumptions. The helper definitions remain available above, while the [paired TRC/force guide](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_trc_forces_files.py) demonstrates alignment on two recorded trials.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Is a readable text file automatically self-describing?

    2. Does a round trip prove the physical units were correct?

    3. Why subtract the target origin from the point position?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Checkpoint answers': mo.md(r"""
    1. No; a schema and units are still required.
    2. No; it checks serialization, not physical interpretation.
    3. The cross product needs the lever arm relative to the target origin.
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
    **Problem 1.** Convert 2,000 N·mm to N·m.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 1': mo.md(r"""
    2 N·m.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** Give the moment of force `(0,10,0)` N applied at `(0,0,0.3)` m about the origin, with zero free moment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 2': mo.md(r"""
    `(−3, 0, 0)` N·m.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Consult the export documentation for the software version that produced a file.

    ### Video lectures on the Internet

    [OpenSim video tutorials](https://www.youtube.com/@OpenSimVideos) illustrate how labelled motion and force data enter a model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Motion Analysis](https://www.motionanalysis.com/).

    - [OpenSim TRC format](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089972/Marker+.trc+Files).

    - [Cortex helper source](https://github.com/BMClab/BMC/blob/master/functions/io_cortexmac.py).
    """)
    return


if __name__ == "__main__":
    app.run()
