"""Read and write Cortex Motion Analysis Corporation ASCII related files.

Read .trc file:
  read_trc(fname, fname2, units, dropna, na, df_multi, show_msg)
Read .anc file
  read_anc(fname, show_msg)
Read .cal file
  read_cal(fname, show_msg)
Read .forces file
  read_forces(fname, time, show_msg)
Read .mot file
  read mot file format from OpenSim
Read Delsys file
  read Delsys csv file from Cortex MAC
Write .trc file
  write_trc(fname, header, df, show_msg)
Write Visual3d text file from .trc and .forces files or dataframes
  write_v3dtxt(fname, trc, forces, freq=0, show_msg)
Calculate force plate moments around its origin given 3 forces, 2 COPs,
  1 free moment and its geometric position
  grf_moments(data, O, show_msg)

"""

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.3"
__license__ = "MIT"

import csv
import os
from fractions import Fraction

import numpy as np
import pandas as pd
from critic_damp import critic_damp
from detecta import detect_onset
from linear_envelope import linear_envelope
from scipy import signal


def read_trc(
    fname, fname2="", units="", dropna=False, na=0.0, df_multi=True, show_msg=True
):
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
    with open(file=fname, mode="rt", encoding="utf-8", newline="") as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end="")
        # get header information
        read = csv.reader(f, delimiter="\t")
        header = [next(read) for x in range(6)]
        # actual number of markers
        nmarkers = int((len(header[3]) - 2) / 3)
        # column labels
        markers = np.asarray(header[3])[np.arange(2, 2 + 3 * nmarkers, 3)].tolist()
        # rename duplicated markers
        x = markers
        markers = [
            v + str(x[:i].count(v) + 1)
            if (x.count(v) > 1 and x[:i].count(v) > 0)
            else v
            for i, v in enumerate(x)
        ]
        markers3 = [m for m in markers for i in range(3)]
        markersxyz = [a + b for a, b in zip(markers3, ["x", "y", "z"] * nmarkers)]
        # read data
        df = pd.read_csv(
            f,
            sep="\t",
            names=["Frame#", "Time"] + markersxyz,
            index_col=False,
            encoding="utf-8",
            engine="c",
        )
        # Drop only whole empty markers; partial XYZ groups must remain intact.
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
        # update header
        nmarkers = int((df.shape[1] - 2) / 3)
        if header[2][3] != str(nmarkers):
            if show_msg:
                print(f" Number of markers changed from {header[2][3]} to {nmarkers}.")
            header[2][3] = str(nmarkers)
        header[3] = [
            "" if c[-1] in ["y", "z"] else c[:-1] if c[-1] in ["x"] else c
            for c in df.columns.values.tolist()
        ] + [""]
        markers = np.asarray(header[3])[np.arange(2, 2 + 3 * nmarkers, 3)].tolist()
        n3 = np.repeat(range(1, nmarkers + 1), 3).tolist()
        xyz = [a + str(b) for a, b in zip(["X", "Y", "Z"] * nmarkers, n3)]
        header[4] = ["", ""] + xyz
        if units in ("m", "mm") and units != header[2][4]:
            df = df.astype({column: float for column in df.columns[2:]})
        if units == "m":
            if header[2][4] == "mm":
                df.iloc[:, 2:] = df.iloc[:, 2:] / 1000
                header[2][4] = "m"
                if show_msg:
                    print(" Units changed from {} to {}".format('"mm"', '"m"'))
        elif units == "mm":
            if header[2][4] == "m":
                df.iloc[:, 2:] = df.iloc[:, 2:] * 1000
                header[2][4] = "mm"
                if show_msg:
                    print(" Units changed from {} to {}".format('"m"', '"mm"'))

        if show_msg:
            print("done.")

    # save file
    if len(fname2):
        if fname2 == "=":
            fname2 = fname
        elif len(fname2) <= 3:
            name, extension = os.path.splitext(fname)
            fname2 = name + fname2 + extension

        write_trc(fname2, header, df, show_msg)

    # outputs
    h = {
        "header": header,
        "data_rate": float(header[2][0]),
        "camera_rate": float(header[2][1]),
        "nframes": int(header[2][2]),
        "nmarkers": int(header[2][3]),
        "markers": markers,
        "xyz": xyz,
        "units": header[2][4],
        "fname": fname,
        "fname2": fname2,
    }
    if df_multi:  # dataframe with multiple labels
        df.drop(labels="Frame#", axis=1, inplace=True)
        df.set_index("Time", inplace=True)
        df.index.name = "Time"
        cols = [s[:-1] for s in df.columns.str.replace(r".", r"_")]
        df.columns = [cols, list("XYZ") * int(df.shape[1] / 3), xyz]
        df.columns.set_names(
            names=["Marker", "Coordinate", "XYZ"], level=[0, 1, 2], inplace=True
        )

    return h, df


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
    with open(file=fname, mode="rt", encoding="utf-8", newline="") as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end="")
        # get header information
        read = csv.reader(f, delimiter="\t")
        header = [next(read) for x in range(11)]
        h = {
            "nbits": int(header[3][1]),
            "polarity": header[1][3],
            "nchannels": int(header[2][7]),
            "data_rate": float(header[3][3]),
            "ch_names": header[8],
            "ch_ranges": header[10],
        }
        h["ch_names"] = h["ch_names"][1:-1]
        h["ch_ranges"] = np.asarray(h["ch_ranges"][1:-1], dtype=float)
        # analog data
        data = pd.read_csv(
            f,
            sep="\t",
            names=h["ch_names"],
            engine="c",
            usecols=np.arange(1, 1 + h["nchannels"]),
        )
        # convert ADC (bit) values to engineering units:
        data *= h["ch_ranges"] / (2 ** h["nbits"] / 2 - 2)
        if show_msg:
            print("done.")

    return h, data


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


def read_mot(fname, show_msg=True):
    """Read .mot file format from OpenSim.

    Reads the OpenSim header written by `write_mot`, and also the older SIMM
    header ("datacolumns 19", "datarows 1260", "range 0 2.797778").

    Example of .mot file structure (OpenSim format):

    walk_grf.mot
    version=1
    nRows=1260
    nColumns=19
    inDegrees=no
    endheader
    time	R_ground_force_vx	R_ground_force_vy	R_ground_force_vz ...
    ...

    Parameters
    ----------
    fname : string
        full file name of the .mot file to be opened
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).

    Returns
    -------
    h : Python dictionary
        .mot header information: the name (first line of the file) and the
        header fields, e.g., version, nRows, nColumns (including the time
        column) and inDegrees (bool) for the OpenSim format, or datacolumns,
        datarows and range for the SIMM format.
    df : pandas dataframe
        data with the column labels of the file and time as index

    """
    int_keys = {"version", "nRows", "nColumns", "datacolumns", "datarows"}

    with open(file=fname, mode="rt", encoding="utf-8", newline="") as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end="")
        # the first line is the name; the header ends with "endheader"
        name = f.readline().strip()
        h = {"name": name[5:].strip() if name.startswith("name ") else name}
        for line in f:
            line = line.strip()
            if line == "endheader":
                break
            if not line:
                continue
            key, value = line.split("=", 1) if "=" in line else line.split(None, 1)
            key, value = key.strip(), value.strip()
            if key in int_keys:
                value = int(value)
            elif key == "inDegrees":
                value = value.lower() == "yes"
            elif key == "range":
                value = [float(v) for v in value.split()]
            h[key] = value
        # data, with the column labels in the first line after the header
        df = pd.read_csv(f, sep="\t", index_col=0, skipinitialspace=True, engine="c")
        # trailing tabs create empty, unnamed columns
        empty = [
            c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()
        ]
        df = df.drop(columns=empty)
        if show_msg:
            print("done.")

    return h, df


def read_delsys(
    fname,
    fname2="",
    sensors=None,
    freq_trc=150,
    emg=True,
    imu=False,
    resample=[1200, 150],
    freqs=[20, 20, 450],
    show_msg=True,
    show=False,
    ax=None,
    suptitle="",
):
    """Read Delsys csv file from Cortex MAC (Asynchronous device data file).

    Parameters
    ----------
    fname : string
        Full file name of the Delsys csv file from Cortex file to be opened.
    fname2 : string, optional (default = '')
        Full file name of the text file to be saved with data if desired.
        If both parameters `emg` and `imu` are True, you must input a list with
        the two full file names (EMG and IMU).
        If fname2 is '', no file is saved.
        If fname2 is '=', the original file name will be used but its extension
        will be .emg and .imu for the files with EMG data and with IMU data (if
        parameters `emg` and `imu` are True).
    sensors : list of strings, optional
        Names of sensors to be used as column names for the EMG and IM data.
    freq_trc : number, optional (default = 150)
        Sampling frequency of the markers data
    emg : bool, optional (default = True)
        Read and save EMG data
    imu : bool, optional (default = False)
        Read and save IMU data
    resample : list with two numbers, optional (default = [1200, 150])
        Whether to resample the data to have the given frequencies.
        The list order is [freq_emg, freq_imu]. Enter 0 (zero) to not resample.
        It's used signal.resample_poly scipy function.
        For the EMG signal, if the parameter frequency is lower than 1000 Hz,
        first it will be calculated the linear envelope with a low-pass
        frequency given by parameter freqs[0] (but first the EMG data will be
        band-pass filtered with frequencies given by parameters freqs[1], freqs[2].
    freqs : list of three numbers, optional (default = [20, 20, 450])
        Frequencies to be used at the linear envelope calculation if desired.
        See the parameter `resample`.
    show_msg : bool, optional (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    suptitle : string, optional (default = '')
        If string, shows string as suptitle. If empty, doesn't show suptitle.

    Returns
    -------
    data : 1 or 2 pandas dataframe
        df_emg and df_imu if paramters `emg` and `imu`.
        The units of df_emg will be mV (the raw signal is multiplied by 1000).
        The units of the IMU data are according to Delsys specification.
    """
    with open(file=fname, mode="rt", encoding="utf-8", newline=None) as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end="")
        file = f.read().splitlines()
        if file[0] != "Cortex generated Asynchronous device data file (.add)":
            raise ValueError("Not a Cortex asynchronous Delsys export")
        # find start and final lines of data in file
        idx = file.index("[Devices]") + 2
        count = int(file[idx].split("=")[1])
        devices = [name.split(", ")[-1] for name in file[idx + 1 : idx + 1 + count]]
        if sensors is None:
            sensors = devices
        idx = idx + 3 + count
        count2 = int(file[idx].split("=")[1])
        channels = [name for name in file[idx + 1 : idx + 1 + count2]]
        n_im = int((count2 - count) / count)
        # indexes for ini_emg, end_emg, ini_im, end_im
        idxs = np.zeros((count, 4), dtype=int)
        for i, device in enumerate(devices):
            idxs[i, 0] = file.index(device) + 3
            idxs[i, 1] = file[idxs[i, 0] :].index("") + idxs[i, 0] - 1
        idxs[:, 2] = idxs[:, 1] + 3
        idxs[:, 3] = np.r_[idxs[1:, 0] - 6, np.array(len(file) - 3, dtype=int, ndmin=1)]

        # read emg data
        if emg:
            nrows_emg = int(np.min(idxs[:, 1] - idxs[:, 0]) + 1)
            f.seek(0)
            t_emg = pd.read_csv(
                f,
                sep=",",
                header=None,
                names=None,
                index_col=None,
                usecols=[2],
                skiprows=idxs[0, 0],
                nrows=nrows_emg,
                dtype=np.float32,
                encoding="utf-8",
                engine="c",
            ).squeeze("columns").to_numpy()
            # the above is faster than simply:
            # np.array([x.split(',')[2] for x in file[idxs[0, 0]:idxs[0, 1]+1]], dtype=np.float32)
            # and faster than:
            # np.loadtxt(f, dtype=np.float32, comments=None, delimiter=',', skiprows=idxs[0, 0], usecols=2, max_rows=nrows_emg)
            freq_emg = np.mean(freq_trc / np.diff(t_emg))
            if resample[0]:
                fr = Fraction(resample[0] / freq_emg).limit_denominator(1000)
                nrows_emg = int(np.ceil(nrows_emg * fr.numerator / fr.denominator))
                freq_emg2 = resample[0]
            else:
                freq_emg2 = freq_emg
            ys = np.empty((nrows_emg, count), dtype=np.float32)
            for i, sensor in enumerate(sensors):
                f.seek(0)
                y = pd.read_csv(
                    f,
                    sep=",",
                    header=None,
                    names=[sensor],
                    index_col=None,
                    usecols=[3],
                    skiprows=idxs[i, 0],
                    nrows=len(t_emg),
                        dtype=np.float32,
                    encoding="utf-8",
                    engine="c",
                ).squeeze("columns").to_numpy()

                if resample[0]:
                    if resample[0] < 1000:
                        y = linear_envelope(
                            y,
                            freq_emg,
                            fc_bp=[freqs[1], freqs[2]],
                            fc_lp=freqs[0],
                            method="rms",
                        )
                    y = signal.resample_poly(y, fr.numerator, fr.denominator)
                ys[:, i] = y * 1000
            df_emg = pd.DataFrame(data=ys, columns=sensors)
            df_emg.index = float(t_emg[0]) / freq_trc + df_emg.index / freq_emg2
            df_emg.index.name = "Time"

        # read IM data
        if imu:
            nrows_imu = int(np.min(idxs[:, 3] - idxs[:, 2]) + 1)
            cols = [
                sensor + channel.split(",")[3]
                for sensor in sensors
                for channel in channels[1 : int(count2 / count)]
            ]
            f.seek(0)
            t_imu = pd.read_csv(
                f,
                sep=",",
                header=None,
                names=None,
                index_col=None,
                usecols=[2],
                skiprows=idxs[0, 2],
                nrows=nrows_imu,
                dtype=np.float32,
                encoding="utf-8",
                engine="c",
            ).squeeze("columns").to_numpy()
            freq_imu = np.mean(freq_trc / np.diff(t_imu))
            if resample[1]:
                fr = Fraction(resample[1] / freq_imu).limit_denominator(1000)
                nrows_imu = int(np.ceil(nrows_imu * fr.numerator / fr.denominator))
                freq_imu = resample[1]
            ys = np.empty((nrows_imu, count2 - count), dtype=np.float32)
            for i, sensor in enumerate(sensors):
                f.seek(0)
                cs = slice(int(n_im * i), int(n_im * (i + 1)))
                y = pd.read_csv(
                    f,
                    sep=",",
                    header=None,
                    names=cols[cs],
                    index_col=None,
                    usecols=range(3, 3 + n_im),
                    skiprows=idxs[i, 2],
                    nrows=len(t_imu),
                    dtype=np.float32,
                    encoding="utf-8",
                    engine="c",
                ).squeeze("columns").to_numpy()
                if resample[1]:
                    y2 = np.empty((nrows_imu, y.shape[1]), dtype=np.float32)
                    for c in range(y.shape[1]):
                        y2[:, c] = signal.resample_poly(
                            y[:, c], fr.numerator, fr.denominator
                        )
                else:
                    y2 = y
                ys[:, cs] = y2
            df_imu = pd.DataFrame(data=ys, columns=cols)
            df_imu.index = float(t_imu[0]) / freq_trc + df_imu.index / freq_imu
            df_imu.index.name = "Time"

        if show_msg:
            print("done.")

    # save file
    if len(fname2):
        if isinstance(fname2, list):
            fname2_emg = fname2[0]
            fname2_imu = fname2[1]
        else:
            if emg:
                fname2_emg = fname2
            if imu:
                fname2_imu = fname2
        if emg and fname2_emg == "=":
            name, extension = os.path.splitext(fname)
            fname2_emg = name + ".emg"
        if imu and fname2_imu == "=":
            name, extension = os.path.splitext(fname)
            fname2_imu = name + ".imu"
        if emg:
            df_emg.to_csv(fname2_emg, sep="\t", float_format="%.6f")
            if show_msg:
                print(f'Saving file "{fname2_emg}" ... ', end="")
        if imu:
            df_imu.to_csv(fname2_imu, sep="\t", float_format="%.6f")
            if show_msg:
                print(f'\nSaving file "{fname2_imu}" ... ', end="")
        if show_msg:
            print("done.")

    if show and emg:
        _plot_df_emg(df_emg, ax=None, suptitle=suptitle)

    if emg and imu:
        return df_emg, df_imu
    elif emg:
        return df_emg
    elif imu:
        return df_imu


def _plot_df_emg(df, ax, suptitle):
    """Plot EMG data of the read_delsys function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available.")
    else:
        if ax is None:
            nplots = df.shape[1]
            if nplots <= 3:
                nrows, ncols = nplots, 1
            elif nplots == 4:
                nrows, ncols = 2, 2
            else:
                ncols = 3
                nrows = int(np.ceil(nplots / ncols))

            _, ax = plt.subplots(
                nrows, ncols, figsize=(9, 6), sharex="all", constrained_layout=True
            )
            no_ax = True
        else:
            no_ax = False

        ax = df.plot(color="b", ax=ax, subplots=True)

        if suptitle:
            plt.suptitle(suptitle)
        # plt.grid()
        if no_ax:
            plt.show()


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
    with open(file=fname, mode="wt", encoding="utf-8", newline="") as f:
        if show_msg:
            print(f'Saving file "{fname}" ... ', end="")
        f.writelines("\t".join(line) + "\n" for line in header)

        # f.write('\n')  # blank line already included in the header
        # Every data line ends with a tab. Recent versions of Python's csv
        # module reject a line terminator that contains the delimiter ("\t\n"),
        # so the tab is added after writing the (purely numeric) data.
        data = df.to_csv(
            header=None, index=None, sep="\t", lineterminator="\n", float_format="%.6f"
        )
        f.write(data.replace("\n", "\t\n"))
        if show_msg:
            print("done.")


def write_forces(fname, header, df, scale=1, show_msg=True):
    """Write .forces file format from Cortex MAC.

    See the read_forces.py function.

    Parameters
    ----------
    fname : string
        Full file name of the .forces file to be saved.
    header : list of lists
        header for the .forces file
    df : pandas dataframe
        dataframe with data for the .forces file (with frame and time columns)
    scale : number (default = 1)
        number to multiply COP data and convert its units.
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).
    """
    with open(file=fname, mode="wt", encoding="utf-8", newline="") as f:
        if show_msg:
            print(f'Saving file "{fname}" ... ', end="")

        forcepla = list(set([int(fp[-1]) for fp in df.columns.values]))
        cops = [cop + str(fp) for fp in forcepla for cop in ["X", "Y", "Z"]]

        if scale != 1:
            df[cops] = df[cops].values * scale

        h = list(header.keys())
        f.write(f"{header[h[0]]}\n")
        f.writelines(f"{key}={header[key]}\n" for key in h[1:-1])

        df.reset_index(drop=True, inplace=True)
        df.index = df.index.values + 1
        df.index.name = "#Sample"
        df.to_csv(
            f,
            header=header[h[-1]],
            index=True,
            sep="\t",
            lineterminator="\n",
            float_format="%.6f",
        )
        if show_msg:
            print("done.")


def write_mot(fname, df, show_msg=True, in_degrees=False):
    """Write .mot file in the OpenSim format.

    The header has the name of the file, the format version, the numbers of
    rows and columns (including the time column), whether angles are in
    degrees, and "endheader"; the column labels follow, starting with time.
    See the read_mot function.

    Parameters
    ----------
    fname : string
        Full file name of the .mot file to be saved.
    df : pandas dataframe
        data for the .mot file, with time as index
    show_msg : bool (default = True)
        Whether to print messages about the execution of the intermediary steps
        (True) or not (False).
    in_degrees : bool (default = False)
        Whether angles in the data are in degrees (True) or radians (False);
        written to the header as inDegrees=yes or no.
    """
    with open(file=fname, mode="wt", encoding="utf-8", newline="") as f:
        if show_msg:
            print(f'Saving file "{fname}" ... ', end="")

        f.write(f"{os.path.basename(fname)}\n")
        f.write("version=1\n")
        f.write(f"nRows={df.shape[0]}\n")
        f.write(f"nColumns={df.shape[1] + 1}\n")
        f.write(f"inDegrees={'yes' if in_degrees else 'no'}\n")
        f.write("endheader\n")

        df.to_csv(
            f,
            header=df.columns.values,
            index=True,
            index_label="time",
            sep="\t",
            lineterminator="\n",
            float_format="%.6f",
        )
        if show_msg:
            print("done.")


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
    values = np.asarray(data, dtype=float)
    origin = np.asarray(O, dtype=float)
    force = values[:, :3]
    lever = values[:, 3:6] - origin
    moment = np.cross(lever, force)
    moment[:, 2] += values[:, 6]
    return np.column_stack((force, moment))


def step_id(df_f, df_t, forcepla=[2], R="RCAL", L="LCAL", show_msg=True):
    """Identification of step side based on .forces and .trc files."""
    if show_msg:
        print("Step identification ... ", end="")
    if not isinstance(forcepla, list):
        forcepla = [forcepla]
    if not forcepla:
        forcepla = list(set([int(fp[-1]) for fp in df_f.columns.values]))
    forces = [force + str(fp) for fp in forcepla for force in ["FX", "FY", "FZ", "MZ"]]
    time = df_f.index.values
    freq = np.round(np.mean(1 / np.diff(time)), 0)
    # resample trc data to the forces frequency
    R = np.interp(time, df_t.index.values, df_t[R].Y.values[:, 0])
    L = np.interp(time, df_t.index.values, df_t[L].Y.values[:, 0])
    # detect onsets in Fy data
    threshold = 50
    n_above = int(0.1 * freq)
    n_below = int(0.01 * freq)
    threshold2 = 10 * threshold  # N
    n_above2 = int(0.02 * freq)
    idx = detect_onset(
        df_f[forces[1]].values, threshold, n_above, n_below, threshold2, n_above2
    )
    # column names of the .mot dataframe
    cols = [
        "R_ground_force_vx",
        "R_ground_force_vy",
        "R_ground_force_vz",
        "R_ground_force_px",
        "R_ground_force_py",
        "R_ground_force_pz",
        "L_ground_force_vx",
        "L_ground_force_vy",
        "L_ground_force_vz",
        "L_ground_force_px",
        "L_ground_force_py",
        "L_ground_force_pz",
        "R_ground_torque_x",
        "R_ground_torque_y",
        "R_ground_torque_z",
        "L_ground_torque_x",
        "L_ground_torque_y",
        "L_ground_torque_z",
    ]
    data = np.zeros((df_f.shape[0], len(cols)))
    # step side identification
    for ini, end in idx:
        if R[ini : int((ini + end) / 2)].sum() < L[ini : int((ini + end) / 2)].sum():
            data[ini : end + 1, [0, 1, 2, 3, 4, 5, 14]] = df_f.values[ini : end + 1, :]
        else:
            data[ini : end + 1, [6, 7, 8, 9, 10, 11, 17]] = df_f.values[
                ini : end + 1, :
            ]

    df = pd.DataFrame(data=data, columns=cols, index=time)
    df.index.name = "time"
    if show_msg:
        print("done.")

    return df


def filter_forces(
    df, h, forcepla=[2], fc_forces=20, fc_cop=6, threshold=50, show_msg=True
):
    """Filter force data from the treadmill."""
    if not isinstance(forcepla, list):
        forcepla = [forcepla]
    if not forcepla:
        forcepla = list(set([int(fp[-1]) for fp in df.columns.values]))
    forces = [force + str(fp) for fp in forcepla for force in ["FX", "FY", "FZ", "MZ"]]
    cops = [cop + str(fp) for fp in forcepla for cop in ["X", "Z"]]

    df2 = df.copy()
    F = df2[forces[1]].values
    freq = h["SampleRate"]
    npad = int(freq / 4)
    npad2 = 2

    # filter parameters for COP
    b_cd, a_cd, fc_cd = critic_damp(
        fcut=fc_cop, freq=freq, npass=2, fcorr=True, filt="critic"
    )
    if show_msg:
        print(f"Filtering: COP Fc: {fc_cd:.2f} Hz", end=", ")

    n_above = int(0.1 * freq)
    n_below = int(0.01 * freq)
    threshold2 = 10 * threshold  # N
    n_above2 = int(0.02 * freq)

    # detect onsets in Fy data
    idx1 = detect_onset(F, threshold, n_above, n_below, threshold2, n_above2)
    for cop in cops:
        COP = df2[cop].values
        # for each foot strike
        for ini, end in idx1:
            # reliable COP portion
            idx2 = detect_onset(
                F[ini : end + 1],
                4 * threshold,
                n_above,
                n_below,
                None,
                1,
                del_ini_end=False,
            )
            if idx2.shape[0]:
                # fit polynomiun
                y = COP[ini + idx2[0, 0] : ini + idx2[0, 1] + 1]
                t = ini + idx2[0, 0] + np.linspace(0, y.shape[0] - 1, y.shape[0])
                p = np.polyfit(t, y, 2)
                # values at the extremities for using to pad data
                z = np.polyval(p, [ini, end])
                q = np.hstack(
                    (z[0] * np.ones(npad), COP[ini : end + 1], z[1] * np.ones(npad))
                )
                # filter data
                q2 = signal.filtfilt(b_cd, a_cd, q)

                COP[ini - npad2 : end + 1 + npad2] = q2[npad - npad2 : -npad + npad2]

        df2[cop] = COP

    b_cd, a_cd, fc_cd = critic_damp(
        fcut=fc_forces, freq=freq, npass=2, fcorr=True, filt="critic"
    )
    if show_msg:
        print(f"Forces Fc: {fc_cd:.2f} Hz")
    for force in forces:
        df2[force] = signal.filtfilt(b_cd, a_cd, df2[force])

    return df2


def to_mot(fname_f, fname_t, forcepla=[2], R="RCAL", L="LCAL", show_msg=True):
    """Generate .mot file from .forces and .trc files"""
    if not isinstance(forcepla, list):
        forcepla = [forcepla]
    # read .forces and .trc files
    h_f, df_f = read_forces(fname_f, time=True, forcepla=forcepla, show_msg=show_msg)
    h_t, df_t = read_trc(
        fname_t, fname2="", dropna=False, na=0.0, df_multi=True, show_msg=show_msg
    )
    # filter .forces data
    df2_f = filter_forces(df_f, h_f, show_msg=show_msg)
    # save filtered .forces file
    # fname2_f = fname_f.split('.forces')[0] + '_2' + '.forces'
    # write_forces(fname2_f, h_f, df2_f, show_msg=show_msg)
    # generate .mot dataframe
    df_m = step_id(df2_f, df_t, forcepla=forcepla, R=R, L=L, show_msg=show_msg)
    # save .mot file
    fname = fname_f.split(".forces")[0] + ".mot"
    write_mot(fname, df_m, show_msg=show_msg)


def polyfit2d(x, y, z, order=[1, 1], show_msg=True):
    """Fit 2d polynomial of order order to the x, y, z data."""
    A = np.polynomial.polynomial.polyvander2d(x, y, order)
    coeff, r, rank, s = np.linalg.lstsq(A, z)
    if show_msg:
        print("Residuals: ", r)
        print("Rank: ", rank)
        print("Chi2: ", r / (x.shape[0] - rank))

    return coeff, r, rank, s


def polyval2d(coeff, x, y, order=[], grid=True, N=[50, 50]):
    """Evaluate 2d polynomial with coefficients coeff of order order at x, y."""
    if not len(order):
        order = 2 * [int(np.sqrt(coeff.shape[0]) - 1)]

    A = np.polynomial.polynomial.polyvander2d(x, y, order)
    z_fit = A @ coeff

    if grid:
        coeff = np.atleast_2d(coeff)
        x2 = np.linspace(x.min(), x.max(), N[0])
        y2 = np.linspace(y.min(), y.max(), N[1])
        x_grid, y_grid = np.meshgrid(x2, y2, indexing="xy")
        z_grid = np.empty((coeff.shape[1], N[1], N[0]))
        for col in range(coeff.shape[1]):
            V = coeff[:, col].reshape(order[0] + 1, order[1] + 1)
            z_grid[col] = np.polynomial.polynomial.polygrid2d(x2, y2, V).T
        if coeff.shape[1] == 1:
            z_grid = z_grid[0]
        return z_fit, x_grid, y_grid, z_grid
    else:
        return z_fit


def fpcal(fname_cal, fnames, forcepla=2):
    """Calibrate data in files fnames using the coefficients in file fname_cal."""
    ch = [ch + str(forcepla) for ch in ["X", "Z", "FY", "FX", "FZ"]]
    # calibration data
    c = np.load(fname_cal)
    print("Data calibration ...")
    for i, fname in enumerate(fnames):
        print(i, end=" ")
        # load file to be calibrated
        h, df = read_forces(fname, forcepla=[2], mm2m=False, show_msg=False)
        # data calibration
        z_fit = polyval2d(
            c["coeff"],
            df[ch[0]].values / 1000,
            df[ch[1]].values / 1000,
            order=c["order"],
            grid=False,
        )
        df[ch[0]] = (
            df[ch[0]].values + z_fit[:, 0]
        )  # COP correction of X is in milimeters
        df[ch[1]] = (
            df[ch[1]].values + z_fit[:, 1]
        )  # COP correction of Z is in milimeters
        df[ch[2]] = df[ch[2]].values * z_fit[:, 2]  # Fy calibration
        df[ch[3]] = (
            df[ch[3]].values * z_fit[:, 2]
        )  # apply correction of Fy to Fx and Fz
        df[ch[4]] = (
            df[ch[4]].values * z_fit[:, 2]
        )  # apply correction of Fy to Fx and Fz
        # save calibrated data
        write_forces(fname, h, df, show_msg=False)
    print("\nDone.")
