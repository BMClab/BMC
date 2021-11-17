"""Read and write Cortex Motion Analysis Corporation ASCII related files.

    Read .trc file:
      read_trc(fname, fname2, units, dropna, na, fmt, show_msg)
    Write .trc file
      write_trc(fname, header, df, show_msg)

"""

__author__ = "Marcos Duarte, https://github.com/demotu/"
__version__ = "0.0.2"
__license__ = "MIT"

import os
import csv
import numpy as np
import pandas as pd


def read_trc(fname, fname2='', units='', dropna=False, na=0.0, fmt='multi',
             show_msg=True):
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
        True: Delete column if it has only missing or NaN values.
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

    with open(file=fname, mode='rt', encoding='utf-8', newline='') as f:
        if show_msg:
            print('Opening file "{}" ... '.format(fname), end='')
        # get header information
        read = csv.reader(f, delimiter='\t')
        header = [next(read) for x in range(5)]
        # actual number of markers
        nmarkers = int((len(header[3])-2)/3)
        # column labels
        markers = np.asarray(header[3])[np.arange(2, 2+3*nmarkers, 3)].tolist()
        # find marker duplicates if they exist and add suffix to the duplicate markers:
        if len(markers) != len(set(markers)):
            from collections import Counter
            d = {a:[''] + list(range(2, b+1)) if b > 1 else '' for a, b in Counter(markers).items()}
            markers = [i + str(d[i].pop(0)) if len(d[i]) else i for i in markers]
        markers3 = [m for m in markers for i in range(3)]
        # XYZ order
        XYZ = [i[0] for i in header[4][2:5]]
        xyz = [i[0].lower() for i in header[4][2:5]]
        #XYZ = header[0][2].strip('()').split('/')
        #xyz = header[0][2].strip('()').lower().split('/')
        markersxyz = [a+b for a, b in zip(markers3, xyz*nmarkers)]
        # read data
        # 6th line of the file is usually blank; pd.read_csv handles that.
        df = pd.read_csv(f, sep='\t', names=['Frame#', 'Time'] + markersxyz,
                         index_col=False, encoding='utf-8', engine='c',
                         skip_blank_lines=True)
        # drop markers with no data (column has NaN only)
        if dropna:
            df.dropna(axis=1, how='all', inplace=True)
        elif na is not None:
            for col in df:
                if df.loc[:, col].isnull().sum() == df.shape[0]:
                    df.loc[:, col] = na
        # update header
        nmarkers = int((df.shape[1]-2)/3)
        if header[2][3] != str(nmarkers):
            if show_msg:
                print(' Number of markers changed from {} to {}.'
                      .format(header[2][3], nmarkers))
            header[2][3] = str(nmarkers)
        header[3] = ['' if c[-1] in ['y', 'z'] else c[:-1] if c[-1] in ['x']
                     else c for c in df.columns.values.tolist()] + ['']
        markers = np.asarray(header[3])[np.arange(2, 2+3*nmarkers, 3)].tolist()
        n3 = np.repeat(range(1, nmarkers+1), 3).tolist()
        XYZs = [a+str(b) for a, b in zip(XYZ*nmarkers, n3)]
        header[4] = ['', ''] + XYZs
        if units == 'm':
            if header[2][4] == 'mm':
                df.iloc[:, 2:] = df.iloc[:, 2:]/1000
                header[2][4] = 'm'
                if show_msg:
                    print(' Units changed from {} to {}'.format('"mm"', '"m"'))
        elif units == 'mm':
            if header[2][4] == 'm':
                df.iloc[:, 2:] = df.iloc[:, 2:]*1000
                header[2][4] = 'mm'
                if show_msg:
                    print(' Units changed from {} to {}'.format('"m"', '"mm"'))

        if show_msg:
            print('done.')

    # save file
    if fname2:
        if fname2 == '=':
            fname2 = fname
        elif len(fname2) <= 3:
            name, extension = os.path.splitext(fname)
            fname2 = name + fname2 + extension

        write_trc(fname2, header, df, show_msg)

    # outputs
    h = {'header': header,
         'data_rate': float(header[2][0]),
         'camera_rate': float(header[2][1]),
         'nframes': int(header[2][2]),
         'nmarkers': int(header[2][3]),
         'markers': markers,
         'xyz': XYZs,
         'units': header[2][4],
         'fname': fname,
         'fname2': fname2}
    if fmt.lower() == 'uni':  # dataframe with uni labels
        return h, df
    elif fmt.lower() == 'multi':  # dataframe with multiple labels
        df.drop(labels='Frame#', axis=1, inplace=True)
        df.set_index('Time', inplace=True)
        df.index.name = 'Time'
        cols = [s[:-1] for s in df.columns.str.replace(r'.', r'_', regex=True)]
        df.columns = [cols, XYZ*int(df.shape[1]/3), XYZs]
        df.columns.set_names(names=['Marker', 'Coordinate', 'XYZ'],
                             level=[0, 1, 2], inplace=True)
        return h, df
    else:
        import xarray as xr

        name = 'Position data'
        dims = ['time', 'marker', 'component']
        time = df.values[:, 1]
        component = XYZ
        coords = [time, markers, component]  # pages, rows, columns
        #h.pop('header')
        da = xr.DataArray(data=df.values[:, 2:].reshape(-1, nmarkers, 3, order='C'),
                          dims=dims,
                          coords=coords,
                          name=name,
                          attrs=h)
        # xarray uses the coordinate name along with metadata attrs.long_name,
        #  attrs.standard_name, DataArray.name, and attrs.units to label axes.
        da['time'].attrs['units'] = 's'
        da['component'].attrs['units'] = h['units']
        return da


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

    with open(file=fname, mode='wt', encoding='utf-8', newline='') as f:
        if show_msg:
            print('Saving file "{}" ... '.format(fname), end='')
        for line in header:
            f.write('\t'.join(line) + '\n')
        f.write('\n')  # blank line

        df.to_csv(f, header=None, index=None, sep='\t',
                  line_terminator='\t\n', float_format='%.6f')
        if show_msg:
            print('done.')
