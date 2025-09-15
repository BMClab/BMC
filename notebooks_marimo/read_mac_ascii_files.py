import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Read and write Cortex Motion Analysis Corporation ASCII files

        > Marcos Duarte  
        > Laboratory of Biomechanics and Motor Control ([http://demotu.org/](http://demotu.org/))  
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
    import sys
    _sys.path.insert(1, './../functions')
    import io_cortexmac as io
    return (io,)


@app.cell
def _(io):
    print(io.__doc__)
    return


@app.cell
def _():
    import sys, os
    path2 = './../data/'
    fname = _os.path.join(path2, 'arm26_elbow_flex.trc')
    return (fname,)


@app.cell
def _(fname, io):
    h, df = io.read_trc(fname, fname2='_2', units='', df_multi=True)
    return (df,)


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _():
    """Read and write Cortex Motion Analysis Corporation ASCII related files.

        read_trc(fname, fname2='_2', units='', df_multi=True): Read .trc file.
        read_anc(fname): Read .anc file.
        read_cal(fname): Read .cal file.
        read_forces(fname): Read .forces file.
        write_trc(fname, header, df): Write .trc file.
        write_v3dtxt(fname, trc, forces, freq=0): Write Visual3d text file
         from .trc and .forces files or dataframes.
        grf_moments(data, O): Calculate force plate moments around its origin
         given 3 forces, 2 COPs, 1 free moment, and its geometric position.
    """
    __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version__ = '1.0.1'
    __license__ = 'MIT'
    import os
    import csv
    import numpy as np
    import pandas as pd

    def read_trc(fname, fname2='_2', units='', df_multi=True):
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
        fname  : string
            Full file name of the .trc file to be opened.

        fname2  : string (default = '_2')
            Full file name of the .trc file to be saved with updated information
            and data if desired.
            If fname2 is '', no file is saved.
            If fname2 is '=', the original file name will be used.
            If fname2 is a string with length between 1 and 3, this string (other
            than '=') is appended to the original file name.

        units  : string (default = '')
            Change the units of the data if desired.
            Accepted output units are 'm' or 'mm'.

        df_multi  : bool (default = True)
            Whether to output data as pandas multiindex dataframe with "Marker"
            and "Coordinate" as labels and "Time" as index (True) or simple
            pandas dataframe with "Frame#" and "Time" as columns (False).

        Returns
        -------
        h  : Python dictionary with .trc header information
            keys: header (the .trc full header), data_rate (Hz), camera_rate (Hz),
            nframes, nmarkers, markers (names), xyz (X1,Y1,Z1...), units.

        data  : pandas dataframe
            Two possible output formats according to the `df_multi` option:
            Dataframe with shape (nframes, 2+3*nmarkers) with markerxyz as label
            and columns: Frame#, time and position data.
            Dataframe with shape (nframes, 3*nmarkers) with "Marker" and
            "Coordinate" as labels, "Time" as index, and data position as columns.

        """
        with open(file=fname, mode='rt', encoding='utf-8', newline='') as f:
            print('Opening file "{}"'.format(fname))
            read = csv.reader(f, delimiter='\t')
            header = [next(read) for x in range(5)]
            nmarkers = int((len(header[3]) - 2) / 3)
            markers = np.asarray(header[3])[np.arange(2, 2 + 3 * nmarkers, 3)].tolist()
            markers3 = [m for m in markers for i in range(3)]
            markersxyz = [a + b for (a, b) in zip(markers3, ['x', 'y', 'z'] * nmarkers)]
            df = pd.read_csv(f, sep='\t', names=['Frame#', 'Time'] + markersxyz, index_col=False, encoding='utf-8', engine='c')
            df.dropna(axis=1, how='all', inplace=True)
            nmarkers = int((df.shape[1] - 2) / 3)
            if header[2][3] != str(nmarkers):
                print(' Number of markers changed from {} to {}.'.format(header[2][3], nmarkers))
                header[2][3] = str(nmarkers)
            header[3] = ['' if c[-1] in ['y', 'z'] else c[:-1] if c[-1] in ['x'] else c for c in df.columns.values.tolist()] + ['']
            markers = np.asarray(header[3])[np.arange(2, 2 + 3 * nmarkers, 3)].tolist()
            n3 = np.repeat(range(1, nmarkers + 1), 3).tolist()
            xyz = [a + str(b) for (a, b) in zip(['X', 'Y', 'Z'] * nmarkers, n3)]
            header[4] = ['', ''] + xyz
            if units == 'm':
                if header[2][4] == 'mm':
                    df.iloc[:, 2:] = df.iloc[:, 2:] / 1000
                    header[2][4] = 'm'
                    print(' Units changed from {} to {}'.format('"mm"', '"m"'))
            elif units == 'mm':
                if header[2][4] == 'm':
                    df.iloc[:, 2:] = df.iloc[:, 2:] * 1000
                    header[2][4] = 'mm'
                    print(' Units changed from {} to {}'.format('"m"', '"mm"'))
        if len(fname2):
            if fname2 == '=':
                fname2 = fname
            elif len(fname2) <= 3:
                (name, extension) = _os.path.splitext(fname)
                fname2 = name + fname2 + extension
            write_trc(fname2, header, df)
        h = {'header': header, 'data_rate': float(header[2][0]), 'camera_rate': float(header[2][1]), 'nframes': int(header[2][2]), 'nmarkers': int(header[2][3]), 'markers': markers, 'xyz': xyz, 'units': header[2][4], 'fname': fname, 'fname2': fname2}
        if df_multi:
            df.drop(labels='Frame#', axis=1, inplace=True)
            df.set_index('Time', inplace=True)
            df.index.name = 'Time'
            cols = [s[:-1] for s in df.columns.str.replace('.', '')]
            df.columns = [cols, list('XYZ') * int(df.shape[1] / 3)]
            df.columns.set_names(names=['Marker', 'Coordinate'], level=[0, 1], inplace=True)
        return (h, df)

    def read_anc(fname):
        """Read .anc file format from Cortex MAC.

        The .anc (Analog ASCII Row Column) file contain ASCII analog data
        in row-column format. The data is derived from *.anb analog binary
        files. These binary *.anb files are generated simultaneously with
        video *.vc files if an optional analog input board is used in
        conjunction with video data capture.

        Parameters
        ----------
        fname  : string
            full file name of the .anc file to be opened

        Returns
        -------
        h  : Python dictionary
            .anc header information
            keys: nbits, polarity, nchannels, data_rate, ch_names, ch_ranges

        data  : pandas dataframe
            analog data with shape (nframes, nchannels)

        """
        with open(file=fname, mode='rt', encoding='utf-8', newline='') as f:
            read = csv.reader(f, delimiter='\t')
            header = [next(read) for x in range(11)]
            h = {'nbits': int(header[3][1]), 'polarity': header[1][3], 'nchannels': int(header[2][7]), 'data_rate': float(header[3][3]), 'ch_names': header[8], 'ch_ranges': header[10]}
            h['ch_names'] = h['ch_names'][1:-1]
            h['ch_ranges'] = np.asarray(h['ch_ranges'][1:-1], dtype=np.float)
            data = pd.read_csv(f, sep='\t', names=h['ch_names'], engine='c', usecols=np.arange(1, 1 + h['nchannels']))
            data = data * (h['ch_ranges'] / (2 ** h['nbits'] / 2 - 2))
        return (h, data)

    def read_cal(fname):
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
        fname  : string
            full file name of the .trc file to be opened

        Returns
        -------
        forcepla  : Python dictionary
            parameter from the froce plate calibration file
            keys: 'fp', 'scale', 'size', 'cal_matrix', 'origin', 'center', 'orientation'
        """
        (fp, scale, size, cal_matrix, origin, center, orientation) = ([], [], [], [], [], [], [])
        with open(file=fname, mode='rt', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                fp.append(int(row[0][0]))
                n = 8 if row[0][-1] == 'K' else 6
                scale_size = np.array(next(reader)).astype(np.float)
                scale.append(scale_size[0])
                size.append(scale_size[1:])
                matrix = [next(reader) for x in range(n)]
                cal_matrix.append(np.array(matrix).astype(np.float))
                origin.append(np.array(next(reader)).astype(np.float))
                center.append(np.array(next(reader)).astype(np.float))
                orienta = [next(reader) for x in range(3)]
                orientation.append(np.array(orienta).astype(np.float))
        forcepla = {'fp': fp, 'scale': scale, 'size': size, 'cal_matrix': cal_matrix, 'origin': origin, 'center': center, 'orientation': orientation}
        return forcepla

    def read_forces(fname):
        """Read .forces file format from Cortex MAC.

        The .forces file in ASCII contains force plate data. The data is saved
        based on the forcepla.cal file of the trial and converts the raw force
        plate data into calibrated forces. The units used are Newtons and
        Newton-meters and each line in the file equates to one analog sample.

        Parameters
        ----------
        fname  : string
            full file name of the .forces file to be opened

        Returns
        -------
        h  : Python dictionary
            .forces header information
            keys: name, nforceplates, data_rate, nsamples, ch_names

        data  : pandas dataframe
            force plate data with shape (nsamples, 7*nforceplates)

        """
        with open(file=fname, mode='rt', encoding='utf-8', newline='') as f:
            read = csv.reader(f, delimiter='\t')
            header = [next(read) for x in range(5)]
            h = {'name': header[0][0], 'nforceplates': int(header[1][0].split('=')[1]), 'data_rate': float(header[2][0].split('=')[1]), 'nsamples': int(header[3][0].split('=')[1]), 'ch_names': header[4][1:]}
            data = pd.read_csv(f, sep='\t', names=h['ch_names'], index_col=False, usecols=np.arange(1, 1 + 7 * h['nforceplates']), engine='c')
        return (h, data)

    def write_trc(fname, header, df):
        """Write .trc file format from Cortex MAC.

        See the read_trc.py function.

        Parameters
        ----------
        fname  : string
            Full file name of the .trc file to be saved.

        header  : list of lists
            header for the .trc file

        df  : pandas dataframe
            dataframe with data for the .trc file (with frame and time columns)

        """
        with open(file=fname, mode='wt', encoding='utf-8', newline='') as f:
            print('Saving file "{}"'.format(fname))
            for line in header:
                f.write('\t'.join(line) + '\n')
            f.write('\n')
            df.to_csv(f, header=None, index=None, sep='\t', line_terminator='\t\n')

    def write_v3dtxt(fname, trc, forces, freq=0):
        """Write Visual3d text file from .trc and .forces files or dataframes.

        The .trc and .forces data are assumed to correspond to the same time
        interval. If the data have different number of samples (different
        frequencies), the data will be resampled to the highest frequency (or to
        the inputed frequency if it is higher than the former two) using the tnorm
        function.

        Parameters
        ----------
        fname  : string
            Full file name of the Visual3d text file to be saved.

        trc  : pandas dataframe or string
            If string, it is a full file name of the .trc file to read.
            If dataframe, data of the .trc file has shape (nsamples, 2 + 3*nmarkers)
            where the first two columns are from the Frame and Time values.

        forces  : pandas dataframe or string
            If string, it is a full file name of the .forces file to read.
            If dataframe, data of the .forces file has shape (nsamples, 7*nforceplates)

        freq  : float (optional, dafault=0)
            Sampling frequency in Hz to resample data if desired.
            Data will be resampled to the highest frequency between freq, trc, forces.

        """
        if isinstance(trc, str):
            (_, trc) = read_trc(trc, fname2='', units='', df_multi=False)
        if isinstance(forces, str):
            (_, forces) = read_forces(forces)
        if trc.shape[0] != forces.shape[0] or freq:
            from tnorm import tnorm
            freq_trc = 1 / np.nanmean(np.diff(trc.iloc[:, 1].values))
            freq_forces = freq_trc * (forces.shape[0] / trc.shape[0])
            freq = np.max([freq, freq_trc, freq_forces])
            nsample = np.max([trc.shape[0], forces.shape[0]]) * freq / np.max([freq_trc, freq_forces])
            (trc2, _, _) = tnorm(trc.iloc[:, 2:].values, step=-nsample)
            trc2 = np.hstack((np.vstack((np.arange(1, nsample + 1, 1), np.arange(0, nsample, 1) / freq)).T, trc2))
            trc = pd.DataFrame(trc2, index=None, columns=trc.columns)
            (forces2, _, _) = tnorm(forces.values, step=-nsample)
            forces = pd.DataFrame(forces2, index=None, columns=forces.columns)
        ntrc = trc.shape[1]
        nforces = forces.shape[1]
        data = pd.concat([trc, forces], axis=1)
        with open(file=fname, mode='wt', encoding='utf-8', newline='') as f:
            rows = [[''] + ['default'] * (ntrc + nforces - 1), [''] + data.columns.tolist()[1:], [''] + ['FRAME_NUMBERS'] + ['TARGET'] * (ntrc - 2) + ['ANALOG'] * nforces, [''] + ['ORIGINAL'] * (ntrc + nforces - 1), [data.columns[0]] + ['0'] + ['X', 'Y', 'Z'] * int((ntrc - 2) / 3) + ['0'] * nforces]
            write = csv.writer(f, delimiter='\t')
            write.writerows(rows)
            write.writerows(data.values)

    def grf_moments(data, O):
        """Calculate force plate moments around its origin given
        3 forces, 2 COPs, 1 free moment, and its geometric position.

        Parameters
        ----------
        data  : Numpy array (n, 7)
            array with [Fx, Fy, Fz, COPx, COPy, COPz, Tz].
        O  : Numpy array-like or list
            origin [x,y,z] of the force plate in the motion capture coordinate system [in meters].

        Returns
        -------
        grf  : Numpy array (n, 8)
            array with [Fx, Fy, Fz, Mx, My, Mz]
        """
        (Fx, Fy, Fz, COPx, COPy, COPz, Tz) = np.hsplit(data, 7)
        COPz = np.nanmean(COPz)
        Mx = COPy * Fz + COPz * Fy
        My = -COPx * Fz - COPz * Fx
        Mz = Tz + COPx * Fy - COPy * Fx
        Mx = Mx - Fy * O[2] + Fz * O[1]
        My = My - Fz * O[0] + Fx * O[2]
        Mz = Mz - Fx * O[1] + Fy * O[0]
        grf = np.hstack((Fx, Fy, Fz, Mx, My, Mz))
        return grf
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
