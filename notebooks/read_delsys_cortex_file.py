# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read Delsys data exported by Cortex

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    An EMG sensor and its inertial channels have different sampling rates. How do we load both without confusing their time axes or units?

    We will read a Cortex asynchronous export, inspect each sample rate, and compare raw EMG with a low-rate amplitude envelope.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The unavailable private recording is replaced by a clearly labelled synthetic format example. You can optionally upload a compatible Cortex export; uploaded recordings are never sent to a third-party service by this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The parser and envelope helpers are included here. NumPy, pandas, SciPy, and Matplotlib supply their dependencies. No device SDK or native binary is needed.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import tempfile

    return Path, mo, np, plt, tempfile


@app.function
def delsys_demo_text():
    """Return one second of synthetic EMG (1 kHz) and IMU channels (100 Hz)."""
    import math
    lines = ['Cortex generated Asynchronous device data file (.add)', '[Devices]', '', 'Count=1', '0, Demonstration sensor', '', '[Channels]', 'Count=10', '0,0,0,EMG', '0,0,0,AccX', '0,0,0,AccY', '0,0,0,AccZ', '0,0,0,GyroX', '0,0,0,GyroY', '0,0,0,GyroZ', '0,0,0,MagX', '0,0,0,MagY', '0,0,0,MagZ', '', 'Demonstration sensor', 'EMG', 'Sample,SubFrame,CortexTime,EMG']
    for sample in range(1000):
        t = sample / 1000
        emg = 0.001 * math.sin(2 * math.pi * 80 * t) * (0.3 + 0.7 * math.sin(math.pi * t) ** 2)
        lines.append(f'{sample},0,{150 * t:.8f},{emg:.10f}')
    lines.extend(['', 'Sample,SubFrame,CortexTime,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,MagX,MagY,MagZ'])
    for sample in range(100):
        t = sample / 100
        channels = [math.sin(2 * math.pi * t), 0, 1, 0, 0, 0, 1, 0, 0]
        lines.append(f'{sample},0,{150 * t:.8f},' + ','.join(map(str, channels)))
    lines.extend(['', 'End'])
    return '\n'.join(lines) + '\n'


@app.function
def moving_rms(x, window):
    """Moving RMS of 'x' with window size 'window'.
    """
    import numpy as np
    from scipy.signal import convolve
    window = int(np.round(window))
    y = np.sqrt(convolve(x * x, np.ones(window) / window, 'same', method='direct'))
    return y


@app.function
def linear_envelope(x, freq=1000, fc_bp=[10, 400], fc_lp=8, method='rect'):
    """Calculate the linear envelope of a signal.

    Parameters
    ----------
    x : 1D array_like
        Raw signal
    freq : number, optional. Default = 1000
        Sampling frequency.
    fc_bp : list of floats [fc_h, fc_l] or None, optional. Default = [10, 400]
        Cutoff frequencies for the band-pass filter (in Hz). Enter None to
        not use this option.
    fc_lp : number, optional. Default = 8
        For rectification: per-pass Butterworth low-pass cutoff (Hz).
        For RMS: inverse window-duration parameter (Hz), not a half-power cutoff.
    method: string, optional. Default = 'rect'
        Method to calculate the linear envelope. Enter 'rect' for using a
        full-wave rectification followed by low-pass Butterworth filter.
        Enter 'rms' for a moving-RMS filter. For the 'rms' method, the
        window size is the nearest integer of freq/fc_lp. For RMS, fc_lp is
        an inverse window-duration parameter, not a Butterworth cutoff; one
        centered mean-square convolution is followed by a square root.

    Returns
    -------
    y : 1D array_like
        Linear envelope of the signal

    Notes
    -----
    A 2nd-order Butterworth filter with zero lag is used for the filtering
    The moving-RMS filter is calculated by one convolution with mode
    equal to 'same', so the extremities of the signal of size 'window'/2 are
    not reliable.

    See this notebook [1]_.

    References
    ----------
    .. [1] https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/electromyography.py

    Version history
    ---------------
    '1.0.1':
        Included parameter 'method', to select method for the linear envelope.
        Changed parameters fc_bp and fc_lp to be written in lowercase.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt
    if fc_bp is not None:
        fc_bp = np.array(fc_bp)
        b, a = butter(N=2, Wn=fc_bp / (freq / 2), btype='bandpass')
        x = filtfilt(b, a, x)
    if method == 'rect':
        x = abs(x)
        b, a = butter(N=2, Wn=fc_lp / (freq / 2), btype='low')
        y = filtfilt(b, a, x)
    elif method == 'rms':
        y = moving_rms(x, freq / fc_lp)
    else:
        raise ValueError("Valid method is 'rect' or 'rms'")
    return y


@app.cell
def _():
    def _plot_df_emg(df, ax, suptitle):
        import numpy as np
        """Plot EMG data of the read_delsys function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                nplots = df.shape[1]
                if nplots <= 3:
                    nrows, ncols = (nplots, 1)
                elif nplots == 4:
                    nrows, ncols = (2, 2)
                else:
                    ncols = 3
                    nrows = int(np.ceil(nplots / ncols))
                _, ax = plt.subplots(nrows, ncols, figsize=(9, 6), sharex='all', constrained_layout=True)
                no_ax = True
            else:
                no_ax = False
            ax = df.plot(color='b', ax=ax, subplots=True)
            if suptitle:
                plt.suptitle(suptitle)
            if no_ax:
                plt.show()

    return


@app.function
def read_delsys(fname, fname2='', sensors=None, freq_trc=150, emg=True, imu=False, resample=[1200, 150], freqs=[20, 20, 450], show_msg=True, show=False, ax=None, suptitle=''):
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
    import numpy as np
    import pandas as pd
    import os
    from scipy import signal
    from fractions import Fraction
    with open(file=fname, mode='rt', encoding='utf-8', newline=None) as f:
        if show_msg:
            print(f'Opening file "{fname}" ... ', end='')
        file = f.read().splitlines()
        if file[0] != 'Cortex generated Asynchronous device data file (.add)':
            raise ValueError('Not a Cortex asynchronous Delsys export')
        idx = file.index('[Devices]') + 2
        count = int(file[idx].split('=')[1])
        devices = [name.split(', ')[-1] for name in file[idx + 1:idx + 1 + count]]
        if sensors is None:
            sensors = devices
        idx = idx + 3 + count
        count2 = int(file[idx].split('=')[1])
        channels = [name for name in file[idx + 1:idx + 1 + count2]]
        n_im = int((count2 - count) / count)
        idxs = np.zeros((count, 4), dtype=int)
        for i, device in enumerate(devices):
            idxs[i, 0] = file.index(device) + 3
            idxs[i, 1] = file[idxs[i, 0]:].index('') + idxs[i, 0] - 1
        idxs[:, 2] = idxs[:, 1] + 3
        idxs[:, 3] = np.r_[idxs[1:, 0] - 6, np.array(len(file) - 3, dtype=int, ndmin=1)]
        if emg:
            nrows_emg = int(np.min(idxs[:, 1] - idxs[:, 0]) + 1)
            f.seek(0)
            t_emg = pd.read_csv(f, sep=',', header=None, names=None, index_col=None, usecols=[2], skiprows=idxs[0, 0], nrows=nrows_emg, dtype=np.float32, encoding='utf-8', engine='c').squeeze('columns').to_numpy()
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
                y = pd.read_csv(f, sep=',', header=None, names=[sensor], index_col=None, usecols=[3], skiprows=idxs[i, 0], nrows=len(t_emg), dtype=np.float32, encoding='utf-8', engine='c').squeeze('columns').to_numpy()
                if resample[0]:
                    if resample[0] < 1000:
                        y = linear_envelope(y, freq_emg, fc_bp=[freqs[1], freqs[2]], fc_lp=freqs[0], method='rms')
                    y = signal.resample_poly(y, fr.numerator, fr.denominator)
                ys[:, i] = y * 1000
            df_emg = pd.DataFrame(data=ys, columns=sensors)
            df_emg.index = float(t_emg[0]) / freq_trc + df_emg.index / freq_emg2
            df_emg.index.name = 'Time'
        if imu:
            nrows_imu = int(np.min(idxs[:, 3] - idxs[:, 2]) + 1)
            cols = [sensor + channel.split(',')[3] for sensor in sensors for channel in channels[1:int(count2 / count)]]
            f.seek(0)
            t_imu = pd.read_csv(f, sep=',', header=None, names=None, index_col=None, usecols=[2], skiprows=idxs[0, 2], nrows=nrows_imu, dtype=np.float32, encoding='utf-8', engine='c').squeeze('columns').to_numpy()
            freq_imu = np.mean(freq_trc / np.diff(t_imu))
            if resample[1]:
                fr = Fraction(resample[1] / freq_imu).limit_denominator(1000)
                nrows_imu = int(np.ceil(nrows_imu * fr.numerator / fr.denominator))
                freq_imu = resample[1]
            ys = np.empty((nrows_imu, count2 - count), dtype=np.float32)
            for i, sensor in enumerate(sensors):
                f.seek(0)
                cs = slice(int(n_im * i), int(n_im * (i + 1)))
                y = pd.read_csv(f, sep=',', header=None, names=cols[cs], index_col=None, usecols=range(3, 3 + n_im), skiprows=idxs[i, 2], nrows=len(t_imu), dtype=np.float32, encoding='utf-8', engine='c').squeeze('columns').to_numpy()
                if resample[1]:
                    y2 = np.empty((nrows_imu, y.shape[1]), dtype=np.float32)
                    for c in range(y.shape[1]):
                        y2[:, c] = signal.resample_poly(y[:, c], fr.numerator, fr.denominator)
                else:
                    y2 = y
                ys[:, cs] = y2
            df_imu = pd.DataFrame(data=ys, columns=cols)
            df_imu.index = float(t_imu[0]) / freq_trc + df_imu.index / freq_imu
            df_imu.index.name = 'Time'
        if show_msg:
            print('done.')
    if len(fname2):
        if isinstance(fname2, list):
            fname2_emg = fname2[0]
            fname2_imu = fname2[1]
        else:
            if emg:
                fname2_emg = fname2
            if imu:
                fname2_imu = fname2
        if emg and fname2_emg == '=':
            name, extension = os.path.splitext(fname)
            fname2_emg = name + '.emg'
        if imu and fname2_imu == '=':
            name, extension = os.path.splitext(fname)
            fname2_imu = name + '.imu'
        if emg:
            df_emg.to_csv(fname2_emg, sep='\t', float_format='%.6f')
            if show_msg:
                print(f'Saving file "{fname2_emg}" ... ', end='')
        if imu:
            df_imu.to_csv(fname2_imu, sep='\t', float_format='%.6f')
            if show_msg:
                print(f'\nSaving file "{fname2_imu}" ... ', end='')
        if show_msg:
            print('done.')
    if show and emg:
        _plot_df_emg(df_emg, ax=None, suptitle=suptitle)
    if emg and imu:
        return (df_emg, df_imu)
    elif emg:
        return df_emg
    elif imu:
        return df_imu


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose the source

    With no upload, the example contains a **synthetic** EMG waveform at 1,000 Hz and nine synthetic inertial channels at 100 Hz. It is a parser demonstration, not an experimental recording. The time column stores Cortex frame units, converted using the Cortex rate below. Confirm that rate for a real export; it cannot be inferred from a filename.
    """)
    return


@app.cell
def _(mo):
    delsys_upload = mo.ui.file(filetypes=[".csv", ".add"], multiple=False, label="Optional Cortex/Delsys export")
    cortex_rate = mo.ui.number(value=150, start=1, label="Cortex frame rate [Hz]")
    mo.vstack([delsys_upload, cortex_rate])
    return cortex_rate, delsys_upload


@app.cell
def _(Path, delsys_upload, mo, tempfile):
    _folder = Path(tempfile.mkdtemp(prefix="bmc-delsys-"))
    input_path = _folder / "input.add"
    if delsys_upload.value:
        input_path.write_bytes(delsys_upload.value[0].contents)
        source_label = "Uploaded recording"
    else:
        input_path.write_text(delsys_demo_text(), encoding="utf-8")
        source_label = "Synthetic format demonstration"
    mo.md(f"**Source: {source_label}.**")
    return (input_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** The EMG time column increases by 0.15 Cortex frames per sample and the Cortex rate is 150 Hz. What is the EMG sample rate?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 1': mo.md(r"""
    One EMG sample spans $0.15/150=0.001$ s, so its sampling rate is 1,000 Hz. The Cortex rate and EMG rate are different quantities.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the original sample rates

    `resample=[0,0]` preserves both native grids. The parser converts EMG volts to millivolts. Inertial-channel units depend on the sensor/export specification and are not inferred from their order. `fname2="\"` disables output-file writing.
    """)
    return


@app.cell
def _(cortex_rate, input_path, mo):
    df_emg, df_imu = read_delsys(input_path, fname2="", sensors=None, freq_trc=cortex_rate.value, emg=True, imu=True, resample=[0, 0], show_msg=False)
    mo.vstack([df_emg.head(), df_imu.head()])
    return df_emg, df_imu


@app.cell
def _(df_emg, df_imu, np):
    {"EMG [Hz]": 1 / np.mean(np.diff(df_emg.index)), "IMU [Hz]": 1 / np.mean(np.diff(df_imu.index))}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** What is a voltage of 0.002 V in millivolts? Does that conversion change its sampling rate?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 2': mo.md(r"""
    It is 2 mV. Multiplying amplitudes by 1,000 changes the unit scale but leaves timestamps and sampling frequency unchanged.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A lower-rate EMG envelope is a different signal

    The original example requested 150 Hz for both outputs. In this reader, requesting an EMG output rate below 1,000 Hz first computes an amplitude envelope after band-pass filtering. That output is an envelope, not raw EMG. The default 20–450 Hz band requires a sufficiently high input sampling rate; choose cutoffs for the actual recording. Here we demonstrate the transformation on the known synthetic source so its assumptions are explicit.
    """)
    return


@app.cell
def _(input_path, plt):
    _demo_path = input_path.parent / "synthetic_for_envelope.add"
    _demo_path.write_text(delsys_demo_text(), encoding="utf-8")
    envelope, imu_resampled = read_delsys(_demo_path, fname2="", freq_trc=150, emg=True, imu=True, resample=[150, 150], show_msg=False)
    _fig, _axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    envelope.plot(ax=_axs[0], legend=False)
    imu_resampled.iloc[:, :3].plot(ax=_axs[1])
    _axs[0].set_ylabel("EMG envelope [mV]")
    _axs[1].set_ylabel("Synthetic IMU [a.u.]")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** Can a raw 450 Hz EMG component be represented without aliasing on a 150 Hz grid?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 3': mo.md(r"""
    No. The new Nyquist frequency is 75 Hz. The example computes an amplitude envelope before resampling; it does not claim to preserve a 450 Hz carrier.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Are EMG and IMU rows necessarily synchronous one-to-one?

    2. What information must accompany uploaded data?

    3. Does the notebook overwrite the input file?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Checkpoint answers': mo.md(r"""
    1. No; compare their times.
    2. The Cortex frame rate, format version, channel interpretation, and units.
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
    **Problem 1.** One second of EMG at 1,000 Hz and IMU at 100 Hz contains how many samples in each native grid?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 1': mo.md(r"""
    1,000 EMG samples and 100 IMU samples.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** What is the Nyquist frequency on the 150 Hz output grid?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 2': mo.md(r"""
    75 Hz.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Inspect the [EMG guide](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/electromyography.py) before interpreting an envelope as muscle activation.

    ### Video lectures on the Internet

    [Delsys video resources](https://www.youtube.com/@DelsysInc) demonstrate EMG acquisition and sensor workflows.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Delsys support](https://delsys.com/support/).

    - [Motion Analysis Cortex](https://www.motionanalysis.com/).

    - [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
    """)
    return


if __name__ == "__main__":
    app.run()
