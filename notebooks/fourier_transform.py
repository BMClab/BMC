# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14", "psd2>=0.0.5", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fourier transform

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A signal recorded in time can also be described by the frequencies it contains. The Fourier transform is the mathematical tool that moves between the two descriptions, and the fast Fourier transform (FFT) is the algorithm that makes it practical. Filters, sampling rates, cutoff frequencies and the fatigue of a muscle are all discussed in terms of frequency, so it pays to be able to compute and read a spectrum yourself.

    This notebook continues [Fourier series](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/fourier_series.py). It defines the Fourier transform and its discrete version, computes the FFT of simple signals and reads their amplitude and phase, shows the traps of a finite recording (frequency resolution and leakage), and then estimates the power spectral density of noisy signals and of a real muscle. It ends with the spectrogram, which shows how the frequency content changes in time. You will also need [Basic properties of signals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/signal_basic_properties.py): sampling, the Nyquist frequency, power and RMS.

    Read it in order and run each cell as you reach it. Try the distributed Exercises before opening their worked solutions. The final Problems have collapsed final answers for independent practice. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and being wrong is the most useful thing that can happen to you here.

    **Challenge 0.** Pick a signal from your own field: walking, a tremor, an EMG, the force under a runner's foot. Write down the range of frequencies you think it contains, from the lowest to the highest that matters. Keep it; by the end you will be able to check a guess like this with a few lines of code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy has the FFT itself, SciPy the spectral estimators, pandas reads the data file, and Matplotlib draws the plots. The declared dependencies also include `psd2` for spectral summaries. Open this file in marimo; the EMG example needs access to the repository data. The synthetic examples need no data download.
    """)
    return


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



@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy import integrate, signal

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return integrate, np, pd, plt, signal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What frequencies are in a muscle's activity?

    Here is the surface electromyography (EMG) used in [Basic properties of signals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/signal_basic_properties.py) and [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py): 3.4 s of the electrical activity of a muscle, sampled at 1000 Hz, with its mean removed. The bottom panel zooms in on 100 ms of the second burst of activity.
    """)
    return


@app.cell
def _(np, pd, plt):
    _data = pd.read_csv(
        repo_file('data/emg.csv'), header=None
    ).to_numpy()
    emg_time = _data[:, 0]
    emg = _data[:, 1] - np.mean(_data[:, 1])  # remove the DC component
    emg_fs = 1 / np.mean(np.diff(emg_time))

    _fig, _axs = plt.subplots(2, 1, figsize=(10, 5.5))
    _axs[0].plot(emg_time, emg, linewidth=0.8)
    _axs[0].axvspan(1.5, 1.6, color="tab:orange", alpha=0.3)
    _axs[0].set_xlabel("Time [s]")
    _axs[0].set_ylabel("EMG [V]")
    _zoom = (emg_time >= 1.5) & (emg_time <= 1.6)
    _axs[1].plot(emg_time[_zoom], emg[_zoom], ".-", color="tab:orange", linewidth=1)
    _axs[1].set_xlabel("Time [s]")
    _axs[1].set_ylabel("EMG [V]")
    plt.tight_layout()
    plt.show()
    return emg, emg_fs, emg_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Guiding questions 0.**

    1. Look at the zoom. Count roughly how many times the signal crosses zero in those 100 ms. What frequency does that suggest?
    2. Is there a single frequency in this signal, or many? Could you tell their relative sizes from the plot?
    3. When a muscle fatigues, its EMG is known to shift towards lower frequencies. How would you measure such a shift from a plot like this one?

    The time plot cannot answer the last two questions. A spectrum can, and by the end of this notebook you will compute this EMG's median frequency, the number used to follow muscle fatigue.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Fourier transform

    In continuation to [Fourier series](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/fourier_series.py), the [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) is a mathematical transformation of functions between the time (or spatial) domain and the frequency domain. Going from time to frequency is called Fourier analysis; the inverse is Fourier synthesis.

    The Fourier transform of a continuous function $x(t)$ is by definition:

    $$
    X(f) = \int_{-\infty}^{\infty} x(t)\:\mathrm{e}^{-i2\pi ft} \:\mathrm{d}t
    $$

    and the inverse Fourier transform is:

    $$
    x(t) = \int_{-\infty}^{\infty} X(f)\:\mathrm{e}^{\:i2\pi tf} \:\mathrm{d}f
    $$

    A Fourier series describes a *periodic* function by a discrete set of harmonics of its fundamental frequency. The Fourier transform extends the idea to functions that need not be periodic: think of it as the period growing without limit, so that the harmonics become infinitely close and the sum becomes an integral over a continuum of frequencies. $X(f)$ is complex: its magnitude says how much of the frequency $f$ the signal contains, and its angle says the phase of that component.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discrete Fourier transform

    A recorded signal is neither continuous nor infinite: it is $N$ samples $x[n]$, taken every $\Delta t = 1/f_s$ seconds. Its Discrete Fourier Transform (DFT) is another sequence $X$, also with $N$ elements:

    $$
    X[k] = \sum_{n=0}^{N-1}  x[n]\, \mathrm{e}^{-i2\pi kn/N} \;,\quad 0 \leq k \leq N-1
    $$

    The Inverse Discrete Fourier Transform (IDFT) inverts this operation and gives back the original data:

    $$
    x[n] = \frac{1}{N} \sum_{k=0}^{N-1}  X[k]\, \mathrm{e}^{i2\pi kn/N} \;,\quad 0 \leq n \leq N-1
    $$

    The element $X[k]$ corresponds to the frequency

    $$
    f_k = \frac{k}{N\Delta t} = k\,\frac{f_s}{N}
    $$

    so the frequencies in a DFT are spaced by $\Delta f = 1/(N\Delta t)$, the inverse of the duration of the recording. This is the **DFT bin spacing**: a 5 s record has bins 0.2 Hz apart, whatever its sampling rate. Resolving two nearby components also depends on the window, their amplitudes and noise; bin spacing alone does not guarantee that they will be distinguishable. The elements above $k = N/2$, beyond the Nyquist frequency $f_s/2$, are the negative frequencies $f_k - f_s$ (see the section on aliasing in [Basic properties of signals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/signal_basic_properties.py)).

    Writing the complex exponential of the inverse transform as $\cos + i\sin$, a real signal can be written as a sum of sinusoids, as in a Fourier series:

    $$
    x[n] = \sum_{k=0}^{N-1} a_k\cos\left(\frac{2\pi kn}{N}\right)+b_k\sin\left(\frac{2\pi kn}{N}\right) \;,\quad 0 \leq n \leq N-1
    $$

    with the Fourier coefficients:

    $$
    a_k = \frac{\text{Real}(X[k])}{N} \;, \qquad b_k = -\frac{\text{Imag}(X[k])}{N}
    $$

    and $a_0 = X[0]/N$ is the mean of the data, its DC component.

    **Guiding questions 1.**

    1. What is the frequency resolution of a 2 s recording sampled at 1000 Hz? And of a 20 s recording sampled at 100 Hz?
    2. For the EMG above, $N = 3360$ and $f_s = 1000$ Hz. What frequency does $X[100]$ correspond to? And $X[3000]$?
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** With 500 samples at 100 Hz, locate the DFT bins for DC, 5 Hz, and Nyquist. What record duration would halve the bin spacing?

    /// details | Solution to Exercise 1
    The duration is $N/f_s=5$ s and $\Delta f=f_s/N=0.2$ Hz. The indices are $k=0$, $k=25$ and $k=250$. Double the record duration to 10 s to obtain 0.1 Hz spacing. Increasing only the sampling rate while retaining 5 s does not do this.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fast Fourier Transform (FFT)

    Computed from its definition, the DFT takes about $N^2$ operations. The [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform) is a family of fast algorithms that compute exactly the same DFT in about $N\log_2 N$ operations: for a million samples, the difference between minutes and milliseconds. NumPy has it in `numpy.fft` (SciPy has an equivalent `scipy.fft`; the older `scipy.fftpack` is legacy).

    Our first signal is a sine wave with an amplitude of 2, a frequency of 5 Hz and a phase of $45^o$, plus a DC component of 1, sampled at 100 Hz for 5 s:
    """)
    return


@app.cell
def _(np, plt):
    fs = 100  # sampling frequency [Hz]
    dt = 1 / fs
    t = np.arange(0, 500) / fs  # 5 s
    x = 2 * np.sin(2 * np.pi * 5 * t + np.pi / 4) + 1
    N = x.size

    _, _ax = plt.subplots(1, 1, figsize=(9, 3))
    _ax.plot(t, x, linewidth=2)
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    return N, dt, fs, t, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Its FFT is simply `np.fft.fft(x)`, and `np.fft.fftfreq` gives the frequency of each element. The amplitude of each component is the magnitude of $X[k]$ divided by $N$, and its phase is the angle of $X[k]$. The phase of a component with no amplitude is meaningless (it is the angle of a number that is zero up to rounding errors), so we set it to zero wherever the amplitude is negligible.

    **Before you run the next cell**, predict at which frequencies there will be peaks, and their heights. Think of $\sin\theta = (\mathrm{e}^{i\theta} - \mathrm{e}^{-i\theta})/2i$.
    """)
    return


@app.cell
def _(N, dt, np, plt, x):
    X = np.fft.fft(x)  # FFT
    freqs = np.fft.fftfreq(N, dt)  # frequency of each element [Hz]
    amp = np.abs(X) / N  # amplitude
    phase = np.where(amp > 1e-6 * amp.max(), np.angle(X, deg=True), 0)  # phase [degrees]

    _fig, _axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    _axs[0].plot(np.fft.fftshift(freqs), np.fft.fftshift(amp), ".-")
    _axs[1].plot(np.fft.fftshift(freqs), np.fft.fftshift(phase), ".-")
    _axs[0].set_ylabel("Amplitude")
    _axs[0].set_ylim(-0.01, 1.1)
    _axs[1].set_ylabel("Phase [$^o$]")
    _axs[1].set_xlabel("Frequency [Hz]")
    _axs[1].set_xlim(-50, 50)
    plt.tight_layout()
    plt.show()

    for _k in np.nonzero(amp > 0.01)[0]:
        print(f"f = {freqs[_k]:5.1f} Hz: amplitude = {amp[_k]:.3f}, phase = {phase[_k]:6.1f} degrees")
    return X, amp, freqs, phase


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three peaks: the DC component, with its full value of 1, and two peaks of 1, half the amplitude of the sine, at $+5$ and $-5$ Hz. A real sinusoid is the sum of two complex exponentials, one at each frequency, and each carries half of it.

    The phase is $-45^o$ at $+5$ Hz, not $+45^o$. The FFT measures phase relative to a *cosine*, and $2\sin(\omega t + 45^o) = 2\cos(\omega t - 45^o)$. Keep that convention in mind whenever you read a phase off an FFT.

    For any real signal, $X[-k]$ is the complex conjugate of $X[k]$: the amplitudes at negative and positive frequencies are the same, and the phases have opposite signs, as in the plot. (If the signal is also even, $X$ is real; if it is odd, $X$ is imaginary.) The negative frequencies carry no new information, so we usually plot only the positive ones and double their amplitudes, except for DC and, for an even number of samples, the Nyquist component, each of which appears only once. This is the one-sided spectrum, which NumPy computes directly with `np.fft.rfft` and `np.fft.rfftfreq`:
    """)
    return


@app.cell
def _(N, dt, np, plt, x):
    _X2 = np.fft.rfft(x)
    _freqs2 = np.fft.rfftfreq(N, dt)
    _amp2 = np.abs(_X2) / N
    _amp2[1:-1 if N % 2 == 0 else None] *= 2
    _phase2 = np.angle(_X2, deg=True)

    _fig, _axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    _axs[0].plot(_freqs2, _amp2, ".-")
    _axs[1].plot(_freqs2, _phase2, ".-")
    _axs[0].set_ylabel("Amplitude")
    _axs[0].set_ylim(-0.01, 2.1)
    _axs[1].set_ylabel("Phase [$^o$]")
    _axs[1].set_xlabel("Frequency [Hz]")
    _axs[1].set_xlim(-0.1, 50)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the amplitude at 5 Hz is the amplitude of the sine, 2, and the DC component is 1.

    ### Fourier synthesis

    We can get the data back from the Fourier coefficients, by adding up all $N$ sinusoids of the formula above. **Before you run the next cell**, predict how close the synthesis will be to the original data.
    """)
    return


@app.cell
def _(N, X, dt, np, plt, t, x):
    # Fourier coefficients
    _a = np.real(X) / N
    _b = -np.imag(X) / N

    # Fourier synthesis: a sum of N sinusoids, one per column
    _w = 2 * np.pi * np.arange(N) / (N * dt)  # angular frequencies
    _y = _a * np.cos(np.outer(t, _w)) + _b * np.sin(np.outer(t, _w))
    _xfft = np.sum(_y, axis=1)

    # or, simply, the inverse FFT
    _xifft = np.real(np.fft.ifft(X))

    _, _ax = plt.subplots(1, 1, figsize=(9, 3))
    _ax.plot(t, x, linewidth=3, label="Original data")
    _ax.plot(t, _xfft, "r--", linewidth=2, label="Fourier synthesis")
    _ax.plot(t, _xifft, "k:", linewidth=2, label="Inverse FFT")
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Amplitude")
    _ax.legend(framealpha=0.7, loc="upper right")
    plt.tight_layout()
    plt.show()
    print(f"Largest difference, synthesis: {np.max(np.abs(_xfft - x)):.1e}")
    print(f"Largest difference, inverse FFT: {np.max(np.abs(_xifft - x)):.1e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Identical, to the rounding errors of the computer. The DFT loses nothing: the $N$ complex numbers $X[k]$ are another, complete description of the $N$ samples. (Between the samples it is another matter: the sum of sinusoids at frequencies above the Nyquist frequency does not interpolate the data sensibly. The DFT describes the samples, not the continuous signal they came from.)
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** For $x(t)=1+2\sin(2\pi5t+\pi/4)$, explain the positive-frequency FFT amplitude and phase. What must a one-sided plot do to DC and Nyquist?

    /// details | Solution to Exercise 2
    Write the sine as $2\cos(2\pi5t-\pi/4)$. The conjugate frequency pair each has amplitude 1 after division by $N$; combining the pair gives amplitude 2 and cosine phase $-45^\circ$. DC remains 1. DC and the even-length Nyquist bin have no distinct partners, so doubling either would be an error.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequency resolution and leakage

    The 5 Hz sine fitted exactly 25 periods in the 5 s of data, and 5 Hz is exactly one of the DFT frequencies, $k\,\Delta f$ with $\Delta f = 0.2$ Hz. Real signals are never that polite. **Before you run the next cell**, predict what the one-sided spectrum of a sine of amplitude 2 at 5.1 Hz, halfway between two DFT frequencies, looks like. Will its peak still be 2?
    """)
    return


@app.cell
def _(N, fs, np, plt, signal, t):
    _f = np.fft.rfftfreq(N, 1 / fs)
    _hann = signal.windows.hann(N)

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for _ax, _freq in zip(_axs, (5, 5.1)):
        _x = 2 * np.sin(2 * np.pi * _freq * t)
        _amp = 2 * np.abs(np.fft.rfft(_x)) / N
        _amp_hann = 2 * np.abs(np.fft.rfft(_x * _hann)) / np.sum(_hann)
        _amp[[0, -1]] /= 2  # N is even: do not double DC or Nyquist
        _amp_hann[[0, -1]] /= 2
        _ax.plot(_f, _amp, "o-", label="no window")
        _ax.plot(_f, _amp_hann, "s--", label="Hann window")
        _ax.set_xlim(3, 7)
        _ax.set_title(f"Sine of amplitude 2 at {_freq} Hz")
        _ax.set_xlabel("Frequency [Hz]")
        print(
            f"{_freq} Hz: peak {_amp.max():.2f} (Hann {_amp_hann.max():.2f}); "
            f"amplitude at 10 Hz {_amp[_f == 10][0]:.4f} (Hann {_amp_hann[_f == 10][0]:.5f})"
        )
    _axs[0].set_ylabel("Amplitude")
    _axs[0].legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At 5.1 Hz the peak falls to about 1.3, and the rest of the amplitude *leaks* into the neighbouring frequencies, spreading over the whole spectrum: even 10 Hz gets some. Nothing is wrong with the sine. The DFT implicitly treats the 5 s of data as one period of a periodic signal, and 5.1 Hz does not fit a whole number of periods in 5 s: the implied periodic signal jumps where the end meets the beginning, and a jump contains all frequencies.

    The usual remedy is a **window**: multiplying the data by a function that goes smoothly to zero at both ends, such as the Hann window, removes the jump. The leakage far from the peak falls by orders of magnitude; the peak itself becomes wider, and with a correction for the window's area its height is closer to 2. Windows trade resolution for leakage, and the spectral estimators below use one by default.

    **Challenge 1.** Change 5.1 Hz to 5.2 Hz in the cell above. Then keep 5.1 Hz but use 10 s of data instead of 5 s. Explain both results with $\Delta f = 1/(N\Delta t)$.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** Answer Challenge 1: why does 5.2 Hz fit the 5 s record, and why does 5.1 Hz fit a 10 s record? Does a Hann window create more frequency resolution?

    /// details | Solution to Exercise 3
    $5.2\times5=26$ and $5.1\times10=51$ are whole numbers of cycles, so both tones fall on their respective DFT grids. A Hann window reduces distant sidelobes by tapering the boundaries, but widens the main lobe. It trades leakage suppression for separation of close components; it does not add information.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A signal buried in noise

    Another example: the sum of two sines, of amplitudes 2 and 1 at 5 and 20 Hz, plus random noise with a standard deviation of 1, sampled at 100 Hz for 5 s. **Before you run the next two cells**, predict whether you will be able to see the two sines in the time plot, and in the spectrum.
    """)
    return


@app.cell
def _(np, plt):
    freq_n = 100.0  # sampling frequency [Hz]
    t_n = np.arange(0, 5, 1 / freq_n)
    _rng = np.random.default_rng(seed=42)
    y_n = (
        2 * np.sin(2 * np.pi * 5 * t_n)
        + np.sin(2 * np.pi * 20 * t_n)
        + _rng.standard_normal(t_n.size)
    )

    _, _ax = plt.subplots(1, 1, figsize=(9, 3))
    _ax.set_title("Time domain")
    _ax.plot(t_n, y_n, "b", linewidth=1)
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("y [V]")
    plt.tight_layout()
    plt.show()
    return freq_n, t_n, y_n


@app.cell
def _(freq_n, np, plt, y_n):
    # one-sided amplitude spectrum
    _amp = np.abs(np.fft.rfft(y_n)) / y_n.size
    _amp[1:-1 if y_n.size % 2 == 0 else None] *= 2
    _freqs = np.fft.rfftfreq(y_n.size, 1 / freq_n)

    _, _ax = plt.subplots(1, 1, figsize=(9, 3))
    _ax.set_title("Frequency domain")
    _ax.plot(_freqs, _amp, "r", linewidth=1.5)
    _ax.set_xlabel("Frequency [Hz]")
    _ax.set_ylabel("Amplitude [V]")
    plt.tight_layout()
    plt.show()
    _top = np.argsort(_amp)[::-1][:3]
    print("Three largest peaks:", ", ".join(f"{_freqs[_i]:.1f} Hz ({_amp[_i]:.2f} V)" for _i in _top))
    print(f"Median amplitude: {np.median(_amp):.3f} V")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the time plot the sines are hard to make out; in the spectrum they stand out clearly, at 5 and 20 Hz, with amplitudes close to 2 and 1. The noise has twice the power of the 20 Hz sine (a variance of 1, against $1^2/2$), but it is spread over all 250 frequencies of the spectrum, while each sine is concentrated in one. That is the power of the frequency domain: it separates what is concentrated in frequency from what is spread out.

    ### FFTW, the Fastest Fourier Transform in the West

    [FFTW](https://www.fftw.org/) is a free collection of fast C routines for computing the DFT, among the fastest available. NumPy and SciPy use their own fast implementations, which are more than enough for the signals in biomechanics; if speed ever becomes a concern, the Python wrapper [pyFFTW](https://pypi.org/project/pyFFTW/) gives access to FFTW.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** A 1 V amplitude sine is mixed with zero-mean noise of variance $0.2$ V². What mean-square power should a PSD recover on average? Is the units label V² correct for PSD?

    /// details | Solution to Exercise 4
    The sine contributes $1^2/2=0.5$ V² and independent zero-mean noise contributes $0.2$ V², so expected total power is $0.7$ V². A density has units V²/Hz; multiplying each bin by its width in Hz and summing gives V². Finite records fluctuate around the expectation.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Power spectral density

    For noisy signals, the amplitude of each frequency is less useful than the **power spectral density** (PSD), the power of the signal per unit of frequency, in units squared per hertz (for instance, V²/Hz). Its integral over all frequencies is the total power of the signal, that is, its mean squared value or, without the DC component, its variance ([Parseval's theorem](https://en.wikipedia.org/wiki/Parseval%27s_theorem)).

    The simplest estimate of the PSD is the [periodogram](https://en.wikipedia.org/wiki/Periodogram), the squared magnitude of the FFT, scaled:

    $$
    P[k] = \frac{|X[k]|^2}{N f_s}
    $$

    doubled for the positive frequencies other than DC and Nyquist, as for the one-sided amplitude. The periodogram of a noisy signal is itself very noisy: each of its values is computed from a single "sample" of the noise. [Welch's method](https://en.wikipedia.org/wiki/Welch%27s_method) reduces that variance by splitting the data into overlapping segments, computing a windowed periodogram of each, and averaging them; the price is a coarser frequency resolution, set by the length of the segments.

    **Before you run the next cell**, predict the total power of the noisy signal above: the power of a sine of amplitude $A$ is $A^2/2$, and the power of the noise is its variance.
    """)
    return


@app.cell
def _(freq_n, np, plt, signal, t_n, y_n):
    _N = y_n.size
    _fp, _Pp = signal.periodogram(y_n, freq_n, window="boxcar", nfft=_N)
    _fw, _Pw = signal.welch(y_n, freq_n, window="hann", nperseg=_N // 4)

    # quick and simple PSD, which is the periodogram
    _P = np.abs(np.fft.rfft(y_n - np.mean(y_n))) ** 2 / _N / freq_n
    _P[1:-1] = 2 * _P[1:-1]  # one-sided, N even: DC and Nyquist appear once

    _fig, _axs = plt.subplots(3, 1, figsize=(10, 8))
    _axs[0].set_title("Time domain")
    _axs[0].plot(t_n, y_n, "b", linewidth=1)
    _axs[0].set_xlabel("Time [s]")
    _axs[0].set_ylabel("y [V]")
    _axs[1].set_title("Periodogram")
    _axs[1].plot(_fp, _Pp, "r", linewidth=1.5)
    _axs[1].set_ylabel("PSD(y) [V$^2$/Hz]")
    _axs[2].set_title(f"Welch's method, segments of {_N // 4} samples")
    _axs[2].plot(_fw, _Pw, "r", linewidth=1.5)
    _axs[2].set_xlabel("Frequency [Hz]")
    _axs[2].set_ylabel("PSD(y) [V$^2$/Hz]")
    plt.tight_layout()
    plt.show()

    print("Quick PSD equals the periodogram:", np.allclose(_P, _Pp))
    print(f"Variance of the data:      {np.var(y_n):.3f} V2")
    print(f"Bin-sum of periodogram:    {np.sum(_Pp) * (_fp[1] - _fp[0]):.3f} V2")
    print(f"Bin-sum of Welch's PSD:    {np.sum(_Pw) * (_fw[1] - _fw[0]):.3f} V2")
    print(f"Bin spacing: periodogram {_fp[1]:.2f} Hz, Welch {_fw[1]:.2f} Hz")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The expected power is $2^2/2 + 1^2/2 + 1 = 3.5$ V². The data have a variance of about 3.4 V² (this particular noise sample has a little less power than its nominal value), and the periodogram bin sum equals that variance (its default detrending removes DC). Welch estimates a weighted segment average and need not equal the full-record variance exactly: the PSD tells you *where in frequency* the power of a signal is. Welch's estimate is smoother and its peaks wider, because each segment is a quarter of the data and has bins four times farther apart, as well as a wider main lobe from the Hann window. For the discrete periodogram, summing bin power preserves Parseval exactly; trapezoidal integration would give half weight to the endpoint bins.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Frequency characteristics of a PSD

    A PSD is often summarized by a few frequencies:

    - the **peak frequency**, $F_{max}$, where the PSD is largest;
    - the **mean frequency**, its centroid:

    $$
    F_{mean} = \frac{ \sum_{i=1}^{N} F_i\,P_i }{ \sum_{i=1}^{N} P_i }
    $$

    - and the frequency **percentiles**: $F_{50\%}$, the **median frequency**, below which lies half of the total power, and $F_{95\%}$, below which lies 95% of it, an estimate of the bandwidth of the signal.

    The function `psd2`, from the Python package [psd2](https://pypi.org/project/psd2/) (included in this notebook’s dependencies), estimates the PSD with Welch's method (it is a wrapper of `scipy.signal.welch`) and computes these characteristics; it returns the frequency percentiles (for example, `fpcntile[50]` is the median frequency), the mean frequency, the peak frequency, the total power, and the PSD itself, and plots the results. Its signature is:

    ```python
    fpcntile, mpf, fmax, Ptotal, f, P = psd2(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
                                             detrend='constant', show=True, ax=None, scales='linear', xlim=None, units='V')
    ```

    Run `help(psd2)` to read its documentation, or read its code on [GitHub](https://github.com/demotu/psd2).
    """)
    return


@app.cell
def _():
    from psd2 import psd2

    return (psd2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's try it on the example from the documentation of `scipy.signal.welch`: a sine wave of 2 V RMS at 1234 Hz, corrupted by white noise of 0.001 V²/Hz, sampled at 10 kHz for 10 s. **Before you run the next cell**, predict the peak frequency and the total power. Will the median and the mean frequency also be near 1234 Hz?
    """)
    return


@app.cell
def _(np, psd2):
    _fs = 10e3
    _N = 100_000
    _amp = 2 * np.sqrt(2)
    _freq = 1234.0
    _noise_power = 0.001 * _fs / 2
    _time = np.arange(_N) / _fs
    _rng = np.random.default_rng(seed=42)
    _x = _amp * np.sin(2 * np.pi * _freq * _time)
    _x += _rng.normal(scale=np.sqrt(_noise_power), size=_time.shape)

    _fpcntile, _mpf, _fmax, _Ptotal, _, _ = psd2(_x, fs=_fs)
    print(f"Peak frequency:   {_fmax:7.1f} Hz")
    print(f"Median frequency: {_fpcntile[50]:7.1f} Hz")
    print(f"Mean frequency:   {_mpf:7.1f} Hz")
    print(f"Total power:      {_Ptotal:7.2f} V2 (variance of the data: {np.var(_x):.2f} V2)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The peak and the median are at 1234 Hz, and the total power is about 9 V²: 4 V² from the sine (2 V RMS, squared) plus 5 V² from the noise (0.001 V²/Hz over 5000 Hz). The mean frequency, though, is near 1940 Hz, where there is nothing but noise. The noise floor looks negligible next to the peak, but it extends over 5000 Hz, and the mean frequency weighs every hertz of it. The median is much less sensitive. Choose the summary frequency with the noise in mind.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back to the muscle

    Now we can answer the questions from the beginning. Let's estimate the PSD of the EMG, with Welch's method and segments of 256 samples (a resolution of about 4 Hz). **Before you run the next cell**, go back to your answer to Guiding question 0.1 and predict the median frequency of this EMG.
    """)
    return


@app.cell
def _(emg, emg_fs, psd2):
    _fpcntile, _mpf, _fmax, _Ptotal, _, _ = psd2(emg, fs=emg_fs, nperseg=256)
    print(f"Peak frequency:   {_fmax:6.1f} Hz")
    print(f"Median frequency: {_fpcntile[50]:6.1f} Hz")
    print(f"Mean frequency:   {_mpf:6.1f} Hz")
    print(f"95% of the power below {_fpcntile[95]:.1f} Hz")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The power of this EMG is spread over a broad band, from a few tens of hertz to about 400 Hz, with a median frequency of about 164 Hz and a mean frequency a little higher. No single frequency dominates: an EMG is the sum of the electrical activity of many motor units, each firing irregularly, and its spectrum is broad.

    That breadth is what makes the median frequency useful. During a sustained contraction, as the muscle fatigues, the conduction velocity of its fibres decreases and the whole spectrum shifts to lower frequencies; the median frequency of successive windows of the EMG falls, and this is one commonly used indicator of muscle fatigue (De Luca, 1997). Electrode geometry, recruitment, motion and analysis choices can also change the spectrum, so a median-frequency change alone does not establish fatigue. Here 95% of the recorded power is below 400 Hz. That describes the stored signal; it cannot prove that an analog anti-alias filter removed frequencies above 500 Hz before acquisition.

    **Guiding questions 2.**

    1. How close was your estimate from counting zero crossings to the median frequency? Why might the two differ?
    2. Compute the PSD again with `nperseg=64` and with `nperseg=1024`. How do the curve and the median frequency change? Which would you report?
    3. This recording has three bursts of activity. How would you check whether the median frequency changes from one burst to the next?
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 5.** You have 2 s of EMG sampled at 1000 Hz. Compare Welch windows of 250 and 1000 samples, and explain why a zero-crossing estimate need not equal the median frequency.

    /// details | Solution to Exercise 5
    The segment lengths are 0.25 and 1 s, giving bin spacings 4 and 1 Hz. Short windows permit more averages and track shorter changes, but broaden spectral features. Median frequency divides spectral power into equal halves; zero crossings depend on the waveform and noise and are not that power-weighted statistic. Use equal windowing and preprocessing when comparing bursts.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Short-time Fourier transform

    The spectrum of a whole recording averages over its whole duration, and the last guiding question needs something else: how the frequency content *changes in time*. The short-time Fourier transform answers it by computing the FFT of short, overlapping windows of the data, one after the other. The result, plotted as a colour map of power against time and frequency, is a **spectrogram**.

    A simple test is a [chirp](https://en.wikipedia.org/wiki/Chirp), a sinusoid whose frequency rises linearly, here from 100 Hz to 300 Hz in about 4 s, sampled at 1000 Hz. **Before you run the next cell**, sketch what its spectrogram should look like.
    """)
    return


@app.cell
def _(np, plt, signal):
    _fs = 1000
    _t = np.arange(2**12) / _fs
    _c = signal.chirp(_t, f0=100, f1=300, t1=_t[-1], method="linear")

    _, _ax = plt.subplots(1, 1, figsize=(10, 4.5))
    _ax.specgram(_c, NFFT=256, Fs=_fs, noverlap=128, cmap="gist_heat")
    _ax.set_title("Spectrogram of a chirp")
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A straight line rising from 100 to 300 Hz, as it should be. Each column is the spectrum of 256 samples, a quarter of a second, so the spectrogram can follow changes that are slow compared with a quarter of a second, and place DFT bins about 4 Hz apart (the window main lobe is wider). A shorter window would follow faster changes and blur the frequencies more; a longer one, the reverse. This is the time-frequency trade-off of every spectrogram, the same $\Delta f = 1/(N\Delta t)$ applied to each window.

    Now the EMG:
    """)
    return


@app.cell
def _(emg, emg_fs, np, plt):
    _fig, _axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _axs[0].plot(np.arange(emg.size) / emg_fs, emg, linewidth=0.8)
    _axs[0].set_ylabel("EMG [V]")
    _axs[1].specgram(emg, NFFT=128, Fs=emg_fs, noverlap=64, cmap="gist_heat")
    _axs[1].set_xlim(0, emg.size / emg_fs)
    _axs[1].set_ylim(0, 500)
    _axs[1].set_xlabel("Time [s]")
    _axs[1].set_ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each burst of activity is a vertical band of power spread over a broad range of frequencies, brightest roughly between 50 and 250 Hz; between the bursts, there is almost nothing. To follow the median frequency during a fatiguing contraction, you would compute it for each column of a spectrogram like this one, or for each window of a few hundred milliseconds of the EMG.

    **Challenge 2.** Use `psd2` on each of the three bursts of the EMG separately (they are roughly centred at 0.5, 1.5 and 2.6 s; take 0.5 s around each). Does the median frequency change from burst to burst? Is the difference larger than what you get by moving each window by 0.1 s?
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 6.** For a 256-sample window at 1000 Hz, compute its duration and bin spacing. How does halving the window affect a changing-frequency signal?

    /// details | Solution to Exercise 6
    The duration is $256/1000=0.256$ s and the bin spacing is $1000/256=3.90625$ Hz. A 128-sample window spans 0.128 s with 7.8125 Hz bins. It localizes changes more closely in time, while spreading their frequency representation more broadly.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your guess in Challenge 0. How long would you need to record that signal to obtain DFT bins 0.5 Hz apart, and how fast would you need to sample it?
    2. The FFT of a 3 Hz sine sampled for 1 s at 100 Hz has a clean single peak; sampled for 1.1 s, it does not. Why?
    3. An FFT reports a phase of $-90^o$ for a component. Is that component a sine or a cosine?
    4. Why is the amplitude of the one-sided spectrum doubled, except at zero frequency and (for even-length data) Nyquist?
    5. Why is the periodogram of a noisy signal so noisy, and what does Welch's method give up to make it smoother?
    6. Two EMG recordings have the same median frequency but different mean frequencies. What might be different about them?
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Checkpoint answers
    1. Two seconds gives 0.5 Hz bin spacing. Choose $f_s>2f_{\max}$ with an analog anti-alias filter and transition margin; resolving close noisy tones may need a longer record.
    2. Three cycles fit 1 s; 3.3 cycles do not fit 1.1 s, producing leakage on that DFT grid.
    3. A positive sine, since $\cos(\omega t-\pi/2)=\sin(\omega t)$.
    4. Positive and negative members of a conjugate pair carry equal amplitudes. DC and the even-length Nyquist bin are unpaired.
    5. A periodogram has high pointwise variability. Welch averages segments, sacrificing segment duration and frequency detail.
    6. Their power distributions can have different tails or asymmetry while retaining the same halfway point. This alone does not identify a physiological cause.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Compute the magnitude of the Fourier-transform approximation `dt * np.fft.fft(pulse)` for a rectangular pulse of unit height and 0.5 s duration, in the middle of 10 s of zeros sampled at 100 Hz. Compare it with the magnitude of the analytical Fourier transform of a rectangular pulse, $|T\,\mathrm{sinc}(fT)|$, with $T = 0.5$ s.

    2. Check Parseval's theorem numerically for the noisy signal above: $\sum_n |x[n]|^2 = \frac{1}{N}\sum_k |X[k]|^2$.

    3. Sample a 70 Hz sine at 100 Hz and compute its spectrum. At what frequency does the peak appear? Relate it to the section on aliasing in [Basic properties of signals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/signal_basic_properties.py).

    4. Load Pezzack's benchmark angle from [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py) and plot its PSD on a logarithmic scale (`scales='semilogy'`). Above what frequency does it flatten into a noise floor? Compare with the optimal cutoff frequency found in [Residual analysis](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/residual_analysis.py).

    5. Zero padding: compute the spectrum of the 5.1 Hz sine with `np.fft.rfft(x, n=4*N)`. Does the peak get closer to 2? Does the frequency resolution really improve?

    6. Compute the spectrogram of the noisy two-sine signal with windows of 32, 128 and `min(512, len(y_n))` samples (the record contains only 500 samples), and describe the time-frequency trade-off you see.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers
    1. The zero-frequency magnitude is 0.5 s; the first nonzero nulls are at $\pm2$ Hz, then multiples of 2 Hz. `dt * fft(pulse)` approximates the continuous transform $T\,\mathrm{sinc}(fT)$ in magnitude, with discrete-time deviations away from low frequencies. A one-sided amplitude spectrum scaled by $2/N$ has different normalization and must not be compared directly with this transform.

    2. Both sides are approximately $1698.51629126209$ for the supplied seed and record. Agreement is to floating-point precision.

    3. The peak appears at 30 Hz. At these sample times, $\sin(2\pi70t)=-\sin(2\pi30t)$; amplitude alone cannot distinguish them.

    4. The low-power tail is broadly visible around 10–20 Hz, with its appearance depending on the window and segment length. There is no unique sharp boundary. The residual-analysis choice is about 5.59 Hz under its stated filter convention, below this tail. A visual PSD boundary and a residual criterion need not agree.

    5. With normalization by the original $N=500$, the padded grid contains 5.1 Hz and the peak is 2.000. The grid spacing becomes 0.05 Hz; the information and main-lobe width of the original 5 s record do not improve. Dividing by $4N$ would incorrectly reduce amplitudes fourfold.

    6. The 32-, 128- and 500-sample windows span 0.32, 1.28 and 5 s, with bin spacings 3.125, 0.78125 and 0.2 Hz. Short windows show fluctuations in time more clearly; long windows separate frequency components more clearly. A 500-sample window gives only one full-record segment here. A literal request for 512 samples exceeds the available record.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Fourier series](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/fourier_series.py) — the periodic case, harmonics and their coefficients.
    - [Basic properties of signals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/signal_basic_properties.py) — sampling, aliasing and power.
    - [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py) — filters described by their frequency response.
    - [Electromyography](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/electromyography.py) — processing EMG in time and in frequency.
    - Smith SW (1997) [The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/), chapters 8 to 12 on the DFT and the FFT.
    ### Video lectures on the Internet

    - 3Blue1Brown: [But what is the Fourier Transform? A visual introduction](https://www.youtube.com/watch?v=spUNpyF58BY).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Bartlett MS (1950) Periodogram analysis and continuous spectra. Biometrika, 37, 1-16.

    - De Luca CJ (1997) The use of surface electromyography in biomechanics. Journal of Applied Biomechanics, 13, 135-163.

    - Lyons RG (2010) [Understanding Digital Signal Processing](https://books.google.com.br/books?id=UBU7Y2tpwWUC&hl). 3rd edition. Prentice Hall.

    - Smith SW (1997) [The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/). California Technical Pub.

    - Welch P (1967) The use of the fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. IEEE Transactions on Audio and Electroacoustics, 15, 70-73.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
