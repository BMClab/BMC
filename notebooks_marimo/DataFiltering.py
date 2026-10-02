import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data filtering in signal processing

    > Marcos Duarte, Renato Naville Watanabe,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    This notebook is an introduction to data filtering and to the most basic filters used to process biomechanical data: the moving average, the Butterworth filter, the critically damped filter and a few others. It starts from a concrete problem, getting an acceleration out of measured positions, and keeps coming back to it until the problem is solved reasonably well.

    You should be familiar with the [basic properties of signals](https://github.com/BMClab/BMC/blob/master/notebooks/SignalBasicProperties.ipynb) before proceeding: frequency, harmonics, sampling, noise and the signal-to-noise ratio. Finite differences, as used in [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb), will also come up.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and being wrong is the most useful thing that can happen to you here.

    **Challenge 0.** Think of a quantity you would like to know but can only get by differentiating a measured position: the acceleration of a body segment, the velocity of a marker, the jerk of a reaching movement. If the position has an error of 1 mm, guess how large the error in the acceleration will be. Write the guess down; you will be able to check its order of magnitude on real data in a moment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy and SciPy do the computation (SciPy's `signal` module has the filters), pandas reads the data files, and Matplotlib draws the plots.
    """)
    return


@app.cell
def _():
    import timeit

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy import interpolate, signal

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return interpolate, np, pd, plt, signal, timeit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The problem: an acceleration from measured positions

    In 1977, Pezzack, Norman and Winter published a paper investigating the effects of differentiation and filtering on experimental data. They filmed a bar being rotated by hand in a horizontal plane, digitized its angle from the film, and at the same time measured its angular acceleration directly, with an accelerometer. Since then these data have become a benchmark for testing new algorithms (they are also available from the [ISB website](https://isbweb.org/data/pezzack/index.html)). We will consider the accelerometer's acceleration as the true one, and ask: can we recover it from the angle alone?

    The most direct answer is to differentiate the angle twice with finite differences. **Before you run the next cell**, predict how the double-differentiated angle will compare with the measured acceleration.
    """)
    return


@app.cell
def _(np, pd, plt):
    _data = pd.read_csv(
        "https://raw.githubusercontent.com/BMClab/BMC/master/data/Pezzack.txt",
        sep="\t",
        header=None,
        skiprows=6,
    ).to_numpy()
    pz_time, pz_disp, pz_disp_noisy, pz_aacc = _data.T
    pz_dt = np.mean(np.diff(pz_time))

    # acceleration by the second-order finite difference (2 samples shorter)
    pz_aacc_diff = np.diff(pz_disp, 2) / pz_dt**2
    _rmse = np.sqrt(np.mean((pz_aacc_diff - pz_aacc[1:-1]) ** 2))

    _fig, _axs = plt.subplots(1, 2, figsize=(11, 4))
    _axs[0].plot(pz_time, pz_disp, "b.-")
    _axs[0].set_xlabel("Time [s]")
    _axs[0].set_ylabel("Angular displacement [rad]")
    _axs[1].plot(pz_time, pz_aacc, "g", linewidth=2, label="Accelerometer (true value)")
    _axs[1].plot(pz_time[1:-1], pz_aacc_diff, "r", label="Angle differentiated twice")
    _axs[1].set_xlabel("Time [s]")
    _axs[1].set_ylabel("Angular acceleration [rad/s$^2$]")
    _axs[1].legend(frameon=False, loc="upper left")
    plt.suptitle("Pezzack's benchmark data", fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"Sampling interval: {pz_dt:.4f} s ({1 / pz_dt:.1f} Hz), {pz_time.size} samples")
    print(f"RMSE of the differentiated acceleration: {_rmse:.1f} rad/s2")
    return pz_aacc, pz_disp, pz_disp_noisy, pz_dt, pz_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The angle looks clean: a smooth curve with no visible noise. Its second derivative is not. The red curve follows the true acceleration only roughly, with a jagged error of tens of rad/s² in places and an RMSE (root-mean-square error) of about 11 rad/s².

    The noise comes from small random errors in the digitization of each film frame, a fraction of a degree. Invisible in the angle, they dominate the acceleration. This notebook is about why that happens and what to do about it.

    **Guiding questions 0.**

    1. The errors in the angle are far too small to see in the left panel. Why do they become so large in the right one?
    2. Is the error in the red curve larger where the acceleration is large, or is it spread over the whole recording?
    3. Would sampling the film faster, with the same digitization error per frame, make the acceleration better or worse?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why differentiation amplifies noise

    Removing noise from a signal is rarely trivial, and the problem gets worse with numerical differentiation: noise at frequencies higher than those of the signal is amplified by differentiation, so the signal-to-noise ratio (SNR) drops with each derivative.

    To see why, consider the following function representing some experimental data:

    $$
    f = \sin(\omega t) + 0.1\sin(10\omega t)
    $$

    The first component, with a large amplitude (1) and a low frequency (1 Hz), represents the signal; the second, with a small amplitude (0.1) and a high frequency (10 Hz), represents the noise. The SNR of these data is $(1/0.1)^2 = 100$.

    **Before you read on**, differentiate $f$ twice by hand and predict the SNR of $f'$ and of $f''$.

    $$
    f\,' = \omega \cos(\omega t) + \omega \cos(10\omega t)
    $$

    $$
    f\,'' = -\omega^2 \sin(\omega t) - 10\omega^2 \sin(10\omega t)
    $$

    Each differentiation multiplies a component of frequency $f$ by $2\pi f$, so the 10 Hz noise gains a factor of 10 on the 1 Hz signal every time. For the first derivative, SNR = 1, and for the second, SNR = 0.01: the noise now has a hundred times the power of the signal. The following plots illustrate the problem:
    """)
    return


@app.cell
def _(np, plt):
    _t = np.arange(0, 1, 0.01)
    _w = 2 * np.pi * 1  # 1 Hz
    # signal and noise, and their derivatives
    _s, _n = np.sin(_w * _t), 0.1 * np.sin(10 * _w * _t)
    _sd, _nd = _w * np.cos(_w * _t), _w * np.cos(10 * _w * _t)
    _sdd, _ndd = -_w * _w * np.sin(_w * _t), -_w * _w * 10 * np.sin(10 * _w * _t)

    _fig, _axs = plt.subplots(3, 1, sharex=True, figsize=(9, 6.5))
    _axs[0].set_title("Differentiation of signal and noise")
    for _ax, _sig, _noi, _label in zip(
        _axs, (_s, _sd, _sdd), (_n, _nd, _ndd), ("f", "f '", "f ''")
    ):
        _ax.plot(_t, _sig, "b.-", linewidth=1, label="signal")
        _ax.plot(_t, _noi, "g.-", linewidth=1, label="noise")
        _ax.plot(_t, _sig + _noi, "r.-", linewidth=2, label="signal+noise")
        _ax.set_ylabel(_label)
    _axs[0].legend(frameon=False, fontsize=10, loc="upper right")
    _axs[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the top panel the noise is a small ripple on the signal. In the bottom panel the signal is the small one. That is the Pezzack acceleration in miniature, and it points to the remedy: if the noise lives at higher frequencies than the signal, remove those frequencies *before* differentiating. That is what a low-pass filter does.

    ## Filter and smoothing

    In data acquisition with an instrument, the noise often has higher frequencies and lower amplitudes than the signal of interest. Removing it is called filtering or smoothing.

    <a href="http://en.wikipedia.org/wiki/Filter_(signal_processing)">Filtering</a> is a process that attenuates some unwanted component or feature of a signal. A filter usually removes certain frequency components of the data, according to its frequency response. The [frequency response](http://en.wikipedia.org/wiki/Frequency_response) is the quantitative measure of the output spectrum of a system in response to a stimulus, and is used to characterize the dynamics of the system.

    [Smoothing](http://en.wikipedia.org/wiki/Smoothing) is the removal of local (short-scale) fluctuations from the data while preserving a more global pattern (those local variations could be noise, or just a short-scale phenomenon that is not of interest). A filter with a low-pass frequency response performs smoothing.

    With respect to its implementation, a filter can be an [analog filter](http://en.wikipedia.org/wiki/Passive_analogue_filter_development) or a [digital filter](http://en.wikipedia.org/wiki/Digital_filter). An analog filter is an electronic circuit that filters an input electrical signal and outputs a filtered electrical signal; a simple one can be built with a resistor and a capacitor. A digital filter is a system that filters digital (discrete-time) data. The anti-aliasing filter of an acquisition system must be analog, because it acts before sampling; everything in this notebook is digital, and acts on data already recorded.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: the moving-average filter

    A simple low-pass (smoothing) filter is the moving average: the arithmetic mean of successive subsequences of $m$ terms of the data. For instance, the moving averages with window sizes $m$ equal to 2 and 3 are:

    $$
    \begin{array}{l}
    y_{MA(2)} = \frac{1}{2}[x_1+x_2,\; x_2+x_3,\; \cdots,\; x_{n-1}+x_n] \\
    y_{MA(3)} = \frac{1}{3}[x_1+x_2+x_3,\; x_2+x_3+x_4,\; \cdots,\; x_{n-2}+x_{n-1}+x_n]
    \end{array}
    $$

    with the general formula:

    $$
    y[i] = \frac{1}{m}\sum_{j=0}^{m-1} x[i+j] \quad \text{for} \quad i=1, \; \dots, \; n-m+1
    $$

    where $n$ is the number of data. Here is a naive implementation, with a loop:
    """)
    return


@app.function
def moving_average(x, window):
    """Moving average of `x` with window size `window` (a naive loop)."""
    import numpy as np

    y = np.empty(len(x) - window + 1)
    for i in range(len(y)):
        y[i] = np.sum(x[i : i + window]) / window
    return y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's test it on a signal that jumps from 0 to 1 and back, plus random noise with a standard deviation of 0.1. **Before you run the next cell**, predict what a window of 11 samples will do to the noise, and what it will do to the two jumps.
    """)
    return


@app.cell
def _(np, plt):
    _rng = np.random.default_rng(seed=42)
    _signal = np.zeros(300)
    _signal[100:200] += 1
    _x = _signal + _rng.standard_normal(300) / 10

    _y = moving_average(_x, 11)

    _, _ax = plt.subplots(1, 1, figsize=(9, 4))
    _ax.plot(_x, "b.-", linewidth=1, label="raw data")
    _ax.plot(_y, "r.-", linewidth=2, label="moving average")
    _ax.legend(frameon=False, loc="upper right")
    _ax.set_xlabel("Sample")
    _ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    print(
        f"Standard deviation of the noise, before: {np.std(_x[:90]):.3f}, "
        f"after: {np.std(_y[:85]):.3f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The noise is reduced several times. Averaging $m$ independent values reduces their standard deviation by $\sqrt{m}$ on average, about 3.3 for $m = 11$; an estimate from a stretch of data this short scatters a good deal around that value. The price is paid at the jumps: each one is now a ramp 11 samples long. Every low-pass filter makes this trade between noise and sharpness; the rest of the notebook is about making it well.

    Look also at *where* the jumps are. The raw data jump at samples 100 and 200, the filtered ones about 5 samples earlier. The output `y[i]` is the mean of `x[i]` to `x[i+10]`, centred on sample `i+5`, but we plotted it at `i`. We will come back to this, and to faster implementations of the moving average, later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Digital filters

    In signal processing, a digital filter is a system that performs mathematical operations on a signal to modify certain aspects of it. A digital filter (more precisely, a causal, linear, time-invariant (LTI) digital filter) can be seen as the implementation of the following difference equation in the time domain:

    $$
    \begin{array}{l l}
    y_n &= b_0x_n + b_1x_{n-1} + \cdots + b_Mx_{n-M} - a_1y_{n-1} - \cdots - a_Ny_{n-N} \\
    &= \sum_{k=0}^M b_kx_{n-k} - \sum_{k=1}^N a_ky_{n-k}
    \end{array}
    $$

    where the output $y$ is the filtered version of the input $x$, $a_k$ and $b_k$ are the filter coefficients (real values), and the order of the filter is the larger of $N$ and $M$.

    This general equation describes a recursive filter: the filtered value $y_n$ is computed from current and previous values of $x$ *and* from previous values of $y$, its own output; it is a system with feedback. A filter that does not reuse its outputs as inputs (a system with only feedforward) is called nonrecursive, and its $a$ coefficients are zero. Recursive and nonrecursive filters are also known as infinite impulse response (IIR) and finite impulse response (FIR) filters, respectively.

    A filter with only the terms on previous values of $y$ is also known as an autoregressive (AR) filter; one with only the terms on current and previous values of $x$, as a moving-average (MA) filter; and one with both, as an autoregressive moving-average (ARMA) filter. The moving average of the previous section is an FIR filter with $m$ coefficients $b$, each equal to $1/m$, and all $a$ coefficients equal to zero.

    ### Transfer function

    Another way to characterize a filter is by its [transfer function](http://en.wikipedia.org/wiki/Transfer_function). In simple terms, the transfer function is the ratio, in the frequency domain, between the output and the input signals of a filter. For a continuous-time input $x(t)$ and output $y(t)$, the transfer function $H(s)$ is the ratio between the [Laplace transforms](http://en.wikipedia.org/wiki/Laplace_transform) of the output and of the input:

    $$
    H(s) = \frac{Y(s)}{X(s)}
    $$

    where $s = \sigma + j\omega$, $j$ is the imaginary unit and $\omega$ is the angular frequency, $2\pi f$. In the steady-state response we can take $\sigma=0$, and the Laplace transforms with complex argument reduce to [Fourier transforms](http://en.wikipedia.org/wiki/Fourier_transform) with real argument $\omega$.

    For a discrete-time input and output, the transfer function $H(z)$ is the ratio between their [z-transforms](http://en.wikipedia.org/wiki/Z-transform), and the formalism is similar. Taking the z-transform of the difference equation above gives the transfer function of a linear, time-invariant, causal digital filter:

    $$
    H(z) = \frac{Y(z)}{X(z)} = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2} + \cdots + b_M z^{-M}}{1 + a_1 z^{-1} + a_2 z^{-2} + \cdots + a_N z^{-N}}
    = \frac{\sum_{k=0}^M b_kz^{-k}}{1 + \sum_{k=1}^N a_kz^{-k}}
    $$

    and again the order of the filter is the larger of $N$ and $M$. As with the difference equation, this is the transfer function of a recursive (IIR) filter; if the $a$ coefficients are zero, the denominator equals one and the filter is nonrecursive (FIR).

    ### The Fourier transform

    The [Fourier transform](http://en.wikipedia.org/wiki/Fourier_transform) is a mathematical operation that transforms a signal that is a function of time, $g(t)$, into a function of frequency, $G(f)$:

    $$
    \mathcal{F}[g(t)] = G(f) = \int_{-\infty}^{\infty} g(t)\, e^{-j 2\pi f t}\, dt
    $$

    Its inverse operation is:

    $$
    \mathcal{F}^{-1}[G(f)] = g(t) = \int_{-\infty}^{\infty} G(f)\, e^{j 2\pi f t}\, df
    $$

    $G(f)$ is the representation of the time-domain signal $g(t)$ in the frequency domain, and vice versa; the two are called a Fourier transform pair. See the notebook [Fourier transform](https://github.com/BMClab/BMC/blob/master/notebooks/FourierTransform.ipynb), or [this text](http://www.thefouriertransform.com/transform/fourier.php), for an introduction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Types of filters

    With respect to the frequencies that are *not* removed from the data, with a boundary given by the critical or cutoff frequency, a filter can be low-pass, high-pass, band-pass or band-stop. The frequency responses of these filters are illustrated in the next figure.

    <figure><center><img src="https://upload.wikimedia.org/wikipedia/en/thumb/e/ec/Bandform_template.svg/640px-Bandform_template.svg.png" width=500 alt="Filters"/></center><figcaption><center><i>Figure. Frequency response of filters (<a href="http://en.wikipedia.org/wiki/Filter_(signal_processing)">from Wikipedia</a>).</i></center></figcaption></figure>

    The critical or cutoff frequency of a filter is the frequency at which the power (the amplitude squared) of the filtered signal is half the power of the input signal, or, equivalently, the output amplitude is 0.707 times the input amplitude. For instance, if a low-pass filter has a cutoff frequency of 10 Hz, a 10 Hz component comes out with 50% of its power and about 71% of its amplitude. (The same $1/\sqrt{2}$ as the RMS of a sinusoid, in [Basic properties of signals](https://github.com/BMClab/BMC/blob/master/notebooks/SignalBasicProperties.ipynb).)

    The gain of a filter (the ratio between the output and input powers) is usually expressed in decibels (dB).

    ### Decibel (dB)

    The <a href="http://en.wikipedia.org/wiki/Decibel">decibel (dB)</a> is a logarithmic unit used to express the ratio between two values. For the gain of a filter:

    $$
    Gain=10\,\log_{10}\left(\frac{A_{out}^2}{A_{in}^2}\right)=20\,\log_{10}\left(\frac{A_{out}}{A_{in}}\right)
    $$

    where $A_{out}$ and $A_{in}$ are the amplitudes of the output (filtered) and input (raw) signals. A decibel is one tenth of a bel, a unit named in honor of <a href="http://en.wikipedia.org/wiki/Alexander_Graham_Bell">Alexander Graham Bell</a>.

    **Before you run the next cell**, predict the gain in decibels at the cutoff frequency, where the power is halved, and for an output amplitude attenuated by a factor of 10 and of 1000.
    """)
    return


@app.cell
def _(np):
    print("Amplitude ratio   Power ratio   Gain [dB]")
    for _ratio in (10, 2, 1, 1 / np.sqrt(2), 0.5, 0.1, 0.001):
        print(f"{_ratio:15.4f} {_ratio**2:13.6f} {20 * np.log10(_ratio):11.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cutoff frequency is the **$-3$ dB** point: $10\log_{10}(0.5) \approx -3$ dB. Doubling the power is $+3$ dB. Every factor of ten in amplitude is 20 dB, so an attenuation of the amplitude by 1000 times is $-60$ dB, and $-120$ dB would be a million times. That compression is why the decibel is useful for describing filters, whose attenuation spans many orders of magnitude.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Butterworth filter

    A common filter in biomechanics and motor control is the [Butterworth filter](http://en.wikipedia.org/wiki/Butterworth_filter). It is popular because it is simple to design and to use, and because its frequency response is maximally flat in the pass band: it does not ripple, so the frequencies it keeps are kept with nearly the same gain. It is a recursive (IIR) filter, and both its $a$ and $b$ coefficients are used.

    In SciPy, the function `butter` calculates the filter coefficients:

    ```python
    butter(N, Wn, btype='low', analog=False, output='ba')
    ```

    where `N` is the order of the filter, `Wn` is the cutoff frequency given as a fraction of the [Nyquist frequency](http://en.wikipedia.org/wiki/Nyquist_frequency) (half the sampling frequency), and `btype` is the type of filter (one of `'lowpass'`, `'highpass'`, `'bandpass'`, `'bandstop'`; the default is `'lowpass'`). The filtering itself is done by the function `lfilter`, which implements the difference equation:

    ```python
    lfilter(b, a, x, axis=-1, zi=None)
    ```

    where `b` and `a` are the coefficients calculated by `butter`, and `x` is the data to be filtered.

    Let's filter the signal-plus-noise function from the section on differentiation, sampled at 100 Hz, with a second-order Butterworth low-pass filter at 5 Hz, between the 1 Hz signal and the 10 Hz noise.
    """)
    return


@app.cell
def _(np, plt, signal):
    fs = 100  # sampling frequency [Hz]
    bw_t = np.arange(0, 1, 1 / fs)
    _w = 2 * np.pi * 1  # 1 Hz
    bw_y = np.sin(_w * bw_t) + 0.1 * np.sin(10 * _w * bw_t)

    # second-order Butterworth low-pass filter at 5 Hz
    _b, _a = signal.butter(2, 5 / (fs / 2), btype="low")
    bw_y2 = signal.lfilter(_b, _a, bw_y)

    _, _ax = plt.subplots(1, 1, figsize=(9, 4))
    _ax.plot(bw_t, bw_y, "r.-", linewidth=2, label="raw data")
    _ax.plot(bw_t, bw_y2, "b.-", linewidth=2, label="filter @ 5 Hz")
    _ax.legend(frameon=False)
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    return bw_t, bw_y, bw_y2, fs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ripple is gone. But the filtered signal is also late: the Butterworth filter introduces a phase shift, a delay, between the raw and filtered signals. We will measure it, and then remove it.

    First, a property of the coefficients. For a low-pass filter to leave a constant signal unchanged (a gain of one at zero frequency), its transfer function at $z = 1$ must equal one, and from the expression for $H(z)$ this means that the sum of the $b$ coefficients minus the sum of the $a$ coefficients, excluding the first, is one:
    """)
    return


@app.cell
def _(np, signal):
    print("Low-pass Butterworth filter coefficients (cutoff at 0.1 of the Nyquist frequency)")
    for _order in (1, 2):
        _b, _a = signal.butter(_order, 0.1, btype="low")
        print(
            f"Order {_order}:\n  b: {_b}\n  a: {_a}\n"
            f"  sum(b) - sum(a[1:]) = {np.sum(_b) - np.sum(_a[1:]):.6f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bode plot

    How much the amplitude of the filtered signal is attenuated relative to the raw signal, as a function of frequency, is given by the frequency response. The plots of the magnitude and phase responses form the [Bode plot](http://en.wikipedia.org/wiki/Bode_plot). Here it is for the filter we just used (Butterworth, low-pass at 5 Hz, second order):
    """)
    return


@app.cell
def _(fs, np, plt, signal):
    _b, _a = signal.butter(2, 5 / (fs / 2), btype="low")
    _f, _h = signal.freqz(_b, _a, worN=512, fs=fs)  # frequency response
    _angles = np.rad2deg(np.unwrap(np.angle(_h)))  # phase [degrees]
    _h_db = 20 * np.log10(np.abs(_h))  # magnitude [dB]

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5))
    _ax1.plot(_f, _h_db, linewidth=2)
    _ax1.plot(5, -3.01, "ro")
    _ax1.set_ylim(-80, 3)
    _ax1.set_title("Frequency response")
    _ax1.set_ylabel("Magnitude [dB]")
    _axi = _ax1.inset_axes([0.06, 0.08, 0.33, 0.5])  # zoom around the cutoff
    _axi.plot(_f, _h_db, linewidth=2)
    _axi.plot(5, -3.01, "ro")
    _axi.set_xlim([0, 10])
    _axi.set_ylim([-6, 0.5])
    _axi.grid(True, linestyle=":")
    _ax2.plot(_f, _angles, linewidth=2)
    _ax2.plot(5, -90, "ro")
    _ax2.set_title("Phase response")
    _ax2.set_xlabel("Frequency [Hz]")
    _ax2.set_ylabel("Phase [degrees]")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The inset shows that at the cutoff frequency, 5 Hz, the filtered signal is indeed attenuated by 3 dB. The phase response shows that at the cutoff frequency, the output lags the input by $90^o$. A 5 Hz component has a period of 0.2 s, so $90^o$, a quarter of a period, is a delay of 0.05 s.

    But the delay we saw in the plot was of the 1 Hz signal, not of a 5 Hz component. **Before you run the next cell**, predict the delay of the 1 Hz component. Is it 0.05 s, five times longer, or something else?
    """)
    return


@app.cell
def _(fs, np, signal):
    _b, _a = signal.butter(2, 5 / (fs / 2), btype="low")
    _freqs = np.array([1, 2, 5, 10])
    _, _h = signal.freqz(_b, _a, worN=_freqs, fs=fs)
    _phase = np.angle(_h)
    print("Frequency [Hz]   Gain [dB]   Phase [degrees]   Delay [s]")
    for _f, _hh, _ph in zip(_freqs, _h, _phase):
        print(
            f"{_f:14d} {20 * np.log10(np.abs(_hh)):11.2f} "
            f"{np.rad2deg(_ph):17.1f} {-_ph / (2 * np.pi * _f):11.3f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The 1 Hz component lags by only $16^o$, but $16^o$ of a 1 s period is 0.045 s, almost the same delay as the 5 Hz component. In its pass band, the phase of the Butterworth filter grows nearly in proportion to the frequency, so every component the filter keeps is delayed by roughly the same time, about $\sqrt{2}/(2\pi f_c) \approx 0.045$ s for this second-order filter. The filtered signal is close to a delayed copy of the slow part of the raw one.

    A delay of 45 ms is not a detail. If you filter a kinematic signal this way and compare it with an EMG or a force filtered differently, every timing between them is off by that much.

    ### Order of a filter

    The order of a filter determines how steep the "wall" of the frequency response is around the cutoff frequency. A vertical wall exactly at the cutoff would be ideal, but it is impossible to implement.

    A first-order Butterworth filter attenuates 6 dB for each doubling of the frequency (per octave), or, which is the same, 20 dB each time the frequency is multiplied by 10 (per decade). In technical terms, a first-order filter rolls off at $-6$ dB per octave, or $-20$ dB per decade. A second-order filter rolls off at $-12$ dB per octave ($-40$ dB per decade), and so on, as the next figure shows.
    """)
    return


@app.function
def butterworth_plot():
    """Plot the frequency response of Butterworth filters of different orders."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {1: "b", 2: "r", 4: "g", 6: "y"}
    for order, color in colors.items():
        b, a = signal.butter(order, 10, "low", analog=True)
        w, h = signal.freqs(b, a, worN=np.logspace(0, 2, 500))
        ax[0].plot(w / 10, 20 * np.log10(abs(h)), color, linewidth=2)
        ax[1].plot(w / 10, np.rad2deg(np.unwrap(np.angle(h))), color, linewidth=2)
    for axi in ax:
        axi.axvline(1, color="black")  # cutoff frequency
        axi.set_xscale("log")
        axi.set_xlabel("Frequency / Critical frequency")
        axi.grid(which="both", axis="both")
    ax[0].scatter(1, -3, marker="s", edgecolor="0", facecolor="1", s=400)
    ax[0].set_ylabel("Magnitude [dB]")
    ax[0].set_xlim(0.1, 10)
    ax[0].set_ylim(-120, 10)
    ax[1].legend([str(order) for order in colors], title="Filter order", loc="best")
    ax[1].set_ylabel("Phase [degrees]")
    ax[1].set_yticks(np.arange(0, -300, -45))
    ax[1].set_ylim(-300, 10)
    axi = ax[0].inset_axes([0.08, 0.08, 0.38, 0.45])  # zoom around the cutoff
    for order, color in colors.items():
        b, a = signal.butter(order, 10, "low", analog=True)
        w, h = signal.freqs(b, a, worN=np.linspace(5, 15, 200))
        axi.plot(w / 10, 20 * np.log10(abs(h)), color, linewidth=2)
    axi.set_xticks((0.6, 1, 1.4))
    axi.set_yticks((-6, -3, 0))
    axi.set_ylim([-7, 1])
    axi.set_xlim([0.5, 1.5])
    axi.grid(which="both", axis="both")
    fig.suptitle(
        "Bode plot of low-pass Butterworth filters of different orders", fontsize=15
    )
    plt.tight_layout()
    plt.show()


@app.cell
def _():
    butterworth_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All orders pass through $-3$ dB at the cutoff frequency (the square). Above it, the higher orders fall much faster: at ten times the cutoff a first-order filter has attenuated by 20 dB, a sixth-order one by 120 dB. Note the cost in the right panel, though: the higher the order, the larger the phase lag.

    **Guiding questions 1.**

    1. Read the left panel at twice the cutoff frequency. How many decibels does each order attenuate there? Does it match $-6$ dB per octave per order?
    2. The noise in Pezzack's data extends up to the Nyquist frequency, about 25 Hz. With a 5 Hz cutoff, roughly how much would a second-order filter attenuate noise at 20 Hz?
    3. Why not always use a very high order?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Butterworth filter with zero-phase shift

    The phase introduced by the Butterworth filter can be cancelled in a digital implementation by filtering the data twice, once forward and once backwards. The lag introduced in the first pass is undone by the same lag in the opposite direction in the second pass. The result is a zero-phase shift (or zero-phase lag) filter. Note that it needs the whole recording to run backwards from its end, so it cannot be used in real time.

    There is a catch. Each pass attenuates the power at the cutoff frequency by half, so two passes attenuate it by a quarter, and the cutoff frequency of the combined filter is no longer the one we asked for. We have to correct the cutoff given to `butter` so that, after both passes, the attenuation is again by half. For a second-order Butterworth filter applied $n$ times, the correction factor is (Winter, 2009):

    $$
    C = \sqrt[4]{2^{\frac{1}{n}} - 1}
    $$

    For two passes, $n=2$, $C=\sqrt[4]{2^{\frac{1}{2}} - 1} \approx 0.802$, and the cutoff frequency given to the filter must be:

    $$
    f_{c,\,actual} = \frac{f_{c,\,desired}}{C}
    $$

    For instance, a second-order Butterworth zero-phase filter with a desired cutoff frequency of 10 Hz must be designed with 12.47 Hz.

    SciPy's function `filtfilt` does the forward and backward filtering. **Before you run the next cell**, predict where the filtfilt curve will be relative to the raw data and to the single-pass filter.
    """)
    return


@app.cell
def _(bw_t, bw_y, bw_y2, fs, plt, signal):
    # correct the cutoff frequency for the two passes of the filter
    C = 0.802
    _b, _a = signal.butter(2, (5 / C) / (fs / 2), btype="low")
    _y3 = signal.filtfilt(_b, _a, bw_y)  # zero-phase filtering

    _, _ax = plt.subplots(1, 1, figsize=(9, 4))
    _ax.plot(bw_t, bw_y, "r.-", linewidth=2, label="raw data")
    _ax.plot(bw_t, bw_y2, "b.-", linewidth=2, label="lfilter @ 5 Hz")
    _ax.plot(bw_t, _y3, "g.-", linewidth=2, label="filtfilt @ 5 Hz")
    _ax.legend(frameon=False)
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    return (C,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The filtfilt output sits right on the slow part of the raw data, with no delay, and without the start-up transient of the single pass. Let's also check the correction: at 5 Hz, one pass of the corrected filter should attenuate by about 1.5 dB, so that two passes attenuate by 3 dB.
    """)
    return


@app.cell
def _(C, fs, np, signal):
    _, _h = signal.freqz(*signal.butter(2, (5 / C) / (fs / 2)), worN=[5], fs=fs)
    _, _h0 = signal.freqz(*signal.butter(2, 5 / (fs / 2)), worN=[5], fs=fs)
    print(
        f"Gain at 5 Hz, corrected cutoff:   one pass {20 * np.log10(abs(_h[0])):.2f} dB, "
        f"two passes {40 * np.log10(abs(_h[0])):.2f} dB"
    )
    print(
        f"Gain at 5 Hz, uncorrected cutoff: one pass {20 * np.log10(abs(_h0[0])):.2f} dB, "
        f"two passes {40 * np.log10(abs(_h0[0])):.2f} dB"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Without the correction, filtering twice at a nominal 5 Hz attenuates 5 Hz by 6 dB: the real cutoff would be lower than the one you report in your methods section.

    ### Critically damped digital filter

    A problem with the low-pass Butterworth filter is that it tends to overshoot or undershoot data with rapid changes (see, for example, Winter (2009), Robertson et al. (2013), and Robertson and Dowling (2003)). The Butterworth filter behaves as an underdamped second-order system; a critically damped filter does not have this overshoot.

    The function `critic_damp` below calculates the coefficients (the $b$'s and $a$'s) of an IIR critically damped digital filter, and corrects the cutoff frequency for the number of passes of the filter. The calculation is very similar to that of the Butterworth filter, so the function can also calculate the Butterworth coefficients if asked to; only the damping term and the correction factor differ.
    """)
    return


@app.function
def critic_damp(fcut, freq, npass=2, fcorr=True, filt="critic"):
    """Coefficients of a critically damped or Butterworth digital low-pass filter.

    A problem with a low-pass Butterworth filter is that it tends to overshoot
    or undershoot data with rapid changes (see for example, Winter (2009),
    Robertson et al. (2013), and Robertson & Dowling (2003)).
    The Butterworth filter behaves as an underdamped second-order system and a
    critically damped filter doesn't have this overshoot/undershoot
    characteristic.

    Parameters
    ----------
    fcut : number
        desired cutoff frequency for the low-pass digital filter (Hz).
    freq : number
        sampling frequency (Hz).
    npass : number, optional (default = 2)
        number of passes the filter will be applied.
        choose 2 for a second-order zero-phase lag filter.
    fcorr : bool, optional (default = True)
        correct (True) or not the cutoff frequency for the number of passes.
    filt : string ('critic', 'butter'), optional (default = 'critic')
        'critic' to calculate coefficients for a critically damped filter,
        'butter' to calculate coefficients for a Butterworth filter.

    Returns
    -------
    b : 1D array
        b coefficients for the filter
    a : 1D array
        a coefficients for the filter
    fc : number
        corrected cutoff frequency considering the number of passes

    References
    ----------
    Robertson DG, Dowling JJ (2003) Design and responses of Butterworth and
    critically damped digital filters. J Electromyogr Kinesiol 13, 569-573.
    """
    import warnings

    import numpy as np

    filt = filt.lower()
    if filt not in ("critic", "butter"):
        raise ValueError(f"Invalid option for parameter filt: {filt!r}")
    if fcut > freq / 2:
        warnings.warn("Cutoff frequency can not be greater than Nyquist frequency.")

    # cutoff frequency correction for the number of passes
    if fcorr:
        if filt == "critic":
            corr = 1 / np.power(2 ** (1 / (2 * npass)) - 1, 0.5)
        else:
            corr = 1 / np.power(2 ** (1 / npass) - 1, 0.25)
        fc = fcut * corr
        if fc > freq / 2:
            warnings.warn(
                f"Corrected cutoff frequency ({fc} Hz) is greater than the Nyquist"
                f" frequency ({freq / 2} Hz). Using the uncorrected cutoff"
                f" frequency ({fcut} Hz)."
            )
            fc = fcut
    else:
        fc = fcut

    # corrected angular cutoff frequency
    wc = np.tan(np.pi * fc / freq)
    # low-pass coefficients
    k1 = np.sqrt(2) * wc if filt == "butter" else 2 * wc
    k2 = wc * wc
    a0 = k2 / (1 + k1 + k2)
    a1 = 2 * a0
    a2 = k2 / (1 + k1 + k2)
    b1 = 2 * a0 * (1 / k2 - 1)
    b2 = 1 - (a0 + a1 + a2 + b1)
    # transform parameters to be consistent with SciPy butter output
    b = np.array([a0, a1, a2])
    a = np.array([1, -b1, -b2])

    return b, a, fc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, a sanity check: with `filt='butter'`, the function should give the same coefficients as SciPy's `butter` at the corrected cutoff frequency.
    """)
    return


@app.cell
def _(np, signal):
    _b_bw, _a_bw, _fc_bw = critic_damp(fcut=10, freq=100, npass=2, filt="butter")
    _b_sp, _a_sp = signal.butter(2, _fc_bw / (100 / 2))
    print(f"Butterworth, corrected cutoff: {_fc_bw:.2f} Hz")
    print(
        "Same coefficients as scipy.signal.butter:",
        np.allclose(_b_bw, _b_sp) and np.allclose(_a_bw, _a_sp),
    )

    _b_cd, _a_cd, _fc_cd = critic_damp(fcut=10, freq=100, npass=2, filt="critic")
    print(f"Critically damped, corrected cutoff: {_fc_cd:.2f} Hz")
    print("  b:", _b_cd, "\n  a:", _a_cd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two filters need very different corrections: a critically damped filter applied twice must be designed with more than twice the desired cutoff frequency, because its response falls off more gently.

    Now the reason for its existence. Let's filter a step, the most rapid change possible, sampled at 100 Hz, with both filters at a 10 Hz cutoff, two passes each. We add a third, nonrecursive option for comparison: an FIR low-pass filter with 11 coefficients, designed with SciPy's `firwin` (and, unlike the other two, without correcting its cutoff for the two passes).

    **Before you run the next cell**, predict which filters will overshoot the step, and by how much.
    """)
    return


@app.cell
def _(np, plt, signal):
    _y = np.hstack((np.zeros(20), np.ones(20)))
    _t = np.linspace(0, 0.39, 40) - 0.19

    _b_cd, _a_cd, _ = critic_damp(fcut=10, freq=100, npass=2, filt="critic")
    _b_bw, _a_bw, _ = critic_damp(fcut=10, freq=100, npass=2, filt="butter")
    _b_fir = signal.firwin(numtaps=11, cutoff=10, fs=100)
    _filtered = {
        "Butterworth": ("b", signal.filtfilt(_b_bw, _a_bw, _y)),
        "Critically damped": ("r", signal.filtfilt(_b_cd, _a_cd, _y)),
        "FIR, 11 coefficients": ("g", signal.filtfilt(_b_fir, 1, _y)),
    }

    _, _ax = plt.subplots(1, 1, figsize=(9, 4))
    _ax.plot(_t, _y, "k", linewidth=2, drawstyle="steps-mid", label="raw data")
    for _name, (_color, _yf) in _filtered.items():
        _ax.plot(_t, _yf, _color + ".-", linewidth=2, label=_name)
        print(
            f"{_name:21s} overshoot: {100 * (_yf.max() - 1):5.1f}%, "
            f"undershoot: {max(0.0, -100 * _yf.min()):5.1f}%"
        )
    _ax.legend(frameon=False, loc="upper left")
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Amplitude")
    _ax.set_title("Step response: 100 Hz sampling, 10 Hz cutoff, zero-phase filters")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Butterworth filter overshoots the step by almost 4%, and undershoots it by the same amount *before* the step happens: this is a zero-phase filter, so the ringing appears on both sides. The critically damped filter does neither, and neither does this FIR filter. For a step, an overshoot of 4% may not matter; in the second derivative of a step, such as a foot striking the ground, the ringing becomes a spurious oscillation in the acceleration.

    **Challenge 1.** Change the cutoff frequency in the cell above to 5 Hz and to 20 Hz. Does the overshoot of the Butterworth filter change? What about the width of the transition? What property of the filter determines the overshoot?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Moving filters

    ### Moving-average filter, four ways

    Back to the moving average. Here are four implementations of it: the naive loop from the beginning of the notebook, one with a cumulative sum, one with a convolution, and one with the general filter `lfilter`, using the coefficients $b_k = 1/m$ and no $a$ coefficients.
    """)
    return


@app.function
def moving_average_cumsum(x, window):
    """Moving average of `x` with window size `window`, by a cumulative sum."""
    import numpy as np

    xsum = np.cumsum(x)
    xsum[window:] = xsum[window:] - xsum[:-window]
    return xsum[window - 1 :] / window


@app.function
def moving_average_convolve(x, window):
    """Moving average of `x` with window size `window`, by convolution."""
    import numpy as np

    return np.convolve(x, np.ones(window) / window, "same")


@app.function
def moving_average_lfilter(x, window):
    """Moving average of `x` with window size `window`, as an FIR filter."""
    import numpy as np
    from scipy.signal import lfilter

    return lfilter(np.ones(window) / window, 1, x)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's test these versions on the same kind of step signal as before. **Before you run the next cell**, predict which of the four curves will coincide, and which will be shifted in time relative to the raw data.
    """)
    return


@app.cell
def _(np, plt):
    _rng = np.random.default_rng(seed=1)
    ma_x = _rng.standard_normal(300) / 10
    ma_x[100:200] += 1
    ma_window = 10

    _versions = {
        "loop": (moving_average, "y-"),
        "cumsum": (moving_average_cumsum, "m--"),
        "convolve": (moving_average_convolve, "r-"),
        "lfilter": (moving_average_lfilter, "g-"),
    }

    _, _ax = plt.subplots(1, 1, figsize=(10, 5))
    _ax.plot(ma_x, "b-", linewidth=1, label="raw data")
    for _name, (_func, _style) in _versions.items():
        _y = _func(ma_x, ma_window)
        _ax.plot(_y, _style, linewidth=2, label=f"moving average, {_name}")
        print(f"{_name:9s} output length: {_y.size}")
    _ax.legend(frameon=False, loc="upper right")
    _ax.set_xlabel("Sample")
    _ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    return ma_window, ma_x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The loop and the cumulative-sum versions give identical results, $m-1$ samples shorter than the input. The convolution and `lfilter` versions keep the length of the input. But only the convolution version is aligned with the raw data. The other three compute the same averages and attribute them to different instants:

    - the loop and cumsum versions put the mean of `x[i]` to `x[i+m-1]` at sample `i`, the *start* of the window, so they lead the data by $(m-1)/2$ samples;
    - `lfilter` puts the mean of `x[i-m+1]` to `x[i]` at sample `i`, the *end* of the window, so it lags by $(m-1)/2$ samples, as a causal filter must;
    - `np.convolve` with `'same'` puts it at the *centre* of the window.

    It is the same question as where to place a finite-difference velocity in [Kinematics of a particle](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsParticle.ipynb): an average over an interval belongs to its middle. The first three could be fixed by shifting their output, or, for `lfilter`, by using `filtfilt` instead.

    Now the speed. **Before you run the next cell**, guess how much slower the loop is than the others.
    """)
    return


@app.cell
def _(ma_window, ma_x, timeit):
    for _func in (
        moving_average,
        moving_average_cumsum,
        moving_average_convolve,
        moving_average_lfilter,
    ):
        _time = timeit.timeit(lambda: _func(ma_x, ma_window), number=200) / 200
        print(f"{_func.__name__:24s} {1e6 * _time:9.1f} microseconds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The exact numbers depend on your computer, but the pattern should not: the version with the Python loop is around a hundred times slower than the others, which hand the loop to compiled code inside NumPy or SciPy. Avoid explicit loops over the samples of a signal whenever a vectorized operation exists.

    ### Moving-RMS filter

    The root mean square (RMS) is a measure of the absolute amplitude of the data, useful when the data have positive and negative values, as an EMG does. The RMS is defined as:

    $$
    RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
    $$

    and, like the moving average, the moving RMS applies it to a sliding window of $m$ samples:

    $$
    y[i] = \sqrt{\frac{1}{m}\sum_{j=0}^{m-1} (x[i+j])^2} \quad \text{for} \quad i=1, \; \dots, \; n-m+1
    $$

    Here are two implementations of a moving-RMS filter, very similar to the moving average: one with a convolution over a centred window of $2m+1$ samples, and one with `filtfilt`, which applies a window of $m$ samples forward and then backward.
    """)
    return


@app.function
def moving_rms_convolve(x, window):
    """Moving RMS of `x` over a centred window of 2*`window` + 1 samples."""
    import numpy as np

    window = 2 * window + 1
    return np.sqrt(np.convolve(x * x, np.ones(window) / window, "same"))


@app.function
def moving_rms_filtfilt(x, window):
    """Moving RMS of `x`, a `window`-sample average applied forward and backward."""
    import numpy as np
    from scipy.signal import filtfilt

    return np.sqrt(filtfilt(np.ones(window) / window, [1], x * x))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's apply them to the electromyographic data of [Basic properties of signals](https://github.com/BMClab/BMC/blob/master/notebooks/SignalBasicProperties.ipynb), sampled at 1000 Hz, with the mean removed first:
    """)
    return


@app.cell
def _(np, pd, plt):
    _data = pd.read_csv(
        "https://raw.githubusercontent.com/BMClab/BMC/master/data/emg.csv", header=None
    ).to_numpy()[300:1000]
    emg_time = _data[:, 0]
    emg = _data[:, 1] - np.mean(_data[:, 1])
    emg_window = 50

    _y1 = moving_rms_convolve(emg, emg_window)
    _y2 = moving_rms_filtfilt(emg, emg_window)

    _, _ax = plt.subplots(1, 1, figsize=(9, 5))
    _ax.plot(emg_time, emg, "k-", linewidth=1, label="raw data")
    _ax.plot(emg_time, _y1, "r-", linewidth=2, label="moving RMS, convolve")
    _ax.plot(emg_time, _y2, "b-", linewidth=2, label="moving RMS, filtfilt")
    _ax.legend(frameon=False, loc="upper right")
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Amplitude")
    _ax.set_ylim(-0.1, 0.1)
    plt.tight_layout()
    plt.show()
    return emg, emg_window


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similar, but not the same. Both span about 100 samples (0.1 s), but they weight them differently. The convolution version averages 101 samples with equal weights. The filtfilt version averages 50 samples twice, and a box averaged with itself is a triangle 99 samples wide, which weights the centre more than the edges; its result is a little smoother. Choosing a window, and its shape, is part of the method, and should be reported.

    The convolution version is also faster:
    """)
    return


@app.cell
def _(emg, emg_window, timeit):
    for _func in (moving_rms_convolve, moving_rms_filtfilt):
        _time = timeit.timeit(lambda: _func(emg, emg_window), number=100) / 100
        print(f"{_func.__name__:20s} {1e6 * _time:9.1f} microseconds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Moving-median filter

    The moving-median filter is similar in concept to the other moving filters but uses the median instead of the mean. Because the median ignores the extreme values in the window, it preserves abrupt changes better than the moving average, and it removes isolated spikes almost completely. **Before you run the next cell**, predict how each filter will treat the two jumps.
    """)
    return


@app.cell
def _(np, plt, signal):
    _rng = np.random.default_rng(seed=2)
    _x = _rng.standard_normal(300) / 10
    _x[100:200] += 1
    _window = 11

    _y = np.convolve(_x, np.ones(_window) / _window, "same")
    _y2 = signal.medfilt(_x, _window)

    _, _ax = plt.subplots(1, 1, figsize=(10, 4))
    _ax.plot(_x, "b-", linewidth=1, label="raw data")
    _ax.plot(_y, "r-", linewidth=2, label="moving average")
    _ax.plot(_y2, "g-", linewidth=2, label="moving median")
    _ax.legend(frameon=False, loc="upper right")
    _ax.set_xlabel("Sample")
    _ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The moving median keeps the jumps nearly vertical, where the moving average turns them into ramps. Both filters also show what happens at the ends of the data: `np.convolve` with `'same'` and `medfilt` pad the data with zeros beyond its edges, which here, where the signal is near zero, does little harm. Every filter has to make some assumption about the data it does not have, and the edges are where that shows.

    ### More moving filters

    The library [pandas](https://pandas.pydata.org/) has many types of [moving (rolling) window functions](https://pandas.pydata.org/docs/user_guide/window.html): mean, median, standard deviation, quantiles, weighted windows and others, as in `pd.Series(x).rolling(11, center=True).mean()`. They mark the edges, where the window is incomplete, with `NaN` instead of padding the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filtering before differentiating

    Back to the function $f = \sin(\omega t) + 0.1\sin(10\omega t)$. Let's filter it with a zero-phase Butterworth low-pass at 5 Hz, differentiate the raw and the filtered data twice, and look at the results in the time domain and in the frequency domain, using the [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) to see their frequency content.

    **Before you run the next cell**, predict what the amplitude spectrum of the raw $f''$ will look like: which peak will be larger, the one at 1 Hz or the one at 10 Hz?
    """)
    return


@app.cell
def _(C, bw_t, bw_y, fs, np, plt, signal):
    _b, _a = signal.butter(2, (5 / C) / (fs / 2), btype="low")
    _yf = signal.filtfilt(_b, _a, bw_y)
    # second derivatives
    _ydd = np.diff(bw_y, 2) * fs * fs  # raw data
    _yfdd = np.diff(_yf, 2) * fs * fs  # filtered data


    def _amplitude_spectrum(x):
        """Amplitude spectrum of `x`, up to 25 Hz."""
        _freqs = np.fft.rfftfreq(x.size, 1 / fs)
        _amp = np.abs(np.fft.rfft(x)) / (x.size / 2)
        return _freqs[_freqs <= 25], _amp[_freqs <= 25]


    _fig, _axs = plt.subplots(2, 2, figsize=(11, 5.5))
    _axs[0, 0].set_title("Time domain")
    _axs[0, 0].plot(bw_t, bw_y, "r", linewidth=2, label="raw data")
    _axs[0, 0].plot(bw_t, _yf, "b", linewidth=2, label="filtered @ 5 Hz")
    _axs[0, 0].set_ylabel("f")
    _axs[0, 0].legend(frameon=False)
    _axs[0, 1].set_title("Frequency domain")
    _axs[0, 1].plot(*_amplitude_spectrum(bw_y), "r", linewidth=2, label="raw data")
    _axs[0, 1].plot(
        *_amplitude_spectrum(_yf), "b--", linewidth=2, label="filtered @ 5 Hz"
    )
    _axs[0, 1].set_ylabel("Amplitude of f")
    _axs[0, 1].legend(frameon=False)
    _axs[1, 0].plot(bw_t[1:-1], _ydd, "r", linewidth=2)
    _axs[1, 0].plot(bw_t[1:-1], _yfdd, "b", linewidth=2)
    _axs[1, 0].set_xlabel("Time [s]")
    _axs[1, 0].set_ylabel("f ''")
    _axs[1, 1].plot(*_amplitude_spectrum(_ydd), "r", linewidth=2)
    _axs[1, 1].plot(*_amplitude_spectrum(_yfdd), "b--", linewidth=2)
    _axs[1, 1].set_xlabel("Frequency [Hz]")
    _axs[1, 1].set_ylabel("Amplitude of f ''")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the raw data the 10 Hz peak is a tenth of the 1 Hz peak; in the raw second derivative it is about nine times *larger*, close to the factor of 10 in amplitude computed by hand (a little less, because a finite difference underestimates the derivative of a fast oscillation).

    After filtering, the 10 Hz peak of $f$ has almost disappeared: it is about 1% of the signal. And yet in the second derivative of the filtered data it is back, about as large as the signal itself. Nothing went wrong. Two passes of a second-order filter at 5 Hz reduce the amplitude at 10 Hz only about eight times, and the second derivative multiplies it by 100 again. **A filter that looks perfect on the data can be far too weak for their derivatives**; how much attenuation you need depends on how many times you will differentiate.

    **Challenge 2.** Change the cutoff frequency in the cell above to 3 Hz, then go back to 5 Hz and filter twice with `filtfilt` (four passes in all). Which works better for $f''$? What does each choice cost the 1 Hz signal?

    ## Pezzack's benchmark data, again

    Now we can attack the problem from the beginning of the notebook. The noise in Pezzack's angle comes from small random errors in the digitization of each frame, so it is spread up to the Nyquist frequency, above the frequency content of the movement. Let's try three different methods to attenuate it: a [Butterworth](https://en.wikipedia.org/wiki/Butterworth_filter) filter, a [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) filter, and a smoothing [spline](https://en.wikipedia.org/wiki/Spline_function).

    The Savitzky-Golay filter and the spline both fit polynomials to the data, and both can differentiate the fitted polynomials to get the derivatives directly, instead of differentiating the data numerically. Their signatures in SciPy are:

    ```python
    savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None, full_output=0, per=0, quiet=1)
    ```

    and the spline derivatives are evaluated with:

    ```python
    splev(x, tck, der=0, ext=0)
    ```

    Both fits behave poorly at the ends of the data, where there are points on only one side. A common workaround is to pad the data before fitting, with an *odd extension*: the data reflected about their first and last values, which continues the trend of the data instead of flattening it.
    """)
    return


@app.function
def odd_extension(x, n):
    """Extend `x` by `n` samples at each end, reflected about its end values."""
    import numpy as np

    x = np.asarray(x)
    return np.concatenate((2 * x[0] - x[n:0:-1], x, 2 * x[-1] - x[-2 : -n - 2 : -1]))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will compare the methods with the [root-mean-square error (RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) of each estimated acceleration relative to the true one. The plain double difference had an RMSE of about 11 rad/s². **Before you run the next cell**, guess how much each method will reduce it: by half, by a factor of ten, more?
    """)
    return


@app.cell
def _(C, interpolate, np, plt, pz_aacc, pz_disp, pz_dt, pz_time, signal):
    # Butterworth filter at 9 Hz, zero phase
    _b, _a = signal.butter(2, (9 / C) / ((1 / pz_dt) / 2))
    _aacc_bw = np.diff(signal.filtfilt(_b, _a, pz_disp), 2) / pz_dt**2  # 2 samples shorter

    # pad the data at the extremities to reduce edge effects
    _n = 11
    _disp_pad = odd_extension(pz_disp, _n)
    _time_pad = odd_extension(pz_time, _n)

    # Savitzky-Golay filter
    _aacc_sg = signal.savgol_filter(
        _disp_pad, window_length=5, polyorder=3, deriv=2, delta=pz_dt
    )[_n:-_n]

    # quintic smoothing spline
    _s = 0.15 * np.var(_disp_pad) / np.size(_disp_pad)
    _tck = interpolate.splrep(_time_pad, _disp_pad, k=5, s=_s)
    _aacc_sp = interpolate.splev(_time_pad, _tck, der=2)[_n:-_n]

    _rmse_bw = np.sqrt(np.mean((_aacc_bw - pz_aacc[1:-1]) ** 2))
    _rmse_sg = np.sqrt(np.mean((_aacc_sg - pz_aacc) ** 2))
    _rmse_sp = np.sqrt(np.mean((_aacc_sp - pz_aacc) ** 2))

    _, _ax = plt.subplots(1, 1, figsize=(11, 4.5))
    _ax.plot(pz_time, pz_aacc, "g", linewidth=3, label="Accelerometer (true value)")
    _ax.plot(pz_time[1:-1], _aacc_bw, "r", label=f"Butterworth 9 Hz: RMSE = {_rmse_bw:.2f}")
    _ax.plot(pz_time, _aacc_sg, "b", label=f"Savitzky-Golay 5 points: RMSE = {_rmse_sg:.2f}")
    _ax.plot(
        pz_time, _aacc_sp, "m", label=f"Quintic spline, s = {_s:.5f}: RMSE = {_rmse_sp:.2f}"
    )
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Angular acceleration [rad/s$^2$]")
    _ax.set_title("Pezzack's benchmark data")
    _ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All three methods cut the error by more than half, to about 4–5 rad/s², and all three follow the true acceleration closely. None of them is clearly better than the others here. With all of them, and particularly with the spline, the parameters had to be tuned: the 9 Hz cutoff, the 5-point window, the smoothing factor `s`. The Butterworth filter is often the easiest to tune, because a cutoff frequency is a parameter with a physical meaning for human movement.

    But where did 9 Hz come from? It was chosen by hand, looking at the result, and here we had the luxury of knowing the true acceleration. In a real experiment you would not.

    **Challenge 3.** The data file also has a second, noisier version of the angle, loaded above as `pz_disp_noisy`.

    1. Differentiate it twice without filtering and compare the RMSE with that of `pz_disp`.
    2. Filter it with the same Butterworth filter at 9 Hz. Is 9 Hz still a good choice? Try a few other cutoff frequencies and find the one that gives the smallest RMSE.
    3. Without the accelerometer, how would you have chosen the cutoff? Keep your answer for [Residual analysis](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kinematics of a ball toss

    Filtering is not the only way to deal with noise. Let's analyse the kinematic data of a ball tossed in the air. They were obtained with [Tracker](https://physlets.org/tracker/), a free video analysis and modeling tool built on the [Open Source Physics](https://www.compadre.org/osp/) (OSP) Java framework, from the video *balltossout.mov* in the mechanics video collection on the Tracker website.
    """)
    return


@app.cell
def _(np, pd, plt):
    _data = pd.read_csv(
        "https://raw.githubusercontent.com/BMClab/BMC/master/data/balltoss.txt",
        sep="\t",
        header=None,
        skiprows=2,
    ).to_numpy()
    ball_t, ball_x, ball_y = _data.T
    ball_dt = np.mean(np.diff(ball_t))
    print(f"Time interval: {ball_dt:.4f} s ({1 / ball_dt:.0f} Hz), {ball_t.size} frames")

    _fig, _axs = plt.subplots(1, 3, figsize=(12, 3.2))
    _axs[0].plot(ball_x, ball_y, "go")
    _axs[0].set_xlabel("x [m]")
    _axs[0].set_ylabel("y [m]")
    _axs[1].plot(ball_t, ball_x, "bo")
    _axs[1].set_xlabel("Time [s]")
    _axs[1].set_ylabel("x [m]")
    _axs[2].plot(ball_t, ball_y, "ro")
    _axs[2].set_xlabel("Time [s]")
    _axs[2].set_ylabel("y [m]")
    plt.suptitle("Kinematics of a ball toss", fontsize=14)
    plt.tight_layout()
    plt.show()
    return ball_dt, ball_t, ball_x, ball_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Twenty-two frames at 30 Hz. Let's compute the velocity and acceleration numerically, with two algorithms: the forward difference, $(x_{i+1} - x_i)/\Delta t$, and the central difference, $(x_{i+1} - x_{i-1})/(2\Delta t)$.

    **Before you run the next cells**, predict the horizontal and vertical accelerations of the ball. Which of the two algorithms do you expect to be noisier?
    """)
    return


@app.cell
def _(ball_dt, ball_x, ball_y, np):
    # forward difference algorithm
    vx, vy = np.diff(ball_x) / ball_dt, np.diff(ball_y) / ball_dt
    ax, ay = np.diff(vx) / ball_dt, np.diff(vy) / ball_dt
    # central difference algorithm
    vx2 = (ball_x[2:] - ball_x[:-2]) / (2 * ball_dt)
    vy2 = (ball_y[2:] - ball_y[:-2]) / (2 * ball_dt)
    ax2, ay2 = (vx2[2:] - vx2[:-2]) / (2 * ball_dt), (vy2[2:] - vy2[:-2]) / (2 * ball_dt)
    return ax, ax2, ay, ay2, vx, vx2, vy, vy2


@app.cell
def _(ax, ax2, ay, ay2, ball_t, ball_x, ball_y, plt, vx, vx2, vy, vy2):
    _fig, _axs = plt.subplots(2, 3, sharex=True, figsize=(11, 6))
    _axs[0, 0].plot(ball_t, ball_x, "bo")
    _axs[0, 0].set_ylabel("x [m]")
    _axs[0, 1].plot(ball_t[:-1], vx, "bo", label="forward difference")
    _axs[0, 1].plot(ball_t[1:-1], vx2, "m+", markersize=10, label="central difference")
    _axs[0, 1].set_ylabel("vx [m/s]")
    _axs[0, 1].legend(frameon=False, fontsize=10, loc="upper left", numpoints=1)
    _axs[0, 2].plot(ball_t[:-2], ax, "bo")
    _axs[0, 2].plot(ball_t[2:-2], ax2, "m+", markersize=10)
    _axs[0, 2].set_ylabel("ax [m/s$^2$]")
    _axs[1, 0].plot(ball_t, ball_y, "ro")
    _axs[1, 0].set_ylabel("y [m]")
    _axs[1, 1].plot(ball_t[:-1], vy, "ro")
    _axs[1, 1].plot(ball_t[1:-1], vy2, "m+", markersize=10)
    _axs[1, 1].set_ylabel("vy [m/s]")
    _axs[1, 2].plot(ball_t[:-2], ay, "ro")
    _axs[1, 2].plot(ball_t[2:-2], ay2, "m+", markersize=10)
    _axs[1, 2].set_ylabel("ay [m/s$^2$]")
    for _ax in _axs[1]:
        _ax.set_xlabel("Time [s]")
    plt.suptitle("Kinematics of a ball toss", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The noise shows up in the derivatives, as always. Once in the air, the ball has only gravity acting on it (neglecting air resistance), so its horizontal acceleration should be zero and its vertical acceleration constant, about $-9.8$ m/s². The forward-difference accelerations scatter around those values by about 1 m/s²; the central differences, which here average over a wider interval, scatter less.

    To estimate the acceleration we could filter the data, but with 22 frames there is little room for a filter, and we know the physics of the phenomenon. So we can instead fit a *model* to the data. For the vertical position, the model is that of a particle at constant acceleration:

    $$
    y(t) = y_0 + v_0 t + \frac{1}{2} g t^2
    $$

    a second-order polynomial whose leading coefficient is $g/2$. **Before you run the next cell**, predict how close to $-9.8$ m/s² the fitted $g$ will be.
    """)
    return


@app.cell
def _(ball_t, ball_y, np):
    _p = np.polyfit(ball_t, ball_y, 2)
    print(f"g = {2 * _p[0]:.2f} m/s2 (all {ball_t.size} frames)")
    _p = np.polyfit(ball_t[:-4], ball_y[:-4], 2)
    print(f"g = {2 * _p[0]:.2f} m/s2 (without the last 4 frames)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A good estimate, from data whose point-by-point accelerations scatter by a whole m/s². Fitting a model uses all the frames at once to estimate one parameter, which is why it is so much more precise than any derivative. The price is that you must trust the model.

    Look again at the central-difference vertical acceleration: it drifts from about $-9.7$ to almost $-11$ m/s² over the last frames, which no ball in free flight does. Something is wrong with the last frames, perhaps a distortion of the lens near the edge of the video image, or the ball moving out of the plane of the camera. Leaving out the last four frames changes $g$ by about 0.1 m/s². To read more about fitting a model to data, see [Curve fitting](https://github.com/BMClab/BMC/blob/master/notebooks/CurveFitting.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The optimal cutoff frequency

    You are probably now wondering how to determine automatically the optimal cutoff frequency of a low-pass filter, the one that attenuates as much of the noise as possible without compromising the signal. This is an important topic in signal processing, particularly in movement science, and one method for it is the subject of the notebook [Residual analysis to determine the optimal cutoff frequency](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your guess in Challenge 0. By what factor does each differentiation multiply a noise component at frequency $f$? What does that mean for noise near the Nyquist frequency of a 100 Hz recording?
    2. A single-pass Butterworth filter at 6 Hz is applied to a marker trajectory but not to the EMG recorded with it. What happens to the timing between muscle activation and movement?
    3. Why must the cutoff frequency be corrected when filtering with `filtfilt`? What do you report in your methods if you forget?
    4. When would you prefer a critically damped filter to a Butterworth filter?
    5. Why can `filtfilt` not be used to filter data in real time, for instance in a biofeedback system?
    6. The ball toss was solved by fitting a model, the Pezzack data by filtering. When is each approach appropriate?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Show that the coefficients of the moving-average filter satisfy $\sum b_k - \sum_{k \geq 1} a_k = 1$. What does this property mean for a constant input?

    2. Use `signal.freqz` to plot the frequency response of moving-average filters of 5 and 11 samples, for data sampled at 100 Hz. At what frequency is the first zero of each response? Is the moving average a good low-pass filter? Compare it with a second-order Butterworth filter with its $-3$ dB point at the same frequency.

    3. A high-pass filter is the complement of a low-pass one. Remove the DC component and any slow drift of the EMG above with a second-order Butterworth high-pass filter at 20 Hz. Compare the result with simply subtracting the mean.

    4. The linear envelope of an EMG is obtained by full-wave rectifying it (taking its absolute value) and filtering it with a low-pass filter at a few hertz. Compute the linear envelope of the EMG above with a zero-phase Butterworth filter at 5 Hz, and compare it with the moving RMS. See [Electromyography](https://github.com/BMClab/BMC/blob/master/notebooks/Electromyography.ipynb).

    5. Filter the vertical position of the ball toss with a zero-phase Butterworth low-pass filter, differentiate it twice, and compare the acceleration with the $g$ estimated by the polynomial fit. Which cutoff frequency did you use, and why is filtering harder here than with the Pezzack data?

    6. Repeat the comparison of the Butterworth, Savitzky-Golay and spline methods with the noisy version of Pezzack's angle, `pz_disp_noisy`, adjusting the parameters of each method. Does the ranking of the methods change?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Residual analysis](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb) — choosing the cutoff frequency from the data.
    - [Basic properties of signals](https://github.com/BMClab/BMC/blob/master/notebooks/SignalBasicProperties.ipynb) — sampling, aliasing, quantization and the signal-to-noise ratio.
    - [Fourier transform](https://github.com/BMClab/BMC/blob/master/notebooks/FourierTransform.ipynb) — the frequency content of a signal.
    - [Curve fitting](https://github.com/BMClab/BMC/blob/master/notebooks/CurveFitting.ipynb) — fitting a model instead of filtering.
    - [Electromyography](https://github.com/BMClab/BMC/blob/master/notebooks/Electromyography.ipynb) — filtering, rectifying and enveloping EMG.
    - [dspGuru - Digital Signal Processing Central](http://www.dspguru.com/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Lyons RG (2010) [Understanding Digital Signal Processing](http://books.google.com.br/books?id=UBU7Y2tpwWUC&hl). 3rd edition. Prentice Hall.
    - Pezzack JC, Norman RW, Winter DA (1977) [An assessment of derivative determining techniques used for motion analysis](http://www.health.uottawa.ca/biomech/courses/apa7305/JB-Pezzack-Norman-Winter-1977.pdf). Journal of Biomechanics, 10, 377-382. [PubMed](http://www.ncbi.nlm.nih.gov/pubmed/893476).
    - Robertson DG, Dowling JJ (2003) [Design and responses of Butterworth and critically damped digital filters](https://www.ncbi.nlm.nih.gov/pubmed/14573371). Journal of Electromyography and Kinesiology, 13(6), 569-573.
    - Robertson G, Caldwell G, Hamill J, Kamen G (2013) [Research Methods in Biomechanics](http://books.google.com.br/books?id=gRn8AAAAQBAJ). 2nd edition. Human Kinetics.
    - Vint PF, Hinrichs RN (1996) Endpoint error in smoothing and differentiating raw kinematic data: an evaluation of four popular methods. Journal of Biomechanics, 29, 1637-1642. (The source of the Pezzack data file used here.)
    - Winter DA (2009) [Biomechanics and Motor Control of Human Movement](http://books.google.com.br/books?id=_bFHL08IWfwC). 4th edition. Hoboken, USA: Wiley.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
