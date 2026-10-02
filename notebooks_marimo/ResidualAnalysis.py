import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Residual analysis to determine the optimal cutoff frequency

    > Marcos Duarte, Renato Naville Watanabe,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Every low-pass filter needs a cutoff frequency, and the choice matters: too high and the noise passes through, too low and the signal is distorted. A common problem in signal processing is to determine automatically the cutoff frequency that attenuates as much of the noise as possible without compromising the signal. There is no definitive solution to it, but there are techniques, with different degrees of success.

    This notebook builds one of them, proposed by David Winter, step by step: first the idea, then the computation by hand, then an automatic implementation. It then does what is rarely possible in practice: it checks the answer against the truth, on data where the true acceleration was measured.

    You should know the material of [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) first, in particular the zero-phase Butterworth filter and the correction of its cutoff frequency for two passes.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and being wrong is the most useful thing that can happen to you here.

    **Challenge 0.** In [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb), Pezzack's angle was filtered at 9 Hz before being differentiated, and that value was picked by looking at the result. Suppose you had no accelerometer to compare with. Write down, in two or three sentences, how you would choose the cutoff frequency from the angle alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Winter's idea

    David Winter, in his classic book *Biomechanics and Motor Control of Human Movement*, proposed finding the cutoff frequency from a residual analysis: filter the data with a range of cutoff frequencies, and for each one compute the residual, the RMS difference between the filtered and the unfiltered signals:

    $$
    R(f_c) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \left(x_i - \hat{x}_i(f_c)\right)^2}
    $$

    where $x$ is the raw signal and $\hat{x}(f_c)$ is the signal filtered with the cutoff frequency $f_c$.

    The residual is whatever the filter removed. At high cutoff frequencies the filter removes only noise, and if the noise is spread evenly over the frequencies (white noise), the residual decreases steadily, roughly along a straight line, as the cutoff approaches the Nyquist frequency. At low cutoff frequencies the filter starts removing signal too, and the residual grows quickly. Winter's choice is the cutoff at the transition, where the residual starts to change very little, because from that point on the filter is, ideally, removing mostly noise and little signal.

    The concept is straightforward to implement. Let's do it on data where we can check the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy and SciPy do the computation, pandas reads the data file, and Matplotlib draws the plots.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pezzack's benchmark data

    In 1977, Pezzack, Norman and Winter published a paper investigating the effects of differentiation and filtering on experimental data: the angle of a bar rotated by hand, digitized from film, and its angular acceleration, measured directly with an accelerometer. Since then these data have become a benchmark for testing new algorithms (they are also available from the [ISB website](https://isbweb.org/data/pezzack/index.html)). We will consider the accelerometer's acceleration as the true one. The file has a second, noisier version of the angle, which we will use later.
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
    time, disp, disp_noisy, aacc = _data.T
    freq = 1 / np.mean(np.diff(time))  # sampling frequency [Hz]

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, sharex=True, figsize=(11, 4))
    _ax1.plot(time, disp, "b")
    _ax1.set_xlabel("Time [s]")
    _ax1.set_ylabel("Angular displacement [rad]")
    _ax2.plot(time, aacc, "g")
    _ax2.set_xlabel("Time [s]")
    _ax2.set_ylabel("Angular acceleration [rad/s$^2$]")
    plt.suptitle("Pezzack's benchmark data", fontsize=14)
    plt.tight_layout()
    plt.show()
    print(f"Sampling frequency: {freq:.2f} Hz (Nyquist frequency: {freq / 2:.2f} Hz)")
    return aacc, disp, disp_noisy, freq, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: the residuals

    Let's filter the angle with a second-order zero-phase Butterworth filter at 100 cutoff frequencies, from about 0.25 Hz to just below 20 Hz, and compute the residual for each. The function below does it; the cutoff given to `butter` is corrected for the two passes of `filtfilt`, with $C = 0.802$.
    """)
    return


@app.function
def residuals(y, freq, n=101):
    """Residual RMS between `y` and `y` filtered at `n` cutoff frequencies.

    The filter is a second-order zero-phase Butterworth low-pass filter, with
    the cutoff frequency corrected for its two passes. The cutoff frequencies
    go from 1% of the Nyquist frequency to just below 0.802 times it.

    Returns (cutoff frequencies [Hz], residuals).
    """
    import numpy as np
    from scipy.signal import butter, filtfilt

    C = 0.802  # correction for two passes: C = (2**(1/2) - 1)**0.25
    freqs = np.linspace((freq / 2) / 100, (freq / 2) * C, n, endpoint=False)
    res = np.empty(freqs.size)
    for i, fc in enumerate(freqs):
        b, a = butter(2, (fc / C) / (freq / 2))
        res[i] = np.sqrt(np.mean((filtfilt(b, a, y) - y) ** 2))
    return freqs, res


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Before you run the next cell**, sketch the residual against the cutoff frequency. Where is it largest, where is it smallest, and what does it look like in between?
    """)
    return


@app.cell
def _(disp, freq, np, plt):
    _freqs, _res = residuals(disp, freq)

    _, _ax = plt.subplots(1, 1, figsize=(9, 4))
    _ax.plot(_freqs, _res, "b.", markersize=8)
    _ax.set_xlabel("Cutoff frequency [Hz]")
    _ax.set_ylabel("Residual RMS [rad]")
    _ax.set_title("Residual analysis of Pezzack's angle")
    _ax.set_ylim(0, 0.01)
    _ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()
    print(f"Residual at {_freqs[0]:.2f} Hz: {_res[0]:.4f} rad; at {_freqs[-1]:.2f} Hz: {_res[-1]:.6f} rad")
    print(f"Residual at {_freqs[np.argmin(np.abs(_freqs - 10))]:.2f} Hz: {_res[np.argmin(np.abs(_freqs - 10))]:.4f} rad")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The plot is zoomed in: at the lowest cutoff frequencies the residual is far above the top of the axis, because there the filter removes most of the movement. Below about 5 Hz the residual falls steeply; above about 8 Hz it falls slowly and almost linearly towards zero at the highest cutoff, where the filter removes almost nothing.

    That straight part is the noise. Extrapolate it back to a cutoff of 0 Hz, and its intercept, call it $a$, estimates the residual that would be left if the filter removed only noise, all of it: approximately the RMS of the noise. Winter's choice of cutoff frequency is the one at which the residual curve reaches that value, $R(f_c) = a$. It is a compromise between the signal distortion the filter introduces and the noise it lets through.

    **Guiding questions 1.**

    1. By eye, where would you draw the straight line through the noisy part, and where does it cross the vertical axis?
    2. Where does the curve reach the height of that intercept? That is your residual-analysis cutoff frequency.
    3. The intercept is in radians. Convert it to degrees. Is that a plausible error for an angle digitized from film?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: an automatic search

    Doing this by eye is easy for one signal and impossible for a thousand. The function `optcutfreq` below automates it in three parts (after the help section):

    1. it computes the residuals over a range of cutoff frequencies, with `residuals`;
    2. it finds the noisy region of the residual curve. For that, it fits an exponential decay to the residuals and considers that the noisy, linear tail starts after three lifetimes of the decay (a 95% drop); a straight line is then fitted to the residuals from that point on, and its intercept $a$ is the noise level;
    3. it finds the cutoff frequency at which the residual curve equals $a$, by interpolating the residuals with a spline, and optionally plots the results.

    Winter should not be blamed for the automatic search. It only follows, as closely as possible, his suggestion of fitting a regression line to the noisy part of the residuals. If the search fails, the frequency limits of the noisy part can be given by hand, with the parameter `fclim`.

    The code is long relative to the simplicity of the idea, because of the help section, the automatic search, and a rich plot. Its signature is:

    ```python
    fc_opt = optcutfreq(y, freq=1, fclim=None, show=False, ax=None)
    ```

    (A version of this function is also distributed as the Python package [optcutfreq](https://github.com/demotu/optcutfreq). The copy here is self-contained, so you can read every line of it.)
    """)
    return


@app.function
def optcutfreq(y, freq=1, fclim=None, show=False, ax=None):
    """Automatic search of the optimal filter cutoff frequency by residual analysis.

    This method was proposed by Winter in his book [1]_.
    The 'optimal' cutoff frequency (in the sense that a filter with such cutoff
    frequency removes as much noise as possible without considerably affecting
    the signal) is found by performing a residual analysis of the difference
    between filtered and unfiltered signals over a range of cutoff frequencies.
    The optimal cutoff frequency is the one where the residual starts to change
    very little because it is considered that from this point, it's being
    filtered mostly noise and minimally signal, ideally.

    Parameters
    ----------
    y : 1D array_like
        Data
    freq : float, optional (default = 1)
        sampling frequency of the signal y
    fclim : list with 2 numbers, optional (default = None)
        limit frequencies of the noisy part of the residuals curve
    show : bool, optional (default = False)
        True (1) plots data in a matplotlib figure
        False (0) to not plot
    ax : array of 3 matplotlib.axes.Axes instances, optional (default = None).

    Returns
    -------
    fc_opt : float
             optimal cutoff frequency (None if not found)

    Notes
    -----
    A second-order zero-phase digital Butterworth low-pass filter is used.
    The cutoff frequency is corrected for the number of passes:
    C = (2**(1/npasses) - 1)**0.25. C = 0.802 for a dual pass filter.

    The matplotlib figure with the results will show a plot of the residual
    analysis with the optimal cutoff frequency, a plot with the unfiltered and
    filtered signals at this optimal cutoff frequency (with the RMSE of the
    difference between these two signals), and a plot with the respective
    second derivatives of these signals which should be useful to evaluate
    the quality of the optimal cutoff frequency found.

    Winter should not be blamed for the automatic search algorithm used here.
    The algorithm implemented is just to follow as close as possible Winter's
    suggestion of fitting a regression line to the noisy part of the residuals.

    This function performs well with data where the signal has frequencies
    considerably below the Nyquist frequency and the noise is predominantly
    white in the higher frequency region.

    If the automatic search fails, the lower and upper frequencies of the noisy
    part of the residuals curve can be given as a parameter (fclim).
    These frequencies can be chosen by viewing the plot of the residuals (enter
    show=True as input parameter when calling this function).

    It is known that this residual analysis algorithm results in oversmoothing
    kinematic data [2]_. Use it with moderation.
    This code is described elsewhere [3]_.

    References
    ----------
    .. [1] Winter DA (2009) Biomechanics and motor control of human movement.
    .. [2] http://www.clinicalgaitanalysis.com/faq/cutoff.html
    .. [3] https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb

    Examples
    --------
    >>> rng = np.random.default_rng(seed=42)
    >>> y = np.cumsum(rng.standard_normal(1000))
    >>> # optimal cutoff frequency based on residual analysis and plot:
    >>> fc_opt = optcutfreq(y, freq=1000, show=True)
    >>> # same analysis but specifying the frequency limits and plot:
    >>> optcutfreq(y, freq=1000, fclim=[200, 400], show=True)
    >>> # It's not always possible to find an optimal cutoff frequency
    >>> # or the one found can be wrong:
    >>> y = rng.standard_normal(100)
    >>> optcutfreq(y, freq=100, show=True)
    """
    import numpy as np
    from scipy.interpolate import UnivariateSpline

    freqs, res = residuals(y, freq)

    # find the noisy part of the residuals by fitting an exponential curve
    # y = A*exp(B*x)+C to the residual data and consider that the tail part
    # of the exponential (which should be the noisy part of the residuals)
    # starts after 3 lifetimes (exp(-3), 95% drop)
    if fclim is None or len(fclim) == 0:
        fc1 = 0
        fc2 = int(0.95 * (len(freqs) - 1))
        # log of exponential turns the problem into a first-order polynomial fit
        # make the data always greater than zero before taking the logarithm
        reslog = np.log(np.abs(res[fc1 : fc2 + 1] - res[fc2]) + 1000 * np.finfo(float).eps)
        Blog, Alog = np.polyfit(freqs[fc1 : fc2 + 1], reslog, 1)
        fcini = np.nonzero(freqs >= -3 / Blog)[0]  # 3 lifetimes
        lims = [fcini[0], fc2] if fcini.size else []
    else:
        lims = [np.nonzero(freqs >= fclim[0])[0][0], np.nonzero(freqs >= fclim[1])[0][0]]

    # find fc_opt with linear fit y=A+Bx of the noisy part of the residuals
    B = A = fc_opt = None
    if len(lims) and lims[0] < lims[1]:
        B, A = np.polyfit(freqs[lims[0] : lims[1]], res[lims[0] : lims[1]], 1)
        # optimal cutoff frequency is the frequency where y[fc_opt] = A
        roots = UnivariateSpline(freqs, res - A, s=0).roots()
        fc_opt = float(roots[0]) if len(roots) else None

    if show:
        plot_optcutfreq(y, freq, freqs, res, lims, fc_opt, B, A, ax)

    return fc_opt


@app.function
def plot_optcutfreq(y, freq, freqs, res, lims, fc_opt, B, A, ax=None):
    """Plot the results of the optcutfreq function, see its help."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import butter, filtfilt

    if ax is None:
        plt.figure(figsize=(11, 5))
        ax = np.array([plt.subplot(121), plt.subplot(222), plt.subplot(224)])

    ax[0].plot(freqs, res, "b.", markersize=9)
    time = np.linspace(0, len(y) / freq, len(y))
    ax[1].plot(time, y, "g", linewidth=1, label="Unfiltered")
    ydd = np.diff(y, n=2) * freq**2
    ax[2].plot(time[:-2], ydd, "g", linewidth=1, label="Unfiltered")
    if fc_opt:
        ylin = np.poly1d([B, A])(freqs)
        ax[0].plot(freqs, ylin, "r--", linewidth=2)
        ax[0].plot(freqs[lims[0]], res[lims[0]], "r>", freqs[lims[1]], res[lims[1]], "r<", ms=9)
        ax[0].set_ylim(bottom=0, top=4 * A)
        ax[0].plot([0, freqs[-1]], [A, A], "r-", linewidth=2)
        ax[0].plot([fc_opt, fc_opt], [0, A], "r-", linewidth=2)
        ax[0].plot(
            fc_opt, 0, "ro", markersize=7, clip_on=False, zorder=9,
            label=f"$Fc_{{opt}}$ = {fc_opt:.1f} Hz",
        )
        ax[0].legend(loc="best", numpoints=1, framealpha=0.5)
        # correct the cutoff frequency for the number of passes
        C = 0.802  # for dual pass; C = (2**(1/npasses) - 1)**0.25
        b, a = butter(2, (fc_opt / C) / (freq / 2))
        yf = filtfilt(b, a, y)
        ax[1].plot(time, yf, color=[1, 0, 0, 0.5], linewidth=2, label="Opt. filtered")
        ax[1].legend(loc="best", framealpha=0.5)
        ax[1].set_title(f"Signals (RMSE = {A:.3g})")
        yfdd = np.diff(yf, n=2) * freq**2
        ax[2].plot(time[:-2], yfdd, color=[1, 0, 0, 0.5], linewidth=2, label="Opt. filtered")
        ax[2].legend(loc="best", framealpha=0.5)
        resdd = np.sqrt(np.mean((yfdd - ydd) ** 2))
        ax[2].set_title(f"Second derivatives (RMSE = {resdd:.3g})")
    else:
        ax[0].text(
            0.5, 0.5, "Unable to find optimal cutoff frequency",
            horizontalalignment="center", color="r", zorder=9,
            transform=ax[0].transAxes, fontsize=12,
        )
        ax[1].set_title("Signal")
        ax[2].set_title("Second derivative")

    ax[0].set_xlabel("Cutoff frequency [Hz]")
    ax[0].set_ylabel("Residual RMSE")
    ax[0].set_title("Residual analysis")
    ax[0].grid()
    ax[1].set_xlim(0, time[-1])
    ax[1].grid()
    ax[2].set_xlabel("Time [s]")
    ax[2].set_xlim(0, time[-1])
    ax[2].grid()
    plt.tight_layout()
    plt.show()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: the optimal cutoff frequency of Pezzack's angle

    **Before you run the next cell**, compare your answer to Guiding question 1.2 with the 9 Hz used in [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb). Will the automatic search give a higher or a lower cutoff frequency?
    """)
    return


@app.cell
def _(disp, freq):
    fc_opt = optcutfreq(disp, freq=freq, show=True)
    print(f"Optimal cutoff frequency: {fc_opt:.2f} Hz")
    return (fc_opt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The optimal cutoff frequency found is 5.6 Hz, lower than the 9 Hz chosen by hand. The noise level $a$, the RMSE shown in the title of the top right panel, is about 0.0018 rad, a tenth of a degree.

    Note that the filtering matters only for the derivatives: in the top right panel the unfiltered and filtered angles cannot be told apart. In the bottom right panel, their second derivatives can.

    Let's filter the angle at this cutoff frequency, differentiate it twice, and compare the result with the true acceleration, as in [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb). The function below does the filtering and the double differentiation; its output is two samples shorter than its input.
    """)
    return


@app.function
def filtered_acceleration(y, freq, fc):
    """Second derivative of `y` after a zero-phase Butterworth filter at `fc` Hz.

    The filter is second order, applied forward and backward with its cutoff
    corrected for the two passes, and the derivative is the second-order
    finite difference, so the output is two samples shorter than `y`.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt

    C = 0.802  # correction for two passes
    b, a = butter(2, (fc / C) / (freq / 2))
    return np.diff(filtfilt(b, a, y), 2) * freq**2


@app.cell
def _(aacc, disp, fc_opt, freq, np, plt, time):
    _aacc_bw = filtered_acceleration(disp, freq, fc_opt)
    _rmse = np.sqrt(np.mean((_aacc_bw - aacc[1:-1]) ** 2))

    _, _ax = plt.subplots(1, 1, figsize=(11, 4))
    _ax.plot(time[1:-1], aacc[1:-1], "g", linewidth=3, label="Accelerometer (true value)")
    _ax.plot(time[1:-1], _aacc_bw, "r", label=f"Butterworth {fc_opt:.3g} Hz: RMSE = {_rmse:.2f}")
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
    An RMSE of about 4.2 rad/s², slightly better than the 4.45 rad/s² of the Butterworth filter at 9 Hz in [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb), and found without looking at the true acceleration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is the optimal cutoff frequency optimal?

    "Optimal" here means optimal by Winter's criterion, a compromise judged on the angle. What we actually care about is the acceleration. Because we know the true acceleration, we can do what is impossible in a real experiment: try every cutoff frequency and see which one gives the acceleration closest to the truth.

    It is known that this residual analysis tends to oversmooth kinematic data (see [this discussion](http://www.clinicalgaitanalysis.com/faq/cutoff.html)), that is, to choose a cutoff frequency that is too low. **Before you run the next cell**, predict where the best cutoff frequency for the acceleration will be: below 5.6 Hz, above it, or above 9 Hz?
    """)
    return


@app.cell
def _(aacc, disp, fc_opt, freq, np, plt):
    fc_sweep = np.arange(2, 18.01, 0.25)


    def rmse_sweep(y):
        """RMSE of the acceleration of `y` filtered at each of `fc_sweep`."""
        return np.array(
            [np.sqrt(np.mean((filtered_acceleration(y, freq, _fc) - aacc[1:-1]) ** 2)) for _fc in fc_sweep]
        )


    _rmse = rmse_sweep(disp)
    _best = fc_sweep[np.argmin(_rmse)]

    _, _ax = plt.subplots(1, 1, figsize=(9, 4))
    _ax.plot(fc_sweep, _rmse, "b.-")
    _ax.axvline(fc_opt, color="r", linestyle="--", label=f"residual analysis: {fc_opt:.1f} Hz")
    _ax.axvline(_best, color="g", linestyle="--", label=f"best for the acceleration: {_best:.1f} Hz")
    _ax.set_xlabel("Cutoff frequency [Hz]")
    _ax.set_ylabel("RMSE of the acceleration [rad/s$^2$]")
    _ax.set_ylim(0, 12)
    _ax.legend(frameon=False, loc="upper center")
    plt.tight_layout()
    plt.show()
    for _fc in (fc_opt, _best, 9):
        print(f"Cutoff {_fc:5.2f} Hz: RMSE = {np.sqrt(np.mean((filtered_acceleration(disp, freq, _fc) - aacc[1:-1]) ** 2)):.2f} rad/s2")
    return fc_sweep, rmse_sweep


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The best cutoff frequency for the acceleration is about 6.5 Hz, a little above the 5.6 Hz of the residual analysis: it did oversmooth, as expected. But look at the shape of the curve. It is flat around its minimum, and the price of the oversmoothing is tiny, an RMSE of 4.22 instead of 4.16 rad/s². Cutoff frequencies that are too low are punished hard (at 2 Hz the error is more than twice the minimum); too high, the error grows slowly.

    **Guiding questions 2.**

    1. Why is the curve so asymmetric? Think of what is removed below and above the best cutoff frequency.
    2. The residual analysis judged the cutoff on the angle, but we evaluated it on the acceleration. Would the best cutoff frequency for the angular *velocity* be higher or lower than for the acceleration?
    3. In a real experiment you cannot draw this curve. What does its flatness near the minimum tell you about how precise a cutoff frequency needs to be?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Noisier data

    The data file has a second version of the angle with more noise added to it. A fixed cutoff frequency ignores the noise level; the residual analysis does not. **Before you run the next cell**, predict whether the optimal cutoff frequency for the noisier angle will be higher or lower than 5.6 Hz, and whether the 9 Hz chosen by hand for the clean angle will still work.
    """)
    return


@app.cell
def _(aacc, disp_noisy, fc_sweep, freq, np, rmse_sweep):
    _fc_opt = optcutfreq(disp_noisy, freq=freq, show=True)
    _rmse = rmse_sweep(disp_noisy)
    print(f"Optimal cutoff frequency, noisy angle: {_fc_opt:.2f} Hz")
    print(f"Best cutoff for the acceleration:      {fc_sweep[np.argmin(_rmse)]:.2f} Hz")
    for _fc in (_fc_opt, fc_sweep[np.argmin(_rmse)], 9):
        print(f"Cutoff {_fc:5.2f} Hz: RMSE = {np.sqrt(np.mean((filtered_acceleration(disp_noisy, freq, _fc) - aacc[1:-1]) ** 2)):.2f} rad/s2")
    print(f"No filter at all:     RMSE = {np.sqrt(np.mean((np.diff(disp_noisy, 2) * freq**2 - aacc[1:-1]) ** 2)):.2f} rad/s2")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With more noise, the residual analysis lowers its cutoff to about 3.9 Hz, and its noise level $a$ rises to about 0.007 rad, four times higher. The best cutoff for the acceleration is now 5 Hz, so the residual analysis oversmooths more than before, at a cost of 5.0 instead of 4.5 rad/s². But the 9 Hz that worked well on the clean angle now gives almost twice that error, and no filter at all gives 37 rad/s².

    That is the real value of the method. It is not that it finds *the* optimal cutoff; it is that it adapts the cutoff to the noise in each recording, with a bias towards smoothing too much, which, as the previous section showed, is the cheaper mistake.

    **Challenge 1.** The residual analysis assumes that the noise is white and that the signal has frequencies well below the Nyquist frequency.

    1. Run `optcutfreq(rng.standard_normal(100), freq=100, show=True)`, with `rng = np.random.default_rng(seed=42)`: a signal that is only noise. What happens, and why?
    2. Run it on a random walk, `np.cumsum(rng.standard_normal(1000))`, sampled at 1000 Hz. Is the result plausible? Look at the residual plot and choose the limits of the noisy region yourself with `fclim`.
    3. Add a 60 Hz sinusoid to the angle (power-line interference is not white noise) and run the residual analysis again. Does the method notice?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your answer to Challenge 0. Which parts of it are in Winter's method, and which are not?
    2. What does the residual measure, and why does it fall to zero at the Nyquist frequency?
    3. Why does the intercept of the straight part of the residual curve estimate the RMS of the noise? Which assumption about the noise does that require?
    4. The residual analysis chooses the cutoff from the angle, but you will use the acceleration. Why might the best cutoff for the two be different?
    5. Two recordings of the same movement, one noisier than the other, get different cutoff frequencies from the residual analysis. Is that a problem when you compare them? What would you report?
    6. A colleague uses 6 Hz for every gait study "because it is standard". Give one argument for and one against.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Generate a known signal, such as $x(t) = \sin(2\pi t) + 0.3\sin(6\pi t)$ sampled at 100 Hz for 2 s, and add white noise with standard deviations of 0.01 and 0.05 (use a fixed seed). For each:<br>
       a. Find the optimal cutoff frequency with `optcutfreq`, and compare the noise level $a$ it estimates with the true standard deviation of the noise.<br>
       b. Find the cutoff that minimizes the RMSE of the filtered signal relative to the noise-free one, and the one that minimizes the RMSE of its second derivative. How do they compare with the residual analysis?

    2. Repeat the analysis of this notebook with a critically damped filter instead of a Butterworth filter (see `critic_damp` in [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb)). Does the optimal cutoff frequency change?

    3. Apply the residual analysis to each coordinate of the markers in Winter's Table A.1, used in [Angular kinematics in a plane](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb). Are the optimal cutoff frequencies similar for all markers and coordinates? Which would you use, and why?

    4. Read the [discussion about the cutoff frequency](http://www.clinicalgaitanalysis.com/faq/cutoff.html) in gait analysis and summarize the arguments against residual analysis in a paragraph. Do the results of this notebook support them?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) — the filters used here, and other ways of dealing with noise.
    - [Basic properties of signals](https://github.com/BMClab/BMC/blob/master/notebooks/SignalBasicProperties.ipynb) — noise, the signal-to-noise ratio and the Nyquist frequency.

    To read more about the determination of the optimal cutoff frequency, see the following papers:

    - Pezzack JC, Norman RW, Winter DA (1977) An assessment of derivative determining techniques used for motion analysis. Journal of Biomechanics, 10, 377-382.
    - Giakas G, Baltzopoulos V (1997) A comparison of automatic filtering techniques applied to biomechanical walking data. Journal of Biomechanics, 30, 847-850.
    - Alonso FJ, Salgado DR, Cuadrado J, Pintado P (2009) [Automatic smoothing of raw kinematic signals using SSA and cluster analysis](https://lim.ii.udc.es/docs/proceedings/2009_09_EUROMECH_Automatic.pdf). 7th EUROMECH Solid Mechanics Conference.
    - Kristianslund E, Krosshaug T, van den Bogert AJ (2012) Effect of low pass filtering on joint moments from inverse dynamics: Implications for injury prevention. Journal of Biomechanics, 45, 666-671.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Pezzack JC, Norman RW, Winter DA (1977) An assessment of derivative determining techniques used for motion analysis. Journal of Biomechanics, 10, 377-382. [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/893476).
    - Vint PF, Hinrichs RN (1996) Endpoint error in smoothing and differentiating raw kinematic data: an evaluation of four popular methods. Journal of Biomechanics, 29, 1637-1642. (The source of the Pezzack data file used here, with its noisier version of the angle.)
    - Winter DA (2009) [Biomechanics and Motor Control of Human Movement](https://books.google.com.br/books?id=_bFHL08IWfwC). 4th edition. Hoboken, USA: Wiley.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
