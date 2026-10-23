# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14", "optcutfreq>=0.0.10", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Residual analysis to determine the optimal cutoff frequency

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Every low-pass filter needs a cutoff frequency, and the choice matters: too high and the noise passes through, too low and the signal is distorted. A common problem in signal processing is to determine automatically the cutoff frequency that attenuates as much of the noise as possible without compromising the signal. There is no definitive solution to it, but there are techniques, with different degrees of success.

    This notebook builds one of them, proposed by David Winter, step by step: first the idea, then the computation by hand, then an automatic implementation. It then does what is rarely possible in practice: it checks the answer against an independently measured acceleration reference. That measurement is a benchmark, not an uncertainty-free truth.

    You should know the material of [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py) first, in particular the zero-phase Butterworth filter and the correction of its cutoff frequency for two passes.

    Read it in order and run each cell as you reach it. Try the distributed Exercises before opening their worked solutions. Use the collapsed final answers to check the independent Problems at the end. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and being wrong is the most useful thing that can happen to you here.

    **Challenge 0.** In [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py), Pezzack's angle was filtered at 9 Hz before being differentiated, and that value was picked by looking at the result. Suppose you had no accelerometer to compare with. Write down, in two or three sentences, how you would choose the cutoff frequency from the angle alone.
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

    The residual is whatever the filter removed. If the movement is concentrated at low frequencies, a sufficiently high cutoff may remove mostly noise. With broadband noise, a segment of the residual curve can look approximately straight; that straightness is an empirical local approximation, not a mathematical consequence of whiteness. At low cutoff frequencies the filter starts removing signal too, and the residual grows quickly. Winter's choice is the cutoff at the transition, where the residual starts to change very little, because from that point on the filter is, ideally, removing mostly noise and little signal.

    The concept is straightforward to implement. Let's do it on data where we can check the answer.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A filter removes $[0.002,-0.002,0.002,-0.002]$ radians from four samples. Compute the residual. Does it establish that the removed component was noise?

    /// details | Solution to Exercise 1
    The mean squared difference is $4\times10^{-6}$ rad², giving $R=0.002$ rad. The residual measures the size of what was removed, not its origin. A real movement component of the same amplitude gives exactly the same residual.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy and SciPy do the computation, pandas reads the data file, and Matplotlib draws the plots. `optcutfreq` implements the automatic residual criterion. All five packages are declared in this notebook’s dependency metadata. Open the file in marimo; the Pezzack benchmark requires access to the repository data.
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

    In 1977, Pezzack, Norman and Winter published a paper investigating the effects of differentiation and filtering on experimental data: the angle of a bar rotated by hand, digitized from film, and its angular acceleration, measured directly with an accelerometer. Since then these data have become a benchmark for testing new algorithms (they are also available from the [ISB website](https://isbweb.org/data/pezzack/index.html)). We will use the accelerometer's acceleration as an independent reference, recognizing its own measurement uncertainty. The file has a second, noisier version of the angle, which we will use later.
    """)
    return


@app.cell
def _(np, pd, plt):
    _data = pd.read_csv(
        repo_file('data/Pezzack.txt'),
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

    Let's filter the angle with a second-order zero-phase Butterworth filter at 100 cutoff frequencies, from about 0.25 Hz to just below 20 Hz, and compute the residual for each. The function below does it; the cutoff given to `butter` is corrected for the two passes of `filtfilt`, with $C = 0.802$. This is the historical low-frequency approximation used by `optcutfreq`; we retain it here to reproduce that package’s criterion. Its reported cutoffs are nominal values under this convention. The exact digital correction, explained in [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py), is $f_{design}=(f_s/\pi)\arctan[\tan(\pi f_c/f_s)/C]$. The two conventions increasingly differ toward Nyquist, so report which one you use.
    """)
    return


@app.function
def residuals(y, freq, n=101):
    """Residual RMS between `y` and `y` filtered at `n` cutoff frequencies.

    The filter is a second-order zero-phase Butterworth low-pass filter, with
    the historical approximate cutoff correction used by optcutfreq. The cutoff frequencies
    go from 1% of the Nyquist frequency to just below 0.802 times it.

    Returns (cutoff frequencies [Hz], residuals).
    """
    import numpy as np
    from scipy.signal import butter, filtfilt

    C = 0.802  # historical approximate two-pass correction: C = (2**(1/2) - 1)**0.25
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

    Interpret that approximately straight tail as mostly removed noise, subject to the bandwidth assumptions. Extrapolate it back to a cutoff of 0 Hz, and its intercept, call it $a$, estimates the residual that would be left if the filter removed only noise, all of it: approximately the RMS of the noise. Winter's choice of cutoff frequency is the one at which the residual curve reaches that value, $R(f_c) = a$. It is a compromise between the signal distortion the filter introduces and the noise it lets through.

    **Guiding questions 1.**

    1. By eye, where would you draw the straight line through the noisy part, and where does it cross the vertical axis?
    2. Where does the curve reach the height of that intercept? That is your residual-analysis cutoff frequency.
    3. The intercept is in radians. Convert it to degrees. Is that a plausible error for an angle digitized from film?
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Answer Guiding questions 1: use the nearly straight tail around 8–18 Hz to estimate the noise level and the crossing. Convert the noise estimate to degrees.

    /// details | Solution to Exercise 2
    The automatic fit gives $a\approx0.001829$ rad and a crossing near 5.59 Hz. Multiplying by $180/\pi$ gives about $0.105^\circ$. A visual fit can differ modestly. Such a small angular uncertainty is plausible for digitized film, but plausibility does not independently verify the noise model.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: an automatic search

    Doing this by eye is easy for one signal and impossible for a thousand. The function `optcutfreq`, from the Python package [optcutfreq](https://pypi.org/project/optcutfreq/) (included in the notebook dependencies), automates it in three parts:

    1. it computes the residuals over a range of cutoff frequencies, as `residuals` does above;
    2. it finds the noisy region of the residual curve. For that, it fits an exponential decay to the residuals and considers that the noisy, linear tail starts after three lifetimes of the decay (a 95% drop); a straight line is then fitted to the residuals from that point on, and its intercept $a$ is the noise level;
    3. it finds the cutoff frequency at which the residual curve equals $a$, by interpolating the residuals with a spline, and optionally plots the results.

    Winter should not be blamed for the automatic search. It only follows, as closely as possible, his suggestion of fitting a regression line to the noisy part of the residuals. If the search fails, the frequency limits of the noisy part can be given by hand, with the parameter `fclim`.

    Its signature is:

    ```python
    fc_opt = optcutfreq(y, freq=1, fclim=None, show=False, ax=None)
    ```

    Run `help(optcutfreq)` to read its documentation, or read its code on [GitHub](https://github.com/demotu/optcutfreq). The code is long relative to the simplicity of the idea, because of the help section, the automatic search, and a rich plot.
    """)
    return


@app.cell
def _():
    from optcutfreq import optcutfreq

    return (optcutfreq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: the optimal cutoff frequency of Pezzack's angle

    **Before you run the next cell**, compare your answer to Guiding question 1.2 with the 9 Hz used in [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py). Will the automatic search give a higher or a lower cutoff frequency?
    """)
    return


@app.cell
def _(disp, freq, optcutfreq):
    fc_opt = optcutfreq(disp, freq=freq, show=True)
    print(f"Optimal cutoff frequency: {fc_opt:.2f} Hz")
    return (fc_opt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The optimal cutoff frequency found is 5.6 Hz, lower than the 9 Hz chosen by hand. The noise level $a$, the RMSE shown in the title of the top right panel, is about 0.0018 rad, a tenth of a degree.

    Note that the filtering matters only for the derivatives: in the top right panel the unfiltered and filtered angles cannot be told apart. In the bottom right panel, their second derivatives can.

    Let's filter the angle at this cutoff frequency, differentiate it twice, and compare the result with the reference acceleration, as in [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py). The function below does the filtering and the double differentiation; its output is two samples shorter than its input.
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

    C = 0.802  # historical approximate two-pass correction
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
    An RMSE of about 4.2 rad/s², slightly better than 4.45 rad/s² at a nominal 9 Hz under the same historical correction convention, and found without looking at the reference acceleration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is the optimal cutoff frequency optimal?

    "Optimal" here means optimal by Winter's criterion, a compromise judged on the angle. What we actually care about is the acceleration. Because we have a reference acceleration, we can sweep a grid of cutoffs and compare the estimated accelerations with that reference. This check is unavailable when the experiment records only position.

    The residual criterion can oversmooth for a derivative-based task, but that is not guaranteed for every signal. The measured benchmark lets us check rather than assume the direction of the difference. **Before you run the next cell**, predict where the best cutoff frequency for the acceleration will be: below 5.6 Hz, above it, or above 9 Hz?
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
    The best cutoff frequency for the acceleration is about 6.5 Hz, a little above the 5.6 Hz of the residual analysis: it oversmoothed slightly for this particular acceleration benchmark. But look at the shape of the curve. It is flat around its minimum, and the price of the oversmoothing is tiny, an RMSE of 4.22 instead of 4.16 rad/s². Cutoff frequencies that are too low are punished hard (at 2 Hz the error is more than twice the minimum); too high, the error grows slowly.

    **Guiding questions 2.**

    1. Why is the curve so asymmetric? Think of what is removed below and above the best cutoff frequency.
    2. The residual analysis judged the cutoff on the angle, but we evaluated it on the acceleration. Would the best cutoff frequency for the angular *velocity* be higher or lower than for the acceleration?
    3. In a real experiment you cannot draw this curve. What does its flatness near the minimum tell you about how precise a cutoff frequency needs to be?
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** The residual criterion selects 5.59 Hz, while the acceleration RMSE minimum is 6.5 Hz. Is either answer “wrong”? Would a velocity optimum necessarily lie between them?

    /// details | Solution to Exercise 3
    They optimize different criteria: one uses the angle’s residual curve; the other uses a measured acceleration reference. The latter has RMSE about 4.16 rather than 4.22 rad/s², a small difference here. Differentiation changes the weighting of noise and signal across frequency. There is no general theorem that the velocity optimum lies between those two numbers; calculate it if a suitable reference is available.
    ///
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
def _(aacc, disp_noisy, fc_sweep, freq, np, optcutfreq, rmse_sweep):
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

    That is the real value of the method. It is not that it finds *the* optimal cutoff; it is that it adapts the cutoff to the noise in each recording, without guaranteeing an optimum for the quantity you ultimately analyze. For this benchmark, mild oversmoothing costs little near the RMSE minimum; rapid events or other noise spectra can behave differently.

    **Challenge 1.** The residual analysis assumes that the noise is white and that the signal has frequencies well below the Nyquist frequency.

    1. Run `optcutfreq(rng.standard_normal(100), freq=100, show=True)`, with `rng = np.random.default_rng(seed=42)`: a signal that is only noise. What happens, and why?
    2. Run it on a random walk, `np.cumsum(rng.standard_normal(1000))`, sampled at 1000 Hz. Is the result plausible? Look at the residual plot and choose the limits of the noisy region yourself with `fclim`.
    3. Add a 60 Hz sinusoid evaluated at the angle’s sample times and run the residual analysis again. First compute its alias at this approximately 49.75 Hz sampling rate. Does the residual plot diagnose a narrowband contaminant uniquely?
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** Answer Challenge 1: what happens for pure noise, a random walk, and 60 Hz interference sampled with Pezzack’s time vector?

    /// details | Solution to Exercise 4
    For the specified 100-sample white-noise record, `optcutfreq` returns `None`: it cannot identify a suitable signal/noise transition. A random walk has strongly colored low-frequency content; obtaining a numerical crossing or choosing `fclim` manually does not establish a physically meaningful signal bandwidth. At $f_s\approx49.7512$ Hz, 60 Hz aliases to $|60-f_s|\approx10.2488$ Hz. A residual curve may change, but cannot label this uniquely as interference or undo aliasing.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your answer to Challenge 0. Which parts of it are in Winter's method, and which are not?
    2. What does the residual measure, and why does it approach zero as the filter tends to the identity? (A digital Butterworth cutoff cannot be set exactly at Nyquist.)
    3. Why does the intercept of the straight part of the residual curve estimate the RMS of the noise? Which assumption about the noise does that require?
    4. The residual analysis chooses the cutoff from the angle, but you will use the acceleration. Why might the best cutoff for the two be different?
    5. Two recordings of the same movement, one noisier than the other, get different cutoff frequencies from the residual analysis. Is that a problem when you compare them? What would you report?
    6. A colleague uses 6 Hz for every gait study "because it is standard". Give one argument for and one against.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 5.** Two trials have residual cutoffs 4 and 6 Hz. How would you compare their peak accelerations responsibly?

    /// details | Solution to Exercise 5
    First inspect the residual tails and document the filter family, pass count, cutoff convention and boundary handling. Since different responses can change peaks differently, repeat the comparison with defensible common cutoffs and assess sensitivity. Trial-specific filtering is not automatically invalid, but a peak difference that disappears under reasonable processing choices needs qualified interpretation.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Checkpoint answers
    1. Winter’s method supplies a residual/noise criterion; it does not supply a task-specific model, prove white noise or optimize every derivative.
    2. RMS of raw minus filtered data. It tends to zero when filtering changes nothing; the nominal grid endpoint in this implementation is about $0.802f_s/2$, not $f_s/2$.
    3. The tail extrapolation is an empirical estimate under a low-bandwidth movement/broadband-noise separation. Correlated noise or a poor tail fit can invalidate it.
    4. Differentiation weights high frequencies more strongly, changing the relevant distortion/noise trade-off.
    5. Different noise levels can justify different cutoffs, but the resulting comparisons need sensitivity checks and a report of the per-trial choices and method.
    6. A fixed value makes processing consistent and may be validated for a particular protocol. It cannot automatically accommodate different noise levels, movement bandwidths or outcomes.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Generate a known signal, such as $x(t) = \sin(2\pi t) + 0.3\sin(6\pi t)$ sampled at 100 Hz for 2 s, and add white noise with standard deviations of 0.01 and 0.05 (use `t=np.arange(200)/100` and reset `rng=np.random.default_rng(42)` for each noise level). Use the historical approximate correction above. For each:<br>
       a. Find the optimal cutoff frequency with `optcutfreq`, and compare the noise level $a$ it estimates with the true standard deviation of the noise.<br>
       b. Find the cutoff that minimizes the RMSE of the filtered signal relative to the noise-free one, and the one that minimizes the RMSE of its second derivative. Use cutoffs from 1 to 39.5 Hz in 0.1 Hz steps, all samples for signal RMSE, and the interior samples for the central second difference compared with the analytic second derivative. How do they compare with the residual analysis?

    2. Repeat the analysis of this notebook with a critically damped filter instead of a Butterworth filter (see `critic_damp` in [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py)). To isolate filter shape, use the exact digital correction for both filter families, the 101-point frequency grid from `residuals`, and fit the same tail interval $8\le f_c<18$ Hz. Does the residual-analysis cutoff frequency change?

    3. Apply the residual analysis to each coordinate of the markers in Winter's Table A.1, used in [Angular kinematics in a plane](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py). Are the optimal cutoff frequencies similar for all markers and coordinates? Which would you use, and why?

    4. Write a paragraph assessing residual analysis using this notebook’s assumptions and benchmark results: discuss tail selection, colored noise, differentiation and endpoint effects. Does the evidence support using the returned value without inspection? An optional historical gait-analysis discussion is linked under Go deeper; this problem does not depend on its availability.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers
    1. With the specified seed, grids and historical `optcutfreq` convention:

    | Noise SD | Residual cutoff | Estimated $a$ | Minimum signal-RMSE cutoff | Minimum acceleration-RMSE cutoff |
    |---|---:|---:|---:|---:|
    | 0.01 | 6.2781 Hz | 0.008895 | 8.1 Hz | 5.7 Hz |
    | 0.05 | 4.0501 Hz | 0.047831 | 5.8 Hz | 4.3 Hz |

    The minimum signal RMSEs are approximately 0.003882 and 0.014889; the minimum second-derivative RMSEs are 4.8043 and 12.5421 in the signal’s units per second squared. The residual estimate is a distinct criterion; none of these cutoffs is universally optimal.

    2. With the stated exact digital correction and common 8–18 Hz tail fit, the cleaner Pezzack angle gives approximately 5.9842 Hz and $a=0.0016992$ rad for Butterworth, versus 7.0070 Hz and $a=0.0040652$ rad for the critically damped filter. The cutoff and inferred noise estimate depend on filter shape and tail choice. These values intentionally differ from the package’s automatic historical-convention result.

    3. At the sampling rate inferred from the timestamps, about 69.9534 Hz, `optcutfreq` gives the following nominal cutoffs in Hz:

    | Marker | x | y |
    |---|---:|---:|
    | Rib cage | 3.64 | 2.97 |
    | Hip | 5.17 | 2.87 |
    | Knee | 6.28 | 3.43 |
    | Fibula | 5.91 | 3.08 |
    | Ankle | 6.55 | 4.28 |
    | Heel | 5.56 | 5.35 |
    | MT5 | 5.71 | 5.64 |
    | Toe | 7.28 | 5.44 |

    The values are not uniform. There is no unique justified common cutoff from this table alone. A defensible answer inspects the residual fits, identifies the movement features to retain, and reports the sensitivity of the intended outcome to common versus coordinate-specific filtering.

    4. A suitable conclusion is: residual analysis is a useful diagnostic, but the tail may not isolate noise; colored noise, endpoint behavior and differentiation can alter the appropriate cutoff. The Pezzack examples show modest disagreement with an acceleration-based optimum and increased disagreement after adding noise. They support inspecting the curve and testing sensitivity, not accepting every returned frequency automatically.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py) — the filters used here, and other ways of dealing with noise.
    - [Basic properties of signals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/signal_basic_properties.py) — noise, the signal-to-noise ratio and the Nyquist frequency.

    The [historical clinical gait-analysis discussion](https://www.clinicalgaitanalysis.com/faq/cutoff.html) is optional background; the self-contained critique in Problem 4 uses the evidence in this guide.

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
    ### Video lectures on the Internet

    - MIT OpenCourseWare, Alan Oppenheim: [Digital Butterworth Filters](https://ocw.mit.edu/courses/res-6-008-digital-signal-processing-spring-2011/resources/lecture-16-digital-butterworth-filters/). Use the frequency-response explanation to understand why a residual curve depends on the chosen filter family.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Pezzack JC, Norman RW, Winter DA (1977) An assessment of derivative determining techniques used for motion analysis. Journal of Biomechanics, 10, 377-382. [PubMed](https://pubmed.ncbi.nlm.nih.gov/893476/).

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
