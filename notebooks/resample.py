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
    # Resample a sampled signal

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A force signal is sampled at 100 Hz and you want a 1,000 Hz time grid. Will ten times as many samples reveal ten times as much information?

    We will reproduce the original DataFrame example, inspect the time axis, and distinguish upsampling from new measurement.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The examples are self-contained.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The included helper uses SciPy polyphase resampling. NumPy generates a reproducible demonstration and pandas labels the time axis.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


@app.function
def resample(y, freq_new, freq_old=None, limit=1000, method='resample_poly'):
    """
    Resample data using signal.resample_poly scipy function.

    Parameters
    ----------
    y : numpy array or pandas dataframe column-oriented
        All columns of `y` will be resampled.
    freq_new : number
        Desired new sampling frequency of `y2` (in Hz).
    freq_old : number, optional (default = None)
        Original sampling frequency of `y` (in Hz). If no sampling frequency is
        informed, `y` is assumed to be a pandas dataframe and its sampling
        frequency will be calculated from its index column.
    limit : integer, optional (default = 1000)
        Positive integer limiting the denominator in the calculation of a fraction of
        integers to represent the ratio between up- and down- sampling
        frequencies (freq_new/freq_old) in the resample (for an irrational
        ratio, this number might affect the accuracy of the resampling but
        it will be faster).
    method : string, optional (default = 'resample_poly')
        Method used for resampling data.
        The resample_poly scipy function is the only one implemented so far.

    Returns
    -------
    y2 : numpy array or pandas dataframe
        Resampled data. Same type (numpy array or pandas dataframe) and same
        number of columns as `y`. If `y` is a pandas dataframe, `y2` will have
        its time origin preserved and its index sampled at the effective rational
        approximation to `freq_new` Hz. Boundary samples use zero-padding.
        Upsampling creates interpolated samples, not new measured information.

    """
    import numpy as np
    import pandas as pd
    from scipy import signal
    from fractions import Fraction
    if method != 'resample_poly':
        raise ValueError("Only method='resample_poly' is supported")
    is_frame = isinstance(y, pd.DataFrame)
    origin = float(y.index[0]) if is_frame and len(y) else 0.0
    if freq_old is None:
        if not is_frame or len(y) < 2:
            raise ValueError('Supply freq_old or a DataFrame with a numeric time index')
        intervals = np.diff(np.asarray(y.index, dtype=float))
        if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=1e-05, atol=1e-10):
            raise ValueError('Time index must be increasing and uniformly sampled')
        freq_old = 1 / intervals.mean()
    if not np.isfinite(freq_old) or not np.isfinite(freq_new) or min(freq_old, freq_new) <= 0:
        raise ValueError('Sampling frequencies must be finite and positive')
    data = np.asarray(y)
    if data.ndim not in (1, 2) or not len(data) or (not np.isfinite(data).all()):
        raise ValueError('Data must be a nonempty finite one- or two-dimensional array')
    fr = Fraction(float(freq_new / freq_old)).limit_denominator(limit)
    if fr.numerator == 0:
        raise ValueError('Increase limit to represent the requested frequency ratio')
    actual_rate = freq_old * fr.numerator / fr.denominator
    y2 = signal.resample_poly(data, fr.numerator, fr.denominator, axis=0)
    if is_frame:
        y2 = pd.DataFrame(y2, columns=y.columns, index=origin + np.arange(len(y2)) / actual_rate)
        y2.index.name = y.index.name
    return y2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start with a labelled time grid

    A sampling frequency in hertz gives samples per second. A 100-sample sequence at 100 Hz has sample times 0 through 0.99 s. Its sampling-bin duration is 1 s; the elapsed time between first and last samples is 0.99 s.
    """)
    return


@app.cell
def _(np, pd):
    rng = np.random.default_rng(7)
    y = pd.DataFrame(rng.standard_normal((100, 3)), columns=["A", "B", "C"])
    y.index = np.arange(len(y)) / 100
    y.index.name = "Time [s]"
    y.head()
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Predict the output length and time step when converting this sequence to 1,000 Hz.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 1

    The ratio is 10, so there are 1,000 output samples, spaced by 0.001 s. The first time is 0 and the last is 0.999 s.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Change the sampling grid

    `resample_poly` upsamples, applies a low-pass filter, and downsamples. The filter controls aliasing when reducing the sampling rate. Upsampling interpolates: it does not recover content already lost during acquisition. The default zero padding can affect the ends of a finite record.
    """)
    return


@app.cell
def _(y):
    y2 = resample(y, freq_old=100, freq_new=1000)
    y2.head()
    return (y2,)


@app.cell
def _(plt, y, y2):
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.plot(y.index, y["A"], "o", label="original")
    _ax.plot(y2.index, y2["A"], label="resampled")
    _ax.set(xlabel="Time [s]", ylabel="Signal [a.u.]")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** What is the Nyquist frequency after downsampling a signal from 1,000 to 100 Hz? Why filter first?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 2

    The new Nyquist frequency is 50 Hz. Components above 50 Hz can fold into lower frequencies unless sufficiently attenuated before downsampling. Polyphase resampling includes this filtering.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keep the time origin and report the effective rate

    For a DataFrame, the helper keeps the first timestamp. If `freq_old` is omitted, it checks that the numeric index is increasing and uniformly sampled. A rational approximation represents the rate ratio; `limit` controls its denominator. The output index uses the effective rate of that approximation, rather than silently labelling it with an unattained requested rate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Can upsampling undo aliasing introduced during acquisition?

    2. Why inspect the first and last part of the output?

    3. Can this helper infer a rate from irregular timestamps?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Checkpoint answers

    1. No.
    2. Padding and filtering affect boundaries.
    3. No; provide a justified uniform grid first.
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
    **Problem 1.** Resample 240 samples from 120 Hz to 150 Hz. Give the output length and interval.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 1

    300 samples; interval $1/150$ s.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** A uniformly sampled DataFrame begins at 3 s. What is its first output time after resampling?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 2

    3 s.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Compare resampling with [time normalization](https://github.com/demotu/tnorma): seconds and percent cycle answer different questions.

    ### Video lectures on the Internet

    [MIT Signals and Systems video lectures](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/video_galleries/video-lectures/) include sampling and reconstruction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [SciPy: resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
    """)
    return


if __name__ == "__main__":
    app.run()
