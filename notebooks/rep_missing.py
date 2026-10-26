# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Replace missing data

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A marker disappears for two frames. Should we delete those frames, fill them with a constant, or interpolate?

    We will compare the replacement methods already used in this collection, while keeping missing measurements distinguishable from measured zeros.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The examples are self-contained.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy stores the signal. The helper is included below so this notebook can run as a single file.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.function
def rep_missing(x, value=np.nan, new_value='interp', max_alert=100):
    """Replace missing data in different ways.

    Parameters
    ----------
    x : 1D numpy array_like
        data
    value : number, optional. Default = np.nan.
        Scalar value in x marking missing data. The input is never modified.
    new_value : number or string, optional. Default = 'interp'.
        Value or string for the method to use for replacing missing data:
        'delete': delete missing data.
        new_value: replace missing data with new_value.
        'mean': replace missing data with the mean of the rest of the data.
        'median': replace missing data with the median of the rest of the data.
        'interp': replace missing data by linear interpolating over them.
    max_alert : number, optional. Default = 100.
        Minimal number of sequential data for a message to be printed with
        information about the continuous missing data.
        Set to 0 to not print any message.

    This function can handle NaNs separately from missing data (in case they
    are different), but the 'interp' method will interpolate on both the
    missing data and NaNs. Interpolation uses sample indices, assumes uniform sampling,
    and extends the nearest observed value at missing endpoints.


    Returns
    -------
    y : 1D numpy array
        1D numpy array similar to x but with missing data replaced according
        to value or method specified with the parameter new_value.

    References
    ----------
    .. [1] https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/rep_missing.py

    Examples
    --------
    >>> x = [1, 1, 2, 0, 4, 1, 1, 0, 1, 0, 2]

    """
    import numpy as np
    from numbers import Number
    y = np.array(x, dtype=complex if np.iscomplexobj(x) else float, copy=True)
    if y.ndim != 1:
        raise ValueError('x must be one-dimensional')
    idx = np.isnan(y) if np.isnan(value) else y == value
    if new_value == 'interp':
        idx = idx | np.isnan(y)
    if isinstance(new_value, str) and new_value not in {'delete', 'mean', 'median', 'interp'}:
        raise ValueError('new_value must be numeric, delete, mean, median, or interp')
    if max_alert and idx.any():
        edges = np.diff(np.r_[False, idx, False].astype(int))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        if np.any(lengths >= max_alert):
            import warnings
            warnings.warn('Long missing-data sequence; assess whether replacement is justified.', stacklevel=2)
    if isinstance(new_value, Number):
        y[idx] = new_value
    elif new_value == 'delete':
        y = y[~idx]
    elif idx.any():
        valid = ~idx & np.isfinite(y)
        if not valid.any():
            raise ValueError('Replacement requires at least one finite observed value')
        if new_value == 'mean':
            y[idx] = np.mean(y[valid])
        elif new_value == 'median':
            y[idx] = np.median(y[valid])
        else:
            missing = idx | np.isnan(y)
            samples = np.arange(y.size)
            y[missing] = np.interp(samples[missing], samples[valid], y[valid])
    return y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mark the missing values

    `value` identifies a missing-data sentinel. The default is `np.nan`; a zero is missing only when the acquisition convention says so. The function returns a copy, leaving the measurement unchanged. `max_alert` warns about long consecutive gaps; it does not make filling those gaps scientifically justified.
    """)
    return


@app.cell
def _(np):
    x = np.array([1, 2, np.nan, 4, 20, np.nan, np.nan, 0, 1])
    x
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Which entries in `x` are missing under the default convention? What changes when `value=0`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 1

    With the default convention the missing indices are 2, 5, and 6 (zero-based). With `value=0`, only index 7 is selected. NaNs still remain NaNs for constant, mean, and median replacement; interpolation also fills NaNs.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare the replacements

    A mean or median uses only finite, unflagged observations. Deleting a sample changes the time grid if you subsequently pretend that the remaining samples are equally spaced. Keep the original timestamps when deleting data.
    """)
    return


@app.cell
def _(x):
    replacements = {
        "constant": rep_missing(x, new_value=0),
        "delete": rep_missing(x, new_value="delete"),
        "mean": rep_missing(x, new_value="mean"),
        "median": rep_missing(x, new_value="median"),
        "interpolate": rep_missing(x, new_value="interp"),
    }
    replacements
    return (replacements,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** For `[1, 0, 9]`, treating zero as missing, find the mean and median replacements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 2

    Both give `[1, 5, 9]`: the observed values are 1 and 9, whose mean and median are both 5. Including the sentinel in the calculation would bias the result.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpolate a short gap

    Linear interpolation assumes uniformly spaced samples and a straight segment between adjacent observed values. At missing endpoints this helper extends the nearest observed value, an explicit extrapolation convention. It cannot reconstruct unobserved rapid changes. All-missing input raises a clear error for data-dependent replacement.
    """)
    return


@app.cell
def _(np):
    rep_missing([np.nan, 2, np.nan, 6, np.nan])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** What is the interpolated interior value in `[2, NaN, 6]`, and why is replacing a long gap riskier?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 3

    The interior value is 4 because it lies halfway between two observations. A long gap permits many possible trajectories; linear interpolation chooses one without measuring it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Detect consecutive gaps before filling

    The companion [detecta sequence-detection guide](https://github.com/demotu/detecta) returns inclusive start/end indices and lets you reject gaps longer than an acceptable duration. For the original example, the NaN runs are `[2, 2]` and `[5, 6]`; a run of length two spans two missing samples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Is a numerical zero always missing?

    2. Does filling a gap create a new measurement?

    3. What happens to the original input array?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Checkpoint answers

    1. No; the file convention determines that.
    2. No; the filled values are estimates.
    3. It stays unchanged.
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
    **Problem 1.** Fill `[2, NaN, 8]` by linear interpolation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 1

    `[2, 5, 8]`.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** At 100 Hz, how many samples are missing in a gap marked by inclusive indices 10 through 14?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 2

    5 samples; missing-sample duration 0.05 s.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Compare replacement choices on a short, known gap before applying them to a recording.

    ### Video lectures on the Internet

    [MIT OpenCourseWare: Signals and Systems video lectures](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/video_galleries/video-lectures/) give context for sampling and interpolation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [NumPy: linear interpolation](https://numpy.org/doc/stable/reference/generated/numpy.interp.html).

    - [Sequence detection in detecta](https://github.com/demotu/detecta).
    """)
    return


if __name__ == "__main__":
    app.run()
