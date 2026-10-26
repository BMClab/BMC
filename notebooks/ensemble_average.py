# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14", "tnorma>=0.0.3"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ensemble average

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Ten movement trials have different durations. How can we compare their typical shape without pretending that their timing was identical?

    We will normalize cycles, calculate a mean curve, and compare error bars, shaded bands, and median/quartile displays.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The examples are self-contained.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    We use NumPy for arrays, Matplotlib for plots, SciPy for filtering, and `tnorma` to normalize each trial to the same percent cycle. The notebook declares these dependencies for standalone use.

    An ensemble average is the pointwise mean across aligned trials or subjects. A band of $\pm$1 standard deviation is an optional display of between-trial dispersion; it is not the definition of the mean and is not a confidence interval for that mean.

    Let's simulate some data and explore different aesthetic variations to present the ensemble average.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from tnorma import tnorma
    if not hasattr(np, 'NaN'):  # tnorma 0.0.3 still uses np.NaN, removed in NumPy 2
        np.NaN = np.nan

    return np, plt, tnorma


@app.cell
def _(np, plt):
    # slighly different data with slightly different duration (number of points)
    from scipy.signal import butter, filtfilt
    # Butterworth filter
    b, a = butter(1, (5/(100/2)), btype = 'low')
    _rng = np.random.default_rng(4)
    N = 10
    y = np.empty([120, N]) * np.nan
    for i in range(N):
        t = np.arange(0, 100 + _rng.integers(-20, high=20)) / 100
        y[0: t.size, i] = 2*np.sin(2 * np.pi * t) + _rng.standard_normal(t.size) / 2 + t.size / 20
        y[0: t.size, i] = filtfilt(b, a,  y[0: t.size, i])

    plt.figure(figsize=(9, 4))
    plt.plot(y)
    plt.xlabel('data points')
    plt.title('Plot of trials')
    plt.show()
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Before averaging, why is padding every trial to 120 samples not sufficient to align a movement cycle?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 1

    Padding makes array sizes equal but does not match movement phase. Here each valid trial is mapped to 0–100% cycle. This assumes that corresponding percentages represent comparable phases; it does not align every event within the cycle.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To calculate the mean and standard deviation across these trials the different trials must have the same number of points.<br>
    We can do this with the [time normalization of data](https://github.com/demotu/tnorma), where we will normalize each trial to the same percent cycle (from 0 to 100%):
    """)
    return


@app.cell
def _(np, plt, tnorma, y):
    # yn, tn, indie = tnorma(y, axis=0, step=1, k=3, smooth=0, mask=None,
    #                       nan_at_ext='delete', show=False, ax=None)
    # Normalize each complete trial independently; the NaN padding is not a shared cutoff.
    _normalized = [tnorma(y[:, _i], nan_at_ext="delete")[0].reshape(-1) for _i in range(y.shape[1])]
    yn = np.column_stack(_normalized)
    tn = np.linspace(0, 100, yn.shape[0])
    # plot of the normalized data
    plt.figure(figsize=(10, 5))
    plt.plot(yn)
    plt.xlabel('Cycle [%]')
    plt.title('Plot of normalized trials')
    plt.show()
    return tn, yn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So, the trials have the same number of points, now we can calculate the mean and standard deviation curves and plot the ensemble average
    """)
    return


@app.cell
def _(np, plt, tn, yn):
    ym, ysd = np.mean(yn, axis=1), np.std(yn, axis=1, ddof=1) # one line is all we need
    # plot of the ensemble average
    plt.figure(figsize=(10,5))
    plt.errorbar(tn, ym, ysd, linewidth=2)
    plt.xlabel('Cycle [%]')
    plt.title('Plot of ensemble average')
    plt.show()
    return ym, ysd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** At 50% cycle, three trials have values 2, 4, and 6. Find the mean and sample standard deviation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 2

    The mean is 4. The sample variance is $[(2-4)^2+(4-4)^2+(6-4)^2]/(3-1)=4$, so the sample SD is 2. The plotted mean ± SD band extends from 2 to 6. It describes dispersion, not uncertainty in the mean.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And here are some aesthetic variations to show the ensemble average:
    """)
    return


@app.cell
def _(np, plt, tn, ym, ysd):
    plt.rc('axes', labelsize=10,  titlesize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.figure(figsize=(10, 7))

    ax1 = plt.subplot(221)
    ax1.set_title('Errorbar every two data points and zero cap size')
    ax1.errorbar(tn, ym, ysd, color = [0, 0, 1, 0.5], capsize=0, errorevery=2, lw=4)

    ax2 = plt.subplot(222)
    ax2.set_title('Mean curve and shaded area')
    ax2.fill_between(tn, ym+ysd, ym-ysd, color = [0, 0, 1, 0.5])
    ax2.plot(tn, ym, color = [0, 0, 1, .8], lw=2, label='Data')
    ax2.legend(fontsize=14, loc='best', framealpha=.8)

    ax3 = plt.subplot(223)
    ax3.set_title('Semi-transparent shaded areas overlap well')
    ax3.fill_between(tn, ym+ysd, ym-ysd, color=[0, 0, 1, 0.5], edgecolor=None)
    y2 = np.mean(ym) - ym + ym[0]
    ax3.fill_between(tn, y2+ysd, y2-ysd, color=[1, 0, 0, 0.5], edgecolor=None)
    ax3.set_xlabel('Cycle [%]')
    p1 = plt.Rectangle((0, 0), 1, 1, color=[0, 0, 1, 0.5])
    p2 = plt.Rectangle((0, 0), 1, 1, color=[1, 0, 0, 0.5])
    # Explicit proxy handles give the two shaded regions descriptive labels.
    ax3.legend([p1, p2], ['Group 1', 'Group 2'], fontsize=14, loc='best', framealpha=.8)

    ax4 = plt.subplot(224)
    ax4.set_title('Combine errorbar and shaded area')
    ax4.errorbar(tn, ym, ysd, color = [0, 0, 1, 0.5], capsize=0, lw=1.5)
    y2 = np.mean(ym) - ym + ym[0]
    ax4.fill_between(tn, y2+ysd, y2-ysd, color=[1, 0, 0, 0.5], edgecolor=None)
    ax4.set_xlabel('Cycle [%]')
    p1 = plt.Line2D((0, 1), (1, 1), color=[0, 0, 1, 1], lw=1.5)
    p2 = plt.Rectangle((0, 0), 1, 1, color=[1, 0, 0, 0.5])
    # Explicit proxy handles give the two shaded regions descriptive labels.
    ax4.legend([p1, p2], ['Group 1', 'Group 2'], fontsize=14, loc='best', framealpha=.8)

    plt.suptitle(r'$\mathrm{Aesthetic\,variations\,for\,the\,ensemble\,average}$', fontsize=18, y=1.04)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** If an outlier is added to otherwise similar trials, which summary is generally less sensitive: mean/SD or median/quartiles?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Solution to Exercise 3

    The median and quartiles depend on ranks and are generally less sensitive to an extreme value. Neither summary removes the need to inspect why the trial differs.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of mean and standard deviation we can use median and the first and third quartiles:
    """)
    return


@app.cell
def _(np, plt, tn, yn):
    ym_1 = np.median(yn, axis=1)
    (yq1, yq3) = (np.abs(ym_1 - np.percentile(yn, 25, 1)), np.abs(np.percentile(yn, 75, 1) - ym_1))
    plt.figure(figsize=(9, 5))
    plt.errorbar(tn, ym_1, np.vstack((yq1, yq3)), linewidth=2)
    plt.xlabel('Cycle [%]')
    plt.title('Plot of ensemble average (median and quartiles)')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Does normalization preserve absolute movement duration?

    2. Is mean ± SD a 95% confidence interval?

    3. Should repeated trials from one person automatically count as independent participants?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Checkpoint answers

    1. No; keep the original durations separately.
    2. No.
    3. No; the unit of analysis matters.
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
    **Problem 1.** Four trials have values 1, 3, 5, and 7 at one phase. Find their mean and median.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 1

    Mean = 4; median = 4.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** How many phase samples span 0–100% inclusive with a 1% interval?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 2

    101.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Compare the mean and median displays using the same simulated trials.

    ### Video lectures on the Internet

    [MIT Introduction to Probability and Statistics: video lectures](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/video_galleries/video-lectures/) review averages and dispersion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Time normalization](https://github.com/demotu/tnorma).

    - [NumPy standard deviation and ddof](https://numpy.org/doc/stable/reference/generated/numpy.std.html).
    """)
    return


if __name__ == "__main__":
    app.run()
