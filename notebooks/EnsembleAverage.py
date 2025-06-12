import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Ensemble average

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A usual procedure employed to present the average pattern of a variable as a function of time or movement cycle across trials or across subjects is to show the ensemble average curve, which is a fancy name for (typically) the mean$\pm$1 standard-deviation curve.

        Let's simulate some data and explore different aesthetic variations to present the ensemble average.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    from tnorma import tnorma
    return np, plt, tnorma


@app.cell
def _(np, plt):
    # slighly different data with slightly different duration (number of points)
    from scipy.signal import butter, filtfilt
    # Butterworth filter
    b, a = butter(1, (5/(100/2)), btype = 'low')
    N = 10
    y = np.empty([120, N]) * np.NaN
    for i in range(N):
        t = np.arange(0, 100 + np.random.randint(-20, high=20)) / 100
        y[0: t.size, i] = 2*np.sin(2 * np.pi * t) + np.random.randn(t.size) / 2 + t.size / 20
        y[0: t.size, i] = filtfilt(b, a,  y[0: t.size, i])

    plt.figure(figsize=(9, 4))
    plt.plot(y)
    plt.xlabel('data points')
    plt.title('Plot of trials')
    plt.show()
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To calculate the mean and standard deviation across these trials the different trials must have the same number of points.   
        We can do this with the [time normalization of data](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/TimeNormalization.ipynb), where we will normalize each trial to the same percent cycle (from 0 to 100%):
        """
    )
    return


@app.cell
def _(plt, tnorma, y):
    # yn, tn, indie = tnorma(y, axis=0, step=1, k=3, smooth=0, mask=None,
    #                       nan_at_ext='delete', show=False, ax=None)
    yn, tn, indie = tnorma(y)
    # plot of the normalized data
    plt.figure(figsize=(10, 5))
    plt.plot(yn)
    plt.xlabel('Cycle [%]')
    plt.title('Plot of normalized trials')
    plt.show()
    return tn, yn


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So, the trials have the same number of points, now we can calculate the mean and standard deviation curves and plot the ensemble average
        """
    )
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
    mo.md(
        r"""
        And here are some aesthetic variations to show the ensemble average:
        """
    )
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
    # fill_between() command creates a PolyCollection that is not supported by the legend()
    ax3.legend([p1, p2], ['Group 1', 'Group 2'], fontsize=14, loc='best', framealpha=.8)

    ax4 = plt.subplot(224)
    ax4.set_title('Combine errorbar and shaded area')
    ax4.errorbar(tn, ym, ysd, color = [0, 0, 1, 0.5], capsize=0, lw=1.5)
    y2 = np.mean(ym) - ym + ym[0]
    ax4.fill_between(tn, y2+ysd, y2-ysd, color=[1, 0, 0, 0.5], edgecolor=None)
    ax4.set_xlabel('Cycle [%]')
    p1 = plt.Line2D((0, 1), (1, 1), color=[0, 0, 1, 1], lw=1.5)
    p2 = plt.Rectangle((0, 0), 1, 1, color=[1, 0, 0, 0.5])
    # fill_between() command creates a PolyCollection that is not supported by the legend()
    ax4.legend([p1, p2], ['Group 1', 'Group 2'], fontsize=14, loc='best', framealpha=.8)

    plt.suptitle(r'$\mathrm{Aesthetic\,variations\,for\,the\,ensemble\,average}$', fontsize=18, y=1.04)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Instead of mean and standard deviation we can use median and the first and third quartiles:
        """
    )
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


if __name__ == "__main__":
    app.run()
