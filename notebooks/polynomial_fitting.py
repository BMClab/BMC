# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polynomial fitting with confidence/prediction intervals

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A quadratic can follow curvature that a line misses, but every extra coefficient must be estimated from the same finite data. Here we compare degrees, inspect residual scatter, and calculate intervals from the full polynomial model. Start with [Curve fitting](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/curve_fitting.py) if least squares is new.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** A quadratic fits curved data better than a line. Does that guarantee that a tenth-degree polynomial predicts new observations better still?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy performs the fit, SciPy supplies t quantiles, Matplotlib draws intervals, and pandas computes a rolling SD. The fitting helper is included in this notebook and used directly. The examples generate synthetic data and do not read external files.
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
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's implement the polynomial fit by least squares and calculate the confidence and prediction intervals assuming normal distribution of the residuals.<br>
    The code of the `polyfit.py` function is at the end of this notebook (or download the function from the GitHub repo).
    """)
    return


@app.cell
def _(polyfit_1):
    polyfit = polyfit_1
    return (polyfit,)


@app.cell
def _(polyfit):
    help(polyfit)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit a polynomial, then question it

    The coefficients returned by `np.polyfit` run from highest power to constant: `[3, 1, 2]` means $3x^2+x+2$. This model is nonlinear in x but linear in its coefficients. Before running the noise-added example, sketch that quadratic and predict where a straight line will miss it.

    **Exercise 1.** Evaluate $3x^2+x+2$ at x=0 and x=2. How many parameters does the quadratic fit estimate?

    /// details | Solution to Exercise 1
    The values are 2 and $12+2+2=16$. There are three coefficients, so 50 observations leave $50-3=47$ residual degrees of freedom.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some data to play:
    """)
    return


@app.cell
def _(np):
    N = 50
    x = np.linspace(-2, 4, N)
    y = np.polyval([3, 1, 2], x) + np.random.randn(N)*8
    return N, x, y


@app.cell
def _(polyfit, x, y):
    (p, _perr, _R2, chi2red, yfit, _ci, _pi, MSE) = polyfit(x, y, 2)
    return MSE, chi2red, p


@app.cell
def _(p):
    p
    return


@app.cell
def _(MSE):
    MSE
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Residual scatter and uncertainty scales

    With the default unit weights, reduced chi-square is the residual sum of squares divided by the residual degrees of freedom. Its value is tied to those arbitrary units; a large number alone does not demonstrate a poor model. Compare it with a credible noise scale and inspect the residual pattern.

    Here we estimate a common noise scale from the residuals and refit. This changes the stated scale, not the information in the data. The helper treats supplied `yerr` values as **relative** standard-deviation scales and estimates their common multiplier from residuals (`np.polyfit(..., cov=True)`). Making reduced chi-square one by this construction does not independently validate the fit.
    """)
    return


@app.cell
def _(N, chi2red, np):
    yerr = np.ones(N)*np.sqrt(chi2red)
    print('Average uncertainty in y:', yerr[0])
    return (yerr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's run the fitting again:
    """)
    return


@app.cell
def _(polyfit, x, y, yerr):
    (p_1, _perr, _R2, chi2red_1, yfit_1, _ci, _pi, MSE_1) = polyfit(x, y, 2, yerr=yerr)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** All relative SD scales are multiplied by two. Should the fitted coefficients change? What happens to reduced chi-square?

    /// details | Solution to Exercise 2
    Every residual gets the same extra factor of one-half before squaring. The minimizer is unchanged; reduced chi-square divides by four. With scaled covariance, the estimated coefficient covariance is unchanged because the estimated common noise multiplier compensates.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see the results of linear and quadratic fits:
    """)
    return


@app.cell
def _(plt, polyfit, x, y):
    (fig, ax) = plt.subplots(1, 2, figsize=(12, 5))
    (p_2, _perr, _R2, chi2red_2, yfit_2, _ci, _pi, MSE_2) = polyfit(x, y, degree=1, axis=ax[0])
    (p_2, _perr, _R2, chi2red_2, yfit_2, _ci, _pi, MSE_2) = polyfit(x, y, degree=2, axis=ax[1], ylabel='')
    plt.tight_layout()
    fig
    return (yfit_2,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Intervals follow the chosen degree

    For a design row $v(x)=[x^d,\ldots,x,1]$ and coefficient covariance $C$, the fitted mean’s variance is $vCv^T$. This expression includes every polynomial coefficient and its covariances. A straight-line leverage formula is not valid for a quadratic. A future observation adds its noise variance.

    The plots deliberately compare a **95% confidence interval** with a **68.3% prediction interval**. Because their coverage levels differ, do not use their relative widths as a general comparison of confidence and prediction intervals.

    **Exercise 3.** If a degree-two model has 50 observations, why is a formula based only on intercept and slope incomplete?

    /// details | Solution to Exercise 3
    The fitted prediction also depends on the quadratic coefficient and its covariance with the other two coefficients. The row $[x^2,x,1]$ propagates all those contributions; a two-parameter formula omits them.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P.S.: Calculation of moving standard-deviation

    A rolling SD describes variation within a local window. On a sloping or curved signal it contains variation of the trend as well as noise; it is not automatically a prediction interval. Compare its shape with the model-based band, and ask which variation each includes.
    """)
    return


@app.cell
def _(N, np, y):
    import pandas as pd

    ys = pd.Series(y)
    # use a moving window of one-tenth of the size of the data.
    ys_std = ys.rolling(window=int(np.round(N/10)), min_periods=4, center=True).std()
    return (ys_std,)


@app.cell
def _(plt, x, y, yfit_2, ys_std):
    plt.figure(figsize=(10, 5))
    plt.fill_between(x, yfit_2 + ys_std, yfit_2 - ys_std, color=[1, 0, 0, 0.1], edgecolor=None, label='Moving standard-deviation')
    plt.plot(x, y, 'o', color=[0, 0.1, 0.9, 1], markersize=8)
    plt.plot(x, yfit_2, linewidth=3, color=[1, 0, 0, 0.8], label='Polynomial (degree {}) fit'.format(2))
    plt.legend()
    plt.show()
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** Would a rolling SD of a perfectly noiseless sloping line be zero?

    /// details | Solution to Exercise 4
    No. Different x values within the window have different true y values. Subtracting a suitable trend before summarizing the residuals separates local trend variation from noise.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function `polyfit.py`
    """)
    return


@app.cell
def _(np, plt):
    """Least squares polynomial regression with confidence/prediction intervals.
    """
    __author__ = 'Marcos Duarte, https://github.com/BMClab/BMC'
    __version__ = 'polyfit.py v.1.0.3 20123/07/07'
    __license__ = 'MIT'
    import scipy.stats as stats

    def polyfit_1(x, y, degree, yerr=None, plot=True, xlabel='x', ylabel='y', title=True, legend=True, plotCI=True, plotPI=True, axis=None):
        """Least squares polynomial regression of order degree for x vs. y [1]_

        Parameters
        ----------
        x : numpy array_like, shape (N,)
            Independent variable, x-coordinates of the N points (x[i], y[i]).
        y : numpy array_like, shape (N,)
            Dependent variable, y-coordinates of the N points (x[i], y[i]).
        degree : integer
            Degree of the polynomial to be fitted to the data.
        yerr : numpy array_like, shape (N,), optional (default = None)
            Relative standard-deviation scales in y, finite and positive.
            A common multiplier is estimated from residuals (cov=True);
            equal scales are assumed when omitted.
        plot : bool, optional (default = True)
            Show plot (True) of not (False).
        xlabel : string, optional (default = 'x')
            Label for the x (horizontal) axis.
        ylabel : string, optional (default = 'y')
            Label for the y (vertical) axis.
        title : bool, optional (default = True)
            Show title (True) of not (False) in the plot.
        legend : bool, optional (default = True)
            Show legend (True) of not (False) in the plot.
        plotCI : bool, optional (default = True)
            Plot the shaded area for the confidence interval (True) of not (False).
        plotPI : bool, optional (default = True)
            Plot the shaded area for the prediction interval (True) of not (False).
        axis : matplotlib object, optional (default = None)
            Matplotlib axis object where to plot.

        Returns
        -------
        p : numpy array, shape (deg + 1,)
            Coefficients of the least squares polynomial fit.
        perr : numpy array, shape (deg + 1,)
            Standard-deviation of the coefficients.
        R2 : float
            Coefficient of determination.
        chi2red : float
            Reduced chi-squared
        yfit : numpy array, shape (N,)
            Values of the fitted polynomial evaluated at x.
        ci : numpy array, shape (N,)
            Values of the 95% confidence interval evaluated at x.
        pi : numpy array, shape (N,)
            Values of the 68% prediction interval evaluated at x.
        MSE : float
            Mean squared error of the regression.

        References
        ----------
        .. [1] https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html


        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> N = 50
        >>> x = np.linspace(-2, 4, N)
        >>> y = np.polyval([3, 1, 2], x) + np.random.randn(N)*8
        >>> # simple use:
        >>> polyfit(x, y, 2);
        >>> # compare two models:
        >>> fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        >>> p1, perr1, R21, chi2red1, yfit1, ci1, pi1, MSE = polyfit(x, y, 1, axis=ax[0])
        >>> p2, perr2, R22, chi2red2, yfit2, ci2, pi2, MSE = polyfit(x, y, 2, axis=ax[1])
        >>> plt.tight_layout()
        >>> Enter error (uncertainty) in y:
        >>> yerr = np.ones(N)*8
        >>> polyfit(x, y, 2, yerr);


        Version history
        ---------------
        '1.0.3':
            Correct the order of the terms in title and account for zeros in yerr.
        '1.0.2':
            Included output 'MSE', mean squared error of the regression.

        """
        (x, y) = (np.asarray(x), np.asarray(y))
        N = y.size
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or N <= degree + 1:
            raise ValueError('Use matching one-dimensional data with positive residual degrees of freedom')
        if yerr is None:
            yerr = np.ones(N)
            errorbar = False
        else:
            yerr = np.asarray(yerr, dtype=float)
            if yerr.shape != y.shape or np.any(~np.isfinite(yerr)) or np.any(yerr <= 0):
                raise ValueError('yerr must contain one finite positive scale per observation')
            errorbar = True
        (p, cov) = np.polyfit(x, y, degree, w=1 / yerr, cov=True)
        yfit = np.polyval(p, x)
        _perr = np.sqrt(np.diag(cov))
        res = y - yfit
        chi2red = np.sum(res ** 2 / yerr ** 2) / (N - degree - 1)
        s_err = np.sqrt(np.sum(res ** 2) / (N - degree - 1))
        SSres = np.sum(res ** 2)
        MSE = SSres / N
        SStot = np.sum((y - np.mean(y)) ** 2)
        _R2 = 1 - SSres / SStot
        R2adj = 1 - SSres / (N - degree - 1) / (SStot / (N - 1))
        t95 = stats.t.ppf(0.95 + (1 - 0.95) / 2, N - degree - 1)
        design = np.vander(x, degree + 1)
        fit_variance = np.maximum(np.einsum('ij,jk,ik->i', design, cov, design), 0)
        _ci = t95 * np.sqrt(fit_variance)
        t68 = stats.t.ppf(0.683 + (1 - 0.683) / 2, N - degree - 1)
        _pi = t68 * np.sqrt(fit_variance + chi2red * yerr ** 2)
        if plot:
            if N < 50 or N > 500:
                x2 = np.linspace(np.min(x), np.max(x), 100)
                yfit2 = np.polyval(p, x2)
                design2 = np.vander(x2, degree + 1)
                variance2 = np.maximum(np.einsum('ij,jk,ik->i', design2, cov, design2), 0)
                ci2 = t95 * np.sqrt(variance2)
                order = np.argsort(x)
                noise_scale2 = np.interp(x2, x[order], yerr[order])
                pi2 = t68 * np.sqrt(variance2 + chi2red * noise_scale2**2)
            else:
                order = np.argsort(x)
                (x2, yfit2, ci2, pi2) = (x[order], yfit[order], _ci[order], _pi[order])
            if axis is None:
                (fig, axis) = plt.subplots(1, 1, figsize=(10, 5))
            else:
                fig = 0
            if plotPI:
                axis.fill_between(x2, yfit2 + pi2, yfit2 - pi2, color=[1, 0, 0, 0.1], edgecolor=None, label='68% prediction interval')
            if plotCI:
                axis.fill_between(x2, yfit2 + ci2, yfit2 - ci2, color=[1, 0, 0, 0.3], edgecolor=None, label='95% confidence interval')
            if errorbar:
                axis.errorbar(x, y, yerr=yerr, fmt='o', capsize=0, color=[0, 0.1, 0.9, 1], markersize=8)
            else:
                axis.plot(x, y, 'o', color=[0, 0.1, 0.9, 1], markersize=8)
            axis.plot(x2, yfit2, linewidth=3, color=[1, 0, 0, 0.8], label='Polynomial (degree {}) fit'.format(degree))
            axis.set_xlabel(xlabel, fontsize=16)
            axis.set_ylabel(ylabel, fontsize=16)
            if legend:
                axis.legend()
            if title:
                xs = ['x^{:d}'.format(ideg) if ideg > 1 else ('x' if ideg == 1 else '') for ideg in range(degree, -1, -1)]
                title = ['({:.2f} \\pm {:.2f}) {}'.format(i, j, k) for (i, j, k) in zip(p, _perr, xs)]
                R2str = '\\, (R^2 = ' + '{:.2f}'.format(_R2) + ', \\chi^2_{red} = ' + '{:.1f}'.format(chi2red) + ')'
                title = '$y = ' + '+'.join(title) + R2str + '$'
                axis.set_title(title, fontsize=12, color=[0, 0, 0])
            if fig:
                plt.show()
        return (p, _perr, _R2, chi2red, yfit, _ci, _pi, MSE)

    return (polyfit_1,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Does more polynomial degree always improve prediction?
    2. What does `cov=True` assume about supplied weights?
    3. Is rolling SD a confidence interval for the fitted mean?

    /// details | Checkpoint answers
    1. No; it can overfit and extrapolate poorly.
    2. Their relative scale is trusted and a common residual multiplier is estimated.
    3. No.
    ///

    ## Problems

    1. Give residual degrees of freedom for 30 observations and a cubic.
    2. For coefficient covariance `diag(0.01, 0.04, 0.09)` and quadratic design row `[0,0,1]`, give the fitted-mean standard error.
    3. State the model value at x=1 for coefficients `[2,-1,4]`.

    /// details | Final answers
    1. 26.
    2. 0.3 in y units.
    3. 5.
    ///

    ## Go deeper

    [NumPy polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) — weight and covariance conventions. [Confidence and prediction intervals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/confidence_prediction_intervals.py) — interpretation of interval targets.

    ### Video lectures on the Internet

    [MIT Statistics for Applications lectures](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/video_galleries/lecture-videos/) — the estimation and regression lectures explain uncertainty in fitted parameters and predictions.

    ## References

    - [NumPy polynomial fitting documentation](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html). Marcos Duarte, original `polyfit.py` teaching implementation, revised here for degree-aware intervals.
    """)
    return


if __name__ == "__main__":
    app.run()
