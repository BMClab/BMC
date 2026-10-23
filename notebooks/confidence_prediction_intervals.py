# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14", "urllib3>=2.2", "ipython"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Confidence and prediction intervals

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    You can estimate a group’s mean precisely while still being unable to predict the next person precisely. This guide puts those two intervals side by side, starting with a population whose parameters are known and then estimating them from a finite sample.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** Collecting four times as many independent observations roughly halves the standard error. Does it also halve the spread of the next observation?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy generates simulated samples, SciPy supplies probability distributions, and Matplotlib draws them. The small `pdf_norm_plot` drawing helper is loaded from this repository. All samples are simulated; there is no experimental dataset to download.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a finite population of equally weighted observations, the mean $\mu$ (a measure of central tendency) and variance $\sigma^2$ (a measure of dispersion) of a population are the well known formulas:

    $$
    \mu = \frac{1}{N}\sum_{i=1}^{N} x_i
    $$

    $$
    \sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2
    $$

    For a more general case, a continuous univariate random variable $x$ with [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) (PDF), $f(x)$, the mean and variance of a population are:

    $$
    \mu = \int_{-\infty}^{\infty} x f(x)\: dx
    $$

    $$
    \sigma^2 = \int_{-\infty}^{\infty} (x-\mu)^2 f(x)\: dx
    $$

    The PDF is a function that describes the relative likelihood for the random variable to take on a given value.<br>
    The mean is the first raw moment and the variance is the second central moment, provided these moments exist. The standard deviation $\sigma$ of the population is the square root of the variance.

    The normal distribution is useful partly because of the central limit theorem: for independent, identically distributed observations with finite variance, the standardized sample mean approaches a normal distribution as the number of observations **within each sample** grows. Repeating small samples many times reveals their sampling distribution; it does not itself make that distribution normal.

    The probability density function of a univariate normal (or Gaussian) distribution is:

    $$
    f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr)
    $$

    The only parameters that define the normal distribution are the mean $\mu$ and the variance $\sigma^2$, because of that a normal distribution is usually described as $N(\mu,\:\sigma^2)$.

    Here is a plot of the PDF for the normal distribution:
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
    # import the necessary libraries
    import numpy as np
    # '%matplotlib notebook' command supported automatically in marimo
    import matplotlib.pyplot as plt
    from IPython.display import display, Latex
    from scipy import stats
    import sys
    sys.path.insert(0, str(repo_file('functions/pdf_norm_plot.py').parent))  # add to pythonpath
    return Latex, display, np, plt, stats


@app.cell
def _(plt):
    from pdf_norm_plot import pdf_norm_plot
    pdf_norm_plot()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The horizontal axis above is shown in terms of the number of standard deviations in relation to the mean, which is known as standard score or $z$ score:

    $$
    z = \frac{x - \mu}{\sigma}
    $$

    So, instead of specifying raw values in the distribution, we define the PDF in terms of $z$ scores; this conversion process is called standardizing the distribution (and the result is known as standard normal distribution). If $x$ is normal with mean $\mu$ and SD $\sigma$, its standardized value $z$ is standard normal, with mean zero and variance one.

    The percentage numbers in the plot are the probability (area under the curve) for each interval shown in the horizontal label.<br>
    An interval in terms of z score is specified as: $[\mu-z\sigma,\;\mu+z\sigma]$.<br>
    The interval $[\mu-1\sigma,\;\mu+1\sigma]$ contains 68.3% of the population and the interval $[\mu-2\sigma,\;\mu+2\sigma]$ contains 95.4% of the population.<br>
    These numbers can be calculated using the function `stats.norm.cdf()`, the [cumulative distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function) (CDF) of the normal distribution at a given value:
    """)
    return


@app.cell
def _(Latex, display, stats):
    print('Cumulative distribution function (cdf) of the normal distribution:')
    for _i in range(-3, 4):
        display(Latex('%d$\\sigma:\\;$%.2f' % (_i, stats.norm.cdf(_i, loc=0, scale=1) * 100) + ' %'))
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A normal variable has mean 10 and SD 2. What z-score corresponds to 14, and which `scale` should SciPy receive?

    /// details | Solution to Exercise 1
    The z-score is $(14-10)/2=2$. Use `scale=2`, not variance 4. A density height is not a point probability; interval probability is area under the density.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The parameters `loc` and `scale` are optionals and represent mean and standard deviation of the distribution. The default is `loc=0` and `scale=1`.<br>
    A commonly used proportion is 95%. The value that results is this proportion can be found using the function `stats.norm.ppf()`. If we want to find the $\pm$ value for the interval that will result in 95% of the population inside, we have to consider that 2.5% of the population will stay out of the interval in each tail of the distribution. Because of that, the number we have to use with the `stats.norm.ppf()` is 0.975:
    """)
    return


@app.cell
def _(Latex, display, stats):
    print('Percent point function (inverse of CDF) of the normal distribution:')
    display(Latex(r'ppf(.975) = %.2f' % stats.norm.ppf(.975, loc=0, scale=1)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or we can use the function `stats.norm.interval` which already gives the interval:
    """)
    return


@app.cell
def _(stats):
    print('Confidence interval around the mean:')
    stats.norm.interval(confidence=0.95, loc=0, scale=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So, the interval $[\mu-1.96\sigma,\;\mu+1.96\sigma]$ contains 95% of the population.

    Now that we know how the probability density function of a normal distribution looks like, let's demonstrate the central limit theorem for a uniform distribution. For that, we will generate samples of a uniform distribution, calculate the mean across samples, and plot the histogram of the mean  across samples:
    """)
    return


@app.cell
def _(np, plt):
    _rc = {'axes.labelsize': 12, 'font.size': 12, 'legend.fontsize': 12, 'axes.titlesize': 12}
    plt.rcParams.update(**_rc)
    (_fig, _ax) = plt.subplots(1, 4, sharey=True, squeeze=True, figsize=(9, 3))
    _x = np.linspace(0, 1, 100)
    for (_i, n) in enumerate([1, 2, 3, 10]):
        _f = np.mean(np.random.random((1000, n)), 1)
        (m, s) = (np.mean(_f), np.std(_f, ddof=1))
        _fn = 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-(_x - m) ** 2 / (2 * s ** 2))
        _ax[_i].hist(_f, 20, density=True, color=[0, 0.2, 0.8, 0.6], edgecolor='black')
        _ax[_i].set_title('n=%d' % n, fontsize=12)
        _ax[_i].plot(_x, _fn, color=[1, 0, 0, 0.6], linewidth=2)
        _ax[_i].xaxis.label.set_size(12)
    plt.suptitle('Demonstration of the central limit theorem for a uniform distribution', fontsize=12, y=1.0)
    plt.tight_layout()
    plt.show()
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Does taking 10,000 samples of size 2 make the mean of two uniform observations exactly normal?

    /// details | Solution to Exercise 2
    No. More repetitions reveal the sampling distribution more clearly. The central limit theorem concerns increasing the number of independent observations per mean, under its assumptions.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Statistics for a sample of the population**

    Parameters (such as mean and variance) are characteristics of a population; statistics are the equivalent for a sample. A normal population is specified by its mean and variance. Two sample summaries do not completely describe an observed dataset; inspect its distribution too.

    The difference between sample and population is fundamental for the understanding of probability and statistics.<br>
    In Statistics, a sample is a set of data collected from a population. A population is usually very large and can't be accessed completely; all we have access is a sample (a smaller set) of the population.

    If we have only a sample of a finite univariate random variable with a normal distribution, both mean and variance of the population are unknown and they have to be estimated from the sample:

    $$
    \bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i
    $$

    $$
    s^2 = \frac{1}{N-1}\sum_{i=1}^{N} (x_i - \bar{x})^2
    $$

    The sample $\bar{x}$ and $s^2$ are only estimations of the unknown true mean and variance of the population, but because of the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), as the size of the sample increases, the sample mean and variance have an increased probability of being close to the population mean and variance.

    **Prediction interval around the mean**

    For a sample of a univariate random variable, the area in an interval of the probability density function can't be interpreted anymore as the proportion of the sample lying inside the interval. Rather, that area in the interval is a prediction of the probability that a new value from the population added to the sample will be inside the interval. This is called a [prediction interval](https://en.wikipedia.org/wiki/Prediction_interval). However, there is one more thing to correct. We have to adjust the interval limits for the fact that now we have only a sample of the population and the parameters $\mu$ and $\sigma$ are unknown and have to be estimated. This correction will increase the interval for the same probability value of the interval because we are not so certain about the distribution of the population.<br>
    To calculate the interval given a desired probability, we have to determine the distribution of the z-score equivalent for the case of a sample with unknown mean and variance:

    $$
    \frac{x_{n+i}-\bar{x}}{s\sqrt{1+1/n}}
    $$

    Where $x_{n+i}$ is the new observation for which we want to calculate the prediction interval.<br>
    The distribution of the ratio above is called <a href="https://en.wikipedia.org/wiki/Student's_t-distribution">Student's t-distribution</a> or simply $T$ distribution, with $n-1$ degrees of freedom. A $T$ distribution is symmetric and its pdf tends to that of the
    standard normal as $n$ tends to infinity.

    Then, the prediction interval around the sample mean for a new observation is:

    $$
    \left[\bar{x} - T_{n-1}\:s\:\sqrt{1+1/n},\quad \bar{x} + T_{n-1}\:s\:\sqrt{1+1/n}\right]
    $$

    Where $T_{n-1}$ is the $100((1+p)/2)^{th}$ percentile of the Student's t-distribution with n−1 degrees of freedom.

    For instance, the prediction interval with 95% of probability for a sample ($\bar{x}=0,\;s^2=1$) with size equals to 10 is:
    """)
    return


@app.cell
def _(np, stats):
    np.asarray(stats.t.interval(confidence=0.95, df=25-1, loc=0, scale=1)) * np.sqrt(1+1/10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For large degrees of freedom (e.g., 9999), the t quantile approaches the standard-normal quantile:
    """)
    return


@app.cell
def _(np, stats):
    np.asarray(stats.t.interval(confidence=0.95, df=10000-1, loc=0, scale=1)) * np.sqrt(1+1/10000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is a plot of the pdf for the normal distribution and the pdf for the Student's t-distribution with different number of degrees of freedom (n-1):
    """)
    return


@app.cell
def _(np, plt, stats):
    (_fig, _ax) = plt.subplots(1, 1, figsize=(8, 4))
    _x = np.linspace(-4, 4, 1000)
    _f = stats.norm.pdf(_x, loc=0, scale=1)
    t2 = stats.t.pdf(_x, df=2 - 1)
    t10 = stats.t.pdf(_x, df=10 - 1)
    t100 = stats.t.pdf(_x, df=100 - 1)
    _ax.plot(_x, _f, color='k', linestyle='--', lw=4, label='Normal')
    _ax.plot(_x, t2, color='r', lw=2, label='T (1)')
    _ax.plot(_x, t10, color='g', lw=2, label='T (9)')
    _ax.plot(_x, t100, color='b', lw=2, label='T (99)')
    _ax.legend(title='Distribution', fontsize=14)
    _ax.set_title("Normal and Student's t distributions", fontsize=12)
    _ax.set_xticks(np.linspace(-4, 4, 9))
    xtl = ['%+d$\\sigma$' % _i for _i in range(-4, 5, 1)]
    xtl[4] = '$\\mu$'
    _ax.set_xticklabels(xtl)
    _ax.set_ylim(-0.01, 0.41)
    plt.grid()
    plt.rc('font', size=12)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's common to use 1.96 as value for the 95% prediction interval even when dealing with a sample; let's quantify the error of this approximation for different sample sizes:
    """)
    return


@app.cell
def _(np, stats):
    _T = lambda n: stats.t.ppf(0.975, n - 1) * np.sqrt(1 + 1 / n)
    _N = stats.norm.ppf(0.975)
    for n_1 in [1000, 100, 10]:
        print('\nApproximation error for n = %d' % n_1)
        print('Using Normal distribution: %.1f%%' % (100 * (_N - _T(n_1)) / _T(n_1)))
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** For $n=10$, sample mean 0, SD 1, and $t_{9,0.975}=2.262$, compute the half-width for predicting one new observation.

    /// details | Solution to Exercise 3
    It is $2.262\sqrt{1+1/10}=2.373$. The factor includes variability of the new observation as well as uncertainty in the estimated mean.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For n=1000, the approximation is good, for n=10 it is bad, and it always underestimates.

    **Standard error of the mean**

    The [standard error of the mean](https://en.wikipedia.org/wiki/Standard_error) (sem) is the standard deviation of the sample-mean estimate of a population mean and is given by:

    $$
    sem = \frac{s}{\sqrt{n}}
    $$

    **Confidence interval**

    In statistics, a [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval) (CI) is a type of interval estimate of a population parameter and is used to indicate the reliability of an estimate ([Wikipedia](https://en.wikipedia.org/wiki/Confidence_interval)). For instance, the 95% confidence interval for the sample-mean estimate of a population mean is:

    $$
    \left[\bar{x} - T_{n-1}\:s/\sqrt{n},\quad \bar{x} + T_{n-1}\:s/\sqrt{n}\right]
    $$

    Where $T_{n-1}$ is the $100((1+p)/2)^{th}$ percentile of the Student's t-distribution with n−1 degrees of freedom.<br>
    For instance, the confidence interval for the mean with 95% of probability for a sample ($\bar{x}=0,\;s^2=1$) with size equals to 10 is:
    """)
    return


@app.cell
def _(np, stats):
    stats.t.interval(confidence=0.95, df=10-1, loc=0, scale=1) / np.sqrt(10)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** For the same sample, compute the confidence half-width for the population mean. Why is it smaller?

    /// details | Solution to Exercise 4
    It is $2.262/\sqrt{10}=0.715$. The target is the mean, not a new observation, so the extra observation variance is absent.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The 95% CI means that if we randomly obtain 100 samples of a population and calculate the CI of each sample (i.e., we replicate the experiment 99 times in a independent way), about 95% of these CIs would contain the population mean over many repetitions (the true mean). This is different from the prediction interval, which is larger, and gives the probability that a new observation is inside this interval. Note that the confidence interval DOES NOT give the probability that the true mean (the mean of the population) is inside this interval. The true mean is a parameter (fixed) and it is either inside the calculated interval or not; it is not a matter of chance (probability).

    Let's simulate samples of a population ~ $N(\mu=0, \sigma^2=1)$ and calculate the confidence interval for the samples' mean:
    """)
    return


@app.cell
def _(np, stats):
    n_2 = 20
    _x = np.random.randn(n_2, 100)
    m_1 = np.mean(_x, axis=0)
    s_1 = np.std(_x, axis=0, ddof=1)
    _T = stats.t.ppf(0.975, n_2 - 1)
    ci = m_1 + np.array([-s_1 * _T / np.sqrt(n_2), s_1 * _T / np.sqrt(n_2)])
    out = ci[0, :] * ci[1, :] > 0
    return ci, m_1, n_2, out, s_1


@app.cell
def _(ci, m_1, n_2, np, out, plt):
    (_fig, _ax) = plt.subplots(1, 1, figsize=(9, 5))
    ind = np.arange(1, 101)
    _ax.axhline(y=0, xmin=0, xmax=n_2 + 1, color=[0, 0, 0])
    _ax.plot([ind, ind], ci, color=[0, 0.2, 0.8, 0.8], marker='_', ms=0, linewidth=3)
    _ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=3)
    _ax.plot(ind, m_1, color=[0, 0.8, 0.2, 0.8], marker='.', ms=10, linestyle='')
    _ax.set_xlim(0, 101)
    _ax.set_ylim(-1.1, 1.1)
    _ax.set_title("Confidence interval for the samples' mean estimate of a population ~$N(0, 1)$", fontsize=14)
    _ax.set_xlabel('Sample (with %d observations)' % n_2, fontsize=14)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Count how many intervals miss the true mean in this run. About five misses per hundred are expected, but the observed count varies from run to run.

    And the standard deviation of the samples' mean per definition should be equal to the standard error of the mean:
    """)
    return


@app.cell
def _(m_1, np, s_1):
    print("Samples' mean and standard deviation:")
    print('m = %.3f   s = %.3f' % (np.mean(m_1), np.mean(s_1)))
    print("Standard deviation of the samples' mean:")
    print('%.3f' % np.std(m_1, ddof=1))
    print('Standard error of the mean:')
    print('%.3f' % (np.mean(s_1) / np.sqrt(20)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Likewise, it's common to use 1.96 for the 95% confidence interval even when dealing with a sample; let's quantify the error of this approximation for different sample sizes:
    """)
    return


@app.cell
def _(stats):
    _T = lambda n: stats.t.ppf(0.975, n - 1)
    _N = stats.norm.ppf(0.975)
    for n_3 in [1000, 100, 10]:
        print('\nApproximation error for n = %d' % n_3)
        print('Using Normal distribution: %.1f%%' % (100 * (_N - _T(n_3)) / _T(n_3)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For n=1000, the approximation is good, for n=10 it is bad, and it always underestimates.

    For the case of a multivariate random variable, see [Prediction ellipse and prediction ellipsoid](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/prediction_ellipse_ellipsoid.py).
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 5.** If sample size rises from 25 to 100 with similar SD, what happens to SEM and to the large-sample prediction width?

    /// details | Solution to Exercise 5
    SEM changes from $s/5$ to $s/10$, approximately halving the mean’s interval. The prediction factor changes only from $\sqrt{1.04}$ to $\sqrt{1.01}$; individual variability remains.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Verification of the relation between standard deviation and standard error of the mean
    """)
    return


@app.cell
def _(np, plt):
    _rc = {'axes.labelsize': 12, 'font.size': 12, 'legend.fontsize': 12, 'axes.titlesize': 12}
    plt.rcParams.update(**_rc)
    (_fig, _ax) = plt.subplots(1, 5, sharey=True, squeeze=True, figsize=(9, 3))
    print('{0:10} {1:10} {2:5} {3:10}'.format('n', 'm', 's', '1/np.sqrt(n)'))
    for (_i, n_4) in enumerate([1, 2, 3, 10, 100]):
        _f = np.mean(np.random.randn(1000, n_4), 1)
        (m_2, s_2) = (np.mean(_f), np.std(_f, ddof=1))
        _x = np.linspace(np.min(_f), np.max(_f), 100)
        _fn = 1 / (s_2 * np.sqrt(2 * np.pi)) * np.exp(-(_x - m_2) ** 2 / (2 * s_2 ** 2))
        _ax[_i].hist(_f, 20, density=True, color=[0, 0.2, 0.8, 0.6], edgecolor='black')
        _ax[_i].set_title('n=%d' % n_4, fontsize=12)
        _ax[_i].plot(_x, _fn, color=[1, 0, 0, 0.6], linewidth=2)
        _ax[_i].xaxis.label.set_size(12)
        print('{0:3} {1:10.4f} {2:10.4f} {3:10.4f}'.format(n_4, m_2, s_2, 1 / np.sqrt(n_4)))
    plt.suptitle('Standard deviation and standard error of the mean', fontsize=12, y=1.0)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, plt):
    _rc = {'axes.labelsize': 12, 'font.size': 12, 'legend.fontsize': 12, 'axes.titlesize': 12}
    plt.rcParams.update(**_rc)
    (_fig, _ax) = plt.subplots(1, 5, sharey=True, squeeze=True, figsize=(9, 3))
    print('{0:10} {1:10} {2:5} {3:10}'.format('n', 'm', 's', '1/np.sqrt(n)'))
    for (_i, n_5) in enumerate([1, 2, 3, 10, 100]):
        _f = np.mean(np.random.randn(100, n_5), 1)
        (m_3, s_3) = (np.mean(_f), np.std(_f, ddof=1))
        _x = np.linspace(np.min(_f), np.max(_f), 100)
        _fn = 1 / (s_3 * np.sqrt(2 * np.pi)) * np.exp(-(_x - m_3) ** 2 / (2 * s_3 ** 2))
        _ax[_i].hist(_f, 10, density=True, color=[0, 0.2, 0.8, 0.6], edgecolor='black')
        _ax[_i].set_title('n=%d' % n_5, fontsize=12)
        _ax[_i].plot(_x, _fn, color=[1, 0, 0, 0.6], linewidth=2)
        _ax[_i].xaxis.label.set_size(12)
        print('{0:3} {1:10.4f} {2:10.4f} {3:10.4f}'.format(n_5, m_3, s_3, 1 / np.sqrt(n_5)))
    plt.suptitle('Standard deviation and standard error of the mean', fontsize=12, y=1.0)
    plt.tight_layout()
    plt.show()
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. What does a 95% confidence procedure cover over repeated samples?
    2. Which interval predicts a new independent observation?
    3. What assumptions make the finite-sample t formulas exact?

    /// details | Checkpoint answers
    1. The fixed population mean in 95% of repetitions in the long run.
    2. A prediction interval.
    3. Independent observations from a normal population, with the future observation drawn independently from the same population.
    ///

    ## Problems

    1. With $n=25$, mean 10, SD 2, and $t_{24,0.975}=2.064$, give the 95% CI.
    2. Give the corresponding 95% prediction interval.
    3. If $n$ becomes very large, does the prediction interval collapse to the mean?

    /// details | Final answers
    1. Approximately [9.174, 10.826].
    2. Approximately [5.790, 14.210].
    3. No.
    ///

    ## Go deeper

    [NIST confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section1/prc14.htm) — repeated-sampling interpretation. [Prediction ellipses](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/prediction_ellipse_ellipsoid.py) extends prediction to several variables.

    ### Video lectures on the Internet

    [MIT Statistics for Applications lectures](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/video_galleries/lecture-videos/) — use the lectures on estimation and confidence intervals to connect the numerical examples with repeated sampling.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Hahn GJ, Meeker WQ (1991) [Statistical Intervals: A Guide for Practitioners](https://books.google.com.br/books?id=ADGuRxqt5z4C). John Wiley & Sons.

    - Montgomery (2013) [Applied Statistics and Probability for Engineers](https://books.google.com.br/books?id=_f4KrEcNAfEC). John Wiley & Sons.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
