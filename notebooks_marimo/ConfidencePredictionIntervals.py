import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Confidence and prediction intervals

        > Marcos Duarte  
        > Laboratory of Biomechanics and Motor Control ([http://demotu.org/](http://demotu.org/))  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For a finite univariate random variable with a normal probability distribution, the mean$\mu$(a measure of central tendency) and variance$\sigma^2$(a measure of dispersion) of a population are the well known formulas:$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2$For a more general case, a continuous univariate random variable$x$with [probability density function](http://en.wikipedia.org/wiki/Probability_density_function) (PDF),$f(x)$, the mean and variance of a population are:$\mu = \int_{\infty}^{\infty} x f(x)\: dx$$\sigma^2 = \int_{\infty}^{\infty} (x-\mu)^2 f(x)\: dx$The PDF is a function that describes the relative likelihood for the random variable to take on a given value.   
        Mean and variance are the first and second central moments of a random variable. The standard deviation$\sigma$of the population is the square root of the variance.

        The [normal (or Gaussian) distribution](http://en.wikipedia.org/wiki/Normal_distribution) is a very common and useful distribution, also because of the [central limit theorem](http://en.wikipedia.org/wiki/Central_limit_theorem), which states that for a sufficiently large number of samples (each with many observations) of an independent random variable with an arbitrary probability distribution, the means of the samples will have a normal distribution. That is, even if the underlying probability distribution of a random variable is not normal, if we sample enough this variable, the means of the set of samples will have a normal distribution. 

        The probability density function of a univariate normal (or Gaussian) distribution is:$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr)$The only parameters that define the normal distribution are the mean$\mu$and the variance$\sigma^2$, because of that a normal distribution is usually described as$N(\mu,\:\sigma^2)$.

        Here is a plot of the PDF for the normal distribution:
        """
    )
    return


@app.cell
def _():
    # import the necessary libraries
    import numpy as np
    # '%matplotlib notebook' command supported automatically in marimo
    import matplotlib.pyplot as plt
    from IPython.display import display, Latex
    from scipy import stats
    import sys
    sys.path.insert(1, r'./../functions')  # directory of BMC Python functions
    return Latex, display, np, plt, stats


@app.cell
def _(plt):
    from pdf_norm_plot import pdf_norm_plot
    pdf_norm_plot()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The horizontal axis above is shown in terms of the number of standard deviations in relation to the mean, which is known as standard score or$z$score:$z = \frac{x - \mu}{\sigma}$So, instead of specifying raw values in the distribution, we define the PDF in terms of$z$scores; this conversion process is called standardizing the distribution (and the result is known as standard normal distribution). Note that because$\mu$and$\sigma$are known  parameters,$z$has the same distribution as$x$, in this case, the normal distribution.   

        The percentage numbers in the plot are the probability (area under the curve) for each interval shown in the horizontal label.  
        An interval in terms of z score is specified as:$[\mu-z\sigma,\;\mu+z\sigma]$.  
        The interval$[\mu-1\sigma,\;\mu+1\sigma]$contains 68.3% of the population and the interval$[\mu-2\sigma,\;\mu+2\sigma]$contains 95.4% of the population.   
        These numbers can be calculated using the function `stats.norm.cdf()`, the [cumulative distribution function](http://en.wikipedia.org/wiki/Cumulative_distribution_function) (CDF) of the normal distribution at a given value:
        """
    )
    return


@app.cell
def _(Latex, display, stats):
    print('Cumulative distribution function (cdf) of the normal distribution:')
    for _i in range(-3, 4):
        display(Latex('%d$\\sigma:\\;$%.2f' % (_i, stats.norm.cdf(_i, loc=0, scale=1) * 100) + ' %'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The parameters `loc` and `scale` are optionals and represent mean and variance of the distribution. The default is `loc=0` and `scale=1`.   
        A commonly used proportion is 95%. The value that results is this proportion can be found using the function `stats.norm.ppf()`. If we want to find the$\pm$value for the interval that will result in 95% of the population inside, we have to consider that 2.5% of the population will stay out of the interval in each tail of the distribution. Because of that, the number we have to use with the `stats.norm.ppf()` is 0.975:
        """
    )
    return


@app.cell
def _(Latex, display, stats):
    print('Percent point function (inverse of CDF) of the normal distribution:')
    display(Latex(r'ppf(.975) = %.2f' % stats.norm.ppf(.975, loc=0, scale=1)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or we can use the function `stats.norm.interval` which already gives the interval:
        """
    )
    return


@app.cell
def _(stats):
    print('Confidence interval around the mean:')
    stats.norm.interval(alpha=0.95, loc=0, scale=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So, the interval$[\mu-1.96\sigma,\;\mu+1.96\sigma]$contains 95% of the population.

        Now that we know how the probability density function of a normal distribution looks like, let's demonstrate the central limit theorem for a uniform distribution. For that, we will generate samples of a uniform distribution, calculate the mean across samples, and plot the histogram of the mean  across samples:
        """
    )
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
    mo.md(
        r"""
        **Statistics for a sample of the population**

        Parameters (such as mean and variance) are characteristics of a population; statistics are the equivalent for a sample. For a population and a sample with normal or Gaussian distribution, mean and variance is everything we need to completely characterize this population or sample.     

        The difference between sample and population is fundamental for the understanding of probability and statistics.   
        In Statistics, a sample is a set of data collected from a population. A population is usually very large and can't be accessed completely; all we have access is a sample (a smaller set) of the population.   

        If we have only a sample of a finite univariate random variable with a normal distribution, both mean and variance of the population are unknown and they have to be estimated from the sample:$\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$$s^2 = \frac{1}{N-1}\sum_{i=1}^{N} (x_i - \bar{x})^2$The sample$\bar{x}$and$s^2$are only estimations of the unknown true mean and variance of the population, but because of the [law of large numbers](http://en.wikipedia.org/wiki/Law_of_large_numbers), as the size of the sample increases, the sample mean and variance have an increased probability of being close to the population mean and variance.

        **Prediction interval around the mean**

        For a sample of a univariate random variable, the area in an interval of the probability density function can't be interpreted anymore as the proportion of the sample lying inside the interval. Rather, that area in the interval is a prediction of the probability that a new value from the population added to the sample will be inside the interval. This is called a [prediction interval](http://en.wikipedia.org/wiki/Prediction_interval). However, there is one more thing to correct. We have to adjust the interval limits for the fact that now we have only a sample of the population and the parameters$\mu$and$\sigma$are unknown and have to be estimated. This correction will increase the interval for the same probability value of the interval because we are not so certain about the distribution of the population.  
        To calculate the interval given a desired probability, we have to determine the distribution of the z-score equivalent for the case of a sample with unknown mean and variance:$\frac{x_{n+i}-\bar{x}}{s\sqrt{1+1/n}}$Where$x_{n+i}$is the new observation for which we want to calculate the prediction interval.  
        The distribution of the ratio above is called <a href="http://en.wikipedia.org/wiki/Student's_t-distribution">Student's t-distribution</a> or simply$T$distribution, with$n-1$degrees of freedom. A$T$distribution is symmetric and its pdf tends to that of the
        standard normal as$n$tends to infinity. 

        Then, the prediction interval around the sample mean for a new observation is:$\left[\bar{x} - T_{n-1}\:s\:\sqrt{1+1/n},\quad \bar{x} + T_{n-1}\:s\:\sqrt{1+1/n}\right]$Where$T_{n-1}$is the$100((1+p)/2)^{th}$percentile of the Student's t-distribution with n−1 degrees of freedom.

        For instance, the prediction interval with 95% of probability for a sample ($\bar{x}=0,\;s^2=1$) with size equals to 10 is:
        """
    )
    return


@app.cell
def _(np, stats):
    np.asarray(stats.t.interval(alpha=0.95, df=25-1, loc=0, scale=1)) * np.sqrt(1+1/10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For a large sample (e.g., 10000), the interval approaches the one for a normal distribution (according to the [central limit theorem](http://en.wikipedia.org/wiki/Central_limit_theorem)):
        """
    )
    return


@app.cell
def _(np, stats):
    np.asarray(stats.t.interval(alpha=0.95, df=10000-1, loc=0, scale=1)) * np.sqrt(1+1/10000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here is a plot of the pdf for the normal distribution and the pdf for the Student's t-distribution with different number of degrees of freedom (n-1):
        """
    )
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
    mo.md(
        r"""
        It's common to use 1.96 as value for the 95% prediction interval even when dealing with a sample; let's quantify the error of this approximation for different sample sizes: 
        """
    )
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
    mo.md(
        r"""
        For n=1000, the approximation is good, for n=10 it is bad, and it always underestimates. 

        **Standard error of the mean**

        The [standard error of the mean](http://en.wikipedia.org/wiki/Standard_error) (sem) is the standard deviation of the sample-mean estimate of a population mean and is given by:$sem = \frac{s}{\sqrt{n}}$**Confidence interval**

        In statistics, a [confidence interval](http://en.wikipedia.org/wiki/Confidence_interval) (CI) is a type of interval estimate of a population parameter and is used to indicate the reliability of an estimate ([Wikipedia](http://en.wikipedia.org/wiki/Confidence_interval)). For instance, the 95% confidence interval for the sample-mean estimate of a population mean is:$\left[\bar{x} - T_{n-1}\:s/\sqrt{n},\quad \bar{x} + T_{n-1}\:s/\sqrt{n}\right]$Where$T_{n-1}$is the$100((1+p)/2)^{th}$percentile of the Student's t-distribution with n−1 degrees of freedom.   
        For instance, the confidence interval for the mean with 95% of probability for a sample ($\bar{x}=0,\;s^2=1$) with size equals to 10 is:
        """
    )
    return


@app.cell
def _(np, stats):
    stats.t.interval(alpha=0.95, df=10-1, loc=0, scale=1) / np.sqrt(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The 95% CI means that if we randomly obtain 100 samples of a population and calculate the CI of each sample (i.e., we replicate the experiment 99 times in a independent way), 95% of these CIs should contain the population mean (the true mean). This is different from the prediction interval, which is larger, and gives the probability that a new observation is inside this interval. Note that the confidence interval DOES NOT give the probability that the true mean (the mean of the population) is inside this interval. The true mean is a parameter (fixed) and it is either inside the calculated interval or not; it is not a matter of chance (probability).    

        Let's simulate samples of a population ~$N(\mu=0, \sigma^2=1)$and calculate the confidence interval for the samples' mean:
        """
    )
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
    mo.md(
        r"""
        Four out of 100 95%-CI's don't contain the population mean, about what we predicted.

        And the standard deviation of the samples' mean per definition should be equal to the standard error of the mean:
        """
    )
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
    mo.md(
        r"""
        Likewise, it's common to use 1.96 for the 95% confidence interval even when dealing with a sample; let's quantify the error of this approximation for different sample sizes: 
        """
    )
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
    mo.md(
        r"""
        For n=1000, the approximation is good, for n=10 it is bad, and it always underestimates. 

        For the case of a multivariate random variable, see [Prediction ellipse and prediction ellipsoid](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/PredictionEllipseEllipsoid.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Verification of the relation between standard deviation and standard error of the mean
        """
    )
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
    mo.md(
        r"""
        ## References

        - Hahn GJ, Meeker WQ (1991) [Statistical Intervals: A Guide for Practitioners](http://books.google.com.br/books?id=ADGuRxqt5z4C). John Wiley & Sons.  
        - Montgomery (2013) [Applied Statistics and Probability for Engineers](http://books.google.com.br/books?id=_f4KrEcNAfEC). John Wiley & Sons.  
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
