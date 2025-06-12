import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Curve fitting

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
        Curve fitting is the process of fitting a model, expressed in terms of a mathematical function, that depends on adjustable parameters to a series of data points and once adjusted, that curve has the best fit to the data points.   

        The model can be an arbitrary class of functions such as polynomials and the fit would determine the polynomial coefficients to simplily summarize the data or the model parameters can represent a underlying theory that the data are supposed to satisfy such as fitting an exponential function to data of a decay process to determine its decay rate or a parabola to the position data of an object falling to determine the gravity acceleration.

        The general approach to the fitting procedure involves the definition of a merit function that measures the agreement between data and model. The model parameters are then adjusted to yield the best-fit parameters as a problem of minimization. A fitting procedure should provide (i) parameters, (ii) error estimates on the parameters, and (iii) a statistical measure of goodness-of-fit. When the third item suggests that the model is an unlikely match to the data, then items (i) and (ii) are probably worthless.   
        (Numerical Recipes 2007, Bevington and Robinson 2002)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Least squares

        Consider$n$data points$(x_i, y_i),\: i=1, \dots , n$, where$x_i$is the independent variable (or predictor) and$y_i$is the dependent variable (or response) to be fitted by a model function$y$with$m$adjustable parameters$\beta_i,\: i=1, \dots , m$. The problem is to find the parameter values for the model which best fits the data. A classical solution is to find the best fit by minimizing the sum of the squared differences between data points and the model function (the sum of squared residuals as the merit function), which is known as the least-squares fit:$\sum_{i=1}^{n} \left[ y_i - y(x | \beta_1 \dots \beta_{m}) \right]^2 \;\;\;\;\;\; \mathrm{minimize\; over:} \;\;\; \beta_1 \dots \beta_{m}$**Chi-Square**   
        If we consider that each response$y_i$has a measurement error or uncertainty described by a standard deviation,$\sigma_i$, the problem now is to minimize the following function:$\sum_{i=1}^{n} \left[ \frac{ y_i - y(x | \beta_1 \dots \beta_{m}) }{\sigma_i} \right]^2 = \chi^2$Considering that the residuals are normally distributed, the sum of squared residuals divided by their variance,$\sigma_i^2$, by definition will have a [chi-squared distribution](http://en.wikipedia.org/wiki/Chi-squared_distribution),$\chi^2$. Once the best-fit parameters are found, the terms in the sum above are not all statistically independent and the probability distribution of$\chi^2$will be the chi-squared distribution for$n-m$degrees of freedom.

        The uncertainty$\sigma_i$can be seen as the inverse of the weights in a weighted sum (because less certainty we have about this measure). Larger$\sigma_i$, smaller the weight of$y_i$in the sum. If$y_i$has no uncertainty,$\sigma_i$should be equal to one.   

        A rough estimate of the goodness of fit is the reduced chi-square statistic,$\chi^2_{red}$: the$\chi^2$value divided by the number of degrees of freedom ($n-m$).   
        A good fitting should have$\chi^2_{red}$equals to one.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear fit

        Let's derive the analytical expression for the least-square fit when the model function is a straight line (a.k.a. linear regression) and the dependent variable$y_i$has uncertainty:$y(x) = y(x | a,b) = a + bx$We want to find$a, b$such that minimizes the$\chi^2$function defined above (a.k.a.$\chi^2$fitting):$\chi^2(a,b) = \sum_{i=1}^{n} \left[ \frac{ y_i - (a + bx_i) }{\sigma_i} \right]^2$Using the property that at the minimum of$\chi^2$its derivative is zero:$\frac{\partial \chi^2}{\partial a} = -2 \sum_{i=1}^{n} \frac{ y_i - a - bx_i }{\sigma_i^2} = 0$$\frac{\partial \chi^2}{\partial b} = -2 \sum_{i=1}^{n} \frac{ x_i(y_i - a - bx_i) }{\sigma_i^2} = 0$To solve these two equations, let's define the sums as:$S = \sum_{i=1}^{n} \frac{1}{\sigma_i^2} \;\;\; S_x = \sum_{i=1}^{n} \frac{x_i}{\sigma_i^2} \;\;\; S_y = \sum_{i=1}^{n} \frac{y_i}{\sigma_i^2} \;\;\; S_{xx} = \sum_{i=1}^{n} \frac{x_i^2}{\sigma_i^2} \;\;\; S_{xy} = \sum_{i=1}^{n} \frac{x_i y_i}{\sigma_i^2}$Using these definitions, the former two equations become:$S_y \:\: = aS + bS_x$$S_{xy} = aS_x + bS_{xx}$And solving these two equations for the two unknowns:$a = \frac{S_{xx}S_y - S_x S_{xy}}{\Delta}$$b = \frac{S S_{xy} - S_x S_y}{\Delta}$Where:$\Delta = S S_{xx} - S_x^2$With the parameters above, the straight line will be the best fit in the sense that the sum of the squared residuals are minimum.   

        **Estimating the uncertainty of the parameters**

        The uncertainty of each parameter is given by:$\sigma_a^2 = \frac{Sxx}{\Delta}$$\sigma_b^2 = \frac{S}{\Delta}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Correlation coefficient

        The Pearson product-moment correlation coefficient, or simply the correlation coefficient, is a measure of the linear correlation between two variables$x$and$y$, with values varying from +1 to −1, where 1 is total positive correlation, 0 is no correlation, and −1 is total negative correlation.   
        The correlation coefficient between populations of two random variables is the covariance of the two variables divided by the product of their standard deviations:$\rho_{x, y} = \frac{cov(x, y)}{\sigma_x\sigma_y} = \frac{E[(x-\mu_x)(y-\mu_y)]}{\sqrt{E[(x-\mu_x)^2]}\sqrt{E[(y-\mu_y)^2]}}$Where$E[\cdot]$is the <a href="http://en.wikipedia.org/wiki/Moment_(mathematics)">expectation operator</a>.

        For samples of two random variables, the covariance and standard deviation are given by:$cov(x, y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})$$\sigma_x = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}$So, the correlation coefficient for the samples is:$r_{x, y} = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}$The square of the sample correlation coefficient, denoted$r^2$or$R^2$, is called the coefficient of determination and it can be shown it is related to the linear fit formalism by:$R^2(y, \widehat{y}) = \frac{\sum_{i=1}^{n}(\widehat{y}_i-\bar{y})^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}$Where$\widehat{y}_i$are the fitted values from the linear fit.

        Examining the division above, it consists of the variance of the fitted values around the mean value of$y$divided by the variance of$y_i$. Because of that, it is said that the coefficient of determination is the proportion of variance in$y$explained by a linear function of$x$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Computing the curve fitting

        Python and its ecosystem for scientific computing have plenty of functions ready available for curve fitting. Instead of writting our own code to implement the formula above, let's use the functions available which will cover many more cases (general polynomials, nonlinear functions, etc.).

        First, if we only want to fit polynomials, we can use the Numpy polyfit function:   

            polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)   
                Least squares polynomial fit.   
        
                Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`   
                to points `(x, y)`. Returns a vector of coefficients `p` that minimises   
                the squared error.  
        
        Let's demonstrate how polyfit works:
        """
    )
    return


@app.cell
def _():
    # Import the necessary libraries
    import numpy as np
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    return np, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Generate some data:
        """
    )
    return


@app.cell
def _(np):
    n = 50
    x = np.arange(1, n+1)
    y = x + 10*np.random.randn(n) + 10
    yerr = np.abs(10*np.random.randn(n)) + 5
    return n, x, y, yerr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        First, let's plot the data and perform the curve fitting without considering the uncertainty:
        """
    )
    return


@app.cell
def _(n, np, plt, x, y, yerr):
    (p, _cov) = np.polyfit(x, y, 1, cov=True)
    yfit = np.polyval(p, x)
    perr = np.sqrt(np.diag(_cov))
    R2 = np.corrcoef(x, y)[0, 1] ** 2
    resid = y - yfit
    chi2red = np.sum((resid / yerr) ** 2) / (y.size - 2)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'bo')
    plt.plot(x, yfit, linewidth=3, color=[1, 0, 0, 0.5])
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.title('$y = %.2f \\pm %.2f + (%.2f \\pm %.2f)x \\; [R^2=%.2f,\\, \\chi^2_{red}=%.1f]$' % (p[1], perr[1], p[0], perr[0], R2, chi2red), fontsize=20, color=[0, 0, 0])
    plt.xlim((0, n + 1))
    plt.ylim((-50, 100))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The curve fitting by a line considering the uncertainty:
        """
    )
    return


@app.cell
def _(np, x, y, yerr):
    (p_1, _cov) = np.polyfit(x, y, 1, w=1 / yerr, cov=True)
    yfit_1 = np.polyval(p_1, x)
    perr_1 = np.sqrt(np.diag(_cov))
    R2_1 = np.corrcoef(x, y)[0, 1] ** 2
    resid_1 = y - yfit_1
    chi2red_1 = np.sum((resid_1 / yerr) ** 2) / (y.size - 2)
    return R2_1, chi2red_1, p_1, perr_1, resid_1, yfit_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the plot:
        """
    )
    return


@app.cell
def _(R2_1, chi2red_1, n, p_1, perr_1, plt, x, y, yerr, yfit_1):
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, y, yerr=yerr, fmt='bo', ecolor='b', capsize=0)
    plt.plot(x, yfit_1, linewidth=3, color=[1, 0, 0, 0.5])
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.title('$y = %.2f \\pm %.2f + (%.2f \\pm %.2f)x \\; [R^2=%.2f,\\, \\chi^2_{red}=%.1f]$' % (p_1[1], perr_1[1], p_1[0], perr_1[0], R2_1, chi2red_1), fontsize=20, color=[0, 0, 0])
    plt.xlim((0, n + 1))
    plt.ylim((-50, 100))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        According to our assumptions, the residuals should be normally distributed.  Let's create a function to plot the residuals and a probability plot that will be useful to check if the residuals are normally distributed:
        """
    )
    return


@app.cell
def _(plt, stats):
    def plot_resid(x, resid):
        """ plot residuals and probability plot of residuals for a normal distribution."""
        (_fig, _ax) = plt.subplots(1, 2, figsize=(13, 5))
        _ax[0].plot(x, resid, 'ro')
        _ax[0].plot([0, x[-1]], [0, 0], 'k')
        _ax[0].set_xlabel('x', fontsize=12)
        _ax[0].set_ylabel('Residuals', fontsize=12)
        stats.probplot(resid, dist='norm', plot=plt)
        _ax[1].set_xlabel('Quantiles', fontsize=12)
        _ax[1].set_ylabel('Ordered values', fontsize=12)
        _ax[1].set_title('Probability Plot of the residuals')
        plt.show()
    return (plot_resid,)


@app.cell
def _(plot_resid, resid_1, x):
    plot_resid(x, resid_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We could apply a statistical test to verify if the data above is normallly distributed, but for now, by visual inspection, the residuals indeed seem to be normallly distributed.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The effect of uncertainty on the curve fitting

        To demonstrate the effect of uncertainty on the curve fitting, let's plot the same (x, y) values but with different errors:
        """
    )
    return


@app.cell
def _(np):
    x_1 = np.array([1, 2, 3, 4, 5])
    y_1 = np.array([1, 2, 3, 6, 4])
    yerr_1 = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0.5, 2], [1, 1, 1, 2, 0.5]])
    return x_1, y_1, yerr_1


@app.cell
def _(np):
    def linearfit(x, y, yerr=None):
        w = None if yerr is None or np.sum(yerr) == 0 else 1 / yerr
        (p, _cov) = np.polyfit(x, y, 1, w=w, cov=True)
        yfit = np.polyval(p, x)
        perr = np.sqrt(np.diag(_cov))
        R2 = np.corrcoef(x, y)[0, 1] ** 2
        resid = y - yfit
        chi2red = np.sum((resid / yerr) ** 2) / (y.size - 2) if w is not None else np.nan
        return (yfit, p, R2, chi2red, perr, resid)
    return (linearfit,)


@app.cell
def _(linearfit, plt, x_1, y_1, yerr_1):
    (_fig, _ax) = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    _axs = _ax.flatten()
    for (_i, _ax) in enumerate(_axs):
        (_yf, p_2, R2_2, chi2red_2, perr_2, resid_2) = linearfit(x_1, y_1, yerr=yerr_1[_i, :])
        _ax.errorbar(x_1, y_1, yerr=yerr_1[_i, :], fmt='bo', ecolor='b', capsize=0, elinewidth=2)
        _ax.plot(x_1, _yf, linewidth=3, color=[1, 0, 0, 0.5])
        _ax.set_title('$y = %.2f + %.2f x \\, [R^2=%.2f,\\chi^2_{red}=%.1f]$' % (p_2[1], p_2[0], R2_2, chi2red_2), fontsize=18)
        _ax.grid()
    _ax.set_xlim(0.5, 5.5)
    _fig.subplots_adjust(bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.05)
    plt.suptitle('The effect of uncertainty on the curve fitting (same data, different errors)', fontsize=16, y=1.02)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        From figure above, if the errors (weights) are all equal, the fitting is the same as if we don't input any error (first line).   
        When the errors are different across data, the uncertainty has a strong impact on the curve fitting (second line).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Curve fitting as a model

        We have to be carefull in interpreting the results of a curve fitting to determine whether the fitted model truely captures the relation between the independent variable (predictor) and the dependent variable (response).

        An illustrative example to demonstrate that the result of a curve fitting is not necessarily an indicator of the phenomenon being modelled is the [Anscombe's quartet](http://en.wikipedia.org/wiki/Anscombe%27s_quartet) data. These four sets of data have very similar basic statistical properties and linear fitting parameters, but are very different when visualized. Let's work with these data:
        """
    )
    return


@app.cell
def _(np):
    x_2 = np.array([[10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]])
    y_2 = np.array([[8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68], [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74], [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89]])
    print("The Anscombe's quartet data ha)ve the same basic statistical properties:")
    print('Mean of x    :', np.mean(y_2, axis=1))
    print('Variance of x:', np.var(y_2, axis=1))
    print('Mean of y    :', np.mean(y_2, axis=1))
    print('Variance of y:', np.var(y_2, axis=1))
    return x_2, y_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or we can use Pandas to describe the data:
        """
    )
    return


@app.cell
def _(np, x_2, y_2):
    import pandas as pd
    df = pd.DataFrame(np.vstack((x_2, y_2)).T, columns=['X1', 'X2', 'X3', 'X4', 'Y1', 'Y2', 'Y3', 'Y4'])
    df.describe()
    return


@app.cell
def _(linearfit, np, plt, x_2, y_2):
    (_fig, _ax) = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    _axs = _ax.flatten()
    resid_3 = np.empty_like(y_2)
    for (_i, _ax) in enumerate(_axs):
        (_yf, p_3, R2_3, chi2red_3, perr_3, resid_3[_i, :]) = linearfit(x_2[_i, :], y_2[_i, :], yerr=None)
        _ax.plot(x_2[_i, :], y_2[_i, :], color=[0, 0.2, 1, 0.8], marker='o', linestyle='', markersize=8)
        _ax.plot(x_2[_i, :], _yf, linewidth=3, color=[1, 0, 0, 0.8])
        _ax.set_title('$y = %.2f + %.2f x \\, [R^2=%.2f]$' % (p_3[1], p_3[0], R2_3), fontsize=18)
        _ax.grid()
    _ax.set_xlim(0, 20)
    _fig.subplots_adjust(bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.05)
    plt.suptitle("Linear fit of the Anscombe's quartet data", fontsize=18, y=1.02)
    plt.show()
    return (resid_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And we can check whether the residuals of the linear fit are normally distributed:
        """
    )
    return


@app.cell
def _(plt, resid_3, stats):
    _fig = plt.figure(figsize=(10, 6))
    for _i in range(4):
        _ax = plt.subplot(2, 2, _i + 1)
        stats.probplot(resid_3[_i, :], dist='norm', plot=plt)
    plt.suptitle('Probability plot of the residuals for a normal distribution', fontsize=18, y=1.05)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Even the residuals don't look bad!  
        Exactly the same model fits very different data.   
        We should be very carefull in interpreting the result of a curve fitting as a description of a phenomenon.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Confidence and prediction intervals for the linear fit

        Analog to the case for a random variable (see [Confidence and prediction intervals](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/ConfidencePredictionIntervals.ipynb)), we can estimate confidence and prediction intervals for the linear fit (for the deduction of these intervals see for example Montgomery (2013)).   

        **Confidence interval**   

        A 95% confidence interval for the linear fit gives the 95% probability that this interval around the linear fit,$\hat{\mu}_{y|x0}$, contains the mean response of new values,$\mu_{y|x0}$, at a specified value,$x_0$, and it is given by:$\left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}$Where:$\hat{\mu}_{y|x0} = a + bx_0$is computed from the lineat fit.$T_{n-2}^{.975}$is the$97.5^{th}$percentile of the Student's t-distribution with n−2 degrees of freedom.$\hat{\sigma}$is the standard deviation of the error term in the linear fit (residuals) given by:$\hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}$**Prediction interval**   

        A 95% prediction interval for the linear fit gives the 95% probability that this interval around the linear fit,$\hat{y}_0$, contains a new observation,$y_0$, at a specified value,$x_0$, and it is given by:$\left| \: \hat{y}_0 - y_0 \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{1 + \frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}$Where$\hat{y}_0 = a + bx_0$is computed from the lineat fit.

        **Implementation in Python**

        Here is an implementation of the linear fit with the confidence and prediction intervals:
        """
    )
    return


@app.cell
def _(np, plt, stats):
    def linearfit_1(x, y, yerr):
        """Linear fit of x and y with uncertainty and plots results."""
        (x, y) = (np.asarray(x), np.asarray(y))
        n = y.size
        (p, _cov) = np.polyfit(x, y, 1, w=1 / yerr, cov=True)
        yfit = np.polyval(p, x)
        perr = np.sqrt(np.diag(_cov))
        R2 = np.corrcoef(x, y)[0, 1] ** 2
        resid = y - yfit
        chi2red = np.sum((resid / yerr) ** 2) / (n - 2)
        s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
        t = stats.t.ppf(0.975, n - 2)
        ci = t * s_err * np.sqrt(1 / n + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        pi = t * s_err * np.sqrt(1 + 1 / n + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        plt.figure(figsize=(10, 5))
        plt.fill_between(x, yfit + pi, yfit - pi, color=[1, 0, 0, 0.1], edgecolor=None)
        plt.fill_between(x, yfit + ci, yfit - ci, color=[1, 0, 0, 0.15], edgecolor=None)
        plt.errorbar(x, y, yerr=yerr, fmt='bo', ecolor='b', capsize=0)
        plt.plot(x, yfit, linewidth=3, color=[1, 0, 0, 0.8])
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.title('$y = %.2f \\pm %.2f + (%.2f \\pm %.2f)x \\; [R^2=%.2f,\\, \\chi^2_{red}=%.1f]$' % (p[1], perr[1], p[0], perr[0], R2, chi2red), fontsize=20, color=[0, 0, 0])
        plt.xlim((0, n + 1))
        plt.show()
    return (linearfit_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And an example:
        """
    )
    return


@app.cell
def _(linearfit_1, np):
    n_1 = 20
    x_3 = np.arange(1, n_1 + 1)
    y_3 = x_3 + 5 * np.random.randn(n_1)
    yerr_2 = np.abs(4 * np.random.randn(n_1)) + 2
    linearfit_1(x_3, y_3, yerr_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Non-linear curve fitting

        A more general curve fitting function is the `scipy.optimize.curve_fit`:

            scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, **kw)[source]
                Use non-linear least squares to fit a function, f, to data.
        
        For the `curve_fit` funcion, we need to define a model (e.g., a mathematical expression) for the fit:

            f : callable
            The model function, f(x, ...). It must take the independent variable as the first argument and the parameters to fit as separate remaining arguments.
    
        Let's create a gaussian curve as model:
        """
    )
    return


@app.cell
def _():
    # Import the necessary libraries
    from scipy.optimize import curve_fit
    from IPython.display import display, Math
    # '%matplotlib inline' command supported automatically in marimo
    return Math, curve_fit, display


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Define the function for curve fitting:
        """
    )
    return


@app.cell
def _(Math, display, np):
    def func(x, a, b, c, d):
        # Gauss function
        return a*np.exp(-(x-b)**2/(2*c**2)) + d

    display(Math( r'y = a * exp\left(-\frac{(x-b)^2}{2c^2}\right) + d' ))
    return (func,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Generate numeric data to be fitted:
        """
    )
    return


@app.cell
def _(func, np):
    x_4 = np.linspace(0, 8, 101)
    noise = np.random.randn(len(x_4)) + 1
    y_4 = func(x_4, 10, 4, 1, np.mean(noise)) + noise
    yerr_3 = np.abs(np.random.randn(len(x_4))) + 1
    return x_4, y_4, yerr_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Perform the curve fitting:
        """
    )
    return


@app.cell
def _(Math, curve_fit, display, func, np, x_4, y_4, yerr_3):
    (p_4, _cov) = curve_fit(func, x_4, y_4, sigma=yerr_3)
    yfit_2 = func(x_4, p_4[0], p_4[1], p_4[2], p_4[3])
    perr_4 = np.sqrt(np.diag(_cov))
    resid_4 = y_4 - yfit_2
    chi2red_4 = np.sum((resid_4 / yerr_3) ** 2) / (y_4.size - 4)
    print('Fitted parameters:')
    display(Math('a=%.2f \\pm %.2f' % (p_4[0], perr_4[0])))
    display(Math('b=%.2f \\pm %.2f' % (p_4[1], perr_4[1])))
    display(Math('c=%.2f \\pm %.2f' % (p_4[2], perr_4[2])))
    display(Math('d=%.2f \\pm %.2f' % (p_4[3], perr_4[3])))
    display(Math('\\chi^2_{red}=%.2f' % chi2red_4))
    return (yfit_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Plot data and fitted curve:
        """
    )
    return


@app.cell
def _(plt, x_4, y_4, yerr_3, yfit_2):
    plt.figure(figsize=(10, 5))
    plt.errorbar(x_4, y_4, yerr=yerr_3, fmt='bo', ecolor='b', capsize=0)
    plt.plot(x_4, yfit_2, linewidth=3, color=[1, 0, 0, 0.5])
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Evidently this is not correct.

        We need to enter an initial guess for the parameters (`p0`):
        """
    )
    return


@app.cell
def _(Math, curve_fit, display, func, np, plt, x_4, y_4, yerr_3):
    (p_5, _cov) = curve_fit(func, x_4, y_4, p0=[1, 4, 1, 1], sigma=yerr_3)
    yfit_3 = func(x_4, p_5[0], p_5[1], p_5[2], p_5[3])
    perr_5 = np.sqrt(np.diag(_cov))
    resid_5 = y_4 - yfit_3
    chi2red_5 = np.sum((resid_5 / yerr_3) ** 2) / (y_4.size - 3)
    print('Fitted parameters:')
    display(Math('a=%.2f \\pm %.2f' % (p_5[0], perr_5[0])))
    display(Math('b=%.2f \\pm %.2f' % (p_5[1], perr_5[1])))
    display(Math('c=%.2f \\pm %.2f' % (p_5[2], perr_5[2])))
    display(Math('d=%.2f \\pm %.2f' % (p_5[3], perr_5[3])))
    display(Math('\\chi^2_{red}=%.2f' % chi2red_5))
    plt.figure(figsize=(10, 5))
    plt.errorbar(x_4, y_4, yerr=yerr_3, fmt='bo', ecolor='b', capsize=0)
    plt.plot(x_4, yfit_3, linewidth=3, color=[1, 0, 0, 0.5])
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.show()
    return (resid_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once again, according to our assumptions, the residuals should be normally distributed:
        """
    )
    return


@app.cell
def _(plot_resid, resid_5, x_4):
    plot_resid(x_4, resid_5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - [Bevington and Robinson (2002) Data Reduction and Error Analysis for the Physical Science. McGraw-Hill Science/Engineering/Math; 3rd edition](https://www.mcgraw-hill.co.uk/html/0071199268.html).
        - [Press et al. (2007) Numerical Recipes 3rd Edition: The Art of Scientific Computing. Cambridge University Press](http://www.nr.com/).  
        - [Montgomery (2013) Applied Statistics and Probability for Engineers. John Wiley & Sons](http://books.google.com.br/books?id=_f4KrEcNAfEC).
        - [NIST/SEMATECH e-Handbook of Statistical Methods](http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm)
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
