# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prediction ellipse and prediction ellipsoid

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Two coordinates can vary together: a cloud that is long and tilted is not well described by two independent error bars. This guide builds prediction regions from the covariance matrix, checks their finite-sample scale, and extends the same geometry from an ellipse to an ellipsoid. Read the univariate interval guide first.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** A cloud of points lies close to the line y=x. Would a circular boundary or a tilted ellipse describe its dispersion more efficiently? What information determines the tilt?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy and SciPy calculate covariance and probability quantiles; Matplotlib draws the two- and three-dimensional examples. The repository helper `hyperellipsoid.py` supplies the geometry. Samples are simulated. Optional animation encoding needs external software only if you explicitly save a movie; calculations and static figures do not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To describe the dispersion of data or to make a prediction for a sample, we can calculate a [prediction interval](https://en.wikipedia.org/wiki/Prediction_interval) or [tolerance interval](https://en.wikipedia.org/wiki/Tolerance_interval). For the specific case of a univariate random variable, see [Confidence and prediction intervals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/confidence_prediction_intervals.py). For a multivariate random variable, we need to calculate a prediction hyperellipsoid ([Chew, 1966](http://www.jstor.org/stable/2282774)):

    > The 95% prediction hyperellipsoid is a prediction interval for a sample of a multivariate random variable such that there is 95% of probability that a new observation will lie inside the hyperellipsoid.

    The 95% prediction interval (a p-dimensional hyperellipsoid) for a sample of a multivariate random variable with $p$ dimensions and size $n$ from a population with normal distribution is given by the following probabilistic bounds which holds 95% of the time (Chew, 1966):

    $$
    (x_{n+1}-\bar{x})'\: S^{-1} (x_{n+1}-\bar{x}) \leq \frac{F_{(0.95,\: p,\: n-p)}(n-1)\:p\:(n+1)}{n\:(n-p)}
    $$

    Where $x_{n+1}$ (with dimension $p$) is the new observation for which we want to calculate the prediction interval; $F$ is the [F-distribution](https://en.wikipedia.org/wiki/F-distribution); $\bar{x}$ is the sample mean (with dimension $p$), and $S$ is the [sample covariance matrix](https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance) defined as:

    $$
    S_{j,k} = \frac{1}{n-1}\sum_{i=1}^{n}(x_{ij}-\bar{x}_j)(x_{ik}-\bar{x}_k)
    $$

    This is the formula for the prediction of the next single observation where the population mean and variance are unknown (we only have estimates of these parameters from the sample). This formula is also equivalent to the formula for calculation of a tolerance region (called Type 2 tolerance region by [Chew (1966)](http://www.jstor.org/stable/2282774)):

    > The 95% tolerance hyperellipsoid is a tolerance region for a sample of a multivariate random variable such that the average or expected value of the proportion of the population contained in the hyperellipsoid is 95%.

    In this context, we can refer to the calculated hyperellipsoid as a prediction or tolerance hyperellipsoid (but prediction and tolerance intervals are in general different concepts in statistics).

    The function `hyperellipsoid.py` (code at the end of this text) calculates the prediction or tolerance hyperellipsoid, some related parameters, and plots the results for a given multivariate random variable. The function signature is:
    ```python
    hypervol, saxes, angles, p0, R = hyperellipsoid(P, y=None, z=None, pvalue=.95,
                                                    units=None, show=True, ax=None):
    ```
    Before showing how to use this code, let's look a bit more at the statistical principles behind the estimation of a prediction or tolerance interval.

    <!-- TEASER_END -->
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the region predicts

    The stated finite-sample F formula assumes independent observations from one multivariate normal population, with $n>p$ and a nonsingular sample covariance. It concerns a future independent observation, not a confidence region for the population mean. A 95% prediction procedure has 95% joint coverage over repeated training samples and future observations under this model; the fraction of the training points inside is not forced to be 95%.

    A covariance matrix from collinear points is singular. That is a geometric issue, not something fixed by asking an inverse routine to try harder. Reduce the model dimension or investigate the measurement design before applying the full-dimensional formula.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Univariate variable

    For $p$ = 1 (univariate case) the 95% prediction interval formula above reduces to:

    $$
    \left|\frac{x_{n+1}-\bar{x}}{s}\right| \leq \sqrt{F_{(0.95,\: 1,\: n-1)}(1+1/n)}
    $$

    Where $s$ is the square root of the sample covariance, the sample standard deviation.

    And considering that $F_{(0.95,\: 1,\: n-1)}$ is equal to $T^2_{(0.975,\: n-1)}$, the square of the <a href="https://en.wikipedia.org/wiki/Student's_t-distribution">Student's t-distribution</a>, finally we have:

    $$
    \left|x_{n+1}-\bar{x}\right| \leq s\:T_{(0.975,\: n-1)}\sqrt{1+1/n}
    $$

    Indeed, this results in the 95% prediction interval used for a sample of a univariate variable, see [Confidence and prediction intervals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/confidence_prediction_intervals.py).

    For large $n$, the relevant reference distributions have the limits $T(\infty)=N$, where $N$ is the [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution), and the above statement can be simplified to:

    $$
    \left|x_{n+1}-\bar{x}\right| \leq s\:N_{0.975}
    $$

    $N_{0.975}$ is equal to 1.96, a commonly used value for calculating 95%-probability intervals.<br>
    Let's calculate the error of this approximation for different sample sizes.<br>
    First let's import the necessary Python libraries:
    """)
    return


@app.cell
def _():
    # import the necessary libraries
    import numpy as np
    from scipy import stats

    return np, stats


@app.cell
def _(np, stats):
    F = lambda p, n: np.sqrt(stats.f.ppf(0.95, p, n-p)*(n-1)*p*(n+1)/n/(n-p))  # F distribution
    N = stats.norm.ppf(0.975)                                                  # Normal distribution
    for i in [1000, 100, 30, 10]:
        print('\nApproximation error for n = %d' %i)
        print('Using Normal distribution: %.1f%%' % (100*(N-F(1, i))/F(1, i)))
    return (F,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For n=1000, the approximation is probably good enough, for n=10 it is bad, and it always underestimates.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Why does the one-dimensional prediction interval contain $\sqrt{1+1/n}$ instead of only $1/\sqrt n$?

    /// details | Solution to Exercise 1
    The future observation has its own variability in addition to uncertainty in the estimated mean. The extra 1 represents that observation variance. Without it the interval would target the mean.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bivariate variable

    For $p$ = 2 (bivariate case) the 95% prediction interval formula above reduces to (again, these probabilistic bounds hold 95% of the time ([Chew, 1966](http://www.jstor.org/stable/2282774)):

    $$
    (x_{n+1}-\bar{x})'\: S^{-1} (x_{n+1}-\bar{x}) \leq \frac{2(n-1)(n+1)}{n\:(n-2)}F_{(0.95,\: 2,\: n-2)}
    $$

    This formula is also equivalent to the formula for calculation of a tolerance region such that the average or expected value of the proportion of the population contained in this region is exactly 95% (called Type 2 tolerance region by [Chew (1966)](http://www.jstor.org/stable/2282774)).

    For large $n$, the relevant reference distributions have the limits $2F(2,\infty)=\chi^2(2)$, where $\chi^2$ is the [chi-square distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution), and the formula above for the 95% prediction interval for the bivariate case can be simplified to:

    $$
    (x_{n+1}-\bar{x})'\: S^{-1} (x_{n+1}-\bar{x}) \leq \chi^2_{0.95,\: 2}
    $$

    Or, equivalently, we can use the fact that $\sqrt{2F(2,\infty)}=R$, where $R$ is the [Rayleigh distribution](https://en.wikipedia.org/wiki/Rayleigh_distribution), and:

    $$
    (x_{n+1}-\bar{x})'\: S^{-1} (x_{n+1}-\bar{x}) \leq (R_{0.95})^2
    $$

    Let's calulate the error of this approximation for different sample sizes:
    """)
    return


@app.cell
def _(F, np, stats):
    C = lambda p: np.sqrt(stats.chi2.ppf(0.95, p))
    R = stats.rayleigh.ppf(0.95)
    for i_1 in [1000, 100, 30, 10]:
        print('\nApproximation error for n = %d' % i_1)
        print('Using Chi-square distribution: %.1f%%' % (100 * (C(2) - F(2, i_1)) / F(2, i_1)))
        print('Using Rayleigh distribution: %.1f%%' % (100 * (R - F(2, i_1)) / F(2, i_1)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For n=1000, the approximation is probably good enough, for n=10 it is bad, and it always underestimates.<br>
    To read more about multivariate prediction and confidence intervals, see [Chew (1966)](http://www.jstor.org/stable/2282774), and about the use of prediction ellipse in the field of posturography, see [Schubert and Kirchner (2014)](http://www.sciencedirect.com/science/article/pii/S0966636213005961).
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** For covariance $S=\operatorname{diag}(4,1)$ and threshold $c=9$, find the prediction ellipse’s semiaxes and area.

    /// details | Solution to Exercise 2
    The eigenvectors are the coordinate axes. The semiaxes are $\sqrt{9(4)}=6$ and $\sqrt{9(1)}=3$. The area is $\pi(6)(3)=18\pi$. The threshold is stipulated here; it is not claimed to be a 95% threshold.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Python code to compute the hyperellipsoid

    We can write a code to calculate the ellipse, ellipsoid, and in a more general case, the hyperellipsoid using the formula for the prediction hyperellipsoid (this code is published in Duarte (2015)). The directions and lengths of the hyperellipsoid semi-axes are found, respectively, as the eigenvectors and square roots of the scaled eigenvalues of the covariance matrix of the data using the concept of [principal components analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) or [singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) and the length of the semi-axes are adjusted to account for the necessary prediction probability. The volume of the hyperellipsoid is calculated with the same equation for the [volume of a n-dimensional ball](https://en.wikipedia.org/wiki/Volume_of_an_n-ball) with the radius replaced by the semi-axes of the hyperellipsoid.<br>
    In Python, such code to compute the hypervolume of the 95% prediction hyperellipsoid is:
    ```Python
    import numpy as np                            # import Numpy package
    from scipy.stats import f as F                # import F distribution
    from scipy.special import gamma               # import Gamma function
    n, p = np.asarray(data).shape                 # 2-D array dimensions
    cov = np.cov(data, rowvar=0)                  # covariance matrix of data
    U, s, Vt = np.linalg.svd(cov)                 # singular value decomposition
    f95 = F.ppf(.95,p,n-p)*(n-1)*p*(n+1)/n/(n-p)  # F 95 percent point function
    saxes = np.sqrt(s*f95)                        # semi-axes lengths
    hypervolume = np.pi**(p/2)/gamma(p/2+1)*np.prod(saxes)
    hypervolume
    ```

    For instance, if `data` has two columns, `hyperellipsoid` will contain the area of the 95% prediction ellipse.

    ### Equivalent code for a Matlab-like environment

    The equivalent code for a Matlab-like environment to compute the hypervolume of the 95% prediction hyperellipsoid is:
    ```Matlab
    [n, p] = size(data);                          % 2-D array dimensions
    covar = cov(data);                            % covariance matrix of data
    [U, S, V] = svd(covar);                       % singular value decomposition
    f95 = finv(.95,p,n-p)*(n-1)*p*(n+1)/n/(n-p);  % F 95 percent point function
    saxes = sqrt(diag(S)*f95);                    % semi-axes lengths
    hypervolume = pi^(p/2)/gamma(p/2+1)*prod(saxes);
    hypervolume
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Python function hyperellipsoid.py

    Let's see now a more complete function, `hyperellipsoid.py` (code at the end of this text), to compute the prediction hyperellipsoid and related variables:
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        from notebook_resources import repo_file
    except ModuleNotFoundError as _error:
        if _error.name != "notebook_resources":
            raise
        raise ModuleNotFoundError(
            "This notebook needs notebook_resources.py. Download it from:\n"
            "https://raw.githubusercontent.com/BMClab/BMC/master/functions/notebook_resources.py\n"
            "Save it in the same folder as this notebook, then rerun this cell.",
            name="notebook_resources",
        ) from None
    return (repo_file,)


@app.cell
def _(repo_file):
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    import sys
    sys.path.insert(0, str(repo_file('functions/hyperellipsoid.py').parent))  # add to pythonpath
    from hyperellipsoid import hyperellipsoid

    return hyperellipsoid, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Prediction ellipse

    For a bivariate random data with normal distribution with mean 0 and variance 1, the prediction ellipse should be a circle:
    """)
    return


@app.cell
def _(hyperellipsoid, np):
    P = np.random.randn(10000, 2)
    (area, axes, angles, center, R_1) = hyperellipsoid(P, units='cm', show=True)
    print('Area =', area)
    print('Semi-axes =', axes)
    print('Angles =', angles)
    print('Center =', center)
    print('Rotation matrix =\n', R_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For other data:
    """)
    return


@app.cell
def _(hyperellipsoid, np):
    y = np.cumsum(np.random.randn(3000)) / 50
    x = np.cumsum(np.random.randn(3000)) / 100
    (area_1, axes_1, angles_1, center_1, R_2) = hyperellipsoid(x, y, units='cm', show=True)
    print('Area =', area_1)
    print('Semi-axes =', axes_1)
    print('Angles =', angles_1)
    print('Center =', center_1)
    print('Rotation matrix =\n', R_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Find points outside the ellipse

    The **outside_hyperellipsoid** function can be used to determine whether specific points lie outside the ellipse as demonstrated below.

    Note that the proportion of points outside a 95% ellipse is expected to tend toward 0.05 as n increases.
    """)
    return


@app.cell
def _(hyperellipsoid, np, plt):
    from hyperellipsoid import outside_hyperellipsoid
    np.random.seed(6)
    P_1 = np.random.randn(100, 2)
    (volume, axes_2, angles_2, center_2, R_3) = hyperellipsoid(P_1, pvalue=0.95, show=False)
    b = outside_hyperellipsoid(axes_2, angles_2, center_2, R_3, P_1)
    p = P_1[b]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(p[:, 0], p[:, 1], 'ro', ms=8)
    (volume, axes_2, angles_2, center_2, R_3) = hyperellipsoid(P_1, pvalue=0.95, show=True, ax=ax)
    print('Number of points outside ellipsoid: %d' % b.sum())
    print('Proportion of points outside ellipsoid: %.3f' % b.mean())
    return (outside_hyperellipsoid,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** For the ellipse in Exercise 2, is the point (3,3) inside when the center is (0,0)?

    /// details | Solution to Exercise 3
    Its squared Mahalanobis distance is $3^2/4+3^2/1=11.25$, greater than 9, so it is outside. Checking each coordinate against its semiaxis separately would incorrectly accept it.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Prediction ellipsoid

    For a trivariate random data with normal distribution with mean 0 and variance 1, the prediction ellipsoid should be a sphere:
    """)
    return


@app.cell
def _(hyperellipsoid, np):
    P_2 = np.random.randn(10000, 3)
    (volume_1, axes_3, angles_3, center_3, R_4) = hyperellipsoid(P_2, units='cm', show=True)
    print('Volume =', volume_1)
    print('Semi-axes =', axes_3)
    print('Angles =', angles_3)
    print('Center =', center_3)
    print('Rotation matrix =\n', R_4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For other data:
    """)
    return


@app.cell
def _(hyperellipsoid, np):
    P_3 = np.random.randn(1000, 3)
    P_3[:, 2] = P_3[:, 2] + P_3[:, 1] * 0.5
    P_3[:, 1] = P_3[:, 1] + P_3[:, 0] * 0.5
    (volume_2, axes_4, angles_4, center_4, R_5) = hyperellipsoid(P_3, units='cm', show=True)
    print('Volume =', volume_2)
    print('Semi-axes =', axes_4)
    print('Angles =', angles_4)
    print('Center =', center_4)
    print('Rotation matrix =\n', R_5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Find points outside the (hyper)ellipsoid

    The **outside_hyperellipsoid** function can be used to determine whether specific points lie outside the ellipsoid as demonstrated below.

    Note that the proportion of points outside a 95% ellipsoid is expected to tend toward 0.05 as n increases.
    """)
    return


@app.cell
def _(hyperellipsoid, np, outside_hyperellipsoid, plt):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    from hyperellipsoid import _plot
    np.random.seed(6)
    P_4 = np.random.randn(100, 3)
    (volume_3, axes_5, angles_5, center_5, R_6) = hyperellipsoid(P_4, pvalue=0.95, show=False)
    b_1 = outside_hyperellipsoid(axes_5, angles_5, center_5, R_6, P_4)
    p_1 = P_4[b_1]
    fig_1 = plt.figure(figsize=(7, 7))
    ax_1 = fig_1.add_axes([0, 0, 1, 1], projection='3d')
    ax_1.plot(p_1[:, 0], p_1[:, 1], p_1[:, 2], 'ro', ms=8)
    (volume_3, axes_5, angles_5, center_5, R_6) = hyperellipsoid(P_4, pvalue=0.95, show=True, ax=ax_1)
    print('Number of points outside ellipsoid: %d' % b_1.sum())
    print('Proportion of points outside ellipsoid: %.3f' % b_1.mean())
    return P_4, animation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Static Matplotlib output shows one view. The animation below changes the view without changing the data. Desktop interactive backends can also rotate a figure, but a Qt window is not required and is unavailable in a browser-only runtime:
    """)
    return




@app.cell
def _(P_4, hyperellipsoid):
    hyperellipsoid(P_4, units='cm', show=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's generate an animation where the ellipsoid is rotated to better visualize it in this notebook:
    """)
    return


@app.cell
def _(P_4, animation, hyperellipsoid, plt):
    from hyperellipsoid import _plot
    fig_2 = plt.figure(figsize=(7, 7))
    ax_2 = fig_2.add_axes([0, 0, 1, 1], projection='3d')
    pts = ax_2.plot([], [], [], 'o')
    (volume_4, axes_6, angles_6, center_6, R_7) = hyperellipsoid(P_4, units='cm', show=False, ax=None)
    _plot(P_4, volume_4, axes_6, center_6, R_7, 0.95, 'cm', ax_2)

    def animate(i):
        ax_2.view_init(20, 3 * i)
        fig_2.canvas.draw()
        return pts
    anim = animation.FuncAnimation(fig_2, func=animate, frames=120, interval=30, blit=True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the animation you see in a separate window when you run the code above:

    <div class='center-align'><figure><img src='https://github.com/BMClab/BMC/blob/master/images/ellipsoid.gif?raw=1' width=450 alt='ellipsoid'/></figure></div><br>

    To save the animation generated above as a video file (you need to have [FFmpeg](http://www.ffmpeg.org/) or [ImageMagick](http://www.imagemagick.org/script/index.php) installed):
    """)
    return








@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** If you rotate every observation by an orthogonal matrix, should the ellipse’s area change? What if you rescale one coordinate by two?

    /// details | Solution to Exercise 4
    An orthogonal rotation preserves eigenvalues and area; it changes orientation. Scaling one coordinate by two multiplies the two-dimensional area by two and changes its units/shape. Report coordinate units alongside area.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Why must $n>p$ in the stated finite-sample formula?
    2. Does a 95% ellipse guarantee exactly 5% of fitted observations lie outside?
    3. What do the covariance eigenvectors specify?

    /// details | Checkpoint answers
    1. Positive denominator degrees of freedom are required, and covariance must also have full rank.
    2. No.
    3. Principal directions of the ellipsoid.
    ///

    ## Problems

    1. Give the area of an ellipse with semiaxes 2 cm and 5 cm.
    2. Give the volume of an ellipsoid with semiaxes 1, 2, and 3 cm.
    3. Can the stated full-dimensional formula be used for three variables and only three observations?

    /// details | Final answers
    1. $10\pi$ cm², approximately 31.42 cm².
    2. $8\pi$ cm³, approximately 25.13 cm³.
    3. No.
    ///

    ## Go deeper

    [Confidence and prediction intervals](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/confidence_prediction_intervals.py) — establish the one-dimensional target first. The Chew paper and the posturography exchange below explain why prediction and confidence regions should not share a label.

    ### Video lectures on the Internet

    [MIT Statistics for Applications lectures](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/video_galleries/lecture-videos/) — multivariate regression and covariance connect statistical regions with linear algebra.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Chew V (1966) [Confidence, Prediction, and Tolerance Regions for the Multivariate Normal Distribution](http://www.jstor.org/stable/2282774). Journal of the American Statistical Association, 61, 315, pp. 605-617.

    - Duarte M (2015) [Comments on “Ellipse area calculations and their applicability in posturography” (Schubert and Kirchner, vol.39, pages 518-522, 2014)](http://demotu.org/pubs/GP15.pdf). Gait & Posture, 41, 44-45. <a href="https://pubmed.ncbi.nlm.nih.gov/25194690" target="_blank">PubMed</a>.

    - Schubert P, Kirchner M (2014) [Ellipse area calculations and their applicability in posturography](http://www.sciencedirect.com/science/article/pii/S0966636213005961). Gait & Posture, 39, 518–522.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function hyperellipsoid.py
    """)
    return


@app.cell
def _(mo, repo_file):
    mo.ui.code_editor(repo_file("functions/hyperellipsoid.py").read_text(), language="python", disabled=True)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
