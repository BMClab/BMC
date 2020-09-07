"""
Calculate confidence interval of an estimate by bootstrap or 'normal' method.
"""
__author__ = "Marcos Duarte, https://github.com/demotu/"
__version__ = "0.0.2"
__license__ = "MIT"

import numpy as np


def ci_calc(y, estimate=np.nanmean, level=95, axis=0, method='bootstrap',
            seed=None):
    """Calculate `level`% ci of the `estimate` for `y` by `method` method.

    Parameters
    ----------
    y : 1-d or 2-d array_like
        Calculate the confidence interval of these values.
    estimate : numpy function (optional, default=np.nanmean)
        Function (unknown parameter) to which ci will be estimated
    level : number (optional, default=95)
        Confidence level (in %) of the ci
    axis : integer (optional, default=0)
        Axis of y which ci will be estimated
    method: string (optional, default='bootstrap')
        Valid options: 'bootstrap' (from Seaborn) or 'normal'.
    seed : {None, int, ..., Generator} (optional, default=None)
        Seed for the random number generator

    Returns
    -------
    confidence interval : 1-d or 2-d array_like

    Example
    -------
    >>> rng = np.random.default_rng(12345)
    >>> y = 1 + rng.standard_normal(size=(10000, 4))
    >>> ci_calc(y, method='bootstrap', seed=rng)
    array([[0.98543573, 0.97812353, 0.99513357, 0.98174334],
            [1.02535726, 1.01852985, 1.03386319, 1.02152344]])

    >>> ci_calc(y, method='normal')
    array([[0.98530252, 0.97851995, 0.99611958, 0.98188726],
            [1.02378054, 1.01758363, 1.03539235, 1.02153012]])

    See Also
    --------
    https://stackoverflow.com/questions/60654248/
    https://github.com/BMClab/BMC/blob/master/notebooks/ConfidencePredictionIntervals.ipynb
    """
    if method == 'bootstrap':
        import seaborn as sns
        ci = sns.utils.ci(sns.algorithms.bootstrap(y, n_boot=1000, axis=axis,
                                                   units=None, func=estimate,
                                                   seed=seed),
                          which=level, axis=axis)
    elif method == 'normal':
        from scipy import stats
        n = y.shape[axis]
        std = np.nanstd(y, axis=axis, ddof=1)
        ci = estimate(y, axis=axis) + (stats.t.ppf((1+level/100)/2, n-1) *
                                       np.array([-std, std])/np.sqrt(n))

    return ci
