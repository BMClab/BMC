"""Time normalization (from 0 to 100% with step interval)."""

from __future__ import division, print_function
import numpy as np

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = "1.0.5"
__license__ = "MIT"


def tnorm(y, axis=0, step=1, k=3, smooth=0, mask=None, show=False, ax=None):
    """Time normalization (from 0 to 100% with step interval).

    Time normalization is usually employed for the temporal alignment of data
    obtained from different trials with different duration (number of points).
    This code implements a procedure knwown as the normalization to percent
    cycle, the most simple and common method used among the ones available,
    but may not be the most adequate [1]_.

    NaNs and any value inputted as a mask parameter and that appears at the
    extremities are removed before the interpolation because this code does not
    perform extrapolation. For a 2D array, the entire row with NaN or a mask
    value at the extermity is removed because of alignment issues with the data
    from different columns. NaNs and any value inputted as a mask parameter and
    that appears in the middle of the data (which may represent missing data)
    are ignored and the interpolation is performed throught these points.

    This code can perform simple linear interpolation passing throught each
    datum or spline interpolation (up to quintic splines) passing through each
    datum (knots) or not (in case a smoothing parameter > 0 is inputted).

    See this IPython notebook [2]_.

    Parameters
    ----------
    y : 1-D or 2-D array_like
        Array of independent input data. Must be increasing.
        If 2-D array, the data in each axis will be interpolated.
    axis : int, 0 or 1, optional (default = 0)
        Axis along which the interpolation is performed.
        0: data in each column are interpolated; 1: for row interpolation
    step : float or int, optional (default = 1)
        Interval from 0 to 100% to resample y or the number of points y
        should be interpolated. In the later case, the desired number of
        points should be expressed with step as a negative integer.
        For instance, step = 1 or step = -101 will result in the same
        number of points at the interpolation (101 points).
        If step == 0, the number of points will be the number of data in y.
    k : int, optional (default = 3)
        Degree of the smoothing spline. Must be 1 <= k <= 5.
        If 3, a cubic spline is used.
        The number of data points must be larger than k.
    smooth : float or None, optional (default = 0)
        Positive smoothing factor used to choose the number of knots.
        If 0, spline will interpolate through all data points.
        If None, smooth=len(y).
    mask : None or float, optional (default = None)
        Mask to identify missing values which will be ignored.
        It can be a list of values.
        NaN values will be ignored and don't need to be in the mask.
    show : bool, optional (default = False)
        True (1) plot data in a matplotlib figure.
        False (0) to not plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    yn : 1-D or 2-D array
        Interpolated data (if axis == 0, column oriented for 2-D array).
    tn : 1-D array
        New x values (from 0 to 100) for the interpolated data.
    inds : list
        Indexes of first and last rows without NaNs at the extremities of `y`.
        If there is no NaN in the data, this list is [0, y.shape[0]-1].

    Notes
    -----
    This code performs interpolation to create data with the desired number of
    points using a one-dimensional smoothing spline fit to a given set of data
    points (scipy.interpolate.UnivariateSpline function).

    References
    ----------
    .. [1] http://www.sciencedirect.com/science/article/pii/S0021929010005038
    .. [2] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/TimeNormalization.ipynb

    See Also
    --------
    scipy.interpolate.UnivariateSpline:
    One-dimensional smoothing spline fit to a given set of data points.

    Examples
    --------
    >>> # Default options: cubic spline interpolation passing through
    >>> # each datum, 101 points, and no plot
    >>> y = [5,  4, 10,  8,  1, 10,  2,  7,  1,  3]
    >>> tnorm(y)

    >>> # Linear interpolation passing through each datum
    >>> y = [5,  4, 10,  8,  1, 10,  2,  7,  1,  3]
    >>> yn, tn, indie = tnorm(y, k=1, smooth=0, mask=None, show=True)

    >>> # Cubic spline interpolation with smoothing
    >>> y = [5,  4, 10,  8,  1, 10,  2,  7,  1,  3]
    >>> yn, tn, indie = tnorm(y, k=3, smooth=1, mask=None, show=True)

    >>> # Cubic spline interpolation with smoothing and 50 points
    >>> x = np.linspace(-3, 3, 100)
    >>> y = np.exp(-x**2) + np.random.randn(100)/10
    >>> yn, tn, indie = tnorm(y, step=-50, k=3, smooth=1, show=True)

    >>> # Deal with missing data (use NaN as mask)
    >>> x = np.linspace(-3, 3, 100)
    >>> y = np.exp(-x**2) + np.random.randn(100)/10
    >>> y[0] = np.NaN # first point is also missing
    >>> y[30: 41] = np.NaN # make other 10 missing points
    >>> yn, tn, indie = tnorm(y, step=-50, k=3, smooth=1, show=True)

    >>> # Deal with 2-D array
    >>> x = np.linspace(-3, 3, 100)
    >>> y = np.exp(-x**2) + np.random.randn(100)/10
    >>> y = np.vstack((y-1, y[::-1])).T
    >>> yn, tn, indie = tnorm(y, step=-50, k=3, smooth=1, show=True)
    """

    from scipy.interpolate import UnivariateSpline

    y = np.asarray(y)
    if axis:
        y = y.T
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))
    # turn mask into NaN
    if mask is not None:
        y[y == mask] = np.NaN
    # delete rows with missing values at the extremities
    iini = 0
    iend = y.shape[0]-1
    while y.size and np.isnan(np.sum(y[0])):
        y = np.delete(y, 0, axis=0)
        iini += 1
    while y.size and np.isnan(np.sum(y[-1])):
        y = np.delete(y, -1, axis=0)
        iend -= 1
    # check if there are still data
    if not y.size:
        return None, None, []
    if y.size == 1:
        return y.flatten(), None, [0, 0]
        
    indie = [iini, iend]

    t = np.linspace(0, 100, y.shape[0])
    if step == 0:
        tn = t
    elif step > 0:
        tn = np.linspace(0, 100, np.round(100 / step + 1))
    else:
        tn = np.linspace(0, 100, -step)
    yn = np.empty([tn.size, y.shape[1]]) * np.NaN
    for col in np.arange(y.shape[1]):
        # ignore NaNs inside data for the interpolation
        ind = np.isfinite(y[:, col])
        if np.sum(ind) > 1:  # at least two points for the interpolation
            spl = UnivariateSpline(t[ind], y[ind, col], k=k, s=smooth)
            yn[:, col] = spl(tn)

    if show:
        _plot(t, y, ax, tn, yn)

    if axis:
        y = y.T
    if yn.shape[1] == 1:
        yn = yn.flatten()

    return yn, tn, indie


def _plot(t, y, ax, tn, yn):
    """Plot results of the tnorm function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 5))

        ax.set_prop_cycle('color', ['b', 'r', 'b', 'g', 'b', 'y', 'b', 'c', 'b', 'm'])
        #ax.set_color_cycle(['b', 'r', 'b', 'g', 'b', 'y', 'b', 'c', 'b', 'm'])
        for col in np.arange(y.shape[1]):
            if y.shape[1] == 1:
                ax.plot(t, y[:, col], 'o-', lw=1, label='Original data')
                ax.plot(tn, yn[:, col], '.-', lw=2,
                        label='Interpolated')
            else:
                ax.plot(t, y[:, col], 'o-', lw=1)
                ax.plot(tn, yn[:, col], '.-', lw=2, label='Col= %d' % col)
            ax.locator_params(axis='y', nbins=7)
            ax.legend(fontsize=12, loc='best', framealpha=.5, numpoints=1)
        plt.xlabel('[%]')
        plt.tight_layout()
        plt.show()
