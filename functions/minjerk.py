"""Calculate minimum jerk trajectory."""

from __future__ import division, print_function
import numpy as np

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.0"
__license__ = "MIT"


def minjerk(r0, rf, duration=1, unit='m', N=101,
            show=True, legenda=['X', 'Y', 'Z'], ax=None):

    """Calculate minimum jerk trajectory.

    Parameters
    ----------
    r0 : array_like with 1, 2 or 3 elements
        coordinates (x, y, z) of the initial position.
    rf : array_like with 1, 2 or 3 elements
        coordinates (x, y, z) of the final position.
    duration : float, optional (default = 1)
        duration (in seconds) of the movement.    
    unit : string, optional (default = 'm')
        unit of measurement of the displacement
    N : integer, optional (default = 101)
        number of points (rows) to interpolate the minimum jerk trajectory.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    legenda : list of strings, optional (default=['x','y','z'])
        text for the plot legend
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    t : numpy 1Darray with `N` elements
        time data.
    r : numpy array with 1, 2, or 3 columns and `N` rows
        displacement data.
    v : numpy array with 1, 2, or 3 columns and `N` rows
        velocity data.
    a : numpy array with 1, 2, or 3 columns and `N` rows
        acceleration data.
    j : numpy array with 1, 2, or 3 columns and `N` rows
        jerk data.

    Notes
    -----
    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/MinimumJerkHypothesis.ipynb

    Examples
    --------
    >>> from minjerk import minjerk
    >>> t, r, v, a, j = mjtlin(r0=[1, 0], rf=[0, 1], duration=1)

    """

    # minimum jerk trajectory:
    # r0 + (rf - r0)*(10*(t/d)**3 - 15*(t/d)**4 + 6*(t/d)**5)
    d = duration
    t = np.linspace(0, d, N)
    r = np.zeros((t.size, len(r0)))
    v = np.zeros((t.size, len(r0)))
    a = np.zeros((t.size, len(r0)))
    j = np.zeros((t.size, len(r0)))
    for i in range(len(r0)):
        r[:, i] = r0[i] + (rf[i] - r0[i])*(10*(t/d)**3 - 15*(t/d)**4 + 6*(t/d)**5)
        v[:, i] = (rf[i] - r0[i])*(30*t**2/d**3 - 60*t**3/d**4 + 30*t**4/d**5)
        a[:, i] = (rf[i] - r0[i])*(60*t/d**3 - 180*t**2/d**4 + 120*t**3/d**5)
        j[:, i] = (rf[i] - r0[i])*(60/d**3 - 360*t/d**4 + 360*t**2/d**5)

    if show:
        plot_data(t, r, v, a, j, unit, legenda, ax)

    return t, r, v, a, j


def plot_data(t, r, v, a, j, unit, legenda, ax):
    """Plot kinematics of minimum jerk trajectories."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
        return

    if ax is None:
        _, ax = plt.subplots(1, 4, sharex=True, figsize=(10, 3))

    ax[0].plot(t, r)
    ax[0].set_title(r'Displacement [$\mathrm{%s}$]'%unit)
    ax[0].legend(legenda, framealpha=.5, loc='best')
    ax[1].plot(t, v)
    ax[1].set_title(r'Velocity [$\mathrm{%s/s}$]'%unit)
    ax[2].plot(t, a)
    ax[2].set_title(r'Acceleration [$\mathrm{%s/s^2}$]'%unit)
    ax[3].plot(t, j)
    ax[3].set_title(r'Jerk [$\mathrm{%s/s^3}$]'%unit)
    for i, axi in enumerate(ax.flat):
        axi.set_xlabel(r'Time [$s$]')
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        axi.yaxis.set_major_locator(plt.MaxNLocator(4))

    plt.tight_layout()
    plt.show()
