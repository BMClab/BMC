"""COGv estimation using COP data based on the inverted pendulum model."""

from __future__ import division, print_function
import numpy as np

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = "1.0.2"
__license__ = "MIT"


def cogve(COP, freq, mass, height, show=False, ax=None):
    """COGv estimation using COP data based on the inverted pendulum model.

    This function estimates the center of gravity vertical projection (COGv)
    displacement from the center of pressure (COP) displacement at the
    anterior-posterior direction during quiet upright standing. COP and COGv
    displacements are measurements useful to quantify the postural sway of a
    person while standing.

    The COGv displacement is estimated by low-pass filtering the COP
    displacement in the frequency domain according to the person's moment
    of rotational inertia as a single inverted pendulum [1]_.

    Parameters
    ----------
    COP    : 1D array_like
             center of pressure data [cm]
    freq   : float
             sampling frequency of the COP data
    mass   : float
             body mass of the subject [kg]
    height : float
             height of the subject [cm]
    show   : bool, optional (default = False)
             True (1) plots data and results in a matplotlib figure
             False (0) to not plot
    ax     : matplotlib.axes.Axes instance, optional (default = None)

    Returns
    -------
    COGv   : 1D array
             center of gravity vertical projection data [cm]

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/IP_Model.ipynb

    Examples
    --------
    >>> from cogve import cogve
    >>> y = np.cumsum(np.random.randn(3000))/50
    >>> cogv = cogve(y, freq=100, mass=70, height=170, show=True)
    """

    from scipy.signal._arraytools import odd_ext
    import scipy.fftpack

    COP = np.asarray(COP)
    height = height / 100  # cm to m
    g = 9.8  # gravity acceleration in m/s2
    # height of the COG w.r.t. ankle (McGinnis, 2005; Winter, 2005)
    hcog = 0.56 * height - 0.039 * height
    # body moment of inertia around the ankle
    # (Breniere, 1996), (0.0572 for the ml direction)
    I = mass * 0.0533 * height ** 2 + mass * hcog ** 2
    # Newton-Euler equation of motion for the inverted pendulum
    # COGv'' = w02*(COGv - COP)
    # where w02 is the squared pendulum natural frequency
    w02 = mass * g * hcog / I
    # add (pad) data and remove mean to avoid problems at the extremities
    COP = odd_ext(COP, n=freq)
    COPm = np.mean(COP)
    COP = COP - COPm
    # COGv is estimated by filtering the COP data in the frequency domain
    # using the transfer function for the inverted pendulum equation of motion
    N = COP.size
    COPfft = scipy.fftpack.fft(COP, n=N) / N  # COP fft
    w = 2 * np.pi * scipy.fftpack.fftfreq(n=N, d=1 / freq)  # angular frequency
    # transfer function
    TF = w02 / (w02 + w ** 2)
    COGv = np.real(scipy.fftpack.ifft(TF * COPfft) * N)
    COGv = COGv[0: N]
    # get back the mean and pad off data
    COP, COGv = COP + COPm, COGv + COPm
    COP, COGv = COP[freq: -freq], COGv[freq: -freq]

    if show:
        _plot(COP, COGv, freq, ax)

    return COGv


def _plot(COP, COGv, freq, ax):
    """Plot results of the cogve function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        time = np.linspace(0, COP.size / freq, COP.size)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(time, COP, color=[0, 0, 1, .8], lw=2, label='COP')
        ax.plot(time, COGv, color=[1, 0, 0, .8], lw=2, label='COGv')
        ax.legend(fontsize=14, loc='best', framealpha=.5, numpoints=1)
        ax.set_xlabel('Time [s]', fontsize=14)
        ax.set_ylabel('Amplitude [cm]', fontsize=14)
        ax.set_title('COGv estimation using the COP data', fontsize=16)
        ax.set_xlim(time[0], time[-1])
        plt.grid()
        plt.show()
