"""Inverse kinematics for a planar two-link system."""

from __future__ import division, print_function
import numpy as np

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.0"
__license__ = "MIT"


def invkin(t, r, L1, L2, unit=['m', '^o'],
           show=True, legenda=['X', 'Y', 'Ang 1', 'Ang 2'], ax=None):
    
    """Inverse kinematics for a planar two-link system.

    Parameters
    ----------
    t : numpy 1Darray with `N` elements
        time data.
    r : numpy array with 1, 2, or 3 columns and `N` rows
        displacement data.
    L1 : float, optional (default = 0.5)
        length of link 1 (proximal)
    L2 : float, optional (default = 0.5)
        length of link 2 (distal)
    unit : list of strings, optional (default = ['m', '^o'])
        unit of measurement of the linear and angular displacement
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    legenda : list of strings, optional (default=['X', 'Y', 'Ang 1', 'Ang 2'])
        text for the plot legends
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ang : numpy array with 2 columns and same number of rows as the input `r`
        angular displacement data [proximal, distal] in radians.
    Notes
    -----
    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/KinematicChain.ipynb

    Examples
    --------
    >>> from invkin2_2d import invkin
    >>> from minjerk import minjerk
    >>> t, r, v, a, j = mjtlin(r0=[1, 0], rf=[0, 1])
    >>> ang = invkin(t, r, L1=0.5, L2=0.5)
    """

    # inverse kinematics
    cosang = (r[:, 0]**2 + r[:, 1]**2 - L1**2 - L2**2)/(2*L1*L2)
    if np.max(cosang) > 1 or np.min(cosang) < -1:
        print("Endpoint value outside working area. Value will be coerced.")
        cosang[cosang > 1] = 1
        cosang[cosang < -1] = -1

    a2 = np.arccos(cosang)
    a1 = np.arctan2(r[:, 1], r[:, 0]) - np.arctan2(L2*np.sin(a2), L1 + L2*np.cos(a2))
    ang = np.vstack((a1, a2)).T

    if show:
        plot_invkin(t, r, ang, unit, legenda, ax)

    return ang
    

def plot_invkin(t, r, ang, unit, legenda, ax):
    """Plot inverse kinematics data."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
        return
    
    if ax is None:
        _, ax = plt.subplots(2, 2, figsize=(8, 5))

    if unit[1] in ['^o', 'deg', 'degrees']:
        ang = ang*180/np.pi
        
    ax[0, 0].plot(t, r)
    ax[0, 0].set_xlabel(r'Time [$s$]')
    ax[0, 0].set_ylabel(r'Displacement [$\mathrm{%s}$]'%unit[0])
    ax[0, 0].legend(legenda[0:2], framealpha=.5, loc='best')
    ax[0, 1].plot(t, ang)
    ax[0, 1].set_xlabel(r'Time [$s$]')
    ax[0, 1].set_ylabel(r'Displacement [$\mathrm{%s}$]'%unit[1])
    ax[0, 1].legend(legenda[2:4], framealpha=.5, loc='best')   
    ax[1, 0].plot(r[:, 0], r[:, 1])
    ax[1, 0].set_xlabel('X [$\mathrm{%s}$]'%unit[0])
    ax[1, 0].set_ylabel('Y [$\mathrm{%s}$]'%unit[0])   
    ax[1, 1].plot(ang[:, 0], ang[:, 1])
    ax[1, 1].set_xlabel('Ang 1 [$\mathrm{%s}$]'%unit[1])
    ax[1, 1].set_ylabel('Ang 2 [$\mathrm{%s}$]'%unit[1])   
    for i, axi in enumerate(ax.flat):
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        axi.yaxis.set_major_locator(plt.MaxNLocator(4))
        
    plt.tight_layout()
    plt.show()  
      