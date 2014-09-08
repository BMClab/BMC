__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'triangular_wave_plot.py.py v.1 2014/09/04'


def triangular_wave_plot(ax=None):
    """Plot of a triangular wave function.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 3))
    t = np.array([-3, -2, -1, 0, 1, 2, 3])*np.pi
    x = np.array([0, 1, 0, 1, 0, 1, 0])
    ax.plot(t, x, linewidth=3, label=r'Square wave')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='inout', which='both', length=5)
    ax.set_xlim((-3*np.pi, 3*np.pi))
    ax.set_ylim((-0.1, 1.1))
    ax.set_xticks(np.linspace(-3*np.pi-0.1, 3*np.pi+0.1, 7))
    ax.set_xticklabels(['$-3\pi$', '$-2\pi$', '$-\pi$', '$0$', '$\pi$', '$2\pi$', '$3\pi$'],
                       fontsize=16)
    plt.locator_params(axis='y', nbins=3)
    ax.annotate(r'$t$', xy=(3*np.pi, 0.1), xycoords = 'data', xytext=(0, 0),
                textcoords = 'offset points', size=18, color='k')
    ax.annotate(r'$x[t]$', xy=(.1, 1.03), xycoords = 'data', xytext=(0, 0),
                textcoords = 'offset points', size=18, color='k')
    ax.grid()
    fig.tight_layout()

    return ax
