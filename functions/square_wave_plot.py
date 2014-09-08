__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'square_wave_plot.py v.1 2014/09/04'


def square_wave_plot(ax=None):
    """Plot of a square wave function.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 3))
    N = np.nan
    t = np.array([-3, -2, N, -2, -1, N, -1, 0, N, 0, 1, N, 1, 2, N, 2, 3])
    x = np.array([-1, -1, N, 1, 1, N, -1, -1, N, 1, 1, N, -1, -1, N, 1, 1])
    ax.plot(t, x, linewidth=3, label=r'Square wave')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='inout', which='both', length=5)
    ax.set_xlim((-3, 3))
    ax.set_ylim((-1.4, 1.4))
    plt.locator_params(axis='y', nbins=3)
    ax.annotate(r'$t[s]$', xy=(3, 0.1), xycoords = 'data', xytext=(0, 0),
                textcoords = 'offset points', size=18, color='k')
    ax.annotate(r'$x[t]$', xy=(-.3, 1.2), xycoords = 'data', xytext=(0, 0),
                textcoords = 'offset points', size=18, color='k')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))
    ax.grid()
    fig.tight_layout()

    return ax
