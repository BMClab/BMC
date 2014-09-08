from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'even_odd_plot.py v.1 2014/09/04'


def even_odd_plot(ax=None):
    """Plot even and odd signals.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax = ax.flat
    t = t = np.linspace(-np.pi, np.pi, 101)
    sin = np.sin(t)
    cos = np.cos(t)
    ax[0].plot(t, cos, 'b', linewidth=3, label=r'$cos(t)$')
    ax[0].plot([np.pi, np.pi], [0, -1], 'r:', linewidth=3)
    ax[0].plot([-np.pi, -np.pi], [0, -1], 'r:', linewidth=3)
    ax[0].plot([-np.pi, np.pi], [-1, -1], 'r:', linewidth=3)
    ax[0].set_title(r'$Even\;function:\;cos(t) = cos(-t)$', fontsize=16)
    ax[1].plot(t, sin, 'g', linewidth=3, label=r'$sin(t)$')
    ax[1].plot([-np.pi/2, -np.pi/2], [0, -1], 'r:', linewidth=3)
    ax[1].plot([-np.pi/2, 0], [-1, -1], 'r:', linewidth=3)
    ax[1].plot([np.pi/2, np.pi/2], [0, 1], 'r:', linewidth=3)
    ax[1].plot([0, np.pi/2], [1, 1], 'r:', linewidth=3)
    ax[1].set_title(r'$Odd\;function:\;sin(t) =-sin(-t)$', fontsize=16)
    for axi in ax:
        axi.margins(.02)
        axi.spines['bottom'].set_position('zero')
        axi.spines['top'].set_color('none')
        axi.spines['left'].set_position('zero')
        axi.spines['right'].set_color('none')
        axi.xaxis.set_ticks_position('bottom')
        axi.yaxis.set_ticks_position('left')
        axi.tick_params(axis='both', direction='inout', which='both', length=5)
        axi.locator_params(axis='both', nbins=5)
        axi.set_xlim((-np.pi-0.2, np.pi+0.2))
        axi.set_ylim((-1.1, 1.1))
        axi.set_xticks(np.linspace(-np.pi, np.pi, 5))
        axi.set_xticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'],
                            fontsize=16)
        for label in axi.get_xticklabels() + axi.get_yticklabels():
            label.set_bbox(dict(facecolor='white', edgecolor='None',
                                alpha=0.65))
    fig.tight_layout()
