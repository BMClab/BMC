from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'sinusoid_int_plot.py.py v.1 2014/09/04'


def sinusoid_int_plot(ax=None):
    """Plot sinsuoids and their integrals.
    """
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(9, 4))
    ax = ax.flat
    t = t = np.linspace(0, 1, 101)
    sin = np.sin(2*np.pi*t)
    cos = np.cos(2*np.pi*t)
    ax[0].plot(t, sin, linewidth=2, label=r'$sin(t)$')
    ax[0].fill(t, sin, 'b', alpha=0.3)
    ax[0].set_title('sin(t)', fontsize=14)
    ax[1].plot(t, cos, 'r', linewidth=2, label=r'$cos(t)$')
    ax[1].fill_between(t, cos, 0, color=[1, 0, 0, .3])
    ax[1].set_title('cos(t)', fontsize=14)
    ax[2].plot(t, -cos, linewidth=3, label=r'$\int\; sin(t)$')
    ax[2].set_title('Indefinite integral of sin(t): -cos(t)', fontsize=14)
    ax[3].plot(t, sin, 'r', linewidth=3, label=r'$\int\; cos(t)$')
    ax[3].set_title('Indefinite integral of cos(t): sin(t)', fontsize=14)
    for axi in ax:
        axi.margins(.02)
        axi.spines['bottom'].set_position('zero')
        axi.spines['top'].set_color('none')
        axi.spines['left'].set_position('zero')
        axi.spines['right'].set_color('none')
        axi.xaxis.set_ticks_position('bottom')
        axi.yaxis.set_ticks_position('left')
        axi.tick_params(axis='both', direction='inout', which='both', length=5)
        axi.locator_params(axis='both', nbins=4)
        axi.set_xticklabels(['', '$0$', '$\pi$', '$2\pi$'], fontsize=16)
    fig.tight_layout()
