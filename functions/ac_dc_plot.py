from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'ac_dc_plot.py v.1 2014/09/04'


def ac_dc_plot(ax=None):
    """Plot AC and DC components of signals.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax = ax.flat
    t = t = np.linspace(0, 1, 101)
    AC = np.sin(2*4*np.pi*t)
    DC = 2*np.ones(t.shape)
    ax[0].plot(t, AC, 'b', linewidth=3, label=r'AC')
    ax[0].set_title(r'$AC:\;sin(8\pi t)$', fontsize=18)
    ax[1].plot(t, DC, 'g', linewidth=3, label=r'DC')
    ax[1].set_title(r'$DC:\; 2$', fontsize=18)
    ax[2].plot(t, AC+DC, 'r', linewidth=3, label='AC+DC')
    ax[2].plot(t, AC, 'b:', linewidth=2, label='AC')
    ax[2].plot(t, DC, 'g:', linewidth=2, label='DC')
    ax[2].set_title(r'$AC+DC:\;sin(8\pi t)+2$', fontsize=18)
    for axi in ax:
        axi.set_ylim(-1.2, 3.2)
        axi.margins(.02)
        axi.spines['bottom'].set_position('zero')
        axi.spines['top'].set_color('none')
        axi.spines['left'].set_position('zero')
        axi.spines['right'].set_color('none')
        axi.xaxis.set_ticks_position('bottom')
        axi.yaxis.set_ticks_position('left')
        axi.tick_params(axis='both', direction='inout', which='both', length=5)
        axi.locator_params(axis='both', nbins=4)
    fig.tight_layout()
