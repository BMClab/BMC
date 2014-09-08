from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'quantize_plot.py v.1 2014/09/04'


def quantize_plot(ax=None):
    """Plot of continuous and discrete signals.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(9, 5))
    ax = ax.flat
    t = np.linspace(0, 1, 1001)
    x = np.sin(2*np.pi*t)
    x1 = quantize(x, 1, 2)
    ax[0].plot(t, x, 'b', linewidth=2, color=[0, 0, 1, .3])
    ax[0].plot(t, x1, 'r.', linewidth=1, drawstyle='steps')
    ax[0].set_title('2-bit resolution', fontsize=12)
    x4 = quantize(x, 4, 2)
    ax[1].plot(t, x, 'b', linewidth=2, color=[0, 0, 1, .3])
    ax[1].plot(t, x4, 'r.', linewidth=1, drawstyle='steps')
    ax[1].set_title('4-bit resolution', fontsize=12)
    x16 = quantize(x, 16, 2)
    ax[2].plot(t, x, 'b', linewidth=2, color=[0, 0, 1, .3])
    ax[2].plot(t, x16, 'r.', linewidth=1, drawstyle='steps')
    ax[2].set_title('16-bit resolution', fontsize=12)
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
        axi.set_ylim((-1.1, 1.1))
    fig.tight_layout()

    return ax


def quantize(x, n_bits, v_range):
    """ Quantizes `x` given `nbits` resolution and `v_range` range."""
    r = v_range/2**n_bits
    return r*np.sign(x)*np.ceil(np.abs(x)/r)
    # a true a/d converter should use 'floor' instead of 'ceil':
    # return r*np.ceil(x/r)
