__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'cont_disc_plot.py v.1 2014/09/04'


def cont_disc_plot(ax=None):
    """Plot of continuous and discrete signals.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax = ax.flat
    t = np.linspace(0, 1, 101)
    x = np.sin(2*np.pi*t)
    ax[0].plot(t, x, 'r', linewidth=3)
    ax[0].set_title('Continuous signal', fontsize=16)
    ax[1].stem(t[::5], x[::5], markerfmt='ro', linefmt='b--',
               label=r'$sin(t)$')
    ax[1].set_title('Discrete signal', fontsize=16)
    for axi in ax:
        axi.margins(.02)
        axi.set_ylabel('Amplitude')
        axi.spines['bottom'].set_position('zero')
        axi.spines['top'].set_color('none')
        axi.spines['left'].set_position('zero')
        axi.spines['right'].set_color('none')
        axi.xaxis.set_ticks_position('bottom')
        axi.yaxis.set_ticks_position('left')
        axi.tick_params(axis='both', direction='inout', which='both',
                        length=5)
        axi.locator_params(axis='both', nbins=5)
        axi.set_ylim((-1.1, 1.1))
        axi.set_ylabel(r'Amplitude', fontsize=16)
        axi.annotate(r't[s]', xy=(.95, .1), xycoords = 'data', color='k',
                     size=14, xytext=(0, 0), textcoords = 'offset points')
    fig.tight_layout()

    return ax