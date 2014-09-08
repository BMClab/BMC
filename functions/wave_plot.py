__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'wave_plot.py v.1 2013/12/30'


def wave_plot(freq, t, y, y2, ax=None):
    """Propertie of waves: amplitude, frequency, phase.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(t, y, color=[0, .5, 0, .5], linewidth=5)
    ax.plot(t, y2, color=[0, 0, 1, .5], linewidth=5)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='inout', which='both', length=5)
    ax.set_xlim((-2.05, 2.05))
    ax.set_ylim((-2.1, 2.1))
    ax.locator_params(axis='both', nbins=7)
    ax.xaxis.set_minor_locator(MultipleLocator(.25))
    ax.yaxis.set_minor_locator(MultipleLocator(.5))
    ax.grid(which='both')
    ax.set_title(r'$Asin(2\pi ft+\phi)$', loc='left', size=20, color=[0, 0, 0])
    ax.set_title(r'$x_1=sin(2\pi t)$', loc='center', size=20, color=[0, .5, 0])
    ax.set_title(r'$x_2=2sin(\pi t + \pi/4)$', loc='right', size=20, color=[0, 0, 1])
    ax.annotate('', xy=(.25, 0), xycoords='data', xytext=(.25, 2), size=16,
                textcoords='data',
                arrowprops={'arrowstyle': '<->', 'fc': 'b', 'ec': 'b'})
    ax.annotate(r'$A=2$', xy=(.25, 1.1), xycoords = 'data', xytext=(0, 0),
                textcoords = 'offset points', size=18, color='b')
    ax.annotate('', xy=(0, 1.6), xycoords='data', xytext=(-.25, 1.6), size=16,
                textcoords='data',
                arrowprops={'arrowstyle': '<->', 'fc': 'b', 'ec': 'b'})
    ax.annotate(r'$t_{\phi}=\phi/2\pi f\,(\phi=\pi/4)$', xy=(-1.25, 1.6),
                xycoords = 'data', xytext=(0, -5),
                textcoords = 'offset points', size=18, color='b')
    ax.annotate('', xy=(-.25, 0), xycoords='data', xytext=(-.25, 2),
                textcoords='data', arrowprops={'arrowstyle': '-',
                'linestyle': 'dotted', 'fc': 'b', 'ec': 'b'}, size=16)
    ax.annotate('', xy=(-.75, -1.8), xycoords='data', xytext=(1.25, -1.8),
                textcoords='data', size=10,
                arrowprops={'arrowstyle': '|-|', 'fc': 'b', 'ec': 'b'})
    ax.annotate(r'$T=1/f\,(f=0.5\,Hz)$', xy=(-.16, -1.8), xycoords = 'data',
                 xytext=(0, 8), textcoords = 'offset points', size=18, color='b')
    ax.annotate(r'$t[s]$', xy=(2.05, -.5), xycoords = 'data', xytext=(0, 0),
                textcoords = 'offset points', size=18, color='k')
    plt.suptitle(r'Amplitude ($A$), frequency ($f$), period ($T$), phase ($\phi$)',
                 fontsize=18, y=1.08)

    return ax
