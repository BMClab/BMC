from __future__ import division, print_function

__author__ = 'Marcos Duarte, https://github.com/duartexyz/BMC'
__version__ = 'ellipseoid.py v.1 2013/12/30'


def pdf_norm_plot(m=0, s=1, fig=None, ax=None):
    """Plot the probability density function of the normal distribution.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    m, s = float(m), float(s)  # in case they were passed as command line args
    if s <= 0:
        s = 1
    plt.rc('font', size=16)
    plt.rc(('xtick.major','ytick.major'), pad=8)
    n = 1000
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    x = np.linspace(m-4*s, m+4*s, n)
    # pdf at x for a random variable with normal distribution
    f = norm.pdf(x, loc=m, scale=s)  
    ones = np.arange(np.round(3/8*n)+1, np.round(5/8*n)+1, dtype=int)
    twos = np.arange(np.round(2/8*n)+1, np.round(6/8*n)+1, dtype=int)
    threes = np.arange(np.round(1/8*n)+1, np.round(7/8*n)+1, dtype=int)
    ax.fill_between(x[ones], y1=f[ones], y2=0, color=[0, 0.2, .5, .4])
    ax.fill_between(x[twos], y1=f[twos], y2=0, color=[0, 0.2, .5, .3])
    ax.fill_between(x[threes], y1=f[threes], y2=0, color=[0, 0.2, .5, .3])
    ax.plot(x, f, color=[1, 0, 0, .8], linewidth=4)
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)
    ymax = ax.get_ylim()[1]/0.45
    for i in range(-3, 4):
        ax.axvline(i*s+m, ymin=0, ymax=f[n/2-i/8*n]/ax.get_ylim()[1], c='w', lw=3)
    ax.axvline(-2.5*s+m, ymin=0.01, ymax=.15, c='k', lw=1)
    ax.axvline(2.5*s+m, ymin=0.01, ymax=.15, c='k', lw=1)
    ax.axvline(-3.5*s+m, ymin=0.01, ymax=.05, c='k', lw=1)
    ax.axvline(3.5*s+m, ymin=0.01, ymax=.05, c='k', lw=1)
    ax.text(-3.8*s+m, .03*ymax, '0.1%', color='k')
    ax.text(3.2*s+m, .03*ymax, '0.1%', color='k')
    ax.text(-2.8*s+m, .08*ymax, '2.1%', color='k')
    ax.text(2.2*s+m, .08*ymax, '2.1%', color='k')
    ax.text(-1.85*s+m, .03*ymax, '13.6%',  color='w')
    ax.text(1.15*s+m, .03*ymax, '13.6%', color='w')
    ax.text(-.85*s+m, .08*ymax, '34.1%', color='w')
    ax.text(.15*s+m, .08*ymax, '34.1%', color='w')
    plt.locator_params(axis = 'y', nbins = 5)
    ax.set_xticks(np.linspace(m-4*s, m+4*s, 9))
    xtl = [r'%+d$\sigma$' %i for i in range(-4, 5, 1)]
    xtl[4] = r'$\mu$'
    ax.set_xticklabels(xtl)
    #ax2 = ax.twiny()
    #ax2.xaxis.set_ticks_position('bottom')
    #ax2.spines["bottom"].set_position(("axes", -0.1))
    #ax2.set_xlim(-4*s, 4*s)
    #ax2.set_xticks(np.linspace(m-4*s, m+4*s, 9))
    title='Probability density function of the normal distribution ' +\
    '~ $N(\mu=%.1f,\ \sigma^2=%.1f)$' %(m, s**2)
    plt.title(title, fontsize=14)
    plt.show()

    return fig, ax
    

if __name__ == '__main__':
    import sys
    pdf_norm_plot(*sys.argv[1:])
