#!/usr/bin/env python

from __future__ import division, print_function

__author__ = 'Marcos Duarte'
__version__ = 'ellipseoid.py v.1 2013/12/30'


def wave_plot(freq, t, y, y2, ax=None):
    """Propertie of waves: amplitude, frequency, phase.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
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
    ax.set_title(r'$Asin(2 \pi f t + \phi)$', loc='left', size=22, color=[0, 0, 0])
    ax.set_title(r'$y = sin(2 \pi t )$', loc='center', size=22, color=[0, .5, 0])
    ax.set_title(r'$y_2 = 2sin(\pi t + \pi/4)$', loc='right', size=22, color=[0, 0, 1])
    ax.annotate('', xy=(.25, 0), xycoords='data',xytext=(.25, 2), textcoords='data',
                 arrowprops={'arrowstyle':'<->', 'fc':'b', 'ec':'b'}, size=16)
    ax.annotate(r'$A=2$', xy=(.25, 1.1), xycoords = 'data',
                 xytext=(0, 0), textcoords = 'offset points', size=18, color='b')
    ax.annotate('', xy=(0, 1.6), xycoords='data',xytext=(-.25, 1.6), textcoords='data',
                 arrowprops={'arrowstyle':'<->', 'fc':'b', 'ec':'b'}, size=16)
    ax.annotate(r'$t_{\phi}=\phi/2\pi f\,(\phi=\pi/4)$', xy=(-1.25, 1.6), xycoords = 'data',
                 xytext=(0, -5), textcoords = 'offset points', size=18, color='b')
    ax.annotate('', xy=(-.25, 0), xycoords='data',xytext=(-.25, 2), textcoords='data',
                 arrowprops={'arrowstyle':'-', 'linestyle':'dotted', 'fc':'b', 'ec':'b'}, size=16)
    ax.annotate('', xy=(-.75, -1.8), xycoords='data',xytext=(1.25, -1.8), textcoords='data',
                 arrowprops={'arrowstyle':'|-|', 'fc':'b', 'ec':'b'}, size=10)
    ax.annotate(r'$T=1/f\,(f=0.5\,Hz)$', xy=(-.16, -1.8), xycoords = 'data',
                 xytext=(0, 8), textcoords = 'offset points', size=18, color='b')
    ax.annotate(r'$t[s]$', xy=(2.05,-.5), xycoords = 'data',
                 xytext=(0, 0), textcoords = 'offset points', size=18, color='k')
    plt.suptitle(r'Properties of waves: amplitude, frequency, phase', fontsize=20, y=1.05)
    
    return ax
