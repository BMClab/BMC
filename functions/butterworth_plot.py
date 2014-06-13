#!/usr/bin/env python

"""Plot of frequency response of the Butterworth filter with different orders."""

from __future__ import division, print_function
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'butterworth_plot.py v.1 2014/06/02'


def butterworth_plot(fig=None, ax=None):
    """
    Plot of frequency response of the Butterworth filter with different orders.
    """

    if fig is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
    b1, a1 = signal.butter(1, 10, 'low', analog=True)
    w, h1 = signal.freqs(b1, a1)
    ang1 = np.rad2deg(np.unwrap(np.angle(h1)))
    h1 = 20 * np.log10(abs(h1))
    b2, a2 = signal.butter(2, 10, 'low', analog=True)
    w, h2 = signal.freqs(b2, a2)
    ang2 = np.rad2deg(np.unwrap(np.angle(h2)))
    h2 = 20 * np.log10(abs(h2))
    b4, a4 = signal.butter(4, 10, 'low', analog=True)
    w, h4 = signal.freqs(b4, a4)
    ang4 = np.rad2deg(np.unwrap(np.angle(h4)))
    h4 = 20 * np.log10(abs(h4))
    b6, a6 = signal.butter(6, 10, 'low', analog=True)
    w, h6 = signal.freqs(b6, a6)
    ang6 = np.rad2deg(np.unwrap(np.angle(h6)))
    h6 = 20 * np.log10(abs(h6))
    w = w/10

    # PLOT
    ax[0].plot(w, h1, 'b', w, h2, 'r', w, h4, 'g', w, h6, 'y', linewidth=2)
    ax[0].axvline(1, color='black') # cutoff frequency
    ax[0].scatter(1, -3, marker='s', edgecolor='0', facecolor='1', s=400)
    #ax1.legend(('1', '2', '4', '6'), title='Filter order', loc='best')
    ax[0].set_xscale('log')
    fig.suptitle('Bode plot for low-pass Butterworth filter with different orders',
                 fontsize=16, y=1.05)
    #ax1.set_title('Magnitude', fontsize=14)
    ax[0].set_xlabel('Frequency / Critical frequency', fontsize=14)
    ax[0].set_ylabel('Magnitude [dB]', fontsize=14)
    ax[0].set_xlim(0.1, 10)
    ax[0].set_ylim(-120, 10)
    ax[0].grid(which='both', axis='both')
    ax[1].plot(w, ang1, 'b', w, ang2, 'r', w, ang4, 'g', w, ang6, 'y', linewidth=2)
    ax[1].axvline(1, color='black')  # cutoff frequency
    ax[1].legend(('1', '2', '4', '6'), title='Filter order', loc='best')
    ax[1].set_xscale('log')
    #ax2.set_title('Phase', fontsize=14)
    ax[1].set_xlabel('Frequency / Critical frequency', fontsize=14)
    ax[1].set_ylabel('Phase [degrees]', fontsize=14)
    ax[1].set_yticks(np.arange(0, -300, -45))
    ax[1].set_ylim(-300, 10)
    ax[1].grid(which='both', axis='both')
    plt.tight_layout(w_pad=1)
    axi = plt.axes([.115, .4, .15, .35])  # inset plot
    axi.plot(w, h1, 'b', w, h2, 'r', w, h4, 'g', w, h6, 'y', linewidth=2)
    #ax11.set_yticks(np.arange(0, -7, -3))
    axi.set_xticks((0.6, 1, 1.4))
    axi.set_yticks((-6, -3, 0))
    axi.set_ylim([-7, 1])
    axi.set_xlim([.5, 1.5])
    axi.grid(which='both', axis='both')
