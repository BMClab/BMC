#!/usr/bin/env python

"""Automatic search of filter cutoff frequency based on residual analysis."""

from __future__ import division, print_function
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

__author__ = 'Marcos Duarte <duartexyz@gmail.com>'
__version__ = 'butterworth_plot.py v.1 2014/06/01'


def butterworth_plot(fig=None, ax=None):
    """
    Plot of frequency response of the Butterworth filter with different orders.
    """

    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
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
    ax1.plot(w, h1, 'b', w, h2, 'r', w, h4, 'g', w, h6, 'y', linewidth=2)
    ax1.axvline(1, color='black') # cutoff frequency
    #ax1.legend(('1', '2', '4', '6'), title='Filter order', loc='best')
    ax1.set_xscale('log')
    fig.suptitle('Bode plot for low-pass Butterworth filters', fontsize=18, y=1.06)
    #ax1.set_title('Magnitude', fontsize=14)
    ax1.set_xlabel('Frequency / Critical frequency', fontsize=12)
    ax1.set_ylabel('Magnitude [dB]', fontsize=12)
    ax1.set_ylim(-120, 10)
    ax1.grid(which='both', axis='both')
    ax2.plot(w, ang1, 'b', w, ang2, 'r', w, ang4, 'g', w, ang6, 'y', linewidth=2)
    ax2.axvline(1, color='black')  # cutoff frequency
    ax2.legend(('1', '2', '4', '6'), title='Filter order', loc='best')
    ax2.set_xscale('log')
    #ax2.set_title('Phase', fontsize=14)
    ax2.set_xlabel('Frequency / Critical frequency', fontsize=12)
    ax2.set_ylabel('Phase [degrees]', fontsize=12)
    ax2.set_yticks(np.arange(0, -300, -45))
    ax2.set_ylim(-300, 10)
    ax2.grid(which='both', axis='both')
    plt.tight_layout(w_pad=1)
    ax11 = plt.axes([.115, .35, .15, .35])  # inset plot
    ax11.plot(w, h1, 'b', w, h2, 'r', w, h4, 'g', w, h6, 'y', linewidth=2)
    ax11.set_yticks(np.arange(0, -7, -3))
    ax11.set_ylim([-7, 1])
    ax11.set_xlim([.5, 1.5])
    ax11.grid(which='both', axis='both')