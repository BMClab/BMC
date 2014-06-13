#!/usr/bin/env python

"""Calculate the linear envelope of a signal."""

from __future__ import division, print_function
from scipy.signal import detrend, butter, filtfilt

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'linear_envelope.py v.1 2014/05/31'


def linear_envelope(x, freq=1000, Fc_bp=[10, 400], Fc_lp=8):
    """Calculate the linear envelope of a signal.

    Parameters
    ----------
    x     : 1D array_like
            raw signal
    freq  : number, optional. Default = 1000
            sampling frequency
    Fc_bp : list of floats [Fc_h, Fc_l], optional. Default = [10, 400]
            cutoff frequencies for the band-pass filter (in Hz)
    Fc_lp : float, optional. Default = 8
            cutoff frequency for the low-pass filter (in Hz)

    Returns
    -------
    x     : 1D array_like
            linear envelope of the signal

    Notes
    -----
    A 2nd-order Butterworth filter with zero lag is used for the filtering.  

    See this notebook [1]_.

    References
    ----------
    .. [1] https://github.com/duartexyz/bmc/Electromyography.ipynb

    """
    
    C = 0.802  # Correct the cutoff frequency for the number of passes
    
    x = detrend(x, type='constant')  # subtract the mean
    
    if Fc_bp is not None:
        # band-pass filter
        b, a = butter(2, (Fc_bp/(freq/2)), btype = 'bandpass')
        x = filtfilt(b, a, x)

    if Fc_lp is not None:
        # full-wave rectification
        x = abs(x)
        # low-pass Butterworth filter
        b, a = butter(2, (Fc_lp/(freq/2)/C), btype = 'low')
        x = filtfilt(b, a, x)
    
    return x
