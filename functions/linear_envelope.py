#!/usr/bin/env python

"""Calculate the linear envelope of a signal."""

import numpy as np
from scipy.signal import butter, filtfilt, convolve

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'linear_envelope.py v.1.0.1 2019/03/13'


def linear_envelope(x, freq=1000, fc_bp=[10, 400], fc_lp=8, method='rect'):
    """Calculate the linear envelope of a signal.

    Parameters
    ----------
    x : 1D array_like
        Raw signal
    freq : number, optional. Default = 1000
        Sampling frequency.
    fc_bp : list of floats [fc_h, fc_l] or None, optional. Default = [10, 400]
        Cutoff frequencies for the band-pass filter (in Hz). Enter None to
        not use this option.
    fc_lp : number, optional. Default = 8
        Cutoff frequency for the low-pass filter (in Hz).
    method: string, optional. Default = 'rect'
        Method to calculate the linear envelope. Enter 'rect' for using a
        full-wave rectification followed by low-pass Butterworth filter.
        Enter 'rms' for a moving-RMS filter. For the 'rms' method, the
        window size is computed as the nearest integer of freq/fc_lp and
        the filter is applied twice calling convolution twice.

    Returns
    -------
    y : 1D array_like
        Linear envelope of the signal

    Notes
    -----
    A 2nd-order Butterworth filter with zero lag is used for the filtering
    The moving-RMS filter is calculated by convolution with the paramter mode
    equal to 'same', so the extremities of the signal of size 'window'/2 are
    not reliable.

    See this notebook [1]_.

    References
    ----------
    .. [1] https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb

    Version history
    ---------------
    '1.0.1':
        Included parameter 'method', to select method for the linear envelope.
        Changed parameters fc_bp and fc_lp to be written in lowercase.
    """
    if fc_bp is not None:
        fc_bp = np.array(fc_bp)
        # band-pass filter
        b, a = butter(N=2, Wn=fc_bp/(freq/2), btype='bandpass')
        x = filtfilt(b, a, x)

    if method == 'rect':
        # full-wave rectification
        x = abs(x)
        # low-pass Butterworth filter
        b, a = butter(N=2, Wn=fc_lp/(freq/2), btype='low')
        y = filtfilt(b, a, x)
    elif method == 'rms':
        y = moving_rms(x, freq/fc_lp)
    else:
        raise ValueError("Valid method is 'rect' or 'rms'")

    return y


def moving_rms(x, window):
    """Moving RMS of 'x' with window size 'window'.
    """
    window = int(np.round(window))
    y = np.sqrt(convolve(x*x, np.ones(window)/window, 'same', method='direct'))

    return y
