#!/usr/bin/env python

"""Automatic search of filter cutoff frequency based on residual analysis."""

from __future__ import division, print_function
import numpy as np
from scipy.signal import butter, filtfilt

__author__ = 'Marcos Duarte <duartexyz@gmail.com>'
__version__ = 'residual_analysis.py v.2 2014/05/31'


def residual_analysis(y, freq=1, fclim=[], show=False):
    """ Automatic search of filter cutoff frequency based on residual analysis.

    This method was proposed by Winter in his book [1]_.
    The 'optimal' cutoff frequency (in the sense that a filter with such cutoff
    frequency removes as much noise as possible without considerably affecting
    the signal) is found by performing a residual analysis of the difference
    between filtered and unfiltered signals over a range of cutoff frequencies.
    The optimal cutoff frequency is the one where the residual starts to change
    very little because it is considered that from this point, it's being
    filtered mostly noise and minimally signal, ideally.

    Parameters
    ----------
    y      : 1D array_like
             Data
    freq   : float, optional (default = 1)
             sampling frequency of the signal y
    fclim  : list with 2 numbers, optional (default = [])
             limit frequencies of the noisy part or the residuals curve
    show   : bool, optional (default = False)
             True (1) plots data in a matplotlib figure
             False (0) to not plot

    Returns
    -------
    fc_opt : float
             optimal cutoff frequency (None if not found)

    Notes
    -----
    A second-order zero-phase digital Butterworth low-pass filter is used.
    # The cutoff frequency is correctyed for the number of passes:
    # C = 0.802 # for dual pass; C = (2**(1/npasses) - 1)**0.25
    
    The matplotlib figure with the results will show a plot of the residual
    analysis with the optimal cutoff frequency, a plot with the unfiltered and
    filtered signals at this optimal cutoff frequency (with the RMSE of the
    difference between these two signals), and a plot with the respective
    second derivatives of these signals which should be useful to evaluate
    the quality of the optimal cutoff frequency found.

    Winter should not be blamed for the automatic search algorithm used here.
    The algorithm implemented is just to follow as close as possible Winter's
    suggestion of fitting a regression line to the noisy part of the residuals.

    This function performs well with data where the signal has frequencies
    considerably bellow the Niquist frequency and the noise is predominantly
    white in the higher frequency region.

    If the automatic search fails, the lower and upper frequencies of the noisy
    part of the residuals curve cam be inputed as a parameter (fclim).
    These frequencies can be chosen by viewing the plot of the residuals (enter
    show=True as input parameter when calling this function).


    It is known that this residual analysis algorithm results in oversmoothing
    kinematic data [2]_. Use it with moderation.
    This code is described elsewhere [3]_.

    References
    ----------
    .. [1] Winter DA (2009) Biomechanics and motor control of human movement.
    .. [2] http://www.clinicalgaitanalysis.com/faq/cutoff.html
    .. [3] http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/ResidualAnalysis.ipynb

    Examples
    --------
    >>> import numpy as np
    >>> from residual_analysis import residual_analysis
    >>> y = np.cumsum(np.random.randn(1000))
    >>> # optimal cutoff frequency based on residual analysis and plot:
    >>> fc_opt = residual_analysis(y, freq=1000, show=True)
    >>> # sane analysis but specifying the frequency limits and plot:
    >>> residual_analysis(y, freq=1000, fclim=[200,400], show=True)
    >>> # Not always it's possible to find an optimal cutoff frequency
    >>> # or the one found can be wrong (run this example many times):
    >>> y = np.random.randn(100)
    >>> residual_analysis(y, freq=100, show=True)

    """

    from scipy.interpolate import UnivariateSpline
    
    # Correct the cutoff frequency for the number of passes in the filter
    C = 0.802 # for dual pass; C = (2**(1/npasses) - 1)**0.25

    # signal filtering
    freqs = np.linspace((freq/2) / 101, (freq/2)*C, 101)
    res = []
    for fc in freqs:
        b, a = butter(2, (fc/C) / (freq / 2))
        yf = filtfilt(b, a, y)
        # residual between filtered and unfiltered signals
        res = np.hstack((res, np.sqrt(np.mean((yf - y) ** 2))))

    # find the optimal cutoff frequency by fitting an exponential curve
    # y = A*exp(B*x)+C to the residual data and consider that the tail part
    # of the exponential (which should be the noisy part of the residuals)
    # decay starts after 3 lifetimes (exp(-3), 95% drop)
    if not len(fclim) or np.any(fclim < 0) or np.any(fclim > freq / 2):
        fc1 = 0
        fc2 = 0.95*(len(freqs)-1)
        # log of exponential turns the problem to first order polynomial fit
        # make the data always greater than zero before taking the logarithm
        reslog = np.log(np.abs(res[fc1:fc2 + 1] - res[fc2]) +
                        10 * np.finfo(np.float).eps)
        Blog, Alog = np.polyfit(freqs[fc1:fc2 + 1], reslog, 1)
        fcini = np.nonzero(freqs >= -3 / Blog)  # 3 lifetimes
        fclim = [fcini[0][0], fc2] if np.size(fcini) else []
    else:
        fclim = [np.nonzero(freqs >= fclim[0])[0][0],
                 np.nonzero(freqs >= fclim[1])[0][0]]

    # find fc_opt with linear fit y=A+Bx of the noisy part of the residuals
    if len(fclim) and fclim[0] < fclim[1]:
        B, A = np.polyfit(freqs[fclim[0]:fclim[1]], res[fclim[0]:fclim[1]], 1)
        # optimal cutoff frequency is the frequency where y[fc_opt] = A
        roots = UnivariateSpline(freqs, res - A, s=0).roots()
        fc_opt = roots[0] if len(roots) else None
    else:
        fc_opt = None

    if show:
        _plot(y, freq, freqs, res, fclim, fc_opt, B, A)

    return fc_opt


def _plot(y, freq, freqs, res, fclim, fc_opt, B, A):
    """Plot results of the residual_analysis function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))
        plt.figure(num=None, figsize=(10, 5))
        ax1 = plt.subplot(121)
        plt.rc('axes', labelsize=12,  titlesize=12)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        ax1.plot(freqs, res, 'b.', markersize=9)
        time = np.linspace(0, len(y) / freq, len(y))
        ax2 = plt.subplot(222)
        ax2.plot(time, y, 'g', linewidth=1, label='Unfiltered')
        ydd = np.diff(y, n=2) * freq ** 2
        ax3 = plt.subplot(224)
        ax3.plot(time[:-2], ydd, 'g', linewidth=1, label='Unfiltered')
        if fc_opt:
            ylin = np.poly1d([B, A])(freqs)
            ax1.plot(freqs, ylin, 'r--', linewidth=2)
            ax1.plot(freqs[fclim[0]], res[fclim[0]], 'r>',
                     freqs[fclim[1]], res[fclim[1]], 'r<', ms=9)
            ax1.set_ylim(ymin=0, ymax=4 * A)
            ax1.plot([0, freqs[-1]], [A, A], 'r-', linewidth=2)
            ax1.plot([fc_opt, fc_opt], [0, A], 'r-', linewidth=2)
            ax1.plot(fc_opt, 0, 'ro', markersize=7, clip_on=False,
                     zorder=9, label='$Fc_{opt}$ = %.1f Hz' % fc_opt)
            ax1.legend(fontsize=12, loc='best', numpoints=1, framealpha=.5)
            # Correct the cutoff frequency for the number of passes
            C = 0.802 # for dual pass; C = (2**(1/npasses) - 1)**0.25
            b, a = butter(2, (fc_opt/C) / (freq / 2))
            yf = filtfilt(b, a, y)
            ax2.plot(time, yf, color=[1, 0, 0, .5],
                     linewidth=2, label='Opt. filtered')
            ax2.legend(fontsize=12, loc='best', framealpha=.5)
            ax2.set_title('Signals (RMSE = %.3g)' % A)
            yfdd = np.diff(yf, n=2) * freq ** 2
            ax3.plot(time[:-2], yfdd, color=[1, 0, 0, .5],
                     linewidth=2, label='Opt. filtered')
            ax3.legend(fontsize=12, loc='best', framealpha=.5)
            resdd = np.sqrt(np.mean((yfdd - ydd) ** 2))
            ax3.set_title('Second derivatives (RMSE = %.3g)' % resdd)
        else:
            ax1.text(.5, .5, 'Unable to find optimal cutoff frequency',
                     horizontalalignment='center', color='r', zorder=9,
                     transform=ax1.transAxes, fontsize=12)
            ax2.set_title('Signal')
            ax3.set_title('Second derivative')

        ax1.set_xlabel('Cutoff frequency [Hz]')
        ax1.set_ylabel('Residual RMSE')
        ax1.set_title('Residual analysis')
        ax1.grid()
        #ax2.set_xlabel('Time [s]')
        ax2.set_xlim(0, time[-1])
        ax2.grid()
        ax3.set_xlabel('Time [s]')
        ax3.set_xlim(0, time[-1])
        ax3.grid()
        plt.tight_layout()
        plt.show()
