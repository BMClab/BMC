#!/usr/bin/env python

"""Estimate power spectral density characteristcs using Welch's method."""

from __future__ import division, print_function
import numpy as np

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'tnorm.py v.1 2013/09/16'


def psd(x, fs=1.0, window='hanning', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', show=True, ax=None, scales='linear', xlim=None,
        units='V'):
    """Estimate power spectral density characteristcs using Welch's method.

    This function is just a wrap of the scipy.signal.welch function with
    estimation of some frequency characteristcs and a plot. For completeness,
    most of the help from scipy.signal.welch function is pasted here.

    Welch's method [1]_ computes an estimate of the power spectral density
    by dividing the data into overlapping segments, computing a modified
    periodogram for each segment and averaging the periodograms.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series in units of Hz. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length will be used for nperseg.
        Defaults to 'hanning'.
    nperseg : int, optional
        Length of each segment.  Defaults to half of `x` length.
    noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg / 2``.  Defaults to None.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired.  If None,
        the FFT length is `nperseg`. Defaults to None.
    detrend : str or function, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`. If it is a
        function, it takes a segment and returns a detrended segment.
        Defaults to 'constant'.
    show : bool, optional (default = False)
        True (1) plots data in a matplotlib figure.
        False (0) to not plot.
    ax : a matplotlib.axes.Axes instance (default = None)
    scales : str, optional
        Specifies the type of scale for the plot; default is 'linear' which
        makes a plot with linear scaling on both the x and y axis.
        Use 'semilogy' to plot with log scaling only on the y axis, 'semilogx'
        to plot with log scaling only on the x axis, and 'loglog' to plot with
        log scaling on both the x and y axis.
    xlim : float, optional
        Specifies the limit for the `x` axis; use as [xmin, xmax].
        The defaukt is `None` which sets xlim to [0, Fniquist].
    units : str, optional
        Specifies the units of `x`; default is 'V'.

    Returns
    -------
    Fpcntile : 1D array
        frequency percentiles of the power spectral density
        For example, Fpcntile[50] gives the median power frequency in Hz.
    mpf : float
        Mean power frequency in Hz.
    fmax : float
        Maximum power frequency in Hz.
    Ptotal : float
        Total power in `units` squared.
    f : 1D array
        Array of sample frequencies in Hz.
    P : 1D array
        Power spectral density or power spectrum of x.

    See Also
    --------
    scipy.signal.welch

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements.  For the default 'hanning' window an
    overlap of 50% is a reasonable trade off between accurately estimating
    the signal power, while not over counting any of the data.  Narrower
    windows may require a larger overlap.
    If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.

    Examples (also from scipy.signal.welch)
    --------
    >>> import numpy as np
    >>> from psd import psd
    #Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
    # 0.001 V**2/Hz of white noise sampled at 10 kHz and calculate the PSD:
    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2*np.sqrt(2)
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> x = amp*np.sin(2*np.pi*freq*time)
    >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> psd(x, fs=freq);
    """

    from scipy import signal, integrate

    if not nperseg:
        nperseg = np.ceil(len(x) / 2)
    f, P = signal.welch(x, fs, window, nperseg, noverlap, nfft, detrend)
    Area = integrate.cumtrapz(P, f, initial=0)
    Ptotal = Area[-1]
    mpf = integrate.trapz(f * P, f) / Ptotal  # mean power frequency
    fmax = f[np.argmax(P)]
    # frequency percentiles
    inds = [0]
    Area = 100 * Area / Ptotal  # + 10 * np.finfo(np.float).eps
    for i in range(1, 101):
        inds.append(np.argmax(Area[inds[-1]:] >= i) + inds[-1])
    fpcntile = f[inds]

    if show:
        _plot(x, fs, f, P, mpf, fmax, fpcntile, scales, xlim, units, ax)

    return fpcntile, mpf, fmax, Ptotal, f, P


def _plot(x, fs, f, P, mpf, fmax, fpcntile, scales, xlim, units, ax):
    """Plot results of the ellipse function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        if scales.lower() == 'semilogy' or scales.lower() == 'loglog':
            ax.set_yscale('log')
        if scales.lower() == 'semilogx' or scales.lower() == 'loglog':
            ax.set_xscale('log')
        plt.plot(f, P, linewidth=2)
        ylim = ax.get_ylim()
        plt.plot([fmax, fmax], [np.max(P), np.max(P)], 'ro',
                 label='Fpeak  = %.2f' % fmax)
        plt.plot([fpcntile[50], fpcntile[50]], ylim, 'r', lw=1.5,
                 label='F50%%   = %.2f' % fpcntile[50])
        plt.plot([mpf, mpf], ylim, 'r--', lw=1.5,
                 label='Fmean = %.2f' % mpf)
        plt.plot([fpcntile[95], fpcntile[95]], ylim, 'r-.', lw=2,
                 label='F95%%   = %.2f' % fpcntile[95])
        leg = ax.legend(loc='best', numpoints=1, framealpha=.5,
                        title='Frequencies [Hz]')
        plt.setp(leg.get_title(), fontsize=12)
        plt.xlabel('Frequency [$Hz$]', fontsize=12)
        plt.ylabel('Magnitude [%s$^2/Hz$]' % units, fontsize=12)
        plt.title('Power spectral density', fontsize=12)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.tight_layout()
        plt.grid()
        plt.show()
