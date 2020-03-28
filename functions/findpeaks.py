#!/usr/bin/env python

"""Find peaks in data based on amplitude."""

from __future__ import division, print_function
import warnings
import numpy as np


__author__ = 'Marcos Duarte, https://github.com/demotu'
__version__ = 'findpeaks.py v.1 2014/06/23'

warnings.warn('A newest version is available at https://pypi.org/project/detecta/')


def findpeaks(x, mph=None, mpd=1, threshold=None, edge='rising', kpsh=False,
              show=False, ax=None):

    """Find peaks in data based on amplitude.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, positive number}, optional (default = None)
        finds peaks that are greater than the minimum peak height.
    mpd : positive integer, optional (default = 1)
        finds peaks that are at least separated by minimum peak distance.
        this option slows down the code if `x` has several peaks (>1000);
        try to decrease the number os peaks by tuning the other parameters.
    threshold : {None, positive number}, optional (default = None)
        finds peaks that are greater than `threshold` w.r.t. its neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional
        (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), or only the
        falling edge ('falling'), or both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with the same height even if they are closer than `mpd`.
    show  : bool, optional (default = False)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    Of course, if you want to detect valleys instead of peaks, just negate the
    data (`ind_valleys = findpeaks(-x)`).

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/notebooks/FindPeaks.ipynb

    Examples
    --------
    >>> # find all peaks and plot data
    >>> x = np.random.randn(100)
    >>> ind = findpeaks(x, show=True)
    >>> print(ind)

    >>> from findpeaks import findpeaks
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # setting minimum peak height = 1 and plot data
    >>> findpeaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # setting minimum peak height = 0 and detect both edges
    >>> findpeaks(x, mph=1, edge='both', show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    # deal with NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
    # find indices of all peaks
    dx = np.diff(x)
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    if edge == 'rising' or edge == 'both':
        ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
    if edge == 'falling' or edge == 'both':
        ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # deal with NaN's
    if indnan.size:
        ind = ind[np.in1d(ind, indnan, invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold is not None:
        tmp = []
        for i in range(ind.size):
            if np.min([x[ind]-x[ind-1], x[ind]-x[ind+1]]) < threshold:
                tmp.append(ind[i])
        ind = np.delete(ind, tmp)
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height
                k = x[ind[i]] > x[ind] if kpsh else True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & k
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        _plot(x, mph, mpd, threshold, edge, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, ax, ind):
    """Plot results of the findpeaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8)

        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        ax.set_title('Peak detection (MPH=%s, MPD=%d, threshold=%s, edge=%s)'
                     % (str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()
