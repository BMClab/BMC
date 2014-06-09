#!/usr/bin/env python

"""Detects onset in data based on amplitude threshold."""

from __future__ import division, print_function
import numpy as np

__author__ = 'Marcos Duarte, https://github.com/duartexyz/BMC'
__version__ = 'onset_detection.py v.2 2014/06/09'


def onset_detection(data, threshold=0, n_above=1, n_below=0, show=False, ax=None):
    """Detects onset in data based on amplitude threshold.

    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : number, optional (default = 0)
        minimum amplitude of the data to detect.
    n_above : number, optional (default = 1)
        minimum number of continuous samples greater than or equal to
        threshold to detect (but see the parameter n_below).
    n_below : number, optional (default = 0)
        minimum number of samples (continuous or not) below threshold
        that will be ignored in the detection of data >= threshold.
    show  : bool, optional (default = True)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    inds : 1D array_like [indi, indf]
        initial and final indeces of the onset events.

    Notes
    -----
    You might have to tune the parameters according to the signal-to-noise
    characteristic of the data.

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/OnsetDetection.ipynb

    Examples
    --------
    >>> from onset_detection import onset_detection
    >>> x = np.random.randn(100)
    >>> onset_detection(x, threshold=0, n_above=10, n_below=1, show=True)

    >>> x = np.random.randn(200)/10 
    >>> x[51:151] += np.hstack((np.linspace(0,1,50), np.linspace(1,0,50)))
    >>> inds = onset_detection(x, np.std(x[:50]), 10, 0, True)
    >>> inds

    >>> x = [0, 0, 2, 0, np.nan, 0, 2, 3, 3, 0, 1, 1, 0]
    >>> onset_detection(x, threshold=1, n_above=1, n_below=0, show=True)
    """

    data = np.atleast_1d(data).astype('float64')    
    # deal with NaN's (by definition, NaN's are not greater than threshold)
    data[np.isnan(data)] = -np.inf
    if not data[np.isfinite(data)].size:
        return np.array([])
    # indices of data greater than or equal to threshold
    inds = np.nonzero(data >= threshold)[0]
    if inds.size:
        # initial and final indexes of continuous data
        inds = np.vstack((inds[np.diff(np.hstack((-np.inf, inds))) > n_below+1], \
                          inds[np.diff(np.hstack((inds, np.inf))) > n_below+1])).T
        # indexes of continuous data longer than or equal to n_above
        inds = inds[inds[:, 1]-inds[:, 0] >= n_above-1, :]
    if not inds.size:
        inds = np.array([])  # standardize inds shape
    if show and data.size > 1:  # don't waste my time ploting one datum
        _plot(data, threshold, n_above, n_below, inds, ax)

    return inds


def _plot(data, threshold, n_above, n_below, inds, ax):
    """Plot results of the cogve function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        if inds.size:
            for (indi, indf) in inds:
                if indi == indf:
                    ax.plot(indf, data[indf], 'ro', mec='r', ms=6)
                else:
                    ax.plot(range(indi, indf+1), data[indi:indf+1], 'r', lw=1)
                    ax.axvline(x=indi, color='b', lw=1, ls='--')
                ax.axvline(x=indf, color='b', lw=1, ls='--')
            inds = np.vstack((np.hstack((0, inds[:, 1])),
                              np.hstack((inds[:, 0], data.size-1)))).T
            for (indi, indf) in inds:
                ax.plot(range(indi, indf+1), data[indi:indf+1], 'k', lw=1)
        else:
            ax.plot(data, 'k', lw=1)
            ax.axhline(y=threshold, color='r', lw=1, ls='-')

        ax.set_xlim(-.02*data.size, data.size*1.02-1)
        ymin, ymax = data[np.isfinite(data)].min(), data[np.isfinite(data)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        ax.set_title('Onset detection (threshold=%.3g, n_above=%d, n_below=%d)'\
                     %(threshold, n_above, n_below))
        #plt.grid()
        plt.show()
