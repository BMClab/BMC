#!/usr/bin/env python

"""Detect indices in x of sequential data identical to value."""

import numpy as np

__author__ = 'Marcos Duarte, https://github.com/demotu'
__version__ = 'detect_seq.py v.1.0.1 2019/03/17'


def detect_seq(x, value=np.nan, index=False, min_seq=1, max_alert=0,
               show=False, ax=None):
    """Detect indices in x of sequential data identical to value.

    Parameters
    ----------
    x : 1D numpy array_like
        data
    value : number, optional. Default = np.nan
        Value to be found in data
    index : bool, optional. Default = False
        Set to True to return a 2D array of initial and final indices where
        data is equal to value or set to False to return an 1D array of Boolean
        values with same length as x with True where x is equal to value and 
        False where x is not equal to value.
    min_seq : integer, optional. Default = 1
        Minimum number of sequential values to detect        
    max_alert : number, optional. Default = 0
        Minimal number of sequential data for a message to be printed with
        information about these indices. Set to 0 to not print any message.
    show : bool, optional. Default = False
        Show plot (True) of not (False).
    ax : matplotlib object, optional. Default = None
        Matplotlib axis object where to plot.
        
    Returns
    -------
    idx : 1D or 2D numpy array_like
        2D numpy array [indi, indf] of initial and final indices (if index is
        equal to True) or 1D array of Boolean values with same length as x (if
        index is equal to False).
            
    References
    ----------
    .. [1] https://github.com/demotu/detecta/blob/master/docs/detect_seq.ipynb

    Examples
    --------
    >>> x = [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]
    >>> detect_seq(x, 0)

    >>> detect_seq(x, 0, index=True)

    >>> detect_seq(x, 0, index=True, min_seq=2)  

    >>> detect_seq(x, 10)

    >>> detect_seq(x, 10, index=True)

    >>> detect_seq(x, 0, index=True, min_seq=2, show=True)

    >>> detect_seq(x, 0, index=True, max_alert=2)

    Version history
    ---------------
    '1.0.1':
        Part of the detecta module - https://pypi.org/project/detecta/  
    """

    idx = np.r_[False, np.isnan(x) if np.isnan(value) else np.equal(x, value), False]

    if index or min_seq > 1 or max_alert or show:
        idx2 = np.where(np.abs(np.diff(idx))==1)[0].reshape(-1, 2)
        if min_seq > 1:
            idx2 = idx2[np.where(np.diff(idx2, axis=1) >= min_seq)[0]]
            if not index:
                idx = idx[1:-1]*False
                for i, f in idx2:
                    idx[i:f] = True           
        idx2[:, 1] = idx2[:, 1] - 1

    if index:
        idx = idx2
    elif len(idx) > len(x):
        idx = idx[1:-1]

    if max_alert and idx2.shape[0]:
        seq = np.diff(idx2, axis=1)
        for j in range(idx2.shape[0]):
            bitlen = seq[j]
            if bitlen >= max_alert:
                text = 'Sequential data equal or longer than {}: ({}, {})'
                print(text.format(max_alert, bitlen, idx2[j]))

    if show:
        _plot(x, value, min_seq, ax, idx2)
        
    return idx


def _plot(x, value, min_seq, ax, idx):
    """Plot results of the detect_seq function, see its help.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        x = np.asarray(x)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        if idx.size:
            for (indi, indf) in idx:
                if indi == indf:
                    ax.plot(indf, x[indf], 'ro', mec='r', ms=6)
                else:
                    ax.plot(range(indi, indf+1), x[indi:indf+1], 'b', lw=1)
                    ax.axvline(x=indi, color='r', lw=1, ls='--')
                    ax.plot(indi, x[indi], 'r>', mec='r', ms=6)
                    ax.plot(indf, x[indf], 'r<', mec='r', ms=6)
                ax.axvline(x=indf, color='r', lw=1, ls='--')
            idx = np.vstack((np.hstack((0, idx[:, 1])),
                             np.hstack((idx[:, 0], x.size-1)))).T
            for (indi, indf) in idx:
                ax.plot(range(indi, indf+1), x[indi:indf+1], 'k', lw=1)
        else:
            ax.plot(x, 'k', lw=1)

        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        text = 'Value=%.3g, minimum number=%d'
        ax.set_title(text % (value, min_seq))
        plt.show()
