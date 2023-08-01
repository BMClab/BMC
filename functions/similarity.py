"""Select data vectors by similarity using a metric score.
"""

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'similarity.py v.1.0.0 20123/07/31'
__license__ = "MIT"

import numpy as np

def mse(y, axis1=0, axis2=1, central=np.nanmedian, normalization=np.nanmedian):
    """Mean Squared Error of `y` w.r.t. `central` across axis2 over axis1.

    Parameters
    ----------
    y : numpy array
        array for the calculation of mse w.r.t. to a central statistics
    axis1 : integer, optional (default = 0)
        axis to slice `y` ndarray in the calculation of mse.
    axis2 : integer, optional (default = 1)
        axis to slice `y` ndarray in the calculation of the `central`.
    central : Python function, optional (default = np.nanmedian)
        function to calculate statistics on `y` w.r.t. mse is computed.
    normalization : Python function, optional (default = np.nanmedian)
        function to normalize the calculated mse values

    Returns
    -------
    score : numpy array
        Mean Squared Error values

    References
    ----------
    .. [1] https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/similarity.ipynb

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> y = rng.random((100, 10))
    >>> y +=  np.atleast_2d(np.sin(2*np.pi*np.linspace(0, 1, 100))).T
    >>> mse(y, axis1=0, axis2=1, central=np.nanmedian, normalization=np.nanmedian)

    Version history
    ---------------
    '1.0.0':
        First release version
    """

    import numpy as np

    score = np.empty((y.shape[axis2]))
    score.fill(np.nan)
    idx = np.where(~np.all(np.isnan(y), axis=axis1))[0]  # masked array is slow
    y = y.swapaxes(0, axis2)[idx, ...].swapaxes(0, axis2)  # faster than .take
    score[idx] = np.nanmean((y - central(y, axis=axis2, keepdims=True))**2, axis=axis1)
    if normalization is not None:
        score = score/normalization(score)
    return score


def similarity(y, axis1=0, axis2=1, threshold=0, nmin=3,
               recursive=True, metric=mse, msg=True, **kwargs):
    """Select data vectors by similarity using a metric score.

    Parameters
    ----------
    y : numpy array
        Array for the calculation of mse w.r.t. to a central statistics.
    axis1 : integer, optional (default = 0)
        Axis to slice `y` ndarray in the calculation of mse.
    axis2 : integer, optional (default = 1)
        Axis to slice `y` ndarray in the calculation of the `central`.
    threshold : float, optional (default = 0)
        If greater than 0, vector with mse above it will be discarded.
        If 0, threshold will be automatically calculated as the
        minimum of [pct[1] + 1.5*(pct[2]-pct[0]), score[-2], 3], where
        score[-2] is the before largest mse value among the vectors
        calculated at the first time (not updated by `recursive` option.
    nmin : interger, optional (default = 3)
        If greater than 0, minumum number of vectors to keep.
        If lower than 0, maximum number of vectors to discard.
    recursive :bool, optional (default = True)
        Whether to calculate similarity metric recursevely. With the
        recursive option, the mse values are computed again each time a
        vector is discarded.
    metric :
        Function to use as metric to compute similarity.
    msg : bool, optional (default = True)
        Whether to print some messages.
    kwargs : optional
        Options for the metric function (see mse function).

    Returns
    -------
    y : numpy array
        Array similar to input `y` but with vectors discarded.
    ikept : numpy array
        Indexes of kept vectors.
    inotkept : numpy array
        Indexes of not kept (discarded) vectors.
    score_all : numpy array
        Mean Squared Error values.

    References
    ----------
    .. [1] https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/similarity.ipynb

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> t, n = 100, 10
    >>> y = rng.random((t, n))
    >>> y +=  np.atleast_2d(2*np.sin(2*np.pi*np.linspace(0, 1, t))).T
    >>> for i in range(0, n, 2):
    >>>    j = rng.integers(t-20)
    >>>    p = rng.integers(20)
    >>>    y[j:j+p, i] = y[j:j+p, i] + rng.integers(10) - 5
    >>>    y[:, i] += rng.integers(4) - 2
    >>> ys, ikept, inotkept, score_all = similarity(y)
    >>> fig, axs = plt.subplots(2, 1, sharex=True)
    >>> axs[0].plot(y, label=list(range(n)))
    >>> axs[0].legend(loc=(1.01, 0))
    >>> axs[1].plot(ys, label= ikept.tolist())
    >>> axs[1].legend(loc=(1.01, 0))
    >>> plt.show()

    Version history
    ---------------
    '1.0.0':
        First release version
    """

    import numpy as np

    if y.ndim < 2:
        raise Exception('The input array must be at least a 2-D array.')
    y = y.copy()
    score = metric(y, axis1=axis1, axis2=axis2, **kwargs)
    score_all = np.atleast_2d(score)
    ikept = np.where(~np.isnan(score))[0]  # indexes of kept vectors
    inotkept = np.where(np.isnan(score))[0]  # indexes of not kept (discarded) vectors
    idx = np.argsort(score)
    score = score[idx]
    n = np.count_nonzero(~np.isnan(score))  # number of kept vectors
    if n < 3:
        raise Exception('The input array must have at least 3 valid vectors.')
    if nmin < 0:
        nmin = np.max([3, n + nmin])
    if threshold == 0:
        pct = np.nanpercentile(score, [25, 50, 75])
        threshold = np.min([pct[1] + 1.5*(pct[2]-pct[0]), score[-2], 3])
        if msg:
            print(f'Calculated threshold: {threshold}')
    if not recursive:  # discard all vectors at once
        idx2 = np.nonzero(score > threshold)[0]  # vectors to discard
        if len(idx2) > 0:
            if n > nmin:  # keep at least nmin vectors
                inotkept = np.r_[inotkept, idx[idx2[-(y.shape[axis2] - nmin):]][::-1]]
                y.swapaxes(0, axis2)[inotkept, ...] = np.nan
                score = metric(y, axis1=axis1, axis2=axis2, **kwargs)
                score_all = np.vstack((score_all, score))
            elif msg:
                print(f'Number of vectors to discard is greater than number to keep ({n}).')
    else:  # discard vectors with largest updated score one by one
        while n > nmin and score[n-1] > threshold:
            inotkept = np.r_[inotkept, idx[n-1]]
            y.swapaxes(0, axis2)[inotkept[-1], ...] = np.nan
            score = metric(y, axis1=axis1, axis2=axis2, **kwargs)
            score_all = np.vstack((score_all, score))
            idx = np.argsort(score)
            score = score[idx]
            n = n - 1
        if msg and n == nmin and score[n-1] > threshold:
            print(f'Number of vectors to discard is greater than number to keep ({n}).')

    if len(inotkept):
        ikept = np.setdiff1d(ikept, inotkept)
        y = y.swapaxes(0, axis2)[ikept, ...].swapaxes(0, axis2)
        if msg:
            print(f'Data vectors discarded (in dimension {axis2}, n={len(inotkept)}): {inotkept}')

    return y, ikept, inotkept, score_all