"""Select data vectors by similarity using a metric score.
"""

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'simila.py v.1.0.0 20123/07/31'
__license__ = "MIT"

import numpy as np


def mse(y: np.ndarray, axis1: int=0, axis2: int=1, central: callable=np.nanmedian,
         normalization: callable=np.nanmedian
         ) -> np.ndarray:
    """Mean Squared Error of `y` w.r.t. `central` across axis2 over axis1.

    Parameters
    ----------
    y : numpy array
        an at least 2D array of data for the calculation of mean squared error
        w.r.t. to a `central` statistics.
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
    .. [1] https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Similarity.ipynb

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


    score = np.empty((y.shape[axis2]))
    score.fill(np.nan)
    idx = np.where(~np.all(np.isnan(y), axis=axis1))[0]  # masked array is slow
    y = y.swapaxes(0, axis2)[idx, ...].swapaxes(0, axis2)  # faster than .take
    score[idx] = np.nanmean((y - central(y, axis=axis2, keepdims=True))**2,
                            axis=axis1)
    if normalization is not None:
        score = score/normalization(score)
    return score


def similarity(y: np.ndarray, axis1: int=0, axis2: int=1, threshold: float=0,
               nmin: int=3, recursive: bool=True, metric: callable=mse,
               drop=False, msg: bool=True, **kwargs: callable
               ) ->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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
        minimum of [q[1] + 1.5*(q[2]-q[0]), score[-2], 3], where q's are the
        quantiles and score[-2] is the before-last largest score of `metric`
        among vectors calculated at the first time, not updated by `recursive`
        option.
    nmin : integer, optional (default = 3)
        If greater than 0, minumum number of vectors to keep.
        If lower than 0, maximum number of vectors to discard.
    recursive :bool, optional (default = True)
        Whether to calculate similarity `metric` recursevely, updating the score
        calculation each time a vector is discarded.
        With `recursive` True, the output `scores` will contain at each row
        the updated score values for the used `metric` for each data vector.
        The first row will contain the calculated original scores before any
        vector was discarded. On the subsequent rows, the vector discarded is
        represented by a NaN value and the kept vectors by their updated scores.
        The last row will contain the updated scores of the final vectors kept.
        With the `recursive` False, the comparison of score values with `threshold`
        are made only once and vectors are discarded accordingly at once.
        In this case, the output `scores` will contain only two rows, the
        first row will contain the calculated original scores before any vectors
        were discarded. At the second row, the vectors discarded are represented
        with NaN values and the kept vectors by their updated scores.
    metric : optional (default=mse)
        Function to use as metric to compute similarity.
    drop : bool, optional (default = False)
        Whether to drop (delete) the discarded vectors from `y` in the output.
        If False, the values of the vectors discarded will be replaced by nans
        and the returned array will have the same dimensions as the original
        array.
    msg : bool, optional (default = True)
        Whether to print some messages.
    kwargs : optional
        Options for the metric function (e.g., see `mse` function).

    Returns
    -------
    y : numpy array
        Array similar to input `y` but with vectors discarded (deleted) if
        option `drop` is True or with all values of vectors discarded replaced
        by nans if option `drop` is False.
    ikept : numpy array
        Indexes of kept vectors.
    inotkept : numpy array
        Indexes of not kept (discarded or replaced by nan) vectors.
    scores : 2-D numpy array
        Metric score values of each vector (as columns) for each round of
        vector selection (one row per round plus the final values).

    References
    ----------
    .. [1] https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/Similarity.ipynb

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
    >>> ys, ikept, inotkept, scores = similarity(y)
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


    if y.ndim < 2:
        raise ValueError('The input array must be at least a 2-D array.')
    y = y.copy()
    score = metric(y, axis1=axis1, axis2=axis2, **kwargs)
    scores = np.atleast_2d(score)
    ikept = np.where(~np.isnan(score))[0]  # indexes of kept vectors
    # indexes of not kept (discarded) vectors
    inotkept = np.where(np.isnan(score))[0]
    idx = np.argsort(score)
    score = score[idx]
    nkept = np.count_nonzero(~np.isnan(score))  # number of kept vectors
    if nkept < 3:
        raise ValueError('The input array must have at least 3 valid vectors.')
    if nmin < 0:
        nmin = np.max([3, nkept + nmin])
    if threshold == 0:
        q = np.nanquantile(a=score, q=[.25, .50, .75])
        threshold = np.min([q[1] + 1.5*(q[2]-q[0]), score[-2], 3])
        if msg:
            print(f'Calculated threshold: {threshold}')
    if not recursive:  # discard all vectors at once
        idx2 = np.nonzero(score > threshold)[0]  # vectors to discard
        if len(idx2) > 0:
            if nkept > nmin:  # keep at least nmin vectors
                inotkept = np.r_[inotkept, idx[idx2[-(y.shape[axis2] - nmin):]][::-1]]
                y.swapaxes(0, axis2)[inotkept, ...] = np.nan
                score = metric(y, axis1=axis1, axis2=axis2, **kwargs)
                scores = np.vstack((scores, score))
    else:  # discard vectors with largest updated score one by one
        while nkept > nmin and score[nkept-1] > threshold:
            inotkept = np.r_[inotkept, idx[nkept-1]]
            y.swapaxes(0, axis2)[inotkept[-1], ...] = np.nan
            score = metric(y, axis1=axis1, axis2=axis2, **kwargs)
            scores = np.vstack((scores, score))
            idx = np.argsort(score)
            score = score[idx]
            nkept = nkept - 1

    if len(inotkept):
        ikept = np.setdiff1d(ikept, inotkept)
        if drop:
            y = y.swapaxes(0, axis2)[ikept, ...].swapaxes(0, axis2)
        if msg:
            print(
                f'Vectors discarded (in dimension {axis2}, n={len(inotkept)}): {inotkept}')

    return y, ikept, inotkept, scores
