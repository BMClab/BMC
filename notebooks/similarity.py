# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Select data vectors by similarity using a metric score

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this notebook

    Treat each column as a trial and each row as a common time sample. Predict which trials will receive large scores, then compare repeated selection with one-pass selection. Keep the selected and excluded indices so that every decision can be traced back to the original trial.

    ## Opening challenge

    Ten trajectories mostly agree, but two differ. Are those two corrupt measurements, or valid movements from a second condition? A similarity rule can flag differences; it cannot decide their cause. This guide makes that distinction explicit while showing how the algorithm works.

    ## Setup

    NumPy and Matplotlib support the simulation and plots. The example retains the `simila` algorithm and its documentation below, uses seed 0, and needs no external dataset. Rows must already represent corresponding times or phases: similarity does not perform time alignment.
    """)
    return


@app.function
def repo_file(path, *, base_url="https://raw.githubusercontent.com/BMClab/BMC/master/"):
    """Find a repository file beside this notebook, or fetch a temporary copy.

    Local files take precedence. Only file acquisition falls back to GitHub;
    parsing errors remain visible to the calling notebook. urllib3 selects its
    browser transport automatically without platform-specific imports.
    """
    import hashlib
    import tempfile
    from pathlib import Path, PurePosixPath
    from urllib.parse import quote

    import marimo as mo
    import urllib3

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise ValueError("Use a repository-relative path such as data/walk.trc")
    location = mo.notebook_dir()
    if location is not None:
        directory = Path(location).resolve()
        for base in (directory, *directory.parents):
            local = base.joinpath(*relative.parts)
            if local.is_file():
                return local
    url = base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")
    key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]
    cached = (Path(tempfile.gettempdir()) / "bmc-notebooks-v2" / key).joinpath(*relative.parts)
    if not cached.is_file():
        response = urllib3.request("GET", url, timeout=30.0, retries=2)
        if response.status != 200:
            raise OSError(f"Cannot load {path}: HTTP {response.status} from {url}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as download:
            temporary = Path(download.name)
            download.write(response.data)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    return cached


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=3)
    return np, plt


@app.cell
def _(similarity):
    help(similarity)
    return


@app.cell
def _(mse):
    help(mse)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define a score before defining an exclusion

    For $n$ matched samples, the unnormalized mean squared error is $s_j=n^{-1}\sum_i(y_{ij}-r_i)^2$. It has squared signal units and is squared Euclidean distance **divided by $n$** for complete vectors of equal length. MSE does not satisfy the triangle inequality, so “metric” here means a score function, not a mathematical distance metric.

    The default reference is the pointwise median across trials. Dividing by the median of the trial scores produces a relative, dimensionless score when that scale is positive. With an exactly zero scale, the revised implementation assigns zero to exact matches and infinity to nonzero error; missing trials remain NaN. Use `normalization=None` when an absolute, unit-bearing threshold is more appropriate.

    **Exercise 1 — calculate the score.** Reference samples are $(1,2,3)$ and a trial is $(2,2,5)$. Find squared Euclidean distance and MSE. If the median trial MSE is 0.5, find the normalized score.

    /// details | Worked solution — Exercise 1

    The squared differences are $(1,0,4)$, whose sum is 5. MSE is $5/3$. Dividing by 0.5 gives $10/3$. All three values answer different questions; specify both normalization and sample correspondence when comparing them.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example
    """)
    return


@app.cell
def _(np, plt, similarity):
    _rng = np.random.default_rng(0)
    _t, _n = 100, 10
    y = _rng.random((_t, _n)) / 2
    y +=  np.atleast_2d(2*np.sin(2*np.pi*np.linspace(0, 1, _t))).T
    for _i in range(0, _n, 2):
       _j = _rng.integers(_t-20)
       _p = _rng.integers(20)
       y[_j:_j+_p, _i] = y[_j:_j+_p, _i] + _rng.integers(10) - 5
       y[:, _i] += _rng.integers(4) - 2
    _ysr, _ikeptr, _inotkeptr, _scoresr = similarity(y)
    _ysn, _ikeptn, _inotkeptn, _scoresn = similarity(y, repeat=False)
    _fig, _axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
    _axs[0].plot(y, label=list(range(_n)))
    _axs[0].legend(loc=(1.01, 0), ncol=2)
    _axs[0].set_title(f'Original vectors (n={_n})')
    _axs[1].plot(_ysr, label= _ikeptr.tolist())
    _axs[1].set_title(f'Vectors maintained with repeat selection (n={len(_ikeptr)})')
    _axs[1].legend(loc=(1.01, 0))
    _axs[2].plot(_ysn, label= _ikeptn.tolist())
    _axs[2].set_title(f'Vectors maintained without repeat selection (n={len(_ikeptn)})')
    _axs[2].legend(loc=(1.01, 0))
    plt.show()
    return (y,)


@app.cell
def _(mse, np, similarity, y):
    ys, ikept, inotkept, scores = similarity(y, threshold=0, nmin=3, repeat=True, metric=mse,
                                             msg=1, central=np.nanmedian, normalization=np.nanmedian)
    return ikept, inotkept, scores, ys


@app.cell
def _(ys):
    ys.shape
    return


@app.cell
def _(ikept):
    ikept
    return


@app.cell
def _(inotkept):
    inotkept
    return


@app.cell
def _(scores):
    scores
    return


@app.cell
def _(mse, np, similarity, y):
    (ys_1, ikept_1, inotkept_1, scores_1) = similarity(y, threshold=0, nmin=3, repeat=True, metric=mse, drop=False, msg=True, central=np.nanmedian, normalization=np.nanmedian)
    return (ys_1,)


@app.cell
def _(ys_1):
    ys_1.shape
    return


@app.cell
def _(mse, np, similarity, y):
    (ys_2, ikept_2, inotkept_2, scores_2) = similarity(y, threshold=0, nmin=3, repeat=False, metric=mse, msg=True, central=np.nanmedian, normalization=np.nanmedian)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — retention and iteration.** Five trial scores are (0.5,0.7,1,2,8), the threshold is 3, and at least three trials must be retained. Which trial is removed in one pass? Why could repeated selection lead to another decision?

    /// details | Worked solution — Exercise 2 — retention and iteration

    The fifth trial exceeds 3 and is removed, leaving four. Repeated selection recomputes the reference and normalized scores after each removal, so the remaining values need not stay (0.5,0.7,1,2). The minimum-count rule limits how many may be excluded; it does not certify the quality of the retained trials.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function profiling
    """)
    return


@app.cell
def _(similarity, y):
    (ys_3, ikept_3, inotkept_3, score_all) = similarity(y, repeat=True, msg=False)
    return


@app.cell
def _(similarity, y):
    (ys_4, ikept_4, inotkept_4, score_all_1) = similarity(y, repeat=False, msg=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The original profiling run attributed 86% of runtime to the score calculation. That is a historical measurement, not a performance guarantee. Compare the actual functions on the current 100-by-10 example below.
    """)
    return


@app.cell
def _(mse, np, y):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    score = mse(y, central=np.nanmedian, normalization=np.nanmedian)
    return


@app.cell
def _(mse, np, y):
    score_1 = mse(y, central=np.nanmean, normalization=np.nanmean)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The original run reported a roughly fourfold median/mean timing difference. The ratio depends on data shape, missingness, versions and hardware; measure it again. Mean and median also estimate different centres, so speed alone does not decide which is suitable.
    """)
    return






@app.cell
def _(mse, np, y):
    from timeit import repeat
    _median_s = min(repeat(lambda: mse(y, central=np.nanmedian, normalization=np.nanmedian), number=20, repeat=3)) / 20
    _mean_s = min(repeat(lambda: mse(y, central=np.nanmean, normalization=np.nanmean), number=20, repeat=3)) / 20
    print(f"100 × 10 data: median {_median_s:.6g} s/call; mean {_mean_s:.6g} s/call; ratio {_median_s/_mean_s:.2f}")
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function `similarity`
    """)
    return


@app.cell
def _(np):
    """Select vectors in array by their similarity using a metric score.
    """

    import logging
    from typing import Callable

    __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version__ = 'simila.py v.1.0.0 20123/07/31'
    __license__ = "MIT"

    # set logger and configure it not to confuse with matplotlib debug messages
    logger = logging.getLogger(__name__)
    logger.setLevel(20)  # 0: NOTSET, 10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR, 50: CRITICAL
    ch = logging.StreamHandler()
    formatter = logging.Formatter('{name}: {levelname}: {message}', style='{')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    def mse(y: np.ndarray, target: np.ndarray | None = None, axis1: int = 0, axis2: int = 1,
            central: Callable = np.nanmedian, normalization: Callable = np.nanmedian
            ) -> np.ndarray:
        """Mean Squared Error of `y` w.r.t. `target` or `central` along `axis2` at `axis1`.

        Parameters
        ----------
        y : numpy.ndarray
            At least a 2-D numpy.ndarray of data for the calculation of mean squared
            error w.r.t. a `target` or a `central` statistics of the data.
        target : 1-D numpy.ndarray of length `axis1`, optional, default = None
            Reference value to calculate the mean squared error of `y` w.r.t. this
            vector. If it is None, the mse value will be calculated w.r.t. a `central`
            calculated along `axis2` of `y`.
        axis1 : integer, optional, default = 0
            Axis of `y` for which the mse will be calculated at each value.
        axis2 : integer, optional, default = 1
            Axis of `y` along which the `central` statistics might be calculated or
            along which the target will be subtracted.
        central : Python function, optional, default = np.nanmedian
            Function to calculate statistics on `y` w.r.t. mse is computed.
        normalization : Python function, optional, default = np.nanmedian
            Function to normalize the calculated mse values. With a zero scale,
            exact matches have score zero and positive errors have score infinity.

        Returns
        -------
        score : numpy.ndarray
            Mean Squared Error values

        References
        ----------
        .. [1] similarity.py

        Examples
        --------
        >>>     >>> rng = np.random.default_rng()
        >>> y = rng.random((100, 10))
        >>> y = y + np.atleast_2d(np.sin(2*np.pi*np.linspace(0, 1, 100))).T
        >>> mse(y, axis1=0, axis2=1, central=np.nanmedian, normalization=np.nanmedian)

        Version history
        ---------------
        '1.0.0':
            First release version
        """

        logger.debug('mse...')

        score: np.ndarray = np.empty((y.shape[axis2]), dtype=float)
        score.fill(np.nan)
        idx: np.ndarray = np.where(~np.all(np.isnan(y), axis=axis1))[0]
        y = y.swapaxes(0, axis2)[idx, ...].swapaxes(0, axis2)  # faster than .take
        if target is not None:
            logger.debug('target shape: %s', target.shape)
            reference = np.asarray(target)
            if reference.ndim == 1:
                shape = [1] * y.ndim
                shape[axis1] = reference.size
                reference = reference.reshape(shape)
            score[idx] = np.nanmean((y - reference)**2, axis=axis1)
        else:
            score[idx] = np.nanmean((y - central(y, axis=axis2, keepdims=True))**2, axis=axis1)

        if normalization is not None:
            scale = normalization(score)
            if scale == 0:
                score = np.where(score > 0, np.inf, score)
            else:
                score = score/scale
        logger.debug('idx: %s, score: %s', idx, score)

        return score


    def similarity(y: np.ndarray, axis1: int = 0, axis2: int = 1, threshold: float = 0,
                   nmin: int = 3, repeat: bool = True, metric: Callable = mse,
                   drop=True, msg: bool = True, **kwargs: Callable
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """Select vectors in array by their similarity using a metric score.

        For example, if `y` is a 2-D numpy.ndarray, with shape (n, m), `axis1`=0
        (n is the number of rows) and `axis2`=1 (m is the number of columns), this
        function will select the vectors along the columns, that are more similar
        to a `central` statistics of `y` or to a `target`, using a `metric` score.
        The metric score can be calculated repeatedly until all selected vectors
        have a `metric` score not greater than a `threshold`, but the minimum
        number of vectors to keep or the maximum number of vectors to discard
        can be specified with parameter `nmin`.

        The default `metric` and target are the mean squared error (`mse`) of `y`
        w.r.t. the median of `y` along `axis2`. The `mse` metric is equal
        to squared Euclidean distance divided by the sample count and it is preferred because it
        penalizes largest differences more than the Euclidean distance. But any
        other `metric` that can be calculated with a function can be used.

        A possible use of this function is to discard the time-series data from bad
        trials (treated as outliers) stored in a 2-D array (each trial as a column
        and instants as rows) where the criterion is the similarity of the trial
        w.r.t. the median trial (the median statistics is more robust to remove
        outliers than the mean in case there are very bad trials). After the bad
        trials are discarded, the mean of all trials could then be calculated more
        reliably.

        Parameters
        ----------
        y : numpy.ndarray
            Array for the calculation of similarity (defined by a `metric`) of its
            vectors w.r.t. a `target` or a `central` statistics of `y`.
        axis1 : integer, optional, default = 0
            Axis of `y` for which the `metric` will be calculated at each value and
            possibly averaged in the `metric` calculation.
        axis2 : integer, optional, default = 1
            Axis of `y` along which the different vectors are going to be compared
            with the `threshold` for their similarity (using their `metric` score).
        threshold : float, optional, default = 0
            If greater than 0, vector with `metric` score above this value will be
            discarded. If 0, threshold will be automatically calculated as the
            minimum of [qs[2] + 1.5*(qs[2]-qs[0]), score[-2], 3], where qs are the
            quantiles and score[-2] is the before-last largest score of `metric`
            among vectors calculated at the first time, not updated by the `repeat`
            option.
        nmin : integer, optional, default = 3
            If greater than 0, minumum number of vectors to keep.
            If lower than 0, maximum number of vectors to discard.
        repeat : bool, optional, default = True
            Whether to calculate similarity `metric` repeatedly, updating the
            score calculation each time a vector is discarded.
            With `repeat` True, the output `scores` will contain at each row
            the updated score values for the used `metric` for each data vector.
            The first row will contain the calculated original scores before any
            vector was discarded. On the next equent rows, the vectors discarded
            are represented by NaN values and the kept vectors by their updated
            scores.
            The last row will contain the updated scores of the final vectors kept.
            With the `repeat` False, the comparison of score values with
            `threshold` are made only once and vectors are discarded accordingly at
            once. In this case, the output `scores` will contain only two rows, the
            first row will contain the calculated original scores before any
            vectors were discarded. At the second row, the vectors discarded are
            represented with NaN values and the kept vectors by their updated
            scores.
        metric : optional, default=mse
            Function to use as metric to compute similarity.
        drop : bool, optional, default = True
            Whether to drop (delete) the discarded vectors from `y` in the output.
            If False, the values of the vectors discarded will be replaced by nans
            and the returned array will have the same dimensions as the original
            array.
        msg : bool, optional, default = True
            Whether to print some messages.
        kwargs : optional
            Options for the metric function (e.g., see `mse` function).

        Returns
        -------
        y : numpy.ndarray
            Array similar to input `y` but with vectors discarded (deleted) if
            option `drop` is True or with all values of vectors discarded replaced
            by nans if option `drop` is False.
        ikept : numpy.ndarray
            Indexes of kept vectors.
        inotkept : numpy.ndarray
            Indexes of not kept (discarded or replaced by nan) vectors.
        scores : 2-D numpy.ndarray
            Metric score values of each vector (as columns) for each round of
            vector selection (one row per round plus the final values).

        References
        ----------
        .. [1] similarity.py

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> rng = np.random.default_rng()
        >>> t, n = 100, 10
        >>> y = rng.random((t, n)) / 2
        >>> y = y + np.atleast_2d(2*np.sin(2*np.pi*np.linspace(0, 1, t))).T
        >>> for i in range(0, n, 2):
        >>>    j = rng.integers(t-20)
        >>>    p = rng.integers(20)
        >>>    y[j:j+p, i] = y[j:j+p, i] + rng.integers(10) - 5
        >>>    y[:, i] += rng.integers(4) - 2
        >>> ysr, ikeptr, inotkeptr, scoresr = similarity(y)
        >>> ysn, ikeptn, inotkeptn, scoresn = similarity(y, repeat=False)
        >>> fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        >>> axs[0].plot(y, label=list(range(n)))
        >>> axs[0].legend(loc=(1.01, 0), ncol=2)
        >>> axs[0].set_title(f'Original vectors (n={n})')
        >>> axs[1].plot(ysr, label= ikeptr.tolist())
        >>> axs[1].set_title(f'Vectors maintained with repeat selection (n={len(ikeptr)})')
        >>> axs[1].legend(loc=(1.01, 0))
        >>> axs[2].plot(ysn, label= ikeptn.tolist())
        >>> axs[2].set_title(f'Vectors maintained without repeat selection (n={len(ikeptn)})')
        >>> axs[2].legend(loc=(1.01, 0))
        >>> plt.show()

        Version history
        ---------------
        '1.0.0':
            First release version
        """

        logger.debug('Similarity...')

        if y.ndim < 2:
            raise ValueError('The input array must be at least a 2-D array.')
        y = y.copy()
        score: np.ndarray = metric(y, axis1=axis1, axis2=axis2, **kwargs)
        scores: np.ndarray = np.atleast_2d(score)
        ikept: np.ndarray = np.where(~np.isnan(score))[0]  # indexes of kept vectors
        inotkept: np.ndarray = np.where(np.isnan(score))[0]  # indexes of discarded vectors
        idx: np.ndarray = np.argsort(score)
        score = score[idx]
        nkept: int = np.count_nonzero(~np.isnan(score))  # number of kept vectors
        if nkept < 3:
            logger.debug('nkept: %s', nkept)
            raise ValueError('The input array must have at least 3 valid vectors.')
        if nmin < 0:
            nmin = np.max([3, nkept + nmin])
        if threshold == 0:  # automatic threshold calculation
            # Infinite scores denote nonzero error against an exactly zero baseline.
            # Keep them eligible for rejection without interpolating 0 and infinity.
            qs: np.ndarray = np.quantile(a=score[np.isfinite(score)], q=[.25, .50, .75])
            threshold = np.min([qs[2] + 1.5*(qs[2]-qs[0]), score[-2], 3.])
            if msg:
                print(f'Calculated threshold: {threshold}')

        if not repeat:  # discard all vectors at once
            idx2: np.ndarray = np.nonzero(score > threshold)[0]  # vectors to discard
            if len(idx2) > 0:
                if nkept > nmin:  # keep at least nmin vectors
                    inotkept = np.r_[inotkept, idx[idx2[-(y.shape[axis2] - nmin):]][::-1]]
                    y.swapaxes(0, axis2)[inotkept, ...] = np.nan
                    score = metric(y, axis1=axis1, axis2=axis2, **kwargs)
                    scores = np.vstack((scores, score))
                    logger.debug('not repeat - score: %s', score)
        else:  # discard vector with largest updated score one by one
            while nkept > nmin and score[nkept-1] > threshold:
                inotkept = np.r_[inotkept, idx[nkept-1]]
                y.swapaxes(0, axis2)[inotkept[-1], ...] = np.nan
                score = metric(y, axis1=axis1, axis2=axis2, **kwargs)
                scores = np.vstack((scores, score))
                idx = np.argsort(score)
                score = score[idx]
                nkept = nkept - 1
                logger.debug('repeat - nkept: %s, score: %s', nkept, score)

        if len(inotkept):
            ikept = np.setdiff1d(ikept, inotkept)
            if drop:
                y = y.swapaxes(0, axis2)[ikept, ...].swapaxes(0, axis2)
            if msg:
                print(
                      f'Vectors discarded (dimension {axis2}, n={len(inotkept)}): {inotkept}')

        return y, ikept, inotkept, scores
    return mse, similarity


@app.cell
def _():
    import marimo as mo

    return (mo,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — keep the audit trail.** A 100-by-10 array loses trials with indices 2 and 7. Compare the shapes and interpretation with `drop=True` and `drop=False`. Then explain why two valid experimental conditions should not be pooled and filtered solely for agreement.

    /// details | Worked solution — Exercise 3

    Dropping columns returns shape (100,8); the kept indices map those columns back to the input. Keeping shape returns (100,10) with columns 2 and 7 set to NaN. Preserve both original data and exclusions. A genuine condition effect can make an entire subgroup dissimilar, so analyse experimental labels and measurement quality before treating difference as corruption.

    ///

    ## Checkpoint questions

    - Does a low score prove that a trial is physiologically correct?
    - Why can the final scores change after removing just one trial?
    - Which threshold has squared signal units, and which is dimensionless?
    - Does profiling establish that the faster reference estimator is scientifically preferable?

    /// details | Checkpoint answers

    No. Common artefacts may affect all trials and valid subgroups may differ. Reference and normalization are recomputed during repeated selection. Unnormalized MSE has squared signal units; normalization by a positive score scale is dimensionless. Timing measures cost for the stated input, while estimator choice depends on the scientific question and robustness needed.

    ///

    ## Problems

    1. For trial $(1,3)$ and reference $(1,1)$, report unnormalized MSE and squared Euclidean distance.
    2. Scores before normalization are $(1,2,4,8,16)$. Report median-normalized scores and the trial indices exceeding threshold 3 (zero-based indexing).
    3. With 20 samples and six trials, two trials are excluded. Give the output shape for both `drop` choices.
    4. Four trajectories are identical and a fifth differs; the median unnormalized error is zero. State the revised normalized scores for the exact matches and the different trial.
    5. Are retained indices sufficient to claim that excluded trials are measurement failures?

    /// details | Final answers to the problems

    1. 2 and 4.
    2. $(0.25,0.5,1,2,4)$; index 4.
    3. `(20,4)` for `drop=True`; `(20,6)` for `drop=False`.
    4. Zero for each exact match; positive infinity for the different trial.
    5. No; exclusion needs interpretation using acquisition and experimental context.

    ///

    ## Go deeper

    Use an independently defined target trajectory and compare the retained set with a pointwise-median reference. Vary the threshold and report selection stability. Check time alignment and missing-data coverage: `nanmean` can compare trials using different numbers of valid samples, which changes their comparability.

    ### Video lectures on the Internet

    - [MIT 18.06SC: projection matrices and least squares](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/projection-matrices-and-least-squares/) — review squared residuals and geometric distance, then distinguish that fixed-target calculation from the changing median reference here.

    ## References

    - [BMC `simila.py`](https://github.com/BMClab/BMC/blob/master/functions/simila.py) — algorithm, parameters and original attribution.

    - [NumPy `nanmean`](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html) — averaging with missing values.

    - [NumPy `nanmedian`](https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html) — the default centre and score normalization.
    """)
    return

if __name__ == "__main__":
    app.run()
