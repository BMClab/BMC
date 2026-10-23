#!/usr/bin/env python

"""Replace missing data in different ways."""

import numpy as np


__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'rep_missing.py v.1.0.0 2019/03/17'


def rep_missing(x, value=np.nan, new_value='interp', max_alert=100):
    """Replace missing data in different ways.

    Parameters
    ----------
    x : 1D numpy array_like
        data
    value : number, optional. Default = np.nan.
        Scalar value in x marking missing data. The input is never modified.
    new_value : number or string, optional. Default = 'interp'.
        Value or string for the method to use for replacing missing data:
        'delete': delete missing data.
        new_value: replace missing data with new_value.
        'mean': replace missing data with the mean of the rest of the data.
        'median': replace missing data with the median of the rest of the data.
        'interp': replace missing data by linear interpolating over them.
    max_alert : number, optional. Default = 100.
        Minimal number of sequential data for a message to be printed with
        information about the continuous missing data.
        Set to 0 to not print any message.
        
    This function can handle NaNs separately from missing data (in case they
    are different), but the 'interp' method will interpolate on both the
    missing data and NaNs. Interpolation uses sample indices, assumes uniform sampling,
    and extends the nearest observed value at missing endpoints.
    
        
    Returns
    -------
    y : 1D numpy array
        1D numpy array similar to x but with missing data replaced according
        to value or method specified with the parameter new_value.
            
    References
    ----------
    .. [1] https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/rep_missing.py

    Examples
    --------
    >>> x = [1, 1, 2, 0, 4, 1, 1, 0, 1, 0, 2]

    """
    
    from numbers import Number

    y = np.array(x, dtype=complex if np.iscomplexobj(x) else float, copy=True)
    if y.ndim != 1:
        raise ValueError("x must be one-dimensional")
    idx = np.isnan(y) if np.isnan(value) else y == value
    if new_value == "interp":
        idx = idx | np.isnan(y)
    if isinstance(new_value, str) and new_value not in {"delete", "mean", "median", "interp"}:
        raise ValueError("new_value must be numeric, delete, mean, median, or interp")
    if max_alert and idx.any():
        edges = np.diff(np.r_[False, idx, False].astype(int))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        if np.any(lengths >= max_alert):
            import warnings
            warnings.warn("Long missing-data sequence; assess whether replacement is justified.", stacklevel=2)
    if isinstance(new_value, Number):
        y[idx] = new_value
    elif new_value == "delete":
        y = y[~idx]
    elif idx.any():
        valid = ~idx & np.isfinite(y)
        if not valid.any():
            raise ValueError("Replacement requires at least one finite observed value")
        if new_value == "mean":
            y[idx] = np.mean(y[valid])
        elif new_value == "median":
            y[idx] = np.median(y[valid])
        else:
            missing = idx | np.isnan(y)
            samples = np.arange(y.size)
            y[missing] = np.interp(samples[missing], samples[valid], y[valid])
    return y
