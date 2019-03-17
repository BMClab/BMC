#!/usr/bin/env python

"""Replace missing data in different ways."""

import numpy as np
from detect_seq import detect_seq
try:
    from tnorm import tnorm
except:
    print('Function tnorm.py not found. Method "interp" not available.')

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'rep_missing.py v.1.0.0 2019/03/17'


def rep_missing(x, value=np.nan, new_value='interp', max_alert=100):
    """Replace missing data in different ways.

    Parameters
    ----------
    x : 1D numpy array_like
        data
    value : number, optional. Default = np.nan.
        Value to be found in x marking missing data.
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
    missing data and NaNs.    
    
        
    Returns
    -------
    y : 1D numpy array
        1D numpy array similar to x but with missing data replaced according
        to value or method specified with the parameter new_value.
            
    References
    ----------
    .. [1] http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/rep_missing.ipynb

    Examples
    --------
    >>> x = [1, 1, 2, 0, 4, 1, 1, 0, 1, 0, 2]

    """
    
    y = np.asarray(x)

    idx = detect_seq(y, value, index=False, max_alert=max_alert)

    if idx.any():
        if new_value == 'delete':
            y = y[~idx]
        elif np.isreal(new_value) or np.iscomplex(new_value):
            y[idx] = new_value
        elif new_value == 'mean':
            y[idx] = np.nanmean(y[~idx])
        elif new_value == 'median':
            y[idx] = np.nanmedian(y[~ĩdx]) 
        elif new_value == 'interp':
            y = y.astype(float)
            y[idx] = np.nan
            y, t, indie = tnorm(y, step=0, k=1, smooth=0, nan_at_ext='replace')
            
    return y
