"""Resample data using resample_poly scipy function.

"""

__author__ = "Marcos Duarte, https://github.com/BMClab/"
__version__ = "1.0.0"
__license__ = "MIT"

import numpy as np
import pandas as pd
from scipy import signal
from fractions import Fraction


def resample(y, freq_new, freq_old=None, limit=1000, method='resample_poly'):
    """
    Resample data using signal.resample_poly scipy function.
    
    Parameters
    ----------
    y : numpy array or pandas dataframe column-oriented
        All columns of `y` will be resampled.
    freq_new : number
        Desired new sampling frequency of `y2` (in Hz).
    freq_old : number, optional (default = None)
        Original sampling frequency of `y` (in Hz). If no sampling frequency is
        informed, `y` is assumed to be a pandas dataframe and its sampling
        frequency will be calculated from its index column.
    limit : integer, optional (default = 1000)
        Positive integer limiting the denominator in the calculation of a fraction of
        integers to represent the ratio between up- and down- sampling
        frequencies (freq_new/freq_old) in the resample (for an irrational
        ratio, this number might affect the accuracy of the resampling but
        it will be faster).
    method : string, optional (default = 'resample_poly')
        Method used for resampling data.
        The resample_poly scipy function is the only one implemented so far.
        
    Returns
    -------
    y2 : numpy array or pandas dataframe
        Resampled data. Same type (numpy array or pandas dataframe) and same
        number of columns as `y`. If `y` is a pandas dataframe, `y2` will have
        its time origin preserved and its index sampled at the effective rational
        approximation to `freq_new` Hz. Boundary samples use zero-padding.
        Upsampling creates interpolated samples, not new measured information.
        
    """
    
    if method != "resample_poly":
        raise ValueError("Only method='resample_poly' is supported")
    is_frame = isinstance(y, pd.DataFrame)
    origin = float(y.index[0]) if is_frame and len(y) else 0.0
    if freq_old is None:
        if not is_frame or len(y) < 2:
            raise ValueError("Supply freq_old or a DataFrame with a numeric time index")
        intervals = np.diff(np.asarray(y.index, dtype=float))
        if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=1e-5, atol=1e-10):
            raise ValueError("Time index must be increasing and uniformly sampled")
        freq_old = 1 / intervals.mean()
    if not np.isfinite(freq_old) or not np.isfinite(freq_new) or min(freq_old, freq_new) <= 0:
        raise ValueError("Sampling frequencies must be finite and positive")
    data = np.asarray(y)
    if data.ndim not in (1, 2) or not len(data) or not np.isfinite(data).all():
        raise ValueError("Data must be a nonempty finite one- or two-dimensional array")
    fr = Fraction(float(freq_new / freq_old)).limit_denominator(limit)
    if fr.numerator == 0:
        raise ValueError("Increase limit to represent the requested frequency ratio")
    actual_rate = freq_old * fr.numerator / fr.denominator
    y2 = signal.resample_poly(data, fr.numerator, fr.denominator, axis=0)
    if is_frame:
        y2 = pd.DataFrame(y2, columns=y.columns, index=origin + np.arange(len(y2)) / actual_rate)
        y2.index.name = y.index.name
    return y2
