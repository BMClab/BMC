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
        Number to limit the denominator in the calculation of a fraction of
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
        its index column sampled at `freq_new` Hz.
        
    """
    
    if freq_old is None:
        freq_old = np.mean(1/np.diff(y.index))
    if isinstance(y, pd.DataFrame):
        cols = y.columns
        index_name = y.index.name
        y = y.values
    else:
        cols = None

    fr = Fraction(freq_new/freq_old).limit_denominator(limit)
    y2 = np.empty((int(np.ceil(y.shape[0]*fr.numerator/fr.denominator)),
                   y.shape[1]))

    for c in range(y.shape[1]):
        if method == 'resample_poly':
            y2[:, c] = signal.resample_poly(y[:, c], fr.numerator, fr.denominator)
    
    if cols is not None:
        y2 = pd.DataFrame(data=y2, columns=cols)
        y2.index = y2.index/freq_new
        y2.index.name = index_name   
    
    return y2
