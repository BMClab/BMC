"""Coefficients of critically damped or Butterworth digital lowpass filter."""

import numpy as np
import warnings

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = "1.0.0"
__license__ = "MIT"



def critic_damp(fcut, freq, npass=2, filt='critic', fcorr=True):
    """Coefficients of critically damped or Butterworthdigital lowpass filter.
    
    A problem with a lowpass Butterworth filter is that it tends to overshoot
    or undershoot data with rapid changes (see for example, Winter (2009),
    Robertson et at. (2013), and Robertson & Dowling (2003)).  
    The Butterworth filter behaves as an underdamped second-order system and a
    critically damped filter doesn't have this overshoot/undershoot
    characteristic.

    This function calculates the coefficients (the b's and a's) for an IIR
    critically damped digital filter and corrects the cutoff frequency for
    the number of passes of the filter. The calculation of these coefficients
    is very similar to the calculation for the Butterworth filter and this
    function can also calculate the Butterworth coefficients if this option
    is chosen.
    
    Parameters
    ----------
    fcut : number
        desired cutoff frequency for the lowpass digital filter (Hz).
    freq : number
        sampling frequency (Hz).
    npass : number, optional (default = 2)
        number of passes the filter will be applied.
        choose 2 for a second order zero phase lag filter
    filt : string ('critic', 'butter'), optional (default = 'critic')
        'critic' to calculate coefficients for critically damped lowpass filter
        'butter' to calculate coefficients for Butterworth lowpass filter
    fcorr : bool, optional (default = True)
        correct (True) or not the cutoff frequency for the number of passes.

    Returns
    -------
    b : 1D array
        b coefficients for the filter
    a : 1D array
        a coefficients for the filter
    fc : number
        actual cutoff frequency considering the number of passes
        
    Notes
    -----
    See this Jupyter notebook [1]_
    See documentation for the Scipy butter function [2]_

    References
    ----------
    .. [1] http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
    .. [2] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html

    Examples
    --------
    >>> from critic_damp import critic_damp
    >>> print('Critically damped filter')
    >>> b_cd, a_cd, fc_cd = critic_damp(fcut=10, freq=100, npass=2, fcorr=True, filt='critic')
    >>> print('b:', b_cd, '\na:', a_cd, '\nActual Fc:', fc_cd)
    >>> print('Butterworth filter')
    >>> b_bw, a_bw, fc_bw = critic_damp(fcut=10, freq=100, npass=2, fcorr=True, filt='butter')
    >>> print('b:', b_bw, '\na:', a_bw, '\nActual Fc:', fc_bw)
    >>> # illustrate the filter in action
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import signal
    >>> y = np.hstack((np.zeros(20), np.ones(20)))
    >>> t = np.linspace(0, 0.39, 40) - .19
    >>> y_cd = signal.filtfilt(b_cd, a_cd, y)
    >>> y_bw = signal.filtfilt(b_bw, a_bw, y)
    >>> fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    >>> ax.plot(t,  y, 'k', linewidth=2, label = 'raw data')
    >>> ax.plot(t,  y_cd, 'r', linewidth=2, label = 'Critically damped')
    >>> ax.plot(t,  y_bw, 'b', linewidth=2, label = 'Butterworth')     
    >>> ax.legend()
    >>> ax.set_xlabel('Time (s)')
    >>> ax.set_ylabel('Amplitude')
    >>> ax.set_title('Freq = 100 Hz, Fc = 10 Hz, 2nd order and zero-phase shift filters')
    >>> plt.show()
    
    """

    if fcut > freq/2:
        warnings.warn('Cutoff frequency can not be greater than Nyquist frequency.')

    # cutoff frequency correction for number of passes
    if filt.lower() == 'critic':
        if fcorr:
            corr = 1/np.power(2**(1/(2*npass))-1, 0.5)
    elif filt.lower() == 'butter':
        if fcorr:
            corr = 1/np.power(2**(1/npass)-1, 0.25)
    else:
        warnings.warn('Invalid option for paraneter filt:', filt)

    # corrected cutoff frequency
    if fcorr:
        fc = fcut*corr
        if fc > (freq/2):
            text = 'Warning: corrected cutoff frequency ({} Hz) is greater'+\
            ' than Nyquist frequency ({} Hz). Using the uncorrected cutoff'+\
            ' frequency ({} Hz).'
            print(text.format(fc, freq/2, fcut))
            fc = fcut
    else:
        fc = fcut

    # corrected angular cutoff frequency
    wc = np.tan(np.pi*fc/freq)
    # lowpass coefficients
    k1 = np.sqrt(2)*wc if filt.lower() == 'butter' else 2*wc
    k2 = wc*wc
    a0 = k2/(1+k1+k2)
    a1 = 2*a0
    a2 = k2/(1+k1+k2)
    b1 = 2*a0*(1/k2-1)
    b2 = 1-(a0+a1+a2+b1)
    # transform parameters to be consistent with SciPy butter output
    b = np.array([a0, a1, a2])
    a = np.array([1, -b1, -b2])
    
    return b, a, fc