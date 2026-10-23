"""Coefficients of critically damped or Butterworth digital low-pass filters."""

__author__ = "Marcos Duarte, https://github.com/BMClab/BMC"
__version__ = "1.1.0"
__license__ = "MIT"


def critic_damp(fcut, freq, npass=2, fcorr=True, filt="critic"):
    """Coefficients of a critically damped or Butterworth digital low-pass filter.

    A problem with a low-pass Butterworth filter is that it tends to overshoot
    or undershoot data with rapid changes (see for example, Winter (2009),
    Robertson et al. (2013), and Robertson & Dowling (2003)).
    The analog critically damped response avoids step overshoot. Check the
    actual digital response, particularly for cutoffs near Nyquist.

    Parameters
    ----------
    fcut : number
        desired cutoff frequency for the low-pass digital filter (Hz).
    freq : number
        sampling frequency (Hz).
    npass : number, optional (default = 2)
        number of passes the filter will be applied.
        choose 2 for two passes of the second-order section (fourth-order roll-off).
    fcorr : bool, optional (default = True)
        correct (True) or not the cutoff frequency for the number of passes.
    filt : string ('critic', 'butter'), optional (default = 'critic')
        'critic' to calculate coefficients for a critically damped filter,
        'butter' to calculate coefficients for a Butterworth filter.

    Returns
    -------
    b : 1D array
        b coefficients for the filter
    a : 1D array
        a coefficients for the filter
    fc : number
        digital design-frequency parameter after exact prewarped correction

    References
    ----------
    Robertson DG, Dowling JJ (2003) Design and responses of Butterworth and
    critically damped digital filters. J Electromyogr Kinesiol 13, 569-573.
    """
    import numpy as np

    filt = filt.lower()
    if filt not in ("critic", "butter"):
        raise ValueError(f"Invalid option for parameter filt: {filt!r}")
    if not np.isfinite(freq) or freq <= 0 or not 0 < fcut < freq / 2:
        raise ValueError("Require a positive sampling rate and 0 < fcut < freq/2.")
    if not np.isfinite(npass) or npass < 1 or int(npass) != npass:
        raise ValueError("npass must be a positive integer.")

    # Correct the prewarped frequency, not its digital frequency in hertz.
    corr = 1.0
    if fcorr:
        if filt == "critic":
            corr = 1 / np.sqrt(2 ** (1 / (2 * npass)) - 1)
        else:
            corr = 1 / (2 ** (1 / npass) - 1) ** 0.25
    wc = np.tan(np.pi * fcut / freq) * corr
    fc = freq / np.pi * np.arctan(wc)
    # low-pass coefficients
    k1 = np.sqrt(2) * wc if filt == "butter" else 2 * wc
    k2 = wc * wc
    a0 = k2 / (1 + k1 + k2)
    a1 = 2 * a0
    a2 = k2 / (1 + k1 + k2)
    b1 = 2 * a0 * (1 / k2 - 1)
    b2 = 1 - (a0 + a1 + a2 + b1)
    # transform parameters to be consistent with SciPy butter output
    b = np.array([a0, a1, a2])
    a = np.array([1, -b1, -b2])

    return b, a, fc
