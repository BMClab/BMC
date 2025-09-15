import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Fourier transform

        > Marcos Duarte  
        > Laboratory of Biomechanics and Motor Control ([http://demotu.org/](http://demotu.org/))  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In continuation to the notebook about [Fourier series](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/FourierSeries.ipynb), the [Fourier transform](http://en.wikipedia.org/wiki/Fourier_transform) is a mathematical transformation to transform functions between time (or spatial) domain and frequency domain. The process of transforming from time to frequency domain is called Fourier analysis, the inverse is the Fourier synthesis.

        The Fourier transform of a continuous function$x(t)$is by definition:$X(f) = \int_{-\infty}^{\infty} x(t)\:\mathrm{e}^{-i2\pi ft} \:\mathrm{d}t$And the inverse Fourier transform is:$x(t) = \int_{-\infty}^{\infty} X(f)\:\mathrm{e}^{\:i2\pi tf} \:\mathrm{d}f$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Discrete Fourier transform

        For discrete data,$x$with length$N$, its Discrete Fourier Transform (DFT) is another data$X$, also with length$N$and elements:$X[k] = \sum_{n=0}^{N-1}  x[n] \mathrm{e}^{-i2\pi kn/N} \;,\quad 0 \leq k \leq N-1$The Inverse Discrete Fourier Transform (IDFT) inverts this operation and gives back the original data$x$:$x[n] = \frac{1}{N} \sum_{k=0}^{N-1}  X[k] \mathrm{e}^{i2\pi kn/N} \;,\quad 0 \leq n \leq N-1$The relationship between the DFT and the Fourier coefficients$a$and$b$in$x[n] = a_0 + \sum_{k=1}^{N-1} a[k]\cos\left(\frac{2\pi kt[n]}{Ndt}\right)+b[k]\sin\left(\frac{2\pi kt[n]}{Ndt}\right) \;,\quad 0 \leq n \leq N-1$is:$\begin{array}{l}
        a_0 = X[0]/N \\\
        \\\
        a[k] = \;\; \text{Real}(X[k+1])/N \\\
        \\\
        b[k] = -\text{Imag}(X[k+1])/N
        \end{array}$Where$x$is a length$N$discrete signal sampled at times$t$with spacing$dt$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Fast Fourier Transform (FFT)

        The [FFT](http://en.wikipedia.org/wiki/Fast_Fourier_transform) is a fast algorithm to compute the DFT. Let's see how to use the FFT algorithm from `scipy.fftpack`.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    import sys
    sys.path.insert(1, r'./../functions')
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A sine wave with amplitude of 2, frequency of 5 Hz, and phase of 45$^o$sampled at 100 Hz:
        """
    )
    return


@app.cell
def _(np, plt):
    A = 2 # amplitude
    freq = 5 # Hz
    phase = np.pi/4 # radians (45 degrees)
    fs = 100
    dt = 1 / fs
    time = np.arange(0, 500) / fs
    x = A * np.sin(2 * np.pi * freq * time + phase) + 1
    x = np.asarray(x)
    N = x.shape[0]
    t = np.arange(0, N) / fs

    fig, ax = plt.subplots(1, 1, figsize=(9, 3))
    ax.plot(time, x, linewidth=2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    fig.tight_layout()
    return N, dt, t, time, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Its FFT is simply:
        """
    )
    return


@app.cell
def _(N, dt, np, x):
    from scipy.fftpack import fft, fftfreq, fftshift
    X = fft(x, N)
    freqs = fftfreq(N, dt)
    amp = np.abs(X) / N
    phase_1 = -np.imag(X) / N * 180 / np.pi
    return amp, fftfreq, freqs, phase_1


@app.cell
def _(amp, freqs, phase_1, plt):
    (fig_1, ax_1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax_1[0].plot(freqs, amp)
    ax_1[1].plot(freqs, phase_1)
    ax_1[0].set_ylabel('Amplitude')
    ax_1[1].set_xlabel('Frequency [Hz]')
    ax_1[1].set_ylabel('Phase$[\\;^o]$')
    ax_1[0].set_ylim(-0.01, 1.1)
    ax_1[1].set_xlim(-50, 50)
    fig_1.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For real signals, the FFT values are the same for negative and positive frequencies and the phase is negated if the signal is odd and the same if the signal is even. Because that, we usually don't care about the negative frequencies and plot only the FFT for the positive frequencies:
        """
    )
    return


@app.cell
def _(N, amp, freqs, np, phase_1, plt):
    freqs2 = freqs[:int(np.floor(N / 2))]
    amp2 = amp[:int(np.floor(N / 2))]
    amp2[1:] = 2 * amp2[1:]
    phase2 = phase_1[:int(np.floor(N / 2))]
    (fig_2, ax_2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax_2[0].plot(freqs2, amp2)
    ax_2[1].plot(freqs2, phase2)
    ax_2[0].set_ylabel('Amplitude')
    ax_2[1].set_xlabel('Frequency [Hz]')
    ax_2[1].set_ylabel('Phase [$^o$]')
    ax_2[0].set_ylim(-0.01, 2.1)
    ax_2[1].set_xlim(-0.1, 50)
    ax_2[1].set_ylim(-0.2, 50)
    fig_2.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can get back the$x$data with the FFT synthesis:
        """
    )
    return


@app.cell
def _(N, dt, fft_1, np, t, x):
    from scipy.fftpack import fft
    X_1 = fft_1(x, N)
    a = np.zeros(N)
    b = np.zeros(N)
    a[0] = np.real(X_1[0]) / N
    a[1:N] = +np.real(X_1[1:N]) / N
    b[1:N] = -np.imag(X_1[1:N]) / N
    y = np.zeros((N, N))
    for k in np.arange(0, N):
        w = 2 * np.pi * k / (N * dt)
        y[:, k] = a[k] * np.cos(w * t) + b[k] * np.sin(w * t)
    xfft = np.sum(y, axis=1)
    return X_1, xfft


@app.cell
def _(plt, t, time, x, xfft):
    (fig_3, ax_3) = plt.subplots(1, 1, figsize=(9, 3))
    ax_3.plot(time, x, linewidth=2, label='Original data')
    ax_3.plot(t, xfft, 'r--', linewidth=2, label='FFT synthesis')
    ax_3.set_xlabel('Time [s]')
    ax_3.set_ylabel('Amplitude')
    ax_3.legend(framealpha=0.7)
    fig_3.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But employing the function `scipy.fftpack.fft` is simpler:
        """
    )
    return


@app.cell
def _(X_1, np, plt, t, time, x):
    from scipy.fftpack import ifft
    xfft2 = np.real(ifft(X_1))
    (fig_4, ax_4) = plt.subplots(1, 1, figsize=(9, 3))
    ax_4.plot(time, x, linewidth=2, label='Original data')
    ax_4.plot(t, xfft2, 'r--', linewidth=2, label='iFFT synthesis')
    ax_4.set_xlabel('Time [s]')
    ax_4.set_ylabel('Amplitude')
    ax_4.legend(framealpha=0.7)
    fig_4.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Another example:
        """
    )
    return


@app.cell
def _(np, plt):
    freq_1 = 100.0
    t_1 = np.arange(0, 5, 0.01)
    y_1 = 2 * np.sin(5 * np.pi * 2 * t_1) + np.sin(2 * np.pi * 20 * t_1) + np.random.randn(t_1.size)
    (fig_5, ax_5) = plt.subplots(1, 1, squeeze=True, figsize=(9, 3))
    ax_5.set_title('Temporal domain', fontsize=18)
    ax_5.plot(t_1, y_1, 'b', linewidth=2)
    ax_5.set_xlabel('Time [s]')
    ax_5.set_ylabel('y')
    ax_5.locator_params(axis='both', nbins=5)
    fig_5.tight_layout()
    return freq_1, y_1


@app.cell
def _(fft_1, fftfreq, freq_1, np, y_1):
    N_1 = y_1.size
    yfft = fft_1(y_1, N_1)
    yfft = 2 * np.abs(yfft) / N_1
    freqs_1 = fftfreq(N_1, 1.0 / freq_1)
    freqs_1 = freqs_1[:int(np.floor(N_1 / 2))]
    yfft = yfft[:int(np.floor(N_1 / 2))]
    return freqs_1, yfft


@app.cell
def _(freqs_1, plt, yfft):
    (fig_6, ax_6) = plt.subplots(1, 1, squeeze=True, figsize=(9, 3))
    ax_6.set_title('Frequency domain', fontsize=18)
    ax_6.plot(freqs_1, yfft, 'r', linewidth=2)
    ax_6.set_xlabel('Frequency [Hz]')
    ax_6.set_ylabel('FFT(y)')
    ax_6.locator_params(axis='both', nbins=5)
    fig_6.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### FFTW - the Fastest Fourier Transform in the West

        [FFTW](http://www.fftw.org/) is a free collection of fast C routines for computing the DFT. Indeed, FFTW is probably the fastest FFT library in the market and you should use it in case speed is a major concern. To use it in Python, you will need to install FFTW and the Python wrapper around FFTW, [pyfftw](https://pypi.python.org/pypi/pyFFTW).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Power spectral density 

        The function `psd.py` (code at the end of this text) estimates power spectral density characteristics using Welch's method. This function is just a wrap of the scipy.signal.welch function with estimation of some frequency characteristics and a plot. The `psd.py` returns power spectral density data, frequency percentiles of the power spectral density (for example, Fpcntile[50] gives the median power frequency in Hz); mean power frequency; maximum power frequency; total power, and plots power spectral density data.

        Let's exemplify the use of `psd.py`.
        """
    )
    return


@app.cell
def _():
    from psd import psd
    help(psd)
    return (psd,)


@app.cell
def _(np, plt):
    fs_1 = 10000.0
    N_2 = 100000.0
    amp_1 = 2 * np.sqrt(2)
    freq_2 = 1234.0
    noise_power = 0.001 * fs_1 / 2
    time_1 = np.arange(N_2) / fs_1
    x_1 = amp_1 * np.sin(2 * np.pi * freq_2 * time_1)
    x_1 = x_1 + np.random.normal(scale=np.sqrt(noise_power), size=time_1.shape)
    plt.figure(figsize=(10, 5))
    plt.plot(time_1, x_1, linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    return freq_2, x_1


@app.cell
def _(freq_2, psd, x_1):
    (fpcntile, mpf, fmax, Ptotal, f, P) = psd(x_1, fs=freq_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Periodogram
        """
    )
    return


@app.cell
def _(np):
    import scipy
    freq_3 = 100.0
    t_2 = np.arange(0, 5, 0.01)
    y_2 = 2 * np.sin(5 * np.pi * 2 * t_2) + np.sin(2 * np.pi * 20 * t_2) + np.random.randn(t_2.size)
    N_3 = y_2.shape[0]
    from scipy import signal, integrate
    (fp, Pp) = signal.periodogram(y_2, freq_3, window='boxcar', nfft=N_3)
    (fw, Pw) = signal.welch(y_2, freq_3, window='hanning', nperseg=N_3, noverlap=0, nfft=N_3)
    P_1 = np.abs(scipy.fftpack.fft(y_2 - np.mean(y_2), N_3))[:int(np.floor(N_3 / 2))] ** 2 / N_3 / freq_3
    P_1[1:-1] = 2 * P_1[1:-1]
    fs_2 = np.linspace(0, freq_3 / 2, len(P_1))
    return N_3, Pp, Pw, fp, freq_3, fw, integrate, signal, t_2, y_2


@app.cell
def _(Pp, Pw, fp, fw, plt, t_2, y_2):
    (fig_7, (ax1, ax2, ax3)) = plt.subplots(3, 1, squeeze=True, figsize=(12, 8))
    ax1.set_title('Temporal domain', fontsize=18)
    ax1.plot(t_2, y_2, 'b', linewidth=2)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('y [V]')
    ax1.locator_params(axis='both', nbins=5)
    ax2.set_title('Frequency domain', fontsize=18)
    ax2.plot(fp, Pp, 'r', linewidth=2)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('PSD(y)$[V^2/Hz]$')
    ax2.locator_params(axis='both', nbins=5)
    ax3.set_title('Frequency domain', fontsize=18)
    ax3.plot(fw, Pw, 'r', linewidth=2)
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('PSD(y)$[V^2/Hz]$')
    ax3.locator_params(axis='both', nbins=5)
    fig_7.tight_layout()
    return


@app.cell
def _(N_3, freq_3, integrate, np, signal, y_2):
    (F, P_2) = signal.welch(y_2, fs=freq_3, window='hanning', nperseg=N_3 / 2, noverlap=N_3 / 4, nfft=N_3 / 2)
    A_1 = integrate.cumtrapz(P_2, F)
    fm = np.trapz(F * P_2, F) / np.trapz(P_2, F)
    f50 = F[np.nonzero(A_1 >= 0.5 * A_1[-1])[0][0]]
    f95 = F[np.nonzero(A_1 >= 0.95 * A_1[-1])[0][0]]
    fmax_1 = F[np.argmax(P_2)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""$F_{mean} = \frac{ \sum_{i=1}^{N} F_i*P_i }{ \sum_{i=1}^{N} P_i }$"""
    )
    return


@app.cell
def _(freq_3, psd, y_2):
    (fp_1, mf, fmax_2, Ptot, F_1, P_3) = psd(y_2, fs=freq_3, scales='linear', units='V')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Short FFT
        """
    )
    return


@app.cell
def _(freq_3, plt, y_2):
    (fig_8, ax1_1) = plt.subplots(1, 1, figsize=(12, 6))
    (P_4, freqs_2, t_3, im) = plt.specgram(y_2, NFFT=64, Fs=freq_3, noverlap=32, cmap=plt.cm.gist_heat)
    ax1_1.set_title('Short FFT', fontsize=18)
    ax1_1.set_xlabel('Time [s]')
    ax1_1.set_ylabel('Frequency [Hz]')
    return


@app.cell
def _(np, plt, scipy_1):
    import scipy.signal
    t_4 = np.linspace(0, (2 ** 12 - 1) / 1000, 2 ** 12)
    c = scipy_1.signal.chirp(t_4, f0=100, f1=300, t1=t_4[-1], method='linear')
    (fig_9, ax1_2) = plt.subplots(1, 1, figsize=(12, 6))
    (P_5, freqs_3, t_4, im_1) = plt.specgram(c, NFFT=256, Fs=t_4.size / t_4[-1], noverlap=128, cmap=plt.cm.gist_heat)
    ax1_2.set_title('Short FFT', fontsize=18)
    ax1_2.set_xlabel('Time [s]')
    ax1_2.set_ylabel('Frequency [Hz]')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Function psd.py
        """
    )
    return


app._unparsable_cell(
    r"""
    # %load ./../functions/psd.py
    #!/usr/bin/env python

    \"\"\"Estimate power spectral density characteristcs using Welch's method.\"\"\"

    from __future__ import division, print_function

    __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version__ = 'tnorm.py v.1 2013/09/16'


    def psd(x, fs=1.0, window='hanning', nperseg=None, noverlap=None, nfft=None,
            detrend='constant', show=True, ax=None, scales='linear', xlim=None,
            units='V'):
        \"\"\"Estimate power spectral density characteristcs using Welch's method.

        This function is just a wrap of the scipy.signal.welch function with
        estimation of some frequency characteristcs and a plot. For completeness,
        most of the help from scipy.signal.welch function is pasted here.

        Welch's method [1]_ computes an estimate of the power spectral density
        by dividing the data into overlapping segments, computing a modified
        periodogram for each segment and averaging the periodograms.

        Parameters
        ----------
        x : array_like
            Time series of measurement values
        fs : float, optional
            Sampling frequency of the `x` time series in units of Hz. Defaults
            to 1.0.
        window : str or tuple or array_like, optional
            Desired window to use. See `get_window` for a list of windows and
            required parameters. If `window` is array_like it will be used
            directly as the window and its length will be used for nperseg.
            Defaults to 'hanning'.
        nperseg : int, optional
            Length of each segment.  Defaults to half of `x` length.
        noverlap: int, optional
            Number of points to overlap between segments. If None,
            ``noverlap = nperseg / 2``.  Defaults to None.
        nfft : int, optional
            Length of the FFT used, if a zero padded FFT is desired.  If None,
            the FFT length is `nperseg`. Defaults to None.
        detrend : str or function, optional
            Specifies how to detrend each segment. If `detrend` is a string,
            it is passed as the ``type`` argument to `detrend`. If it is a
            function, it takes a segment and returns a detrended segment.
            Defaults to 'constant'.
        show : bool, optional (default = False)
            True (1) plots data in a matplotlib figure.
            False (0) to not plot.
        ax : a matplotlib.axes.Axes instance (default = None)
        scales : str, optional
            Specifies the type of scale for the plot; default is 'linear' which
            makes a plot with linear scaling on both the x and y axis.
            Use 'semilogy' to plot with log scaling only on the y axis, 'semilogx'
            to plot with log scaling only on the x axis, and 'loglog' to plot with
            log scaling on both the x and y axis.
        xlim : float, optional
            Specifies the limit for the `x` axis; use as [xmin, xmax].
            The defaukt is `None` which sets xlim to [0, Fniquist].
        units : str, optional
            Specifies the units of `x`; default is 'V'.

        Returns
        -------
        Fpcntile : 1D array
            frequency percentiles of the power spectral density
            For example, Fpcntile[50] gives the median power frequency in Hz.
        mpf : float
            Mean power frequency in Hz.
        fmax : float
            Maximum power frequency in Hz.
        Ptotal : float
            Total power in `units` squared.
        f : 1D array
            Array of sample frequencies in Hz.
        P : 1D array
            Power spectral density or power spectrum of x.

        See Also
        --------
        scipy.signal.welch

        Notes
        -----
        An appropriate amount of overlap will depend on the choice of window
        and on your requirements.  For the default 'hanning' window an
        overlap of 50% is a reasonable trade off between accurately estimating
        the signal power, while not over counting any of the data.  Narrower
        windows may require a larger overlap.
        If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.

        References
        ----------
        .. [1] P. Welch, \"The use of the fast Fourier transform for the
               estimation of power spectra: A method based on time averaging
               over short, modified periodograms\", IEEE Trans. Audio
               Electroacoust. vol. 15, pp. 70-73, 1967.
        .. [2] M.S. Bartlett, \"Periodogram Analysis and Continuous Spectra\",
               Biometrika, vol. 37, pp. 1-16, 1950.

        Examples (also from scipy.signal.welch)
        --------
        #Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
        # 0.001 V**2/Hz of white noise sampled at 10 kHz and calculate the PSD:
        >>> from psd import psd
        >>> fs = 10e3
        >>> N = 1e5
        >>> amp = 2*np.sqrt(2)
        >>> freq = 1234.0
        >>> noise_power = 0.001 * fs / 2
        >>> time = np.arange(N) / fs
        >>> x = amp*np.sin(2*np.pi*freq*time)
        >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        >>> psd(x, fs=freq);
        \"\"\"


        if not nperseg:
            nperseg = np.ceil(len(x) / 2)
        f, P = signal.welch(x, fs, window, nperseg, noverlap, nfft, detrend)
        Area = integrate.cumtrapz(P, f, initial=0)
        Ptotal = Area[-1]
        mpf = integrate.trapz(f * P, f) / Ptotal  # mean power frequency
        fmax = f[np.argmax(P)]
        # frequency percentiles
        inds = [0]
        Area = 100 * Area / Ptotal  # + 10 * np.finfo(np.float).eps
        for i in range(1, 101):
            inds.append(np.argmax(Area[inds[-1]:] >= i) + inds[-1])
        fpcntile = f[inds]

        if show:
            _plot(x, fs, f, P, mpf, fmax, fpcntile, scales, xlim, units, ax)

        return fpcntile, mpf, fmax, Ptotal, f, P


    def _plot(x, fs, f, P, mpf, fmax, fpcntile, scales, xlim, units, ax):
        \"\"\"Plot results of the ellipse function, see its help.\"\"\"
        try:
        except ImportError:
            print('matplotlib is not available.')
        else:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            if scales.lower() == 'semilogy' or scales.lower() == 'loglog':
                ax.set_yscale('log')
            if scales.lower() == 'semilogx' or scales.lower() == 'loglog':
                ax.set_xscale('log')
            ax.plot(f, P, linewidth=2)
            ylim = ax.get_ylim()
            ax.plot([fmax, fmax], [np.max(P), np.max(P)], 'ro',
                     label='Fpeak  = %.2f' % fmax)
            ax.plot([fpcntile[50], fpcntile[50]], ylim, 'r', lw=1.5,
                     label='F50%%   = %.2f' % fpcntile[50])
            ax.plot([mpf, mpf], ylim, 'r--', lw=1.5,
                     label='Fmean = %.2f' % mpf)
            ax.plot([fpcntile[95], fpcntile[95]], ylim, 'r-.', lw=2,
                     label='F95%%   = %.2f' % fpcntile[95])
            leg = ax.legend(loc='best', numpoints=1, framealpha=.5,
                            title='Frequencies [Hz]')
            plt.setp(leg.get_title(), fontsize=12)
            ax.set_xlabel('Frequency [$Hz$]', fontsize=12)
            ax.set_ylabel('Magnitude [%s$^2/Hz$]' % units, fontsize=12)
            ax.set_title('Power spectral density', fontsize=12)
            if xlim:
                ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.tight_layout()
            plt.show()
    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
