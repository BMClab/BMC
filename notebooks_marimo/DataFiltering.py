import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a href="https://colab.research.google.com/github/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Data filtering in signal processing

        > Marcos Duarte, Renato Naville Watanabe  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here will see an introduction to data filtering and the most basic filters typically used in signal processing of biomechanical data.  
        You should be familiar with the [basic properties of signals](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/SignalBasicProperties.ipynb) before proceeding.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Setup
        """
    )
    return


@app.cell
def _():
    # Import the necessary libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display
    # scipy and numpy have too many future warnings
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return np, pd, plt, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Filter and smoothing

        In data acquisition with an instrument, it's common that the noise has higher frequencies and lower amplitudes than the desired signal. To remove this noise from the signal, a procedure known as filtering or smoothing is employed in the signal processing.  
        <a href="http://en.wikipedia.org/wiki/Filter_(signal_processing)">Filtering</a> is a process to attenuate from a signal some unwanted component or feature. A filter usually removes certain frequency components from the data according to its frequency response.  
        [Frequency response](http://en.wikipedia.org/wiki/Frequency_response) is the quantitative measure of the output spectrum of a system or device in response to a stimulus, and is used to characterize the dynamics of the system.  
        [Smoothing](http://en.wikipedia.org/wiki/Smoothing) is the process of removal of local (at short scale) fluctuations in the data while preserving a more global pattern in the data (such local variations could be noise or just a short scale phenomenon that is not interesting). A filter with a low-pass frequency response performs smoothing.  
        With respect to the filter implementation, it can be classified as [analog filter](http://en.wikipedia.org/wiki/Passive_analogue_filter_development) or [digital filter](http://en.wikipedia.org/wiki/Digital_filter).  
        An analog filter is an electronic circuit that performs filtering of the input electrical signal (analog data) and outputs a filtered electrical signal (analog data). A simple analog filter can be implemented with a electronic circuit with a resistor and a capacitor. A digital filter, is  a system that implement the filtering of a digital data (time-discrete data).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: the moving-average filter

        An example of a low-pass (smoothing) filter is the moving average, which is performed taking the arithmetic mean of subsequences of$m$terms of the data. For instance, the moving averages with window sizes (m) equal to 2 and 3 are:$\begin{array}{}
        &y_{MA(2)} = \frac{1}{2}[x_1+x_2,\; x_2+x_3,\; \cdots,\; x_{n-1}+x_n] \\
        &y_{MA(3)} = \frac{1}{3}[x_1+x_2+x_3,\; x_2+x_3+x_4,\; \cdots,\; x_{n-2}+x_{n-1}+x_n]
        \end{array}$Which has the general formula:$y[i] = \sum_{j=0}^{m-1} x[i+j] \quad for \quad i=1, \; \dots, \; n-m+1$Where$n$is the number (length) of data.

        Let's implement a simple version of the moving average filter.  
        First, let's import the necessary Python libraries and configure the environment:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A naive moving-average function definition:
        """
    )
    return


@app.cell
def _(np):
    def moving_average(x, window):
        """Moving average of 'x' with window size 'window'."""
        y = np.empty(len(x)-window+1)
        for i in range(len(y)):
            y[i] = np.sum(x[i:i+window])/window
        return y
    return (moving_average,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's generate some data to test this function:
        """
    )
    return


@app.cell
def _(moving_average, np, plt):
    signal = np.zeros(300)
    signal[100:200] = signal[100:200] + 1
    noise = np.random.randn(300) / 10
    x = signal + noise
    window = 11
    y = moving_average(x, window)
    (fig, ax) = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, 'b.-', linewidth=1, label='raw data')
    ax.plot(y, 'r.-', linewidth=2, label='moving average')
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Later we will look on better ways to calculate the moving average.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Digital filters

        In signal processing, a digital filter is a system that performs mathematical operations on a signal to modify certain aspects of that signal. A digital filter (in fact, a causal, linear time-invariant (LTI) digital filter) can be seen as the implementation of the following difference equation in the time domain:$\begin{array}{}
        y_n &= \quad b_0x_n + \; b_1x_{n-1} + \cdots + b_Mx_{n-M} - \; a_1y_{n-1} - \cdots - a_Ny_{n-N} \\
        & = \quad \sum_{k=0}^M b_kx_{n-k} - \sum_{k=1}^N a_ky_{n-k}
        \end{array}$Where the output$y$is the filtered version of the input$x$,$a_k$and$b_k$are the filter coefficients (real values), and the order of the filter is the larger of N or M.

        This general equation is for a recursive filter where the filtered signal y is calculated based on current and previous values of$x$and on previous values of$y$(the own output values, because of this it is said to be a system with feedback). A filter that does not re-use its outputs as an input (and it is said to be a system with only feedforward) is called nonrecursive filter (the$a$coefficients of the equation are zero). Recursive and nonrecursive filters are also known as infinite impulse response (IIR) and finite impulse response (FIR) filters, respectively.  

        A filter with only the terms based on the previous values of$y$is also known as an autoregressive (AR) filter. A filter with only the terms based on the current and previous values of$x$is also known as an moving-average (MA) filter. The filter with all terms is also known as an autoregressive moving-average (ARMA) filter. The moving-average filter can be implemented by making$n$$b$coefficients each equals to$1/n$and the$a$coefficients equal to zero in the difference equation.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Transfer function

        Another form to characterize a digital filter is by its [transfer function](http://en.wikipedia.org/wiki/Transfer_function). In simple terms, a transfer function is the ratio in the frequency domain between the input and output signals of a filter.  
        For continuous-time input signal$x(t)$and output$y(t)$, the transfer function$H(s)$is given by the ratio between the [Laplace transforms](http://en.wikipedia.org/wiki/Laplace_transform) of input$x(t)$and output$y(t)$:$H(s) = \frac{Y(s)}{X(s)}$Where$s = \sigma + j\omega$;$j$is the imaginary unit and$\omega$is the angular frequency,$2\pi f$.  

        In the steady-state response case, we can consider$\sigma=0$and the Laplace transforms with complex arguments reduce to the [Fourier transforms](http://en.wikipedia.org/wiki/Fourier_transform) with real argument$\omega$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For discrete-time input signal$x(t)$and output$y(t)$, the transfer function$H(z)$will be given by the ratio between the [z-transforms](http://en.wikipedia.org/wiki/Z-transform) of input$x(t)$and output$y(t)$, and the formalism is similar.

        The transfer function of a digital filter (in fact for a linear, time-invariant, and causal filter), obtained by taking the z-transform of the difference equation shown earlier, is given by:$H(z) = \frac{Y(z)}{X(z)} = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2} + \cdots + b_N z^{-N}}{1 + a_1 z^{-1} + a_2 z^{-2} + \cdots + a_M z^{-M}}$$H(z) = \frac{\sum_{k=0}^M b_kz^{-k}}{1 + \sum_{k=1}^N a_kz^{-k}}$And the order of the filter is the larger of N or M.  

        Similar to the difference equation, this transfer function is for a recursive (IIR) filter. If the$a$coefficients are zero, the denominator is equal to one, and the filter becomes nonrecursive (FIR).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The Fourier transform

        The [Fourier transform](http://en.wikipedia.org/wiki/Fourier_transform) is a mathematical operation to transform a signal which is function of time,$g(t)$, into a signal which is function of frequency,$G(f)$, and it is defined by:  
        <br />$\mathcal{F}[g(t)] = G(f) = \int_{-\infty}^{\infty} g(t) e^{-j 2\pi f t} dt$Its inverse operation is:  
        <br />$\mathcal{F}^{-1}[G(f)] = g(t) = \int_{-\infty}^{\infty} G(f) e^{j 2\pi f t} df$The function$G(f)$is the representation in the frequency domain of the time-domain signal,$g(t)$, and vice-versa. The functions$g(t)$and$G(f)$are referred to as a Fourier integral pair, or Fourier transform pair, or simply the Fourier pair. [See this text for an introduction to Fourier transform](http://www.thefouriertransform.com/transform/fourier.php).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Types of filters

        In relation to the frequencies that are not removed from the data (and a boundary is specified by the critical or cutoff frequency), a filter can be a low-pass, high-pass, band-pass, and band-stop. The frequency response of such filters is illustrated in the next figure.

        <div class='center-align'><figure><img src="http://upload.wikimedia.org/wikipedia/en/thumb/e/ec/Bandform_template.svg/640px-Bandform_template.svg.png" alt="Filters" /><figcaption><i>Frequency response of filters (<a href="http://en.wikipedia.org/wiki/Filter_(signal_processing)" target="_blank">from Wikipedia</a>).</i></figcaption></figure></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The critical or cutoff frequency for a filter is defined as the frequency where the power (the amplitude squared) of the filtered signal is half of the power of the input signal (or the output amplitude is 0.707 of the input amplitude).  
        For instance, if a low-pass filter has a cutoff frequency of 10 Hz, it means that at 10 Hz the power of the filtered signal is 50% of the power of the original signal (and the output amplitude will be about 71% of the input amplitude).

        The gain of a filter (the ratio between the output and input powers) is usually expressed in the decibel (dB) unit.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Decibel (dB)

        The <a href="http://en.wikipedia.org/wiki/Decibel" target="_blank">decibel (dB)</a> is a logarithmic unit used to express the ratio between two values.    
        In the case of the filter gain measured in the decibel unit:$Gain=10\,log\left(\frac{A_{out}^2}{A_{in}^2}\right)=20\,log\left(\frac{A_{out}}{A_{in}}\right)$Where$A_{out}$and$A_{in}$are respectively the amplitudes of the output (filtered) and input (raw) signals.

        For instance, the critical or cutoff frequency for a filter, the frequency where the power (the amplitude squared) of the filtered signal is half of the power of the input signal, is given in decibel as:$10\,log\left(0.5\right) \approx -3 dB$If the power of the filtered signal is twice the power of the input signal, because of the logarithm, the gain in decibel is$10\,log\left(2\right) \approx 3 dB$.  
        If the output power is attenuated by ten times, the gain is$10\,log\left(0.1\right) \approx -10 dB$, but if the output amplitude is attenuated by ten times, the gain is$20\,log\left(0.1\right) \approx -20 dB$, and if the output amplitude is amplified by ten times, the gain is$20 dB$.  
        For each 10-fold variation in the amplitude ratio, there is an increase (or decrease) of$20 dB$.

        The decibel unit is useful to represent large variations in a measurement, for example,$-120 dB$represents an attenuation of 1,000,000 times.  
        A decibel is one tenth of a bel, a unit named in honor of <a href="http://en.wikipedia.org/wiki/Alexander_Graham_Bell" target="_blank">Alexander Graham Bell</a>.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Butterworth filter

        A common filter employed in biomechanics and motor control fields is the [Butterworth filter](http://en.wikipedia.org/wiki/Butterworth_filter). This filter is used because its simple design,  it has a more flat frequency response and linear phase response in the pass and stop bands, and it is simple to use.    
        The Butterworth filter is a recursive filter (IIR) and both$a$and$b$filter coefficients are used in its implementation.   
        Let's implement the Butterworth filter. We will use the function `butter` to calculate the filter coefficients:  
        ```python
        butter(N, Wn, btype='low', analog=False, output='ba')
        ```
        Where `N` is the order of the filter, `Wn` is the cutoff frequency specified as a fraction of the [Nyquist frequency](http://en.wikipedia.org/wiki/Nyquist_frequency) (half of the sampling frequency), and `btype` is the type of filter (it can be any of {'lowpass', 'highpass', 'bandpass', 'bandstop'}, the default is 'lowpass'). See the help of `butter` for more details. The filtering itself is performed with the function `lfilter`:   
        ```python
        lfilter(b, a, x, axis=-1, zi=None)
        ```
        Where `b` and `a` are the Butterworth coefficients calculated with the function `butter` and `x` is the variable with the data to be filtered.
        """
    )
    return


@app.cell
def _(np, plt, signal_1):
    from scipy import signal
    freq = 100
    t = np.arange(0, 1, 0.01)
    w = 2 * np.pi * 1
    y_1 = np.sin(w * t) + 0.1 * np.sin(10 * w * t)
    (b, a) = signal_1.butter(2, 5 / (freq / 2), btype='low')
    y2 = signal_1.lfilter(b, a, y_1)
    (fig_1, ax1) = plt.subplots(1, 1, figsize=(9, 4))
    ax1.plot(t, y_1, 'r.-', linewidth=2, label='raw data')
    ax1.plot(t, y2, 'b.-', linewidth=2, label='filter @ 5 Hz')
    ax1.legend(frameon=False, fontsize=14)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    plt.show()
    return (freq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The plot above shows that the Butterworth filter introduces a phase (a delay or lag in time) between the raw and the filtered signals. We will see how to account for that later.  

        Let's look at the values of the `b` and `a` Butterworth filter coefficients for different orders and see a characteristic of them; from the general difference equation shown earlier, it follows that the sum of the `b` coefficients minus the sum of the `a` coefficients (excluding the first coefficient of `a`) is one:
        """
    )
    return


@app.cell
def _(np, signal_1):
    print('Low-pass Butterworth filter coefficients')
    (b_1, a_1) = signal_1.butter(1, 0.1, btype='low')
    print('Order 1:', '\nb:', b_1, '\na:', a_1, '\nsum(b)-sum(a):', np.sum(b_1) - np.sum(a_1[1:]))
    (b_1, a_1) = signal_1.butter(2, 0.1, btype='low')
    print('Order 2:', '\nb:', b_1, '\na:', a_1, '\nsum(b)-sum(a):', np.sum(b_1) - np.sum(a_1[1:]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Bode plot

        How much the amplitude of the filtered signal is attenuated in relation to the amplitude of the raw signal (gain or magnitude) as a function of frequency is given in the frequency response plot. The plots of the frequency and phase responses (the [bode plot](http://en.wikipedia.org/wiki/Bode_plot)) of this filter implementation (Butterworth, lowpass at 5 Hz, second-order) is shown below:
        """
    )
    return


@app.cell
def _(freq, np, plt, signal_1):
    (b_2, a_2) = signal_1.butter(2, 5 / (freq / 2), btype='low')
    (w_1, h) = signal_1.freqz(b_2, a_2)
    angles = np.rad2deg(np.unwrap(np.angle(h)))
    w_1 = w_1 / np.pi * freq / 2
    h = 20 * np.log10(np.absolute(h))
    (fig_2, (ax1_1, ax2)) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax1_1.plot(w_1, h, linewidth=2)
    ax1_1.set_ylim(-80, 1)
    ax1_1.set_title('Frequency response')
    ax1_1.set_ylabel('Magnitude [dB]')
    ax1_1.plot(5, -3.01, 'ro')
    ax11 = plt.axes([0.17, 0.59, 0.2, 0.2])
    ax11.plot(w_1, h, linewidth=2)
    ax11.plot(5, -3.01, 'ro')
    ax11.set_ylim([-6, 0.5])
    ax11.set_xlim([0, 10])
    ax2.plot(w_1, angles, linewidth=2)
    ax2.set_title('Phase response')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [degrees]')
    ax2.plot(5, -90, 'ro')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The inset plot in the former figure shows that at the cutoff frequency (5 Hz), the power of the filtered signal is indeed attenuated by 3 dB.  

        The phase-response plot shows that at the cutoff frequency, the Butterworth filter presents about 90 degrees of phase between the raw and filtered signals. A 5 Hz signal has a period of 0.2 s and 90 degrees of phase corresponds to 0.05 s of lag. Looking at the plot with the raw and filtered signals employing or not the phase correction, we can see that the delay is indeed about 0.05 s.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Order of a filter

        The order of a filter is related to the inclination of the 'wall' in the frequency response plot that attenuates or not the input signal at the vicinity of the cutoff frequency. A vertical wall exactly at the cutoff frequency would be ideal but this is impossible to implement.   
        A Butterworth filter of first order attenuates 6 dB of the power of the signal each doubling of the frequency (per octave) or, which is the same, attenuates 20 dB each time the frequency varies by an order of 10 (per decade). In more technical terms, one simply says that a first-order filter rolls off -6 dB per octave or that rolls off -20 dB per decade. A second-order filter rolls off -12 dB per octave (-40 dB per decade), and so on, as shown in the next figure.  
        """
    )
    return


@app.cell
def _(np, plt, signal_1):
    """Plot of frequency response of the Butterworth filter with different orders."""
    __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version__ = 'butterworth_plot.py v.1 2023/10/23'

    def butterworth_plot(fig=None, ax=None):
        """
        Plot of frequency response of the Butterworth filter with different orders.
        """
        if fig is None:
            (fig, ax) = plt.subplots(1, 2, figsize=(10, 4))
        (b1, a1) = signal_1.butter(1, 10, 'low', analog=True)
        (w, h1) = signal_1.freqs(b1, a1)
        ang1 = np.rad2deg(np.unwrap(np.angle(h1)))
        h1 = 20 * np.log10(abs(h1))
        (b2, a2) = signal_1.butter(2, 10, 'low', analog=True)
        (w, h2) = signal_1.freqs(b2, a2)
        ang2 = np.rad2deg(np.unwrap(np.angle(h2)))
        h2 = 20 * np.log10(abs(h2))
        (b4, a4) = signal_1.butter(4, 10, 'low', analog=True)
        (w, h4) = signal_1.freqs(b4, a4)
        ang4 = np.rad2deg(np.unwrap(np.angle(h4)))
        h4 = 20 * np.log10(abs(h4))
        (b6, a6) = signal_1.butter(6, 10, 'low', analog=True)
        (w, h6) = signal_1.freqs(b6, a6)
        ang6 = np.rad2deg(np.unwrap(np.angle(h6)))
        h6 = 20 * np.log10(abs(h6))
        w = w / 10
        ax[0].plot(w, h1, 'b', w, h2, 'r', w, h4, 'g', w, h6, 'y', linewidth=2)
        ax[0].axvline(1, color='black')
        ax[0].scatter(1, -3, marker='s', edgecolor='0', facecolor='1', s=400)
        ax[0].set_xscale('log')
        fig.suptitle('Bode plot for low-pass Butterworth filter with different orders', fontsize=16, y=1.05)
        ax[0].set_xlabel('Frequency / Critical frequency', fontsize=14)
        ax[0].set_ylabel('Magnitude [dB]', fontsize=14)
        ax[0].set_xlim(0.1, 10)
        ax[0].set_ylim(-120, 10)
        ax[0].grid(which='both', axis='both')
        ax[1].plot(w, ang1, 'b', w, ang2, 'r', w, ang4, 'g', w, ang6, 'y', linewidth=2)
        ax[1].axvline(1, color='black')
        ax[1].legend(('1', '2', '4', '6'), title='Filter order', loc='best')
        ax[1].set_xscale('log')
        ax[1].set_xlabel('Frequency / Critical frequency', fontsize=14)
        ax[1].set_ylabel('Phase [degrees]', fontsize=14)
        ax[1].set_yticks(np.arange(0, -300, -45))
        ax[1].set_ylim(-300, 10)
        ax[1].grid(which='both', axis='both')
        plt.tight_layout(w_pad=1)
        axi = plt.axes([0.115, 0.4, 0.15, 0.35])
        axi.plot(w, h1, 'b', w, h2, 'r', w, h4, 'g', w, h6, 'y', linewidth=2)
        axi.set_xticks((0.6, 1, 1.4))
        axi.set_yticks((-6, -3, 0))
        axi.set_ylim([-7, 1])
        axi.set_xlim([0.5, 1.5])
        axi.grid(which='both', axis='both')
    return (butterworth_plot,)


@app.cell
def _(butterworth_plot):
    butterworth_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Butterworth filter with zero-phase shift

        The phase introduced by the Butterworth filter can be corrected in the digital implementation by cleverly filtering the data twice, once forward and once backwards. So, the lag introduced in the first filtering is zeroed by the same lag in the opposite direction at the second pass. The result is a zero-phase shift (or zero-phase lag) filtering.  
        However, because after each pass the output power at the cutoff frequency is attenuated by two, by passing twice the second order Butterworth filter, the final output power will be attenuated by four. We have to correct the actual cutoff frequency value so that when employing the two passes, the filter will attenuate only by two.  
        The following formula gives the desired cutoff frequency for a second-order Butterworth filter according to the number of passes,$n$, (see Winter, 2009):$C = \sqrt[4]{2^{\frac{1}{n}} - 1}$For instance, for two passes,$n=2$,$C=\sqrt[4]{2^{\frac{1}{2}} - 1} \approx 0.802$.  
        The actual filter cutoff frequency will be:$fc_{actual} = \frac{fc_{desired}}{C}$For instance, for a second-order Butterworth filter with zero-phase shift and a desired 10 Hz cutoff frequency, the actual cutoff frequency should be 12.47 Hz.

        Let's implement this forward and backward filtering using the function `filtfilt` and compare with the single-pass filtering we just did it.
        """
    )
    return


@app.cell
def _(np, plt):
    from scipy.signal import butter, lfilter, filtfilt
    freq_1 = 100
    t_1 = np.arange(0, 1, 0.01)
    w_2 = 2 * np.pi * 1
    y_2 = np.sin(w_2 * t_1) + 0.1 * np.sin(10 * w_2 * t_1)
    (b_3, a_3) = butter(2, 5 / (freq_1 / 2), btype='low')
    y2_1 = lfilter(b_3, a_3, y_2)
    C = 0.802
    (b_3, a_3) = butter(2, 5 / C / (freq_1 / 2), btype='low')
    y3 = filtfilt(b_3, a_3, y_2)
    (fig_3, ax1_2) = plt.subplots(1, 1, figsize=(9, 4))
    ax1_2.plot(t_1, y_2, 'r.-', linewidth=2, label='raw data')
    ax1_2.plot(t_1, y2_1, 'b.-', linewidth=2, label='filter  @ 5 Hz')
    ax1_2.plot(t_1, y3, 'g.-', linewidth=2, label='filtfilt @ 5 Hz')
    ax1_2.legend(frameon=False, fontsize=14)
    ax1_2.set_xlabel('Time [s]')
    ax1_2.set_ylabel('Amplitude')
    plt.show()
    return t_1, y_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Critically damped digital filter

        A problem with a lowpass Butterworth filter is that it tends to overshoot or undershoot data with rapid changes (see for example, Winter (2009), Robertson et at. (2013), and Robertson & Dowling (2003)).  
        The Butterworth filter behaves as an underdamped second-order system and a critically damped filter doesn't have this overshoot/undershoot characteristic.   
        The function `crit_damp.py` calculates the coefficients (the b's and a's) for an IIR critically damped digital filter and corrects the cutoff frequency for the number of passes of the filter. The calculation of these coefficients is very similar to the calculation for the Butterworth filter, see the `critic_damp.py` code. This function can also calculate the Butterworth coefficients if this option is chosen.  
        The signature of `critic_damp.py` function is:  
        ```python
        critic_damp(fcut, freq, npass=2, fcorr=True, filt='critic')
        ```
        And here is an example of `critic_damp.py`:
        """
    )
    return


@app.cell
def _(np, warnings):
    """Coefficients of critically damped or Butterworth digital lowpass filter."""
    __author___1 = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version___1 = '1.0.0'
    __license__ = 'MIT'

    def critic_damp(fcut, freq, npass=2, fcorr=True, filt='critic'):
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
        fcorr : bool, optional (default = True)
            correct (True) or not the cutoff frequency for the number of passes.
        filt : string ('critic', 'butter'), optional (default = 'critic')
            'critic' to calculate coefficients for critically damped lowpass filter
            'butter' to calculate coefficients for Butterworth lowpass filter

        Returns
        -------
        b : 1D array
            b coefficients for the filter
        a : 1D array
            a coefficients for the filter
        fc : number
            corrected cutoff frequency considering the number of passes

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
        >>> print('b:', b_cd, '
    a:', a_cd, '
    Corrected Fc:', fc_cd)
        >>> print('Butterworth filter')
        >>> b_bw, a_bw, fc_bw = critic_damp(fcut=10, freq=100, npass=2, fcorr=True, filt='butter')
        >>> print('b:', b_bw, '
    a:', a_bw, '
    Corrected Fc:', fc_bw)
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
        if fcut > freq / 2:
            warnings.warn('Cutoff frequency can not be greater than Nyquist frequency.')
        if filt.lower() == 'critic':
            if fcorr:
                corr = 1 / np.power(2 ** (1 / (2 * npass)) - 1, 0.5)
        elif filt.lower() == 'butter':
            if fcorr:
                corr = 1 / np.power(2 ** (1 / npass) - 1, 0.25)
        else:
            warnings.warn('Invalid option for paraneter filt:', filt)
        if fcorr:
            fc = fcut * corr
            if fc > freq / 2:
                text = 'Warning: corrected cutoff frequency ({} Hz) is greater' + ' than Nyquist frequency ({} Hz). Using the uncorrected cutoff' + ' frequency ({} Hz).'
                print(text.format(fc, freq / 2, fcut))
                fc = fcut
        else:
            fc = fcut
        wc = np.tan(np.pi * fc / freq)
        k1 = np.sqrt(2) * wc if filt.lower() == 'butter' else 2 * wc
        k2 = wc * wc
        a0 = k2 / (1 + k1 + k2)
        a1 = 2 * a0
        a2 = k2 / (1 + k1 + k2)
        b1 = 2 * a0 * (1 / k2 - 1)
        b2 = 1 - (a0 + a1 + a2 + b1)
        b = np.array([a0, a1, a2])
        a = np.array([1, -b1, -b2])
        return (b, a, fc)
    return (critic_damp,)


app._unparsable_cell(
    r"""
    >>> print('Critically damped filter')
        >>> b_cd, a_cd, fc_cd = critic_damp(fcut=10, freq=100, npass=2, fcorr=True, filt='critic')
        >>> print('b:', b_cd, '\na:', a_cd, '\nCorrected Fc:', fc_cd)
        >>> print('Butterworth filter')
        >>> b_bw, a_bw, fc_bw = critic_damp(fcut=10, freq=100, npass=2, fcorr=True, filt='butter')
        >>> print('b:', b_bw, '\na:', a_bw, '\nCorrected Fc:', fc_bw)
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
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Comparison of FIR filter with a critically damped filter
        """
    )
    return


@app.cell
def _(critic_damp, signal_1, y_2):
    print('Critically damped filter')
    (b_cd, a_cd, fc_cd) = critic_damp(fcut=10, freq=100, npass=2, fcorr=True, filt='critic')
    print('b:', b_cd, '\na:', a_cd, '\nCorrected Fc:', fc_cd)
    print('FIR filter')
    b_fir = signal_1.firwin(numtaps=3, cutoff=10, fs=1000)
    print('b:', b_fir)
    y_cd = signal_1.filtfilt(b_cd, a_cd, y_2)
    y_fir = signal_1.filtfilt(b_fir, 1, y_2)
    return y_cd, y_fir


@app.cell
def _(plt, t_1, y_2, y_cd, y_fir):
    (fig_4, ax_1) = plt.subplots(1, 1, figsize=(9, 4))
    ax_1.plot(t_1, y_2, 'k', linewidth=2, label='raw data')
    ax_1.plot(t_1, y_cd, 'r', linewidth=2, label='Critically damped')
    ax_1.plot(t_1, y_fir, 'g', linewidth=2, label='FIR (freq. not corrected)')
    ax_1.legend()
    ax_1.set_xlabel('Time (s)')
    ax_1.set_ylabel('Amplitude')
    ax_1.set_title('Freq = 100 Hz, Fc = 10 Hz, 2nd order and zero-phase shift filters')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Moving-average filter

        Here are four different versions of a function to implement the moving-average filter:
        """
    )
    return


@app.cell
def _(lfilter_1, np):
    def moving_averageV1(x, window):
        """Moving average of 'x' with window size 'window'."""
        y = np.empty(len(x) - window + 1)
        for i in range(len(y)):
            y[i] = np.sum(x[i:i + window]) / window
        return y

    def moving_averageV2(x, window):
        """Moving average of 'x' with window size 'window'."""
        xsum = np.cumsum(x)
        xsum[window:] = xsum[window:] - xsum[:-window]
        return xsum[window - 1:] / window

    def moving_averageV3(x, window):
        """Moving average of 'x' with window size 'window'."""
        return np.convolve(x, np.ones(window) / window, 'same')
    from scipy.signal import lfilter

    def moving_averageV4(x, window):
        """Moving average of 'x' with window size 'window'."""
        return lfilter_1(np.ones(window) / window, 1, x)
    return (
        moving_averageV1,
        moving_averageV2,
        moving_averageV3,
        moving_averageV4,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's test these versions:
        """
    )
    return


@app.cell
def _(
    moving_averageV1,
    moving_averageV2,
    moving_averageV3,
    moving_averageV4,
    np,
    plt,
):
    x_1 = np.random.randn(300) / 10
    x_1[100:200] = x_1[100:200] + 1
    window_1 = 10
    y1 = moving_averageV1(x_1, window_1)
    y2_2 = moving_averageV2(x_1, window_1)
    y3_1 = moving_averageV3(x_1, window_1)
    y4 = moving_averageV4(x_1, window_1)
    (fig_5, ax_2) = plt.subplots(1, 1, figsize=(10, 5))
    ax_2.plot(x_1, 'b-', linewidth=1, label='raw data')
    ax_2.plot(y1, 'y-', linewidth=2, label='moving average V1')
    ax_2.plot(y2_2, 'm--', linewidth=2, label='moving average V2')
    ax_2.plot(y3_1, 'r-', linewidth=2, label='moving average V3')
    ax_2.plot(y4, 'g-', linewidth=2, label='moving average V4')
    ax_2.legend(frameon=False, loc='upper right', fontsize=12)
    ax_2.set_xlabel('Data #')
    ax_2.set_ylabel('Amplitude')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A test of the performance of the four versions (using the magick IPython function `timeit`):
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %timeit moving_averageV1(x, window)
    # magic command not supported in marimo; please file an issue to add support
    # %timeit moving_averageV2(x, window)
    # magic command not supported in marimo; please file an issue to add support
    # %timeit moving_averageV3(x, window)
    # magic command not supported in marimo; please file an issue to add support
    # %timeit moving_averageV4(x, window)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The version with the cumsum function produces identical results to the first version of the moving average function but it is much faster (the fastest of the four versions).  
        Only the version with the convolution function produces a result without a phase or lag between the input and output data, although we could improve the other versions to account for that (for example, calculating the moving average of `x[i-window/2:i+window/2]` and using `filtfilt` instead of `lfilter`).  
        And avoid as much as possible the use of loops in Python! The version with the for loop is about one hundred times slower than the other versions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Moving-RMS filter

        The root-mean square (RMS) is a measure of the absolute amplitude of the data and it is useful when the data have positive and negative values. The RMS is defined as:$RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$Similar to the moving-average measure, the moving RMS is defined as:$y[i] = \sqrt{\sum_{j=0}^{m-1} (x[i+j])^2} \;\;\;\; for \;\;\; i=1, \; \dots, \; n-m+1$Here are two implementations for a moving-RMS filter (very similar to the moving-average filter):
        """
    )
    return


@app.cell
def _(filtfilt_1, np):
    from scipy.signal import filtfilt

    def moving_rmsV1(x, window):
        """Moving RMS of 'x' with window size 'window'."""
        window = 2 * window + 1
        return np.sqrt(np.convolve(x * x, np.ones(window) / window, 'same'))

    def moving_rmsV2(x, window):
        """Moving RMS of 'x' with window size 'window'."""
        return np.sqrt(filtfilt_1(np.ones(window) / window, [1], x * x))
    return moving_rmsV1, moving_rmsV2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's filter electromyographic data:
        """
    )
    return


@app.cell
def _(np, pd):
    # load data file with EMG signal
    data = pd.read_csv('https://raw.githubusercontent.com/BMClab/BMC/master/data/emg.csv').to_numpy()
    data = data[300:1000,:]
    time = data[:, 0]
    data = data[:, 1] - np.mean(data[:, 1])
    return data, time


@app.cell
def _(data, moving_rmsV1, moving_rmsV2):
    window_2 = 50
    y1_1 = moving_rmsV1(data, window_2)
    y2_3 = moving_rmsV2(data, window_2)
    return y1_1, y2_3


@app.cell
def _(data, plt, time, y1_1, y2_3):
    (fig_6, ax_3) = plt.subplots(1, 1, figsize=(9, 5))
    ax_3.plot(time, data, 'k-', linewidth=1, label='raw data')
    ax_3.plot(time, y1_1, 'r-', linewidth=2, label='moving RMS V1')
    ax_3.plot(time, y2_3, 'b-', linewidth=2, label='moving RMS V2')
    ax_3.legend(frameon=False, loc='upper right', fontsize=12)
    ax_3.set_xlabel('Time [s]')
    ax_3.set_ylabel('Amplitude')
    ax_3.set_ylim(-0.1, 0.1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Similar, but not the same, results.
        An advantage of the filter employing the convolution method is that it behaves better to abrupt changes in the data, such as when filtering data that change from a baseline at zero to large positive values. The filter with the `filter` or `filtfilt` function would introduce negative values in this case.  
        Another advantage for the convolution method is that it is much faster:
        """
    )
    return


@app.cell
def _():
    print('Filter with convolution:')
    # magic command not supported in marimo; please file an issue to add support
    # %timeit moving_rmsV1(data, window)
    print('Filter with filtfilt:')
    # magic command not supported in marimo; please file an issue to add support
    # %timeit moving_rmsV2(data, window)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Moving-median filter

        The moving-median filter is similar in concept than the other moving filters but uses the median instead. This filter has a sharper response to abrupt changes in the data than the moving-average filter:
        """
    )
    return


@app.cell
def _(np, plt):
    from scipy.signal import medfilt
    x_2 = np.random.randn(300) / 10
    x_2[100:200] = x_2[100:200] + 1
    window_3 = 11
    y_3 = np.convolve(x_2, np.ones(window_3) / window_3, 'same')
    y2_4 = medfilt(x_2, window_3)
    (fig_7, ax_4) = plt.subplots(1, 1, figsize=(10, 4))
    ax_4.plot(x_2, 'b-', linewidth=1, label='raw data')
    ax_4.plot(y_3, 'r-', linewidth=2, label='moving average')
    ax_4.plot(y2_4, 'g-', linewidth=2, label='moving median')
    ax_4.legend(frameon=False, loc='upper right', fontsize=12)
    ax_4.set_xlabel('Data #')
    ax_4.set_ylabel('Amplitude')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### More moving filters

        The library [pandas](https://pandas.pydata.org/) has several types of [moving-filter functions](https://pandas.pydata.org/pandas-docs/version/0.15/computation.html#moving-rolling-statistics-moments).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Numerical differentiation of data with noise

        How to remove noise from a signal is rarely a trivial task and this problem gets worse with numerical differentiation of the data because the amplitudes of the noise with higher frequencies than the signal are amplified with differentiation (for each differentiation step, the SNR decreases).  
        To demonstrate this problem, consider the following function representing some experimental data:$f = sin(\omega t) + 0.1sin(10\omega t)$The first component, with large amplitude (1) and small frequency (1 Hz), represents the signal and the second component, with small amplitude (0.1) and large frequency (10 Hz), represents the noise. The signal-to-noise ratio (SNR) for these data is equal to (1/0.1)$^2$= 100. Let's see what happens with the SNR for the first and second derivatives of$f$:$f\:'\:= \omega cos(\omega t) + \omega cos(10\omega t)$$f\:''= -\omega^2 sin(\omega t) - 10\omega^2 sin(10\omega t)$For the first derivative, SNR = 1, and for the second derivative, SNR = 0.01!   
        The following plots illustrate this problem:
        """
    )
    return


@app.cell
def _(np, plt):
    t_2 = np.arange(0, 1, 0.01)
    w_3 = 2 * np.pi * 1
    s = np.sin(w_3 * t_2)
    n = 0.1 * np.sin(10 * w_3 * t_2)
    sd = w_3 * np.cos(w_3 * t_2)
    nd = w_3 * np.cos(10 * w_3 * t_2)
    sdd = -w_3 * w_3 * np.sin(w_3 * t_2)
    ndd = -w_3 * w_3 * 10 * np.sin(10 * w_3 * t_2)
    plt.rc('axes', labelsize=16, titlesize=16)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    (fig_8, (ax1_3, ax2_1, ax3)) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    ax1_3.set_title('Differentiation of signal and noise')
    ax1_3.plot(t_2, s, 'b.-', linewidth=1, label='signal')
    ax1_3.plot(t_2, n, 'g.-', linewidth=1, label='noise')
    ax1_3.plot(t_2, s + n, 'r.-', linewidth=2, label='signal+noise')
    ax2_1.plot(t_2, sd, 'b.-', linewidth=1)
    ax2_1.plot(t_2, nd, 'g.-', linewidth=1)
    ax2_1.plot(t_2, sd + nd, 'r.-', linewidth=2)
    ax3.plot(t_2, sdd, 'b.-', linewidth=1)
    ax3.plot(t_2, ndd, 'g.-', linewidth=1)
    ax3.plot(t_2, sdd + ndd, 'r.-', linewidth=2)
    ax1_3.legend(frameon=False, fontsize=10)
    ax1_3.set_ylabel('f')
    ax2_1.set_ylabel("f '")
    ax3.set_ylabel("f ''")
    ax3.set_xlabel('Time (s)')
    fig_8.tight_layout(pad=0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's see how the use of a low-pass Butterworth filter can attenuate the high-frequency noise and how the derivative is affected.    
        We will also calculate the [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) of these data to look at their frequencies content.
        """
    )
    return


@app.cell
def _(np, signal_2):
    from scipy import signal, fftpack
    freq_2 = 100
    t_3 = np.arange(0, 1, 0.01)
    w_4 = 2 * np.pi * 1
    y_4 = np.sin(w_4 * t_3) + 0.1 * np.sin(10 * w_4 * t_3)
    C_1 = 0.802
    (b_4, a_4) = signal_2.butter(2, 5 / C_1 / (freq_2 / 2), btype='low')
    y2_5 = signal_2.filtfilt(b_4, a_4, y_4)
    ydd = np.diff(y_4, 2) * freq_2 * freq_2
    y2dd = np.diff(y2_5, 2) * freq_2 * freq_2
    yfft = np.abs(fftpack.fft(y_4)) / (y_4.size / 2)
    y2fft = np.abs(fftpack.fft(y2_5)) / (y_4.size / 2)
    freqs = fftpack.fftfreq(y_4.size, 1.0 / freq_2)
    yddfft = np.abs(fftpack.fft(ydd)) / (ydd.size / 2)
    y2ddfft = np.abs(fftpack.fft(y2dd)) / (ydd.size / 2)
    freqs2 = fftpack.fftfreq(ydd.size, 1.0 / freq_2)
    return freqs, t_3, y2_5, y2dd, y2ddfft, y2fft, y_4, ydd, yddfft, yfft


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the plots:
        """
    )
    return


@app.cell
def _(freqs, plt, t_3, y2_5, y2dd, y2ddfft, y2fft, y_4, ydd, yddfft, yfft):
    (fig_9, ((ax1_4, ax2_2), (ax3_1, ax4))) = plt.subplots(2, 2, figsize=(11, 5))
    ax1_4.set_title('Temporal domain', fontsize=14)
    ax1_4.plot(t_3, y_4, 'r', linewidth=2, label='raw data')
    ax1_4.plot(t_3, y2_5, 'b', linewidth=2, label='filtered @ 5 Hz')
    ax1_4.set_ylabel('f')
    ax1_4.legend(frameon=False, fontsize=12)
    ax2_2.set_title('Frequency domain', fontsize=14)
    ax2_2.plot(freqs[:int(yfft.size / 4)], yfft[:int(yfft.size / 4)], 'r', linewidth=2, label='raw data')
    ax2_2.plot(freqs[:int(yfft.size / 4)], y2fft[:int(yfft.size / 4)], 'b--', linewidth=2, label='filtered @ 5 Hz')
    ax2_2.set_ylabel('FFT(f)')
    ax2_2.legend(frameon=False, fontsize=12)
    ax3_1.plot(t_3[:-2], ydd, 'r', linewidth=2, label='raw')
    ax3_1.plot(t_3[:-2], y2dd, 'b', linewidth=2, label='filtered @ 5 Hz')
    ax3_1.set_xlabel('Time [s]')
    ax3_1.set_ylabel("f ''")
    ax4.plot(freqs[:int(yddfft.size / 4)], yddfft[:int(yddfft.size / 4)], 'r', linewidth=2, label='raw')
    ax4.plot(freqs[:int(yddfft.size / 4)], y2ddfft[:int(yddfft.size / 4)], 'b--', linewidth=2, label='filtered @ 5 Hz')
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel("FFT(f '')")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Pezzack's benchmark data

        In 1977, Pezzack, Norman and Winter published a paper where they investigated the effects of differentiation and filtering processes on experimental data (the angle of a bar manipulated in space). Since then, these data have became a benchmark to test new algorithms. Let's work with these data (available at [https://isbweb.org/data/pezzack/index.html](https://isbweb.org/data/pezzack/index.html)). The data have the angular displacement measured by video and the angular acceleration  directly measured by an accelerometer, which we will consider as the true acceleration.
        """
    )
    return


@app.cell
def _(pd):
    data_1 = pd.read_csv('https://raw.githubusercontent.com/BMClab/BMC/master/data/Pezzack.txt', sep='\t', header=None, skiprows=6).to_numpy()
    (time_1, disp, disp2, aacc) = (data_1[:, 0], data_1[:, 1], data_1[:, 2], data_1[:, 3])
    return aacc, disp, time_1


@app.cell
def _(aacc, disp, np, plt, time_1):
    dt = np.mean(np.diff(time_1))
    (fig_10, (ax1_5, ax2_3)) = plt.subplots(1, 2, sharex=True, figsize=(11, 4))
    plt.suptitle("Pezzack's benchmark data", fontsize=20)
    ax1_5.plot(time_1, disp, 'b.-')
    ax1_5.set_xlabel('Time [s]')
    ax1_5.set_ylabel('Angular displacement [rad]', fontsize=12)
    ax2_3.plot(time_1, aacc, 'g.-')
    ax2_3.set_xlabel('Time [s]')
    ax2_3.set_ylabel('Angular acceleration [rad/s$^2$]', fontsize=12)
    plt.subplots_adjust(wspace=0.3)
    return (dt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The challenge is how to obtain the acceleration using the disclacement data dealing with the noise.   
        A simple double differentiation of these data will amplify the noise:
        """
    )
    return


@app.cell
def _(aacc, disp, dt, np, plt, time_1):
    aacc2 = np.diff(disp, 2) / dt / dt
    (fig_11, ax1_6) = plt.subplots(1, 1, figsize=(11, 4))
    plt.suptitle("Pezzack's benchmark data", fontsize=20)
    ax1_6.plot(time_1, aacc, 'g', label='Analog acceleration (true value)')
    ax1_6.plot(time_1[1:-1], aacc2, 'r', label='Acceleration by 2-point difference')
    ax1_6.set_xlabel('Time [s]', fontsize=12)
    ax1_6.set_ylabel('Angular acceleration [rad/s$^2$]', fontsize=12)
    plt.legend(frameon=False, fontsize=12, loc='upper left')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The source of noise in these data is due to random small errors in the digitization process which occur at each frame, because that the frequency content of the noise is up to half of the sampling frequency, higher the frequency content of the movement being analyzed.   
        Let's try different filters ([Butterworth](https://en.wikipedia.org/wiki/Butterworth_filter), [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_smoothing_filter), and [spline](https://en.wikipedia.org/wiki/Spline_function)) to attenuate this noise.  

        Both Savitzky-Golay and the spline functions are based on fitting polynomials to the data and they allow to differentiate the polynomials in order to get the derivatives of the data (instead of direct numerical differentiation of the data).  
        The Savitzky-Golay and the spline functions have the following signatures:  
        ```python
        savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)    
        splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None, full_output=0, per=0, quiet=1)
        ```
        And to evaluate the spline derivatives:  
        ```python
        splev(x, tck, der=0, ext=0)
        ```

        And let's employ the [root-mean-square error (RMSE)](https://en.wikipedia.org/wiki/RMSE) metric to compare their performance:
        """
    )
    return


@app.cell
def _(aacc, disp, dt, np, signal_3, time_1):
    from scipy import signal, interpolate
    C_2 = 0.802
    (b_5, a_5) = signal_3.butter(2, 9 / C_2 / (1 / dt / 2))
    dispBW = signal_3.filtfilt(b_5, a_5, disp)
    aaccBW = np.diff(dispBW, 2) / dt / dt
    disp_pad = signal_3._arraytools.odd_ext(disp, n=11)
    time_pad = signal_3._arraytools.odd_ext(time_1, n=11)
    aaccSG = signal_3.savgol_filter(disp_pad, window_length=5, polyorder=3, deriv=2, delta=dt)[11:-11]
    tck = interpolate.splrep(time_pad, disp_pad, k=5, s=0.15 * np.var(disp_pad) / np.size(disp_pad))
    aaccSP = interpolate.splev(time_pad, tck, der=2)[11:-11]
    rmseBW = np.sqrt(np.mean((aaccBW - aacc[1:-1]) ** 2))
    rmseSG = np.sqrt(np.mean((aaccSG - aacc) ** 2))
    rmseSP = np.sqrt(np.mean((aaccSP - aacc) ** 2))
    return aaccBW, aaccSG, aaccSP, rmseBW, rmseSG, rmseSP


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the plots:
        """
    )
    return


@app.cell
def _(aacc, aaccBW, aaccSG, aaccSP, plt, rmseBW, rmseSG, rmseSP, time_1):
    (fig_12, ax1_7) = plt.subplots(1, 1, figsize=(11, 4))
    plt.suptitle("Pezzack's benchmark data", fontsize=20)
    ax1_7.plot(time_1, aacc, 'g', label='Analog acceleration:         (True value)')
    ax1_7.plot(time_1[1:-1], aaccBW, 'r', label='Butterworth 9 Hz:             RMSE = %0.2f' % rmseBW)
    ax1_7.plot(time_1, aaccSG, 'b', label='Savitzky-Golay 5 points:   RMSE = %0.2f' % rmseSG)
    ax1_7.plot(time_1, aaccSP, 'm', label='Quintic spline, s=0.0005: RMSE = %0.2f' % rmseSP)
    ax1_7.set_xlabel('Time [s]')
    ax1_7.set_ylabel('Angular acceleration [rad/s$^2$]', fontsize=12)
    plt.legend(frameon=False, fontsize=12, loc='upper left')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        At this case, the Butterworth, Savitzky-Golay, and spline filters produced similar results with good fits to the original curve. However, with all of them, particularly with the spline smoothing, it is necessary some degree of tuning for choosing the right parameters. The Butterworth filter is the easiest one because the cutoff frequency choice sound more familiar for human movement analysis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Kinematics of a ball toss

        Let's now analyse the kinematic data of a ball tossed to the space. These data were obtained using [Tracker](https://physlets.org/tracker/), which is a free video analysis and modeling tool built on the [Open Source Physics](https://www.compadre.org/osp/) (OSP) Java framework.   
        The data are from the analysis of the video *balltossout.mov* from the mechanics video collection which can be obtained in the Tracker website.
        """
    )
    return


@app.cell
def _(np, pd):
    data_2 = pd.read_csv('https://raw.githubusercontent.com/BMClab/BMC/master/data/balltoss.txt', sep='\t', header=None, skiprows=2).to_numpy()
    (t_4, x_3, y_5) = (data_2[:, 0], data_2[:, 1], data_2[:, 2])
    dt_1 = np.mean(np.diff(t_4))
    print('Time interval: %f s' % dt_1)
    print('x and y values:')
    (x_3, y_5)
    return dt_1, t_4, x_3, y_5


@app.cell
def _(plt, t_4, x_3, y_5):
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    (fig_13, (ax1_8, ax2_4, ax3_2)) = plt.subplots(1, 3, figsize=(12, 3))
    plt.suptitle('Kinematics of a ball toss', fontsize=14, y=1)
    ax1_8.plot(x_3, y_5, 'go')
    ax1_8.set_ylabel('y [m]')
    ax1_8.set_xlabel('x [m]')
    ax2_4.plot(t_4, x_3, 'bo')
    ax2_4.set_ylabel('x [m]')
    ax2_4.set_xlabel('Time [s]')
    ax3_2.plot(t_4, y_5, 'ro')
    ax3_2.set_ylabel('y [m]')
    ax3_2.set_xlabel('Time [s]')
    plt.subplots_adjust(wspace=0.4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Calculate the velocity and acceleration numerically:
        """
    )
    return


@app.cell
def _(dt_1, np, x_3, y_5):
    (vx, vy) = (np.diff(x_3) / dt_1, np.diff(y_5) / dt_1)
    (ax_5, ay) = (np.diff(vx) / dt_1, np.diff(vy) / dt_1)
    (vx2, vy2) = ((x_3[2:] - x_3[:-2]) / (2 * dt_1), (y_5[2:] - y_5[:-2]) / (2 * dt_1))
    (ax2_5, ay2) = ((vx2[2:] - vx2[:-2]) / (2 * dt_1), (vy2[2:] - vy2[:-2]) / (2 * dt_1))
    return ax2_5, ax_5, ay, ay2, vx, vx2, vy, vy2


@app.cell
def _(ax2_5, ax_5, ay, ay2, plt, t_4, vx, vx2, vy, vy2, x_3, y_5):
    (fig_14, axarr) = plt.subplots(2, 3, sharex=True, figsize=(11, 6))
    axarr[0, 0].plot(t_4, x_3, 'bo')
    axarr[0, 0].set_ylabel('x [m]')
    axarr[0, 1].plot(t_4[:-1], vx, 'bo', label='forward difference')
    axarr[0, 1].set_ylabel('vx [m/s]')
    axarr[0, 1].plot(t_4[1:-1], vx2, 'm+', markersize=10, label='central difference')
    axarr[0, 1].legend(frameon=False, fontsize=10, loc='upper left', numpoints=1)
    axarr[0, 2].plot(t_4[:-2], ax_5, 'bo')
    axarr[0, 2].set_ylabel('ax [m/s$^2$]')
    axarr[0, 2].plot(t_4[2:-2], ax2_5, 'm+', markersize=10)
    axarr[1, 0].plot(t_4, y_5, 'ro')
    axarr[1, 0].set_ylabel('y [m]')
    axarr[1, 1].plot(t_4[:-1], vy, 'ro')
    axarr[1, 1].set_ylabel('vy [m/s]')
    axarr[1, 1].plot(t_4[1:-1], vy2, 'm+', markersize=10)
    axarr[1, 2].plot(t_4[:-2], ay, 'ro')
    axarr[1, 2].set_ylabel('ay [m/s$^2$]')
    axarr[1, 2].plot(t_4[2:-2], ay2, 'm+', markersize=10)
    axarr[1, 1].set_xlabel('Time [s]')
    plt.tight_layout(w_pad=-0.5, h_pad=0)
    plt.suptitle('Kinematics of a ball toss', fontsize=14, y=1.05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can observe the noise, particularly in the derivatives of the data. For example, the vertical acceleration of the ball should be constant, approximately g=9.8 m/s$^2$.   
        To estimate the acceleration, we can get rid off the noise by filtering the data or, because we know the physics of the phenomenon, we can fit a model to the data. Let's try the latter option.
        """
    )
    return


@app.cell
def _(np, t_4, y_5):
    p = np.polyfit(t_4, y_5, 2)
    print('g = %0.2f m/s2' % (2 * p[0]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A good estimation but is seems there is a problem with the video because the acceleration at the end seems to increase (see figure above); maybe there is a distortion in the video at its extremity.

        To read more about fitting a model to data (in this case a mathematical equation), read the text [curve fitting](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The optimal cutoff frequency

        Probably after reading this tutorial you are wondering how to automatically determine the optimal cutoff frequency that should be employed in a low-pass filter to attenuate as much as possible the noise without compromising the signal content in the data.   
        This is an important topic in signal processing, particularly in movement science, and we discuss one method for that in the text [Residual analysis to determine the optimal cutoff frequency](https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - David A. Winter (2009) [Biomechanics and motor control of human movement](http://books.google.com.br/books?id=_bFHL08IWfwC). 4th edition. Hoboken: Wiley.
        - [dspGuru - Digital Signal Processing Central](http://www.dspguru.com/).
        - Gordon Robertson, Graham Caldwell, Joseph Hamill, Gary Kamen (2013) [Research Methods in Biomechanics](http://books.google.com.br/books?id=gRn8AAAAQBAJ). 2nd Edition. Human Kinetics.  
        - Pezzack JC, Norman RW, & Winter DA (1977). [An assessment of derivative determining techniques used for motion analysis](http://www.health.uottawa.ca/biomech/courses/apa7305/JB-Pezzack-Norman-Winter-1977.pdf). Journal of Biomechanics, 10, 377-382. [PubMed](http://www.ncbi.nlm.nih.gov/pubmed/893476).
        - Richard G. Lyons (2010) [Understanding Digital Signal Processing](http://books.google.com.br/books?id=UBU7Y2tpwWUC&hl). 3rd edition. Prentice Hall.  
        - Robertson DG, Dowling JJ (2003) [Design and responses of Butterworth and critically damped digital filters](https://www.ncbi.nlm.nih.gov/pubmed/14573371). J Electromyogr Kinesiol. 13(6), 569-573.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
