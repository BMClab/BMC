import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic properties of signals

    > Marcos Duarte, Renato Naville Watanabe,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Every measurement in biomechanics ends up as a signal: a force plate, a marker on the heel, an electrode on a muscle. Before anything can be filtered, differentiated or interpreted, it helps to know the vocabulary used to describe a signal, and what a computer actually stores when it records one.

    This notebook introduces that vocabulary: amplitude, frequency, period and phase; harmonics; AC and DC components; even and odd functions; continuous and discrete, analog and digital signals; sampling and quantization; signal and noise. Along the way it keeps returning to one real recording, the electrical activity of a muscle, and by the end you will have read off that file how often it was sampled and how fine the steps of the converter that digitized it were.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it; the prediction is the point, and being wrong is the most useful thing that can happen to you here.

    **Challenge 0.** Pick a sensor you have used or would like to use: a force plate, a motion-capture camera, an accelerometer, an EMG electrode. Write down two guesses. How many numbers per second does it store? And how many *different* values can each of those numbers take? Keep the guesses; the sections on sampling and quantization answer both questions for one real sensor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the computation, pandas reads the data file, and Matplotlib draws the plots.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A recorded signal

    Here is a fraction of a second of surface electromyography (EMG): the electrical activity of a muscle, picked up by electrodes on the skin. The file is in this repository's [data folder](https://github.com/BMClab/BMC/blob/master/data/emg.csv); its first column is time, in seconds, and the second is the EMG amplitude. The bottom panel zooms in on 25 ms of it and shows every value in the file as a dot.
    """)
    return


@app.cell
def _(np, pd, plt):
    emg_data = pd.read_csv(
        "https://raw.githubusercontent.com/BMClab/BMC/master/data/emg.csv", header=None
    ).to_numpy()
    emg_time, emg = emg_data[:, 0], emg_data[:, 1]

    _fig, _axs = plt.subplots(2, 1, figsize=(10, 6))
    _axs[0].plot(emg_time, emg, color="tab:blue", linewidth=1)
    _axs[0].axvspan(0.5, 0.525, color="tab:orange", alpha=0.3)
    _axs[0].set_xlabel("Time [s]")
    _axs[0].set_ylabel("EMG")
    _axs[0].set_title("Surface EMG")
    _zoom = (emg_time >= 0.5) & (emg_time <= 0.525)
    _axs[1].plot(emg_time[_zoom], emg[_zoom], ".-", color="tab:orange", drawstyle="steps-mid")
    _axs[1].set_xlabel("Time [s]")
    _axs[1].set_ylabel("EMG")
    _axs[1].set_title("Zoom on the shaded 25 ms")
    _axs[1].grid(True, linestyle=":")
    for _ax in _axs:
        _ax.axhline(np.mean(emg), color="k", linewidth=0.8)
    plt.tight_layout()
    plt.show()
    return emg, emg_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Guiding questions 0.**

    1. Count the dots in the zoom. Roughly how many values per second did the recording system store?
    2. In the zoom the signal moves in steps, and some values repeat exactly. Do you think the muscle's electrical activity really changes in steps?
    3. The black line is the mean of the whole signal. Is it zero? Should it be?
    4. Which parts of this recording would you call *signal*, and which *noise*?

    Keep your answers. The rest of the notebook gives you the words, and the code, to answer each of them precisely.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What a signal is

    A signal is a set of data that conveys information about some phenomenon (Bendat and Piersol, 2010; Lathi, 2009; Lyons, 2010). It can be represented mathematically by a function of one or more independent variables, and we often refer to a signal simply as data. The time-dependent voltage of an electric circuit, the acceleration of a moving body and the EMG above are all signals.

    What follows is a brief description of the basic properties of signals. For more detail, see Bendat and Piersol (2010), Lathi (2009), Lyons (2010) and Smith (1997).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Amplitude, frequency, period, and phase

    A periodic function can be characterized by its amplitude, frequency, period and phase. Let's see these properties in a periodic function composed of a single frequency, the sine wave or sinusoid ([trigonometric function](https://en.wikipedia.org/wiki/Trigonometric_functions)):

    $$
    x(t) = A\sin(2 \pi f t + \phi)
    $$

    where $A$ is the amplitude, $f$ the frequency, $\phi$ the phase, and $T=1/f$ the period of the function $x(t)$.

    We can define $\omega=2\pi f = 2\pi/T$ as the angular frequency, and then:

    $$
    x(t) = A\sin(\omega t + \phi)
    $$

    Let's visualize two such functions: $x_1 = \sin(2\pi t)$, with unit amplitude, a frequency of 1 Hz and no phase, and $x_2 = 2\sin(\pi t + \pi/4)$, with an amplitude of 2, a frequency of 0.5 Hz and a phase of $\pi/4$ rad ($45^o$).
    """)
    return


@app.cell
def _(np):
    t = np.linspace(-2, 2, 101)  # time vector [s]
    A = 2  # amplitude
    freq = 0.5  # frequency [Hz]
    phase = np.pi / 4  # phase [rad] (45 degrees)
    x1 = 1 * np.sin(2 * np.pi * 1 * t + 0)  # sinusoid 1
    x2 = A * np.sin(2 * np.pi * freq * t + phase)  # sinusoid 2
    return t, x1, x2


@app.function
def wave_plot(t, x1, x2, x3=None):
    """Plot two sinusoids annotated with amplitude, period and phase.

    If `x3` is given, it is plotted too, as the sum of the other two.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    _, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ax.plot(t, x1, color=[0, 0.5, 0, 0.5], linewidth=5)
    ax.plot(t, x2, color=[0, 0, 1, 0.5], linewidth=5)
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="both", direction="inout", which="both", length=5)
    ax.set_xlim((-2.05, 2.05))
    ax.set_ylim((-2.1, 2.1))
    ax.locator_params(axis="both", nbins=7)
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(which="both")
    ax.set_title(r"$A\sin(2\pi ft+\phi)$", loc="left", size=18, color=[0, 0, 0])
    ax.set_title(r"$x_1=\sin(2\pi t)$", loc="center", size=18, color=[0, 0.5, 0])
    ax.set_title(r"$x_2=2\sin(\pi t + \pi/4)$", loc="right", size=18, color=[0, 0, 1])
    ax.annotate(
        "", xy=(0.25, 0), xycoords="data", xytext=(0.25, 2), size=16,
        textcoords="data", arrowprops={"arrowstyle": "<->", "fc": "b", "ec": "b"},
    )
    ax.annotate(
        r"$A=2$", xy=(0.25, 1.1), xycoords="data", xytext=(0, 0),
        textcoords="offset points", size=18, color="b",
    )
    ax.annotate(
        "", xy=(0, 1.6), xycoords="data", xytext=(-0.25, 1.6), size=16,
        textcoords="data", arrowprops={"arrowstyle": "<->", "fc": "b", "ec": "b"},
    )
    ax.annotate(
        r"$t_{\phi}=\phi/2\pi f\,(\phi=\pi/4)$", xy=(-1.25, 1.6), xycoords="data",
        xytext=(0, -5), textcoords="offset points", size=18, color="b",
    )
    ax.annotate(
        "", xy=(-0.25, 0), xycoords="data", xytext=(-0.25, 2), textcoords="data",
        arrowprops={"arrowstyle": "-", "linestyle": "dotted", "fc": "b", "ec": "b"},
        size=16,
    )
    ax.annotate(
        "", xy=(-0.75, -1.8), xycoords="data", xytext=(1.25, -1.8),
        textcoords="data", size=10,
        arrowprops={"arrowstyle": "|-|", "fc": "b", "ec": "b"},
    )
    ax.annotate(
        r"$T=1/f\,(f=0.5\,Hz)$", xy=(-0.16, -1.8), xycoords="data",
        xytext=(0, 8), textcoords="offset points", size=18, color="b",
    )
    ax.annotate(
        r"$t[s]$", xy=(2.05, -0.5), xycoords="data", xytext=(0, 0),
        textcoords="offset points", size=18, color="k",
    )
    if x3 is not None:
        ax.plot(t, x3, "r", linewidth=6)
        ax.annotate(
            r"$x_3 = x_1 + x_2$", xy=(1.2, 2.3), xycoords="data", size=20, color="r"
        )
        ax.set_ylim((-3.1, 3.1))
    plt.suptitle(
        r"Amplitude ($A$), frequency ($f$), period ($T$), phase ($\phi$)",
        fontsize=18, y=1.02,
    )
    plt.tight_layout()
    plt.show()


@app.cell
def _(t, x1, x2):
    wave_plot(t, x1, x2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The phase shifts the curve in time. A phase $\phi$ corresponds to a time shift of $t_\phi = \phi/(2\pi f)$, which for $x_2$ is $(\pi/4)/\pi = 0.25$ s: $x_2$ crosses zero going up at $t=-0.25$ s instead of at $t=0$. A positive phase moves the curve to the *left*, earlier in time. We say that $x_2$ *leads* a sinusoid with no phase.

    **Guiding questions 1.**

    1. Read the period of $x_1$ off the plot. Does it agree with its frequency?
    2. What phase would make $x_2$ cross zero going up at $t = +0.25$ s?
    3. A sine and a cosine of the same frequency differ only in phase. By how much?

    ### The sum of two sinusoids

    Can you guess the shape of the sum of the two curves we just plotted?

    $$
    x_3 = x_1 + x_2 = \sin(2 \pi t) + 2\sin(\pi t + \pi/4)
    $$

    **Before you run the next cell**, sketch $x_3$ on your scratchpad and predict three numbers: its period, its largest value, and its smallest value. Are they $\pm(1 + 2) = \pm 3$?
    """)
    return


@app.cell
def _(np, t, x1, x2):
    x3 = x1 + x2
    wave_plot(t, x1, x2, x3)
    print(f"Largest value of x3:  {np.max(x3):5.2f}, at t = {t[np.argmax(x3)]:.2f} s")
    print(f"Smallest value of x3: {np.min(x3):5.2f}, at t = {t[np.argmin(x3)]:.2f} s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The largest value is 3, but the smallest is only about $-1.5$. Both sinusoids peak at $t = 0.25$ s, so there their amplitudes add; their troughs never coincide, so the sum never reaches $-3$. Whether amplitudes add depends on the phases. With $\phi = 0$ for $x_2$, the peaks no longer meet and the sum stays within about $\pm 2.6$.

    The sum of two sinusoids of different frequencies is not a sinusoid, and it need not even be symmetric about zero. It repeats, though, and the next section is about how often.

    ### Magnitude and power

    Two terms related to amplitude are magnitude and power. The magnitude is the absolute value of the amplitude, that is, the amplitude without its sign. The power of a signal is proportional to its amplitude (or magnitude) squared. Doubling the amplitude of a signal quadruples its power.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Periodic function, fundamental frequency, and harmonics

    A function is said to be periodic with period $T$ if $x(t+T) = x(t)$ for all values of $t$, that is, the function repeats itself after a period. Two important consequences of this definition are that the sum of periodic functions with a common period, and a constant times a periodic function, are also periodic functions. It also follows that a periodic function repeats itself after $2T,\: 3T, \dots$

    The shortest period after which the function repeats itself is said to be its fundamental period, and its inverse is the fundamental frequency of the periodic function. A [harmonic](http://en.wikipedia.org/wiki/Harmonic) is a component frequency of the function that is an integer multiple of the fundamental frequency, that is, the harmonics have frequencies $f,\: 2f,\: 3f, \dots$ (with periods $T,\: T/2,\: T/3, \dots$) and are referred to as the first, second, and third harmonics, and so on.

    **Challenge 1.** Go back to $x_3 = x_1 + x_2$.

    1. What is its fundamental period, and its fundamental frequency? Check your earlier prediction against the plot.
    2. Which of $x_1$ and $x_2$ is the first harmonic of $x_3$, and which is the second?
    3. Replace the frequency of $x_1$ by $\sqrt{2}$ Hz. Is the sum still periodic? Why not?

    This idea, that a periodic signal can be built from a fundamental and its harmonics, is the starting point of the [Fourier series](https://github.com/BMClab/BMC/blob/master/notebooks/FourierSeries.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AC and DC components

    A signal can also be characterized by its [AC and DC components](https://en.wikipedia.org/wiki/DC_bias). The terms come from electronics, where they mean alternating current and direct current. The DC component, also called the DC offset or DC bias, is simply the average (mean) value of the function, its constant part. The AC component is the oscillatory part of the function, found by subtracting the average value (the DC component) from the function itself. The next figure illustrates these components.
    """)
    return


@app.function
def ac_dc_plot():
    """Plot the AC and DC components of a signal, and their sum."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    t = np.linspace(0, 1, 101)
    ac = np.sin(2 * 4 * np.pi * t)
    dc = 2 * np.ones(t.shape)
    ax[0].plot(t, ac, "b", linewidth=3, label="AC")
    ax[0].set_title(r"$AC:\;\sin(8\pi t)$", fontsize=16)
    ax[1].plot(t, dc, "g", linewidth=3, label="DC")
    ax[1].set_title(r"$DC:\; 2$", fontsize=16)
    ax[2].plot(t, ac + dc, "r", linewidth=3, label="AC+DC")
    ax[2].plot(t, ac, "b:", linewidth=2, label="AC")
    ax[2].plot(t, dc, "g:", linewidth=2, label="DC")
    ax[2].set_title(r"$AC+DC:\;\sin(8\pi t)+2$", fontsize=16)
    for axi in ax:
        axi.set_ylim(-1.2, 3.2)
        axi.margins(0.02)
        axi.spines["bottom"].set_position("zero")
        axi.spines["top"].set_color("none")
        axi.spines["left"].set_position("zero")
        axi.spines["right"].set_color("none")
        axi.xaxis.set_ticks_position("bottom")
        axi.yaxis.set_ticks_position("left")
        axi.tick_params(axis="both", direction="inout", which="both", length=5)
        axi.locator_params(axis="both", nbins=4)
    fig.tight_layout()
    plt.show()


@app.cell
def _():
    ac_dc_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that for a periodic function like $x(t)=A\cos(2\pi f t)$, when its frequency is zero, $x(t)=A$. That is, the function has only a DC component. For that reason we say that the DC component of a function has a frequency equal to zero (an infinite period).

    Let's separate the two components of the right-hand signal numerically. Besides the DC component, we will compute the RMS (root mean square) of the AC component, the square root of its mean squared value, a common measure of the size of an oscillation:

    $$
    RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
    $$

    **Before you run the next cell**, predict the DC component and the RMS of the AC component. The AC component has a peak of 1; is its RMS larger or smaller than that?
    """)
    return


@app.cell
def _(np):
    _t = np.arange(0, 1, 0.01)  # four whole periods of the AC component
    _x = np.sin(2 * 4 * np.pi * _t) + 2

    _dc = np.mean(_x)
    _ac = _x - _dc
    print(f"DC component:          {_dc:.3f}")
    print(f"Peak of the AC part:   {np.max(np.abs(_ac)):.3f}")
    print(f"RMS of the AC part:    {np.sqrt(np.mean(_ac**2)):.3f}")
    print(f"RMS of the whole x:    {np.sqrt(np.mean(_x**2)):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The RMS of a sinusoid is its amplitude divided by $\sqrt{2}$, about 0.707 times the peak. Remember that number: it reappears in [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) as the gain of a filter at its cutoff frequency. Note also that the RMS of the whole signal, 2.12, is dominated by the DC component; to describe the oscillation alone, remove the mean first.

    Now back to the EMG. **Before you run the next cell**, look at the black line in the first figure again, and predict whether the DC component of the EMG is large or small compared with the RMS of its AC part.
    """)
    return


@app.cell
def _(emg, np):
    _dc = np.mean(emg)
    print(f"EMG DC component:       {_dc:.5f}")
    print(f"EMG RMS of the AC part: {np.sqrt(np.mean((emg - _dc) ** 2)):.5f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The DC component of this EMG is about one percent of its RMS, small but not zero. The electrical activity of a muscle, picked up between two electrodes, has no physiological reason to have a mean different from zero; a DC offset in an EMG comes from the electrodes and the amplifier. That is why the first step in processing EMG is usually to subtract the mean, as [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) does before computing a moving RMS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Even and odd functions

    [Even and odd functions](https://en.wikipedia.org/wiki/Even_and_odd_functions) are functions that satisfy symmetry relations with respect to taking additive inverses. The [additive inverse](https://en.wikipedia.org/wiki/Additive_inverse) of a number $x$ is the number that, added to $x$, yields zero.

    The function $x(t)$ is even if $x(t) = x(-t)$, and odd if $x(t) = -x(-t)$.

    The cosine function is even, for instance, $\cos(\pi)=\cos(-\pi)$, and the sine function is odd, for instance, $\sin(\pi/2)=-\sin(-\pi/2)$; see the next figure.
    """)
    return


@app.function
def even_odd_plot():
    """Plot an even function (cosine) and an odd function (sine)."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    t = np.linspace(-np.pi, np.pi, 101)
    ax[0].plot(t, np.cos(t), "b", linewidth=3, label=r"$\cos(t)$")
    ax[0].plot([np.pi, np.pi], [0, -1], "r:", linewidth=3)
    ax[0].plot([-np.pi, -np.pi], [0, -1], "r:", linewidth=3)
    ax[0].plot([-np.pi, np.pi], [-1, -1], "r:", linewidth=3)
    ax[0].set_title(r"Even function: $\cos(t) = \cos(-t)$", fontsize=15)
    ax[1].plot(t, np.sin(t), "g", linewidth=3, label=r"$\sin(t)$")
    ax[1].plot([-np.pi / 2, -np.pi / 2], [0, -1], "r:", linewidth=3)
    ax[1].plot([-np.pi / 2, 0], [-1, -1], "r:", linewidth=3)
    ax[1].plot([np.pi / 2, np.pi / 2], [0, 1], "r:", linewidth=3)
    ax[1].plot([0, np.pi / 2], [1, 1], "r:", linewidth=3)
    ax[1].set_title(r"Odd function: $\sin(t) = -\sin(-t)$", fontsize=15)
    for axi in ax:
        axi.margins(0.02)
        axi.spines["bottom"].set_position("zero")
        axi.spines["top"].set_color("none")
        axi.spines["left"].set_position("zero")
        axi.spines["right"].set_color("none")
        axi.xaxis.set_ticks_position("bottom")
        axi.yaxis.set_ticks_position("left")
        axi.tick_params(axis="both", direction="inout", which="both", length=5)
        axi.set_xlim((-np.pi - 0.2, np.pi + 0.2))
        axi.set_ylim((-1.1, 1.1))
        axi.set_xticks(np.linspace(-np.pi, np.pi, 5))
        axi.set_xticklabels(
            [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"], fontsize=14
        )
        for label in axi.get_xticklabels() + axi.get_yticklabels():
            label.set_bbox(dict(facecolor="white", edgecolor="None", alpha=0.65))
    fig.tight_layout()
    plt.show()


@app.cell
def _():
    even_odd_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most functions are neither even nor odd, but every function can be written as the sum of an even part and an odd part:

    $$
    x(t) = \underbrace{\frac{x(t) + x(-t)}{2}}_{\text{even}} + \underbrace{\frac{x(t) - x(-t)}{2}}_{\text{odd}}
    $$

    **Challenge 2.**

    1. Check that the first term is indeed even and the second is odd.
    2. Find the even and odd parts of $x(t) = e^{t}$. Do you recognize them?
    3. Is $x_3 = \sin(2\pi t) + 2\sin(\pi t + \pi/4)$ even, odd, or neither? Plot its even and odd parts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuous and discrete signals

    A [continuous signal](https://en.wikipedia.org/wiki/Continuous_signal) depends on a continuous variable, that is, its independent variable varies continuously (it has a continuum domain). A [discrete signal](https://en.wikipedia.org/wiki/Discrete-time_signal) depends on a discrete variable, defined only on a discrete set of values. For instance, the temperature throughout the day, $T(t)$, is a continuous signal, because it depends on time, which varies continuously. If we measure the temperature only at certain times, say every hour, the new signal is discrete, because its independent variable is discrete ($t$ = 8 am, 9 am, 10 am, ...).

    The following figure illustrates continuous and discrete signals (although, since we are using a digital computer, the continuous signal below is in fact not authentic!):
    """)
    return


@app.function
def cont_disc_plot():
    """Plot a continuous and a discrete version of the same sinusoid."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    t = np.linspace(0, 1, 101)
    x = np.sin(2 * np.pi * t)
    ax[0].plot(t, x, "r", linewidth=3)
    ax[0].set_title("Continuous signal", fontsize=15)
    ax[1].stem(t[::5], x[::5], markerfmt="ro", linefmt="b--")
    ax[1].set_title("Discrete signal", fontsize=15)
    for axi in ax:
        axi.margins(0.02)
        axi.spines["bottom"].set_position("zero")
        axi.spines["top"].set_color("none")
        axi.spines["left"].set_position("zero")
        axi.spines["right"].set_color("none")
        axi.xaxis.set_ticks_position("bottom")
        axi.yaxis.set_ticks_position("left")
        axi.tick_params(axis="both", direction="inout", which="both", length=5)
        axi.locator_params(axis="both", nbins=5)
        axi.set_ylim((-1.1, 1.1))
        axi.set_ylabel("Amplitude", fontsize=14)
        axi.annotate(
            "t[s]", xy=(0.95, 0.1), xycoords="data", color="k", size=13,
            xytext=(0, 0), textcoords="offset points",
        )
    fig.tight_layout()
    plt.show()


@app.cell
def _():
    cont_disc_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling

    The reduction of a continuous signal to a discrete signal is called <a href="https://en.wikipedia.org/wiki/Sampling_(signal_processing)">sampling</a>, and the frequency at which it is performed is called the sampling frequency or sampling rate, in hertz (Hz). For instance, the discrete signal plotted above has a sampling frequency of 20 Hz: it was sampled every 0.05 s.

    Sampling is the basis for using digital computers to record and store data from an observed phenomenon. Computers have a finite memory, so they cannot store a continuous signal (how many instants are there in one second?).

    The EMG file stores time in its first column, so its sampling frequency can be read off the data. **Before you run the next cell**, compare your answer to Guiding question 0.1 with the guess you made in Challenge 0.
    """)
    return


@app.cell
def _(emg_time, np):
    _dt = np.diff(emg_time)
    print(f"Sampling interval: {np.mean(_dt) * 1000:.3f} ms (from {_dt.min() * 1000:.3f} to {_dt.max() * 1000:.3f} ms)")
    print(f"Sampling frequency: {1 / np.mean(_dt):.0f} Hz")
    print(f"Number of samples: {emg_time.size}, duration: {emg_time[-1] - emg_time[0]:.3f} s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One thousand samples per second, at perfectly regular intervals. Is that fast enough? The answer depends on what frequencies the signal contains, and that is the subject of the next section.

    ### Nyquist-Shannon sampling theorem

    Some requirements must be satisfied for the proper discretization of a signal, and an important one is expressed in the [Nyquist-Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem), in Shannon's words:

    > "If a function x(t) contains no frequencies higher than B hertz, it is completely determined by giving its ordinates at a series of points spaced 1/(2B) seconds apart."

    That is, to properly acquire data from a phenomenon whose highest frequency component is $f_B$, the sampling frequency $f_s$ must be at least twice that, $f_s\geq2f_B$. The Nyquist frequency, the highest frequency that can be represented in the sampled signal, is half of the sampling frequency. For the EMG above it is 500 Hz, which is enough: most of the power of surface EMG lies roughly between 20 and 450 Hz.

    When the theorem is not satisfied, and a continuous signal is sampled at less than twice its highest frequency, an effect called [aliasing](https://en.wikipedia.org/wiki/Aliasing) occurs: the discrete signal no longer contains the same information as the continuous one. It does not just lose the high frequencies. It shows them disguised as low ones.

    **Before you run the next cell**, predict what you will see when a 9 Hz sinusoid is sampled at 10 Hz, well below the 18 Hz the theorem demands.
    """)
    return


@app.cell
def _(np, plt):
    _t = np.linspace(0, 2, 2001)  # (almost) continuous time
    _ts = np.arange(0, 2, 1 / 10)  # sampled at 10 Hz
    _x9 = np.sin(2 * np.pi * 9 * _ts)
    _x1 = -np.sin(2 * np.pi * 1 * _ts)

    plt.figure(figsize=(10, 3.5))
    plt.plot(_t, np.sin(2 * np.pi * 9 * _t), color="0.7", linewidth=1, label="9 Hz sinusoid")
    plt.plot(_t, -np.sin(2 * np.pi * 1 * _t), "b--", linewidth=2, label="1 Hz sinusoid (inverted)")
    plt.plot(_ts, _x9, "ro", markersize=8, label="9 Hz sampled at 10 Hz")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.show()
    print(f"Largest difference between the samples of the two sinusoids: {np.max(np.abs(_x9 - _x1)):.1e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The samples of the 9 Hz sinusoid are, to the last decimal place, the samples of a 1 Hz sinusoid. Nothing in the sampled data can tell you which one was there. A frequency $f$ above the Nyquist frequency reappears at the alias frequency $|f - k f_s|$, for the integer $k$ that brings it into the range $[0, f_s/2]$; here $|9 - 10| = 1$ Hz.

    This has a practical consequence that is easy to miss: **aliasing cannot be removed after sampling**. Once the 9 Hz has become 1 Hz, no digital filter can separate it from a genuine 1 Hz. That is why acquisition systems have an analog *anti-aliasing* low-pass filter before the analog-to-digital converter, and why the sampling frequency has to be chosen before the experiment, not fixed afterwards.

    **Guiding questions 2.**

    1. A marker on the heel during running has relevant content up to about 15 Hz. What is the minimum sampling frequency? Would you sample at exactly that rate?
    2. A film of a spinning wheel can show it turning slowly backwards. Explain it with aliasing. (See also Challenge 1 of [Angular kinematics in a plane](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb).)
    3. At what frequency would an 11 Hz sinusoid appear if sampled at 10 Hz? And a 10 Hz one?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analog and digital signals

    The amplitude of an [analog signal](https://en.wikipedia.org/wiki/Analog_signal) can take on any value (an infinite number of possible values) in a continuous range. The amplitude of a [digital signal](https://en.wikipedia.org/wiki/Digital_signal) can take only certain values (a finite number of possible values). For instance, a binary signal can take only two values.

    The continuous and discrete properties refer to the independent variable (for instance, time); the analog and digital properties refer to the amplitude of the dependent variable (the signal itself). A recorded signal is both discrete and digital: sampled in time and quantized in amplitude.

    ### Quantization

    The reduction of an analog signal to a digital signal is called <a href="https://en.wikipedia.org/wiki/Quantization_(signal_processing)">quantization</a>, and in measurement it is typically performed by an [analog-to-digital (A/D) converter](https://en.wikipedia.org/wiki/Analog-to-digital_converter). The number of discrete values that the converter can output over its input range is its resolution. Because the output is stored as a binary number, the resolution is expressed in bits and the number of levels is a power of two. A resolution of 1 bit encodes the analog input into one of 2 ($2^1$) levels, 4 bits into one of 16 ($2^4$) levels, and a good commercial A/D converter, with 16 bits, into one of 65,536 ($2^{16}$) levels.

    If we know the voltage range of the A/D converter (the maximum minus the minimum voltage it can read), we can express the resolution in volts, as the size of one step: the range divided by the number of levels. For a typical range of 10 V, from $-5$ to $5$ V, the step is 5 V for 1 bit, 0.625 V for 4 bits, and about 0.00015 V (0.15 mV) for 16 bits.

    Let's write an idealized converter. It divides its range into $2^{n}$ steps of equal size and replaces each value by the centre of the step it falls in; values outside the range are clipped to the first or last step.
    """)
    return


@app.function
def quantize(x, n_bits, v_range):
    """Quantize `x` with an ideal `n_bits` A/D converter of range `v_range`.

    The range is centred at zero, [-v_range/2, v_range/2], and divided into
    2**n_bits steps of equal size; each value is replaced by the centre of its
    step. Values outside the range are clipped to the first or last step.
    """
    import numpy as np

    step = v_range / 2**n_bits
    level = np.clip(np.floor((np.asarray(x) + v_range / 2) / step), 0, 2**n_bits - 1)
    return -v_range / 2 + (level + 0.5) * step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Before you run the next cell**, predict for a sinusoid of unit amplitude quantized with a range of 2 (from $-1$ to $1$): how many distinct values will the 1-bit and the 4-bit versions have, and what is the largest error each one makes?
    """)
    return


@app.cell
def _(np, plt):
    _t = np.linspace(0, 1, 101)
    _x = np.sin(2 * np.pi * _t)

    _fig, _axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for _ax, _bits in zip(_axs, (1, 4, 16)):
        _xq = quantize(_x, _bits, 2)
        _ax.plot(_t, _x, linewidth=2, color=[0, 0, 1, 0.3])
        _ax.plot(_t, _xq, "r.-", linewidth=1, drawstyle="steps-mid")
        _ax.set_title(f"{_bits}-bit resolution")
        _ax.set_xlabel("Time [s]")
        _ax.set_ylim((-1.1, 1.1))
        print(
            f"{_bits:2d} bits: step = {2 / 2**_bits:.6f}, distinct values = "
            f"{np.unique(_xq).size:3d}, largest error = {np.max(np.abs(_xq - _x)):.6f}"
        )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With 1 bit the sinusoid becomes a square wave with two values, $\pm 0.5$, and the error reaches half a step, 0.5. With 4 bits the staircase is visible but already follows the curve, and the error never exceeds half a step, 0.0625. With 16 bits the staircase is invisible at this scale. Note that the 16-bit version shows only about 50 distinct values, not 65,536: there are only 101 samples, and many fall on the same level. The resolution is how many values the converter *can* output, not how many a given recording uses.

    ### Back to the EMG: reading the converter off the data

    Now we can answer Guiding question 0.2. The steps in the zoomed EMG are quantization. If the smallest difference between two distinct values in the file is the step of the converter, we can work out its resolution.

    **Before you run the next cell**, predict the number of distinct values in the EMG file, which has 3360 samples.
    """)
    return


@app.cell
def _(emg, np):
    _levels = np.unique(emg)
    _step = np.min(np.diff(_levels))
    print(f"Distinct values in the file: {_levels.size}")
    print(f"Smallest step between them:  {_step:.5f}")
    for _bits in (8, 12, 16):
        print(f"  step of a {_bits:2d}-bit converter with a 10 V range: {10 / 2**_bits:.5f}")
    print(f"Range used by the EMG: from {emg.min():.4f} to {emg.max():.4f}, "
          f"{100 * (emg.max() - emg.min()) / 10:.1f}% of a 10 V range")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Of 3360 samples, only 82 different values. The smallest step, 0.00244, is exactly the step of a 12-bit converter with a 10 V range, $10/2^{12}$. (The file does not say so; this is the most likely reading if its values are in volts at the converter's input.) The other steps between levels are whole multiples of it.

    So a 12-bit converter, with 4096 levels available, recorded this muscle with fewer than a hundred, because the signal used only a few percent of the converter's range. The amplification before the converter was too low for this recording: the resolution of a measurement depends both on the converter and on how much of its range the signal fills.

    **Guiding questions 3.**

    1. If the amplifier gain had been ten times larger, how many levels would the EMG have used? What could go wrong if it had been a hundred times larger?
    2. With the same gain, how many levels would a 16-bit converter have used?
    3. Go back to the guess you made in Challenge 0 about the number of distinct values. Was it about the converter or about the recording?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deterministic and random signals

    A deterministic signal can be described by an explicit mathematical function; a random signal cannot. A sine wave and a parabola are deterministic. The outcomes of tossing a coin or throwing a die are random: although the set of possible values is limited and known, no mathematical function predicts the exact outcome, only its probability. Most measurements of observed phenomena in nature contain both deterministic and random parts.

    Since, by definition, there is no explicit mathematical function to generate a random signal, computers cannot generate truly random data either. What they produce are *pseudorandom* numbers, from a deterministic algorithm started at a seed. For true randomness, one needs to extract data from nature and inject it into the computer; read more in [Introduction to Randomness and Random Numbers](https://www.random.org/randomness/).

    The determinism is easy to see, and it is useful: **before you run the next cell**, predict whether two generators started with the same seed produce the same numbers.
    """)
    return


@app.cell
def _(np):
    _rng1 = np.random.default_rng(seed=42)
    _rng2 = np.random.default_rng(seed=42)
    print("Generator 1:", np.round(_rng1.standard_normal(5), 4))
    print("Generator 2:", np.round(_rng2.standard_normal(5), 4))
    print("New seed:   ", np.round(np.random.default_rng(seed=7).standard_normal(5), 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Identical. Fixing the seed is how a simulation with "random" noise is made reproducible, and the notebooks in this collection do it whenever they add noise to a signal.

    ## Signal and noise

    With respect to what is being measured, data can be classified as signal or noise. <a href="https://en.wikipedia.org/wiki/Signal_(electrical_engineering)">Signal</a> is the part of the data you want, the part you believe truly represents the phenomenon being measured (or modeled). [Noise](https://en.wikipedia.org/wiki/Noise) is what you don't want in a measurement because, in principle, it has no significant role in the understanding of the observed phenomenon.

    As tautological as this sounds, the distinction between noise and signal depends on what you think is relevant about the phenomenon. The power-line interference in an EMG is noise to a physiologist and signal to the engineer looking for a bad ground connection. Experimental data are contaminated by noise, and it is rarely possible to completely separate signal from noise.

    The [signal-to-noise ratio](https://en.wikipedia.org/wiki/Signal-to-noise_ratio) (SNR or S/N) quantifies the level of a desired signal relative to the background noise. It is the ratio between the power of the signal and the power of the noise, often expressed in decibels:

    $$
    SNR = \frac{P_{signal}}{P_{noise}}, \qquad SNR_{dB} = 10\log_{10}\left(\frac{P_{signal}}{P_{noise}}\right)
    $$

    **Before you run the next cell**, predict the SNR of a sinusoid of amplitude 1 plus random noise with a standard deviation of 0.1. The power of a signal with zero mean is its mean squared value, which for the sinusoid you computed in the section on AC and DC components.
    """)
    return


@app.cell
def _(np, plt):
    _rng = np.random.default_rng(seed=42)
    _t = np.arange(0, 2, 0.001)
    _s = np.sin(2 * np.pi * 2 * _t)
    _n = 0.1 * _rng.standard_normal(_t.size)

    _snr = np.mean(_s**2) / np.mean(_n**2)
    print(f"Signal power: {np.mean(_s**2):.3f}, noise power: {np.mean(_n**2):.4f}")
    print(f"SNR = {_snr:.0f} = {10 * np.log10(_snr):.1f} dB")

    plt.figure(figsize=(10, 3))
    plt.plot(_t, _s + _n, color="tab:red", linewidth=1, label="signal + noise")
    plt.plot(_t, _s, "k", linewidth=2, label="signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An SNR of about 50, or 17 dB: the sinusoid has 50 times the power of the noise, although the noise is plainly visible. The power ratio, not the look of the plot, is what changes dramatically when data are differentiated, as you will see in [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb).

    ## Signal and system

    In engineering, a signal is often associated with a system, an entity (a physical or virtual device) that takes a signal as input and produces another signal as output. In mathematics, if a signal can be represented as a function, a system can be represented by a differential equation. For instance, a mass attached to a spring is a system that takes a force (a signal) as input and produces a displacement (another signal); it can be described by a second-order ordinary differential equation. A filter, the subject of the next notebook, is also a system.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to the two guesses in Challenge 0. Which of them is set by the sampling frequency and which by the resolution of the converter?
    2. A sinusoid and its sum with a constant have the same AC component. Do they have the same RMS?
    3. Why can aliasing not be fixed after the data are recorded, while noise can at least be attenuated?
    4. A force plate is sampled at 1000 Hz with a 16-bit converter. A colleague proposes saving disk space by keeping every tenth sample. What is lost, and when would it not matter?
    5. An EMG uses only 2% of the range of its converter. Name two ways to improve its resolution, and the risk of each.
    6. Give an example from your own field of a component of the data that is noise in one study and signal in another.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Plot the following functions in the interval $t=[-1, 1]$:<br>
       a. $x(t) = 1 + 0.5\cos(t + \pi/4)$<br>
       b. $x(t) = |t|$<br>
       c. $x(t) = 2t^3$<br>
       d. $x(t) = \sin(2\pi t) + \cos(2\pi t)$<br>
       e. $x(t) = \sin^2(2\pi t) + \cos^2(2\pi t)$<br>
       f. $x(t) = \sin(2\pi t)/t$

    2. Plot the following periodic function in the interval $t=[-3\pi, 3\pi]$:

       $$
       x(t) = \left\{
       \begin{array}{l l}
           1+t/\pi & \quad \text{if } -\pi < t < 0\\
           1-t/\pi & \quad \text{if } 0 < t < +\pi
       \end{array} \right.
       \qquad
       x(t + 2\pi) = x(t), \quad \text{for all } t
       $$

    3. What are the amplitude, frequency, period, and phase of the periodic functions in problems 1 and 2?

    4. Calculate the AC and DC components of the functions in problems 1 and 2.

    5. Which functions in problems 1 and 2 are even, and which are odd?

    6. The [power line frequency](https://en.wikipedia.org/wiki/Utility_frequency) in Brazil is 60 Hz (the same as in the US; in Europe it is 50 Hz; [see an online measurement of the frequency in Europe](https://www.mainsfrequency.com/)). The power line voltage is usually specified as 110 V or 127 V RMS. For a discrete signal, RMS is given by:

       $$
       RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
       $$

       and for a continuous periodic signal with period $T$:

       $$
       RMS = \sqrt{\frac{1}{T}\int_{t_0}^{t_0+T} x(t)^2 \:\mathrm{d}t}
       $$

       If the power line waveform is a sinusoid, what is the amplitude that results in 110 V RMS?

    7. Calculate the average power and the RMS values of the signals in problem 1.

    8. Consider the continuous signal represented by the function $x(t) = 2\sin(8\pi t)$.<br>
       a. What is the minimum sampling frequency, $f_N$, that satisfies the Nyquist-Shannon theorem for this signal?<br>
       b. Plot discretized versions of this signal for the following sampling frequencies: $10f_N$, $2f_N$, $f_N$, and $f_N/2$. In which of them do you recognize the original signal?

    9. In scientific computing, it's common to use a random number generator to generate noise and add it to a signal. For example, the following code generates a sinusoid (signal) plus random noise:

       ```python
       rng = np.random.default_rng(seed=42)
       t = np.arange(0, 1, 0.01)
       x = np.sin(2 * 2 * np.pi * t) + rng.standard_normal(t.size) / 10
       ```

       Plot this function and play with different levels of noise. For each, compute the SNR in decibels.

    10. Write a Python function that, for a given input signal, calculates its average value, peak-to-peak amplitude, average power, and RMS value. Apply it to the EMG above, before and after removing its mean.

    11. Quantize the EMG with `quantize`, using a 10 V range and 8 bits. How many distinct values remain, and does the signal still look like an EMG?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Fourier series](https://github.com/BMClab/BMC/blob/master/notebooks/FourierSeries.ipynb) — building periodic signals from a fundamental and its harmonics.
    - [Fourier transform](https://github.com/BMClab/BMC/blob/master/notebooks/FourierTransform.ipynb) — describing any signal by its frequency content.
    - [Data filtering](https://github.com/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb) — attenuating noise, and why differentiation makes it worse.
    - [Residual analysis](https://github.com/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb) — choosing the cutoff frequency of a filter from the data.
    - [Electromyography](https://github.com/BMClab/BMC/blob/master/notebooks/Electromyography.ipynb) — processing the kind of signal used throughout this notebook.
    - [dspGuru - Digital Signal Processing Central](https://www.dspguru.com/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Bendat JS, Piersol AG (2010) [Random Data: Analysis and Measurement Procedures](https://books.google.com.br/books?id=qYSViFRNMlwC). 4th Edition. John Wiley & Sons, Inc.
    - Lathi BP (2009) [Linear Systems and Signals](https://books.google.com.br/books?id=JC18PwAACAAJ). Oxford University Press.
    - Lyons RG (2010) [Understanding Digital Signal Processing](https://books.google.com.br/books?id=UBU7Y2tpwWUC&hl). 3rd edition. Prentice Hall.
    - Smith SW (1997) [The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/). California Technical Pub.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
