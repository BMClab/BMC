import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Análise dos dados da trajetória da bola: filtragem
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)


@app.cell
def _():
    import matplotlib  # data visualization
    import matplotlib.pyplot as plt  # data visualization
    import numpy as np  # algorithms and convenience functions for scientific computing with Python
    import pandas as pd  # labelled tables with numeric and string data
    import scipy as sp  # collection of mathematical algorithms and convenience functions built on NumPy
    import seaborn as sns  # data visualization
    import sympy  # symbolic programming
    from watermark import watermark  # date, version numbers and hardware information

    print(
        watermark(
            updated=True,
            current_time=True,
            current_date=True,
            machine=True,
            python=True,
            iversions=True,
            globals_={
                "numpy": np,
                "pandas": pd,
                "matplotlib": matplotlib,
                "seaborn": sns,
                "sympy": sympy,
                "scipy": sp,
            },
        )
    )
    return np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Configuração do ambiente
    """)


@app.cell
def _(sns):
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})
    sns.set_style("whitegrid")
    colors = sns.color_palette()

    # Parâmetros
    g = -9.8  # aceleração de gravidade, m/s2

    colors
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Importar dados
    """)


@app.cell
def _(pd):
    # importar dados da internet
    url = (
        "https://raw.githubusercontent.com/BMClab/BMC/refs/heads/master/data/dados.txt"
    )
    dados = pd.read_csv(url, skiprows=1, usecols=[0, 1, 2], sep=",")
    print(f"tamanho dos dados: {dados.shape}")
    dados
    return (dados,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotar dados
    """)


@app.cell
def _(dados, plt):
    dados.plot(
        x="t",
        y=["x", "y"],
        style=[".", "."],
        figsize=(8, 3),
        xlabel="Time [s]",
        title="Position [m]",
        subplots=True,
    )
    plt.show()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Análise dos dados
    """)


@app.cell
def _():
    freq = 120  # Hz
    # freq = 1 / np.mean(np.diff(dados.values[:, 0]))
    return (freq,)


@app.cell
def _(mo, np):
    mo.doc(np.gradient)


@app.cell
def _(dados, freq, np):
    # calculate velocity
    vx = np.gradient(dados.x, 1 / freq)
    vy = np.gradient(dados.y, 1 / freq)
    # calculate acceleration
    ax = np.gradient(vx, 1 / freq)
    ay = np.gradient(vy, 1 / freq)
    return ax, ay, vx, vy


@app.cell
def _(ax, ay, dados, vx, vy):
    dados_deriv = dados.assign(vx=vx, vy=vy, ax=ax, ay=ay)
    dados_deriv
    return (dados_deriv,)


@app.cell
def _(dados_deriv, plt):
    dados_deriv.plot(
        x="t",
        y=["vx", "vy"],
        style=[".", "."],
        xlabel="Time [s]",
        title="Velocity [m/s]",
        subplots=True,
    )
    plt.show()


@app.cell
def _(dados_deriv, plt):
    dados_deriv.plot(
        x="t",
        y=["ax", "ay"],
        style=[".", "."],
        xlabel="Time [s]",
        title="Acceleration [m/s$^2$]",
        subplots=True,
    )
    plt.show()


@app.cell
def _(dados_deriv):
    dados_deriv.mean()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dados são muito ruidosos para estimar os valores da velocidade e aceleração...

    ## O que fazer?

    ## Uma possível solução: filtrar dados

    *   Data filtering in signal processing:  https://colab.research.google.com/github/BMClab/BMC/blob/master/notebooks/DataFiltering.ipynb
    *   Residual analysis to determine the optimal cutoff frequency:  https://colab.research.google.com/github/BMClab/BMC/blob/master/notebooks/ResidualAnalysis.ipynb
    *   Revisão: Basic properties of signals:  https://colab.research.google.com/github/BMClab/BMC/blob/master/notebooks/SignalBasicProperties.ipynb
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Duas estratégias para filtrar

    Vamos usar um filtro Butterworth passa-baixas de 2ª ordem, aplicado duas vezes (para frente e para trás, com `filtfilt`), o que não introduz atraso (*zero-lag*). Mas há duas maneiras de combiná-lo com as derivadas:

    - **A. Filtrar a posição, depois derivar:** filtrar $y$ uma única vez e então calcular $v_y$ e $a_y$ a partir da posição filtrada.
    - **B. Derivar e filtrar a cada passo:** calcular $v_y$ a partir da posição sem filtrar, filtrar $v_y$, calcular $a_y$ e filtrar $a_y$ de novo.

    Um detalhe: a cada passagem, o filtro atenua a potência na frequência de corte pela metade, e com duas passagens (para frente e para trás) ela seria atenuada a um quarto. Por isso a frequência de corte informada ao filtro é corrigida pelo fator

    $$
    C = \left(2^{1/n_{passagens}} - 1\right)^{1/(2\,ordem)} \approx 0.802 \quad \text{(2ª ordem, 2 passagens)}
    $$

    para que o filtro final tenha de fato a frequência de corte desejada.
    """)


@app.function
def filtra(x, freq, fc, ordem=2):
    """Filtro Butterworth passa-baixas com atraso zero (filtfilt).

    A frequência de corte é corrigida para as duas passagens do filtro, de
    modo que a frequência de corte final seja `fc`.
    """
    from scipy import signal

    C = (2 ** (1 / 2) - 1) ** (1 / (2 * ordem))  # 0.802 para 2ª ordem
    b, a = signal.butter(ordem, (fc / C) / (freq / 2), btype="low")
    return signal.filtfilt(b, a, x)


@app.function
def estrategia_a(y, freq, fc):
    """A: filtra a posição uma vez, depois deriva. Retorna (yf, vy, ay).

    As derivadas usam o intervalo de amostragem 1/freq.
    """
    import numpy as np

    yf = filtra(y, freq, fc)
    vy = np.gradient(yf, 1 / freq)
    ay = np.gradient(vy, 1 / freq)
    return yf, vy, ay


@app.function
def estrategia_b(y, freq, fc):
    """B: deriva e filtra a velocidade, deriva e filtra a aceleração. Retorna (vy, ay).

    As derivadas usam o intervalo de amostragem 1/freq.
    """
    import numpy as np

    vy = filtra(np.gradient(y, 1 / freq), freq, fc)
    ay = filtra(np.gradient(vy, 1 / freq), freq, fc)
    return vy, ay


@app.function
def compara_estrategias(dados, freq, fc, g=-9.8, trecho=slice(50, -20)):
    """Plota a velocidade e a aceleração vertical pelas estratégias A e B.

    Também imprime a média da aceleração vertical no `trecho` dos dados.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    t, y = dados.t.to_numpy(), dados.y.to_numpy()
    _, vy_a, ay_a = estrategia_a(y, freq, fc)
    vy_b, ay_b = estrategia_b(y, freq, fc)
    vy_bruto = np.gradient(y, 1 / freq)
    ay_bruto = np.gradient(vy_bruto, 1 / freq)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    axs[0].plot(t, vy_bruto, ".", color="0.7", label="sem filtro")
    axs[0].plot(t, vy_a, "-", linewidth=2, label="A: filtra a posição")
    axs[0].plot(t, vy_b, "--", linewidth=2, label="B: filtra a cada derivada")
    axs[0].set_ylabel("vy [m/s]")
    axs[0].legend(loc="best")
    axs[1].plot(t, ay_bruto, ".", color="0.7")
    axs[1].plot(t, ay_a, "-", linewidth=2)
    axs[1].plot(t, ay_b, "--", linewidth=2)
    axs[1].axhline(g, color="k", linestyle=":", label=f"g = {g} m/s$^2$")
    axs[1].axvspan(
        t[trecho][0], t[trecho][-1], color="0.9", zorder=0, label="trecho da média"
    )
    axs[1].set_ylim(3 * g, -g)
    axs[1].set_ylabel("ay [m/s$^2$]")
    axs[1].set_xlabel("Time [s]")
    axs[1].legend(loc="best")
    fig.suptitle(f"Velocidade e aceleração verticais, filtro com fc = {fc:.1f} Hz")
    plt.tight_layout()
    plt.show()

    print(f"Média de ay no trecho, A: {np.mean(ay_a[trecho]):6.2f} m/s2")
    print(f"Média de ay no trecho, B: {np.mean(ay_b[trecho]):6.2f} m/s2")
    print(f"Média de ay no trecho, sem filtro: {np.mean(ay_bruto[trecho]):6.2f} m/s2")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore: escolha a frequência de corte

    Mova o controle abaixo e observe as duas estratégias. O trecho sombreado é usado para calcular a média da aceleração (sem os extremos dos dados, onde os filtros e as derivadas são menos confiáveis).
    """)


@app.cell
def _(mo):
    fc_slider = mo.ui.slider(
        start=1,
        stop=40,
        step=0.5,
        value=10,
        show_value=True,
        label="Frequência de corte [Hz]",
    )
    fc_slider
    return (fc_slider,)


@app.cell
def _(dados, fc_slider, freq, g):
    compara_estrategias(dados, freq, fc_slider.value, g=g)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Para explorar:**

    1. Com que frequência de corte a aceleração fica razoavelmente constante? E a velocidade?
    2. Para uma mesma frequência de corte, qual das estratégias, A ou B, suaviza mais a aceleração? Por quê? (Dica: quantas vezes cada uma passa o filtro pelos dados?)
    3. O que acontece nos extremos dos dados, no início e no fim do movimento?
    4. Com frequências de corte muito baixas, a média da aceleração ainda se aproxima de $g$? E a forma da curva?
    5. As derivadas acima usam o intervalo de amostragem $1/120$ s. A coluna `t` dos dados está arredondada para o milissegundo, e seus intervalos alternam entre 0.008 e 0.009 s. Troque `1 / freq` por `dados.t` em `np.gradient` (nas funções `estrategia_a` e `estrategia_b`) e veja o que acontece com a aceleração. Por que a estratégia A é mais afetada?
    6. Como escolher a frequência de corte sem saber de antemão qual deveria ser a aceleração? Veja a seguir.
    """)


@app.function
def optcutfreq(y, freq=1, fclim=[], show=False, ax=None):
    """Automatic search of optimal filter cutoff frequency based on residual analysis.

    This method was proposed by Winter in his book [1]_.
    The 'optimal' cutoff frequency (in the sense that a filter with such cutoff
    frequency removes as much noise as possible without considerably affecting
    the signal) is found by performing a residual analysis of the difference
    between filtered and unfiltered signals over a range of cutoff frequencies.
    The optimal cutoff frequency is the one where the residual starts to change
    very little because it is considered that from this point, it's being
    filtered mostly noise and minimally signal, ideally.

    Parameters
    ----------
    y : 1D array_like
        Data
    freq : float, optional (default = 1)
        sampling frequency of the signal y
    fclim : list with 2 numbers, optional (default = [])
        limit frequencies of the noisy part or the residuals curve
    show : bool, optional (default = False)
        True (1) plots data in a matplotlib figure
        False (0) to not plot
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    fc_opt : float
             optimal cutoff frequency (None if not found)

    Notes
    -----
    A second-order zero-phase digital Butterworth low-pass filter is used.
    # The cutoff frequency is corrected for the number of passes:
    # C = (2**(1/npasses) - 1)**0.25. C = 0.802 for a dual pass filter.

    The matplotlib figure with the results will show a plot of the residual
    analysis with the optimal cutoff frequency, a plot with the unfiltered and
    filtered signals at this optimal cutoff frequency (with the RMSE of the
    difference between these two signals), and a plot with the respective
    second derivatives of these signals which should be useful to evaluate
    the quality of the optimal cutoff frequency found.

    Winter should not be blamed for the automatic search algorithm used here.
    The algorithm implemented is just to follow as close as possible Winter's
    suggestion of fitting a regression line to the noisy part of the residuals.

    This function performs well with data where the signal has frequencies
    considerably below the Nyquist frequency and the noise is predominantly
    white in the higher frequency region.

    If the automatic search fails, the lower and upper frequencies of the noisy
    part of the residuals curve can be input as a parameter (fclim).
    These frequencies can be chosen by viewing the plot of the residuals (enter
    show=True as input parameter when calling this function).

    It is known that this residual analysis algorithm results in oversmoothing
    kinematic data [2]_. Use it with moderation.
    This code is described elsewhere [3]_.

    References
    ----------
    .. [1] Winter DA (2009) Biomechanics and motor control of human movement.
    .. [2] http://www.clinicalgaitanalysis.com/faq/cutoff.html
    .. [3] https://github.com/demotu/optcutfreq/blob/master/docs/optcutfreq.ipynb

    Examples
    --------
    >>> y = np.cumsum(np.random.randn(1000))
    >>> # optimal cutoff frequency based on residual analysis and plot:
    >>> fc_opt = optcutfreq(y, freq=1000, show=True)
    >>> # same analysis but specifying the frequency limits and plot:
    >>> optcutfreq(y, freq=1000, fclim=[200,400], show=True)
    >>> # It's not always possible to find an optimal cutoff frequency
    >>> # or the one found can be wrong (run this example many times):
    >>> y = np.random.randn(100)
    >>> optcutfreq(y, freq=100, show=True)

    """
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import butter, filtfilt

    y = np.asarray(y)
    # Correct the cutoff frequency for the number of passes in the filter
    C = 0.802  # for dual pass; C = (2**(1/npasses)-1)**0.25

    # signal filtering
    freqs = np.linspace((freq / 2) / 100, (freq / 2) * C, 101, endpoint=False)
    res = []
    for fc in freqs:
        b, a = butter(2, (fc / C) / (freq / 2))
        yf = filtfilt(b, a, y)
        # residual between filtered and unfiltered signals
        res = np.hstack((res, np.sqrt(np.mean((yf - y) ** 2))))

    # find the optimal cutoff frequency by fitting an exponential curve
    # y = A*exp(B*x)+C to the residual data and consider that the tail part
    # of the exponential (which should be the noisy part of the residuals)
    # decay starts after 3 lifetimes (exp(-3), 95% drop)
    fclim = np.asarray(fclim)
    if not len(fclim) or np.any(fclim < 0) or np.any(fclim > freq / 2):
        fc1 = 0
        fc2 = int(0.95 * (len(freqs) - 1))
        # log of exponential turns the problem to first order polynomial fit
        # make the data always greater than zero before taking the logarithm
        reslog = np.log(
            np.abs(res[fc1 : fc2 + 1] - res[fc2]) + 1000 * np.finfo(float).eps
        )
        Blog, Alog = np.polyfit(freqs[fc1 : fc2 + 1], reslog, 1)
        fcini = np.nonzero(freqs >= -3 / Blog)  # 3 lifetimes
        fclim = [fcini[0][0], fc2] if np.size(fcini) else []
    else:
        fclim = [
            np.nonzero(freqs >= fclim[0])[0][0],
            np.nonzero(freqs >= fclim[1])[0][0],
        ]

    # find fc_opt with linear fit y=A+Bx of the noisy part of the residuals
    B = A = None
    if len(fclim) and fclim[0] < fclim[1]:
        B, A = np.polyfit(freqs[fclim[0] : fclim[1]], res[fclim[0] : fclim[1]], 1)
        # optimal cutoff frequency is the frequency where y[fc_opt] = A
        roots = UnivariateSpline(freqs, res - A, s=0).roots()
        fc_opt = roots[0] if len(roots) else None
    else:
        fc_opt = None

    if show:
        plot_optcutfreq(y, freq, freqs, res, fclim, fc_opt, B, A, ax)

    return fc_opt


@app.function
def plot_optcutfreq(y, freq, freqs, res, fclim, fc_opt, B, A, ax):
    """Plot results of the optcutfreq function, see its help."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import butter, filtfilt

    if ax is None:
        plt.figure(num=None, figsize=(10, 5))
        ax = np.array([plt.subplot(121), plt.subplot(222), plt.subplot(224)])

    plt.rc("axes", labelsize=12, titlesize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    ax[0].plot(freqs, res, "b.", markersize=9)
    time = np.linspace(0, len(y) / freq, len(y))
    ax[1].plot(time, y, "g", linewidth=1, label="Unfiltered")
    ydd = np.diff(y, n=2) * freq**2
    ax[2].plot(time[:-2], ydd, "g", linewidth=1, label="Unfiltered")
    if fc_opt:
        ylin = np.poly1d([B, A])(freqs)
        ax[0].plot(freqs, ylin, "r--", linewidth=2)
        ax[0].plot(
            freqs[fclim[0]],
            res[fclim[0]],
            "r>",
            freqs[fclim[1]],
            res[fclim[1]],
            "r<",
            ms=9,
        )
        ax[0].set_ylim(bottom=0, top=4 * A)
        ax[0].plot([0, freqs[-1]], [A, A], "r-", linewidth=2)
        ax[0].plot([fc_opt, fc_opt], [0, A], "r-", linewidth=2)
        ax[0].plot(
            fc_opt,
            0,
            "ro",
            markersize=7,
            clip_on=False,
            zorder=9,
            label="$Fc_{opt}$ = %.1f Hz" % fc_opt,
        )
        ax[0].legend(fontsize=12, loc="best", numpoints=1, framealpha=0.5)
        # Correct the cutoff frequency for the number of passes
        C = 0.802  # for dual pass; C = (2**(1/npasses) - 1)**0.25
        b, a = butter(2, (fc_opt / C) / (freq / 2))
        yf = filtfilt(b, a, y)
        ax[1].plot(time, yf, color=[1, 0, 0, 0.5], linewidth=2, label="Opt. filtered")
        ax[1].legend(fontsize=12, loc="best", framealpha=0.5)
        ax[1].set_title("Signals (RMSE = %.3g)" % A)
        yfdd = np.diff(yf, n=2) * freq**2
        ax[2].plot(
            time[:-2], yfdd, color=[1, 0, 0, 0.5], linewidth=2, label="Opt. filtered"
        )
        ax[2].legend(fontsize=12, loc="best", framealpha=0.5)
        resdd = np.sqrt(np.mean((yfdd - ydd) ** 2))
        ax[2].set_title("Second derivatives (RMSE = %.3g)" % resdd)
    else:
        ax[0].text(
            0.5,
            0.5,
            "Unable to find optimal cutoff frequency",
            horizontalalignment="center",
            color="r",
            zorder=9,
            transform=ax[0].transAxes,
            fontsize=12,
        )
        ax[1].set_title("Signal")
        ax[2].set_title("Second derivative")

    ax[0].set_xlabel("Cutoff frequency [Hz]")
    ax[0].set_ylabel("Residual RMSE")
    ax[0].set_title("Residual analysis")
    ax[0].grid()
    ax[1].set_xlim(0, time[-1])
    ax[1].grid()
    ax[2].set_xlabel("Time [s]")
    ax[2].set_xlim(0, time[-1])
    ax[2].grid()
    plt.tight_layout()
    plt.show()


@app.cell
def _(dados, freq):
    fc_opt = optcutfreq(dados.y, freq=freq, show=True)
    fc_opt
    return (fc_opt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A frequência de corte ótima foi calculada para a posição $y$, com o mesmo filtro usado nas duas estratégias (Butterworth de 2ª ordem, duas passagens, frequência corrigida). Vejamos as duas estratégias com ela:
    """)


@app.cell
def _(dados, fc_opt, freq, g):
    compara_estrategias(dados, freq, fc_opt, g=g)


@app.cell
def _(dados, fc_opt, freq):
    _yf, _vy_a, _ay_a = estrategia_a(dados.y, freq, fc_opt)
    _vy_b, _ay_b = estrategia_b(dados.y, freq, fc_opt)
    dados_fopt = dados.assign(yf=_yf, vy_A=_vy_a, ay_A=_ay_a, vy_B=_vy_b, ay_B=_ay_b)
    dados_fopt
    return (dados_fopt,)


@app.cell
def _(dados_fopt):
    dados_fopt.iloc[50:-20,].mean()


@app.cell
def _(dados, np):
    # Model: y = y0 + v0*t + 1/2*g*t^2
    # fit a second order polynomial to the data
    p = np.polyfit(dados.t, dados.y, 2)
    print("g = %0.2f m/s2" % (2 * p[0]))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Referências

    - Welcome to Colab!, https://colab.research.google.com/
    - Tracker Video Analysis and Modeling Tool, http://physlets.org/tracker/
    - marimo, https://marimo.io/
    """)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
