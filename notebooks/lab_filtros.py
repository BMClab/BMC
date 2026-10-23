# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14", "optcutfreq>=0.0.10", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Uso de filtros — laboratório de Biomecânica I

    > Renato Naville Watanabe, Marcos Duarte<br>
    > [Laboratório de Biomecânica e Controle Motor](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Universidade Federal do ABC, Brasil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Como usar este roteiro

    Uma trajetória pode ficar visualmente mais suave e, ainda assim, produzir uma aceleração pior. Vamos usar os dados do laboratório de Biomecânica I de 2022 para comparar média móvel, filtragem causal, filtragem de fase zero, Butterworth e análise de resíduos.

    Execute as células na ordem da leitura e descreva o que espera antes de alterar uma frequência de corte. Resolva os **Exercícios** distribuídos pelo texto e abra suas soluções somente depois da tentativa. As **Questões de verificação** revisam as ideias; os **Problemas** finais têm apenas respostas finais recolhidas.

    **Desafio inicial.** Uma média móvel com quatro amostras e a aplicação dessa mesma média para frente e para trás produzem a mesma curva, apenas sem atraso? Faça sua previsão antes de observar os gráficos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparação em Python e protocolo da atividade

    Usaremos NumPy para vetores, pandas para a tabela, Matplotlib para os gráficos, SciPy para filtros e `optcutfreq` para a análise de resíduos. As dependências estão declaradas no próprio arquivo; não é preciso montar o Google Drive nem executar `pip install` dentro da aula.

    O exemplo usa uma cópia pública arquivada de `massa_A.txt`, fixada à revisão `5366ffa7` para continuar disponível após a remoção da pasta antiga. Primeiro reproduza os resultados com o arquivo da turma. Para comparar os filtros, mantenha o mesmo trecho de voo livre e confira os intervalos de amostragem antes de usar a grade uniforme preparada adiante. Compare atraso, atenuação e efeitos nas extremidades, além da aparência da curva: uma trajetória mais suave não garante uma derivada mais fiel. Se enviar suas próprias medidas, calibre a escala espacial e exporte tempo em segundos e posições x,y em metros. Contatos, quadros ausentes, paralaxe e erros de calibração precisam ser identificados; a filtragem não os transforma em ruído aleatório.

    O arquivo original tem uma linha de identificação, seguida do cabeçalho `t`, `x`, `y`, separado por tabulações. Você pode enviar outro arquivo no mesmo formato pelo controle abaixo. Ele será lido em memória; o arquivo original permanece preservado.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt, lfilter, sosfiltfilt
    from optcutfreq import optcutfreq

    return butter, filtfilt, lfilter, mo, np, optcutfreq, pd, plt, sosfiltfilt


@app.function
def repo_file(path, *, base_url="https://raw.githubusercontent.com/BMClab/BMC/master/"):
    """Find a repository file beside this notebook, or fetch a temporary copy.

    Local files take precedence. Only file acquisition falls back to GitHub;
    parsing errors remain visible to the calling notebook. urllib3 selects its
    browser transport automatically without platform-specific imports.
    """
    import hashlib
    import tempfile
    from pathlib import Path, PurePosixPath
    from urllib.parse import quote

    import marimo as mo
    import urllib3

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise ValueError("Use a repository-relative path such as data/walk.trc")
    location = mo.notebook_dir()
    if location is not None:
        directory = Path(location).resolve()
        for base in (directory, *directory.parents):
            local = base.joinpath(*relative.parts)
            if local.is_file():
                return local
    url = base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")
    key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]
    cached = (Path(tempfile.gettempdir()) / "bmc-notebooks-v2" / key).joinpath(*relative.parts)
    if not cached.is_file():
        response = urllib3.request("GET", url, timeout=30.0, retries=2)
        if response.status != 200:
            raise OSError(f"Cannot load {path}: HTTP {response.status} from {url}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as download:
            temporary = Path(download.name)
            download.write(response.data)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    return cached


@app.cell
def _(mo):
    arquivo = mo.ui.file(label="Opcional: enviar tabela de posições (.txt)")
    arquivo
    return (arquivo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Abertura do arquivo

    Observe a tabela e depois a trajetória no plano x–y. Esses dois gráficos respondem a perguntas diferentes: a trajetória mostra onde o objeto passou; as séries temporais mostram quando ele passou. A unidade de uma derivada depende das unidades da tabela.
    """)
    return


@app.cell
def _(arquivo, np, pd):
    from io import BytesIO
    _conteudo = arquivo.contents()
    _fonte = BytesIO(_conteudo) if _conteudo is not None else repo_file("courses/biomechanicsI_2022/massa_A.txt", base_url="https://raw.githubusercontent.com/BMClab/BMC/5366ffa7e12fc61b3c2410c173128e468a9fe524/")
    dado = pd.read_csv(_fonte, sep="\t", skiprows=1)
    if not {"t", "x", "y"}.issubset(dado.columns):
        raise ValueError("O arquivo precisa das colunas t, x e y após a linha de identificação.")
    dado = dado[["t", "x", "y"]].astype(float)
    if len(dado) < 30 or not np.isfinite(dado.to_numpy()).all():
        raise ValueError("Use pelo menos 30 amostras finitas, sem lacunas nas colunas.")
    tempo = dado["t"].to_numpy()
    x = dado["x"].to_numpy()
    y = dado["y"].to_numpy()
    if not np.all(np.diff(tempo) > 0):
        raise ValueError("Os tempos precisam ser estritamente crescentes.")
    dado
    return tempo, x, y


@app.cell
def _(plt, tempo, x, y):
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4))
    _ax[0].plot(x, y, ".-")
    _ax[0].set(xlabel="x (m)", ylabel="y (m)", title="Trajetória medida")
    _ax[0].set_aspect("equal", adjustable="datalim")
    _ax[1].plot(tempo, x, label="x")
    _ax[1].plot(tempo, y, label="y")
    _ax[1].set(xlabel="Tempo (s)", ylabel="Posição (m)")
    _ax[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Período de amostragem: confira todos os intervalos

    O roteiro original calculava `t[2]-t[1]` e usava esse único número no registro inteiro. Vamos verificar a hipótese antes de empregá-la. No arquivo da turma, o primeiro intervalo é 0,046 s e os seguintes alternam aproximadamente 0,008 e 0,009 s. Há uma lacuna inicial e arredondamento dos tempos; um único intervalo de 0,009 s não representa toda a aquisição.

    Para derivar, usaremos os próprios tempos. Para os filtros digitais, que supõem uma grade temporal uniforme, construiremos adiante uma grade explícita no trecho contínuo após a lacuna inicial.
    """)
    return


@app.cell
def _(np, pd, tempo):
    intervalos = np.diff(tempo)
    dt_original = tempo[2] - tempo[1]
    pd.DataFrame({"medida": ["primeiro intervalo", "intervalo usado no roteiro original", "menor intervalo", "maior intervalo"],
                  "tempo (s)": [intervalos[0], dt_original, intervalos.min(), intervalos.max()]})
    return (intervalos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 1.** Uma posição muda 0,018 m em 0,009 s. Qual é a velocidade média nesse intervalo? Se o intervalo fosse 0,008 s, o resultado seria igual?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solução do exercício 1': mo.md(r"""
    $v=\Delta x/\Delta t=2$ m/s no primeiro caso e 2,25 m/s no segundo. Dividir todos os incrementos por um intervalo errado introduz um erro de escala, mesmo sem ruído espacial.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derivada numérica com `for`, `np.diff` e `np.gradient`

    Para cada par de amostras, $ar v_i=(x_{i+1}-x_i)/(t_{i+1}-t_i)$. Existem N−1 intervalos para N posições. Não acrescente um zero artificial na última amostra: ele pareceria uma parada que não foi medida. Desenharemos essa velocidade no ponto médio do intervalo.
    """)
    return


@app.cell
def _(np, plt, tempo, x):
    dxdt_1 = np.zeros(len(tempo) - 1)
    for _i in range(len(tempo) - 1):
        dxdt_1[_i] = (x[_i + 1] - x[_i]) / (tempo[_i + 1] - tempo[_i])
    tempo_meio = (tempo[:-1] + tempo[1:]) / 2
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(tempo_meio, dxdt_1)
    _ax.set(xlabel="Tempo (s)", ylabel="Velocidade x (m/s)", title="Diferenças por intervalo: for")
    _fig
    return dxdt_1, tempo_meio


@app.cell
def _(dxdt_1, np, plt, tempo, tempo_meio, x):
    dxdt_2 = np.diff(x) / np.diff(tempo)
    np.testing.assert_allclose(dxdt_1, dxdt_2)
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(tempo_meio, dxdt_2)
    _ax.set(xlabel="Tempo (s)", ylabel="Velocidade x (m/s)", title="A mesma operação: np.diff")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `np.gradient(x, tempo, edge_order=2)` devolve N valores nos tempos originais: combina diferenças centrais no interior e fórmulas unilaterais nas bordas. Aceita coordenadas não uniformes. Isso não elimina o ruído e não transforma os resultados de borda em medidas exatas.
    """)
    return


@app.cell
def _(np, plt, tempo, x):
    dxdt_3 = np.gradient(x, tempo, edge_order=2)
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(tempo, dxdt_3)
    _ax.set(xlabel="Tempo (s)", ylabel="Velocidade x (m/s)", title="Derivada nos tempos originais: gradient")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 2.** Considere $x(t)=t^2$ em t=0, 1, 2 s. Quais velocidades aparecem nos dois intervalos com diferenças simples? E quais valores fornece a derivada exata nos três tempos?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solução do exercício 2': mo.md(r"""
    As diferenças dão 1 e 3 m/s, associadas aos tempos médios 0,5 e 1,5 s. A derivada exata $2t$ vale 0, 2 e 4 m/s. `gradient(..., edge_order=2)` recupera esses três valores para essa parábola sem ruído.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Velocidades, acelerações e estimativa de g

    Derive x e y duas vezes. Com y positivo para cima e resistência do ar desprezível, o voo livre ideal tem $a_y=-g$ e $a_x=0$. No ápice, a velocidade vertical é zero por um instante, mas a gravidade continua atuando.

    Primeiro reproduzimos o recorte original `[10:]`. Depois comparamos um recorte que também exclui dez amostras finais. Esse número é um exercício de sensibilidade, não uma regra universal para bordas ou para identificar voo livre.
    """)
    return


@app.cell
def _(np, plt, tempo, x, y):
    dxdt = np.gradient(x, tempo, edge_order=2)
    dydt = np.gradient(y, tempo, edge_order=2)
    d2xdt2 = np.gradient(dxdt, tempo, edge_order=2)
    d2ydt2 = np.gradient(dydt, tempo, edge_order=2)
    d2ydt2_cut = d2ydt2[10:]
    t_cut = tempo[10:]
    g_direto = -np.mean(d2ydt2[10:-10])
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3))
    _ax[0].plot(tempo, d2ydt2, label="Todas as amostras")
    _ax[1].plot(t_cut, d2ydt2_cut, label="Recorte original [10:]")
    for _a in _ax:
        _a.set(xlabel="Tempo (s)", ylabel="Aceleração y (m/s²)")
        _a.legend()
    _fig.tight_layout()
    _fig
    return (d2ydt2,)


@app.cell
def _(d2ydt2, np, pd):
    pd.DataFrame({"trecho": ["todo o registro", "recorte original [10:]", "recorte [10:-10]"],
                  "média de ay (m/s²)": [np.mean(d2ydt2), np.mean(d2ydt2[10:]), np.mean(d2ydt2[10:-10])]})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 3.** Duas estimativas de aceleração média vertical são −9,4 e −10,2 m/s², com y para cima. Quais estimativas de g correspondem a elas? Uma curva mais lisa garante a melhor estimativa?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solução do exercício 3': mo.md(r"""
    Os módulos estimados são 9,4 e 10,2 m/s². Suavidade não garante exatidão: o filtro pode reduzir a variância e ao mesmo tempo alterar curvatura, bordas e valor médio. A qualidade deve ser avaliada com o modelo, a calibração e a sensibilidade ao processamento.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uma grade uniforme para comparar filtros

    Conservamos todas as amostras na análise anterior. Somente para filtrar, retiramos a primeira amostra quando seu intervalo excede duas vezes a mediana e interpolamos linearmente o trecho restante em uma grade com o mesmo número de pontos e os mesmos extremos. No arquivo da turma isso dá aproximadamente 119,92 Hz. A interpolação trata o arredondamento dos tempos; não recupera informação perdida na lacuna inicial. Lacunas grandes no interior precisam ser tratadas antes desta análise.
    """)
    return


@app.cell
def _(intervalos, np, pd, tempo, x, y):
    inicio_filtro = 1 if intervalos[0] > 2 * np.median(intervalos) else 0
    _tt = tempo[inicio_filtro:]
    if np.max(np.diff(_tt)) > 2 * np.median(np.diff(_tt)):
        raise ValueError("Há lacunas internas grandes: selecione um trecho contínuo antes de filtrar.")
    tempo_uniforme = np.linspace(_tt[0], _tt[-1], len(_tt))
    dt = float(tempo_uniforme[1] - tempo_uniforme[0])
    fs = 1 / dt
    fn = fs / 2
    x_uniforme = np.interp(tempo_uniforme, tempo, x)
    y_uniforme = np.interp(tempo_uniforme, tempo, y)
    pd.DataFrame({"dt (s)": [dt], "fs (Hz)": [fs], "Nyquist (Hz)": [fn], "amostras": [len(tempo_uniforme)]})
    return dt, fs, tempo_uniforme, x_uniforme, y_uniforme


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Média móvel com `for` e `lfilter`

    A média das quatro amostras mais recentes é $ar x_i=(x_i+x_{i-1}+x_{i-2}+x_{i-3})/4$. Ela é causal e atrasa as componentes de baixa frequência em 1,5 amostra. O laço conserva os três primeiros valores sem filtrar, como no roteiro original. `lfilter`, sem estado inicial informado, começa com amostras anteriores iguais a zero. Compare o início antes de concluir que os algoritmos discordam.
    """)
    return


@app.cell
def _(plt, tempo_uniforme, x_uniforme):
    x_mm = x_uniforme.copy()
    for _i in range(3, len(x_uniforme)):
        x_mm[_i] = (x_uniforme[_i] + x_uniforme[_i - 1] + x_uniforme[_i - 2] + x_uniforme[_i - 3]) / 4
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(tempo_uniforme, x_uniforme, label="Original na grade uniforme")
    _ax.plot(tempo_uniforme, x_mm, label="Média móvel: for")
    _ax.set(xlabel="Tempo (s)", ylabel="x (m)")
    _ax.legend()
    _fig
    return (x_mm,)


@app.cell
def _(lfilter, np, plt, tempo_uniforme, x_mm, x_uniforme):
    b_ma = np.ones(4) / 4
    a_ma = [1.0]
    x_mm_2 = lfilter(b_ma, a_ma, x_uniforme)
    np.testing.assert_allclose(x_mm[3:], x_mm_2[3:])
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(tempo_uniforme, x_uniforme, label="Original")
    _ax.plot(tempo_uniforme, x_mm, label="for")
    _ax.plot(tempo_uniforme, x_mm_2, "--", label="lfilter, estado inicial zero")
    _ax.set(xlabel="Tempo (s)", ylabel="x (m)")
    _ax.legend()
    _fig
    return a_ma, b_ma, x_mm_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 4.** Para uma série constante de valor 8, quais são as quatro primeiras saídas de `lfilter([1/4]*4, [1], x)` sem estado inicial? Quando a saída coincide com a média móvel completa?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solução do exercício 4': mo.md(r"""
    As saídas são 2, 4, 6 e 8. O filtro inicialmente completa o passado com zeros; a partir da quarta amostra há quatro valores reais na janela. Para este FIR, o laço e `lfilter` coincidem do índice 3 em diante.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Média móvel com `filtfilt`: fase zero e outra resposta de amplitude

    `filtfilt` aplica o filtro para frente e para trás. Isso remove a fase no interior, mas eleva ao quadrado a resposta de magnitude: não é simplesmente deslocar a curva causal para a esquerda. O resultado depende de valores futuros e da extensão usada nas bordas. Compare x e y, como no roteiro original.
    """)
    return


@app.cell
def _(
    a_ma,
    b_ma,
    filtfilt,
    lfilter,
    plt,
    tempo_uniforme,
    x_mm,
    x_mm_2,
    x_uniforme,
    y_uniforme,
):
    x_mm_3 = filtfilt(b_ma, a_ma, x_uniforme)
    y_mm_2 = lfilter(b_ma, a_ma, y_uniforme)
    y_mm_3 = filtfilt(b_ma, a_ma, y_uniforme)
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4))
    _ax[0].plot(tempo_uniforme, x_uniforme, label="Original")
    _ax[0].plot(tempo_uniforme, x_mm, label="for")
    _ax[0].plot(tempo_uniforme, x_mm_2, label="lfilter")
    _ax[0].plot(tempo_uniforme, x_mm_3, label="filtfilt")
    _ax[1].plot(tempo_uniforme, y_uniforme, label="Original")
    _ax[1].plot(tempo_uniforme, y_mm_2, label="lfilter")
    _ax[1].plot(tempo_uniforme, y_mm_3, label="filtfilt")
    for _a, _nome in zip(_ax, ["x", "y"]):
        _a.set(xlabel="Tempo (s)", ylabel=f"{_nome} (m)")
        _a.legend()
    _fig.tight_layout()
    _fig
    return y_mm_2, y_mm_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filtro Butterworth

    Mantemos a comparação original: um Butterworth de ordem 3, corte de projeto em 10 Hz, aplicado nos dois sentidos. A implementação em seções de segunda ordem evita formar um único polinômio de ordem alta. Com `fs=fs`, a frequência de corte já é fornecida em hertz.

    A ordem efetiva depois das duas passagens é 6 e a magnitude no corte de projeto é aproximadamente −6 dB. As frequências 10 e 5 Hz são escolhas para esta atividade, não valores universais. A faixa de Nyquist limita as escolhas válidas.
    """)
    return


@app.cell
def _(
    butter,
    fs,
    plt,
    sosfiltfilt,
    tempo_uniforme,
    y_mm_2,
    y_mm_3,
    y_uniforme,
):
    fc_10 = min(10.0, 0.4 * fs)
    sos_10 = butter(N=3, Wn=fc_10, fs=fs, output="sos")
    y_mm_4 = sosfiltfilt(sos_10, y_uniforme)
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(tempo_uniforme, y_uniforme, label="Original")
    _ax.plot(tempo_uniforme, y_mm_2, label="Média móvel causal")
    _ax.plot(tempo_uniforme, y_mm_3, label="Média móvel nos dois sentidos")
    _ax.plot(tempo_uniforme, y_mm_4, label=f"Butterworth N=3, {fc_10:.2f} Hz, dois sentidos")
    _ax.set(xlabel="Tempo (s)", ylabel="y (m)")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 5.** Um filtro tem magnitude 1/√2 em certa frequência. Qual será a magnitude depois da aplicação para frente e para trás? E qual é o atraso de uma média móvel de quatro pontos quando fs=120 Hz?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solução do exercício 5': mo.md(r"""
    A magnitude torna-se $(1/\sqrt2)^2=1/2$, aproximadamente −6,02 dB. O atraso de baixa frequência da média móvel causal é $(4-1)/(2f_s)=1,5/120=0,0125$ s. A fase zero não elimina a atenuação de amplitude.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filtrar antes de derivar

    O exemplo seguinte mantém a escolha original N=1 e 5 Hz, agora ajustável. Observe a aceleração, além da posição. Reduzir o corte pode diminuir as oscilações e alterar a estimativa de g. Compare o trecho interior com as bordas usando os mesmos índices na grade uniforme.
    """)
    return


@app.cell
def _(fs, mo):
    _limite = float(0.4 * fs)
    corte = mo.ui.slider(start=float(0.01 * fs), stop=_limite, value=min(5.0, _limite), step=float(0.005 * fs), label="Corte de projeto (Hz)")
    corte
    return (corte,)


@app.cell
def _(
    butter,
    corte,
    dt,
    fs,
    np,
    plt,
    sosfiltfilt,
    tempo_uniforme,
    x_uniforme,
    y_uniforme,
):
    sos_1 = butter(N=1, Wn=corte.value, fs=fs, output="sos")
    x_filtrado = sosfiltfilt(sos_1, x_uniforme)
    y_filtrado = sosfiltfilt(sos_1, y_uniforme)
    dxdt_filtrado = np.gradient(x_filtrado, dt, edge_order=2)
    dydt_filtrado = np.gradient(y_filtrado, dt, edge_order=2)
    d2xdt2_filtrado = np.gradient(dxdt_filtrado, dt, edge_order=2)
    d2ydt2_filtrado = np.gradient(dydt_filtrado, dt, edge_order=2)
    ay_uniforme = np.gradient(np.gradient(y_uniforme, dt, edge_order=2), dt, edge_order=2)
    g_filtrado = -np.mean(d2ydt2_filtrado[10:-10])
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(tempo_uniforme, ay_uniforme, label="Sem filtro na grade uniforme")
    _ax.plot(tempo_uniforme, d2ydt2_filtrado, label=f"Filtrada, corte {corte.value:.2f} Hz")
    _ax.set(xlabel="Tempo (s)", ylabel="Aceleração y (m/s²)")
    _ax.legend()
    _fig
    return ay_uniforme, d2ydt2_filtrado


@app.cell
def _(ay_uniforme, d2ydt2_filtrado, np, pd):
    pd.DataFrame({"processamento": ["sem filtro", "Butterworth N=1, dois sentidos"],
                  "média de ay, tudo (m/s²)": [np.mean(ay_uniforme), np.mean(d2ydt2_filtrado)],
                  "média de ay, [10:-10] (m/s²)": [np.mean(ay_uniforme[10:-10]), np.mean(d2ydt2_filtrado[10:-10])]})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequência de corte pela análise dos resíduos

    `optcutfreq` examina a diferença entre dados filtrados e não filtrados ao longo de frequências de corte. Ele tenta identificar uma região dominada por ruído. Essa escolha depende das hipóteses sobre sinal e ruído; o algoritmo pode não encontrar um valor ou sugerir suavização excessiva. O filtro usado internamente pelo método não é idêntico ao exemplo N=1 acima.

    Inspecione o gráfico dos resíduos e as derivadas. A palavra “ótima” no nome do método não significa que a aceleração verdadeira seja conhecida.
    """)
    return


@app.cell
def _(fs, mo, optcutfreq, y_uniforme):
    fc_opt = optcutfreq(y_uniforme, freq=fs, show=True)
    mo.md("A análise não encontrou um corte automático; examine os resíduos e as hipóteses." if fc_opt is None else f"Corte sugerido pela análise de resíduos: **{fc_opt:.3f} Hz**.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Uma comparação com o modelo de voo livre

    Se todo o trecho escolhido representar voo livre com aceleração constante, $y(t)=c_0+c_1t+c_2t^2$ implica $g=-2c_2$. Ajustar uma parábola usa a hipótese física diretamente e oferece uma comparação com a média da segunda derivada. Não use o ajuste para esconder contatos ou arrasto relevante. Para a atividade, compare também os resíduos do ajuste com a trajetória.
    """)
    return


@app.cell
def _(np, plt, tempo, y):
    coef_parabola = np.polyfit(tempo, y, 2)
    y_modelo = np.polyval(coef_parabola, tempo)
    g_modelo = -2 * coef_parabola[0]
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3))
    _ax[0].plot(tempo, y, ".", label="Medição")
    _ax[0].plot(tempo, y_modelo, label="Parábola ajustada")
    _ax[0].set(xlabel="Tempo (s)", ylabel="y (m)")
    _ax[0].legend()
    _ax[1].plot(tempo, y-y_modelo)
    _ax[1].axhline(0, color="gray", lw=1)
    _ax[1].set(xlabel="Tempo (s)", ylabel="Resíduo (m)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 6.** Um ajuste fornece $y(t)=0,2+3t-4,8t^2$, em unidades SI. Qual é a estimativa de g? A posição do ápice depende do filtro usado para construir esse ajuste?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solução do exercício 6': mo.md(r"""
    A segunda derivada é −9,6 m/s², então g=9,6 m/s². Ajustar dados previamente filtrados pode alterar os coeficientes e o ápice. Compare o ajuste nos dados medidos com alternativas e relate a cadeia de processamento.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questões de verificação

    1. Por que N posições produzem N−1 velocidades por diferenças simples?
    2. Por que o primeiro intervalo da tabela não deve ser ignorado na análise de amostragem?
    3. Por que `for` e `lfilter` diferem nas três primeiras amostras deste exemplo?
    4. Fase zero significa ausência de distorção de amplitude?
    5. Uma média de acelerações é sempre uma estimativa válida de g?
    6. O que você deve fazer se `optcutfreq` devolver `None`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Respostas às questões de verificação': mo.md(r"""
    1. Há N−1 pares consecutivos; cada velocidade representa um intervalo.

    2. Ele revela uma lacuna; usar um dt único sem conferir os tempos muda a escala das derivadas.

    3. O laço conserva os valores iniciais e o filtro causal começa com um passado nulo.

    4. Não. As magnitudes das duas passagens se multiplicam e as bordas também exigem atenção.

    5. Não. É preciso voo livre, convenção de eixos, calibração e processamento compatíveis; contatos e ruído podem invalidar a interpretação.

    6. Inspecionar a curva de resíduos e as hipóteses; não tratar `None` como uma frequência numérica.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problemas

    1. Para fs=120 Hz e uma média móvel de quatro pontos, determine o atraso causal de baixa frequência em amostras e segundos.

    2. Um Butterworth passa-baixas de ordem 3 é aplicado para frente e para trás. Informe a ordem efetiva e a magnitude no corte de projeto.

    3. No arquivo original, construa a grade uniforme entre a segunda e a última amostra, mantendo 125 pontos. Determine fs e a frequência de Nyquist.

    4. Por que a filtragem bidirecional não pode ser usada em tempo real com atraso nulo para dados que ainda não foram medidos?

    5. Compare cortes de 2, 5 e 10 Hz no exemplo N=1, mantendo os demais parâmetros. Qual deles deve obrigatoriamente dar a estimativa mais exata de g?

    6. Se uma senóide tem amplitude 4 antes de um filtro cuja magnitude nessa frequência é 0,5, qual a amplitude depois de uma passagem e depois de duas passagens nos sentidos opostos, longe das bordas?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Respostas finais dos problemas': mo.md(r"""
    1. 1,5 amostra; 0,0125 s.

    2. Ordem efetiva 6; magnitude 0,5, aproximadamente −6,0206 dB.

    3. $f_s=119,922631$ Hz; Nyquist $=59,961315$ Hz.

    4. A passagem para trás depende de amostras futuras; o método é apropriado para processamento posterior do registro completo.

    5. Nenhum é garantidamente o mais exato. A escolha depende do espectro do movimento, do ruído, das bordas e do objetivo da estimativa.

    6. Amplitudes 2 e 1, respectivamente.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Para aprofundar

    - [Análise de resíduos](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/residual_analysis.py): examine as hipóteses usadas para sugerir uma frequência de corte.
    - [Filtragem de dados](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py): relacione ordem, resposta em frequência e processamento em dois sentidos.
    - [Cinemática de partículas](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_particle.py): retome os significados físicos de posição, velocidade e aceleração.
    - [Laboratório de gravidade](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/lab_acel_gravidade.py) e [laboratório de filtros](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/lab_filtros.py): compare os dois objetivos usando o mesmo conjunto de dados.

    ### Videoaulas na Internet

    - [MIT — Classical Mechanics](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/): as aulas iniciais de cinemática e gravidade ajudam a interpretar a aceleração durante o voo livre (em inglês).
    - [MIT — Filtering](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/resources/lecture-12-filtering/): resposta de sistemas e efeito da filtragem (em inglês).
    - [MIT — Butterworth Filters](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/resources/lecture-24-butterworth-filters/): projeto e interpretação da resposta em frequência (em inglês).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Referências

    - Watanabe, R. N. (2022). Roteiros de laboratório de Biomecânica I e arquivo `massa_A.txt`, material original deste repositório.

    - NumPy. [gradient](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) e [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html): derivadas e diferenças discretas.

    - SciPy. [butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html), [lfilter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html), [filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html) e [sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html): projeto e aplicação dos filtros usados aqui.

    - Duarte, M. [optcutfreq](https://github.com/demotu/optcutfreq): implementação e limitações da análise de resíduos proposta por Winter.
    """)
    return


if __name__ == "__main__":
    app.run()
