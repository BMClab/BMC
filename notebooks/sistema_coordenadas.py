# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Determinação de sistemas de coordenadas anatômicos com dados experimentais

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratório de Biomecânica e Controle Motor](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Universidade Federal do ABC, Brasil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Como usar este guia

    Um sistema de captura de movimento entrega, para cada marcador e em cada quadro, três números: as coordenadas X, Y e Z no laboratório. Com 28 marcadores e 364 quadros são mais de trinta mil números, e nenhum deles diz se a coxa está flexionada ou para onde o pé aponta. Para isso, cada segmento do corpo precisa de um sistema de coordenadas próprio, uma origem e três versores que se movem com ele, e esses versores precisam apontar para direções que tenham sentido anatômico.

    Este notebook constrói os sistemas de coordenadas anatômicos da pelve, do pé, da perna e da coxa direitos de uma pessoa andando, seguindo a convenção do projeto [VAKHUM](https://github.com/BMClab/BMC/blob/master/courses/refs/VAKHUM.pdf). No caminho, aparecem os problemas que surgem com dados reais: um laboratório cujos eixos não são o que parecem, um centro articular em que não se pode colocar um marcador, e a pergunta de sempre, se a base construída é mesmo ortonormal e dextrógira.

    Leia em ordem e execute cada célula quando chegar a ela. Onde houver um **Desafio** ou **Perguntas norteadoras**, pare e responda num rascunho antes de seguir. Várias delas pedem que você preveja um número *antes* de o código imprimi-lo. Faça a previsão; um palpite errado que o código corrige ensina mais do que um certo. Os **Exercícios** têm soluções recolhidas logo abaixo: abra uma solução só depois de ter a sua resposta.

    Você vai precisar das operações com vetores de [Scalar and vector](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/scalar_vector.py) (norma, produto escalar e produto vetorial) e da construção de uma base a partir de três marcadores, de [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py).

    **Desafio 0.** Coloque as mãos na cintura e procure, com os dedos, as duas espinhas ilíacas ântero-superiores, as saliências ósseas na frente da pelve. Agora aponte onde você acha que fica o centro da articulação do seu quadril, a cabeça do fêmur. Fica abaixo da espinha? Para dentro ou para fora dela? Quantos centímetros? Anote; a seção da coxa vai calcular essa posição.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Perguntas norteadoras 0.**

    1. Por que um segmento precisa de pelo menos três marcadores não colineares para ter sua orientação definida?
    2. Com três marcadores dá para construir muitas bases diferentes. Quem decide qual delas é "a" base da coxa, e para onde aponta cada eixo?
    3. O centro da articulação do quadril está dentro do corpo. Como usá-lo para definir a coxa, se não podemos colocar um marcador nele?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuração do Python

    O pandas lê o arquivo, o NumPy faz as contas e o Matplotlib desenha as figuras.
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        from notebook_resources import repo_file
    except ModuleNotFoundError as _error:
        if _error.name != "notebook_resources":
            raise
        raise ModuleNotFoundError(
            "This notebook needs notebook_resources.py. Download it from:\n"
            "https://raw.githubusercontent.com/BMClab/BMC/master/functions/notebook_resources.py\n"
            "Save it in the same folder as this notebook, then rerun this cell.",
            name="notebook_resources",
        ) from None
    return (repo_file,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Duas funções vão ser usadas em todas as seções: `check_basis` verifica se três versores, dados quadro a quadro, formam uma base ortonormal e dextrógira, e `plot_basis` desenha os marcadores de um segmento e a sua base num instante.
    """)
    return


@app.function
def check_basis(e1, e2, e3):
    """Check, frame by frame, that e1, e2, e3 (one row per frame) form a right-handed orthonormal basis."""
    import numpy as np

    norms = all(np.allclose(np.linalg.norm(e, axis=1), 1) for e in (e1, e2, e3))
    orthogonal = all(
        np.allclose(np.sum(a * b, axis=1), 0)
        for a, b in ((e1, e2), (e1, e3), (e2, e3))
    )
    det = np.linalg.det(np.stack((e1, e2, e3), axis=1))
    print(f"Versores de norma 1: {norms}")
    print(f"Versores ortogonais entre si: {orthogonal}")
    print(
        f"Determinante de [e1; e2; e3]: de {det.min():.4f} a {det.max():.4f} (dextrógira: {bool(np.allclose(det, 1))})"
    )


@app.function
def plot_basis(
    origin,
    basis,
    markers,
    labels,
    axis_names=("x", "y", "z"),
    ax=None,
    length=0.12,
    azim=35,
):
    """Draw markers and a basis (versors in the rows) at `origin`, in the laboratory, with Y upwards.

    If `ax` is given, draw on it and leave its limits alone; otherwise create a figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    new_figure = ax is None
    if new_figure:
        ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    origin, basis, markers = (
        np.asarray(origin),
        np.asarray(basis),
        np.atleast_2d(markers),
    )
    ax.scatter(*markers.T, s=40, color="gold", edgecolors="k", depthshade=False)
    for p, label in zip(markers, labels):
        ax.text(*p, f"  {label}", fontsize=9)
    for e, name, color in zip(
        basis, axis_names, ["tab:red", "tab:green", "tab:blue"]
    ):
        ax.quiver(
            *origin,
            *(length * e),
            color=color,
            linewidth=2.5,
            arrow_length_ratio=0.15,
        )
        ax.text(
            *(origin + 1.2 * length * e),
            name,
            color=color,
            fontsize=12,
            fontweight="bold",
        )
    if new_figure:
        equal_axes(ax, np.vstack((markers, origin, origin + length * basis)))
        ax.view_init(elev=15, azim=azim, vertical_axis="y")
    return ax


@app.function
def equal_axes(ax, points, margin=0.05):
    """Give a 3D axes the same scale on X, Y and Z, enclosing `points` (one per row)."""
    import numpy as np

    points = np.asarray(points)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    half = (points.max(axis=0) - points.min(axis=0)).max() / 2 + margin
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Leitura do arquivo e pré-processamento

    O arquivo `.trc` é texto separado por tabulações. Depois de três linhas de cabeçalho vêm duas linhas de rótulos: os nomes dos marcadores, cada um ocupando três colunas, e os nomes das coordenadas. A célula abaixo junta as duas numa única linha de nomes, como `RASISX`, `RASISY` e `RASISZ`, descarta a coluna do número do quadro e uma coluna vazia no fim, e converte as coordenadas de milímetros para metros:
    """)
    return


@app.cell
def _(mo, pd, repo_file):
    _path = mo.notebook_dir() / ".." / "data" / "walk1.trc"
    if not _path.exists():  # e.g., on molab, where only the notebook is available
        _path = repo_file("data/walk1.trc")
    data_trc = pd.read_csv(_path, skiprows=3, sep="\t")
    _cols = list(data_trc.columns)
    _cols = [_col.replace(".", "") for _col in _cols]
    for _i in range(2, len(_cols) - 3, 3):
        _marca = _cols[_i]
        _cols[_i] = _marca + "X"
        _cols[_i + 1] = _marca + "Y"
        _cols[_i + 2] = _marca + "Z"

    _cols = [_c.upper() for _c in _cols]
    data_trc.columns = _cols
    data_trc = data_trc.drop(0, axis=0)
    data_trc = data_trc.drop(data_trc.columns[[0, -1]], axis=1)
    data_trc = data_trc.astype(float)
    data_trc.iloc[:, 1:] = data_trc.iloc[:, 1:].values / 1000  # mm to m
    print(list(data_trc.columns))
    data_trc
    return (data_trc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Para onde aponta cada eixo do laboratório?

    A convenção VAKHUM, como a da Sociedade Internacional de Biomecânica (ISB), usa no laboratório o eixo $\mathbf{Y}$ vertical e para cima, $\mathbf{X}$ horizontal e para a frente, na direção da progressão, e $\mathbf{Z}$ horizontal e para a direita. Com $\mathbf{X}$ para a frente e $\mathbf{Y}$ para cima, um sistema dextrógiro *exige* $\mathbf{Z}$ para a direita: $\hat{\mathbf{k}} = \hat{\mathbf{i}} \times \hat{\mathbf{j}}$.

    Mas os eixos de um laboratório são definidos na calibração do sistema de captura, e nada garante que sigam a convenção. Antes de construir qualquer base, vale perguntar aos dados.

    **Antes de executar a próxima célula**, preveja: a espinha ilíaca direita, `RASIS`, deve ter a coordenada $Z$ maior ou menor que a esquerda, `LASIS`?
    """)
    return


@app.cell
def _(data_trc):
    _asis = (
        data_trc[["RASISX", "RASISY", "RASISZ"]].values
        - data_trc[["LASISX", "LASISY", "LASISZ"]].values
    )
    _travel = data_trc[["RASISX", "RASISY", "RASISZ"]].values[[0, -1]]
    print("RASIS - LASIS, média no ensaio [m]:", _asis.mean(axis=0).round(3))
    print(
        "Deslocamento da RASIS do primeiro ao último quadro [m]:",
        (_travel[1] - _travel[0]).round(3),
    )
    print("Altura média da RASIS [m]:", data_trc["RASISY"].mean().round(3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A pessoa anda 2,86 m no sentido positivo de $\mathbf{X}$, e a espinha ilíaca está a cerca de 1 m de altura em $\mathbf{Y}$: até aqui, tudo conforme a convenção. Mas a espinha direita está 23 cm *abaixo* da esquerda em $Z$. O eixo $\mathbf{Z}$ deste laboratório aponta para a *esquerda* da pessoa, e com $\mathbf{X}$ para a frente e $\mathbf{Y}$ para cima isso faz dele um sistema **levógiro**.

    Isso importa porque o produto vetorial é definido pelas coordenadas: num sistema levógiro, $\vec{\mathbf{a}} \times \vec{\mathbf{b}}$ calculado com os números aponta para o lado oposto ao que a regra da mão direita indicaria. Toda a construção das bases a seguir depende de produtos vetoriais, e a matriz de rotação, os ângulos de Cardan e a velocidade angular dependem de bases dextrógiras.

    A solução mais simples é corrigir os dados uma vez, logo na entrada: trocar o sinal de todas as coordenadas $Z$. Isso espelha o laboratório, e o novo $\mathbf{Z}$ aponta para a direita, como na convenção. Daqui em diante, `data` é o laboratório dextrógiro:
    """)
    return


@app.cell
def _(data_trc):
    data = data_trc.copy()
    _z_columns = [_col for _col in data.columns if _col.endswith("Z")]
    data[_z_columns] = -data[_z_columns]  # Z to the right: a right-handed laboratory
    t = data["TIME"].values
    return data, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pelve

    **Planos anatômicos da pelve** (VAKHUM)

    - **Plano quase-transverso**: definido pelas espinhas ilíacas ântero-superiores direita e esquerda (**rasis** e **lasis**) e pelo ponto médio entre as póstero-superiores (**rpsis** e **lpsis**).
    - **Plano quase-coronal**: ortogonal ao plano quase-transverso e contendo **rasis** e **lasis**.
    - **Plano quase-sagital**: perpendicular aos planos quase-transverso e quase-coronal.

    **Sistema de coordenadas anatômico da pelve** $x_p y_p z_p$

    - **Origem**: o ponto médio entre **rasis** e **lasis**.
    - **Eixo $z_p$**: ao longo da linha que passa por **rasis** e **lasis**, com sentido positivo para a direita.
    - **Eixo $x_p$**: no plano quase-transverso e perpendicular a $z_p$, com sentido positivo para a frente.
    - **Eixo $y_p$**: perpendicular a $x_p$ e a $z_p$, para cima.

    O relatório VAKHUM chama a origem de $O_a$ e o ponto médio das espinhas póstero-superiores de $O_p$. No código, a origem da pelve é `Op` e o ponto médio das póstero-superiores é `mPSIS`. Os versores $\hat{\mathbf{e}}_1$, $\hat{\mathbf{e}}_2$ e $\hat{\mathbf{e}}_3$ de cada segmento correspondem aos eixos $x$, $y$ e $z$.

    ### Seleção das marcas necessárias
    """)
    return


@app.cell
def _(data):
    RASIS = data[["RASISX", "RASISY", "RASISZ"]].values
    LASIS = data[["LASISX", "LASISY", "LASISZ"]].values
    RPSIS = data[["RPSISX", "RPSISY", "RPSISZ"]].values
    LPSIS = data[["LPSISX", "LPSISY", "LPSISZ"]].values

    Op = (RASIS + LASIS) / 2
    return LASIS, LPSIS, Op, RASIS, RPSIS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cada variável tem uma linha por quadro e uma coluna por coordenada, e todas as contas a seguir são feitas para todos os quadros de uma vez.

    ### $\hat{\mathbf{e}}_3$, para a direita

    O eixo $z_p$ é o mais fácil: a direção de **lasis** para **rasis**, normalizada.
    """)
    return


@app.cell
def _(LASIS, RASIS, np):
    _v3 = RASIS - LASIS
    e3p = _v3 / np.linalg.norm(_v3, axis=1, keepdims=True)
    return (e3p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $\hat{\mathbf{e}}_2$, para cima

    O eixo $y_p$ é perpendicular ao plano quase-transverso, e um vetor perpendicular a um plano é o produto vetorial de dois vetores contidos nele. Do ponto médio das póstero-superiores, um vetor vai para **rasis** (para a frente e para a direita) e outro para **lasis** (para a frente e para a esquerda). Pela regra da mão direita, frente-direita $\times$ frente-esquerda aponta para cima; na ordem inversa, apontaria para baixo.
    """)
    return


@app.cell
def _(LASIS, LPSIS, RASIS, RPSIS, np):
    mPSIS = (RPSIS + LPSIS) / 2
    _v2 = np.cross(RASIS - mPSIS, LASIS - mPSIS)
    e2p = _v2 / np.linalg.norm(_v2, axis=1, keepdims=True)
    return (e2p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $\hat{\mathbf{e}}_1$, para a frente

    O terceiro versor completa a base dextrógira: $\hat{\mathbf{e}}_1 = \hat{\mathbf{e}}_2 \times \hat{\mathbf{e}}_3$. Como $\hat{\mathbf{e}}_2$ e $\hat{\mathbf{e}}_3$ são unitários e perpendiculares, o resultado já é unitário, e fica no plano quase-transverso, perpendicular a $z_p$, como pede a definição de $x_p$.
    """)
    return


@app.cell
def _(e2p, e3p, np):
    e1p = np.cross(e2p, e3p)
    return (e1p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Verificar normas, ortogonalidade e orientação

    Uma base ortonormal tem versores de norma 1, perpendiculares dois a dois. Isso não basta: uma base com um dos versores invertido também passa nesses dois testes. A orientação aparece no determinante da matriz com os versores nas linhas, $+1$ para uma base dextrógira e $-1$ para uma levógira.
    """)
    return


@app.cell
def _(e1p, e2p, e3p):
    check_basis(e1p, e2p, e3p)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizar a base

    No primeiro quadro, com a base desenhada na origem da pelve e as quatro espinhas ilíacas:
    """)
    return


@app.cell
def _(LASIS, LPSIS, Op, RASIS, RPSIS, e1p, e2p, e3p, np, plt):
    _markers = np.vstack((RASIS[0], LASIS[0], RPSIS[0], LPSIS[0]))
    _basis = np.vstack((e1p[0], e2p[0], e3p[0]))

    plot_basis(
        Op[0],
        _basis,
        _markers,
        ["RASIS", "LASIS", "RPSIS", "LPSIS"],
        axis_names=("xp", "yp", "zp"),
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    O eixo $x_p$ aponta para a frente mas um pouco para baixo: as espinhas ântero-superiores estão mais baixas que as póstero-superiores, a inclinação anterior da pelve.

    **Exercício 1.** Qual é o ângulo de inclinação anterior da pelve no primeiro quadro, o ângulo entre $x_p$ e o plano horizontal? Quanto ele varia durante o ensaio?
    """)
    return


@app.cell(hide_code=True)
def _(e1p, mo, np):
    _tilt = np.degrees(np.arcsin(-e1p[:, 1]))

    mo.accordion(
        {
            "Solução do Exercício 1": mo.md(
                rf"""
    O plano horizontal é perpendicular a $\mathbf{{Y}}$, então o ângulo entre $\hat{{\mathbf{{e}}}}_1$ e esse plano é o complemento do ângulo entre $\hat{{\mathbf{{e}}}}_1$ e $\hat{{\mathbf{{j}}}}$. Com o sinal positivo quando $x_p$ aponta para baixo,

    $$
    \theta_{{tilt}} = \arcsin\left(-\hat{{\mathbf{{e}}}}_1\cdot\hat{{\mathbf{{j}}}}\right) = \arcsin\left(-e_{{1Y}}\right)
    $$

    No primeiro quadro, $e_{{1Y}} = {e1p[0, 1]:.3f}$ e a inclinação anterior é de ${_tilt[0]:.1f}^\circ$. Durante o ensaio ela fica entre ${_tilt.min():.1f}^\circ$ e ${_tilt.max():.1f}^\circ$: a pelve oscila uns poucos graus a cada passo. Esse ângulo depende da convenção, porque o plano quase-transverso passa pelas espinhas ilíacas, e não por algum plano "verdadeiramente" horizontal da pelve; outra convenção daria outro número para a mesma postura.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 2.** Suponha que você não tivesse corrigido o eixo $\mathbf{Z}$ do laboratório e tivesse aplicado exatamente as mesmas operações aos dados originais, `data_trc`. Para onde apontaria cada versor? Qual seria o determinante? E se, para fazer $\hat{\mathbf{e}}_2$ apontar para cima, você trocasse a ordem do produto vetorial?
    """)
    return


@app.cell(hide_code=True)
def _(data_trc, mo, np):
    def _versor(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def _marker(name):
        return data_trc[[name + _c for _c in "XYZ"]].values

    _ra, _la = _marker("RASIS"), _marker("LASIS")
    _mp = (_marker("RPSIS") + _marker("LPSIS")) / 2
    _e3 = _versor(_ra - _la)
    _e2 = _versor(np.cross(_ra - _mp, _la - _mp))  # same order as above
    _e1 = np.cross(_e2, _e3)
    _det = np.linalg.det(np.stack((_e1, _e2, _e3), axis=1))
    _e2b = _versor(np.cross(_la - _mp, _ra - _mp))  # order swapped
    _e1b = np.cross(_e3, _e2b)  # to keep e1 forward
    _detb = np.linalg.det(np.stack((_e1b, _e2b, _e3), axis=1))

    mo.accordion(
        {
            "Solução do Exercício 2": mo.md(
                rf"""
    Com as mesmas operações sobre os dados originais, no primeiro quadro,

    $$
    \hat{{\mathbf{{e}}}}_1 = [{_e1[0, 0]:.3f},\; {_e1[0, 1]:.3f},\; {_e1[0, 2]:.3f}], \quad
    \hat{{\mathbf{{e}}}}_2 = [{_e2[0, 0]:.3f},\; {_e2[0, 1]:.3f},\; {_e2[0, 2]:.3f}], \quad
    \hat{{\mathbf{{e}}}}_3 = [{_e3[0, 0]:.3f},\; {_e3[0, 1]:.3f},\; {_e3[0, 2]:.3f}]
    $$

    e o determinante é ${_det[0]:.0f}$. Numericamente a base é dextrógira, mas $\hat{{\mathbf{{e}}}}_2$ aponta para *baixo*: o produto vetorial calculado com as coordenadas de um laboratório levógiro dá o sentido oposto ao da regra da mão direita. $\hat{{\mathbf{{e}}}}_3$ continua apontando para a direita da pessoa, porque é uma diferença, não um produto vetorial, e $\hat{{\mathbf{{e}}}}_1$ ainda aponta para a frente, porque os dois sinais trocados se compensam.

    Trocando a ordem dos produtos vetoriais para que cada eixo aponte para o lado anatômico certo, $\hat{{\mathbf{{e}}}}_2$ volta a apontar para cima, mas o determinante passa a ser ${_detb[0]:.0f}$: a base é levógira. As setas estariam certas num desenho, mas os números descreveriam uma reflexão, e não uma rotação. Num laboratório levógiro não há como ter as duas coisas; por isso corrigimos o laboratório antes de tudo.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pé

    **Planos anatômicos do pé** (VAKHUM)

    - **Plano quase-transverso**: definido pelo calcâneo (**ca**) e pelas cabeças do primeiro e do quinto metatarsos (**fm** e **vm**).
    - **Plano quase-sagital**: ortogonal ao plano quase-transverso e contendo **ca** e a cabeça do segundo metatarso (**sm**).
    - **Plano quase-coronal**: perpendicular aos planos quase-transverso e quase-sagital.

    **Sistema de coordenadas anatômico do pé** $x_f y_f z_f$

    - **Origem** $O_f$: o ponto **ca**.
    - **Eixo $y_f$**: a interseção dos planos quase-coronal e quase-sagital, para cima.
    - **Eixo $z_f$**: no plano quase-transverso e perpendicular a $y_f$, para a direita.
    - **Eixo $x_f$**: perpendicular a $y_f$ e a $z_f$, para a frente.

    No arquivo, **ca** é o marcador do calcanhar (`R.Heel`), e **fm**, **sm** e **vm** são `R.MT1`, `R.MT2` e `R.MT5`:
    """)
    return


@app.cell
def _(data):
    CA = data[["RHEELX", "RHEELY", "RHEELZ"]].values
    FM = data[["RMT1X", "RMT1Y", "RMT1Z"]].values
    SM = data[["RMT2X", "RMT2Y", "RMT2Z"]].values
    VM = data[["RMT5X", "RMT5Y", "RMT5Z"]].values

    Of = CA
    return CA, FM, Of, SM, VM


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A interseção dos planos quase-coronal e quase-sagital é uma reta perpendicular a ambos e, como os dois são perpendiculares ao quase-transverso, é a normal do plano quase-transverso. No pé direito, o quinto metatarso fica à direita e o primeiro à esquerda, então $(\mathbf{vm} - \mathbf{ca}) \times (\mathbf{fm} - \mathbf{ca})$ aponta para cima:
    """)
    return


@app.cell
def _(CA, FM, VM, np):
    _v2 = np.cross(VM - CA, FM - CA)
    e2f = _v2 / np.linalg.norm(_v2, axis=1, keepdims=True)
    return (e2f,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    O eixo $z_f$ é perpendicular ao plano quase-sagital, que contém $\mathbf{sm} - \mathbf{ca}$ (para a frente) e $\hat{\mathbf{e}}_2$ (para cima); frente $\times$ cima aponta para a direita:
    """)
    return


@app.cell
def _(CA, SM, e2f, np):
    _v3 = np.cross(SM - CA, e2f)
    e3f = _v3 / np.linalg.norm(_v3, axis=1, keepdims=True)
    return (e3f,)


@app.cell
def _(e2f, e3f, np):
    e1f = np.cross(e2f, e3f)

    check_basis(e1f, e2f, e3f)
    return (e1f,)


@app.cell
def _(CA, FM, Of, SM, VM, e1f, e2f, e3f, np, plt):
    _markers = np.vstack((CA[0], FM[0], SM[0], VM[0]))
    _basis = np.vstack((e1f[0], e2f[0], e3f[0]))

    plot_basis(
        Of[0],
        _basis,
        _markers,
        ["CA", "FM", "SM", "VM"],
        axis_names=("xf", "yf", "zf"),
        length=0.08,
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Desafio 1.** A convenção pede a crista superior do calcâneo, mas o marcador `R.Heel` costuma ficar na face posterior do calcanhar, alguns centímetros abaixo. Desloque `CA` 2 cm para cima (some $0{,}02$ à coordenada $Y$) e construa a base do pé de novo. De quantos graus muda o eixo $x_f$? Qual dos três eixos muda menos, e por quê?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perna

    **Planos anatômicos da perna** (VAKHUM)

    - **Plano quase-coronal**: definido pelo ápice da cabeça da fíbula (**hf**), pelo maléolo lateral (**lm**) e pelo ponto médio $O_s$ entre os maléolos lateral e medial (**lm** e **mm**).
    - **Plano quase-sagital**: ortogonal ao plano quase-coronal e contendo $O_s$ e a tuberosidade da tíbia (**tt**).
    - **Plano quase-transverso**: perpendicular aos planos quase-coronal e quase-sagital.

    **Sistema de coordenadas anatômico da perna** $x_s y_s z_s$

    - **Origem** $O_s$: o ponto médio entre **lm** e **mm**.
    - **Eixo $y_s$**: a interseção dos planos quase-coronal e quase-sagital, para cima.
    - **Eixo $z_s$**: no plano quase-coronal e perpendicular a $y_s$, para a direita.
    - **Eixo $x_s$**: perpendicular a $y_s$ e a $z_s$, para a frente.

    No arquivo, **hf** é `R.HF`, **tt** é `R.TT`, **lm** é `R.Ankle` e **mm** é `R.Ankle.Medial`:
    """)
    return


@app.cell
def _(data):
    HF = data[["RHFX", "RHFY", "RHFZ"]].values
    TT = data[["RTTX", "RTTY", "RTTZ"]].values
    MM = data[["RANKLEMEDIALX", "RANKLEMEDIALY", "RANKLEMEDIALZ"]].values
    ML = data[["RANKLEX", "RANKLEY", "RANKLEZ"]].values

    Os = (MM + ML) / 2
    return HF, ML, MM, Os, TT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    O eixo $x_s$ é a normal do plano quase-coronal. Da cabeça da fíbula, um vetor desce até o maléolo lateral e outro desce até o medial, à esquerda dele na perna direita; para baixo $\times$ (para baixo e para a esquerda) aponta para a frente. O código usa os dois maléolos em vez de **lm** e $O_s$: é o mesmo plano (Exercício 3).
    """)
    return


@app.cell
def _(HF, ML, MM, np):
    _v1 = np.cross(ML - HF, MM - HF)
    e1s = _v1 / np.linalg.norm(_v1, axis=1, keepdims=True)
    return (e1s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    O eixo $z_s$ é a normal do plano quase-sagital, que contém $\hat{\mathbf{e}}_1$ (para a frente) e $\mathbf{tt} - O_s$ (para cima); frente $\times$ cima aponta para a direita. Por fim, $\hat{\mathbf{e}}_2 = \hat{\mathbf{e}}_3 \times \hat{\mathbf{e}}_1$ fica nos dois planos, a interseção que a definição pede:
    """)
    return


@app.cell
def _(Os, TT, e1s, np):
    _v3 = np.cross(e1s, TT - Os)
    e3s = _v3 / np.linalg.norm(_v3, axis=1, keepdims=True)
    return (e3s,)


@app.cell
def _(e1s, e3s, np):
    e2s = np.cross(e3s, e1s)

    check_basis(e1s, e2s, e3s)
    return (e2s,)


@app.cell
def _(HF, ML, MM, Os, TT, e1s, e2s, e3s, np, plt):
    _markers = np.vstack((HF[0], TT[0], MM[0], ML[0]))
    _basis = np.vstack((e1s[0], e2s[0], e3s[0]))

    plot_basis(
        Os[0], _basis, _markers, ["HF", "TT", "MM", "ML"], axis_names=("xs", "ys", "zs")
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 3.** A convenção define o plano quase-coronal da perna por **hf**, **lm** e $O_s$, e o código usa **hf**, **lm** e **mm**. Mostre que os dois trios de pontos definem o mesmo plano e que as normais $(\mathbf{lm} - \mathbf{hf}) \times (O_s - \mathbf{hf})$ e $(\mathbf{lm} - \mathbf{hf}) \times (\mathbf{mm} - \mathbf{hf})$ têm a mesma direção e o mesmo sentido.

    /// details | Solução do Exercício 3

    Chame $\vec{\mathbf{a}} = \mathbf{lm} - \mathbf{hf}$ e $\vec{\mathbf{b}} = \mathbf{mm} - \mathbf{hf}$. Como $O_s = (\mathbf{lm} + \mathbf{mm})/2$,

    $$
    O_s - \mathbf{hf} = \frac{(\mathbf{lm} - \mathbf{hf}) + (\mathbf{mm} - \mathbf{hf})}{2} = \frac{\vec{\mathbf{a}} + \vec{\mathbf{b}}}{2}
    $$

    e, como $\vec{\mathbf{a}} \times \vec{\mathbf{a}} = \vec{\mathbf{0}}$,

    $$
    \vec{\mathbf{a}} \times \frac{\vec{\mathbf{a}} + \vec{\mathbf{b}}}{2} = \frac{1}{2}\,\vec{\mathbf{a}} \times \vec{\mathbf{b}}
    $$

    As duas normais diferem só pelo fator positivo $1/2$, que desaparece na normalização: mesma direção, mesmo sentido, mesmo $\hat{\mathbf{e}}_1$. Geometricamente, $O_s$ está no segmento entre **lm** e **mm**, então qualquer plano que contenha **lm** e **mm** contém $O_s$.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coxa

    **Planos anatômicos da coxa** (VAKHUM)

    - **Plano quase-coronal**: definido pelos epicôndilos medial e lateral do fêmur (**me** e **le**) e pelo centro da cabeça do fêmur (**fh**).
    - **Plano quase-sagital**: ortogonal ao plano quase-coronal e contendo $O_t$ e **fh**.
    - **Plano quase-transverso**: perpendicular aos planos quase-coronal e quase-sagital.

    **Sistema de coordenadas anatômico da coxa** $x_t y_t z_t$

    - **Origem** $O_t$: o ponto médio entre **le** e **me**.
    - **Eixo $y_t$**: ao longo da linha que passa por $O_t$ e **fh**, para cima.
    - **Eixo $z_t$**: no plano quase-coronal e perpendicular a $y_t$, para a direita.
    - **Eixo $x_t$**: perpendicular a $y_t$ e a $z_t$, para a frente.

    ### Onde está a cabeça do fêmur?

    Para determinar o sistema de coordenadas anatômico usando a convenção VAKHUM é necessária a posição da cabeça do fêmur (FH), que é o centro articular do quadril (HJC). Obviamente não é possível colocar um marcador na cabeça do fêmur para rastrear sua posição, porque é uma região dentro do corpo.

    O que usualmente se faz é usar algum método indireto para estimar esta posição, por exemplo a partir de equações de regressão que relacionam as dimensões da pelve com a posição da cabeça do fêmur. Estas equações de regressão são geralmente obtidas do estudo de uma amostra da população, em que a posição da cabeça do fêmur é medida por algum outro método (por exemplo, raio-X).

    Para estimar a posição da cabeça do fêmur, vamos utilizar o método proposto por Bell, Brand e Pedersen (1989):

    $$
    \vec{\mathbf{r}}_{FH} = \vec{\mathbf{r}}_{O_p} + d_{ASIS}\left(-0{,}19\,\hat{\mathbf{e}}_1 - 0{,}30\,\hat{\mathbf{e}}_2 + 0{,}36\,\hat{\mathbf{e}}_3\right)
    $$

    onde $O_p$ é a origem do sistema de coordenadas da pelve, $d_{ASIS}$ é a distância entre as duas espinhas ilíacas ântero-superiores, e $\hat{\mathbf{e}}_1$, $\hat{\mathbf{e}}_2$ e $\hat{\mathbf{e}}_3$ são os versores da pelve: o centro do quadril direito fica 19% de $d_{ASIS}$ para trás, 30% para baixo e 36% para a direita da origem da pelve.

    Note que os coeficientes são componentes na base *da pelve*, e não do laboratório: a pelve inclina, gira e oscila, e o quadril vai com ela.

    **Antes de executar a próxima célula**, volte ao Desafio 0: a que distância da espinha ilíaca direita você esperava encontrar o centro do quadril?
    """)
    return


@app.cell
def _(LASIS, Op, RASIS, e1p, e2p, e3p, np):
    dASIS = np.linalg.norm(RASIS - LASIS, axis=1, keepdims=True)
    FH = Op + dASIS * (-0.19 * e1p - 0.30 * e2p + 0.36 * e3p)

    _d = FH[0] - RASIS[0]
    print(
        f"Distância entre as espinhas ilíacas ântero-superiores: {dASIS[0, 0] * 100:.1f} cm"
    )
    print(
        "FH em relação à RASIS, na base da pelve [cm]:",
        f"{_d @ e1p[0] * 100:.1f} (frente), {_d @ e2p[0] * 100:.1f} (cima), {_d @ e3p[0] * 100:.1f} (direita)",
    )
    return FH, dASIS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    O centro do quadril fica 4,4 cm atrás da espinha ilíaca, 6,9 cm abaixo dela e 3,2 cm para dentro (os 36% para a direita da origem deixam o quadril 14% de $d_{ASIS}$ para dentro da espinha, que está a 50%). Vamos checar nossa estimativa do marcador FH plotando junto com o sistema de coordenadas da pelve e o marcador do trocânter maior, `R.GTR`, a saliência óssea do fêmur que se palpa na lateral do quadril:
    """)
    return


@app.cell
def _(FH, LASIS, LPSIS, Op, RASIS, RPSIS, data, e1p, e2p, e3p, np, plt):
    GTR = data[["RGTRX", "RGTRY", "RGTRZ"]].values
    _markers = np.vstack((RASIS[0], LASIS[0], RPSIS[0], LPSIS[0], FH[0], GTR[0]))
    _basis = np.vstack((e1p[0], e2p[0], e3p[0]))

    plot_basis(
        Op[0],
        _basis,
        _markers,
        ["RASIS", "LASIS", "RPSIS", "LPSIS", "FH", "GTR"],
        axis_names=("xp", "yp", "zp"),
        azim=60,
    )
    plt.show()

    _d = GTR - FH
    print(
        "GTR em relação a FH no primeiro quadro, na base da pelve [cm]:",
        f"{_d[0] @ e1p[0] * 100:.1f} (frente), {_d[0] @ e2p[0] * 100:.1f} (cima), {_d[0] @ e3p[0] * 100:.1f} (direita)",
    )
    print(
        f"Distância FH-GTR no ensaio: de {np.linalg.norm(_d, axis=1).min() * 100:.1f} a {np.linalg.norm(_d, axis=1).max() * 100:.1f} cm"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    O centro estimado fica 9,3 cm para dentro do marcador do trocânter maior, um pouco acima dele e à frente: onde a cabeça do fêmur deveria estar, na ponta do colo do fêmur, que sobe e vai para dentro e para a frente a partir do trocânter. A distância entre os dois varia menos de 1 cm durante o ensaio, embora um seja calculado a partir da pelve e o outro seja um marcador na pele da coxa.

    Uma regressão como essa erra, para uma pessoa em particular, por um ou dois centímetros, e métodos mais recentes, baseados em ressonância magnética (Harrington et al., 2007) ou no movimento da coxa em relação à pelve (métodos funcionais), erram menos. Mas, para a direção do eixo longo da coxa, de 45 cm, um centímetro corresponde a pouco mais de um grau.

    Agora a coxa. No arquivo, **le** é `R.Knee` e **me** é `R.Knee.Medial`:
    """)
    return


@app.cell
def _(data):
    LE = data[["RKNEEX", "RKNEEY", "RKNEEZ"]].values
    ME = data[["RKNEEMEDIALX", "RKNEEMEDIALY", "RKNEEMEDIALZ"]].values

    Ot = (LE + ME) / 2
    return LE, ME, Ot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    O eixo $y_t$ vai da origem até a cabeça do fêmur. O eixo $x_t$ é a normal do plano quase-coronal, que contém $\hat{\mathbf{e}}_2$ (para cima) e $\mathbf{le} - O_t$ (para a direita, na coxa direita); cima $\times$ direita aponta para a frente. E $\hat{\mathbf{e}}_3 = \hat{\mathbf{e}}_1 \times \hat{\mathbf{e}}_2$ fica no plano quase-coronal, perpendicular a $y_t$:
    """)
    return


@app.cell
def _(FH, Ot, np):
    _v2 = FH - Ot
    e2t = _v2 / np.linalg.norm(_v2, axis=1, keepdims=True)
    return (e2t,)


@app.cell
def _(LE, Ot, e2t, np):
    _v1 = np.cross(e2t, LE - Ot)
    e1t = _v1 / np.linalg.norm(_v1, axis=1, keepdims=True)
    return (e1t,)


@app.cell
def _(e1t, e2t, np):
    e3t = np.cross(e1t, e2t)

    check_basis(e1t, e2t, e3t)
    return (e3t,)


@app.cell
def _(FH, LE, ME, Ot, e1t, e2t, e3t, np, plt):
    _markers = np.vstack((LE[0], ME[0], FH[0]))
    _basis = np.vstack((e1t[0], e2t[0], e3t[0]))

    plot_basis(
        Ot[0],
        _basis,
        _markers,
        ["LE", "ME", "FH"],
        axis_names=("xt", "yt", "zt"),
        length=0.15,
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercício 4.** Uma forma comum de errar a equação de Bell, Brand e Pedersen é somar os coeficientes diretamente às coordenadas do laboratório, $\vec{\mathbf{r}}_{FH} = \vec{\mathbf{r}}_{O_p} + d_{ASIS}\,[-0{,}19,\; -0{,}30,\; 0{,}36]$. Calcule FH assim e compare com a estimativa feita com a base da pelve. De quanto é a diferença? Em que situação ela seria grande?

    **Desafio 2.** O fêmur é rígido, mas calcule a distância entre FH e $O_t$, o comprimento da coxa, em todos os quadros, e faça um gráfico dela ao longo do tempo. Ela é constante? De onde vem a variação?
    """)
    return


@app.cell(hide_code=True)
def _(FH, Op, dASIS, e1p, mo, np):
    _fh_lab = Op + dASIS * np.array([-0.19, -0.30, 0.36])
    _diff = np.linalg.norm(_fh_lab - FH, axis=1) * 100
    _heading = np.degrees(np.arctan2(-e1p[:, 2], e1p[:, 0]))

    mo.accordion(
        {
            "Solução do Exercício 4": mo.md(
                rf"""
    As duas estimativas diferem em ${_diff[0]:.1f}$ cm no primeiro quadro e em até ${_diff.max():.1f}$ cm no ensaio (${_diff.mean():.1f}$ cm em média). A diferença é pequena aqui porque a base da pelve quase coincide com a do laboratório: a pessoa anda ao longo de $\mathbf{{X}}$ e, no plano horizontal, a pelve gira só entre ${_heading.min():.1f}^\circ$ e ${_heading.max():.1f}^\circ$ em relação a ele, além de inclinar uns $8^\circ$ para a frente.

    Ela seria grande se a pessoa andasse em outra direção. Andando ao longo de $\mathbf{{Z}}$, por exemplo, os 36% "para a direita" no laboratório cairiam à frente ou atrás da pelve, e o quadril estimado ficaria a uns $0{{,}}36\,d_{{ASIS}}\sqrt{{2}} \approx {0.36 * np.sqrt(2) * dASIS[0, 0] * 100:.0f}$ cm de onde deveria. Coeficientes de regressão como esses são sempre componentes numa base anatômica.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## O membro inteiro

    Com as quatro bases calculadas em todos os quadros, podemos ver o membro inferior direito em qualquer instante do ensaio. Use os controles para percorrer o passo e girar a vista:
    """)
    return


@app.cell
def _(mo, t):
    frame_slider = mo.ui.slider(0, len(t) - 1, step=1, value=0, label="Quadro")
    azim_slider = mo.ui.slider(
        -180, 180, step=5, value=35, label="Azimute da vista [graus]"
    )
    mo.vstack([frame_slider, azim_slider])
    return azim_slider, frame_slider


@app.cell
def _(
    CA,
    FH,
    FM,
    HF,
    LASIS,
    LE,
    LPSIS,
    ME,
    ML,
    MM,
    Of,
    Op,
    Os,
    Ot,
    RASIS,
    RPSIS,
    SM,
    TT,
    VM,
    azim_slider,
    e1f,
    e1p,
    e1s,
    e1t,
    e2f,
    e2p,
    e2s,
    e2t,
    e3f,
    e3p,
    e3s,
    e3t,
    frame_slider,
    np,
    plt,
    t,
):
    _i = frame_slider.value
    _ax = plt.figure(figsize=(7, 7)).add_subplot(projection="3d")
    _segments = [
        (Op, (e1p, e2p, e3p), (RASIS, LASIS, RPSIS, LPSIS, FH)),
        (Ot, (e1t, e2t, e3t), (LE, ME)),
        (Os, (e1s, e2s, e3s), (HF, TT, MM, ML)),
        (Of, (e1f, e2f, e3f), (CA, FM, SM, VM)),
    ]
    _points = []
    for _origin, _basis, _markers in _segments:
        _m = np.vstack([_mk[_i] for _mk in _markers])
        plot_basis(
            _origin[_i],
            np.vstack([_e[_i] for _e in _basis]),
            _m,
            (),
            axis_names=("", "", ""),
            ax=_ax,
            length=0.1,
        )
        _points.append(_m)
    _chain = np.vstack((FH[_i], Ot[_i], Os[_i], CA[_i]))
    _ax.plot(*_chain.T, color="gray", linewidth=1.5, linestyle="--")
    equal_axes(_ax, np.vstack(_points))
    _ax.view_init(elev=15, azim=azim_slider.value, vertical_axis="y")
    _ax.set_title(f"Membro inferior direito, t = {t[_i]:.3f} s")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Os versores estão em vermelho ($x$, para a frente), verde ($y$, para cima) e azul ($z$, para a direita), e a linha tracejada liga o centro do quadril, a origem da coxa, a origem da perna e o calcâneo.
    """)
    return


@app.cell(hide_code=True)
def _(e1f, e1s, e2s, mo, np, t):
    _ankle = np.degrees(np.arctan2(np.sum(e1f * e2s, axis=1), np.sum(e1f * e1s, axis=1)))
    _imax, _imin = np.argmax(_ankle), np.argmin(_ankle)

    mo.accordion(
        {
            "Solução do Exercício 5": mo.md(
                rf"""
    O eixo do pé que aponta para a frente, $\hat{{\mathbf{{e}}}}_{{1f}}$, escrito na base da perna, tem componentes $\hat{{\mathbf{{e}}}}_{{1f}}\cdot\hat{{\mathbf{{e}}}}_{{1s}}$ (para a frente da perna) e $\hat{{\mathbf{{e}}}}_{{1f}}\cdot\hat{{\mathbf{{e}}}}_{{2s}}$ (para cima, ao longo da perna). Na dorsiflexão a ponta do pé sobe em direção à perna, então

    $$
    \theta_{{tornozelo}} = \arctan\left(\frac{{\hat{{\mathbf{{e}}}}_{{1f}}\cdot\hat{{\mathbf{{e}}}}_{{2s}}}}{{\hat{{\mathbf{{e}}}}_{{1f}}\cdot\hat{{\mathbf{{e}}}}_{{1s}}}}\right)
    $$

    No ensaio, a maior dorsiflexão é de ${_ankle[_imax]:.1f}^\circ$, em $t = {t[_imax]:.3f}$ s, e a maior flexão plantar, de ${-_ankle[_imin]:.1f}^\circ$, em $t = {t[_imin]:.3f}$ s.

    Com a pessoa em pé, parada, o pé e a perna formam um ângulo de cerca de $90^\circ$, mas os eixos anatômicos não precisam ser exatamente perpendiculares: o eixo $x_f$ vai do calcâneo, na parte de trás e de cima do calcanhar, às cabeças dos metatarsos, mais baixas, e o eixo $y_s$ depende de onde estão a cabeça da fíbula e a tuberosidade da tíbia. Por isso, em biomecânica é comum subtrair o ângulo medido numa postura de referência, em pé, e relatar os ângulos da marcha em relação a ela.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Questões de verificação

    Pare aqui antes dos problemas.

    1. Volte ao Desafio 0. Onde você tinha posto o centro do quadril, e onde a regressão o pôs?
    2. Os versores de uma base passaram nos testes de norma e de ortogonalidade. O que ainda pode estar errado com ela?
    3. Por que foi preciso olhar para os dados para descobrir para onde aponta o eixo $\mathbf{Z}$ do laboratório? O que teria acontecido com as bases se não tivéssemos corrigido isso?
    4. Em cada segmento, um eixo foi obtido como normal de um plano definido por três marcadores. Que eixo foi esse na pelve, no pé, na perna e na coxa? E qual foi obtido por último, como produto vetorial dos outros dois?
    5. Por que os coeficientes da equação de Bell, Brand e Pedersen multiplicam os versores da pelve, e não os do laboratório?
    6. A base da coxa depende de um ponto estimado, a cabeça do fêmur. Que consequências isso tem para os ângulos do quadril e do joelho?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problemas

    1. Construa os sistemas de coordenadas anatômicos do pé, da perna e da coxa *esquerdos*, com os marcadores `L.Heel`, `L.MT1`, `L.MT2`, `L.MT5`, `L.HF`, `L.TT`, `L.Ankle`, `L.Ankle.Medial`, `L.Knee` e `L.Knee.Medial`, e com o centro do quadril esquerdo (que coeficiente da equação de Bell, Brand e Pedersen muda?). Os eixos $z$ continuam apontando para a direita. Em quais produtos vetoriais é preciso trocar a ordem? Verifique as bases com `check_basis`.

    2. Harrington et al. (2007) propuseram, a partir de imagens de ressonância magnética, outra regressão para o centro do quadril, em milímetros e na base da pelve: $x = -0{,}24\,PD - 9{,}9$, $y = -0{,}30\,PW - 10{,}9$ e $z = 0{,}33\,PW + 7{,}3$, onde $PW$ é a distância entre as espinhas ântero-superiores e $PD$ é a distância entre os pontos médios das espinhas ântero-superiores e das póstero-superiores. Calcule o centro do quadril direito com essa regressão. Quanto ele difere da estimativa de Bell, Brand e Pedersen? Quanto muda a direção do eixo $y_t$ da coxa?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Respostas finais dos problemas

    1. No lado esquerdo, o coeficiente lateral de Bell muda de +0,36 para −0,36. O eixo $z$ continua dirigido à direita: inverta a diferença medial/lateral usada para orientar o pé e as linhas de referência da perna e da coxa; feche sempre a tríade com $x\times y=z$. A base correta satisfaz $RR^T=I$ e $\det R=+1$; espelhar apenas uma coluna produz uma reflexão, não uma rotação.
    2. No primeiro quadro, centro direito de Harrington aproximadamente $(−0,217337,0,969077,0,451764)$ m. Diferença para Bell: 16,16 mm no primeiro quadro e média de 16,32 mm no ensaio. O eixo longo muda 1,59° no primeiro quadro e até 2,08° no registro, usando as regressões e a origem especificadas no enunciado.

    ///
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Para aprofundar

    - [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py): a construção de uma base a partir de três marcadores e como os erros dos marcadores se tornam erros nos eixos.
    - [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py): referenciais, bases e as convenções anatômicas para a direção de cada eixo.
    - [Rigid-body transformations in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py): matrizes de rotação e ângulos de Cardan, o próximo passo para descrever as articulações.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Videoaulas na Internet

    [Lynch e Park: matrizes de rotação](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-1-of-2/) apresenta ortogonalidade, orientação dextrógira e o significado das colunas da matriz. O vídeo está em inglês; compare a convenção de colunas com os versores nas linhas usados aqui.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Referências

    - Bell AL, Brand RA, Pedersen DR (1989) Prediction of hip joint centre location from external landmarks. Human Movement Science, 8(1), 3–16.

    - Cappozzo A, Catani F, Della Croce U, Leardini A (1995) Position and orientation in space of bones during movement: anatomical frame definition and determination. Clinical Biomechanics, 10(4), 171–178.

    - Harrington ME, Zavatsky AB, Lawson SEM, Yuan Z, Theologis TN (2007) Prediction of the hip joint centre in adults, children, and patients with cerebral palsy based on magnetic resonance imaging. Journal of Biomechanics, 40(3), 595–602.

    - Hilal I, Van Sint Jan S, Leardini A, Della Croce U. [D3.2. Technical report on data collection procedure, Annex I](https://github.com/BMClab/BMC/blob/master/courses/refs/VAKHUM.pdf). VAKHUM project (IST-1999-10954).
    """)
    return


if __name__ == "__main__":
    app.run()
