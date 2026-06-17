import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Modelos Viscoelásticos de Tecidos Biológicos

    Este notebook implementa os modelos viscoelásticos apresentados no Capítulo 2 do livro
    *"Dynamic Modeling of Musculoskeletal Motion: A Vectorized Approach"* de Gary T. Yamaguchi.

    Modelos de parâmetros concentrados (*lumped-parameter models*) são utilizados para descrever
    o comportamento viscoelástico de tecidos biológicos moles (músculos, tendões, ligamentos).

    ### Conteúdo:
    1. **Elementos básicos** — Mola ideal e Amortecedor ideal (dashpot)
    2. **Modelo de Maxwell** — Mola e amortecedor em série
    3. **Modelo de Voigt** — Mola e amortecedor em paralelo
    4. **Comparação com tecido real** — Limitações de Maxwell e Voigt
    5. **Modelo de Kelvin (Sólido Linear Padrão)** — Mola em paralelo com ramo série (mola + amortecedor)
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({
        'figure.figsize': (10, 5),
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
    })
    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Elementos Mecânicos Básicos

    Os modelos viscoelásticos são construídos a partir de dois elementos idealizados fundamentais:

    ### 1.1 Mola Ideal (Spring)

    Uma mola linear ideal gera uma força proporcional à sua deformação:

    $$F_k = k \cdot x$$

    onde:
    - $F_k$ é a força exercida pela mola
    - $k$ é a constante de rigidez (stiffness)
    - $x$ é a extensão além do comprimento relaxado

    **Propriedades ideais:**
    - Sem massa
    - Deformação instantânea quando submetida a uma força
    - Sem limite de extensão
    - Sem deformação residual
    - Sem dissipação de energia

    ### 1.2 Amortecedor Ideal (Dashpot)

    Um amortecedor ideal gera uma força proporcional à sua velocidade de deformação:

    $$F_b = b \cdot \dot{x}$$

    onde:
    - $F_b$ é a força exercida pelo amortecedor
    - $b$ é o coeficiente de amortecimento viscoso
    - $\dot{x}$ é a taxa de extensão (velocidade)

    **Propriedades ideais:**
    - Sem massa
    - **Não pode ser deformado instantaneamente** por uma força finita
    - Dissipa energia na forma de calor
    """)
    return


@app.cell
def _(np, plt):
    fig_elem, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Mola
    x_spring = np.linspace(0, 2, 100)
    k_values = [1, 2, 4]
    for _k in k_values:
        ax1.plot(x_spring, _k * x_spring, label=f'k = {_k} N/m')
    ax1.set_xlabel('Deformação x (m)')
    ax1.set_ylabel('Força $F_k$ (N)')
    ax1.set_title('Mola Ideal: $F_k = kx$')
    ax1.legend()

    # Amortecedor
    xdot = np.linspace(0, 2, 100)
    b_values = [1, 2, 4]
    for _b in b_values:
        ax2.plot(xdot, _b * xdot, label=f'b = {_b} N·s/m')
    ax2.set_xlabel(r'Velocidade $\dot{x}$ (m/s)')
    ax2.set_ylabel(r'Força $F_b$ (N)')
    ax2.set_title(r'Amortecedor Ideal: $F_b = b\dot{x}$')
    ax2.legend()

    fig_elem.tight_layout()
    fig_elem
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercícios — Elementos Básicos**

    1. Uma mola com $k = 5$ N/m é deformada em $x = 0.3$ m. Qual a força exercida pela mola?

    2. Um amortecedor com $b = 8$ N·s/m é submetido a uma velocidade de extensão $\dot{x} = 0.5$ m/s. Qual a força no amortecedor?

    3. Se dobrarmos a constante de rigidez $k$ de uma mola, o que acontece com a energia armazenada $E = \frac{1}{2}kx^2$ para a mesma deformação $x$?

    4. Um amortecedor é um elemento **dissipativo**. O que isso significa em termos de energia? A energia pode ser recuperada?

    5. Qual dos dois elementos (mola ou amortecedor) pode mudar a força **instantaneamente** quando o comprimento muda? Qual elemento faz a resposta depender do **tempo**?

    6. Em um arranjo em **série**, qual grandeza é compartilhada entre os elementos (força ou deformação)? E em um arranjo em **paralelo**?
    """), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Modelo de Maxwell

    O modelo de Maxwell consiste em uma **mola e um amortecedor conectados em série**.

    ### Esquema:
    ```
    ──/\/\/──[████]──
      mola   amortecedor
      (k)      (b)
    ```

    ### Relações fundamentais:

    Em uma conexão em série:
    - As **forças** na mola e no amortecedor são **iguais**: $F = F_k = F_b$
    - As **deformações** são **aditivas**: $x = x_k + x_b$

    ### Equação diferencial governante:

    Derivando a deformação total:

    $$\dot{x} = \dot{x}_k + \dot{x}_b = \frac{\dot{F}}{k} + \frac{F}{b}$$

    Reorganizando:

    $$\boxed{\dot{F} + \frac{k}{b}F = k\dot{x}}$$

    ### Caso I: Resposta a Carga Constante ($F$ = constante, $\dot{F} = 0$)

    A equação se reduz a $\frac{F}{b} = \dot{x}$, com solução:

    $$\boxed{x(t) = \frac{F}{k} + \frac{F}{b}t}$$

    O primeiro termo é a deformação instantânea da mola; o segundo é a deformação linear
    crescente do amortecedor. **O tecido se deforma indefinidamente** — comportamento
    inadequado para longo prazo.

    ### Caso II: Deformação Constante ($x = x_0$ = constante, $\dot{x} = 0$)

    A equação se reduz a $\dot{F} + \frac{k}{b}F = 0$, com solução:

    $$\boxed{F(t) = kx_0 \cdot e^{-t/\tau}}$$

    onde $\tau = b/k$ é a constante de tempo. A força **decai exponencialmente a zero** —
    também inadequado, pois tecidos reais mantêm uma tensão residual.
    """)
    return


@app.cell
def _(mo):
    slider_k_maxwell = mo.ui.slider(0.5, 10.0, step=0.5, value=2.0, label="k (N/m)")
    slider_b_maxwell = mo.ui.slider(0.5, 20.0, step=0.5, value=5.0, label="b (N·s/m)")
    slider_F_maxwell = mo.ui.slider(1.0, 20.0, step=1.0, value=10.0, label="F (N)")
    slider_x0_maxwell = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="x₀ (m)")

    mo.md(f"""
    ### Parâmetros do Modelo de Maxwell

    {mo.hstack([slider_k_maxwell, slider_b_maxwell, slider_F_maxwell, slider_x0_maxwell])}
    """)
    return (
        slider_F_maxwell,
        slider_b_maxwell,
        slider_k_maxwell,
        slider_x0_maxwell,
    )


@app.cell
def _(
    np,
    plt,
    slider_F_maxwell,
    slider_b_maxwell,
    slider_k_maxwell,
    slider_x0_maxwell,
):
    k_m = slider_k_maxwell.value
    b_m = slider_b_maxwell.value
    F_m = slider_F_maxwell.value
    x0_m = slider_x0_maxwell.value
    tau_m = b_m / k_m

    T_final_m = 10 * tau_m
    N_m = 2000
    dt_m = T_final_m / N_m
    t_maxwell = np.zeros(N_m + 1)

    # Caso I: Carga constante — EDO: dx/dt = F/b
    # Condição inicial: x(0) = F/k (deformação instantânea da mola)
    x_maxwell_const_F = np.zeros(N_m + 1)
    x_maxwell_const_F[0] = F_m / k_m  # deformação instantânea da mola
    for _i in range(N_m):
        t_maxwell[_i + 1] = t_maxwell[_i] + dt_m
        _dxdt = F_m / b_m
        x_maxwell_const_F[_i + 1] = x_maxwell_const_F[_i] + _dxdt * dt_m

    # Caso II: Deformação constante — EDO: dF/dt = -(k/b)*F
    # Condição inicial: F(0) = k*x0
    F_maxwell_const_x = np.zeros(N_m + 1)
    F_maxwell_const_x[0] = k_m * x0_m
    for _i in range(N_m):
        _dFdt = -(k_m / b_m) * F_maxwell_const_x[_i]
        F_maxwell_const_x[_i + 1] = F_maxwell_const_x[_i] + _dFdt * dt_m

    fig_maxwell, (ax_m1, ax_m2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Caso I
    ax_m1.plot(t_maxwell, x_maxwell_const_F, color='#2196F3', linewidth=2.5)
    ax_m1.axhline(y=F_m / k_m, color='gray', linestyle='--', alpha=0.5, label=f'Deformação inicial = F/k = {F_m/k_m:.2f}')
    ax_m1.set_xlabel('Tempo (s)')
    ax_m1.set_ylabel('Deformação x(t) (m)')
    ax_m1.set_title(f'Maxwell — Carga Constante (F = {F_m} N)\nτ = b/k = {tau_m:.2f} s  [Euler, dt = {dt_m:.4f} s]')
    ax_m1.legend(fontsize=10)
    ax_m1.set_ylim(bottom=0)

    # Plot Caso II
    ax_m2.plot(t_maxwell, F_maxwell_const_x, color='#F44336', linewidth=2.5)
    ax_m2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_m2.axvline(x=tau_m, color='orange', linestyle=':', alpha=0.7, label=f'τ = {tau_m:.2f} s')
    ax_m2.set_xlabel('Tempo (s)')
    ax_m2.set_ylabel('Força F(t) (N)')
    ax_m2.set_title(f'Maxwell — Deformação Constante (x₀ = {x0_m} m)\nτ = b/k = {tau_m:.2f} s  [Euler, dt = {dt_m:.4f} s]')
    ax_m2.legend(fontsize=10)
    ax_m2.set_ylim(bottom=-0.1)

    fig_maxwell.tight_layout()
    fig_maxwell
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercícios — Modelo de Maxwell**

    1. Para um modelo de Maxwell com $k = 4$ N/m e $b = 12$ N·s/m, calcule a constante de tempo $\tau$.

    2. Sob carga constante $F = 8$ N com $k = 2$ N/m e $b = 10$ N·s/m:
        - Qual a deformação instantânea em $t = 0$?
        - Qual a velocidade de deformação do amortecedor?
        - Qual a deformação total em $t = 5$ s?

    3. Sob deformação constante $x_0 = 2$ m com $k = 3$ N/m e $b = 6$ N·s/m:
        - Qual a força inicial $F(0)$?
        - Qual a força em $t = \tau$? (Lembre-se: $e^{-1} \approx 0.368$)
        - Após quanto tempo a força cai para metade do valor inicial?

    4. Por que o modelo de Maxwell é inadequado para descrever o comportamento de longo prazo de tecidos biológicos sob carga constante?

    5. Modifique os parâmetros nos sliders acima. O que acontece com a curva de relaxação quando $b$ aumenta (mantendo $k$ fixo)? Por quê?

    6. No gráfico de carga constante, identifique qual parte da deformação total vem da **mola** e qual vem do **amortecedor**. Dobre $b$ e depois dobre $k$ — qual mudança altera a inclinação da curva?

    7. **Auditoria de parâmetros:** Na resposta a deformação constante, dobrar $b$ ou dobrar $k$ — qual das duas mudanças *aumenta* a constante de tempo de relaxação $\tau$?
    """), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Modelo de Voigt

    O modelo de Voigt consiste em uma **mola e um amortecedor conectados em paralelo**.

    ### Esquema:
    ```
        ┌──/\/\/──┐
    ────┤  mola(k) ├────
        └──[████]──┘
        amortecedor(b)
    ```

    ### Relações fundamentais:

    Em uma conexão em paralelo:
    - As **deformações** na mola e no amortecedor são **iguais**: $x = x_k = x_b$
    - As **forças** são **aditivas**: $F = F_k + F_b$

    ### Equação diferencial governante:

    $$\boxed{F = kx + b\dot{x}}$$

    ### Caso I: Resposta a Carga Constante ($F$ = constante, $\dot{F} = 0$)

    A solução, com condição inicial $x(0) = 0$ (o amortecedor impede deformação instantânea):

    $$\boxed{x(t) = \frac{F}{k}\left(1 - e^{-t/\tau}\right)}$$

    onde $\tau = b/k$. A deformação **cresce exponencialmente até o valor assintótico** $F/k$.
    O comportamento de longo prazo é bom, mas **falta a deformação instantânea** observada
    em tecidos reais.

    ### Caso II: Resposta a Deformação Constante ($\dot{x} = 0$)

    A equação se reduz trivialmente a:

    $$\boxed{F = kx}$$

    A resposta da força é **constante e trivial**. Porém, para impor uma deformação
    instantânea no modelo de Voigt, seria necessária uma **força infinita** (pois o
    amortecedor impede deformação instantânea), o que não é fisicamente realizável.
    """)
    return


@app.cell
def _(mo):
    slider_k_voigt = mo.ui.slider(0.5, 10.0, step=0.5, value=2.0, label="k (N/m)")
    slider_b_voigt = mo.ui.slider(0.5, 20.0, step=0.5, value=5.0, label="b (N·s/m)")
    slider_F_voigt = mo.ui.slider(1.0, 20.0, step=1.0, value=10.0, label="F (N)")

    mo.md(f"""
    ### Parâmetros do Modelo de Voigt

    {mo.hstack([slider_k_voigt, slider_b_voigt, slider_F_voigt])}
    """)
    return slider_F_voigt, slider_b_voigt, slider_k_voigt


@app.cell
def _(np, plt, slider_F_voigt, slider_b_voigt, slider_k_voigt):
    k_v = slider_k_voigt.value
    b_v = slider_b_voigt.value
    F_v = slider_F_voigt.value
    tau_v = b_v / k_v

    T_final_v = 10 * tau_v
    N_v = 2000
    dt_v = T_final_v / N_v

    # Caso I: Carga constante — EDO: dx/dt = (F - k*x) / b
    # Condição inicial: x(0) = 0 (amortecedor impede deformação instantânea)
    t_voigt = np.zeros(N_v + 1)
    x_voigt_const_F = np.zeros(N_v + 1)
    x_voigt_const_F[0] = 0.0
    for _i in range(N_v):
        t_voigt[_i + 1] = t_voigt[_i] + dt_v
        _dxdt = (F_v - k_v * x_voigt_const_F[_i]) / b_v
        x_voigt_const_F[_i + 1] = x_voigt_const_F[_i] + _dxdt * dt_v

    fig_voigt, (ax_v1, ax_v2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Caso I
    ax_v1.plot(t_voigt, x_voigt_const_F, color='#4CAF50', linewidth=2.5)
    ax_v1.axhline(y=F_v / k_v, color='gray', linestyle='--', alpha=0.5, label=f'Assíntota = F/k = {F_v/k_v:.2f}')
    ax_v1.axvline(x=tau_v, color='orange', linestyle=':', alpha=0.7, label=f'τ = {tau_v:.2f} s')
    ax_v1.set_xlabel('Tempo (s)')
    ax_v1.set_ylabel('Deformação x(t) (m)')
    ax_v1.set_title(f'Voigt — Carga Constante (F = {F_v} N)\nτ = b/k = {tau_v:.2f} s  [Euler, dt = {dt_v:.4f} s]')
    ax_v1.legend(fontsize=10)
    ax_v1.set_ylim(bottom=0)

    # Plot Caso II (trivial — não precisa de integração numérica)
    x_val_v = 1.0
    t_voigt2 = np.linspace(0, 5, 500)
    F_voigt_const_x = np.ones_like(t_voigt2) * k_v * x_val_v
    ax_v2.plot(t_voigt2, F_voigt_const_x, color='#F44336', linewidth=2.5, label=f'F = kx = {k_v*x_val_v:.1f} N')
    ax_v2.set_xlabel('Tempo (s)')
    ax_v2.set_ylabel('Força F(t) (N)')
    ax_v2.set_title(f'Voigt — Deformação Constante (x = {x_val_v} m)\nF = kx (constante, trivial)')
    ax_v2.legend(fontsize=10)
    ax_v2.set_ylim(bottom=0, top=k_v * x_val_v * 3.5)

    # Seta estilo impulso de Dirac em t = 0 representando F → ∞
    _F_base = k_v * x_val_v
    _F_top = _F_base * 3.0
    ax_v2.annotate('', xy=(0, _F_top), xytext=(0, _F_base),
                   arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2.5,
                                   mutation_scale=20))
    # Triângulo preenchido na base da seta (como na notação de impulso)
    ax_v2.plot(0, _F_base, marker='^', markersize=0, color='#D32F2F')
    # Linha vertical do impulso
    ax_v2.plot([0, 0], [0, _F_base], color='#D32F2F', linewidth=2.5)

    ax_v2.annotate(r'$F \to \infty$ (impulso)' + '\n' + r'$b\dot{x}\,\delta(t)$',
                   xy=(0, _F_top), xytext=(1.5, _F_top * 0.9),
                   arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
                   fontsize=11, color='#D32F2F', ha='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#D32F2F', alpha=0.8))

    fig_voigt.tight_layout()
    fig_voigt
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercícios — Modelo de Voigt**

    1. Para um modelo de Voigt com $k = 3$ N/m, $b = 9$ N·s/m sob carga constante $F = 15$ N:
        - Qual a constante de tempo $\tau$?
        - Qual o valor assintótico da deformação $x(\infty)$?
        - Qual a deformação em $t = \tau$? (Use $1 - e^{-1} \approx 0.632$)

    2. No modelo de Voigt, por que a deformação inicial é zero ($x(0) = 0$)? Qual elemento impede a deformação instantânea?

    3. Explique fisicamente por que seria necessária uma **força infinita** para impor uma deformação instantânea no modelo de Voigt.

    4. Compare a EDO do Voigt ($F = kx + b\dot{x}$) com a do Maxwell ($\dot{F} + \frac{k}{b}F = k\dot{x}$). Qual é de primeira ordem em $x$? Qual é de primeira ordem em $F$?

    5. Usando os sliders, observe o que acontece quando $b \to 0$. A resposta do Voigt se aproxima de qual elemento simples?

    6. **Problema de projeto:** Escolha valores de $k$ e $b$ para que o modelo de Voigt atinja **95%** da deformação final em $t = 3$ s sob carga constante. *Dica: $1 - e^{-3} \approx 0.95$, portanto $\tau$ deve valer...*
    """), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Comparação: Maxwell vs Voigt vs Tecido Real

    A tabela abaixo resume as limitações de cada modelo comparado com o comportamento
    real de tecidos biológicos (Figura 2.16):

    | Comportamento | Tecido Real | Maxwell | Voigt |
    |:---|:---:|:---:|:---:|
    | **Carga constante — deformação instantânea** | ✅ Sim | ✅ Sim | ❌ Não |
    | **Carga constante — assíntota finita** | ✅ Sim | ❌ Não (cresce linearmente) | ✅ Sim |
    | **Deformação constante — força inicial** | ✅ Finita | ✅ Finita | ❌ Infinita |
    | **Deformação constante — relaxação para valor >0** | ✅ Sim | ❌ Não (decai a zero) | ✅ Sim (trivial) |

    **Conclusão:** Nenhum dos dois modelos sozinho consegue reproduzir adequadamente
    o comportamento de tecidos biológicos em **todas** as condições. Precisamos de um
    modelo que combine as qualidades de ambos — esse é o **Modelo de Kelvin** (Seção 5).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Comportamento real de tecidos biológicos (Figura 2.16 — Yamaguchi)

    A figura abaixo, extraída do livro, mostra o comportamento **experimentalmente observado**
    em tecidos biológicos reais:

    - **A — Carga constante (*creep*):** deformação instantânea + deformação lenta até assíntota finita
    - **B — Deformação constante (*relaxação*):** força inicial alta que decai até valor assintótico **não nulo**
    """)
    return


@app.cell
def _(mo):
    _img = mo.image(src="https://raw.githubusercontent.com/BMClab/BMC/refs/heads/master/notebooks_marimo/fig2_16.png", width=600)
    mo.vstack([
        _img,
        mo.md("*Figura 2.16 — Respostas de tecidos fisiológicos a carga e deformação constantes (Yamaguchi)*"),
        mo.md(r"""
        Observe como **nenhum** dos modelos abaixo (Maxwell ou Voigt) consegue reproduzir
        **simultaneamente** os dois comportamentos mostrados na figura.
        """)
    ])
    return


@app.cell
def _(np, plt):
    # Comparação visual: Maxwell vs Voigt (sem Kelvin — ainda não apresentado)
    # Todos resolvidos pelo método de Euler
    k_comp = 2.0
    b_comp = 5.0
    F_comp = 10.0
    x0_comp = 1.0
    tau_comp = b_comp / k_comp

    T_final_comp = 10 * tau_comp
    N_comp = 3000
    dt_comp = T_final_comp / N_comp
    t_comp = np.zeros(N_comp + 1)

    # --- Carga Constante (Euler) ---
    # Maxwell: dx/dt = F/b, x(0) = F/k
    x_maxwell_comp = np.zeros(N_comp + 1)
    x_maxwell_comp[0] = F_comp / k_comp
    # Voigt: dx/dt = (F - k*x)/b, x(0) = 0
    x_voigt_comp = np.zeros(N_comp + 1)

    for _i in range(N_comp):
        t_comp[_i + 1] = t_comp[_i] + dt_comp
        # Maxwell
        x_maxwell_comp[_i + 1] = x_maxwell_comp[_i] + (F_comp / b_comp) * dt_comp
        # Voigt
        x_voigt_comp[_i + 1] = x_voigt_comp[_i] + ((F_comp - k_comp * x_voigt_comp[_i]) / b_comp) * dt_comp

    # --- Deformação Constante (Euler) ---
    # Maxwell: dF/dt = -(k/b)*F, F(0) = k*x0
    F_maxwell_comp = np.zeros(N_comp + 1)
    F_maxwell_comp[0] = k_comp * x0_comp
    # Voigt: F = k*x (trivial, constante)
    F_voigt_comp = np.ones(N_comp + 1) * k_comp * x0_comp

    for _i in range(N_comp):
        # Maxwell
        F_maxwell_comp[_i + 1] = F_maxwell_comp[_i] + (-(k_comp / b_comp) * F_maxwell_comp[_i]) * dt_comp

    fig_comp, axes_comp = plt.subplots(2, 2, figsize=(14, 10))

    axes_comp[0, 0].plot(t_comp, x_maxwell_comp, '--', color='#2196F3', linewidth=2, label='Maxwell')
    axes_comp[0, 0].plot(t_comp, x_voigt_comp, '--', color='#4CAF50', linewidth=2, label='Voigt')
    axes_comp[0, 0].set_xlabel('Tempo (s)')
    axes_comp[0, 0].set_ylabel('Deformação x(t)')
    axes_comp[0, 0].set_title('Carga Constante — Deformação x(t)  [Euler]')
    axes_comp[0, 0].legend()
    axes_comp[0, 0].annotate('Maxwell: cresce\nindefinidamente ❌',
                             xy=(t_comp[-1], x_maxwell_comp[-1]),
                             xytext=(t_comp[-1]*0.5, x_maxwell_comp[-1]*0.8),
                             fontsize=9, color='#1565C0',
                             arrowprops=dict(arrowstyle='->', color='#1565C0'))
    axes_comp[0, 0].annotate('Voigt: sem deformação\ninstantânea ❌',
                             xy=(0, 0), xytext=(t_comp[-1]*0.3, x_voigt_comp[-1]*0.4),
                             fontsize=9, color='#2E7D32',
                             arrowprops=dict(arrowstyle='->', color='#2E7D32'))

    axes_comp[0, 1].axhline(y=F_comp, color='black', linewidth=2)
    axes_comp[0, 1].set_xlabel('Tempo (s)')
    axes_comp[0, 1].set_ylabel('Força F (N)')
    axes_comp[0, 1].set_title(f'Carga Aplicada (F = {F_comp} N)')
    axes_comp[0, 1].set_ylim(0, F_comp * 1.5)

    axes_comp[1, 0].plot(t_comp, F_maxwell_comp, '--', color='#2196F3', linewidth=2, label='Maxwell')
    axes_comp[1, 0].plot(t_comp, F_voigt_comp, '--', color='#4CAF50', linewidth=2, label='Voigt')
    axes_comp[1, 0].set_xlabel('Tempo (s)')
    axes_comp[1, 0].set_ylabel('Força F(t) (N)')
    axes_comp[1, 0].set_title('Deformação Constante — Força F(t)  [Euler]')
    axes_comp[1, 0].legend()
    axes_comp[1, 0].annotate('Maxwell: decai a zero ❌',
                             xy=(t_comp[-1], 0),
                             xytext=(t_comp[-1]*0.5, F_maxwell_comp[0]*0.3),
                             fontsize=9, color='#1565C0',
                             arrowprops=dict(arrowstyle='->', color='#1565C0'))

    axes_comp[1, 1].axhline(y=x0_comp, color='black', linewidth=2)
    axes_comp[1, 1].set_xlabel('Tempo (s)')
    axes_comp[1, 1].set_ylabel('Deformação x (m)')
    axes_comp[1, 1].set_title(f'Deformação Aplicada (x₀ = {x0_comp} m)')
    axes_comp[1, 1].set_ylim(0, x0_comp * 1.5)

    fig_comp.suptitle('Comparação: Maxwell vs Voigt  [Método de Euler]', fontsize=14, fontweight='bold', y=1.01)
    fig_comp.tight_layout()
    fig_comp
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Modelo de Kelvin

    O modelo de Kelvin consiste em uma **mola ($k_p$) em paralelo** com um **ramo série
    de mola ($k_s$) e amortecedor ($b$)**, conforme a Figura 2.17C do livro.

    ### Esquema (Fig. 2.17C — Yamaguchi):
    ```
         ┌──/\/\/──[████]──┐
    o────┤   ks       b    ├────o
         └────/\/\/────────┘
               kp
    ```

    - Ramo superior: mola $k_s$ em **série** com amortecedor $b$ (modelo de Maxwell)
    - Ramo inferior: mola $k_p$ em **paralelo**

    ### Relações fundamentais:

    - $F = F_{k_p} + F_{serie}$ (força total = mola paralela + ramo série)
    - $F_{serie} = F_{k_s} = F_b$ (no ramo série, as forças são iguais)
    - $x = x_{k_p} = x_{k_s} + x_b$ (mesma deformação total; no ramo série, deformações se somam)

    ### Equação diferencial governante:

    $$\boxed{F + \tau_\varepsilon \dot{F} = k_s\left(x + \tau_\sigma \dot{x}\right)}$$

    onde as constantes de tempo são:

    $$\tau_\varepsilon = \frac{b}{k_p} \quad \text{(constante de tempo para deformação constante — relaxação)}$$

    $$\tau_\sigma = \frac{b(k_p + k_s)}{k_p \cdot k_s} \quad \text{(constante de tempo para carga constante — creep)}$$

    Note que $\tau_\sigma > \tau_\varepsilon$ sempre (pois $k_p + k_s > k_p$).

    ### Caso I: Carga Constante ($F$ = constante, $\dot{F} = 0$)

    $$\boxed{x(t) = \frac{F}{k_s} + \frac{F}{k_p}\left(1 - e^{-t/\tau_\sigma}\right)}$$

    - Em $t = 0$: deformação instantânea $x(0) = F/k_s$ (mola em série)
    - Em $t \to \infty$: $x(\infty) = F/k_s + F/k_p$ (assíntota finita) ✅

    ### Caso II: Deformação Constante ($\dot{x} = 0$)

    $$\boxed{F(t) = \frac{k_p k_s}{k_p + k_s}x + \left(k_s - \frac{k_p k_s}{k_p + k_s}\right)x \cdot e^{-t/\tau_\varepsilon}}$$

    - Em $t = 0$: $F(0) = k_s \cdot x$ (mola série reage instantaneamente) ✅
    - Em $t \to \infty$: $F(\infty) = \frac{k_p k_s}{k_p + k_s} x > 0$ (relaxação para valor não-nulo) ✅
    """)
    return


@app.cell
def _(mo):
    slider_ks_kelvin = mo.ui.slider(0.5, 10.0, step=0.5, value=3.0, label="ks (N/m)")
    slider_kp_kelvin = mo.ui.slider(0.5, 10.0, step=0.5, value=1.5, label="kp (N/m)")
    slider_b_kelvin = mo.ui.slider(0.5, 20.0, step=0.5, value=5.0, label="b (N·s/m)")
    slider_F_kelvin = mo.ui.slider(1.0, 20.0, step=1.0, value=10.0, label="F (N)")
    slider_x0_kelvin = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="x₀ (m)")

    mo.md(f"""
    ### Parâmetros do Modelo de Kelvin

    {mo.hstack([slider_ks_kelvin, slider_kp_kelvin, slider_b_kelvin])}
    {mo.hstack([slider_F_kelvin, slider_x0_kelvin])}
    """)
    return (
        slider_F_kelvin,
        slider_b_kelvin,
        slider_kp_kelvin,
        slider_ks_kelvin,
        slider_x0_kelvin,
    )


@app.cell
def _(
    np,
    plt,
    slider_F_kelvin,
    slider_b_kelvin,
    slider_kp_kelvin,
    slider_ks_kelvin,
    slider_x0_kelvin,
):
    ks_kel = slider_ks_kelvin.value
    kp_kel = slider_kp_kelvin.value
    b_kel = slider_b_kelvin.value
    F_kel = slider_F_kelvin.value
    x0_kel = slider_x0_kelvin.value

    tau_eps_kel = b_kel / kp_kel
    tau_sig_kel = b_kel * (kp_kel + ks_kel) / (kp_kel * ks_kel)
    k_eq_kel = (kp_kel * ks_kel) / (kp_kel + ks_kel)

    T_final_kel = 10 * tau_sig_kel
    N_kel = 3000
    dt_kel = T_final_kel / N_kel
    t_kelvin = np.zeros(N_kel + 1)

    # Caso I: Carga constante — Euler
    # Estado: x_b (deformação da parte Voigt: amortecedor + mola paralela)
    # EDO: dx_b/dt = (F - kp*x_b) / b
    # x_total = F/ks + x_b (mola série deforma instantaneamente)
    x_b_kel = np.zeros(N_kel + 1)
    x_kelvin_const_F = np.zeros(N_kel + 1)
    x_kelvin_const_F[0] = F_kel / ks_kel  # deformação instantânea da mola série
    for _i in range(N_kel):
        t_kelvin[_i + 1] = t_kelvin[_i] + dt_kel
        _dx_b_dt = (F_kel - kp_kel * x_b_kel[_i]) / b_kel
        x_b_kel[_i + 1] = x_b_kel[_i] + _dx_b_dt * dt_kel
        x_kelvin_const_F[_i + 1] = F_kel / ks_kel + x_b_kel[_i + 1]

    # Caso II: Deformação constante — Euler
    # EDO: dF/dt = -(F - k_eq*x0) / tau_eps
    # Condição inicial: F(0) = ks*x0
    F_kelvin_const_x = np.zeros(N_kel + 1)
    F_kelvin_const_x[0] = ks_kel * x0_kel
    for _i in range(N_kel):
        _dFdt = -(F_kelvin_const_x[_i] - k_eq_kel * x0_kel) / tau_eps_kel
        F_kelvin_const_x[_i + 1] = F_kelvin_const_x[_i] + _dFdt * dt_kel

    fig_kelvin, (ax_k1, ax_k2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Caso I
    ax_k1.plot(t_kelvin, x_kelvin_const_F, color='#9C27B0', linewidth=2.5)
    ax_k1.axhline(y=F_kel / ks_kel, color='#E91E63', linestyle=':', alpha=0.7, label=f'Deformação instantânea = F/ks = {F_kel/ks_kel:.2f}')
    ax_k1.axhline(y=F_kel / ks_kel + F_kel / kp_kel, color='gray', linestyle='--', alpha=0.5,
                  label=f'Assíntota = F/ks + F/kp = {F_kel/ks_kel + F_kel/kp_kel:.2f}')
    ax_k1.axvline(x=tau_sig_kel, color='orange', linestyle=':', alpha=0.7, label=f'τ_σ = {tau_sig_kel:.2f} s')
    ax_k1.set_xlabel('Tempo (s)')
    ax_k1.set_ylabel('Deformação x(t) (m)')
    ax_k1.set_title(f'Kelvin — Carga Constante (F = {F_kel} N)  [Euler]')
    ax_k1.legend(fontsize=9)
    ax_k1.set_ylim(bottom=0)

    # Plot Caso II
    ax_k2.plot(t_kelvin, F_kelvin_const_x, color='#FF5722', linewidth=2.5)
    ax_k2.axhline(y=ks_kel * x0_kel, color='#E91E63', linestyle=':', alpha=0.7, label=f'F(0) = ks·x₀ = {ks_kel*x0_kel:.2f}')
    ax_k2.axhline(y=k_eq_kel * x0_kel, color='gray', linestyle='--', alpha=0.5,
                  label=f'F(∞) = kp·ks/(kp+ks)·x₀ = {k_eq_kel*x0_kel:.2f}')
    ax_k2.axvline(x=tau_eps_kel, color='orange', linestyle=':', alpha=0.7, label=f'τ_ε = {tau_eps_kel:.2f} s')
    ax_k2.set_xlabel('Tempo (s)')
    ax_k2.set_ylabel('Força F(t) (N)')
    ax_k2.set_title(f'Kelvin — Deformação Constante (x₀ = {x0_kel} m)  [Euler]')
    ax_k2.legend(fontsize=9)
    ax_k2.set_ylim(bottom=0)

    fig_kelvin.tight_layout()
    fig_kelvin
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ✅ Kelvin reproduz o comportamento do tecido real

    Compare os gráficos acima com a **Figura 2.16** (Seção 4):

    | Comportamento | Tecido Real (Fig. 2.16) | Kelvin |
    |:---|:---:|:---:|
    | Deformação instantânea sob carga constante | ✅ | ✅ ($x(0) = F/k_s$) |
    | Assíntota finita sob carga constante | ✅ | ✅ ($x(\infty) = F/k_s + F/k_p$) |
    | Força inicial finita sob deformação constante | ✅ | ✅ ($F(0) = k_s x_0$) |
    | Relaxação para valor **não nulo** | ✅ | ✅ ($F(\infty) = k_{eq} x_0 > 0$) |

    O modelo de Kelvin é o **mais simples** que reproduz **qualitativamente** todos
    os comportamentos observados em tecidos biológicos reais.
    """)
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercícios — Modelo de Kelvin**

    1. Para $k_s = 4$ N/m, $k_p = 2$ N/m, $b = 6$ N·s/m:
        - Calcule $\tau_\varepsilon$ e $\tau_\sigma$. Qual é maior? Por quê?
        - Sob carga $F = 10$ N, qual a deformação instantânea? E a deformação final?

    2. Sob deformação constante $x_0 = 1$ m com os parâmetros acima:
        - Qual a força inicial $F(0)$?
        - Qual a força de equilíbrio $F(\infty)$? (Calcule $k_{eq} = \frac{k_p k_s}{k_p + k_s}$)

    3. Mostre que $\tau_\sigma > \tau_\varepsilon$ sempre. *Dica: compare as expressões e use o fato de que $k_s > 0$.*

    4. O que acontece com o modelo de Kelvin quando $k_p \to \infty$? Ele se reduz a qual modelo mais simples?

    5. E quando $k_s \to \infty$? A qual modelo ele se reduz?

    6. **Problema de projeto:** Escolha valores de $k_s$, $k_p$ e $b$ para que o Kelvin tenha uma deformação instantânea **grande** mas apenas um pequeno creep adicional. *Dica: pense na relação entre $F/k_s$ e $F/k_p$.*

    7. O que acontece com o tamanho da deformação instantânea quando $k_s$ **aumenta**? Use os sliders para verificar.
    """), kind="warn")
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercícios — Comparação dos Modelos**

    1. Preencha a tabela abaixo com ✅ ou ❌ para cada modelo:

    | Comportamento | Maxwell | Voigt | Kelvin |
    |:---|:---:|:---:|:---:|
    | Deformação instantânea sob carga | ? | ? | ? |
    | Assíntota finita sob carga constante | ? | ? | ? |
    | Relaxação para valor $> 0$ | ? | ? | ? |

    2. Um tecido biológico é submetido a uma carga constante. Observa-se uma deformação instantânea seguida de uma deformação crescente que tende a um valor finito. Qual dos três modelos melhor descreve esse comportamento? Justifique.

    3. Por que precisamos de **pelo menos 3 parâmetros** (como no modelo de Kelvin) para reproduzir adequadamente o comportamento viscoelástico de tecidos biológicos?

    4. Escolha **uma** curva dos gráficos de comparação (Seção 4) e explique qual **elemento físico** é responsável: (a) pelo valor inicial, (b) pela inclinação, e (c) pelo valor final da curva.
    """), kind="warn")
    return


if __name__ == "__main__":
    app.run()
