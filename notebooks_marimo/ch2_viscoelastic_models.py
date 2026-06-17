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
    # Viscoelastic Models of Biological Tissues

    This notebook implements the viscoelastic models presented in Chapter 2 of the book
    *"Dynamic Modeling of Musculoskeletal Motion: A Vectorized Approach"* by Gary T. Yamaguchi.

    Lumped-parameter models are used to describe
    the viscoelastic behavior of soft biological tissues (muscles, tendons, ligaments).

    ### Contents:
    1. **Basic Elements** — Ideal Spring and Ideal Dashpot
    2. **Maxwell Model** — Spring and dashpot in series
    3. **Voigt Model** — Spring and dashpot in parallel
    4. **Comparison with real tissue** — Limitations of Maxwell and Voigt
    5. **Kelvin Model (Standard Linear Solid)** — Spring in parallel with series branch (spring + dashpot)
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
    ## 1. Basic Mechanical Elements

    Viscoelastic models are built from two fundamental idealized elements:

    ### 1.1 Ideal Spring

    An ideal linear spring generates a force proportional to its deformation:

    $$F_k = k \cdot x$$

    where:
    - $F_k$ is the force exerted by the spring
    - $k$ is the stiffness constant
    - $x$ is the extension beyond the relaxed length

    **Ideal properties:**
    - Massless
    - Instantaneous deformation when subjected to a force
    - No extension limit
    - No residual deformation
    - No energy dissipation

    ### 1.2 Ideal Dashpot

    An ideal dashpot generates a force proportional to its rate of deformation:

    $$F_b = b \cdot \dot{x}$$

    where:
    - $F_b$ is the force exerted by the dashpot
    - $b$ is the viscous damping coefficient
    - $\dot{x}$ is the rate of extension (velocity)

    **Ideal properties:**
    - Massless
    - **Cannot be deformed instantaneously** by a finite force
    - Dissipates energy as heat
    """)
    return


@app.cell
def _(np, plt):
    fig_elem, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Spring
    x_spring = np.linspace(0, 2, 100)
    k_values = [1, 2, 4]
    for _k in k_values:
        ax1.plot(x_spring, _k * x_spring, label=f'k = {_k} N/m')
    ax1.set_xlabel('Deformation x (m)')
    ax1.set_ylabel('Force $F_k$ (N)')
    ax1.set_title('Ideal Spring: $F_k = kx$')
    ax1.legend()

    # Dashpot
    xdot = np.linspace(0, 2, 100)
    b_values = [1, 2, 4]
    for _b in b_values:
        ax2.plot(xdot, _b * xdot, label=f'b = {_b} N·s/m')
    ax2.set_xlabel(r'Velocity $\dot{x}$ (m/s)')
    ax2.set_ylabel(r'Force $F_b$ (N)')
    ax2.set_title(r'Ideal Dashpot: $F_b = b\dot{x}$')
    ax2.legend()

    fig_elem.tight_layout()
    fig_elem
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercises — Basic Elements**

    1. A spring with $k = 5$ N/m is deformed by $x = 0.3$ m. What is the force exerted by the spring?

    2. A dashpot with $b = 8$ N·s/m is subjected to an extension velocity $\dot{x} = 0.5$ m/s. What is the force in the dashpot?

    3. If we double the stiffness constant $k$ of a spring, what happens to the stored energy $E = \frac{1}{2}kx^2$ for the same deformation $x$?

    4. A dashpot is a **dissipative** element. What does this mean in terms of energy? Can the energy be recovered?

    5. Which of the two elements (spring or dashpot) can change the force **instantaneously** when the length changes? Which element makes the response depend on **time**?

    6. In a **series** arrangement, which quantity is shared between the elements (force or deformation)? And in a **parallel** arrangement?
    """), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Maxwell Model

    The Maxwell model consists of a **spring and a dashpot connected in series**.

    ### Schematic:
    ```
    ──/\/\/──[████]──
      spring   dashpot
      (k)      (b)
    ```

    ### Fundamental relations:

    In a series connection:
    - The **forces** in the spring and dashpot are **equal**: $F = F_k = F_b$
    - The **deformations** are **additive**: $x = x_k + x_b$

    ### Governing differential equation:

    Differentiating the total deformation:

    $$\dot{x} = \dot{x}_k + \dot{x}_b = \frac{\dot{F}}{k} + \frac{F}{b}$$

    Rearranging:

    $$\boxed{\dot{F} + \frac{k}{b}F = k\dot{x}}$$

    ### Case I: Constant Load Response ($F$ = constant, $\dot{F} = 0$)

    The equation reduces to $\frac{F}{b} = \dot{x}$, with solution:

    $$\boxed{x(t) = \frac{F}{k} + \frac{F}{b}t}$$

    The first term is the instantaneous deformation of the spring; the second is the linearly
    increasing deformation of the dashpot. **The tissue deforms indefinitely** — inadequate
    behavior for long-term response.

    ### Case II: Constant Deformation ($x = x_0$ = constant, $\dot{x} = 0$)

    The equation reduces to $\dot{F} + \frac{k}{b}F = 0$, with solution:

    $$\boxed{F(t) = kx_0 \cdot e^{-t/\tau}}$$

    where $\tau = b/k$ is the time constant. The force **decays exponentially to zero** —
    also inadequate, since real tissues maintain a residual tension.
    """)
    return


@app.cell
def _(mo):
    slider_k_maxwell = mo.ui.slider(0.5, 10.0, step=0.5, value=2.0, label="k (N/m)")
    slider_b_maxwell = mo.ui.slider(0.5, 20.0, step=0.5, value=5.0, label="b (N·s/m)")
    slider_F_maxwell = mo.ui.slider(1.0, 20.0, step=1.0, value=10.0, label="F (N)")
    slider_x0_maxwell = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="x₀ (m)")

    mo.md(f"""
    ### Maxwell Model Parameters

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

    # Case I: Constant load — ODE: dx/dt = F/b
    # Initial condition: x(0) = F/k (instantaneous deformation of the spring)
    x_maxwell_const_F = np.zeros(N_m + 1)
    x_maxwell_const_F[0] = F_m / k_m  # instantaneous deformation of the spring
    for _i in range(N_m):
        t_maxwell[_i + 1] = t_maxwell[_i] + dt_m
        _dxdt = F_m / b_m
        x_maxwell_const_F[_i + 1] = x_maxwell_const_F[_i] + _dxdt * dt_m

    # Case II: Constant deformation — ODE: dF/dt = -(k/b)*F
    # Initial condition: F(0) = k*x0
    F_maxwell_const_x = np.zeros(N_m + 1)
    F_maxwell_const_x[0] = k_m * x0_m
    for _i in range(N_m):
        _dFdt = -(k_m / b_m) * F_maxwell_const_x[_i]
        F_maxwell_const_x[_i + 1] = F_maxwell_const_x[_i] + _dFdt * dt_m

    fig_maxwell, (ax_m1, ax_m2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Case I
    ax_m1.plot(t_maxwell, x_maxwell_const_F, color='#2196F3', linewidth=2.5)
    ax_m1.axhline(y=F_m / k_m, color='gray', linestyle='--', alpha=0.5, label=f'Initial deformation = F/k = {F_m/k_m:.2f}')
    ax_m1.set_xlabel('Time (s)')
    ax_m1.set_ylabel('Deformation x(t) (m)')
    ax_m1.set_title(f'Maxwell — Constant Load (F = {F_m} N)\nτ = b/k = {tau_m:.2f} s  [Euler, dt = {dt_m:.4f} s]')
    ax_m1.legend(fontsize=10)
    ax_m1.set_ylim(bottom=0)

    # Plot Case II
    ax_m2.plot(t_maxwell, F_maxwell_const_x, color='#F44336', linewidth=2.5)
    ax_m2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_m2.axvline(x=tau_m, color='orange', linestyle=':', alpha=0.7, label=f'τ = {tau_m:.2f} s')
    ax_m2.set_xlabel('Time (s)')
    ax_m2.set_ylabel('Force F(t) (N)')
    ax_m2.set_title(f'Maxwell — Constant Deformation (x₀ = {x0_m} m)\nτ = b/k = {tau_m:.2f} s  [Euler, dt = {dt_m:.4f} s]')
    ax_m2.legend(fontsize=10)
    ax_m2.set_ylim(bottom=-0.1)

    fig_maxwell.tight_layout()
    fig_maxwell
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercises — Maxwell Model**

    1. For a Maxwell model with $k = 4$ N/m and $b = 12$ N·s/m, calculate the time constant $\tau$.

    2. Under constant load $F = 8$ N with $k = 2$ N/m and $b = 10$ N·s/m:
        - What is the instantaneous deformation at $t = 0$?
        - What is the deformation velocity of the dashpot?
        - What is the total deformation at $t = 5$ s?

    3. Under constant deformation $x_0 = 2$ m with $k = 3$ N/m and $b = 6$ N·s/m:
        - What is the initial force $F(0)$?
        - What is the force at $t = \tau$? (Recall: $e^{-1} \approx 0.368$)
        - After how long does the force drop to half of the initial value?

    4. Why is the Maxwell model inadequate for describing the long-term behavior of biological tissues under constant load?

    5. Modify the parameters using the sliders above. What happens to the relaxation curve when $b$ increases (keeping $k$ fixed)? Why?

    6. In the constant load plot, identify which part of the total deformation comes from the **spring** and which comes from the **dashpot**. Double $b$ and then double $k$ — which change alters the slope of the curve?

    7. **Parameter audit:** In the constant deformation response, doubling $b$ or doubling $k$ — which of the two changes *increases* the relaxation time constant $\tau$?
    """), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Voigt Model

    The Voigt model consists of a **spring and a dashpot connected in parallel**.

    ### Schematic:
    ```
        ┌──/\/\/──┐
    ────┤  spring(k) ├────
        └──[████]──┘
        dashpot(b)
    ```

    ### Fundamental relations:

    In a parallel connection:
    - The **deformations** in the spring and dashpot are **equal**: $x = x_k = x_b$
    - The **forces** are **additive**: $F = F_k + F_b$

    ### Governing differential equation:

    $$\boxed{F = kx + b\dot{x}}$$

    ### Case I: Constant Load Response ($F$ = constant, $\dot{F} = 0$)

    The solution, with initial condition $x(0) = 0$ (the dashpot prevents instantaneous deformation):

    $$\boxed{x(t) = \frac{F}{k}\left(1 - e^{-t/\tau}\right)}$$

    where $\tau = b/k$. The deformation **grows exponentially toward the asymptotic value** $F/k$.
    The long-term behavior is good, but the **instantaneous deformation** observed
    in real tissues is missing.

    ### Case II: Constant Deformation Response ($\dot{x} = 0$)

    The equation trivially reduces to:

    $$\boxed{F = kx}$$

    The force response is **constant and trivial**. However, to impose an instantaneous
    deformation on the Voigt model, an **infinite force** would be required (since the
    dashpot prevents instantaneous deformation), which is not physically achievable.
    """)
    return


@app.cell
def _(mo):
    slider_k_voigt = mo.ui.slider(0.5, 10.0, step=0.5, value=2.0, label="k (N/m)")
    slider_b_voigt = mo.ui.slider(0.5, 20.0, step=0.5, value=5.0, label="b (N·s/m)")
    slider_F_voigt = mo.ui.slider(1.0, 20.0, step=1.0, value=10.0, label="F (N)")

    mo.md(f"""
    ### Voigt Model Parameters

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

    # Case I: Constant load — ODE: dx/dt = (F - k*x) / b
    # Initial condition: x(0) = 0 (dashpot prevents instantaneous deformation)
    t_voigt = np.zeros(N_v + 1)
    x_voigt_const_F = np.zeros(N_v + 1)
    x_voigt_const_F[0] = 0.0
    for _i in range(N_v):
        t_voigt[_i + 1] = t_voigt[_i] + dt_v
        _dxdt = (F_v - k_v * x_voigt_const_F[_i]) / b_v
        x_voigt_const_F[_i + 1] = x_voigt_const_F[_i] + _dxdt * dt_v

    fig_voigt, (ax_v1, ax_v2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Case I
    ax_v1.plot(t_voigt, x_voigt_const_F, color='#4CAF50', linewidth=2.5)
    ax_v1.axhline(y=F_v / k_v, color='gray', linestyle='--', alpha=0.5, label=f'Asymptote = F/k = {F_v/k_v:.2f}')
    ax_v1.axvline(x=tau_v, color='orange', linestyle=':', alpha=0.7, label=f'τ = {tau_v:.2f} s')
    ax_v1.set_xlabel('Time (s)')
    ax_v1.set_ylabel('Deformation x(t) (m)')
    ax_v1.set_title(f'Voigt — Constant Load (F = {F_v} N)\nτ = b/k = {tau_v:.2f} s  [Euler, dt = {dt_v:.4f} s]')
    ax_v1.legend(fontsize=10)
    ax_v1.set_ylim(bottom=0)

    # Plot Case II (trivial — no numerical integration needed)
    x_val_v = 1.0
    t_voigt2 = np.linspace(0, 5, 500)
    F_voigt_const_x = np.ones_like(t_voigt2) * k_v * x_val_v
    ax_v2.plot(t_voigt2, F_voigt_const_x, color='#F44336', linewidth=2.5, label=f'F = kx = {k_v*x_val_v:.1f} N')
    ax_v2.set_xlabel('Time (s)')
    ax_v2.set_ylabel('Force F(t) (N)')
    ax_v2.set_title(f'Voigt — Constant Deformation (x = {x_val_v} m)\nF = kx (constant, trivial)')
    ax_v2.legend(fontsize=10)
    ax_v2.set_ylim(bottom=0, top=k_v * x_val_v * 3.5)

    # Dirac impulse-style arrow at t = 0 representing F → ∞
    _F_base = k_v * x_val_v
    _F_top = _F_base * 3.0
    ax_v2.annotate('', xy=(0, _F_top), xytext=(0, _F_base),
                   arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2.5,
                                   mutation_scale=20))
    # Filled triangle at the base of the arrow (as in impulse notation)
    ax_v2.plot(0, _F_base, marker='^', markersize=0, color='#D32F2F')
    # Vertical line of the impulse
    ax_v2.plot([0, 0], [0, _F_base], color='#D32F2F', linewidth=2.5)

    ax_v2.annotate(r'$F \to \infty$ (impulse)' + '\n' + r'$b\dot{x}\,\delta(t)$',
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
    **📝 Exercises — Voigt Model**

    1. For a Voigt model with $k = 3$ N/m, $b = 9$ N·s/m under constant load $F = 15$ N:
        - What is the time constant $\tau$?
        - What is the asymptotic value of the deformation $x(\infty)$?
        - What is the deformation at $t = \tau$? (Use $1 - e^{-1} \approx 0.632$)

    2. In the Voigt model, why is the initial deformation zero ($x(0) = 0$)? Which element prevents instantaneous deformation?

    3. Explain physically why an **infinite force** would be needed to impose an instantaneous deformation on the Voigt model.

    4. Compare the Voigt ODE ($F = kx + b\dot{x}$) with the Maxwell ODE ($\dot{F} + \frac{k}{b}F = k\dot{x}$). Which is first-order in $x$? Which is first-order in $F$?

    5. Using the sliders, observe what happens when $b \to 0$. The Voigt response approaches which simple element?

    6. **Design problem:** Choose values of $k$ and $b$ so that the Voigt model reaches **95%** of the final deformation at $t = 3$ s under constant load. *Hint: $1 - e^{-3} \approx 0.95$, so $\tau$ should be...*
    """), kind="warn")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Comparison: Maxwell vs Voigt vs Real Tissue

    The table below summarizes the limitations of each model compared with the real
    behavior of biological tissues (Figure 2.16):

    | Behavior | Real Tissue | Maxwell | Voigt |
    |:---|:---:|:---:|:---:|
    | **Constant load — instantaneous deformation** | ✅ Yes | ✅ Yes | ❌ No |
    | **Constant load — finite asymptote** | ✅ Yes | ❌ No (grows linearly) | ✅ Yes |
    | **Constant deformation — initial force** | ✅ Finite | ✅ Finite | ❌ Infinite |
    | **Constant deformation — relaxation to value >0** | ✅ Yes | ❌ No (decays to zero) | ✅ Yes (trivial) |

    **Conclusion:** Neither model alone can adequately reproduce
    the behavior of biological tissues under **all** conditions. We need a
    model that combines the qualities of both — this is the **Kelvin Model** (Section 5).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Real behavior of biological tissues (Figure 2.16 — Yamaguchi)

    The figure below, extracted from the book, shows the **experimentally observed** behavior
    in real biological tissues:

    - **A — Constant load (*creep*):** instantaneous deformation + slow deformation toward a finite asymptote
    - **B — Constant deformation (*relaxation*):** high initial force that decays to a **non-zero** asymptotic value
    """)
    return


@app.cell
def _(mo):
    _img = mo.image(src="fig2_16.png", width=600)
    mo.vstack([
        _img,
        mo.md("*Figure 2.16 — Physiological tissue responses to constant load and constant deformation (Yamaguchi)*"),
        mo.md(r"""
        Notice how **none** of the models below (Maxwell or Voigt) can reproduce
        **simultaneously** both behaviors shown in the figure.
        """)
    ])
    return


@app.cell
def _(np, plt):
    # Visual comparison: Maxwell vs Voigt (without Kelvin — not yet presented)
    # All solved by the Euler method
    k_comp = 2.0
    b_comp = 5.0
    F_comp = 10.0
    x0_comp = 1.0
    tau_comp = b_comp / k_comp

    T_final_comp = 10 * tau_comp
    N_comp = 3000
    dt_comp = T_final_comp / N_comp
    t_comp = np.zeros(N_comp + 1)

    # --- Constant Load (Euler) ---
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

    # --- Constant Deformation (Euler) ---
    # Maxwell: dF/dt = -(k/b)*F, F(0) = k*x0
    F_maxwell_comp = np.zeros(N_comp + 1)
    F_maxwell_comp[0] = k_comp * x0_comp
    # Voigt: F = k*x (trivial, constant)
    F_voigt_comp = np.ones(N_comp + 1) * k_comp * x0_comp

    for _i in range(N_comp):
        # Maxwell
        F_maxwell_comp[_i + 1] = F_maxwell_comp[_i] + (-(k_comp / b_comp) * F_maxwell_comp[_i]) * dt_comp

    fig_comp, axes_comp = plt.subplots(2, 2, figsize=(14, 10))

    axes_comp[0, 0].plot(t_comp, x_maxwell_comp, '--', color='#2196F3', linewidth=2, label='Maxwell')
    axes_comp[0, 0].plot(t_comp, x_voigt_comp, '--', color='#4CAF50', linewidth=2, label='Voigt')
    axes_comp[0, 0].set_xlabel('Time (s)')
    axes_comp[0, 0].set_ylabel('Deformation x(t)')
    axes_comp[0, 0].set_title('Constant Load — Deformation x(t)  [Euler]')
    axes_comp[0, 0].legend()
    axes_comp[0, 0].annotate('Maxwell: grows\nindefinitely ❌',
                             xy=(t_comp[-1], x_maxwell_comp[-1]),
                             xytext=(t_comp[-1]*0.5, x_maxwell_comp[-1]*0.8),
                             fontsize=9, color='#1565C0',
                             arrowprops=dict(arrowstyle='->', color='#1565C0'))
    axes_comp[0, 0].annotate('Voigt: no instantaneous\ndeformation ❌',
                             xy=(0, 0), xytext=(t_comp[-1]*0.3, x_voigt_comp[-1]*0.4),
                             fontsize=9, color='#2E7D32',
                             arrowprops=dict(arrowstyle='->', color='#2E7D32'))

    axes_comp[0, 1].axhline(y=F_comp, color='black', linewidth=2)
    axes_comp[0, 1].set_xlabel('Time (s)')
    axes_comp[0, 1].set_ylabel('Force F (N)')
    axes_comp[0, 1].set_title(f'Applied Load (F = {F_comp} N)')
    axes_comp[0, 1].set_ylim(0, F_comp * 1.5)

    axes_comp[1, 0].plot(t_comp, F_maxwell_comp, '--', color='#2196F3', linewidth=2, label='Maxwell')
    axes_comp[1, 0].plot(t_comp, F_voigt_comp, '--', color='#4CAF50', linewidth=2, label='Voigt')
    axes_comp[1, 0].set_xlabel('Time (s)')
    axes_comp[1, 0].set_ylabel('Force F(t) (N)')
    axes_comp[1, 0].set_title('Constant Deformation — Force F(t)  [Euler]')
    axes_comp[1, 0].legend()
    axes_comp[1, 0].annotate('Maxwell: decays to zero ❌',
                             xy=(t_comp[-1], 0),
                             xytext=(t_comp[-1]*0.5, F_maxwell_comp[0]*0.3),
                             fontsize=9, color='#1565C0',
                             arrowprops=dict(arrowstyle='->', color='#1565C0'))

    axes_comp[1, 1].axhline(y=x0_comp, color='black', linewidth=2)
    axes_comp[1, 1].set_xlabel('Time (s)')
    axes_comp[1, 1].set_ylabel('Deformation x (m)')
    axes_comp[1, 1].set_title(f'Applied Deformation (x₀ = {x0_comp} m)')
    axes_comp[1, 1].set_ylim(0, x0_comp * 1.5)

    fig_comp.suptitle('Comparison: Maxwell vs Voigt  [Euler Method]', fontsize=14, fontweight='bold', y=1.01)
    fig_comp.tight_layout()
    fig_comp
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Kelvin Model (Standard Linear Solid)

    The Kelvin model consists of a **spring ($k_p$) in parallel** with a **series branch
    of spring ($k_s$) and dashpot ($b$)**, as shown in Figure 2.17C of the book.

    ### Diagram (Fig. 2.17C — Yamaguchi):
    ```
         ┌──/\/\/──[████]──┐
    o────┤   ks       b    ├────o
         └────/\/\/────────┘
               kp
    ```

    - Upper branch: spring $k_s$ in **series** with dashpot $b$ (Maxwell model)
    - Lower branch: spring $k_p$ in **parallel**

    ### Fundamental relationships:

    - $F = F_{k_p} + F_{series}$ (total force = parallel spring + series branch)
    - $F_{series} = F_{k_s} = F_b$ (in the series branch, forces are equal)
    - $x = x_{k_p} = x_{k_s} + x_b$ (same total deformation; in the series branch, deformations add up)

    ### Governing differential equation:

    $$\boxed{F + \tau_\varepsilon \dot{F} = k_s\left(x + \tau_\sigma \dot{x}\right)}$$

    where the time constants are:

    $$\tau_\varepsilon = \frac{b}{k_p} \quad \text{(time constant for constant deformation — relaxation)}$$

    $$\tau_\sigma = \frac{b(k_p + k_s)}{k_p \cdot k_s} \quad \text{(time constant for constant load — creep)}$$

    Note that $\tau_\sigma > \tau_\varepsilon$ always (since $k_p + k_s > k_p$).

    ### Case I: Constant Load ($F$ = constant, $\dot{F} = 0$)

    $$\boxed{x(t) = \frac{F}{k_s} + \frac{F}{k_p}\left(1 - e^{-t/\tau_\sigma}\right)}$$

    - At $t = 0$: instantaneous deformation $x(0) = F/k_s$ (series spring)
    - At $t \to \infty$: $x(\infty) = F/k_s + F/k_p$ (finite asymptote) ✅

    ### Case II: Constant Deformation ($\dot{x} = 0$)

    $$\boxed{F(t) = \frac{k_p k_s}{k_p + k_s}x + \left(k_s - \frac{k_p k_s}{k_p + k_s}\right)x \cdot e^{-t/\tau_\varepsilon}}$$

    - At $t = 0$: $F(0) = k_s \cdot x$ (series spring responds instantaneously) ✅
    - At $t \to \infty$: $F(\infty) = \frac{k_p k_s}{k_p + k_s} x > 0$ (relaxation to non-zero value) ✅
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
    ### Kelvin Model Parameters

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

    # Case I: Constant load — Euler
    # State: x_b (deformation of the Voigt part: dashpot + parallel spring)
    # ODE: dx_b/dt = (F - kp*x_b) / b
    # x_total = F/ks + x_b (series spring deforms instantaneously)
    x_b_kel = np.zeros(N_kel + 1)
    x_kelvin_const_F = np.zeros(N_kel + 1)
    x_kelvin_const_F[0] = F_kel / ks_kel  # instantaneous deformation of the series spring
    for _i in range(N_kel):
        t_kelvin[_i + 1] = t_kelvin[_i] + dt_kel
        _dx_b_dt = (F_kel - kp_kel * x_b_kel[_i]) / b_kel
        x_b_kel[_i + 1] = x_b_kel[_i] + _dx_b_dt * dt_kel
        x_kelvin_const_F[_i + 1] = F_kel / ks_kel + x_b_kel[_i + 1]

    # Case II: Constant deformation — Euler
    # ODE: dF/dt = -(F - k_eq*x0) / tau_eps
    # Initial condition: F(0) = ks*x0
    F_kelvin_const_x = np.zeros(N_kel + 1)
    F_kelvin_const_x[0] = ks_kel * x0_kel
    for _i in range(N_kel):
        _dFdt = -(F_kelvin_const_x[_i] - k_eq_kel * x0_kel) / tau_eps_kel
        F_kelvin_const_x[_i + 1] = F_kelvin_const_x[_i] + _dFdt * dt_kel

    fig_kelvin, (ax_k1, ax_k2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Case I
    ax_k1.plot(t_kelvin, x_kelvin_const_F, color='#9C27B0', linewidth=2.5)
    ax_k1.axhline(y=F_kel / ks_kel, color='#E91E63', linestyle=':', alpha=0.7, label=f'Instantaneous deformation = F/ks = {F_kel/ks_kel:.2f}')
    ax_k1.axhline(y=F_kel / ks_kel + F_kel / kp_kel, color='gray', linestyle='--', alpha=0.5,
                  label=f'Asymptote = F/ks + F/kp = {F_kel/ks_kel + F_kel/kp_kel:.2f}')
    ax_k1.axvline(x=tau_sig_kel, color='orange', linestyle=':', alpha=0.7, label=f'τ_σ = {tau_sig_kel:.2f} s')
    ax_k1.set_xlabel('Time (s)')
    ax_k1.set_ylabel('Deformation x(t) (m)')
    ax_k1.set_title(f'Kelvin — Constant Load (F = {F_kel} N)  [Euler]')
    ax_k1.legend(fontsize=9)
    ax_k1.set_ylim(bottom=0)

    # Plot Case II
    ax_k2.plot(t_kelvin, F_kelvin_const_x, color='#FF5722', linewidth=2.5)
    ax_k2.axhline(y=ks_kel * x0_kel, color='#E91E63', linestyle=':', alpha=0.7, label=f'F(0) = ks·x₀ = {ks_kel*x0_kel:.2f}')
    ax_k2.axhline(y=k_eq_kel * x0_kel, color='gray', linestyle='--', alpha=0.5,
                  label=f'F(∞) = kp·ks/(kp+ks)·x₀ = {k_eq_kel*x0_kel:.2f}')
    ax_k2.axvline(x=tau_eps_kel, color='orange', linestyle=':', alpha=0.7, label=f'τ_ε = {tau_eps_kel:.2f} s')
    ax_k2.set_xlabel('Time (s)')
    ax_k2.set_ylabel('Force F(t) (N)')
    ax_k2.set_title(f'Kelvin — Constant Deformation (x₀ = {x0_kel} m)  [Euler]')
    ax_k2.legend(fontsize=9)
    ax_k2.set_ylim(bottom=0)

    fig_kelvin.tight_layout()
    fig_kelvin
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ✅ Kelvin reproduces the behavior of real tissue

    Compare the plots above with **Figure 2.16** (Section 4):

    | Behavior | Real Tissue (Fig. 2.16) | Kelvin |
    |:---|:---:|:---:|
    | Instantaneous deformation under constant load | ✅ | ✅ ($x(0) = F/k_s$) |
    | Finite asymptote under constant load | ✅ | ✅ ($x(\infty) = F/k_s + F/k_p$) |
    | Finite initial force under constant deformation | ✅ | ✅ ($F(0) = k_s x_0$) |
    | Relaxation to **non-zero** value | ✅ | ✅ ($F(\infty) = k_{eq} x_0 > 0$) |

    The Kelvin model is the **simplest** model that **qualitatively** reproduces all
    the behaviors observed in real biological tissues.
    """)
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercises — Kelvin Model**

    1. For $k_s = 4$ N/m, $k_p = 2$ N/m, $b = 6$ N·s/m:
        - Calculate $\tau_\varepsilon$ and $\tau_\sigma$. Which is larger? Why?
        - Under load $F = 10$ N, what is the instantaneous deformation? And the final deformation?

    2. Under constant deformation $x_0 = 1$ m with the parameters above:
        - What is the initial force $F(0)$?
        - What is the equilibrium force $F(\infty)$? (Calculate $k_{eq} = \frac{k_p k_s}{k_p + k_s}$)

    3. Show that $\tau_\sigma > \tau_\varepsilon$ always. *Hint: compare the expressions and use the fact that $k_s > 0$.*

    4. What happens to the Kelvin model when $k_p \to \infty$? Which simpler model does it reduce to?

    5. And when $k_s \to \infty$? Which model does it reduce to?

    6. **Design problem:** Choose values of $k_s$, $k_p$, and $b$ so that the Kelvin model has a **large** instantaneous deformation but only a small additional creep. *Hint: think about the relationship between $F/k_s$ and $F/k_p$.*

    7. What happens to the size of the instantaneous deformation when $k_s$ **increases**? Use the sliders to verify.
    """), kind="warn")
    return


@app.cell
def _(mo):
    mo.callout(mo.md(r"""
    **📝 Exercises — Model Comparison**

    1. Fill in the table below with ✅ or ❌ for each model:

    | Behavior | Maxwell | Voigt | Kelvin |
    |:---|:---:|:---:|:---:|
    | Instantaneous deformation under load | ? | ? | ? |
    | Finite asymptote under constant load | ? | ? | ? |
    | Relaxation to value $> 0$ | ? | ? | ? |

    2. A biological tissue is subjected to a constant load. An instantaneous deformation is observed followed by an increasing deformation that tends to a finite value. Which of the three models best describes this behavior? Justify.

    3. Why do we need **at least 3 parameters** (as in the Kelvin model) to adequately reproduce the viscoelastic behavior of biological tissues?

    4. Choose **one** curve from the comparison plots (Section 4) and explain which **physical element** is responsible for: (a) the initial value, (b) the slope, and (c) the final value of the curve.
    """), kind="warn")
    return


if __name__ == "__main__":
    app.run()
