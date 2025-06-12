import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Basic passive models
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Maxwell Model
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""$F = F_k=F_b$$x = x_k+x_b$$\dot x = \dot x_k + \dot x_b$$F_k = kx_k \rightarrow \dot x_k = \frac{\dot F_k}{k} = \frac{\dot F}{k}$$F_b = b\dot x_b \rightarrow \dot x_b = \frac{F_b}{b} = \frac{F}{b}$$\dot x = \frac{\dot F}{k} + \frac{F}{b}$$\dot F =   - \frac{kF}{b} + k\dot x$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Length step
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""$\dot{x_1} = -\frac{kx_1}{b} + x$$F = -\frac{k^2x_1}{b} + kx$"""
    )
    return


@app.cell
def _(np, plt):
    _x = 0.3
    _dt = 0.01
    _k = 0.5
    _b = 0.1
    _t = np.arange(0, 3, _dt)
    _F = np.zeros_like(_t)
    _x1 = 0
    for _i in range(len(_t)):
        if _t[_i] > 1:
            _x = 0.3
        else:
            _x = 0
        _x1dot = -_k * _x1 / _b + _x
        _x1 = _x1 + _dt * _x1dot
        _F[_i] = _k * _x1dot
    plt.figure()
    plt.plot(_t, _F)
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sinusoidal length
        """
    )
    return


@app.cell
def _(np, plt):
    _dt = 0.01
    _k = 0.5
    _b = 0.1
    _t = np.arange(0, 3, _dt)
    _F = np.zeros_like(_t)
    _x1 = 0
    for _i in range(len(_t)):
        if _t[_i] > 1:
            _x = 0.3 + 0.15 * np.cos(2 * np.pi * _t[_i])
        else:
            _x = 0
        _x1dot = -_k * _x1 / _b + _x
        _x1 = _x1 + _dt * _x1dot
        _F[_i] = -_k ** 2 * _x1 / _b + _k * _x
    plt.figure()
    plt.plot(_t, _F)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Voight Model
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""$F = F_k+F_b$$x = x_k=x_b$$F_k = kx_k \rightarrow F_k = kx$$F_b = b\dot x_b \rightarrow F_b = b\dot x$$F = kx+b\dot x$$\dot x =   - \frac{kx}{b} + \frac{F}{b}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""$\dot x_1 = -\frac{kx_1}{b} + \frac{F}{b}$$x = x_1$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Constant force


        """
    )
    return


@app.cell
def _(np, plt):
    _dt = 0.01
    _k = 0.5
    _b = 0.1
    _t = np.arange(0, 10, _dt)
    _x = np.zeros_like(_t)
    _x1 = 0
    for _i in range(len(_t)):
        if 0 < _t[_i] <= 1:
            _F = 0
        elif 1 < _t[_i] <= 4:
            _F = 1
        else:
            _F = 0
        _x1dot = -_k / _b * _x1 + _F / _b
        _x1 = _x1 + _dt * _x1dot
        _x[_i] = _x1
    plt.figure()
    plt.plot(_t, _x)
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sinusoidal force
        """
    )
    return


@app.cell
def _(np, plt):
    _dt = 0.01
    _k = 0.5
    _b = 0.1
    _t = np.arange(0, 10, _dt)
    _F = 1 + 0.2 * np.cos(2 * np.pi * 2 * _t)
    _x = np.zeros_like(_t)
    _x1 = 1
    for _i in range(len(_t)):
        _x1dot = -_k / _b * _x1 + _F[_i] / _b
        _x1 = _x1 + _dt * _x1dot
        _x[_i] = _x1
    plt.figure()
    plt.plot(_t, _x)
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Kelvin Model
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""$F = F_1+F_2$$F_1 = F_{k_s} = F_b$$F_2 = F_{k_p}$$x = x_s + x_b$$\dot x = \dot x_s + \dot x_b$$F_{k_s} = k_sx_s \rightarrow \dot F_{k_s} = k_s\dot x_s \rightarrow \dot x_s = \frac{\dot F_{k_s}}{k_s}$$F_{b} = b\dot x_b \rightarrow \dot x_b = \frac{F_{b}}{b}$$F_{k_p} =$$\dot x = \frac{\dot F_{1}}{k_s} + \frac{F_{1}}{b}$$\dot F_{1} = - \frac{k_sF_{1}}{b} + k_s\dot x$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Length step$\dot x_1 =  - \frac{k_sx_{1}}{b} + x$$F_1 =  - \frac{k_s^2x_{1}}{b} + k_sx$$F = F_1+k_px$"""
    )
    return


@app.cell
def _(np, plt):
    _x = 0.3
    _dt = 0.01
    _ks = 0.5
    _kp = 0.1
    _b = 1
    _t = np.arange(0, 10, _dt)
    _F = np.zeros_like(_t)
    _x1 = 0
    for _i in range(len(_t)):
        if 1 < _t[_i] < 1.3:
            _x = 3.3 * (_t[_i] - 1)
        elif _t[_i] >= 1.3:
            _x = 1
        else:
            _x = 0
        _x1dot = -_ks * _x1 / _b + _x
        _x1 = _x1 + _dt * _x1dot
        _F[_i] = _ks * _x1dot + _kp * _x
    plt.figure()
    plt.plot(_t, _F)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Force step$\dot x = \frac{\dot F - k_p\dot x}{k_s} + \frac{F - k_px}{b}$$\dot x\left(\frac{k_s+k_p}{k_s} \right) = \frac{\dot F}{k_s} + \frac{F - k_px}{b}$$\dot x = \frac{\dot F}{k_s+k_p} + \frac{k_sF}{b(k_s+k_p)} - \frac{k_sk_px}{b(k_s+k_p)}$$\dot x_1 = F - \frac{k_sk_px}{b(k_s+k_p)}$$x = \frac{\dot x_1}{k_s+k_p} + \frac{k_sx_1}{b(k_s+k_p)}$"""
    )
    return


@app.cell
def _(np, plt):
    _dt = 0.01
    _ks = 0.5
    _kp = 0.1
    _b = 0.1
    _t = np.arange(0, 10, _dt)
    _x = np.zeros_like(_t)
    _x1 = 0
    for _i in range(len(_t)):
        if 0 < _t[_i] < 1:
            _F = 0
        elif 1 < _t[_i] < 5:
            _F = 1
        else:
            _F = 0
        _x1dot = -_ks * _kp / (_ks + _kp) / _b * _x1 + _F
        _x1 = _x1 + _dt * _x1dot
        _x[_i] = _ks / (_b * (_ks + _kp)) * _x1 + _x1dot / (_ks + _kp)
    plt.figure()
    plt.plot(_t, _x)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Yamaguchi, GT, DYNAMIC MODELING OF MUSCULOSKELETAL MOTION A Vectorized Approach
        for Biomechanical Analysis in Three Dimensions, 2001
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
