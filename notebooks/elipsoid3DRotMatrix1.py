import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import sympy as sym
    import numpy as np
    return np, sym


@app.cell
def _(np):
    def rotationGlobalX(alpha):
        return np.array([[1, 0, 0], [0, np.cos(_alpha), -np.sin(_alpha)], [0, np.sin(_alpha), np.cos(_alpha)]])

    def rotationGlobalY(beta):
        return np.array([[np.cos(_beta), 0, np.sin(_beta)], [0, 1, 0], [-np.sin(_beta), 0, np.cos(_beta)]])

    def rotationGlobalZ(gamma):
        return np.array([[np.cos(_gamma), -np.sin(_gamma), 0], [np.sin(_gamma), np.cos(_gamma), 0], [0, 0, 1]])

    def rotationLocalX(alpha):
        return np.array([[1, 0, 0], [0, np.cos(_alpha), np.sin(_alpha)], [0, -np.sin(_alpha), np.cos(_alpha)]])

    def rotationLocalY(beta):
        return np.array([[np.cos(_beta), 0, -np.sin(_beta)], [0, 1, 0], [np.sin(_beta), 0, np.cos(_beta)]])

    def rotationLocalZ(gamma):
        return np.array([[np.cos(_gamma), np.sin(_gamma), 0], [-np.sin(_gamma), np.cos(_gamma), 0], [0, 0, 1]])
    return (
        rotationGlobalX,
        rotationGlobalY,
        rotationGlobalZ,
        rotationLocalX,
        rotationLocalY,
        rotationLocalZ,
    )


@app.cell
def _(
    np,
    rotationGlobalX,
    rotationGlobalY,
    rotationGlobalZ,
    rotationLocalX,
    rotationLocalY,
    rotationLocalZ,
):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (10, 10)
    _coefs = (1, 3, 15)
    (_rx, _ry, _rz) = 1 / np.sqrt(_coefs)
    _u = np.linspace(0, 2 * np.pi, 30)
    _v = np.linspace(0, np.pi, 30)
    _x = _rx * np.outer(np.cos(_u), np.sin(_v))
    _y = _ry * np.outer(np.sin(_u), np.sin(_v))
    _z = _rz * np.outer(np.ones_like(_u), np.cos(_v))
    _fig = plt.figure(figsize=plt.figaspect(1))
    _ax = _fig.add_subplot(111, projection='3d')
    _xr = np.reshape(_x, (1, -1))
    _yr = np.reshape(_y, (1, -1))
    _zr = np.reshape(_z, (1, -1))
    RX = rotationGlobalX(np.pi / 3)
    RY = rotationGlobalY(np.pi / 3)
    RZ = rotationGlobalZ(np.pi / 3)
    _Rx = rotationLocalX(np.pi / 3)
    _Ry = rotationLocalY(np.pi / 3)
    _Rz = rotationLocalZ(np.pi / 3)
    _rRotx = RZ @ RY @ RX @ np.vstack((_xr, _yr, _zr))
    print(np.shape(_rRotx))
    _ax.plot_surface(np.reshape(_rRotx[0, :], (30, 30)), np.reshape(_rRotx[1, :], (30, 30)), np.reshape(_rRotx[2, :], (30, 30)), rstride=4, cstride=4, color='b')
    _max_radius = max(_rx, _ry, _rz)
    for _axis in 'xyz':
        getattr(_ax, 'set_{}lim'.format(_axis))((-_max_radius, _max_radius))
    plt.show()
    return (plt,)


@app.cell
def _(
    np,
    plt,
    rotationGlobalX,
    rotationGlobalY,
    rotationGlobalZ,
    rotationLocalX,
    rotationLocalY,
    rotationLocalZ,
):
    _coefs = (1, 3, 15)
    (_rx, _ry, _rz) = 1 / np.sqrt(_coefs)
    _u = np.linspace(0, 2 * np.pi, 30)
    _v = np.linspace(0, np.pi, 30)
    _x = _rx * np.outer(np.cos(_u), np.sin(_v))
    _y = _ry * np.outer(np.sin(_u), np.sin(_v))
    _z = _rz * np.outer(np.ones_like(_u), np.cos(_v))
    _fig = plt.figure(figsize=plt.figaspect(1))
    _ax = _fig.add_subplot(111, projection='3d')
    _xr = np.reshape(_x, (1, -1))
    _yr = np.reshape(_y, (1, -1))
    _zr = np.reshape(_z, (1, -1))
    RX_1 = rotationGlobalX(np.pi / 3)
    RY_1 = rotationGlobalY(np.pi / 3)
    RZ_1 = rotationGlobalZ(np.pi / 3)
    _Rx = rotationLocalX(np.pi / 3)
    _Ry = rotationLocalY(np.pi / 3)
    _Rz = rotationLocalZ(np.pi / 3)
    _rRotx = RY_1 @ RX_1 @ np.vstack((_xr, _yr, _zr))
    print(np.shape(_rRotx))
    _ax.plot_surface(np.reshape(_rRotx[0, :], (30, 30)), np.reshape(_rRotx[1, :], (30, 30)), np.reshape(_rRotx[2, :], (30, 30)), rstride=4, cstride=4, color='b')
    _max_radius = max(_rx, _ry, _rz)
    for _axis in 'xyz':
        getattr(_ax, 'set_{}lim'.format(_axis))((-_max_radius, _max_radius))
    plt.show()
    return RX_1, RY_1, RZ_1


@app.cell
def _(np):
    np.sin(np.arccos(0.7))
    return


@app.cell
def _(RX_1, RY_1, RZ_1):
    print(RZ_1 @ RY_1 @ RX_1)
    return


@app.cell
def _(sym):
    sym.init_printing()
    return


@app.cell
def _(sym):
    a,b,g = sym.symbols('alpha, beta, gamma')
    return a, b, g


@app.cell
def _(a, b, g, sym):
    RX_2 = sym.Matrix([[1, 0, 0], [0, sym.cos(a), -sym.sin(a)], [0, sym.sin(a), sym.cos(a)]])
    RY_2 = sym.Matrix([[sym.cos(b), 0, sym.sin(b)], [0, 1, 0], [-sym.sin(b), 0, sym.cos(b)]])
    RZ_2 = sym.Matrix([[sym.cos(g), -sym.sin(g), 0], [sym.sin(g), sym.cos(g), 0], [0, 0, 1]])
    (RX_2, RY_2, RZ_2)
    return RX_2, RY_2, RZ_2


@app.cell
def _(RX_2, RY_2, RZ_2):
    R = RZ_2 @ RY_2 @ RX_2
    R
    return


@app.cell
def _(np):
    mm = np.array([2.71, 10.22, 26.52])
    lm = np.array([2.92, 10.10, 18.85])
    fh = np.array([5.05, 41.90, 15.41])
    mc = np.array([8.29, 41.88, 26.52])
    ajc = (mm + lm)/2
    kjc = (fh + mc)/2
    return ajc, kjc, lm, mm


@app.cell
def _(ajc, kjc, lm, mm, np):
    i = np.array([1,0,0])
    j = np.array([0,1,0])
    k = np.array([0,0,1])
    v1 = kjc - ajc
    v1 = v1 / np.sqrt(v1[0]**2+v1[1]**2+v1[2]**2)
    v2 = (mm-lm) - ((mm-lm)@v1)*v1
    v2 = v2/ np.sqrt(v2[0]**2+v2[1]**2+v2[2]**2)
    v3 = k - (k@v1)*v1 - (k@v2)*v2
    v3 = v3/ np.sqrt(v3[0]**2+v3[1]**2+v3[2]**2)
    return v1, v2, v3


@app.cell
def _(v1):
    v1
    return


@app.cell
def _(np, v1, v2, v3):
    R_1 = np.array([v1, v2, v3])
    RGlobal = R_1.T
    RGlobal
    return RGlobal, R_1


@app.cell
def _(RGlobal, np):
    _alpha = np.arctan2(RGlobal[2, 1], RGlobal[2, 2]) * 180 / np.pi
    _alpha
    return


@app.cell
def _(RGlobal, np):
    _beta = np.arctan2(-RGlobal[2, 0], np.sqrt(RGlobal[2, 1] ** 2 + RGlobal[2, 2] ** 2)) * 180 / np.pi
    _beta
    return


@app.cell
def _(RGlobal, np):
    _gamma = np.arctan2(RGlobal[1, 0], RGlobal[0, 0]) * 180 / np.pi
    _gamma
    return


@app.cell
def _(np):
    R2 = np.array([[0, 0.71, 0.7],[0,0.7,-0.71],[-1,0,0]])
    R2
    return (R2,)


@app.cell
def _(R_1, np):
    _alpha = np.arctan2(R_1[2, 1], R_1[2, 2]) * 180 / np.pi
    _alpha
    return


@app.cell
def _(R_1, np):
    _gamma = np.arctan2(R_1[1, 0], R_1[0, 0]) * 180 / np.pi
    _gamma
    return


@app.cell
def _(R_1, np):
    _beta = np.arctan2(-R_1[2, 0], np.sqrt(R_1[2, 1] ** 2 + R_1[2, 2] ** 2)) * 180 / np.pi
    _beta
    return


@app.cell
def _(RX_2, RY_2, RZ_2):
    R_2 = RY_2 @ RZ_2 @ RX_2
    R_2
    return


@app.cell
def _(R2, np):
    _alpha = np.arctan2(-R2[1, 2], R2[1, 1]) * 180 / np.pi
    _alpha
    return


@app.cell
def _():
    _gamma = 0
    return


@app.cell
def _():
    _beta = 90
    return


@app.cell
def _(sym):
    sym.init_printing()
    return


@app.cell
def _(sym):
    (a_1, b_1, g_1) = sym.symbols('alpha, beta, gamma')
    return a_1, b_1, g_1


@app.cell
def _(a_1, b_1, g_1, sym):
    RX_3 = sym.Matrix([[1, 0, 0], [0, sym.cos(a_1), -sym.sin(a_1)], [0, sym.sin(a_1), sym.cos(a_1)]])
    RY_3 = sym.Matrix([[sym.cos(b_1), 0, sym.sin(b_1)], [0, 1, 0], [-sym.sin(b_1), 0, sym.cos(b_1)]])
    RZ_3 = sym.Matrix([[sym.cos(g_1), -sym.sin(g_1), 0], [sym.sin(g_1), sym.cos(g_1), 0], [0, 0, 1]])
    (RX_3, RY_3, RZ_3)
    return RX_3, RY_3, RZ_3


@app.cell
def _(RX_3, RY_3, RZ_3):
    RXYZ = RZ_3 * RY_3 * RX_3
    RXYZ
    return


@app.cell
def _(RX_3, RY_3, RZ_3):
    RZXY = RZ_3 * RX_3 * RY_3
    RZXY
    return


if __name__ == "__main__":
    app.run()
