# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "sympy>=1.13"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ellipsoids, rotation sequences and a marker-derived basis

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil

    ## How to use this guide

    An ellipsoid makes orientation visible: unlike a sphere, its unequal axes let us see a rotation. Predict the long-axis direction before running each plot, then compare a rotated point with the rotated surface. The Exercises have collapsed worked solutions; the independent Problems provide final answers only.

    **Opening challenge.** The surface is $x^2+3y^2+15z^2=1$. Which axis is longest? If it is rotated about $x$, does that longest axis move?

    ## Python setup

    NumPy builds the surface mesh and Matplotlib renders it. SymPy also shows rotation products algebraically. No external files are needed. You need matrix multiplication, sines and cosines.

    ## From the equation to a surface

    Write the ellipsoid as $(x/a)^2+(y/b)^2+(z/c)^2=1$. Its semiaxes are $a=1$, $b=1/\sqrt3$ and $c=1/\sqrt{15}$. The sampled angles describe surface points, not the orientation angles of the whole body. A rotation acts on the stacked Cartesian coordinates.

    **Exercise 1.** Find the three positive axis intercepts and check one in the original equation.

    /// details | Worked solution — Exercise 1

    The intercepts are $(1,0,0)$, $(0,1/\sqrt3,0)$ and $(0,0,1/\sqrt{15})$. For the second, $3(1/\sqrt3)^2=1$. The $x$ semiaxis is longest, and rotation about $x$ leaves it fixed.

    ///
    """)
    return

@app.cell
def _():
    import sympy as sym
    import numpy as np

    return np, sym


@app.cell
def _(np):
    def rotationGlobalX(alpha):
        return np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])

    def rotationGlobalY(beta):
        return np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])

    def rotationGlobalZ(gamma):
        return np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    def rotationLocalX(alpha):
        return np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])

    def rotationLocalY(beta):
        return np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])

    def rotationLocalZ(gamma):
        return np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

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
    _ax.set_box_aspect((1, 1, 1))
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
    _ax.set_box_aspect((1, 1, 1))
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
    v3 = np.cross(v1, v2)  # preserve right-handedness for any nondegenerate marker set
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
    _q = np.sqrt(0.5)
    R2 = np.array([[0, _q, _q], [0, _q, -_q], [-1, 0, 0]])
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


@app.cell
def _():
    import marimo as mo
    return (mo,)






@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the sequence and the basis

    With column vectors, the rightmost matrix acts first: `RZ @ RY @ RX` applies fixed-axis $x$, then $y$, then $z$ rotations. The functions named `rotationLocalX/Y/Z` in the preserved example return the transposes of the corresponding active matrices; their names alone do not specify a moving-axis sequence.

    The marker example builds two orthogonal directions first and closes the triad with their cross product. This makes handedness explicit. The final gimbal-lock example now uses exact $1/\sqrt2$ entries: the old rounded 0.70 and 0.71 values were not exactly orthogonal.

    **Exercise 2.** Why can an inverse rotation be obtained with a transpose, while reversing a product also requires reversing the factor order?

    /// details | Worked solution — Exercise 2

    Orthogonality gives $R^{-1}=R^T$. For $R=R_zR_yR_x$, the inverse is $R_x^TR_y^TR_z^T$: the last operation must be undone first. Keeping the original factor order would generally give another orientation.

    ///

    **Exercise 3.** A matrix passes $R^TR=I$ but has determinant −1. Can its entries be treated as an anatomical rotation? What if the middle Cardan angle is 90°?

    /// details | Worked solution — Exercise 3

    Determinant −1 means a reflection, so one axis convention must be corrected before calling it a rotation. At a 90° middle angle in this Cardan convention, the orientation matrix remains valid, but two angle coordinates become coupled: separate first and third angles are not unique. This is a coordinate singularity, not a loss of physical orientation.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. What properties distinguish a rotation from scaling?
    2. Does transposing a rotation undo it?
    3. Are three orthogonal unit vectors automatically right-handed?

    /// details | Checkpoint answers

    1. A rotation preserves lengths and angles and has determinant +1. Scaling generally changes lengths. 2. Yes, for a proper orthogonal rotation matrix. 3. No; the determinant must also be +1.

    ///

    ## Problems

    1. Rotate $(0,1,0)$ by $60^\circ$ about the fixed positive $x$ axis.
    2. Give the ellipsoid’s volume and say whether rotation changes it.
    3. Let $R=R_z(90^\circ)R_x(90^\circ)$. Give $R(0,1,0)$ and compare it with the reversed order.

    /// details | Final answers to the problems

    1. $(0,1/2,\sqrt3/2)$.
    2. $4\pi/(3\sqrt{45})\approx0.6244$ cubic coordinate units; unchanged.
    3. $(0,0,1)$; the reversed order gives $(-1,0,0)$.

    ///

    ## Go deeper

    [Transformation3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py) develops active/passive conventions and sequence-dependent angles. [CoordinateSystem](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py) connects the marker calculations below the surfaces to anatomical frames.

    ### Video lectures on the Internet

    [Lynch and Park: rotation matrices](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-1-of-2/) explains the orthogonality and determinant tests used here.

    ## References

    - Lynch KM, Park FC, *Modern Robotics*, Chapter 3: rotation matrices and rigid configurations.

    - Original BMC ellipsoid and marker examples, preserved from the paired Jupyter notebook.
    """)
    return



if __name__ == "__main__":
    app.run()
