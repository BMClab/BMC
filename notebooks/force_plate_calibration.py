# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14", "seaborn>=0.13", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Algorithm for force plate calibration

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this notebook

    Follow the six measured loads through the matrix model, a simulation with known answers, and the alternating estimates of calibration and load-cell orientation. Keep N, N m and m visible in your calculations. Try each exercise before opening its derivation, then use the final problems as an independent check.

    ## Opening challenge

    A plate reports a force that is 5% too large. A calibration can correct that gain. But what if a sideways force also leaks into a moment channel, and the reference load cell is rotated? Can one experiment identify all of those effects?

    ## Setup

    NumPy, SciPy, Matplotlib and seaborn support the retained simulations and optimization. Seeded synthetic forces make repeated runs comparable. The example has known true calibration and rotation angles; no laboratory acquisition is needed. Agreement with this synthetic case verifies the calculation under its assumptions, not the accuracy of a physical plate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook demonstrates the algorithm for force plate calibration proposed by Cedraro et al. (2008, 2009).

    A force plate (FP) is an electromechanical device that measures the components of the vectors force $(\mathbf{F} = [F_X,\, F_Y,\, F_Z])$ and moment of force or torque $(\mathbf{M} = [M_X,\, M_Y,\, M_Z])$ applied to the it. The FP is composed by a transducer that transforms a mechanical deformation to an electrical signal usually using strain gauges or piezoelectric sensors. The transformation from electrical signals (input) to force and moment of force (output) as a function of time in a six-component FP usually is given by the following linear relationship:

    $$
    \mathbf{L}(t) = \mathbf{C}\mathbf{V}(t)
    $$

    Where $\mathbf{L}(t)$ is the force plate output vector $([\mathbf{F}(t), \mathbf{M}(t)]^T)$, in N and Nm, $\mathbf{V}(t)$ is the vector of electrical signals (six voltage signals, in V) and  $\mathbf{C}$ is known as the six-by-six (constant) calibration matrix (in N/V or Nm/V). Note that we used the term vector here to refer to an uni-dimensional matrix (usual in scientific computing), which is different from vector/scalar concept in Mechanics.<br>
    The expansion of the former equiation at a given instant is:

    $$
    \begin{bmatrix}
    F_x \\ F_y \\ F_z \\ M_x \\ M_y \\ M_z
    \end{bmatrix}\, = \,
    \begin{bmatrix}
    C_{11} && C_{12} && C_{13} && C_{14} && C_{15} && C_{16} \\
    C_{21} && C_{22} && C_{23} && C_{24} && C_{25} && C_{26} \\
    C_{31} && C_{32} && C_{33} && C_{34} && C_{35} && C_{36} \\
    C_{41} && C_{42} && C_{43} && C_{44} && C_{45} && C_{46} \\
    C_{51} && C_{52} && C_{53} && C_{54} && C_{55} && C_{56} \\
    C_{61} && C_{62} && C_{63} && C_{64} && C_{65} && C_{66}
    \end{bmatrix}\,
    \begin{bmatrix}
    V_1 \\ V_2 \\ V_3 \\ V_4 \\ V_5 \\ V_6
    \end{bmatrix}
    $$

    The terms off-diagonal are known as the crosstalk terms and represent the effect of a load applied in one direction on the other direction. For a FP with none or small crosstalk, the off-diagonal terms are zero or very small compared to the main-diagonal terms. Note that the equation above is in fact a system of six linear independent equations with six unknowns each (where $V_1 ... V_6$ are the measured inputs):

    \begin{cases}
        F_x &=& C_{11}V_1 + C_{12}V_2 + C_{13}V_3 + C_{14}V_4 + C_{15}V_5 + C_{16}V_6 \\
        F_y &=& C_{21}V_1 + C_{22}V_2 + C_{23}V_3 + C_{24}V_4 + C_{25}V_5 + C_{26}V_6 \\
        F_z &=& C_{31}V_1 + C_{32}V_2 + C_{33}V_3 + C_{34}V_4 + C_{35}V_5 + C_{36}V_6 \\
        M_x &=& C_{41}V_1 + C_{42}V_2 + C_{43}V_3 + C_{44}V_4 + C_{45}V_5 + C_{46}V_6 \\
        M_y &=& C_{51}V_1 + C_{52}V_2 + C_{53}V_3 + C_{54}V_4 + C_{55}V_5 + C_{56}V_6 \\
        M_z &=& C_{61}V_1 + C_{62}V_2 + C_{63}V_3 + C_{64}V_4 + C_{65}V_5 + C_{66}V_6
    \end{cases}

    Of course, an important aspect of the FP functionning is that it should be calibrated, i.e., the calibration matrix must be known and accurate (it comes with the force plate when you buy one). Cedraro et al. (2008) proposed a method for in situ re-calibration of FP and their algorithm is presented next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Algorithm

    Consider that in a re-calibration procedure we apply on the FP known forces, $\mathbf{F}_I = [F_{X_I},\, F_{Y_I},\, F_{Z_I}]^T$, at known places, $\mathbf{COP} = [X_{COP},\,  Y_{COP},\,  Z_{COP}]$ (the center of pressure coordinates in the FP reference frame).<br>
    The moments of forces, $\mathbf{M}_I = [M_{X_I},\, M_{Y_I},\, M_{Z_I}]^T$, due to these forces can be found using the equation $\mathbf{M}_I = \mathbf{COP} \times \mathbf{F}_I$, which can be expressed in matrix form as:

    $$
    \mathbf{M}_I =
    \begin{bmatrix}
    0 && -Z_{COP} && Y_{COP} \\
    Z_{COP} && 0 && -X_{COP} \\
    -Y_{COP} && X_{COP} && 0
    \end{bmatrix}\, \mathbf{F}_I \, = \, \mathbf{A}_{COP}\mathbf{F}_I
    $$

    $\mathbf{A}_{COP}$ (a [skew-symmetric matrix](https://en.wikipedia.org/wiki/Skew-symmetric_matrix)) is simply the COP position in matrix form in order to calculate the [cross product with matrix multiplication](https://en.wikipedia.org/wiki/Cross_product).

    These known loads on the FP can also be represented as:

    $$
    \mathbf{L}_I =
    \begin{bmatrix}
    \mathbf{F}_I \\
    \mathbf{M}_I
    \end{bmatrix}
    $$
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — known moments.** A force $(0,0,500)$ N is applied at $(0.1,-0.2,0)$ m. Calculate the moment about the origin and the full six-component known load.

    /// details | Worked solution — Exercise 1 — known moments

    Use $\mathbf M=\mathbf r\times\mathbf F=(-100,-50,0)$ N m. Thus $\mathbf L_I=(0,0,500,-100,-50,0)^T$. The cross-product matrix has units of length, which is why the moment block cannot be treated as having force units.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Linear re-calibration

    For a linear re-calibration, the relationship between the measured FP output, $\mathbf{L}$, and the known loads, $\mathbf{L}_I$, is approximated by a linear equation:

    $$
    \mathbf{L}_I = \mathbf{C}\mathbf{L} + \mathbf{E}
    $$

    Where $\mathbf{C}$ now is the six-by-six re-calibration matrix (with units that depend on the block: force/force and moment/moment are dimensionless, moment/force has units m, and force/moment has units 1/m) and $\mathbf{E}$ is a gaussian, uncorrelated, zero mean
    noise six-by-one matrix.<br>
    The re-calibration matrix can be found by solving the equation above and then $\mathbf{C}$ can be later used to re-calibrate the FP output:

    $$
    \mathbf{L}_C = \mathbf{C}\mathbf{L}
    $$

    Where $\mathbf{L}_C$ is the re-calibrated FP output. For a perfectly calibrated FP, $\mathbf{L}_C = \mathbf{L}$ and $\mathbf{C} = \mathbf{I}$, the six-by-six identity matrix.

    Cedraro et al. (2008, 2009) proposed to use a calibrated three-component load cell (LC) to measure the loads $\mathbf{F}_I(t)$ applied on the FP at $k$ known measurements sites. The LC measures the loads in its own coordinate system $(xyz)$: $\mathbf{F}_{LC}(t) = [F_x(t),\, F_y(t),\, F_z(t)]^T$, which is probaly rotated (by an unknown value, represented by rotation matrix $\mathbf{R}^k$) in relation to the FP coordinate system (the coordinate systems are also translated to each other but the translation is known and given by the COP position).<br>
    For each measurement site, the equation for the determination of the re-calibration matrix will be given by:

    $$
    \mathbf{P}^k\mathbf{R}^k\mathbf{F}^k_{LC}(t)= \mathbf{P}^k\mathbf{F}_I^k(t) = \mathbf{C}\mathbf{L}^k(t) + \mathbf{E}^k(t) \quad k = 1, ..., n
    $$

    Where:

    $$
    \mathbf{P}^k =
    \begin{bmatrix}
    \mathbf{I}_3 \\
    \mathbf{A}_{COP}
    \end{bmatrix}
    $$

    and $I_3$ is the three-by-three identity matrix.

    Using a typical load cell, with a flat bottom, on top the FP, a realistic assumption is to consider that $z$ of LC is aligned to $Z$ of FP (the vertical direction); in this case the rotation matrix is:

    $$
    \mathbf{R}^k =
    \begin{bmatrix}
    \cos\alpha^k && -\sin\alpha^k && 0 \\
    \sin\alpha^k && \cos\alpha^k && 0 \\
    0 && 0 && 1
    \end{bmatrix}
    $$

    Cedraro et al. (2008) propose the following algorithm to estimate $\mathbf{C}$:
    1. The misalignments, $\alpha^k$, are initialized: $\mathbf{\alpha} = [\alpha^1, \cdots, \alpha^n]$;
    2. $\mathbf{C}$ is calculated by a least-squares approach;
    3. The residual errors are estimated as: $\mathbf{E}^k(t) = \mathbf{P}^k\mathbf{R}^k\mathbf{F}^k_{LC}(t) - \mathbf{C}\mathbf{L}^k(t)$;
    4. The increment $\mathbf{\Delta\alpha}$ is calculated by minimizing the cost function $\sum_{k,t}\mathbf{E}^k(t)^T\mathbf{E}^k(t)$, assuming dimensional unitary weights;
    5. The parameters are updated: $\mathbf{\alpha} = \mathbf{\alpha} + \mathbf{\Delta\alpha}$.

    The iteration of steps 2–5 stops when each $|\Delta\alpha^k| < \varepsilon_0$, where $\varepsilon_0=10^{-10}$ is the chosen threshold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before fitting, distinguish a voltage calibration from a re-calibration of existing force/moment channels. The latter still mixes units across its off-diagonal blocks. Likewise, summing raw squared force and moment residuals implicitly chooses a numerical weighting tied to the chosen units. For experimental use, specify residual scales or an uncertainty-based weighting matrix; changing metres to millimetres without adjusting that weighting changes the fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulation

    Let's simulate some data to test this calibration procedure. Cedraro et al. (2008) employed sinusoids, cosenoids, and ramps as sintetic signals to simulate the calibration process:
    """)
    return


@app.cell
def _():
    import numpy as np
    from numpy.linalg import inv
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    import seaborn as sns
    sns.set_context("notebook", font_scale=1.4,
                    rc={"lines.linewidth": 3, "lines.markersize": 8, "axes.titlesize": 'x-large'})
    return inv, np, plt


@app.cell
def _(np, plt):
    # simulated forces measured by the load cell in its local coordinate system
    _rng = np.random.default_rng(0)
    samples = np.linspace(1, 6000, 6000)
    ns = samples.shape[0]
    Flc = np.array([100*np.sin(5*2*np.pi*samples/ns) + 2*_rng.standard_normal(6000),
                    100*np.cos(5*2*np.pi*samples/ns) + 2*_rng.standard_normal(6000),
                    samples/15 + 200 + 5*_rng.standard_normal(6000)])
    # plots
    fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex='all')
    axs[0].plot(samples, Flc[0])
    axs[0].set_ylabel('Fx (N)')
    axs[0].locator_params(axis='y', nbins=3)
    axs[0].yaxis.set_label_coords(-.08, 0.5)
    axs[1].plot(samples, Flc[1])
    axs[1].set_ylabel('Fy (N)')
    axs[1].locator_params(axis='y', nbins=3)
    axs[1].yaxis.set_label_coords(-.08, 0.5)
    axs[2].plot(samples, Flc[2])
    axs[2].set_ylabel('Fz (N)')
    axs[2].set_xlabel('Samples')
    axs[2].locator_params(axis='y', nbins=3)
    axs[2].yaxis.set_label_coords(-.08, 0.5)
    plt.tight_layout(pad=.5, h_pad=.025)
    plt.show()
    return Flc, ns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And Cedraro et al. (2008) also proposed five measurement sites and a re-calibration matrix for the simulated re-calibration:
    """)
    return


@app.cell
def _(Flc, np):
    C = np.array([[1.0354, -0.0053, -0.0021, -0.0289, -0.0402, 0.0081], [0.0064, 1.0309, -0.0031, 0.0211, 0.0135, -0.0001], [0.0, -0.0004, 1.0022, -0.0005, -0.0182, 0.03], [-0.0012, -0.0385, 0.0002, 0.9328, 0.0007, 0.0017], [0.0347, 0.0003, 0.0008, -0.0002, 0.9325, -0.0024], [-0.0004, -0.0013, -0.0003, -0.0023, 0.0035, 1.0592]])
    COP = np.array([[0, 112, 112, -112, -112], [0, 192, -192, 192, -192], [-124, -124, -124, -124, -124]]) / 1000
    nk = COP.shape[1]
    Acop = lambda x, y, z: np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    Flc_1 = np.tile(Flc, nk)
    return Acop, C, COP, Flc_1, nk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's generate the loads measured by the FP given the re-calibration matrix and the simulated forces measured by the load cell (we will consider no rotation for now). For that we will have to solve the equation:

    $$
    \mathbf{L}_I = \mathbf{C}\mathbf{L}
    $$

    Which is:

    $$
    \mathbf{L} = \mathbf{C}^{-1}\mathbf{L}_I
    $$

    $\mathbf{C}$ is a square (6-by-6) matrix and the computation of its inverse is straightforward.
    """)
    return


@app.cell
def _(Acop, C, COP, Flc_1, inv, nk, np, ns):
    Li = np.empty((6, ns * nk))
    P = np.empty((6, 3, nk))
    for (k, cop) in enumerate(COP.T):
        P[:, :, k] = np.vstack((np.eye(3), Acop(*cop)))
        Li[:, k * ns:(k + 1) * ns] = P[:, :, k] @ Flc_1[:, k * ns:(k + 1) * ns]
    L = inv(C) @ Li
    return L, Li


@app.cell
def _(mo):
    mo.md(r"""
    In the calculations above we took advantage of the [new operator for matrix multiplcation in Python 3](https://www.python.org/dev/peps/pep-0465/): `@` (mnemonic: `@` is `*` for mATrices).

    We can now simulate the re-calibration procedure by determining the re-calibration matrix using these loads. Of course, the re-calibration matrix to be determined should be equal to the simulated re-calibration matrix we started with, but this is the fun of the simulation - we know where we want to go.

    The re-calibration matrix can be found by solving the following equation (considering the angles equal zero for now):

    $$
    \mathbf{L}_I = \mathbf{C}\mathbf{L}
    $$

    $$
    \mathbf{L}_I \mathbf{L}^{-1} = \mathbf{C}\mathbf{L} \mathbf{L}^{-1} = \mathbf{C}\mathbf{I}
    $$

    $$
    \mathbf{C} = \mathbf{L}_I\mathbf{L}^{-1}
    $$

    The problem is that $\mathbf{L}$ in general is a non-square matrix and its inverse is not defined (unless you perform exactly six measurements and then $\mathbf{L}$ would be a six-by-six square matrix, but this is too restrictive). However, we still can solve the equation with some extra manipulation:

    $$
    \mathbf{L}_I = \mathbf{C}\mathbf{L}
    $$

    $$
    \mathbf{L}_I \mathbf{L}^T = \mathbf{C}\mathbf{L} \mathbf{L}^T
    $$

    $$
    \mathbf{L}_I \mathbf{L}^T(\mathbf{L}\mathbf{L}^T)^{-1} = \mathbf{C}\mathbf{L} \mathbf{L}^T (\mathbf{L}\mathbf{L}^T)^{-1} = \mathbf{C}\mathbf{I}
    $$

    $$
    \mathbf{C} = \mathbf{L}_I\mathbf{L}^T(\mathbf{L}\mathbf{L}^T)^{-1}
    $$

    Note that $\mathbf{L} \mathbf{L}^T$ is a square matrix and is invertible (also [nonsingular](https://en.wikipedia.org/wiki/Invertible_matrix)) if $\mathbf{L}$ is L.I. ([linearly independent rows/columns](https://en.wikipedia.org/wiki/Linear_independence)). The matrix $\mathbf{L}^T(\mathbf{L}\mathbf{L}^T)^{-1}$ is known as the [generalized inverse or Moore–Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse), a generalization of the inverse matrix. If we denote this pseudoinverse matrix by $\mathbf{L}^+$, we can state the solution of the equation simply as:

    $$
    \mathbf{L}_I = \mathbf{C}\mathbf{L}
    $$

    $$
    \mathbf{C} = \mathbf{L}_I \mathbf{L}^+
    $$

    To compute the Moore–Penrose pseudoinverse, we could calculate it by the naive approach in Python:
    ```python
    Linv = L.T @ inv(L @ L.T)
    ```
    But both Numpy and Scipy have functions to calculate the pseudoinverse, which might give greater numerical stability (but read [Inverses and pseudoinverses. Numerical issues, speed, symmetry](http://vene.ro/blog/inverses-pseudoinverses-numerical-issues-speed-symmetry.html)).<br>
    Of note, [numpy.linalg.pinv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html) calculates the pseudoinverse of a matrix using its singular-value decomposition (SVD) and including all large singular values (using the [LAPACK (Linear Algebra Package)](https://en.wikipedia.org/wiki/LAPACK) routine gesdd), whereas [scipy.linalg.pinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv.html) also uses the SVD (it used a least-squares solver until SciPy 1.7, when the SVD-based `scipy.linalg.pinv2` was folded into it; `pinv2` was removed in SciPy 1.9).
    Let's use [scipy.linalg.pinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv.html):
    """)
    return


@app.cell
def _(L):
    from scipy.linalg import pinv as _pinv
    Lpinv = _pinv(L)
    return (Lpinv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, the re-calibration matrix is:
    """)
    return


@app.cell
def _(Li, Lpinv):
    C2 = Li @ Lpinv
    return (C2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which is indeed the same as the initial calibration matrix:
    """)
    return


@app.cell
def _(C, C2, np):
    np.allclose(C, C2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The residual error between the old loads and new loads after re-calibration is:
    """)
    return


@app.cell
def _(C2, L, Li, np):
    E = Li - C2 @ L
    e = np.sum(E * E)
    print('Average residual error between old and new loads:', e)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — rank and pseudoinverse.** A calibration experiment applies only vertical forces at the plate centre. Can it identify an arbitrary $6\times6$ re-calibration matrix? How does an SVD pseudoinverse change the answer?

    /// details | Worked solution — Exercise 2 — rank and pseudoinverse

    The measured loads span only one direction in this idealized experiment, so infinitely many calibration matrices fit them. A pseudoinverse returns one minimum-norm least-squares solution; it does not create missing excitation. To identify the full linear map, the six load rows must have full row rank, and adequate conditioning and independent validation are also needed.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optimization

    Let's now implement the full algorithm considering the likely rotation of the load cell during a re-calibration.

    The idea is to guess initial values for the angles, estmate the re-calibration matrix, estimate new values for the angles that minimize the equation for the residuals and then estimate again the re-calibration matrix in an iterative approach until the estimated angles converge to the actual angles of the load cell in the different sites. This is a typical problem of [optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) where the angles are the design variables and the equation for the residuals is the cost function (see this [notebook about optimization](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/optimization.py)).

    Let's code the optimization in a complete function for the force plate re-calibration, named `fpcalibra.py`, with the following signature:
    ```python
       C, ang = fpcalibra(Lfp, Flc, COP, threshold=1e-10)
    ```

    Let's run the embedded, corrected implementation with the original simulation:
    """)
    return






@app.cell
def _(fpcalibra, inv, np):
    # simulated true re-calibration matrix
    _C = np.array([[ 1.0354, -0.0053, -0.0021, -0.0289, -0.0402,  0.0081],
                  [ 0.0064,  1.0309, -0.0031,  0.0211,  0.0135, -0.0001],
                  [ 0.0000, -0.0004,  1.0022, -0.0005, -0.0182,  0.0300],
                  [-0.0012, -0.0385,  0.0002,  0.9328,  0.0007,  0.0017],
                  [ 0.0347,  0.0003,  0.0008, -0.0002,  0.9325, -0.0024],
                  [-0.0004, -0.0013, -0.0003, -0.0023,  0.0035,  1.0592]])
    # simulated 5 measurements sites (in m)
    _COP = np.array([[   0,  112,  112, -112, -112],
                    [   0,  192, -192,  192, -192],
                    [-124, -124, -124, -124, -124]])/1000
    _nk = _COP.shape[1]
    # simulated forces measured by the load cell (in N) before rotation
    _samples = np.linspace(1, 6000, 6000)
    _ns = _samples.shape[0]
    _Flc = np.empty((3, _nk*_ns))
    _rng = np.random.default_rng(0)
    for _k in range(_nk):
        _Flc[:, _k*_ns:(_k+1)*_ns] = np.array([100*np.sin(5*2*np.pi*_samples/_ns) + 2*_rng.standard_normal(_ns),
                                          100*np.cos(5*2*np.pi*_samples/_ns) + 2*_rng.standard_normal(_ns),
                                          _samples/15 + 200 + 5*_rng.standard_normal(_ns)])
    # function for the COP skew-symmetric matrix
    _Acop = lambda x,y,z : np.array([[.0, -z, y], [z, .0, -x], [-y, x, .0]])
    # simulated loads measured by the force plate
    _Li = np.empty((6, _ns*_nk))
    _P = np.empty((6, 3, _nk))
    for _k, _cop in enumerate(_COP.T):
        _P[:, :, _k] = np.vstack((np.eye(3), _Acop(*_cop)))
        _Li[:, _k*_ns:(_k+1)*_ns] = _P[:, :, _k] @ _Flc[:, _k*_ns:(_k+1)*_ns]
    _Lfp = inv(_C) @  _Li
    # simulated angles of rotaton of the measurement sites
    _ang = np.array([20, -10, 0, 15, -5])/180*np.pi
    # function for the rotation matrix
    _R = lambda a : np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [ 0, 0, 1]])
    # simulated forces measured by the load cell after rotation
    for _k in range(_nk):
        _Flc[:, _k*_ns:(_k+1)*_ns] = _R(_ang[_k]).T @ _Flc[:, _k*_ns:(_k+1)*_ns]

    _C2, _ang2 = fpcalibra(_Lfp, _Flc, _COP)

    _e = np.sqrt(np.sum((_C2-_C)**2))
    print('Residual between simulated and optimal re-calibration matrices:', _e)
    _e = np.sqrt(np.sum((_ang2-_ang)**2))
    print('Residual between simulated and optimal rotation angles:', _e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulation works as expected and the function was able to estimate accurately the known initial re-calibration matrix and angles of rotation.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — convergence.** Angle changes at five sites are $(0,0.02,0.01,0.03,0.01)$ rad and the stopping tolerance is $10^{-6}$ rad. Has the alternating fit converged? Which condition should continue the loop?

    /// details | Worked solution — Exercise 3 — convergence

    No: four sites still change. Continue while **any** absolute change exceeds tolerance, `np.any(delta_ang > threshold)`. Using `all` would stop as soon as one site becomes stationary. Recompute the calibration using the final angles so both returned quantities belong to the same final iterate.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Non-linear algorithm for force plate calibration

    Cappello et al. (2011) extended the algorithm described earlier and proposed an algorithm for non-linear re-calibration of FPs.<br>
    The idea is that a load applied on the FP produces bending which depends on the point of force application and in turn will result in systematic errors in the COP determination. Consequently, this non-linearity could be modeled and compensated with a re-calibration which takes into account the COP coordinates measured by the FP and added to the linear re-calibration we deduced above (Cappello et al., 2011).<br>
    The re-calibration equation will be (Cappello et al., 2011):

    $$
    \begin{array}{l l}
    \mathbf{L}_C = \mathbf{C}_0\mathbf{L} + \,
    \begin{bmatrix}
    C_{x_{11}} & C_{x_{12}} & 0 & C_{x_{14}} & C_{x_{15}} & C_{x_{16}} \\
    C_{x_{21}} & C_{x_{22}} & 0 & C_{x_{24}} & C_{x_{25}} & C_{x_{26}} \\
    C_{x_{31}} & C_{x_{32}} & 0 & C_{x_{34}} & C_{x_{35}} & C_{x_{36}} \\
    C_{x_{41}} & C_{x_{42}} & 0 & C_{x_{44}} & C_{x_{45}} & C_{x_{46}} \\
    C_{x_{51}} & C_{x_{52}} & 0 & C_{x_{54}} & C_{x_{55}} & C_{x_{56}} \\
    C_{x_{61}} & C_{x_{62}} & 0 & C_{x_{64}} & C_{x_{65}} & C_{x_{66}}
    \end{bmatrix}\,
    \begin{bmatrix}
    F_x \\ F_y \\ F_z \\ M_x \\ M_y \\ M_z
    \end{bmatrix} COP_x +
    \begin{bmatrix}
    C_{y_{11}} & C_{y_{12}} & 0 & 0 & C_{y_{15}} & C_{y_{16}} \\
    C_{y_{21}} & C_{y_{22}} & 0 & 0 & C_{y_{25}} & C_{y_{26}} \\
    C_{y_{31}} & C_{y_{32}} & 0 & 0 & C_{y_{35}} & C_{y_{36}} \\
    C_{y_{41}} & C_{y_{42}} & 0 & 0 & C_{y_{45}} & C_{y_{46}} \\
    C_{y_{51}} & C_{y_{52}} & 0 & 0 & C_{y_{55}} & C_{y_{56}} \\
    C_{y_{61}} & C_{y_{62}} & 0 & 0 & C_{y_{65}} & C_{y_{66}}
    \end{bmatrix}\,
    \begin{bmatrix}
    F_x \\ F_y \\ F_z \\ M_x \\ M_y \\ M_z
    \end{bmatrix} COP_y
    \\[6pt]
    \mathbf{L}_C = (\mathbf{C}_0 + \mathbf{C}_x COP_x + \mathbf{C}_y COP_y)\mathbf{L} = \mathbf{C}_{NL}\mathbf{L}
    \end{array}
    $$

    Where $\mathbf{C}_0$ is the linear re-calibration matrix, $\mathbf{L}$ is the measured FP output, $\mathbf{C}_x$ and  $\mathbf{C}_y$ are the non-linear re-calibration matrices.

    To estimate $\mathbf{C}_{NL}$, Cappello et al. (2011) suggest to employ the algorithm proposed by Cedraro et al. (2008) to estimate the linear re-calibration described earlier.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The position-dependent model below is retained as theory from Cappello et al. The following `_fpcalibra_nl` block is an **unfinished historical draft**: its body repeats the linear algorithm and does not fit the extra $C_x,C_y$ terms. It is included for reading and extension, not called as a completed nonlinear calibration. A complete implementation needs the expanded design, constraints, identifiability checks and independent validation.
    """)
    return


@app.cell
def _(minimize, np, pinv, time):
    # Draft (never run in the original notebook): the body of the linear algorithm of
    # fpcalibra, as a starting point for implementing the non-linear re-calibration
    def _fpcalibra_nl(Lfp, Flc, COP, threshold=1e-10, method='SVD'):
        # number of sites
        nk = COP.shape[1]
        # number of samples
        ns = int(Lfp.shape[1]/nk)
        # function for the COP skew-symmetric matrix
        Acop = lambda x,y,z : np.array([[.0, -z, y], [z, .0, -x], [-y, x, .0]])
        P = np.empty((6, 3, nk))
        for k, cop in enumerate(COP.T):
            P[:, :, k] = np.vstack((np.eye(3), Acop(*cop)))
        # function for the 2D rotation matrix
        R = lambda a : np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [ 0, 0, 1]])
        # Pseudoiverse of the loads measured by the force plate
        if method.lower() == 'svd':
            Lpinv = pinv(Lfp)
        else:
            Lpinv = pinv(Lfp)
        # cost function for the optimization
        def costfun(ang, P, R, Flc, CLfp, nk, ns, E):
            for k in range(nk):
                E[:,k*ns:(k+1)*ns] = (P[:,:,k] @ R(ang[k])) @ Flc[:,k*ns:(k+1)*ns] - CLfp[:,k*ns:(k+1)*ns]
            return np.sum(E * E)
        # inequality constraints
        bnds = [(-np.pi/2, np.pi/2) for k in range(nk)]
        # some initialization
        ang0 = np.zeros(nk)
        E = np.empty((6, ns*nk))
        da = []
        delta_ang = 10*threshold
        Li = np.empty((6, ns*nk))
        start = time.time()
        # the optimization
        while np.any(delta_ang > threshold):
            for k in range(nk):
                Li[:,k*ns:(k+1)*ns] = (P[:,:,k] @ R(ang0[k])) @ Flc[:,k*ns:(k+1)*ns]
            C = Li @ Lpinv
            CLfp = C @ Lfp
            res = minimize(fun=costfun, x0=ang0, args=(P, R, Flc, CLfp, nk, ns, E),
                           bounds=bnds, method='TNC', options={'disp': False})
            delta_ang = np.abs(res.x - ang0)
            ang0 = res.x
            da.append(delta_ang.sum())

        tdelta = time.time() - start
        print('\nOptimization finished in %.1f s after %d steps.\n' %(tdelta, len(da)))
        print('Optimal calibration matrix:\n', C)
        print('\nOptimal angles:\n', res.x*180/np.pi)
        print('\n')

        return C, res.x
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function fpcalibra.py
    """)
    return


@app.cell
def _(np):
    """Force plate calibration algorithm.
    """

    __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version__ = 'fpcalibra.py v.1.0.1 2016/08/19'
    __license__ = "MIT"

    from scipy.linalg import pinv
    from scipy.optimize import minimize
    import time


    def fpcalibra(Lfp, Flc, COP, threshold=1e-10, method='SVD'):
        """Force plate calibration algorithm.

        For a force plate (FP) re-calibration, the relationship between the
        measured FP output (L) and the known loads (Li) is approximated by:
        Li = C@L + E (@ is the operator for matrix multiplication).
        Where C is the 6-by-6 re-calibration matrix and E is a gaussian,
        uncorrelated, zero mean noise six-by-one matrix.

        The re-calibration matrix can be found by solving the equation above and
        then C can be later used to re-calibrate the FP output: Lc = C@L.
        Where Lc is the re-calibrated FP output.

        Cedraro et al. (2008) [1]_ proposed to use a calibrated three-component
        load cell to measure the forces applied on the FP at known measurement
        sites and an algorithm for the re-calibration.

        This code implements the re-calibration algorithm, see [2]_

        Parameters
        ----------
        Lfp : numpy 2-D array (6, nsamples*nksites)
            loads [Fx, Fy, Fz, Mx, My, Mz] (in N and Nm) measured by the force
            plate due to the corresponding forces applied at the measurement sites
        Flc : numpy 2-D array (3, nsamples*nksites)
            forces [Fx, Fy, Fz] (in N) measured by the load cell at the
            measurement sites
        COP : numpy 2-D array (3, nksites)
            positions [COPx, COPy, COPz] (in m) of the load cell at the
            measurement sites
        threshold  : float, optional
            threshold to stop the optimization (default 1e-10)
        method  : string, optional
            legacy pseudoinverse selector, 'SVD' (default) or 'lstsq'. Both values
            now use scipy.linalg.pinv, which is SVD-based.

        Returns
        -------
        C   : numpy 2-D (6-by-6) array
            force plate re-calibration matrix. Force/force and moment/moment blocks
            are dimensionless; force/moment and moment/force blocks have units 1/m
            and m, respectively, when loads use N and Nm.
        ang : numpy 1-D array [ang0, ..., angk]
            optimal angles of rotation (in rad) of the load cells at the
            measurement sites

        References
        ----------
        .. [1] Cedraro A, Cappello A, Chiari L (2008) Gait & Posture, 28, 488–494.
        .. [2] force_plate_calibration.py

        Example
        -------
        >>> from fpcalibra import fpcalibra
        >>>     >>> from numpy.linalg import inv
        >>>
        >>> # simulated true re-calibration matrix
        >>> C = np.array([[ 1.0354, -0.0053, -0.0021, -0.0289, -0.0402,  0.0081],
        >>>               [ 0.0064,  1.0309, -0.0031,  0.0211,  0.0135, -0.0001],
        >>>               [ 0.0000, -0.0004,  1.0022, -0.0005, -0.0182,  0.0300],
        >>>               [-0.0012, -0.0385,  0.0002,  0.9328,  0.0007,  0.0017],
        >>>               [ 0.0347,  0.0003,  0.0008, -0.0002,  0.9325, -0.0024],
        >>>               [-0.0004, -0.0013, -0.0003, -0.0023,  0.0035,  1.0592]])
        >>> # simulated 5 measurements sites (in m)
        >>> COP = np.array([[   0,  112,  112, -112, -112],
        >>>                 [   0,  192, -192,  192, -192],
        >>>                 [-124, -124, -124, -124, -124]])/1000
        >>> nk = COP.shape[1]
        >>> # simulated forces measured by the load cell (in N) before rotation
        >>> samples = np.linspace(1, 6000, 6000)
        >>> ns = samples.shape[0]
        >>> Flc = np.empty((3, nk*ns))
        >>> for k in range(nk):
        >>>     Flc[:, k*ns:(k+1)*ns] = np.array([100*np.sin(5*2*np.pi*samples/ns) + 2*np.random.randn(ns),
        >>>                                       100*np.cos(5*2*np.pi*samples/ns) + 2*np.random.randn(ns),
        >>>                                       samples/15 + 200 + 5*np.random.randn(ns)])
        >>> # function for the COP skew-symmetric matrix
        >>> Acop = lambda x,y,z : np.array([[.0, -z, y], [z, .0, -x], [-y, x, .0]])
        >>> # simulated loads measured by the force plate
        >>> Li = np.empty((6, ns*nk))
        >>> P = np.empty((6, 3, nk))
        >>> for k, cop in enumerate(COP.T):
        >>>     P[:, :, k] = np.vstack((np.eye(3), Acop(*cop)))
        >>>     Li[:, k*ns:(k+1)*ns] = P[:, :, k] @ Flc[:, k*ns:(k+1)*ns]
        >>> Lfp = inv(C) @  Li
        >>> # simulated angles of rotaton of the measurement sites
        >>> ang = np.array([20, -10, 0, 15, -5])/180*np.pi
        >>> # function for the rotation matrix
        >>> R = lambda a : np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [ 0, 0, 1]])
        >>> # simulated forces measured by the load cell after rotation
        >>> for k in range(nk):
        >>>     Flc[:, k*ns:(k+1)*ns] = R(ang[k]).T @ Flc[:, k*ns:(k+1)*ns]
        >>>
        >>> C2, ang2 = fpcalibra(Lfp, Flc, COP)
        >>>
        >>> e = np.sqrt(np.sum((C2-C)**2))
        >>> print('Residual between simulated and optimal re-calibration matrices:', e)
        >>> e = np.sqrt(np.sum((ang2-ang)**2))
        >>> print('Residual between simulated and optimal rotation angles:', e)
        """

        # number of sites
        nk = COP.shape[1]
        # number of samples
        ns = int(Lfp.shape[1]/nk)
        # function for the COP skew-symmetric matrix
        Acop = lambda x,y,z : np.array([[.0, -z, y], [z, .0, -x], [-y, x, .0]])
        P = np.empty((6, 3, nk))
        for k, cop in enumerate(COP.T):
            P[:, :, k] = np.vstack((np.eye(3), Acop(*cop)))
        # function for the 2D rotation matrix
        R = lambda a : np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [ 0, 0, 1]])
        # Pseudoiverse of the loads measured by the force plate
        if method.lower() == 'svd':
            Lpinv = pinv(Lfp)  # pinv2 was removed in SciPy 1.9; pinv now also uses the SVD
        else:
            Lpinv = pinv(Lfp)
        # cost function for the optimization
        def costfun(ang, P, R, Flc, CLfp, nk, ns, E):
            for k in range(nk):
                E[:,k*ns:(k+1)*ns] = (P[:,:,k] @ R(ang[k])) @ Flc[:,k*ns:(k+1)*ns] - CLfp[:,k*ns:(k+1)*ns]
            return np.sum(E * E)
        # inequality constraints
        bnds = [(-np.pi/2, np.pi/2) for k in range(nk)]
        # some initialization
        ang0 = np.zeros(nk)
        E = np.empty((6, ns*nk))
        da = []
        delta_ang = 10*threshold
        Li = np.empty((6, ns*nk))
        start = time.time()
        # the optimization
        while np.any(delta_ang > threshold):
            for k in range(nk):
                Li[:,k*ns:(k+1)*ns] = (P[:,:,k] @ R(ang0[k])) @ Flc[:,k*ns:(k+1)*ns]
            C = Li @ Lpinv
            CLfp = C @ Lfp
            res = minimize(fun=costfun, x0=ang0, args=(P, R, Flc, CLfp, nk, ns, E),
                           bounds=bnds, method='TNC', options={'disp': False})
            delta_ang = np.abs(res.x - ang0)
            ang0 = res.x
            da.append(delta_ang.sum())

        # Return a calibration consistent with the final angle estimate.
        for k in range(nk):
            Li[:,k*ns:(k+1)*ns] = (P[:,:,k] @ R(ang0[k])) @ Flc[:,k*ns:(k+1)*ns]
        C = Li @ Lpinv

        tdelta = time.time() - start
        print('\nOptimization finished in %.1f s after %d steps.\n' %(tdelta, len(da)))
        print('Optimal calibration matrix:\n', C)
        print('\nOptimal angles:\n', res.x*180/np.pi)
        print('\n')

        return C, res.x
    return fpcalibra, minimize, pinv, time


@app.cell
def _():
    import marimo as mo

    return (mo,)



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4 — a position-dependent correction.** A single channel obeys $F_c=(1+0.2x)F$, with $x$ in m. Evaluate it at $x=0$ and $x=0.1$ m for $F=500$ N. What units must 0.2 carry?

    /// details | Worked solution — Exercise 4

    At the centre the result is 500 N. At 0.1 m the gain is 1.02 and the result is 510 N. The coefficient has units m⁻¹ so that its product with position is dimensionless. This example illustrates the extra position dependence; it is not a fit of the full Cappello model.

    ///

    ## Checkpoint questions

    - Does a small residual establish that all calibration coefficients are identifiable?
    - Why are load-cell rotation and calibration alternated in the example?
    - What further evidence is needed before a more flexible nonlinear model is preferable?

    /// details | Checkpoint answers

    No: inspect excitation, rank and conditioning, as well as residuals. The reference forces must be expressed in the plate frame; the unknown orientations and calibration affect each other's estimates. A nonlinear model needs sufficient spatial/loading coverage and improved performance on independent validation data, not just a smaller training residual.

    ///

    ## Problems

    1. A measured force is 525 N when the true force is 500 N. Find the scalar re-calibration gain.
    2. For $\mathbf r=(0.2,0.1,0)$ m and $\mathbf F=(10,20,300)$ N, give $\mathbf r\times\mathbf F$.
    3. State the units of a coefficient multiplying $F_z$ to predict $M_x$, and one multiplying $M_y$ to predict $F_x$.
    4. A $6\times1000$ load matrix has rank 4. Is an arbitrary six-by-six re-calibration unique? What can the pseudoinverse provide?
    5. Changes are $(10^{-8},10^{-3})$ rad with tolerance $10^{-6}$ rad. Stop or continue? Does the retained nonlinear draft estimate $C_x,C_y$?

    /// details | Final answers to the problems

    1. $20/21\approx0.952381$.
    2. $(30,-60,3)$ N m.
    3. m; m⁻¹.
    4. No; a minimum-norm least-squares solution, with unidentifiable directions remaining.
    5. Continue; the draft does not estimate those matrices.

    ///

    ## Go deeper

    Compare an independently loaded validation trial before and after calibration. Report force and moment errors separately and specify any weighting. For the nonlinear extension, examine whether your calibration sites cover the positions used in validation. Keep the original Cedraro and Cappello papers below as the algorithmic sources.

    ### Video lectures on the Internet

    - [MIT 18.06SC: projection matrices and least squares](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/projection-matrices-and-least-squares/) — relate the projection view to the load-matrix pseudoinverse.
    """)
    return







@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Cedraro A, Cappello A, Chiari L (2008) A portable system for in-situ re-calibration of force platforms: theoretical validation. Gait & Posture, 28, 488–494](https://pubmed.ncbi.nlm.nih.gov/18450453/).

    - [Cedraro A, Cappello A, Chiari L (2009) A portable system for in-situ re-calibration of force platforms: experimental validation. Gait & Posture, 29, 449–453](https://pubmed.ncbi.nlm.nih.gov/19111467/).

    - [Cappello A, Bagala F, Cedraro A, Chiari L (2011) Non-linear re-calibration of force platforms. Gait & Posture, 33, 724–726](https://pubmed.ncbi.nlm.nih.gov/21392999/).

    - [SciPy `pinv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv.html) — the current SVD-based pseudoinverse, including rank tolerance.

    - [Modern Robotics: wrenches](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-4-wrenches/) — force/moment representation and change of reference frame.
    """)
    return


if __name__ == "__main__":
    app.run()
