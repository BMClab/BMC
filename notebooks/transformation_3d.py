# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "sympy>=1.13", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rigid-body transformations in three dimensions

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    In a plane, rotations are easy. One angle describes an orientation, and angles add: turn by 30° and then by 20°, and you have turned by 50°, in whichever order you like. In three dimensions neither statement survives. An orientation needs three angles, and the result of two rotations depends on the order in which they are made. Almost everything that is hard about three-dimensional kinematics comes from that one fact.

    This notebook builds the tools to handle it. It starts with the rotation matrix, the object that describes an orientation without ambiguity, and then turns to Euler angles, the three numbers that people actually report, with their conventions and their failure mode, the gimbal lock. It ends where motion analysis does: combining rotation and translation, first on a static example from a textbook and then on every frame of a real walking trial.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a result *before* the code prints it. Make the prediction; a wrong guess that the code then corrects teaches more than a right one.

    The numbered **Exercises** come with solutions, folded away under each exercise. Try the exercise first, and open the solution only to check your answer or when you are stuck.

    The notebook follows the same rationale as [Rigid-body transformations in a plane (2D)](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py) and skips the fundamental concepts covered there, so if you have not read that notebook yet, read it first. You will also need matrix multiplication (see [Matrix](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/matrix.py)), and the construction of a basis from three markers, explained in [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py) and [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py).

    **Challenge 0.** Take a book and hold it upright in front of you, with its spine towards you. Turn it by 90° about a horizontal axis pointing to your right, and then by 90° about the vertical axis. Note where the spine ends up. Now start again and make the same two turns in the opposite order. Sketch both results; the section on rotation shows the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Codman's paradox

    Stand up and let your right arm hang at your side, with the palm against your thigh and the thumb pointing forward. Keeping the elbow straight, make three movements, each a quarter of a turn:

    1. Raise the arm forward until it is horizontal (90° of flexion).
    2. Keeping it horizontal, swing it out to the side (90° of horizontal abduction).
    3. Lower it to your side again (90° of adduction).

    The arm is back where it started. Now look at your palm.

    This is known as Codman's paradox, after the surgeon Ernest Codman, who described it while studying the shoulder. None of the three movements turned the arm about its own long axis, and yet the hand did not come back the way it left. It is not really a paradox (Politti et al., 1998): it is what rotations in three dimensions do, and later in this notebook you will compute it with three matrices.

    **Guiding questions 0.**

    1. Where does your palm face at the end? At which of the three movements, if any, did your arm turn about its long axis?
    2. Make the same three movements in a different order: first raise the arm sideways to horizontal, then swing it forward, then lower it in front of you. Do you end in the same place?
    3. If three quarter turns about three perpendicular axes can add up to a turn about a fourth axis, what does that tell you about adding angles in three dimensions?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the numerical work, [SymPy](https://www.sympy.org) the symbolic work, and Matplotlib the figures. The helper functions `CCS`, which draws coordinate systems in 3D, and `read_trc`, which reads motion capture files, come from the `functions` folder of this repository and are loaded where they are first needed.
    """)
    return


@app.cell
def _():
    import numpy as np
    import sympy as sym
    import matplotlib
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=4, suppress=True)  # suppress scientific notation for small numbers
    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return np, plt, sym


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Many results in this notebook are matrices, symbolic or numerical. The function below displays one as a formula:
    """)
    return


@app.function
def show_matrix(lhs, matrix, digits=None):
    """Display `lhs` = `matrix` as LaTeX, for a SymPy or NumPy matrix.

    With `digits`, the entries are evaluated numerically to that number of
    significant digits, and entries that are zero to within rounding are shown
    as zero.
    """
    import marimo as mo
    import sympy as sym

    M = sym.Matrix(matrix)
    if digits is not None:
        M = M.n(digits, chop=True)
    return mo.md(rf"$${lhs} = {sym.latex(M, mat_str='matrix')}$$")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rigid bodies and their transformations

    The kinematics of a rigid body is completely described by its pose, that is, its position and orientation in space, and by the changes of pose, translation and rotation. In three-dimensional space, at least three coordinates and three angles are needed to describe the pose of a rigid body: six degrees of freedom.

    In motion analysis, to describe the translation and rotation of a rigid body with respect to a coordinate system, we typically attach another coordinate system to the rigid body and determine the transformation between the two coordinate systems.

    A transformation is any function mapping a set to another set. For the kinematics of rigid bodies we are interested only in the rigid, or Euclidean, transformations (denoted SE(3) for three-dimensional space), because they preserve the distance between every pair of points of the body, which is rigid by definition. Translations and rotations are rigid transformations. A reflection also preserves distances, but it turns a right-handed coordinate system into a left-handed one, which is usually not of interest. In turn, rigid transformations are examples of [affine transformations](https://en.wikipedia.org/wiki/Affine_transformation); shear and scaling are other affine transformations, which do not preserve lengths.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Translation

    A pure three-dimensional translation of a rigid body (or of a coordinate system attached to it) in relation to another rigid body (with another coordinate system) is illustrated in the figure below.

    <figure aria-describedby="figure-caption-1"><center><img src="https://github.com/BMClab/BMC/blob/master/images/translation3D.png?raw=1" width=450 alt="translation 3D"/></center></figure>

    <span id="figure-caption-1" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. A point in three-dimensional space represented in two coordinate systems, with one coordinate system translated.</span>

    The position of the point $\mathbf{P}$, originally described in the $xyz$ (local) coordinate system but now described in the $\mathbf{XYZ}$ (Global) coordinate system, is in vector form

    $$
    \mathbf{P_G} = \mathbf{L_G} + \mathbf{P_l}
    $$

    or, in terms of its components,

    $$
    \begin{array}{ll}
    \mathbf{P_X} =& \mathbf{L_X} + \mathbf{P}_x \\
    \mathbf{P_Y} =& \mathbf{L_Y} + \mathbf{P}_y \\
    \mathbf{P_Z} =& \mathbf{L_Z} + \mathbf{P}_z
    \end{array}
    $$

    and in matrix form:

    $$
    \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y} \\
    \mathbf{P_Z}
    \end{bmatrix} =
    \begin{bmatrix}
    \mathbf{L_X} \\
    \mathbf{L_Y} \\
    \mathbf{L_Z}
    \end{bmatrix} +
    \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y \\
    \mathbf{P}_z
    \end{bmatrix}
    $$

    In classical mechanics, this is an example of a [Galilean transformation](https://en.wikipedia.org/wiki/Galilean_transformation), the same one that relates a runner on a treadmill to a runner on the ground in [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py).

    For example, if the local coordinate system is translated by $\mathbf{L_G}=[1, 2, 3]$ in relation to the Global coordinate system, a point with coordinates $\mathbf{P_l}=[4, 5, 6]$ in the local coordinate system has the position $\mathbf{P_G}=[5, 7, 9]$ in the Global coordinate system:
    """)
    return


@app.cell
def _(np):
    _LG = np.array([1, 2, 3])  # NumPy array
    _Pl = np.array([4, 5, 6])
    _PG = _LG + _Pl

    print("PG =", _PG)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same operation works for several points at once, one per row: NumPy adds the translation to every row (this is called broadcasting).
    """)
    return


@app.cell
def _(np):
    _LG = np.array([1, 2, 3])
    _Pl = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 2D array with 3 rows and 3 columns
    _PG = _LG + _Pl

    print("PG =\n", _PG)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A force plate is installed with its axes parallel to those of the laboratory, and the origin of its coordinate system is at $\mathbf{L_G} = [0.40,\, 0.00,\, 0.30]$ m in the laboratory. The plate reports the centre of pressure of a foot at $\mathbf{P_l} = [0.12,\, 0.00,\, 0.25]$ m in its own coordinate system. Where is the centre of pressure in the laboratory? And where, in the coordinate system of the plate, is a heel marker measured at $[0.55,\, 0.05,\, 0.50]$ m in the laboratory?

    /// details | Solution to Exercise 1
    With parallel axes, changing coordinate system is a pure translation. From the plate to the laboratory, add the position of the plate's origin:

    $$
    \mathbf{P_G} = \mathbf{L_G} + \mathbf{P_l} = [0.40 + 0.12,\; 0.00 + 0.00,\; 0.30 + 0.25] = [0.52,\; 0.00,\; 0.55]\;\mathrm{m}
    $$

    From the laboratory to the plate, subtract it:

    $$
    \mathbf{P_l} = \mathbf{P_G} - \mathbf{L_G} = [0.55 - 0.40,\; 0.05 - 0.00,\; 0.50 - 0.30] = [0.15,\; 0.05,\; 0.20]\;\mathrm{m}
    $$

    The heel marker is 5 cm above the surface of the plate, as a marker on a shoe should be.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rotation

    A pure three-dimensional rotation of an $xyz$ (local) coordinate system in relation to another, $\mathbf{XYZ}$ (Global), coordinate system, and the position of a point in these two coordinate systems, are illustrated in the next figure. Remember that this is equivalent to describing a rotation between two rigid bodies.

    <figure aria-describedby="figure-caption-2"><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotation3D.png?raw=1" width=450 alt="rotation 3D"/></center></figure>

    <span id="figure-caption-2" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. A point in three-dimensional space represented in two coordinate systems, with one coordinate system rotated.</span>

    An important characteristic of rotations in three-dimensional space is that angles cannot be treated as vectors: the result of a sequence of rotations of a rigid body about different axes depends on the order of the rotations, as the next figure shows for the book of Challenge 0.

    <figure aria-describedby="figure-caption-3"><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotationsseqs2.png?raw=1" width=700 alt="sequences of rotations of a book"/></center></figure>

    <span id="figure-caption-3" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. The result of a sequence of rotations about different axes of a coordinate system depends on the order of the rotations. In the first example (first row), the rotations are about a Global (fixed) coordinate system. In the second example (second row), the rotations are about a local (rotating) coordinate system.</span>

    Compare the figure with your sketches from Challenge 0. Notice also something that the figure shows without saying: the two orders *about the fixed axes* give results that reappear, in the opposite order, *about the rotating axes*. We will come back to that.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Determination of the rotation matrix using direction cosines

    As for rotations in two dimensions, we can calculate the rotation matrix that describes the rotation of the $xyz$ (local) coordinate system in relation to the $\mathbf{XYZ}$ (Global) coordinate system from the direction cosines between the axes of the two coordinate systems:

    $$
    \mathbf{R_{Gl}} = \begin{bmatrix}
    \cos\mathbf{X}x & \cos\mathbf{X}y & \cos\mathbf{X}z \\
    \cos\mathbf{Y}x & \cos\mathbf{Y}y & \cos\mathbf{Y}z \\
    \cos\mathbf{Z}x & \cos\mathbf{Z}y & \cos\mathbf{Z}z
    \end{bmatrix}
    $$

    Note, however, that for rotations about more than one axis these angles do not lie in the main planes ($\mathbf{XY, YZ, ZX}$) of the $\mathbf{XYZ}$ coordinate system, as illustrated in the figure below for the direction angles of the $y$ axis alone. Determining these angles by simple inspection, as we did in two dimensions, is no longer simple.

    <figure aria-describedby="figure-caption-4"><center><img src="https://github.com/BMClab/BMC/blob/master/images/directioncosine3D.png?raw=1" width=260 alt="direction angles 3D"/></center></figure>

    <span id="figure-caption-4" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Definition of the direction angles of the $y$ axis of the local coordinate system in relation to the $\mathbf{XYZ}$ Global coordinate system.</span>

    The nine angles in the matrix are also obviously redundant, since only three angles are needed to describe the orientation of a rigid body in three-dimensional space.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Determination of the rotation matrix using the basis method

    As described in [Rigid-body transformations in a plane (2D)](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py), the columns (or rows) of the 3D rotation matrix $\mathbf{R_{Gl}}$ (or $\mathbf{R_{lG}}$) between two coordinate systems are the versors of the basis of the rotated coordinate system.

    Suppose we have measured the positions of at least three noncollinear markers placed on the rigid body. Let's calculate a basis from these positions, and then the rotation matrix. Given three markers $\mathbf{m_1}$, $\mathbf{m_2}$, $\mathbf{m_3}$, a basis of three orthogonal versors can be found as follows (see [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py) for this recipe in detail):

    - First axis, $\mathbf{v_1}$: the vector $\mathbf{m_2}-\mathbf{m_1}$.
    - Second axis, $\mathbf{v_2}$: the cross product between the vectors $\mathbf{v_1}$ and $\mathbf{m_3}-\mathbf{m_1}$.
    - Third axis, $\mathbf{v_3}$: the cross product between the vectors $\mathbf{v_1}$ and $\mathbf{v_2}$.

    Each of these vectors is then normalized, resulting in three orthogonal versors.

    For example, given the positions $\mathbf{m_1} = [1,0,0]$, $\mathbf{m_2} = [0,1,0]$, $\mathbf{m_3} = [0,0,1]$, a basis can be found:
    """)
    return


@app.cell
def _(np):
    m1 = np.array([1, 0, 0])
    m2 = np.array([0, 1, 0])
    m3 = np.array([0, 0, 1])

    v1 = m2 - m1
    v2 = np.cross(v1, m3 - m1)
    v3 = np.cross(v1, v2)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)
    print("Versors:", "\nv1 =", v1, "\nv2 =", v2, "\nv3 =", v3)

    _RGl = np.array([v1, v2, v3]).T  # or np.vstack((v1, v2, v3)).T
    print("\nRotation matrix RGl:\n", _RGl)
    return v1, v2, v3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A rotation matrix is not just any matrix. Its columns are orthogonal versors, so it is **orthonormal**: $\mathbf{R}^T\mathbf{R} = \mathbf{I}$, which means that its inverse is its transpose. And because the basis is right-handed, its determinant is $+1$. Let's check both, together with the norms and the dot products of the versors:
    """)
    return


@app.cell
def _(np, v1, v2, v3):
    _RGl = np.array([v1, v2, v3]).T

    print("Norm of each versor:", np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3))
    print(f"Dot products: v1.v2 = {np.dot(v1, v2):.1e}, v1.v3 = {np.dot(v1, v3):.1e}, v2.v3 = {np.dot(v2, v3):.1e}")
    print("\nRGl^T RGl =\n", (_RGl.T @ _RGl).round(10) + 0.0)
    print(f"\ndet(RGl) = {np.linalg.det(_RGl):g}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, we have the same problem as with the rotation matrix deduced from the direction cosines: there is no simple way to read angles of rotation off this matrix.

    **Exercise 2.** Which of the following matrices are rotation matrices? For those that are, find the axis and the angle of the rotation by inspection.

    $$
    \mathbf{A} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad
    \mathbf{B} = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad
    \mathbf{C} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0.8 & -0.6 \\ 0 & 0.6 & 0.8 \end{bmatrix}, \quad
    \mathbf{D} = \begin{bmatrix} 1 & 0.1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
    $$

    /// details | Solution to Exercise 2
    A rotation matrix needs both properties: orthonormal columns and a determinant of $+1$.

    - $\mathbf{A}$ has orthonormal columns, but its determinant is $-1$. It swaps the $\mathbf{X}$ and $\mathbf{Y}$ axes, a reflection in the plane $\mathbf{X} = \mathbf{Y}$, which would turn a right-handed coordinate system into a left-handed one. Not a rotation.
    - $\mathbf{B}$ is orthonormal with determinant $+1$. Its third column is $[0, 0, 1]$, so $\mathbf{Z}$ is unchanged and the axis is $\mathbf{Z}$. Its first column, the image of $\mathbf{X}$, is $[0, 1, 0]$: $\mathbf{X}$ was turned into $\mathbf{Y}$, a rotation of $+90^o$ about $\mathbf{Z}$.
    - $\mathbf{C}$ is orthonormal with determinant $0.8^2 + 0.6^2 = 1$. Its first column is $[1, 0, 0]$, so the axis is $\mathbf{X}$, and comparing it with the elemental rotation about $\mathbf{X}$, $\cos\alpha = 0.8$ and $\sin\alpha = 0.6$: $\alpha = \arctan2(0.6, 0.8) = 36.87^o$.
    - $\mathbf{D}$ has determinant $+1$, but it is not orthonormal: its second column, $[0.1, 1, 0]$, has norm $\sqrt{1.01}$ and is not perpendicular to the first. It is a shear, which changes angles and lengths. A determinant of $+1$ alone is not enough.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Euler angles

    There are different ways to describe a three-dimensional rotation of a rigid body (or of a coordinate system). The most straightforward would probably be spherical coordinates, but it would be difficult to give them an anatomical or clinical interpretation. The solution most often employed in biomechanics is Euler angles. Under certain conditions Euler angles have an anatomical interpretation, but the representation also has caveats, as we will see.

    In the 18th century, [Leonhard Euler](https://en.wikipedia.org/wiki/Leonhard_Euler) showed that two three-dimensional coordinate systems with a common origin can be related by a sequence of up to three elemental rotations about the axes of the local coordinate system, where no two successive rotations may be about the same axis. The angles of these rotations are now known as [Euler (or Eulerian) angles](https://en.wikipedia.org/wiki/Euler_angles).

    <figure aria-describedby="figure-caption-5"><center><img src="https://upload.wikimedia.org/wikipedia/commons/8/85/Euler2a.gif" width=300 alt="Euler angles"/></center></figure>

    <span id="figure-caption-5" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Euler angles: a way to reach any orientation using a specific sequence of elemental rotations (<a href="https://en.wikipedia.org/wiki/Euler_angles">image from Wikipedia</a>).</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Elemental rotations

    First, let's look at rotations about a fixed Global coordinate system, as we did in two dimensions. The next figure illustrates elemental rotations of the local coordinate system about each axis of the fixed Global coordinate system.

    <figure aria-describedby="figure-caption-6"><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotations.png?raw=1" width=700 alt="elemental rotations"/></center></figure>

    <span id="figure-caption-6" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Elemental rotations of the $xyz$ coordinate system about each axis, $\mathbf{X}$, $\mathbf{Y}$ and $\mathbf{Z}$, of the fixed $\mathbf{XYZ}$ coordinate system. For clarity, the axis of each rotation is drawn perpendicular to the page.</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rotations about the fixed coordinate system

    The rotation matrices for the elemental rotations about each axis of the fixed $\mathbf{XYZ}$ coordinate system (rotations of the local coordinate system in relation to the Global coordinate system) are:

    About the $\mathbf{X}$ axis:

    $$
    \mathbf{R_{Gl,\,X}} =
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos\alpha & -\sin\alpha \\
    0 & \sin\alpha & \cos\alpha
    \end{bmatrix}
    $$

    About the $\mathbf{Y}$ axis:

    $$
    \mathbf{R_{Gl,\,Y}} =
    \begin{bmatrix}
    \cos\beta & 0 & \sin\beta \\
    0 & 1 & 0 \\
    -\sin\beta & 0 & \cos\beta
    \end{bmatrix}
    $$

    About the $\mathbf{Z}$ axis:

    $$
    \mathbf{R_{Gl,\,Z}} =
    \begin{bmatrix}
    \cos\gamma & -\sin\gamma & 0\\
    \sin\gamma & \cos\gamma & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    $$

    These are the rotation matrices of the two-dimensional case plus the terms for the third axes of the local and Global coordinate systems, which stay parallel. To see why those terms are ones and zeros, remember that they are direction cosines. The cosines between $\mathbf{X}x$, $\mathbf{Y}y$ and $\mathbf{Z}z$, for the elemental rotations about the $\mathbf{X}$, $\mathbf{Y}$ and $\mathbf{Z}$ axes respectively, are all 1, because those axes are parallel ($\cos 0^o$). The other cosines in the same row and column are zero, because the axis of each rotation is perpendicular to the other axes of the coordinate systems ($\cos 90^o$).

    Let's write the three matrices as Python functions of an angle in degrees; we will use them throughout the notebook:
    """)
    return


@app.function
def rotX(angle):
    """Elemental rotation matrix about the fixed X axis (angle in degrees)."""
    import numpy as np

    t = np.radians(angle)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


@app.function
def rotY(angle):
    """Elemental rotation matrix about the fixed Y axis (angle in degrees)."""
    import numpy as np

    t = np.radians(angle)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


@app.function
def rotZ(angle):
    """Elemental rotation matrix about the fixed Z axis (angle in degrees)."""
    import numpy as np

    t = np.radians(angle)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Before you run the next cell**, predict: where does the vector $[1, 0, 0]$ end up after a rotation of 90° about $\mathbf{Z}$ followed by a rotation of 90° about $\mathbf{X}$? And in the opposite order? (Each new rotation about a fixed axis multiplies the previous result from the left.)
    """)
    return


@app.cell
def _(np):
    _v = np.array([1, 0, 0])

    print("First Z, then X:", (rotX(90) @ rotZ(90) @ _v).round(4) + 0.0)
    print("First X, then Z:", (rotZ(90) @ rotX(90) @ _v).round(4) + 0.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first rotation, about $\mathbf{Z}$, takes $[1,0,0]$ to $[0,1,0]$, and the rotation about $\mathbf{X}$ then lifts it to $[0,0,1]$. In the opposite order, the rotation about $\mathbf{X}$ leaves $[1,0,0]$ where it is, because the vector lies on the axis of rotation, and the rotation about $\mathbf{Z}$ then takes it to $[0,1,0]$. Two quarter turns, two different results: matrix multiplication is not commutative, and neither are rotations.

    **Exercise 3.** And two half turns? A trampolinist makes a half somersault, $180^o$ about the medio-lateral axis $\mathbf{Z}$ of the laboratory, and a half turn, $180^o$ about the vertical axis $\mathbf{Y}$, both about the fixed axes. Does the order matter? At the end, where do the athlete's front, initially $[1, 0, 0]$, and head, initially $[0, 1, 0]$, point?
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _somersault_first = rotY(180) @ rotZ(180)  # first Z, then Y
    _turn_first = rotZ(180) @ rotY(180)  # first Y, then Z
    _same = np.allclose(_somersault_first, _turn_first)
    _half_turn_x = np.allclose(_somersault_first, rotX(180))
    _front = (_somersault_first @ np.array([1, 0, 0])).round(4) + 0.0
    _head = (_somersault_first @ np.array([0, 1, 0])).round(4) + 0.0

    mo.accordion(
        {
            "Solution to Exercise 3": mo.vstack(
                [
                    show_matrix(r"\mathbf{R_Y}(180^o)\,\mathbf{R_Z}(180^o)", _somersault_first, digits=3),
                    show_matrix(r"\mathbf{R_Z}(180^o)\,\mathbf{R_Y}(180^o)", _turn_first, digits=3),
                    mo.md(
                        f"The two orders give the same matrix (same: {_same}), and it is a half turn about "
                        + r"$\mathbf{X}$"
                        + f" (equal to a rotation of 180° about X: {_half_turn_x}). "
                        + "The order does not matter here: half turns about two perpendicular axes commute, "
                        + "and together they make a half turn about the third axis. "
                        + f"The athlete's front ends at {_front}, facing the original direction again, "
                        + f"and the head at {_head}: upside down.\n\n"
                        + "So the rule is that the order of rotations *can* matter, not that it always does. "
                        + "When in doubt, a product of two matrices is quicker than an argument."
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rotations about the local coordinate system

    The rotation matrices for the elemental rotations about each axis of the $xyz$ coordinate system (rotations of the Global coordinate system in relation to the local coordinate system) are, as in the two-dimensional case, simply the transposes of the matrices above.

    About the $x$ axis:

    $$
    \mathbf{R}_{\mathbf{lG},\,x} =
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos\alpha & \sin\alpha \\
    0 & -\sin\alpha & \cos\alpha
    \end{bmatrix}
    $$

    About the $y$ axis:

    $$
    \mathbf{R}_{\mathbf{lG},\,y} =
    \begin{bmatrix}
    \cos\beta & 0 & -\sin\beta \\
    0 & 1 & 0 \\
    \sin\beta & 0 & \cos\beta
    \end{bmatrix}
    $$

    About the $z$ axis:

    $$
    \mathbf{R}_{\mathbf{lG},\,z} =
    \begin{bmatrix}
    \cos\gamma & \sin\gamma & 0\\
    -\sin\gamma & \cos\gamma & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    $$

    This is equivalent to rotating the Global coordinate system by $-\alpha, -\beta, -\gamma$ in relation to the axes of the local coordinate system, instead of rotating the local coordinate system by $\alpha, \beta, \gamma$ in relation to the axes of the Global coordinate system; remember that $\cos(-\:\cdot)=\cos(\cdot)$ and $\sin(-\:\cdot)=-\sin(\cdot)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sequence of elemental rotations

    Consider now a sequence of elemental rotations about the $\mathbf{X}$, $\mathbf{Y}$ and $\mathbf{Z}$ axes of the fixed $\mathbf{XYZ}$ coordinate system, illustrated in the next figure.

    <figure aria-describedby="figure-caption-7"><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotations_XYZ.png?raw=1" width=700 alt="sequence of rotations about the fixed axes"/></center></figure>

    <span id="figure-caption-7" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Sequence of elemental rotations of the $xyz$ coordinate system about each axis, $\mathbf{X}$, $\mathbf{Y}$, $\mathbf{Z}$, of the fixed $\mathbf{XYZ}$ coordinate system.</span>

    This sequence of elemental rotations, each one of the local coordinate system with respect to the fixed Global coordinate system, is represented mathematically by a product of rotation matrices:

    $$
    \begin{array}{ll}
    \mathbf{R_{Gl,\;XYZ}} & = \mathbf{R_{Z}} \mathbf{R_{Y}} \mathbf{R_{X}} \\
    \\
    & = \begin{bmatrix}
    \cos\gamma & -\sin\gamma & 0\\
    \sin\gamma & \cos\gamma & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    \cos\beta & 0 & \sin\beta \\
    0 & 1 & 0 \\
    -\sin\beta & 0 & \cos\beta
    \end{bmatrix}
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos\alpha & -\sin\alpha \\
    0 & \sin\alpha & \cos\alpha
    \end{bmatrix} \\
    \\
    & =
    \begin{bmatrix}
    \cos\beta\:\cos\gamma \;&\;
    \sin\alpha\:\sin\beta\:\cos\gamma-\cos\alpha\:\sin\gamma \;&\;
    \cos\alpha\:\sin\beta\:\cos\gamma+\sin\alpha\:\sin\gamma \;\;\; \\
    \cos\beta\:\sin\gamma \;&\;
    \sin\alpha\:\sin\beta\:\sin\gamma+\cos\alpha\:\cos\gamma \;&\;
    \cos\alpha\:\sin\beta\:\sin\gamma-\sin\alpha\:\cos\gamma \;\;\;  \\
    -\sin\beta \;&\; \sin\alpha\:\cos\beta \;&\; \cos\alpha\:\cos\beta \;\;\;
    \end{bmatrix}
    \end{array}
    $$

    Note the order of the matrices: the first rotation is the rightmost one.

    We can check this matrix multiplication with [SymPy](https://www.sympy.org):
    """)
    return


@app.cell
def _(sym):
    cos, sin = sym.cos, sym.sin
    a, b, g = sym.symbols("alpha, beta, gamma")

    # Elemental rotation matrices of xyz in relation to XYZ:
    RX = sym.Matrix([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])
    RY = sym.Matrix([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]])
    RZ = sym.Matrix([[cos(g), -sin(g), 0], [sin(g), cos(g), 0], [0, 0, 1]])

    # Rotation matrix of xyz in relation to XYZ:
    RXYZ = RZ @ RY @ RX

    show_matrix(r"\mathbf{R_{Gl,\,XYZ}}", RXYZ)
    return RX, RXYZ, RY, RZ, a, b, cos, g, sin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For instance, here is the numerical rotation matrix for sequential elemental rotations by $90^o$ about $\mathbf{X}$, $\mathbf{Y}$ and $\mathbf{Z}$:
    """)
    return


@app.cell
def _(RXYZ, a, b, g, np, sym):
    _R = sym.lambdify((a, b, g), RXYZ, "numpy")(np.pi / 2, np.pi / 2, np.pi / 2)

    show_matrix(r"\mathbf{R_{Gl,\,XYZ\,}}(90^o, 90^o, 90^o)", _R, digits=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's draw the rotated coordinate system with `CCS`. The first cell loads it, together with `read_trc`, which we will need at the end. In the figure, the black arrows are the Global axes $\mathbf{X}$, $\mathbf{Y}$, $\mathbf{Z}$, and the red, green and blue arrows, which `CCS` labels i, j and k, are the local axes $x$, $y$, $z$. Since the columns of $\mathbf{R_{Gl}}$ are the local versors written in the Global coordinate system, those are the vectors we draw. Use the sliders to try other angles.
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
def _(repo_file):
    import sys

    _functions = [repo_file(f"functions/{_m}.py") for _m in ("CCS", "read_trc")]
    sys.path.insert(0, str(_functions[0].parent))  # add to pythonpath
    from CCS import CCS
    from read_trc import read_trc

    return CCS, read_trc


@app.cell
def _(mo):
    alpha_XYZ = mo.ui.slider(start=-180, stop=180, step=5, value=90, label="α about X [degrees]", show_value=True)
    beta_XYZ = mo.ui.slider(start=-180, stop=180, step=5, value=90, label="β about Y [degrees]", show_value=True)
    gamma_XYZ = mo.ui.slider(start=-180, stop=180, step=5, value=90, label="γ about Z [degrees]", show_value=True)
    mo.hstack([alpha_XYZ, beta_XYZ, gamma_XYZ], justify="start")
    return alpha_XYZ, beta_XYZ, gamma_XYZ


@app.cell
def _(CCS, RXYZ, a, alpha_XYZ, b, beta_XYZ, g, gamma_XYZ, np, sym):
    _angles = np.radians([alpha_XYZ.value, beta_XYZ.value, gamma_XYZ.value])
    _R = sym.lambdify((a, b, g), RXYZ, "numpy")(*_angles)

    # the columns of RGl are the local versors in the Global coordinate system
    _ccs = CCS(Oijk=np.zeros(3), Oxyz=np.zeros(3), ijk=_R.T, xyz=np.eye(3), vector=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the sliders at their initial 90°, the matrix and the figures agree: the rotated $x$ axis (first column of the matrix) is $[0,0,-1]$, pointing along $-\mathbf{Z}$; the rotated $y$ axis (second column) is along $\mathbf{Y}$, $[0,1,0]$; and the rotated $z$ axis (third column) is along $\mathbf{X}$, $[1,0,0]$.

    We can also calculate a sequence of elemental rotations about the $x$, $y$, $z$ axes of the rotating $xyz$ coordinate system, illustrated in the next figure.

    <figure aria-describedby="figure-caption-8"><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotations_xyz2.png?raw=1" width=700 alt="sequence of rotations about the local axes"/></center></figure>

    <span id="figure-caption-8" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Sequence of elemental rotations of a second $xyz$ local coordinate system about each axis, $x$, $y$, $z$, of the rotating $xyz$ coordinate system.</span>

    Likewise, this sequence of elemental rotations, each one of the local coordinate system with respect to the rotating local coordinate system, is represented by a product of rotation matrices, which are the inverses of the matrices for the rotations about $\mathbf{X,Y,Z}$, as we saw earlier:

    $$
    \begin{array}{ll}
    \mathbf{R}_{\mathbf{lG},\,xyz} & = \mathbf{R_{z}} \mathbf{R_{y}} \mathbf{R_{x}} \\
    \\
    & = \begin{bmatrix}
    \cos\gamma & \sin\gamma & 0\\
    -\sin\gamma & \cos\gamma & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    \cos\beta & 0 & -\sin\beta \\
    0 & 1 & 0 \\
    \sin\beta & 0 & \cos\beta
    \end{bmatrix}
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos\alpha & \sin\alpha \\
    0 & -\sin\alpha & \cos\alpha
    \end{bmatrix} \\
    \\
    & =
    \begin{bmatrix}
    \cos\beta\:\cos\gamma \;&\;
    \sin\alpha\:\sin\beta\:\cos\gamma+\cos\alpha\:\sin\gamma \;&\;
    -\cos\alpha\:\sin\beta\:\cos\gamma+\sin\alpha\:\sin\gamma \;\;\; \\
    -\cos\beta\:\sin\gamma \;&\;
    -\sin\alpha\:\sin\beta\:\sin\gamma+\cos\alpha\:\cos\gamma \;&\;
    \cos\alpha\:\sin\beta\:\sin\gamma+\sin\alpha\:\cos\gamma \;\;\;  \\
    \sin\beta \;&\; -\sin\alpha\:\cos\beta \;&\; \cos\alpha\:\cos\beta \;\;\;
    \end{bmatrix}
    \end{array}
    $$

    As before, the order of the matrices is from right to left.

    Once again, we can check this matrix multiplication with SymPy:
    """)
    return


@app.cell
def _(a, b, cos, g, sin, sym):
    # Elemental rotation matrices of xyz (local):
    Rx = sym.Matrix([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]])
    Ry = sym.Matrix([[cos(b), 0, -sin(b)], [0, 1, 0], [sin(b), 0, cos(b)]])
    Rz = sym.Matrix([[cos(g), sin(g), 0], [-sin(g), cos(g), 0], [0, 0, 1]])

    # Rotation matrix of xyz' in relation to xyz:
    Rxyz = Rz @ Ry @ Rx

    show_matrix(r"\mathbf{R}_{\mathbf{lG},\,xyz}", Rxyz)
    return Rx, Rxyz, Ry, Rz


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For instance, the numerical rotation matrix for sequential elemental rotations by $90^o$ about $x$, $y$, $z$:
    """)
    return


@app.cell
def _(Rxyz, a, b, g, np, sym):
    _R = sym.lambdify((a, b, g), Rxyz, "numpy")(np.pi / 2, np.pi / 2, np.pi / 2)

    show_matrix(r"\mathbf{R}_{\mathbf{lG},\,xyz\,}(90^o, 90^o, 90^o)", _R, digits=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's compare this matrix with the figure. Remember that it is the Global-to-local rotation matrix, $\mathbf{R}_{\mathbf{lG},\,xyz}$, in which the coordinates of the local versors are the *rows*, not the columns. With this in mind, the figure and the matrix agree: the rotated $x$ axis (first row) is along $\mathbf{Z}$, $[0,0,1]$; the rotated $y$ axis (second row) is along $-\mathbf{Y}$, $[0,-1,0]$; and the rotated $z$ axis (third row) is along $\mathbf{X}$, $[1,0,0]$.

    In fact, this example cannot tell rows from columns, because the matrix above is symmetric. Let's look at the matrix after only the first two rotations, $\mathbf{R}_{\mathbf{lG},\,xy}$:
    """)
    return


@app.cell
def _(Rx, Ry, a, b, np, sym):
    _Rxy = Ry * Rx
    _R = sym.lambdify((a, b), _Rxy, "numpy")(np.pi / 2, np.pi / 2)

    show_matrix(r"\mathbf{R}_{\mathbf{lG},\,xy\,}(90^o, 90^o)", _R, digits=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the third plot of the figure, the versor $x$ is along the Global axis $\mathbf{Y}$, so its coordinates in the Global coordinate system are $[0,1,0]$, and that is indeed the first *row* of the matrix. Check the other two rows.

    What, then, are the columns of this matrix? The coordinates of the Global versors in the local coordinate system. The first column is the coordinates of $\mathbf{X}$, which is aligned with $z$: $[0,0,1]$.

    Here, too, you can try other angles. This time the local versors are the rows of the matrix, so they are what we pass to `CCS`:
    """)
    return


@app.cell
def _(mo):
    alpha_xyz = mo.ui.slider(start=-180, stop=180, step=5, value=60, label="α about x [degrees]", show_value=True)
    beta_xyz = mo.ui.slider(start=-180, stop=180, step=5, value=45, label="β about y [degrees]", show_value=True)
    gamma_xyz = mo.ui.slider(start=-180, stop=180, step=5, value=30, label="γ about z [degrees]", show_value=True)
    mo.hstack([alpha_xyz, beta_xyz, gamma_xyz], justify="start")
    return alpha_xyz, beta_xyz, gamma_xyz


@app.cell
def _(CCS, Rxyz, a, alpha_xyz, b, beta_xyz, g, gamma_xyz, np, sym):
    _angles = np.radians([alpha_xyz.value, beta_xyz.value, gamma_xyz.value])
    _R = sym.lambdify((a, b, g), Rxyz, "numpy")(*_angles)

    # the rows of RlG are the local versors in the Global coordinate system
    _ccs = CCS(Oijk=np.zeros(3), Oxyz=np.zeros(3), ijk=_R, xyz=np.eye(3), vector=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A rotation in one coordinate system is a negative rotation in the other

    We saw for the elemental rotations that rotating the local coordinate system, $xyz$, by $\alpha, \beta, \gamma$ in relation to the axes of the Global coordinate system is equivalent to rotating the Global coordinate system, $\mathbf{XYZ}$, by $-\alpha, -\beta, -\gamma$ in relation to the axes of the local coordinate system. The same property applies to a sequence of rotations: rotations of $xyz$ in relation to $\mathbf{XYZ}$ by $\alpha, \beta, \gamma$ result in the same matrix as rotations of $\mathbf{XYZ}$ in relation to $xyz$ by $-\alpha, -\beta, -\gamma$:

    $$
    \begin{array}{ll}
    \mathbf{R_{Gl,\,XYZ\,}}(\alpha,\beta,\gamma) & = \mathbf{R_{Gl,\,Z}}(\gamma)\, \mathbf{R_{Gl,\,Y}}(\beta)\, \mathbf{R_{Gl,\,X}}(\alpha) \\
    & = \mathbf{R}_{\mathbf{lG},\,z\,}(-\gamma)\, \mathbf{R}_{\mathbf{lG},\,y\,}(-\beta)\, \mathbf{R}_{\mathbf{lG},\,x\,}(-\alpha) \\
    & = \mathbf{R}_{\mathbf{lG},\,xyz\,}(-\alpha,-\beta,-\gamma)
    \end{array}
    $$

    Compare the $\mathbf{R_{Gl,\,XYZ}}$ and $\mathbf{R}_{\mathbf{lG},\,xyz}$ matrices above to confirm it, or let SymPy do it:
    """)
    return


@app.cell
def _(RXYZ, a, b, cos, g, sin, sym):
    # Elemental rotation matrices of XYZ in relation to xyz, with all angles negated:
    _Rx_neg = sym.Matrix([[1, 0, 0], [0, cos(-a), -sin(-a)], [0, sin(-a), cos(-a)]]).T
    _Ry_neg = sym.Matrix([[cos(-b), 0, sin(-b)], [0, 1, 0], [-sin(-b), 0, cos(-b)]]).T
    _Rz_neg = sym.Matrix([[cos(-g), -sin(-g), 0], [sin(-g), cos(-g), 0], [0, 0, 1]]).T

    # Rotation matrix of XYZ in relation to xyz:
    _Rxyz_neg = _Rz_neg * _Ry_neg * _Rx_neg

    print("RGl,XYZ(alpha, beta, gamma) == RlG,xyz(-alpha, -beta, -gamma):", RXYZ == _Rxyz_neg)
    show_matrix(r"\mathbf{R}_{\mathbf{lG},\,xyz\,}(-\alpha,-\beta,-\gamma)", _Rxyz_neg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A rotation in one coordinate system is the transpose of the inverse sequence in the other

    There is another property relating the two kinds of rotation matrix. The Global-to-local rotation matrix for, say, the $xyz$ sequence is the transpose of the rotation matrix for the inverse operation (from local to Global) with the inverse sequence ($\mathbf{ZYX}$), and vice versa:

    $$
    \begin{array}{ll}
    \mathbf{R}_{\mathbf{lG},\,xyz}(\alpha,\beta,\gamma) & = \mathbf{R}_{\mathbf{lG},\,z\,} \mathbf{R}_{\mathbf{lG},\,y\,} \mathbf{R}_{\mathbf{lG},\,x} \\
    & = \mathbf{R_{Gl,\,Z\,}^{-1}} \mathbf{R_{Gl,\,Y\,}^{-1}} \mathbf{R_{Gl,\,X\,}^{-1}} \\
    & = \mathbf{R_{Gl,\,Z\,}^{T}} \mathbf{R_{Gl,\,Y\,}^{T}} \mathbf{R_{Gl,\,X\,}^{T}} \\
    & = (\mathbf{R_{Gl,\,X\,}} \mathbf{R_{Gl,\,Y\,}} \mathbf{R_{Gl,\,Z}})^\mathbf{T} \\
    & = \mathbf{R_{Gl,\,ZYX\,}^{T}}(\gamma,\beta,\alpha)
    \end{array}
    $$

    where we used two properties: the inverse of a rotation matrix (which is orthonormal) is its transpose, and the transpose of a product of matrices is the product of their transposes in reverse order. This is what the figure of the book showed: rotating about the rotating axes in one order ends where rotating about the fixed axes in the opposite order does.

    Let's verify this property with SymPy:
    """)
    return


@app.cell
def _(RX, RY, RZ, Rxyz):
    RZYX = RX * RY * RZ

    print("RlG,xyz(alpha, beta, gamma) == RGl,ZYX(gamma, beta, alpha)^T:", Rxyz == RZYX.T)
    show_matrix(r"\mathbf{R_{Gl,\,ZYX\,}^T}", RZYX.T)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Guiding questions 1.**

    1. A rotation matrix $\mathbf{R_{Gl}}$ has the local versors as its columns. Without computing anything, what are the columns of $\mathbf{R_{lG}}$?
    2. Why does the first rotation of a sequence appear as the *rightmost* matrix of the product?
    3. Look again at the two orders in the first row of the figure of the book. Using the property above, which sequence about the rotating axes gives the same final orientation as each of them?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sequence of rotations of a vector

    We saw in [Rigid-body transformations in a plane (2D)](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py) that the rotation matrix can also be used to rotate a vector (or a point, an image, a solid) by a given angle about an axis of the coordinate system. Let's investigate that in 3D with the book, rotated in different orders about the Global and local coordinate systems.

    Before any rotation, the point drawn in that figure as a round black dot on the spine of the book has coordinates $\mathbf{P}=[0, 1, 2]$ (the book has thickness 0, width 1 and height 2).

    After the first sequence of rotations in the figure (about $\mathbf{X}$ and then $\mathbf{Y}$, by $90^o$ each time), $\mathbf{P}$ has coordinates $\mathbf{P_1}=[1, -2, 0]$ in the Global coordinate system. Let's verify that:
    """)
    return


@app.cell
def _(RX, RY, a, b, np, sym):
    P = np.array([[0, 1, 2]]).T

    _RXY = RY * RX
    _R = sym.lambdify((a, b), _RXY, "numpy")(np.pi / 2, np.pi / 2)
    _P1 = np.dot(_R, P)

    print("P1 =", _P1.T.round(4) + 0.0)
    return (P,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As expected.

    Although we are rotating a vector here, without any need for the idea of a transformation between coordinate systems, we used the local-to-Global rotation matrix, $\mathbf{R_{Gl}}$. As we saw for the 2D case, this matrix performs a counterclockwise (positive) rotation. To rotate the vector clockwise (negative direction), we can use the very same matrix with a negative angle, or the inverse matrix, the Global-to-local rotation matrix $\mathbf{R_{lG}}$, with a positive angle, because $\mathbf{R_{Gl}}(\alpha) = \mathbf{R_{lG}}(-\alpha)$. Bear in mind that even in the latter case we are rotating about the Global coordinate system.

    Now let's deduce algebraically the position of the point $\mathbf{P}$ after the rotations about the *local* coordinate system, the second row of the figure. The point starts at the same position, $\mathbf{P}=[0, 1, 2]$, and the book is rotated about $x$ and then about $y$, by $90^o$ each time. Where is the point now?

    The question implicitly asks for the position in the Global coordinate system, because the local coordinate system rotates with the book and the point never moves in it. By inspection of the figure, the new position is $\mathbf{P_1}=[2, 0, 1]$. Let's naively try to obtain it by repeating the steps above with the local matrices:
    """)
    return


@app.cell
def _(P, Rx, Ry, a, b, np, sym):
    _Rxy = Ry * Rx
    _R = sym.lambdify((a, b), _Rxy, "numpy")(np.pi / 2, np.pi / 2)
    _P1 = np.dot(_R, P)

    print("P1 =", _P1.T.round(4) + 0.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The wrong answer.

    The problem is that we defined the rotation of a vector with the local-to-Global rotation matrix. One correct solution is to keep multiplying the Global-to-local rotation matrices, $\mathbf{R}_{xy} = \mathbf{R}_y\,\mathbf{R}_x$, transpose $\mathbf{R}_{xy}$ to obtain the corresponding local-to-Global rotation matrix, $\mathbf{R_{XY}}=\mathbf{R^T}_{xy}$, and then rotate the vector with this matrix:
    """)
    return


@app.cell
def _(P, Rx, Ry, a, b, np, sym):
    _Rxy = Ry * Rx
    _RXY = _Rxy.T
    _R = sym.lambdify((a, b), _RXY, "numpy")(np.pi / 2, np.pi / 2)
    _P1 = np.dot(_R, P)

    print("P1 =", _P1.T.round(4) + 0.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The correct answer.

    Another solution is to understand that, with the Global-to-local rotation matrix, counterclockwise rotations (as performed with the book in the figure) are negative, not positive, and that the order of the matrix multiplication is inverted: it should be $\mathbf{R\_}_{xyz} = \mathbf{R}_x\,\mathbf{R}_y\,\mathbf{R}_z$ (the underscore is a reminder that this is not the convention adopted here).
    """)
    return


@app.cell
def _(P, Rx, Ry, a, b, np, sym):
    _R_xy = Rx * Ry
    _R = sym.lambdify((a, b), _R_xy, "numpy")(-np.pi / 2, -np.pi / 2)
    _P1 = np.dot(_R, P)

    print("P1 =", _P1.T.round(4) + 0.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The correct answer again.

    You will find texts on 3D rotations that adopt this latter convention as the standard: they introduce the Global-to-local rotation matrix and describe a sequence of rotations as a matrix product in the direct order, $\mathbf{R\_}_{xyz} = \mathbf{R}_x\,\mathbf{R}_y\,\mathbf{R}_z$, the opposite of what we do here. It is a matter of convention, nothing more, but you have to know which convention a text, a paper or a piece of software is using.

    **Challenge 1.** Deduce the position of the point $\mathbf{P}$ after the inverse order of rotations, first about the Global coordinate system and then about the local one, by inspection of the figure. Then check your answers with code. Which of the four results coincide, and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Codman's paradox, computed

    We now have everything needed to explain the arm movements at the start of this notebook. Let's use a Global coordinate system with $\mathbf{X}$ pointing forward, $\mathbf{Y}$ upwards and $\mathbf{Z}$ to the right, the convention recommended for motion analysis by the International Society of Biomechanics (Wu & Cavanagh, 1995). We follow two vectors attached to the right arm: its long axis, pointing from the shoulder down to the hand, $[0, -1, 0]$; and the direction the palm faces, initially towards the thigh, $[0, 0, -1]$.

    The three movements are rotations about fixed axes:

    1. Flexion: the arm swings forward and up, a rotation of $+90^o$ about $\mathbf{Z}$.
    2. Horizontal abduction: the arm swings from the front to the side, a rotation of $-90^o$ about $\mathbf{Y}$.
    3. Adduction: the arm comes down from the side, a rotation of $+90^o$ about $\mathbf{X}$.

    **Before you run the next cell**, check those signs with the right-hand rule, and predict where the palm faces at the end.
    """)
    return


@app.cell
def _(np):
    _arm = np.array([0, -1, 0])  # long axis of the arm, from the shoulder to the hand
    _palm = np.array([0, 0, -1])  # the right palm, facing the thigh
    _steps = [
        ("flexion, +90 about Z", rotZ(90)),
        ("horizontal abduction, -90 about Y", rotY(-90)),
        ("adduction, +90 about X", rotX(90)),
    ]

    _R = np.eye(3)
    for _name, _Ri in _steps:
        _R = _Ri @ _R  # each new rotation about a fixed axis multiplies from the left
        print(f"after {_name:34s} arm: {(_R @ _arm).round(4) + 0.0}  palm: {(_R @ _palm).round(4) + 0.0}")

    print("\nRotation matrix of the whole sequence:\n", _R.round(4) + 0.0)
    print("\nIs it a single rotation of -90 degrees about Y?", np.allclose(_R, rotY(-90)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The arm comes back to $[0,-1,0]$, hanging at the side, but the palm, which started facing the thigh, now faces forward, $[1,0,0]$. And the product of the three matrices is a single rotation of $-90^o$ about the vertical axis, which at the end is also the long axis of the arm: the arm has rotated externally by 90°, without a single movement about its long axis.

    There is no contradiction. Each movement was a rotation about an axis perpendicular to the arm *at that moment*, but those axes were fixed in space while the arm moved, and three rotations compose into a rotation about an axis that none of them used. In a plane that cannot happen: rotations about the one available axis simply add. In three dimensions, the result of a sequence of rotations is the product of the matrices, not the sum of the angles.

    For the shoulder, this is not a curiosity. It means that the axial rotation of the humerus depends on the path the arm took to reach a position, and that a description of the arm's orientation by three angles depends on the order in which the angles are defined.

    **Guiding questions 2.**

    1. Problem 6 asks you to compute the order you tried in Guiding question 0.2. Before computing, predict: will the palm end up facing forward again?
    2. Why did the second movement change the direction of the palm, while the first one did not?
    3. A colleague reports the arm's orientation as the sum of the three angles about $\mathbf{X}$, $\mathbf{Y}$ and $\mathbf{Z}$. What would that sum be for Codman's movement, and what is wrong with it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One rotation is enough

    Codman's three rotations turned out to be a single rotation about one axis. That is not a coincidence of the example. Euler's rotation theorem says that *any* rotation matrix is a single rotation, by some angle $\theta$, about some axis $\hat{\mathbf{n}}$. Both can be read off the matrix. The trace (the sum of the diagonal) gives the angle,

    $$
    \mathrm{tr}(\mathbf{R}) = 1 + 2\cos\theta
    $$

    and the antisymmetric part of the matrix gives the axis,

    $$
    \begin{bmatrix} R_{21} - R_{12} \\ R_{02} - R_{20} \\ R_{10} - R_{01} \end{bmatrix} = 2\sin\theta\;\hat{\mathbf{n}}
    $$

    (rows and columns numbered from 0, as in Python). The formula for the axis fails in two cases: with no rotation, where any axis will do, and for a half turn, where $\sin\theta = 0$; for the half turn, $(\mathbf{R} + \mathbf{I})/2 = \hat{\mathbf{n}}\hat{\mathbf{n}}^T$ gives the axis instead. Here is a function that handles all three cases:
    """)
    return


@app.function
def axis_angle(R):
    """Axis (versor) and angle (degrees, from 0 to 180) of the single rotation equivalent to R.

    R is a local-to-Global rotation matrix (or a rotation of vectors). With no
    rotation the axis is undefined and returned as NaN.
    """
    import numpy as np

    R = np.asarray(R, dtype=float)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.linalg.norm(w) > 1e-9:  # 0 < angle < 180
        return w / np.linalg.norm(w), angle
    if angle < 90:  # no rotation
        return np.full(3, np.nan), angle
    B = (R + np.eye(3)) / 2  # half turn: B = n n^T
    n = B[:, np.argmax(np.diag(B))]
    return n / np.linalg.norm(n), angle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Before you run the next cell**, predict: what single rotation is equivalent to $90^o$ about $\mathbf{X}$, then $90^o$ about $\mathbf{Y}$, then $90^o$ about $\mathbf{Z}$ (you saw its matrix earlier, with the sliders at their initial values)? And how large is the single rotation equivalent to $30^o$, $45^o$ and $60^o$ about the same axes?
    """)
    return


@app.cell
def _():
    _cases = [
        ("90 about X, then 90 about Y, then 90 about Z", rotZ(90) @ rotY(90) @ rotX(90)),
        ("Codman's movement", rotX(90) @ rotY(-90) @ rotZ(90)),
        ("30 about X, then 45 about Y, then 60 about Z", rotZ(60) @ rotY(45) @ rotX(30)),
    ]
    for _name, _R in _cases:
        _axis, _angle = axis_angle(_R)
        print(f"{_name:46s} -> {_angle:5.1f} degrees about {_axis.round(3) + 0.0}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three quarter turns, about three different axes, add up to a single quarter turn about $\mathbf{Y}$. Codman's movement is a quarter turn about $-\mathbf{Y}$, which is the rotation of $-90^o$ about $\mathbf{Y}$ found before: the function always reports an angle between $0^o$ and $180^o$, and lets the direction of the axis carry the sign. And rotations of $30^o$, $45^o$ and $60^o$ add up to a single rotation of $69.4^o$, not $135^o$.

    Unlike the three Euler angles, this angle does not depend on any convention: whatever sequence you choose to describe the orientation, the single equivalent rotation is the same. It is the rotational part of the helical (or screw) axis used to describe joint motion, and a convenient, sequence-free answer to the question "how much did this segment rotate?". Problem 10 asks you to go the other way, from an axis and an angle to the rotation matrix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How much does the order matter?

    Two quarter turns about $\mathbf{Z}$ and $\mathbf{X}$ ended in completely different places depending on their order. Are joint angles of a few degrees just as sensitive?

    **Before you run the next cell**, predict: one body is turned by $10^o$ about $\mathbf{Z}$ and then by $10^o$ about $\mathbf{X}$, and another body by the same two rotations in the opposite order. By how much do their final orientations differ? And for $45^o$?

    The difference between two orientations is itself a rotation, $\mathbf{R_1^T}\mathbf{R_2}$, and `axis_angle` measures it:
    """)
    return


@app.cell
def _():
    print(" angle    Z then X vs X then Z: differ by    axis of the difference")
    for _theta in [1, 5, 10, 20, 45, 90]:
        _z_then_x = rotX(_theta) @ rotZ(_theta)
        _x_then_z = rotZ(_theta) @ rotX(_theta)
        _axis, _diff = axis_angle(_z_then_x.T @ _x_then_z)
        print(f"{_theta:5d}    {_diff:10.3f} degrees              {_axis.round(2) + 0.0}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For rotations of $1^o$, the two orders end within $0.02^o$ of each other; for $10^o$, within $1.7^o$; for $20^o$, $6.9^o$; for $45^o$, $34^o$; and the two quarter turns end $120^o$ apart. The difference grows roughly with the square of the angle: ten times the angle, a hundred times the difference. For small angles it is a rotation about $\mathbf{Y}$, the axis perpendicular to the two that were used; Problem 11 asks you to show why.

    Small rotations nearly commute. That is why, for a joint that moves little outside its main plane, different sequences of Euler angles give nearly the same angles, and why the choice of sequence matters most for large three-dimensional movements, such as those of the shoulder or of the hip in sport.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The 12 different sequences of Euler angles

    The Euler angles are defined in terms of rotations about a rotating local coordinate system. As we saw for the sequence of rotations about $x, y, z$, the axes of the local coordinate system are not fixed in space: after the first elemental rotation, the other two axes have moved.

    Other sequences of rotations can be formed without mixing axes of the two coordinate systems (Global and local) as rotation axes. There are 12 valid sequences of three elemental rotations that can describe the rotation of a coordinate system with respect to another:

    $$
    \begin{array}{ll}
    xyz \quad xzy \quad yzx \quad yxz \quad zxy \quad zyx  \\
    xyx \quad xzx \quad yzy \quad yxy \quad zxz \quad zyz
    \end{array}
    $$

    The first six sequences (first row) are about three different axes and are usually referred to as Cardan or Tait–Bryan angles. The other six (second row) have the first and third rotations about the same axis, but keep in mind that the axis of the third rotation is no longer where it was, because it changed orientation with the second rotation. The sequences with a repeated axis are known as proper or classic Euler angles.

    Which order to use is a matter of convention, but because the order affects the results, it is fundamental to follow a convention and to report it. In engineering mechanics (including biomechanics), the $xyz$ order is more common; in physics, the $zxz$ order is more common (but the letters chosen for the axes are arbitrary; what matters is the directions they represent). In biomechanics, the order of the Cardan angles is most often based on the angle of most interest or of most reliable measurement. Accordingly, the axis of flexion–extension is typically the first axis, the axis of abduction–adduction the second, and the axis of internal–external rotation the last one; for the knee, this is the order of the joint coordinate system of Grood and Suntay (1983). The $zyx$ order is commonly used to describe the orientation of a ship or an aircraft, and the rotations are known as the nautical angles: yaw, pitch and roll, respectively (see next figure).

    <figure aria-describedby="figure-caption-9"><center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Yaw_Axis.svg/330px-Yaw_Axis.svg.png" width=320 alt="principal axes of an aircraft"/></center></figure>

    <span id="figure-caption-9" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. The principal axes of an aircraft and the names of the rotations about them (<a href="https://en.wikipedia.org/wiki/Euler_angles">image from Wikipedia</a>).</span>

    If, instead of rotations about the rotating local coordinate system, we perform rotations about the fixed Global coordinate system, we have another 12 sequences of three elemental rotations, called simply rotation angles. In total there are 24 possible sequences, but they are not independent: with the 12 sequences of Euler angles in the local coordinate system we can obtain the other 12 in the Global coordinate system.

    The function below determines the rotation matrix in algebraic form for any of the 24 sequences (sequences of only one or two axes can also be entered), and in numerical form if a list of angles is given. It is a compact version of the function `euler_rotmat.py`, from the `functions` folder of this repository.
    """)
    return


@app.function
def euler_rotmat(order="xyz", frame="local", angles=None, unit="deg"):
    """Rotation matrix for a sequence of up to three elemental rotations.

    `order` is a string of one to three letters among x, y and z; the first
    letter is the first rotation. With frame="local", the rotations are about
    the axes of the rotating coordinate system (Euler angles) and the matrix is
    Global-to-local; with frame="global", they are about the fixed axes and the
    matrix is local-to-Global. The angles of the first, second and third
    rotations are the symbols alpha, beta and gamma.

    Returns the algebraic matrix and, if `angles` is given, also the matrix with
    those values substituted (use None for an angle that stays symbolic).
    """
    import sympy as sym

    cos, sin = sym.cos, sym.sin
    symbols = sym.symbols("alpha, beta, gamma")

    def elemental(axis, q):
        R = {
            "x": sym.Matrix([[1, 0, 0], [0, cos(q), sin(q)], [0, -sin(q), cos(q)]]),
            "y": sym.Matrix([[cos(q), 0, -sin(q)], [0, 1, 0], [sin(q), 0, cos(q)]]),
            "z": sym.Matrix([[cos(q), sin(q), 0], [-sin(q), cos(q), 0], [0, 0, 1]]),
        }[axis]
        return R.T if frame.lower() == "global" else R

    R = sym.eye(3)
    for axis, q in zip(order.lower(), symbols):
        R = elemental(axis, q) * R  # the matrix product goes from right to left
    if angles is None:
        return R

    values = {}
    for q, angle in zip(symbols, angles):
        if angle is not None:
            values[q] = sym.rad(angle) if unit[:3].lower() == "deg" else angle
    return R, sym.simplify(R.subs(values))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For instance, here are the rotation matrix in algebraic form for the $zxz$ order of Euler angles in the local coordinate system, and the corresponding numerical matrix after three elemental rotations by $90^o$ each:
    """)
    return


@app.cell
def _(mo):
    _Ra, _Rn = euler_rotmat(order="zxz", frame="local", angles=[90, 90, 90])

    mo.vstack(
        [
            show_matrix(r"\mathbf{R}_{local}(z:\alpha, x:\beta, z:\gamma)", _Ra),
            show_matrix(r"\mathbf{R}_{local}(z:90^o, x:90^o, z:90^o)", _Rn),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Line of nodes

    The second axis of rotation in the rotating coordinate system is also referred to as the nodal axis, or line of nodes. It coincides with the intersection of two perpendicular planes, one from each coordinate system, Global (fixed) and local (rotating). The figure below shows an example of rotations and the nodal axis for the $xyz$ sequence of Cardan angles.

    <figure aria-describedby="figure-caption-10"><center><img src="https://github.com/BMClab/BMC/blob/master/images/Node.png?raw=1" width=700 alt="line of nodes"/></center></figure>

    <span id="figure-caption-10" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. First row: example of rotations for the $xyz$ sequence of Cardan angles. The Global (fixed) $XYZ$ coordinate system is shown in green, the local (rotating) $xyz$ coordinate system in blue. The nodal axis (<b>N</b>, in red) is defined by the intersection of the $YZ$ and $xy$ planes, and all rotations can be described in relation to this nodal axis or to an axis perpendicular to it. Second row: starting from no rotation, the local coordinate system is rotated by $\alpha$ about the $x$ axis, then by $\beta$ about the rotated $y$ axis, and finally by $\gamma$ about the twice-rotated $z$ axis. The line of nodes coincides with the $y$ axis of the second rotation.</span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Determination of the Euler angles

    Once a convention is adopted, the corresponding three Euler angles can be found from the rotation matrix. For example, for the $\mathbf{R}_{xyz}$ rotation matrix:
    """)
    return


@app.cell
def _():
    show_matrix(r"\mathbf{R}_{local}(x:\alpha, y:\beta, z:\gamma)", euler_rotmat(order="xyz", frame="local"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The corresponding Cardan angles for the $xyz$ sequence are:

    $$
    \begin{array}{ll}
    \alpha = \arctan\left(\dfrac{\sin(\alpha)}{\cos(\alpha)}\right) = \arctan\left(\dfrac{-\mathbf{R}_{21}}{\;\;\;\mathbf{R}_{22}}\right) \\
    \\
    \beta = \arctan\left(\dfrac{\sin(\beta)}{\cos(\beta)}\right) = \arctan\left(\dfrac{\mathbf{R}_{20}}{\sqrt{\mathbf{R}_{00}^2+\mathbf{R}_{10}^2}}\right) \\
    \\
    \gamma = \arctan\left(\dfrac{\sin(\gamma)}{\cos(\gamma)}\right) = \arctan\left(\dfrac{-\mathbf{R}_{10}}{\;\;\;\mathbf{R}_{00}}\right)
    \end{array}
    $$

    (with the rows and columns numbered from 0, as in Python). We use the function `arctan2` rather than `arcsin`, `arccos` or `arctan`, because those cannot, for example, distinguish $45^o$ from $135^o$, and for better numerical accuracy; see [Angular kinematics in a plane (2D)](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py) for more on these issues.

    Here is a Python function that computes the Euler angles of the rotation from the Global to the local coordinate system for the $xyz$ Cardan sequence:
    """)
    return


@app.function
def euler_angles_from_rot_xyz(rot_matrix, unit="deg"):
    """Compute Euler angles from rotation matrix in the xyz sequence."""
    import numpy as np

    R = np.asarray(rot_matrix, dtype=np.float64)[:3, :3]
    angles = np.zeros(3)

    angles[0] = np.arctan2(-R[2, 1], R[2, 2])
    angles[1] = np.arctan2(R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    angles[2] = np.arctan2(-R[1, 0], R[0, 0])

    if unit[:3].lower() == "deg":  # convert from rad to degree
        angles = np.rad2deg(angles)

    return angles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For instance, consider sequential rotations of $45^o$ about $x$, $y$, $z$. The resulting rotation matrix is:
    """)
    return


@app.cell
def _():
    _Ra, Rn45 = euler_rotmat(order="xyz", frame="local", angles=[45, 45, 45])

    show_matrix(r"\mathbf{R}_{local}(x:45^o, y:45^o, z:45^o)", Rn45, digits=4)
    return (Rn45,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check by calculating the Cardan angles back from this rotation matrix with `euler_angles_from_rot_xyz()`:
    """)
    return


@app.cell
def _(Rn45):
    euler_angles_from_rot_xyz(Rn45, unit="deg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could implement a function to calculate the Euler angles for each of the 12 sequences (24, counting the rotations from and to both coordinate systems), but this is tedious. There is a smarter solution using the concept of [quaternion](https://en.wikipedia.org/wiki/Quaternion), but we will not see it here.

    **Exercise 4.** The round trip gave back the angles we put in, but the angles that produce a given rotation matrix are not unique. The matrix $\mathbf{R}_{local}(x:10^o, y:-30^o, z:50^o)$ gives back exactly $(10^o, -30^o, 50^o)$. Find another set of three $xyz$ angles that produces the same matrix, and check it with `euler_rotmat`. (Hint: start by adding $180^o$ to the first and the third angles, and see what the second one must do to compensate.)
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _, _R = euler_rotmat(order="xyz", frame="local", angles=[10, -30, 50])
    _, _R_alt = euler_rotmat(order="xyz", frame="local", angles=[190, 210, 230])
    _same = np.allclose(np.array(_R, dtype=float), np.array(_R_alt, dtype=float))
    _back = euler_angles_from_rot_xyz(np.array(_R_alt, dtype=float)).round(4) + 0.0

    mo.accordion(
        {
            "Solution to Exercise 4": mo.md(
                rf"""
    Adding $180^o$ to $\alpha$ changes the sign of both $\sin\alpha$ and $\cos\alpha$, and the same for $\gamma$. To compensate, the second angle must keep its sine and change the sign of its cosine, which $180^o - \beta$ does. Every entry of the matrix is then a product in which either none or exactly two of the factors change sign, so the matrix does not change. The second set of angles is

    $$
    (\alpha + 180^o,\; 180^o - \beta,\; \gamma + 180^o) = (190^o,\; 210^o,\; 230^o)
    $$

    or, written between $-180^o$ and $180^o$, $(-170^o, -150^o, -130^o)$.

    Same matrix: {_same}. And `euler_angles_from_rot_xyz` applied to this matrix returns {_back}, the first solution again.

    The function always picks the solution with $\cos\beta \geq 0$, that is, with $\beta$ between $-90^o$ and $90^o$, because it computes the cosine of $\beta$ as a positive square root. That choice is a convention too, and it is the reason why the second angle of a Cardan sequence is reported between $-90^o$ and $90^o$.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Apart from that choice between two solutions, the round trip worked here. It does not always work, and the reason is a problem of Euler angles known as gimbal lock.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gimbal lock

    [Gimbal lock](https://en.wikipedia.org/wiki/Gimbal_lock) is the loss of one degree of freedom that occurs when an axis of rotation is placed parallel to a previous axis of rotation, so that, in a given convention of Euler angles, two of the three rotations are about the same direction. This "locks" the system into rotations in a degenerate two-dimensional space. The system is not really locked, in the sense that it cannot move or cannot reach the other degree of freedom, but it needs an extra rotation for that.

    For instance, look at the $zxz$ sequence of rotations by the angles $\alpha, \beta, \gamma$:

    $$
    \begin{array}{ll}
    \mathbf{R}_{zxz} & = \mathbf{R_{z}} \mathbf{R_{x}} \mathbf{R_{z}} \\
    \\
    & =
    \begin{bmatrix}
    \cos\gamma & \sin\gamma & 0\\
    -\sin\gamma & \cos\gamma & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos\beta & \sin\beta \\
    0 & -\sin\beta & \cos\beta
    \end{bmatrix}
    \begin{bmatrix}
    \cos\alpha & \sin\alpha & 0\\
    -\sin\alpha & \cos\alpha & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \end{array}
    $$

    which results in:
    """)
    return


@app.cell
def _(a, b, cos, g, sin, sym):
    # Elemental rotation matrices of xyz (local) for the zxz sequence:
    Rz1 = sym.Matrix([[cos(a), sin(a), 0], [-sin(a), cos(a), 0], [0, 0, 1]])
    Rx2 = sym.Matrix([[1, 0, 0], [0, cos(b), sin(b)], [0, -sin(b), cos(b)]])
    Rz3 = sym.Matrix([[cos(g), sin(g), 0], [-sin(g), cos(g), 0], [0, 0, 1]])

    # Rotation matrix for the zxz sequence:
    Rzxz = Rz3 * Rx2 * Rz1

    show_matrix(r"\mathbf{R}_{zxz}", Rzxz)
    return Rz1, Rz3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's examine what happens to this rotation matrix when the rotation about the second axis ($x$) by $\beta$ is zero:

    $$
    \mathbf{R}_{zxz}(\alpha, \beta=0, \gamma) =
    \begin{bmatrix}
    \cos\gamma & \sin\gamma & 0\\
    -\sin\gamma & \cos\gamma & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    \cos\alpha & \sin\alpha & 0\\
    -\sin\alpha & \cos\alpha & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    $$

    The second matrix is the identity matrix and has no effect on the product, which is:
    """)
    return


@app.cell
def _(Rz1, Rz3, mo, sym):
    _Rzxz0 = Rz3 * Rz1

    mo.vstack(
        [
            show_matrix(r"\mathbf{R}_{zxz}(\alpha, \beta=0, \gamma)", _Rzxz0),
            show_matrix(r"\mathbf{R}_{zxz}(\alpha, \beta=0, \gamma)", sym.simplify(_Rzxz0)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    which, after simplification (above), is a single rotation about the $z$ axis by the sum $\alpha+\gamma$. Whatever the values of $\alpha$ and $\gamma$, only their sum matters: of the three degrees of freedom, one was lost (and another was fixed by setting $\beta=0$). For movement analysis, this means that one angle is undetermined, because all we can recover from the rotation matrix is the sum of the two. We can set the unknown angle to zero, but that is arbitrary.

    We have already met another example of gimbal lock, in the $xyz$ sequence with rotations by $90^o$. Look again at the figure of those rotations: the first and third rotations were about the same direction, because the second rotation was by $90^o$. Let's do the matrix multiplication with only the second angle replaced by $90^o$:
    """)
    return


@app.cell
def _():
    _Ra, _Rn = euler_rotmat(order="xyz", frame="local", angles=[None, 90.0, None])

    show_matrix(r"\mathbf{R}_{local}(x:\alpha, y:90^o, z:\gamma)", _Rn)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once again, one degree of freedom was lost: only $\alpha+\gamma$ appears, and we cannot determine the three angles uniquely from the rotation matrix of this sequence.

    ### How close is too close?

    A rotation of exactly $90^o$ is rare in measured data. Is there anything to worry about at, say, $89^o$?

    Let's take a rotation in the $xyz$ sequence with $\alpha = 30^o$, $\gamma = 20^o$ and different values of $\beta$, spoil each matrix with a tiny error, a rotation of about 0.14° made of 0.1° about $\mathbf{X}$ and 0.1° about $\mathbf{Y}$, much smaller than the errors of a real measurement, and calculate the angles back.

    **Before you run the next cell**, predict: how large will the errors in the three angles be when $\beta = 45^o$? And when $\beta = 89.9^o$?
    """)
    return


@app.cell
def _(Rxyz, a, b, g, np, sym):
    _R_of = sym.lambdify((a, b, g), Rxyz, "numpy")
    _error = rotX(0.1) @ rotY(0.1)  # a small error in the orientation

    print(" beta     angles from the slightly wrong matrix    errors           alpha + gamma")
    for _beta in [0, 45, 80, 89, 89.9, 89.99]:
        _true = np.array([30, _beta, 20])
        _back = euler_angles_from_rot_xyz(_R_of(*np.radians(_true)) @ _error)
        print(f"{_beta:6.2f}   {_back.round(2)}   {(_back - _true).round(2)}   {_back[0] + _back[2]:6.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Far from the lock, an error of 0.14° in the orientation becomes errors of about 0.1° in the angles, as it should. As $\beta$ approaches $90^o$, the errors in $\alpha$ and $\gamma$ grow without bound: about 2.7° at $\beta = 89^o$, 15° at $89.9^o$, 27° at $89.99^o$. Their sum stays close to $50^o$ throughout. Near the lock, the matrix determines only the sum, and the split between the two angles is decided by the error. So gimbal lock is not a single bad orientation to avoid, but a region around it where two of the angles become meaningless.

    Possible solutions to avoid gimbal lock are: choose a different sequence; avoid the orientations that put the system in gimbal lock (in the examples above, $\beta=90^o$); or add a fourth parameter to the description of the orientation, as quaternions do.

    But if we have a physical system that we describe or control with exactly three Euler angles in a fixed sequence, and we cannot prevent it from reaching certain angles, then we might have to say "Houston, we have a problem". A famous situation where this happened was the Apollo 13 mission. This is an actual conversation between the crew and mission control during that mission (Corke, 2011):

    >`Mission clock: 02 08 12 47`<br>
    **Flight**: *Go, Guidance.*<br>
    **Guido**: *He’s getting close to gimbal lock there.*<br>
    **Flight**: *Roger. CapCom, recommend he bring up C3, C4, B3, B4, C1 and C2 thrusters, and advise he’s getting close to gimbal lock.*<br>
    **CapCom**: *Roger.*

    *Of note, gimbal lock did not cause the Apollo 13 accident; the problem was an oxygen tank explosion.*

    **Guiding questions 3.**

    1. For which value of $\beta$ does the $zxz$ sequence lock? And the $xyz$ sequence? Why are they different?
    2. In the table above, $\beta$ itself is recovered with an error of about 0.1° all the way to the lock. Why is the second angle unaffected?
    3. You measure the shoulder of a person raising the arm sideways to above the head. Why might the sequence you chose for the knee be a poor choice here?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Determination of the rotation matrix

    Let's revisit the example where we determined the rotation matrix by building a basis from three noncollinear points, $\mathbf{m_1} = [1,0,0]$, $\mathbf{m_2} = [0,1,0]$, $\mathbf{m_3} = [0,0,1]$, and now find its Euler angles. The versors of the basis, as rows, form the rotation matrix from the Global to the local coordinate system:
    """)
    return


@app.cell
def _(np, v1, v2, v3):
    _RlG = np.array([v1, v2, v3])

    print("Rotation matrix from Global to local coordinate system:\n", _RlG)
    print("\nAngles of rotation for the xyz sequence:", euler_angles_from_rot_xyz(_RlG))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These angles do not mean anything here, because they are the angles of the axes of the arbitrary basis we computed. In biomechanics, to give the orientation of a coordinate system an anatomical interpretation, we define the versors of the basis along anatomical axes (for the upper arm, for example, one versor along the long axis of the humerus), as seen in [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Determination of the rotation matrix between two local coordinate systems

    As in the [two-dimensional case](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py), the rotation matrix between two local coordinate systems can be computed from the rotation matrices of both coordinate systems:

    $$
    R_{l_1l_2} = R_{Gl_1}^TR_{Gl_2}
    $$

    The Euler angles between the two coordinate systems can then be found with the `arctan2` function, as shown previously. This is how a joint angle is calculated: the knee angle is the orientation of the shank's coordinate system relative to the thigh's.

    For example, suppose the thigh is rotated from the laboratory by 30° about $\mathbf{Z}$ and the shank by $-20^o$ about $\mathbf{Z}$. **Before you run the next cell**, predict the rotation of the shank relative to the thigh.
    """)
    return


@app.cell
def _(np):
    _RGl_thigh = rotZ(30)
    _RGl_shank = rotZ(-20)
    _Rl1l2 = _RGl_thigh.T @ _RGl_shank

    print("Rotation matrix of the shank relative to the thigh:\n", _Rl1l2.round(4) + 0.0)
    print("Equal to a rotation of -50 degrees about Z?", np.allclose(_Rl1l2, rotZ(-50)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A rotation of $-50^o$ about $\mathbf{Z}$: the difference between the two angles. That is only because both rotations are about the same axis, which makes this a two-dimensional problem in disguise. As soon as the two segments are rotated about different axes, the relative angles are no longer differences of angles; Problem 8 asks you to show it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Translation and rotation

    Consider the case where the local coordinate system is translated and rotated in relation to the Global coordinate system, as illustrated in the next figure.

    <figure aria-describedby="figure-caption-11"><center><img src="https://github.com/BMClab/BMC/blob/master/images/transrot3D.png?raw=1" width=450 alt="translation and rotation 3D"/></center></figure>

    <span id="figure-caption-11" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. A point in three-dimensional space represented in two coordinate systems, with one coordinate system translated and rotated.</span>

    The position of the point $\mathbf{P}$, originally described in the local coordinate system, is described in the Global coordinate system in vector form by:

    $$
    \mathbf{P_G} = \mathbf{L_G} + \mathbf{R_{Gl}}\mathbf{P_l}
    $$

    This means that we first *disrotate* the local coordinate system and then correct for the translation between the two coordinate systems. We cannot invert this order: the position of the point is expressed in the local coordinate system, and we cannot add it to a vector expressed in the Global coordinate system before converting both vectors to the same coordinate system.

    To find the position of a point in the local coordinate system, given its position in the Global coordinate system, the rotation matrix and the translation vector, we invert the expression above:

    $$
    \begin{array}{ll}
    \mathbf{P_G} = \mathbf{L_G} + \mathbf{R_{Gl}}\mathbf{P_l} \implies \\
    \\
    \mathbf{R_{Gl}^{-1}}\cdot\mathbf{P_G} = \mathbf{R_{Gl}^{-1}}\left(\mathbf{L_G} + \mathbf{R_{Gl}}\mathbf{P_l}\right) \implies \\
    \\
    \mathbf{R_{Gl}^{-1}}\mathbf{P_G} = \mathbf{R_{Gl}^{-1}}\mathbf{L_G} + \mathbf{R_{Gl}^{-1}}\mathbf{R_{Gl}}\mathbf{P_l} \implies \\
    \\
    \mathbf{P_l} = \mathbf{R_{Gl}^{-1}}\left(\mathbf{P_G}-\mathbf{L_G}\right) =  \mathbf{R_{Gl}^T}\left(\mathbf{P_G}-\mathbf{L_G}\right) \;\;\;\;\; \text{or} \;\;\;\;\; \mathbf{P_l} = \mathbf{R_{lG}}\left(\mathbf{P_G}-\mathbf{L_G}\right)
    \end{array}
    $$

    To perform the inverse operation, from the Global to the local coordinate system, we first translate and then rotate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transformation matrix

    The translation and rotation operations can be combined in a single matrix, called the transformation matrix:

    $$
    \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y} \\
    \mathbf{P_Z} \\
    1
    \end{bmatrix} =
    \begin{bmatrix}
    . & . & . & \mathbf{L_{X}} \\
    . & \mathbf{R_{Gl}} & .  & \mathbf{L_{Y}} \\
    . & . & .  & \mathbf{L_{Z}} \\
    0 & 0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y \\
    \mathbf{P}_z \\
    1
    \end{bmatrix}
    $$

    or simply

    $$
    \mathbf{P_G} = \mathbf{T_{Gl}}\mathbf{P_l}
    $$

    Remember that, in general, the transformation matrix is not orthonormal: its inverse is not equal to its transpose.

    The inverse operation, expressing a position given in the Global coordinate system in the local coordinate system, is

    $$
    \mathbf{P_l} = \mathbf{T_{Gl}^{-1}}\mathbf{P_G}
    $$

    $$
    \begin{bmatrix}
    \mathbf{P_x} \\
    \mathbf{P_y} \\
    \mathbf{P_z} \\
    1
    \end{bmatrix} =
    \begin{bmatrix}
    \cdot & \cdot & \cdot & \cdot \\
    \cdot & \mathbf{R^{-1}_{Gl}} & \cdot  & -\mathbf{R^{-1}_{Gl}}\:\mathbf{L_G} \\
    \cdot & \cdot & \cdot  & \cdot \\
    0 & 0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y} \\
    \mathbf{P_Z} \\
    1
    \end{bmatrix}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example with actual motion analysis data

    *The data for this example are taken from page 183 of David Winter's book.*

    Consider the following positions of markers placed on a leg (described in the laboratory coordinate system with coordinates $x, y, z$ in cm, the $x$ axis pointing forward and the $y$ axis pointing upward): lateral malleolus (**lm** = [2.92, 10.10, 18.85]), medial malleolus (**mm** = [2.71, 10.22, 26.52]), fibular head (**fh** = [5.05, 41.90, 15.41]) and medial condyle (**mc** = [8.29, 41.88, 26.52]). Define the ankle joint center as the centroid of the **lm** and **mm** markers, and the knee joint center as the centroid of the **fh** and **mc** markers. An anatomical coordinate system for the leg can be defined as follows: the quasi-vertical axis ($y$) passes through the ankle and knee joint centers; a temporary medio-lateral axis ($z$) passes through the two markers on the malleoli; the anterior-posterior axis is the cross product of the two former axes; and the origin is at the ankle joint center.

    a) Calculate the anatomical coordinate system for the leg as described above.<br>
    b) Calculate the rotation matrix and the translation vector for the transformation from the anatomical to the laboratory coordinate system.<br>
    c) Calculate the position of each marker and of each joint center in the anatomical coordinate system.<br>
    d) Calculate the Cardan angles using the $zxy$ sequence for the orientation of the leg with respect to the laboratory (but remember that the letters chosen to refer to axes are arbitrary; what matters is the directions they represent).

    We solve parts a) to c) here. Part d) is Problem 3.
    """)
    return


@app.cell
def _(np):
    # calculation of the joint centers
    mm = np.array([2.71, 10.22, 26.52])
    lm = np.array([2.92, 10.10, 18.85])
    fh = np.array([5.05, 41.90, 15.41])
    mc = np.array([8.29, 41.88, 26.52])
    ajc = (mm + lm) / 2
    kjc = (fh + mc) / 2

    print("Position of the ankle joint center:", ajc)
    print("Position of the knee joint center:", kjc)
    return ajc, fh, kjc, lm, mc, mm


@app.cell
def _(ajc, kjc, lm, mm, np):
    # calculation of the anatomical coordinate system axes (basis)
    y = kjc - ajc
    x = np.cross(y, mm - lm)
    z = np.cross(x, y)
    print("Versors:")
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    print("x =", x)
    print("y =", y)
    print("z =", z)
    Oleg = ajc
    print("\nOrigin =", Oleg)
    return Oleg, x, y, z


@app.cell
def _(np, x, y, z):
    # Rotation matrices
    RGl = np.array([x, y, z]).T
    print("Rotation matrix from the anatomical to the laboratory coordinate system:\n", RGl)
    RlG = RGl.T
    print("\nRotation matrix from the laboratory to the anatomical coordinate system:\n", RlG)
    return RGl, RlG


@app.cell
def _(Oleg, np):
    # Translational vector
    OG = np.array([0, 0, 0])  # Laboratory coordinate system origin
    LG = Oleg - OG
    print("Translational vector from the anatomical to the laboratory coordinate system:\n", LG)
    return (LG,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To get the coordinates from the laboratory (Global) coordinate system to the anatomical (local) coordinate system:

    $$
    \mathbf{P_l} = \mathbf{R_{lG}}\left(\mathbf{P_G}-\mathbf{L_G}\right)
    $$

    **Before you run the next cell**, predict two of the results without computing anything: the coordinates of the ankle joint center, and two of the three coordinates of the knee joint center.
    """)
    return


@app.cell
def _(LG, RlG, ajc, fh, kjc, lm, mc, mm, np):
    # position of each marker and of each joint center in the anatomical coordinate system
    mml = np.dot(RlG, (mm - LG))  # equivalent to the algebraic expression RlG*(mm - LG).T
    lml = np.dot(RlG, (lm - LG))
    fhl = np.dot(RlG, (fh - LG))
    mcl = np.dot(RlG, (mc - LG))
    ajcl = np.dot(RlG, (ajc - LG))
    kjcl = np.dot(RlG, (kjc - LG))
    print("Coordinates of mm in the anatomical system:\n", mml.round(4) + 0.0)
    print("Coordinates of lm in the anatomical system:\n", lml.round(4) + 0.0)
    print("Coordinates of fh in the anatomical system:\n", fhl.round(4) + 0.0)
    print("Coordinates of mc in the anatomical system:\n", mcl.round(4) + 0.0)
    print("Coordinates of kjc in the anatomical system:\n", kjcl.round(4) + 0.0)
    print("Coordinates of ajc in the anatomical system (origin):\n", ajcl.round(4) + 0.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results read like a description of the leg. The ankle joint center is the origin, $[0,0,0]$. The knee joint center lies on the $y$ axis, at $[0, 32.01, 0]$: 32 cm up the leg, which is the length of the leg between the two joint centers. The two malleoli have equal and opposite coordinates, because the origin is their midpoint, and they lie in the $yz$ plane ($x = 0$), because the anterior-posterior axis was built perpendicular to the line between them. And the two markers at the knee are 3.5 cm apart in $x$: the fibular head is behind the medial condyle.

    Now the same transformation with the transformation matrix. Let's build $\mathbf{T_{Gl}}$, check that its inverse has the form given above, and check that it is not orthonormal:
    """)
    return


@app.cell
def _(LG, RGl, RlG, mm, np):
    TGl = np.eye(4)
    TGl[:3, :3] = RGl
    TGl[:3, 3] = LG
    _TlG = np.linalg.inv(TGl)

    print("Transformation matrix TGl:\n", TGl)
    print("\nIts inverse equals [RlG, -RlG LG]?", np.allclose(_TlG[:3, :3], RlG) and np.allclose(_TlG[:3, 3], -RlG @ LG))
    print("Its inverse equals its transpose?", np.allclose(_TlG, TGl.T))
    print("\nCoordinates of mm in the anatomical system, with the transformation matrix:", (_TlG @ np.append(mm, 1)).round(4) + 0.0)
    return (TGl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same coordinates as before, now with a fourth component equal to 1, the price of packing a translation into a matrix product.

    **Exercise 5.** A marker is placed on the shin, 4 cm in front of the ankle joint center and 15 cm up the leg: $\mathbf{P_l} = [4,\, 15,\, 0]$ cm in the anatomical coordinate system of the leg. Where is it in the laboratory? Calculate it in two ways, with $\mathbf{R_{Gl}}$ and $\mathbf{L_G}$, and with $\mathbf{T_{Gl}}$, and explain why it is more than 4 cm in front of the ankle joint center.
    """)
    return


@app.cell(hide_code=True)
def _(LG, RGl, TGl, ajc, mo, np):
    _Pl = np.array([4, 15, 0])
    _PG = LG + RGl @ _Pl
    _PG_T = (TGl @ np.append(_Pl, 1))[:3]
    _offset = _PG - ajc

    mo.accordion(
        {
            "Solution to Exercise 5": mo.md(
                rf"""
    From the anatomical to the laboratory coordinate system we rotate first and then translate, $\mathbf{{P_G}} = \mathbf{{L_G}} + \mathbf{{R_{{Gl}}}}\,\mathbf{{P_l}}$:

    `P_G = {_PG.round(2)}` cm

    and with the transformation matrix, $\mathbf{{T_{{Gl}}}}\,[\mathbf{{P_l}}, 1]$ gives the same point, `{_PG_T.round(2)}` cm.

    Relative to the ankle joint center the marker is at `{_offset.round(2)}` cm in the laboratory: {_offset[1]:.1f} cm higher and {_offset[0]:.1f} cm further forward. Of those {_offset[0]:.1f} cm, $4 \times {RGl[0, 0]:.4f} = {4 * RGl[0, 0]:.2f}$ cm come from the marker being in front of the leg, and $15 \times {RGl[0, 1]:.4f} = {15 * RGl[0, 1]:.2f}$ cm from the leg itself, which is inclined forward: the $\mathbf{{X}}$ component of its $y$ axis is {RGl[0, 1]:.4f}.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A whole stride

    Winter's example is a single instant. A motion capture system gives the same markers at every frame, and the same calculation, repeated frame by frame, gives the orientation of the leg throughout a movement. Let's do it for a walking trial recorded in our laboratory, in the file `walk.trc` in the `data` folder of this repository: 184 frames at 150 Hz, with the laboratory's $\mathbf{X}$ axis pointing in the direction of walking, $\mathbf{Y}$ upwards and $\mathbf{Z}$ to the right.

    A `.trc` file is a tab-separated text file with six lines of header before the numbers. pandas can read the numbers directly, with the separator, no header, and the number of rows to skip:
    """)
    return


@app.cell
def _(repo_file):
    import pandas as pd

    _data = pd.read_csv(repo_file("data/walk.trc"), sep="\t", header=None, skiprows=6)
    _data.iloc[:5, :8]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But then the columns have no names. The names of the markers are in the fourth line of the file and the coordinates in the fifth, so it is easier to read the header first and the numbers second. The function `read_trc`, from the `functions` folder, does that, and returns the header and a pandas DataFrame with one column per coordinate of each marker:
    """)
    return


@app.cell
def _(read_trc, repo_file):
    _h, walk = read_trc(str(repo_file("data/walk.trc")), fname2="", dropna=False, na=0.0, fmt="uni")
    walk.iloc[:5, :8]
    return (walk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The right leg carries the same four markers as Winter's example: `R.Ankle` (lateral malleolus), `R.Ankle.Medial`, `R.HF` (fibular head) and `R.Knee.Medial`. We convert them from millimetres to centimetres and apply Winter's definition to every frame at once; NumPy's cross product works row by row.

    One detail needs care. In Winter's data the medial malleolus has the larger $z$ coordinate, so $\mathbf{mm} - \mathbf{lm}$ points along $+z$, and $\mathbf{y} \times (\mathbf{mm} - \mathbf{lm})$ points forward. In our laboratory $\mathbf{Z}$ points to the right, so for the right leg it is the *lateral* malleolus that has the larger $z$, and we must use $\mathbf{lm} - \mathbf{mm}$ to keep the $x$ axis pointing forward. It is the same left-right issue as for the femur in [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py).

    Each frame then gives a rotation matrix $\mathbf{R_{lG}}$, with the versors as rows, and `euler_angles_from_rot_xyz` turns it into three angles: $\alpha$ about the anterior-posterior axis $x$, $\beta$ about the long axis $y$, and $\gamma$ about the medio-lateral axis $z$.
    """)
    return


@app.cell
def _(np, plt, walk):
    _t = walk["Time"].to_numpy()

    def _marker(name):
        return walk[[name + c for c in "xyz"]].to_numpy() / 10  # cm

    _lm, _mm = _marker("R.Ankle"), _marker("R.Ankle.Medial")
    _fh, _mc = _marker("R.HF"), _marker("R.Knee.Medial")
    _ajc, _kjc = (_lm + _mm) / 2, (_fh + _mc) / 2

    _y = _kjc - _ajc
    _x = np.cross(_y, _lm - _mm)  # lateral minus medial: right leg, Z to the right
    _z = np.cross(_x, _y)
    _x, _y, _z = (v / np.linalg.norm(v, axis=1, keepdims=True) for v in (_x, _y, _z))
    _RlG = np.stack((_x, _y, _z), axis=1)  # one rotation matrix per frame, versors as rows

    shank_angles = np.array([euler_angles_from_rot_xyz(_R) for _R in _RlG])
    shank_length = np.linalg.norm(_kjc - _ajc, axis=1)

    _, _ax = plt.subplots(figsize=(9, 4.5))
    _labels = [r"$\alpha$, about $x$ (anterior-posterior)", r"$\beta$, about $y$ (long axis)", r"$\gamma$, about $z$ (medio-lateral)"]
    for _i, _color in enumerate(["tab:red", "tab:green", "tab:blue"]):
        _ax.plot(_t, shank_angles[:, _i], color=_color, linewidth=2, label=_labels[_i])
    _ax.axhline(0, color="0.7", linewidth=1, zorder=0)
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Angle [degrees]")
    _ax.set_title("Orientation of the right shank, xyz Cardan angles")
    _ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return shank_angles, shank_length


@app.cell
def _(shank_angles, shank_length):
    print("Range of each angle [degrees]:")
    for _name, _col in zip(["alpha", "beta", "gamma"], shank_angles.T):
        print(f"  {_name:5s}: from {_col.min():6.1f} to {_col.max():6.1f}")
    print(f"\nDistance between the joint centers: from {shank_length.min():.2f} to {shank_length.max():.2f} cm")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The trial starts with the right leg in swing and covers the following stance phase. The largest rotation is $\gamma$, the inclination of the shank in the sagittal plane: from about $-65^o$ in swing, with the knee far ahead of the ankle, to about $+36^o$ around heel strike, with the ankle ahead of the knee, after which the shank rotates forward over the foot during stance.

    $\beta$, the rotation about the long axis, stays between about $-32^o$ and $-19^o$: the shank's coordinate system is rotated externally, with its $x$ axis pointing forward and to the right. Part of the reason is anatomical: the line through the two malleoli is normally turned outwards relative to the axis of the knee (external tibial torsion), and the definition we used takes its medio-lateral direction from the malleoli. Part is the posture of the whole leg: these angles are measured relative to the laboratory, so any outward turn of the leg relative to the direction of walking adds to them.

    And $\alpha$ swings by about 45°, from $-27^o$ to $+18^o$. Is the shank really tilting sideways that much at every stride? It is not. With a large sagittal rotation applied *last* and a coordinate system turned about its long axis, the $xyz$ decomposition attributes part of the sagittal motion to $\alpha$: the same orientations, described in another order, would give a much smaller frontal angle. This "cross-talk" between angles is another face of the dependence on the sequence, and one reason why the flexion–extension axis is placed first in biomechanics. Problem 8 asks you to check it.

    Finally, the distance between the two joint centers, which should be constant for a rigid leg, varies by about 1.3 cm during the trial. The markers move on the skin, and the leg is rigid only approximately; see [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py) for what that does to the axes.

    **Challenge 2.** Add a horizontal line at the angles of the leg in Winter's static example (calculated with `euler_angles_from_rot_xyz(RlG)`) to the figure. Is Winter's posture close to any instant of this stride?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A marker hidden from the cameras

    In walking, the medial malleolus marker is often hidden from the cameras for a few frames, when the other leg passes in front of it. The transformation matrix is the standard way to fill such a gap. Three other markers on the same segment, visible throughout, define a technical coordinate system at every frame. In the last frame before the gap, we express the hidden marker in that coordinate system, $\mathbf{P_l} = \mathbf{T_{Gl}^{-1}}\,\mathbf{P_G}$, once. If the segment is rigid, those local coordinates do not change, so during the gap the marker is wherever $\mathbf{P_G} = \mathbf{T_{Gl}}\,\mathbf{P_l}$ puts it, frame by frame.

    Our trial has no gap, which is convenient: we can pretend that `R.Ankle.Medial` disappears for 0.2 s of the swing phase, fill the gap from `R.HF`, `R.TT` (on the tibial tuberosity) and `R.Ankle`, and compare the result with the marker that was actually measured. The technical coordinate system is built with the recipe of the basis method, with its origin at the centroid of the three markers:
    """)
    return


@app.function
def transformation_from_markers(m1, m2, m3):
    """Transformation matrices T_Gl of the technical coordinate system of three markers.

    Each marker is an array of positions, one frame per row. The first axis is
    along m2 - m1, the second is perpendicular to the plane of the markers, the
    third completes a right-handed basis, and the origin is the centroid of the
    markers. Returns one 4x4 matrix per frame.
    """
    import numpy as np

    v1 = m2 - m1
    v2 = np.cross(v1, m3 - m1)
    v3 = np.cross(v1, v2)
    T = np.zeros((len(m1), 4, 4))
    T[:, :3, :3] = np.stack([v / np.linalg.norm(v, axis=1, keepdims=True) for v in (v1, v2, v3)], axis=2)
    T[:, :3, 3] = (m1 + m2 + m3) / 3
    T[:, 3, 3] = 1
    return T


@app.function
def fill_gap(hidden, m1, m2, m3, gap, ref):
    """Positions of the marker `hidden` in the frames `gap`, from three markers of the same segment.

    The local coordinates of the hidden marker are taken from the frame `ref`.
    """
    import numpy as np

    T = transformation_from_markers(m1, m2, m3)
    p_local = np.linalg.inv(T[ref]) @ np.append(hidden[ref], 1)  # Global to local, once
    return (T[gap] @ p_local)[:, :3]  # local to Global, at every frame of the gap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Before you run the next cell**, predict: after 0.2 s of swing, how far will the filled marker be from the measured one? A millimetre? A centimetre?
    """)
    return


@app.cell
def _(np, plt, walk):
    walk_time = walk["Time"].to_numpy()
    shank_markers = {
        _name: walk[[_name + _c for _c in "xyz"]].to_numpy() / 10  # cm
        for _name in ["R.HF", "R.TT", "R.Ankle", "R.Ankle.Medial"]
    }
    gap_frames = np.arange(15, 45)  # 30 frames, 0.2 s of swing
    gap_ref = gap_frames[0] - 1  # the last frame in which the marker was seen

    _filled = fill_gap(
        shank_markers["R.Ankle.Medial"], shank_markers["R.HF"], shank_markers["R.TT"], shank_markers["R.Ankle"],
        gap_frames, gap_ref,
    )
    gap_error = 10 * np.linalg.norm(_filled - shank_markers["R.Ankle.Medial"][gap_frames], axis=1)  # mm

    _names = list(shank_markers)
    _changes = [
        np.ptp(np.linalg.norm(shank_markers[_a] - shank_markers[_b], axis=1))
        for _i, _a in enumerate(_names)
        for _b in _names[_i + 1 :]
    ]

    print(f"Gap from {walk_time[gap_frames[0]]:.3f} to {walk_time[gap_frames[-1]]:.3f} s, local coordinates from {walk_time[gap_ref]:.3f} s")
    print(f"Error of the filled marker: mean {gap_error.mean():.1f} mm, largest {gap_error.max():.1f} mm")
    print(f"Largest change in a distance between two shank markers during the trial: {max(_changes):.2f} cm")

    _, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(walk_time[gap_frames], gap_error, color="tab:red", linewidth=2, marker="o", markersize=4)
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Error [mm]")
    _ax.set_title("Filled medial malleolus marker minus the measured one")
    plt.tight_layout()
    plt.show()
    return gap_error, gap_frames, gap_ref, shank_markers, walk_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The error is zero at the reference frame, just before the gap. It reaches about 1 cm within 0.1 s, eases a little, and then climbs to almost 2 cm at the end of the gap. Two things are to blame. First, the shank is not quite rigid as the markers see it: the markers sit on skin, which moves relative to the bone, and the distances between them, which a rigid body would keep constant, change by more than 1.5 cm during the trial. The further the leg moves from the reference frame, the more the cluster has deformed. Second, the technical coordinate system can amplify the errors of its own markers, which is what the next exercise is about.

    In practice, gaps are filled from the frames closest to them, often from both ends of the gap at once, and a filled marker is always an estimate.

    **Exercise 6.** In the technical coordinate system above, the first axis goes from `R.HF` to `R.TT`, two markers only about 8 cm apart, both near the knee. Build it instead with the first axis along the leg, from `R.HF` to `R.Ankle`, about 36 cm apart, keeping the same three markers, and fill the same gap. Before computing, predict: can the change matter at all, since the three markers are the same?
    """)
    return


@app.cell(hide_code=True)
def _(gap_error, gap_frames, gap_ref, mo, np, plt, shank_markers, walk_time):
    _filled = fill_gap(
        shank_markers["R.Ankle.Medial"], shank_markers["R.HF"], shank_markers["R.Ankle"], shank_markers["R.TT"],
        gap_frames, gap_ref,
    )
    _error = 10 * np.linalg.norm(_filled - shank_markers["R.Ankle.Medial"][gap_frames], axis=1)  # mm
    _T = transformation_from_markers(shank_markers["R.HF"], shank_markers["R.TT"], shank_markers["R.Ankle"])
    _p_local = (np.linalg.inv(_T[gap_ref]) @ np.append(shank_markers["R.Ankle.Medial"][gap_ref], 1))[:3]
    _lever = np.linalg.norm(_p_local)  # cm, distance of the hidden marker from the origin

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(walk_time[gap_frames], gap_error, color="tab:red", linewidth=2, label="first axis R.HF to R.TT (8 cm)")
    _ax.plot(walk_time[gap_frames], _error, color="tab:blue", linewidth=2, label="first axis R.HF to R.Ankle (36 cm)")
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Error [mm]")
    _ax.legend(loc="upper left")
    plt.tight_layout()

    mo.accordion(
        {
            "Solution to Exercise 6": mo.vstack(
                [
                    mo.md(
                        f"It matters a lot. With the first axis along the leg, the mean error drops from {gap_error.mean():.1f} to {_error.mean():.1f} mm, "
                        + f"and the largest from {gap_error.max():.1f} to {_error.max():.1f} mm."
                    ),
                    _fig,
                    mo.md(
                        "With noise-free markers, the three markers would define the same coordinate system in any order. "
                        + "With real markers they do not, because the first axis is taken exactly along the first pair, "
                        + "and the same error of a few millimetres tilts a vector 8 cm long four to five times more than a vector 36 cm long. "
                        + f"The hidden marker is {_lever:.0f} cm from the origin of the technical coordinate system, "
                        + f"so every tenth of a degree of error in the axes moves it by {10 * _lever * np.radians(0.1):.1f} mm. "
                        + "In the first half of the gap the two curves agree: that part of the error comes from the skin, and no choice of axes removes it. "
                        + "The late climb came from the short first axis. "
                        + "The rule from [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py) "
                        + "applies: build the first axis from the two markers that are farthest apart."
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your sketches from Challenge 0 and to Codman's paradox. In one sentence each, explain both with rotation matrices.
    2. You are given a 3×3 matrix and told it is a rotation matrix. Name two checks you would make before believing it.
    3. A paper reports "the knee was flexed 40°, abducted 5° and rotated 10°" without stating the sequence. What is missing, and how much could it matter?
    4. Why can't you obtain the orientation of the shank relative to the thigh by subtracting the Euler angles of the two segments?
    5. You analyse a movement in which one segment approaches 90° in the second rotation of your sequence. What do you expect to see in the other two angles, and what would you do about it?
    6. The transformation matrix packs a rotation and a translation together. Why is its inverse not its transpose, when the inverse of the rotation matrix is?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. For the example of how the order of rotations of a rigid body affects its orientation, shown in a figure above, deduce the rotation matrices for each of the 4 cases shown in the figure. For the first two cases, deduce the rotation matrices from the Global to the local coordinate system, and for the other two, the rotation matrices from the local to the Global coordinate system.

    2. Consider the data from problem 5 in the notebook [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py), where the following anatomical landmark positions are given (units in meters): RASIS=[0.5, 0.8, 0.4], LASIS=[0.55, 0.78, 0.1], RPSIS=[0.3, 0.85, 0.2], and LPSIS=[0.29, 0.78, 0.3]. Deduce the rotation matrices for the Global to anatomical coordinate system and for the anatomical to Global coordinate system.

    3. For the data from the example with actual motion analysis data, calculate the Cardan angles using the $zxy$ sequence for the orientation of the leg with respect to the laboratory (but remember that the letters chosen to refer to axes are arbitrary; what matters is the directions they represent).

    4. Write down the 4×4 matrices for each of the following (adapted from a [computer graphics exercise sheet](https://web.archive.org/web/20200224231232/http://www.eecs.qmul.ac.uk/~sgg/cg/Exers/transformations_ex.html) of Queen Mary University of London):<br>
       a. To translate by the vector (1, 2, 3).<br>
       b. To scale with respect to the origin by the amount (2, 4, 6).<br>
       c. To rotate around the z axis by 45 degrees (note that sin 45° = cos 45° = 1/√2).<br>
       d. To rotate around the x axis by 45 degrees.

    5. Compute the variation of Codman's movement from Guiding question 0.2: starting from the arm hanging at the side, abduct the arm 90° in the frontal plane, then swing it horizontally 90° to the front, then lower it 90° in front of the body to the side. Write each movement as a rotation about a fixed axis, with its sign, and find where the palm faces at the end. What single rotation is equivalent to the whole sequence?

    6. For the $zxz$ sequence, repeat the analysis of the section "How close is too close?": find the value of $\beta$ at which the sequence locks, write a function that computes the three $zxz$ angles from a rotation matrix, and show how the errors in the angles grow near the lock.

    7. The coordinate system of the thigh is rotated from the laboratory by the $xyz$ Euler angles (5°, 10°, 30°), and that of the shank by (5°, 15°, −20°). Calculate the rotation matrix of the shank relative to the thigh and its $xyz$ Euler angles. Are they the differences between the angles of the two segments?

    8. Repeat the analysis of the whole stride with the $zxy$ sequence, which puts the rotation about the medio-lateral axis first. Use `euler_rotmat` to obtain the algebraic matrix, derive the formulas for the three angles, and plot them. Compare the frontal-plane angle with $\alpha$ of the $xyz$ sequence, and explain the difference.

    9. Euler's rotation theorem also works the other way round: a rotation by the angle $\theta$ about the axis $\hat{\mathbf{n}}$ has the rotation matrix $\mathbf{R} = \mathbf{I} + \sin\theta\,\mathbf{K} + (1 - \cos\theta)\,\mathbf{K}^2$ (Rodrigues' formula), where $\mathbf{K}$ is the matrix for which $\mathbf{K}\mathbf{v} = \hat{\mathbf{n}} \times \mathbf{v}$ for any vector $\mathbf{v}$. Write $\mathbf{K}$, implement the formula as a function, and check it against `axis_angle` with the rotation of $30^o$, $45^o$ and $60^o$ about $\mathbf{X}$, $\mathbf{Y}$ and $\mathbf{Z}$ from the section "One rotation is enough".

    10. Expand $\mathbf{R_X}(\theta)$ and $\mathbf{R_Z}(\theta)$ to second order in $\theta$ ($\cos\theta \approx 1 - \theta^2/2$, $\sin\theta \approx \theta$) and show that the two orders of the rotations differ by a rotation of approximately $\theta^2$ radians about the $\mathbf{Y}$ axis, as the table in "How much does the order matter?" suggests. Why does the axis of the difference tilt away from $\mathbf{Y}$ for larger angles?

    11. Using the four pelvic markers of `walk.trc` (`R.ASIS`, `L.ASIS`, `R.PSIS` and `L.PSIS`) and the VAKHUM convention of Problem 2, calculate the anatomical coordinate system of the pelvis at every frame of the stride, and plot its $xyz$ Cardan angles relative to the laboratory. Which angle varies most, and by how much? Is the pelvis close to gimbal lock at any moment?

    12. The right thigh in `walk.trc` carries the markers `R.GTR` (greater trochanter), `R.Knee` (lateral epicondyle) and `R.Knee.Medial`. Define a coordinate system for the thigh in the same way as for the shank in the section "A whole stride" (the $y$ axis from the knee joint center, midway between the two knee markers, to the greater trochanter, and a temporary medio-lateral axis from `R.Knee.Medial` to `R.Knee`). Calculate the rotation matrix of the shank relative to the thigh at every frame, and use `axis_angle` to plot the total rotation of the knee during the stride. When is it largest? Why is it not zero when the knee is at its straightest?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers to the problems

    1. For positive 90° turns and the figure’s four rows: fixed X then Y, global-to-local $[[0,0,-1],[1,0,0],[0,-1,0]]$; fixed Y then X, global-to-local $[[0,1,0],[0,0,1],[1,0,0]]$; local x then y, local-to-global $[[0,0,1],[1,0,0],[0,1,0]]$; local y then x, local-to-global $[[0,1,0],[0,0,-1],[-1,0,0]]$.
    2. Global-to-pelvis $R=[[0.980530,-0.098327,0.169977],[0.107934,0.992988,-0.048210],[-0.164045,0.065618,0.984268]]$. Pelvis-to-global is $R^T$.
    3. Intrinsic $zxy$ angles for the static leg example are approximately $(-6.92713,-3.08021,-1.66623)^\circ$.
    4. (a) $[[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]$. (b) $\operatorname{diag}(2,4,6,1)$. With $q=1/\sqrt2$: (c) $[[q,-q,0,0],[q,q,0,0],[0,0,1,0],[0,0,0,1]]$; (d) $[[1,0,0,0],[0,q,-q,0],[0,q,q,0],[0,0,0,1]]$.
    5. With $X$ anterior, $Y$ upward and $Z$ right: $R_Z(-90^\circ)R_Y(90^\circ)R_X(-90^\circ)=R_Y(90^\circ)$. The right palm ends facing backward, opposite to the original sequence’s result.
    6. $zxz$ locks at middle angle $\beta=0$ or $\pi$. For local-to-global matrix $A$ away from the lock, $\beta=\arccos A_{22}$, $\alpha=\operatorname{atan2}(A_{02},-A_{12})$, $\gamma=\operatorname{atan2}(A_{20},A_{21})$. Angle-error amplification scales with $1/|\sin\beta|$; at the lock the individual outer angles are not unique.
    7. Shank-to-thigh rotation is approximately $[[0.639691,0.764917,0.075479],[-0.764257,0.643438,-0.043578],[-0.081900,-0.029809,0.996195]]$. Intrinsic xyz angles are $(2.50477,4.32875,-50.09468)^\circ$; they are not the componentwise differences.
    8. For local-to-global $A$, $\beta=\arcsin A_{21}$, $\alpha=\operatorname{atan2}(-A_{01},A_{11})$, $\gamma=\operatorname{atan2}(-A_{20},A_{22})$. The stride ranges in z,x,y order are approximately [−58.545,29.879]°, [−1.855,6.236]°, [−35.699,−20.707]°. The frontal excursion is much smaller than in xyz because the decomposition assigns rotations differently.
    9. $K=[[0,-n_z,n_y],[n_z,0,-n_x],[-n_y,n_x,0]]$; `I + sin(theta)*K + (1-cos(theta))*(K@K)` reproduces the same axis-angle rotation to numerical precision.
    10. The leading order difference is a rotation of magnitude $\theta^2$ about Y, with sign determined by the order used for comparison. Higher-order commutator terms add X and Z components at finite angles, tilting the equivalent axis.
    11. Pelvis xyz ranges are about [−8.887,3.463]°, [−5.608,9.246]°, [−4.781,1.701]°. The Y angle varies most, by 14.854°; its distance from ±90° confirms that this recording is far from the xyz lock.
    12. Total relative rotation ranges approximately 19.569–69.987°, largest at zero-based frame 19, time 0.127 s. Its nonzero minimum reflects the anatomical/technical axis definitions, three-dimensional rotation and marker/soft-tissue effects; it is not a pure clinical flexion angle.

    ///
    """)
    return








@app.cell
def _():
    import marimo as mo

    return (mo,)




@app.cell(hide_code=True)
def _(mo, np, plt, walk, walk_time):
    def _marker(_name):
        return walk[[_name+_c for _c in "xyz"]].to_numpy()/10
    def _unit(_v):
        return _v/np.linalg.norm(_v,axis=1,keepdims=True)
    _lm,_mm,_fh,_mc=[_marker(_n) for _n in ["R.Ankle","R.Ankle.Medial","R.HF","R.Knee.Medial"]]
    _y=_unit((_fh+_mc-_lm-_mm)/2);_x=_unit(np.cross(_y,_lm-_mm));_z=np.cross(_x,_y)
    _Bs=np.stack([_x,_y,_z],axis=1)
    _zxy=np.degrees(np.c_[np.arctan2(-_Bs[:,1,0],_Bs[:,1,1]),np.arcsin(np.clip(_Bs[:,1,2],-1,1)),np.arctan2(-_Bs[:,0,2],_Bs[:,2,2])])
    _ra,_la,_rp,_lp=[_marker(_n) for _n in ["R.ASIS","L.ASIS","R.PSIS","L.PSIS"]]
    _z=_unit(_ra-_la);_y=_unit(np.cross(_z,(_ra+_la-_rp-_lp)/2));_x=np.cross(_y,_z)
    _Bp=np.stack([_x,_y,_z],axis=1)
    _pq=np.array([euler_angles_from_rot_xyz(_R) for _R in _Bp])
    _knee=(_marker("R.Knee")+_marker("R.Knee.Medial"))/2
    _y=_unit(_marker("R.GTR")-_knee);_x=_unit(np.cross(_y,_marker("R.Knee")-_marker("R.Knee.Medial")));_z=np.cross(_x,_y)
    _Bt=np.stack([_x,_y,_z],axis=1);_rel=_Bt@np.swapaxes(_Bs,-1,-2)
    _angle=np.degrees(np.arccos(np.clip((np.trace(_rel,axis1=1,axis2=2)-1)/2,-1,1)))
    _fig,_ax=plt.subplots(3,1,figsize=(9,9),sharex=True)
    _ax[0].plot(walk_time,_zxy);_ax[0].legend(["z","x","y"]);_ax[0].set(ylabel="Shank zxy [deg]")
    _ax[1].plot(walk_time,_pq);_ax[1].legend(["x","y","z"]);_ax[1].set(ylabel="Pelvis xyz [deg]")
    _ax[2].plot(walk_time,_angle);_ax[2].set(xlabel="Time [s]",ylabel="Knee total rotation [deg]")
    _fig.tight_layout()
    _problem_plot = mo.accordion({"Final stride plots — Problems 8, 11 and 12":mo.as_html(_fig)},expanded=False)
    plt.close(_fig)
    _problem_plot
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read pages 1136–1164 of chapter 21 of [Ruina and Pratap's book](https://ruina.tam.cornell.edu/Book/index.html), an elementary introduction to 3D rigid-body dynamics.
    - [Rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) and [Euler angles](https://en.wikipedia.org/wiki/Euler_angles) on Wikipedia.
    - [Rigid-body transformations in a plane (2D)](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py): the two-dimensional version of this notebook.
    - [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py) and [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py): how the coordinate systems related by these matrices are built from markers.
    - [Angular velocity in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/angular_velocity_3d.py): what comes after orientation.

    ### Video lectures on the Internet

    - [Rotation in R3 around the x-axis](https://www.khanacademy.org/math/linear-algebra/matrix-transformations/lin-trans-examples/v/rotation-in-r3-around-the-x-axis) - Khan Academy
    - [Modern Robotics, Chapter 3.2.1: Rotation Matrices (Part 1 of 2)](https://youtu.be/OZucG1DY_sY) - Northwestern Robotics
    - [Rotations in 3D](https://youtu.be/wg9bI8-Qx2Q)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional assignments from the original sources

    These original reading assignments remain available for further study. Their full statements and source-specific answer keys are external to this guide.

    - Solve the first two problems of [this homework assignment](https://web.archive.org/web/20240420193536/https://rrg.utk.edu/resources/BME473/assignments/BME473_homework_3.pdf) (BME 473, University of Tennessee).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Corke P (2011) [Robotics, Vision and Control: Fundamental Algorithms in MATLAB](https://petercorke.com/RVC/). Springer-Verlag Berlin.

    - Grood ES, Suntay WJ (1983) [A joint coordinate system for the clinical description of three-dimensional motions: application to the knee](https://doi.org/10.1115/1.3138397). Journal of Biomechanical Engineering, 105(2), 136–144.

    - [Maths - Euler Angles](http://www.euclideanspace.com/maths/geometry/rotations/euler/).

    - Murray RM, Li Z, Sastry SS (1994) [A Mathematical Introduction to Robotic Manipulation](http://www.cds.caltech.edu/~murray/mlswiki/index.php/Main_Page). Boca Raton, CRC Press.

    - Politti JC, Goroso G, Valentinuzzi ME, Bravo O (1998) [Codman's paradox of the arm rotations is not a paradox: mathematical validation](https://doi.org/10.1016/S1350-4533(98)00020-4). Medical Engineering & Physics, 20(4), 257–260.

    - Rade D (2017) [Cinemática e Dinâmica para Engenharia](https://www.grupogen.com.br/e-book-cinematica-e-dinamica-para-engenharia). Grupo GEN.

    - Robertson G, Caldwell G, Hamill J, Kamen G (2013) [Research Methods in Biomechanics](https://books.google.com.br/books?id=gRn8AAAAQBAJ). 2nd Edition. Human Kinetics.

    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](https://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.

    - Siciliano B, Sciavicco L, Villani L, Oriolo G (2009) [Robotics - Modelling, Planning and Control](https://books.google.com.br/books/about/Robotics.html?hl=pt-BR&id=jPCAFmE-logC). Springer-Verlag London.

    - Winter DA (2009) [Biomechanics and motor control of human movement](https://books.google.com.br/books?id=_bFHL08IWfwC). 4 ed. Hoboken, USA: Wiley.

    - Wu G, Cavanagh PR (1995) [ISB recommendations for standardization in the reporting of kinematic data](https://doi.org/10.1016/0021-9290(95)00017-C). Journal of Biomechanics, 28(10), 1257–1261.

    - Zatsiorsky VM (1997) [Kinematics of Human Motion](https://books.google.com.br/books/about/Kinematics_of_Human_Motion.html?id=Pql_xXdbrMcC&redir_esc=y). Champaign, Human Kinetics.
    """)
    return


if __name__ == "__main__":
    app.run()
