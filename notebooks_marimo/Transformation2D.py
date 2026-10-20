import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rigid-body transformations in a plane (2D)

    > Marcos Duarte, Renato Naville Watanabe,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A motion capture system measures markers in the coordinate system of the laboratory. But the questions we ask are about body segments: how far the thigh has rotated, where the knee joint is, what angle the knee makes. Answering them means moving back and forth between the laboratory's coordinate system and coordinate systems attached to the segments, and the tool for that is the rigid-body transformation: a translation, a rotation, or both.

    This notebook builds that tool in the simplest setting, a plane, where a rotation is a single angle. It derives the rotation matrix three different ways, shows how to invert it at no cost, builds it from marker positions, uses it for segment and joint angles, and combines it with a translation. It ends by locating a joint that carries no marker, which is what the whole apparatus is for.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a result *before* the code prints it. Make the prediction; a wrong guess that the code then corrects teaches more than a right one.

    The **Exercises** come with worked solutions, folded away under a *Solution* heading. Try each one on paper or in a new cell first, and open the solution only to check your answer.

    You will need the vector operations from [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb), the idea of a basis from [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb), and matrix multiplication from [Matrix](https://github.com/BMClab/BMC/blob/master/notebooks/Matrix.ipynb).

    **Challenge 0.** Put a coin on a sheet of paper and draw a pair of x–y axes on the paper, and another pair on the table. Write down the coin's coordinates in both. Now turn the paper by about 45° on the table, carrying the coin with it, and write down both sets of coordinates again. Which set changed? Keep your notes nearby; the checkpoint questions ask you to come back to them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where is the knee?

    In a gait laboratory, two markers are placed on the side of a person's thigh, seen in the sagittal plane: one on the greater trochanter, near the hip, and one lower on the thigh. With the person standing still, the position of the knee joint centre is also located, for example midway between markers on the two femoral epicondyles. Those knee markers are then taken off, because the one on the inside of the knee would be knocked off by the other leg. In metres, with $X$ pointing forward and $Y$ upwards:

    | | standing (calibration) | during the movement |
    |---|---|---|
    | marker on the greater trochanter | (0.000, 0.900) | (0.150, 0.800) |
    | marker on the lateral thigh | (0.020, 0.700) | (0.333, 0.717) |
    | knee joint centre | (0.010, 0.480) | ? |

    The thigh is a rigid body, so the knee is still exactly where it was *relative to the thigh*. But the thigh has moved and turned, so its position relative to the laboratory has changed, and there is no marker on it to tell us where. **Where is the knee during the movement?**

    **Guiding questions 0.**

    1. The knee is 42 cm from the marker on the greater trochanter. Is knowing where that marker is now enough to locate the knee? What else do you need?
    2. How many numbers are needed to describe where the thigh is in the plane? Which of them can you get from the two markers?
    3. Make a rough guess: during the movement, is the knee in front of or behind the trochanter marker? Above or below it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the numerical work and Matplotlib draws the figures. SymPy appears once, for a symbolic product of rotation matrices. The print options show four decimals and suppress the scientific notation of numbers that are zero up to rounding.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=4, suppress=True)
    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affine transformations

    Translation and rotation are two examples of [affine transformations](https://en.wikipedia.org/wiki/Affine_transformation). Affine transformations preserve straight lines, but not necessarily the distances between points. Other examples are scaling, shear and reflection. The figure below illustrates different affine transformations in a plane. On top of each one is a 3×3 matrix, its transformation matrix, which is the mathematical representation of the physical transformation. We will build exactly this kind of matrix for the translation and rotation of a rigid body.

    <figure><center><img src="https://upload.wikimedia.org/wikipedia/commons/2/2c/2D_affine_transformation_matrix.svg" width=360 alt="Affine transformations"/></center><figcaption><center><i>Figure. Examples of affine transformations in a plane applied to a square (with the letter <b>F</b> in it) and the corresponding transformation matrices (<a href="https://en.wikipedia.org/wiki/Affine_transformation">image from Wikipedia</a>).</i></center></figcaption></figure>

    Translation and rotation keep every distance unchanged. So does reflection, but a reflection turns a right hand into a left hand, which no physical body can do. Translations, rotations and their combinations are the only transformations a rigid body can undergo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kinematics of a rigid body

    The kinematics of a rigid body is completely described by its pose, that is, its position and orientation in space; the corresponding changes are translation and rotation. The translation and rotation of a rigid body are also known as rigid-body transformations, or simply rigid transformations.

    In physics, a [rigid body](https://en.wikipedia.org/wiki/Rigid_body) is a model, an idealization, for a body whose deformation is neglected: the distance between every pair of its points is considered constant. Consequently, the position and orientation of a rigid body can be completely described by a coordinate system attached to it. Two (or more) coordinate systems can then represent the same rigid body at two (or more) instants, or two (or more) rigid bodies in space.

    In motion analysis, rigid-body transformations describe the position and orientation of each body segment, through a local (anatomical) coordinate system attached to the segment, in relation to a global coordinate system fixed in the laboratory. A segment may also carry a technical coordinate system, defined by markers that are not on anatomical landmarks; the positions of the anatomical landmarks are then reconstructed from it, which is exactly the knee problem above. [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb) describes these coordinate systems in more detail.

    We will study rigid-body transformations as transformations between two coordinate systems. For simplicity, we start with planar (two-dimensional) transformations; [Rigid-body transformations in 3D](https://github.com/BMClab/BMC/blob/master/notebooks/Transformation3D.ipynb) extends them to three dimensions, where rotations are considerably more complicated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Translation

    In a plane, two coordinates and one angle are enough to describe the pose of a rigid body: it has three degrees of freedom. Let's look at translation first, then at rotation, and combine them at the end.

    A pure translation of one coordinate system relative to another, and the representation of a point in both, is illustrated below. Remember that this is equivalent to describing the translation between two rigid bodies.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/translation2D.png?raw=1" width=310 alt="translation 2D"/></center><figcaption><center><i>Figure. A point in two-dimensional space represented in two coordinate systems (Global and local), with one system translated.</i></center></figcaption></figure>

    The position of point $\mathbf{P}$, originally described in the local coordinate system, described in the Global coordinate system in vector form is

    $$
    \mathbf{P_G} = \mathbf{L_G} + \mathbf{P_l}
    $$

    where $\mathbf{L_G}$ is the position of the local origin in the Global coordinate system. For each component,

    $$
    \begin{aligned}
    \mathbf{P_X} &= \mathbf{L_X} + \mathbf{P}_x \\
    \mathbf{P_Y} &= \mathbf{L_Y} + \mathbf{P}_y
    \end{aligned}
    $$

    and in matrix form,

    $$
    \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y}
    \end{bmatrix} =
    \begin{bmatrix}
    \mathbf{L_X} \\
    \mathbf{L_Y}
    \end{bmatrix} +
    \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y
    \end{bmatrix}
    $$

    Because position and translation can be treated as vectors, the inverse operation, describing the position in the local coordinate system in terms of the Global one, is simply

    $$
    \mathbf{P_l} = \mathbf{P_G} -\mathbf{L_G}
    \qquad \text{or} \qquad
    \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y
    \end{bmatrix} =
    \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y}
    \end{bmatrix} -
    \begin{bmatrix}
    \mathbf{L_X} \\
    \mathbf{L_Y}
    \end{bmatrix}
    $$

    A translation of the origin is the simplest case of a [Galilean transformation](http://en.wikipedia.org/wiki/Galilean_transformation); [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb) uses one in which the translation grows with time.

    For example, if the local coordinate system is translated by $\mathbf{L_G}=[2, 3]$ in relation to the Global coordinate system, a point with coordinates $\mathbf{P_l}=[4, 5]$ in the local coordinate system has the position $\mathbf{P_G}=[6, 8]$ in the Global coordinate system:
    """)
    return


@app.cell
def _(np):
    LG = np.array([2, 3])  # NumPy 1D array with 2 elements
    _Pl = np.array([4, 5])
    _PG = LG + _Pl

    print("PG =", _PG)
    return (LG,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This operation also works with more than one point at once, one point per row; NumPy knows how to combine arrays of different shapes ([broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)):
    """)
    return


@app.cell
def _(LG, np):
    _Pl = np.array([[4, 5], [6, 7], [8, 9]])  # 2D array with 3 rows and 2 columns
    _PG = LG + _Pl

    print("PG =\n", _PG)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rotation

    A pure rotation of one coordinate system relative to another, and the representation of a point in both, is illustrated below (again, this is equivalent to describing the rotation between two rigid bodies). The rotation is about an axis perpendicular to the page, not shown in the figure; in a three-dimensional coordinate system it would be the $\mathbf{Z}$ axis.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotation2D.png?raw=1" width=270 alt="rotation 2D"/></center><figcaption><center><i>Figure. A point in the two-dimensional space represented in two coordinate systems (Global and local), with one system rotated in relation to the other around an axis orthogonal to both coordinate systems.</i></center></figcaption></figure>

    We want to express the position of point $\mathbf{P}$ in the Global coordinate system knowing only its coordinates in the local coordinate system and the angle of rotation between the two. There are several ways to work it out; here are three.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using trigonometry

    From the figure below, the coordinates of point $\mathbf{P}$ in the Global coordinate system can be found from the sides of the triangles marked in red.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotation2Db.png?raw=1" width=600 alt="rotation 2D, trigonometry"/></center><figcaption><center><i>Figure. The coordinates of a point in the Global coordinate system in terms of its coordinates in the local coordinate system.</i></center></figcaption></figure>

    Then:

    $$
    \begin{aligned}
    \mathbf{P_X} &= \mathbf{P}_x \cos \alpha - \mathbf{P}_y \sin \alpha \\
    \mathbf{P_Y} &= \mathbf{P}_x \sin \alpha + \mathbf{P}_y \cos \alpha
    \end{aligned}
    $$

    In matrix form,

    $$
    \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y}
    \end{bmatrix} =
    \begin{bmatrix}
    \cos\alpha & -\sin\alpha \\
    \sin\alpha & \hphantom{-}\cos\alpha
    \end{bmatrix} \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y
    \end{bmatrix}
    $$

    or simply

    $$
    \mathbf{P_G} = \mathbf{R_{Gl}}\mathbf{P_l}
    $$

    where $\mathbf{R_{Gl}}$ is the rotation matrix that rotates the coordinates from the local to the Global coordinate system:

    $$
    \mathbf{R_{Gl}} = \begin{bmatrix}
    \cos\alpha & -\sin\alpha \\
    \sin\alpha & \hphantom{-}\cos\alpha
    \end{bmatrix}
    $$

    Read the subscripts from right to left: $\mathbf{R_{Gl}}$ takes coordinates in $l$ and returns coordinates in $G$. With it, any position in the local coordinate system can be expressed in the Global one. Let's check that before looking at the other ways of obtaining the matrix.

    Consider a local coordinate system rotated by $45^\circ$ in relation to the Global coordinate system, and a point at $\mathbf{P_l}=[1, 1]$ in the local coordinate system.

    **Before you run the next cell**, predict its Global coordinates. (Hint: where does the diagonal of the local coordinate system point?)
    """)
    return


@app.cell
def _(np):
    _alpha = np.pi / 4
    RGl = np.array([[np.cos(_alpha), -np.sin(_alpha)],
                    [np.sin(_alpha),  np.cos(_alpha)]])
    _Pl = np.array([[1, 1]]).T  # transpose the array for correct matrix multiplication
    _PG = RGl @ _Pl  # the operator @ is used for matrix multiplication of arrays

    print("PG =\n", _PG)
    return (RGl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The local diagonal, at $45^\circ$ in the local system, ends up at $90^\circ$ in the Global one: the point lies on the Global $\mathbf{Y}$ axis, at a distance $\sqrt{2} = 1.4142$ from the origin, as it was in the local system. A rotation changes coordinates, never distances.

    With the points $[1,1]$, $[0,1]$ and $[1,0]$ in the local coordinate system, one per column, their positions in the Global coordinate system are:
    """)
    return


@app.cell
def _(RGl, np):
    _Pl = np.array([[1, 1], [0, 1], [1, 0]]).T  # transpose: one point per column
    _PG = RGl @ _Pl

    print("PG =\n", _PG)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the orientation of the arrays. With the rotation matrix on the left, $\mathbf{R}\,\mathbf{P}$, the points must be *columns*; with data stored one point per row, as marker data usually are, the equivalent product is `P @ R.T`.

    We will build this matrix many more times, so let's write it as a function:
    """)
    return


@app.function
def rotation_2d(alpha):
    """Matrix that rotates planar coordinates by the angle `alpha` [rad],
    counterclockwise; it is R_Gl for a local system rotated by `alpha`."""
    import numpy as np

    return np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha),  np.cos(alpha)]])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A local coordinate system is rotated by $90^\circ$ counterclockwise in relation to the Global one. Without a calculator, write $\mathbf{R_{Gl}}$ and find the Global coordinates of the local points $[2, 0]$ and $[0, 3]$. Check your answers with a sketch.

    /// details | Solution to Exercise 1
    With $\cos 90^\circ = 0$ and $\sin 90^\circ = 1$,

    $$
    \mathbf{R_{Gl}} = \begin{bmatrix} 0 & -1 \\ 1 & \hphantom{-}0 \end{bmatrix}
    $$

    so $[2, 0]$ goes to $[0 \cdot 2 - 1 \cdot 0,\; 1 \cdot 2 + 0 \cdot 0] = [0, 2]$, and $[0, 3]$ goes to $[-3, 0]$.

    In the sketch, the local $x$ axis lies along the Global $Y$ axis and the local $y$ axis along the Global $-X$ axis. A point 2 along local $x$ is therefore 2 up, and a point 3 along local $y$ is 3 to the left. `rotation_2d(np.pi / 2) @ [2, 0]` confirms it.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using direction cosines

    Another way to determine the rotation matrix uses the concept of direction cosine.

    > Direction cosines are the cosines of the angles between any two vectors.

    With two coordinate systems, they are the cosines of the angles between each axis of one coordinate system and each axis of the other. The figure below illustrates these direction angles, expressing the local coordinate system in terms of the Global coordinate system.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/directioncosine2D.png?raw=1" width=285 alt="direction angles 2D"/></center><figcaption><center><i>Figure. Definition of direction angles in two-dimensional space.</i></center></figcaption></figure>

    $$
    \mathbf{R_{Gl}} = \begin{bmatrix}
    \cos\mathbf{X}x & \cos\mathbf{X}y \\
    \cos\mathbf{Y}x & \cos\mathbf{Y}y
    \end{bmatrix} =
    \begin{bmatrix}
    \cos(\alpha) & \cos(90^\circ+\alpha) \\
    \cos(90^\circ-\alpha) & \cos(\alpha)
    \end{bmatrix} =
    \begin{bmatrix}
    \cos\alpha & -\sin\alpha \\
    \sin\alpha & \hphantom{-}\cos\alpha
    \end{bmatrix}
    $$

    The same rotation matrix as before.

    The order of the direction cosines follows from our convention: the first row is for the $\mathbf{X}$ coordinate and the second row for the $\mathbf{Y}$ coordinate (the outputs). For the inputs we follow the same order, first column for the $\mathbf{x}$ coordinate, second column for the $\mathbf{y}$ coordinate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using a basis

    Yet another way to deduce the rotation matrix is to view the axes of the rotated coordinate system as unit vectors, versors, of a <a href="http://en.wikipedia.org/wiki/Basis_(linear_algebra)">basis</a>, as illustrated in the figure below.

    > A basis is a set of linearly independent vectors that can represent every vector in a given vector space, that is, a basis defines a coordinate system.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/basis2D2.png?raw=1" width=285 alt="basis 2D"/></center><figcaption><center><i>Figure. Definition of the rotation matrix using a basis in two-dimensional space.</i></center></figcaption></figure>

    The versors of the local coordinate system, written in terms of the versors of the Global coordinate system, are

    $$
    \begin{aligned}
    \hat{\mathbf{e}}_x &= \hphantom{-}\cos\alpha\:\hat{\mathbf{e}}_X + \sin\alpha\:\hat{\mathbf{e}}_Y \\
    \hat{\mathbf{e}}_y &= -\sin\alpha\:\hat{\mathbf{e}}_X + \cos\alpha\:\hat{\mathbf{e}}_Y
    \end{aligned}
    $$

    As versors, each of them has norm one, which indeed is the case. Placing each versor as a column of a matrix gives the rotation matrix once more:

    $$
    \mathbf{R_{Gl}} = \begin{bmatrix}
    \cos\alpha & -\sin\alpha \\
    \sin\alpha & \hphantom{-}\cos\alpha
    \end{bmatrix}
    $$

    So the rotation matrix *is* the basis of the rotated coordinate system: its columns are the local versors, written in Global coordinates.

    This third way of deriving the rotation matrix is the one most used in motion analysis, because the coordinates of markers in the Global (laboratory) coordinate system are what cameras measure, and the versors are built from them. Before doing that, let's see how to go the other way, from Global to local coordinates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Local-to-Global and Global-to-local rotations

    For the inverse operation, expressing the position of point $\mathbf{P}$ in the local coordinate system in terms of the Global coordinate system, the figure below shows the trigonometry.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotation2Dc.png?raw=1" width=600 alt="rotation 2D, inverse"/></center><figcaption><center><i>Figure. The coordinates of a point in the local coordinate system in terms of its coordinates in the Global coordinate system.</i></center></figcaption></figure>

    Then:

    $$
    \begin{aligned}
    \mathbf{P}_x &= \hphantom{-}\mathbf{P_X} \cos \alpha + \mathbf{P_Y} \sin \alpha \\
    \mathbf{P}_y &= -\mathbf{P_X} \sin \alpha + \mathbf{P_Y} \cos \alpha
    \end{aligned}
    $$

    And in matrix form:

    $$
    \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y
    \end{bmatrix} =
    \begin{bmatrix}
    \hphantom{-}\cos\alpha & \sin\alpha \\
    -\sin\alpha & \cos\alpha
    \end{bmatrix} \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y}
    \end{bmatrix}
    \qquad \text{or} \qquad
    \mathbf{P_l} = \mathbf{R_{lG}}\mathbf{P_G}
    $$

    where $\mathbf{R_{lG}}$ is the rotation matrix that rotates the coordinates from the Global to the local coordinate system (note the inverse order of the subscripts):

    $$
    \mathbf{R_{lG}} = \begin{bmatrix}
    \hphantom{-}\cos\alpha & \sin\alpha \\
    -\sin\alpha & \cos\alpha
    \end{bmatrix}
    $$

    With direction cosines, the angles between the axes have not changed, so the cosines are the same; only their order changes. Now $\mathbf{x, y}$ are the rows (outputs) and $\mathbf{X, Y}$ are the columns (inputs):

    $$
    \mathbf{R_{lG}} = \begin{bmatrix}
    \cos\mathbf{X}x & \cos\mathbf{Y}x \\
    \cos\mathbf{X}y & \cos\mathbf{Y}y
    \end{bmatrix} =
    \begin{bmatrix}
    \cos(\alpha) & \cos(90^\circ-\alpha) \\
    \cos(90^\circ+\alpha) & \cos(\alpha)
    \end{bmatrix} =
    \begin{bmatrix}
    \hphantom{-}\cos\alpha & \sin\alpha \\
    -\sin\alpha & \cos\alpha
    \end{bmatrix}
    $$

    And writing the versors of the Global axes as a basis in terms of the local coordinate system would produce this matrix too.

    The two matrices are very similar, and that is no coincidence. Each performs the inverse operation of the other: each matrix is the inverse of the other. Equivalently, instead of rotating the local coordinate system by $\alpha$ relative to the Global one, we may rotate the Global coordinate system by $-\alpha$ relative to the local one; remember that $\cos(-\alpha)=\cos(\alpha)$ and $\sin(-\alpha)=-\sin(\alpha)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The orthogonality of the rotation matrix

    (See [Matrix](https://github.com/BMClab/BMC/blob/master/notebooks/Matrix.ipynb) for a review of matrices and their main properties.)

    A very useful property of the rotation matrix is that its inverse is its transpose, because its columns (and its rows) are mutually orthogonal and have norm one. We can show it with the matrices we deduced:

    $$
    \begin{aligned}
    \mathbf{R}\:\mathbf{R^T} & =
    \begin{bmatrix}
    \cos\alpha & -\sin\alpha \\
    \sin\alpha & \hphantom{-}\cos\alpha
    \end{bmatrix}
    \begin{bmatrix}
    \hphantom{-}\cos\alpha & \sin\alpha \\
    -\sin\alpha & \cos\alpha
    \end{bmatrix} \\
    & = \begin{bmatrix}
    \cos^2\alpha+\sin^2\alpha & \cos\alpha \sin\alpha-\sin\alpha \cos\alpha \\
    \sin\alpha \cos\alpha-\cos\alpha \sin\alpha & \sin^2\alpha+\cos^2\alpha
    \end{bmatrix} \\
    & = \begin{bmatrix}
    1 & 0 \\
    0 & 1
    \end{bmatrix} = \mathbf{I}
    \quad \Longrightarrow \quad
    \mathbf{R^{-1}} = \mathbf{R^T}
    \end{aligned}
    $$

    If we have a rotation matrix, we have its inverse for free. In NumPy, the transpose is an attribute of the array and the inverse is a function of `numpy.linalg`:
    """)
    return


@app.cell
def _(RGl, np):
    print("Orthogonal matrix (RGl):\n", RGl)
    print("Transpose (RGl.T):\n", RGl.T)
    print("Inverse (np.linalg.inv(RGl)):\n", np.linalg.inv(RGl))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the inverse, or the transpose, the coordinates in the local coordinate system follow from the coordinates in the Global coordinate system and the rotation matrix:

    $$
    \begin{aligned}
    \mathbf{P_G} = \mathbf{R_{Gl}}\mathbf{P_l} & \implies
    \mathbf{R_{Gl}^{-1}}\mathbf{P_G} = \mathbf{R_{Gl}^{-1}}\mathbf{R_{Gl}}\mathbf{P_l} \\
    & \implies \mathbf{R_{Gl}^{-1}}\mathbf{P_G} = \mathbf{I}\:\mathbf{P_l} \\
    & \implies \mathbf{P_l} = \mathbf{R_{Gl}^{-1}}\mathbf{P_G} = \mathbf{R_{Gl}^T}\mathbf{P_G}
    \quad \text{or} \quad \mathbf{P_l} = \mathbf{R_{lG}}\mathbf{P_G}
    \end{aligned}
    $$

    where we called the inverse of $\mathbf{R_{Gl}}$, $\mathbf{R_{Gl}^{-1}}$, by the name $\mathbf{R_{lG}}$ (note the order of the subscripts). In NumPy, going there and back again:
    """)
    return


@app.cell
def _(RGl, np):
    print("Rotation matrix (RGl):\n", RGl)

    _Pl = np.array([[1, 1]]).T  # transpose the array for correct matrix multiplication
    print("Position in the local coordinate system (Pl):\n", _Pl)

    _PG = RGl @ _Pl
    print("Position in the Global coordinate system (PG = RGl Pl):\n", _PG)

    _Pl_back = RGl.T @ _PG
    print("Position in the local coordinate system using the inverse of RGl (Pl = RlG PG):\n", _Pl_back)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rotation of a vector

    The rotation matrix has another use: to rotate a vector by a given angle around an axis of the coordinate system, as in the figure below.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/rotation2Dvector.png?raw=1" width=205 alt="rotation of a vector in 2D"/></center><figcaption><center><i>Figure. Rotation of a position vector $\mathbf{P}$ by an angle $\alpha$ in two-dimensional space.</i></center></figcaption></figure>

    This time the coordinate system stays where it is and the vector turns. We will not prove it, but the same rotation matrix does the job: the coordinates of the position vector $\mathbf{P'}$, rotated by an angle $\alpha$, are the rotation matrix for the angle $\alpha$ times the coordinates of $\mathbf{P}$:

    $$
    \mathbf{P'} = \mathbf{R}_\alpha\mathbf{P}
    $$

    For example, for $\mathbf{P}=[2,1]$ and $\alpha=30^\circ$, the coordinates of $\mathbf{P'}$ are:
    """)
    return


@app.cell
def _(np):
    _alpha = np.pi / 6
    _R = np.array([[np.cos(_alpha), -np.sin(_alpha)],
                   [np.sin(_alpha),  np.cos(_alpha)]])
    _P = np.array([[2, 1]]).T
    _P_rotated = _R @ _P

    print("P' =\n", _P_rotated)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So the same matrix has two readings. Applied to the coordinates of a point that stays put, it converts them from a rotated coordinate system to the original one. Applied to a vector in a coordinate system that stays put, it turns the vector. The two are easy to mix up, and the mix-up costs a sign.

    **Before you run the next cell**, predict: now keep the vector $\mathbf{P}=[2,1]$ where it is, and turn the *coordinate system* by $30^\circ$ counterclockwise instead. What are the coordinates of the same, unmoved, point in the turned coordinate system?
    """)
    return


@app.cell
def _(np):
    _R30 = rotation_2d(np.pi / 6)
    _P = np.array([2, 1])

    print("Vector turned by +30 degrees, coordinate system fixed:  R P   =", _R30 @ _P)
    print("Coordinate system turned by +30 degrees, point fixed:  R.T P =", _R30.T @ _P)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Turning the axes counterclockwise makes a fixed point appear to turn *clockwise* relative to them, by the same angle. So its new coordinates come from $\mathbf{R}_{-\alpha} = \mathbf{R}_{\alpha}^T$, not from $\mathbf{R}_{\alpha}$: $[2.2321, -0.134]$ instead of $[1.2321, 1.866]$. Whenever you take a rotation matrix from a book or a library, find out which of the two readings it was written for.

    **In summary, some of the properties of the rotation matrix are:**

    1. Its columns form a basis of independent unit vectors (versors), and so do its rows, since the transpose of a rotation matrix is another rotation matrix.
    2. It is orthogonal: its rows are mutually orthogonal, and so are its columns (property 1 rewritten). Because each row and column also has norm one, the matrix is sometimes called orthonormal.
    3. Its determinant is $+1$. For the matrix we deduced, $\cos\alpha \cos\alpha - \sin\alpha(-\sin\alpha)=1$. A matrix with orthonormal columns and determinant $-1$ is not a rotation: it includes a reflection, and it appears when one of the two coordinate systems is left-handed, which you should rarely if ever use.
    4. Its inverse is its transpose.

    **On the different meanings of the rotation matrix:**

    - It represents the transformation between the coordinates of a point expressed in two different coordinate systems.
    - It describes the rotation between two coordinate systems. Its columns are the direction cosines (versors) of the axes of the rotated coordinate system in relation to the other coordinate system, and its rows are the direction cosines (versors) for the inverse rotation.
    - It is an operator that rotates a vector within one coordinate system.
    - It represents a rotation numerically, without appealing to any particular specification of angles.

    **Which matrix to use, from local to Global or from Global to local?**

    - In movement analysis there is a fixed Global (laboratory) coordinate system and a local (moving, for example anatomical) coordinate system attached to each body segment. The movement of the segment is measured in the Global coordinate system, by cameras for example, and to reconstruct the coordinates of the markers in the anatomical coordinate system we need the transformation from Global to local.
    - Of course, if you have one of the matrices, the other is just its transpose; you only have to be careful to use the right one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Which of these matrices are rotation matrices? For each one that is, find the angle of rotation.

    $$
    \mathbf{M} = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & \hphantom{-}0.6 \end{bmatrix}, \qquad
    \mathbf{N} = \begin{bmatrix} 0.6 & \hphantom{-}0.8 \\ 0.8 & -0.6 \end{bmatrix}, \qquad
    \mathbf{K} = \begin{bmatrix} 0.8 & -0.6 \\ 0.5 & \hphantom{-}0.8 \end{bmatrix}
    $$

    Use the properties listed above. The solution checks them with NumPy.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _matrices = {
        "M": np.array([[0.6, -0.8], [0.8, 0.6]]),
        "N": np.array([[0.6, 0.8], [0.8, -0.6]]),
        "K": np.array([[0.8, -0.6], [0.5, 0.8]]),
    }
    _rows, _angles = [], {}
    for _name, _M in _matrices.items():
        _norms = np.linalg.norm(_M, axis=0)
        _det = np.linalg.det(_M)
        _orthonormal = np.allclose(_M.T @ _M, np.eye(2))
        _angles[_name] = np.degrees(np.arctan2(_M[1, 0], _M[0, 0]))
        if _orthonormal and np.isclose(_det, 1):
            _verdict = f"rotation by {_angles[_name]:.2f}°"
        elif _orthonormal:
            _verdict = "orthonormal, but a reflection"
        else:
            _verdict = "not orthonormal"
        _rows.append(
            f"| {_name} | {_norms[0]:.3f}, {_norms[1]:.3f} | {round(_M[:, 0] @ _M[:, 1], 2) + 0.0:.2f} | {_det:.2f} | {_verdict} |"
        )

    mo.accordion(
        {
            "Solution to Exercise 2": mo.md(
                "| matrix | norms of the columns | dot product of the columns | determinant | verdict |\n"
                "|---|---|---|---|---|\n"
                + "\n".join(_rows)
                + "\n\n"
                + f"Only $\\mathbf{{M}}$ passes every test: it is a rotation by {_angles['M']:.2f}°, "
                "the angle whose cosine is 0.6 and whose sine is 0.8. "
                + r"$\mathbf{N}$ has orthonormal columns, but its determinant is $-1$: it reflects the plane "
                "about a line, which no rigid body can do. "
                + f"$\\mathbf{{K}}$ fails at once, because its first column is not a versor. Yet `arctan2` "
                f"returns {_angles['K']:.2f}° for $\\mathbf{{K}}$ and {_angles['N']:.2f}° for $\\mathbf{{N}}$ "
                "without complaint. A formula for the angle always gives a number, so check that the matrix "
                "is a rotation before you trust it."
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculation of the rotation matrix using a basis

    A typical scenario in motion analysis is to calculate the rotation matrix from the positions of markers placed on the moving rigid body. With the markers' positions we create a local basis, which by definition is the rotation matrix of the rigid body with respect to the Global (laboratory) coordinate system. To define a coordinate system from the basis, we also need an origin.

    Let's calculate a basis from the markers' positions. Consider markers at $m_1=[1,1]$, $m_2=[1,2]$ and $m_3=[-1,1]$, measured in the Global coordinate system, as illustrated in the figure below:

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/transrot2Db.png?raw=1" width=700 alt="translation and rotation 2D"/></center><figcaption><center><i>Figure. Three points in two-dimensional space, two possible vectors given these points, and the corresponding basis.</i></center></figcaption></figure>

    A possible local coordinate system with origin at $m_1$ is also shown in the figure. The three markers were chosen on purpose so that the two vectors between them are orthogonal. The translation vector between the two coordinate systems is

    $$
    \mathbf{L_{Gl}} = m_1 - [0,0] = [1,1]
    $$

    and the vectors along the axes of the local coordinate system are

    $$
    \overrightarrow{\mathbf{x}} = m_2 - m_1 = [1,2] - [1,1] = [0,1],
    \qquad
    \overrightarrow{\mathbf{y}} = m_3 - m_1 = [-1,1] - [1,1] = [-2,0]
    $$

    These two vectors do not form an orthonormal basis yet, because they are not both unit vectors ($\overrightarrow{\mathbf{y}}$ is not). Let's normalize them:

    $$
    \hat{\mathbf{e}}_x = \frac{\overrightarrow{\mathbf{x}}}{||\overrightarrow{\mathbf{x}}||} = \frac{[0,1]}{\sqrt{0^2+1^2}} = [0,1],
    \qquad
    \hat{\mathbf{e}}_y = \frac{\overrightarrow{\mathbf{y}}}{||\overrightarrow{\mathbf{y}}||} = \frac{[-2,0]}{\sqrt{2^2+0^2}} = [-1,0]
    $$

    Beware that these versors are not drawn exactly as in the right panel of the figure: plotted from their coordinates, they would start at the origin of the Global coordinate system, not at $[1,1]$.

    Because the markers were carefully chosen, $\hat{\mathbf{e}}_x$ and $\hat{\mathbf{e}}_y$ are orthogonal. We could have done this calculation in NumPy, as we will have to with real data:
    """)
    return


@app.cell
def _(np):
    m1 = np.array([1.0, 1.0])  # marker 1
    m2 = np.array([1.0, 2.0])  # marker 2
    m3 = np.array([-1.0, 1.0])  # marker 3

    x_axis = m2 - m1  # vector x
    y_axis = m3 - m1  # vector y

    ex = x_axis / np.linalg.norm(x_axis)  # versor x
    ey = y_axis / np.linalg.norm(y_axis)  # versor y

    print("x =", x_axis, ", y =", y_axis, "\nex =", ex, ", ey =", ey)
    return ex, ey, m1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\hat{\mathbf{e}}_x$ and $\hat{\mathbf{e}}_y$ are unit vectors and they are orthogonal, so they form a basis, and the rotation matrix is just that basis, with the versors as its columns:

    $$
    \mathbf{R_{Gl}} = \begin{bmatrix}
    0 & -1 \\
    1 & \hphantom{-}0
    \end{bmatrix}
    $$

    This makes sense: the figure shows the local coordinate system rotated by $90^\circ$ in relation to the Global one, and the general form of the rotation matrix gives

    $$
    \mathbf{R} = \begin{bmatrix}
    \cos\alpha & -\sin\alpha \\
    \sin\alpha & \hphantom{-}\cos\alpha
    \end{bmatrix} =
    \begin{bmatrix}
    \cos90^\circ & -\sin90^\circ \\
    \sin90^\circ & \hphantom{-}\cos90^\circ
    \end{bmatrix} =
    \begin{bmatrix}
    0 & -1 \\
    1 & \hphantom{-}0
    \end{bmatrix}
    $$

    So the position of any point in the local coordinate system can be expressed in the Global coordinate system by

    $$
    \mathbf{P_G} = \mathbf{L_{Gl}} + \mathbf{R_{Gl}}\mathbf{P_l}
    = \begin{bmatrix} 1 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 & -1 \\ 1 & \hphantom{-}0 \end{bmatrix} \mathbf{P_l}
    $$

    (the combination of a translation and a rotation is discussed in detail below). For example, the point $\mathbf{P_l}=[1,1]$ has the following position in the Global coordinate system:
    """)
    return


@app.cell
def _(ex, ey, m1, np):
    LGl = m1.reshape(2, 1)  # translation vector, as a column
    print("Translation vector:\n", LGl)

    R_basis = np.column_stack((ex, ey))  # the versors are the columns of RGl
    print("Rotation matrix:\n", R_basis)

    _Pl = np.array([[1, 1]]).T
    print("Position in the local coordinate system:\n", _Pl)

    _PG = LGl + R_basis @ _Pl
    print("Position in the Global coordinate system, PG = LGl + RGl Pl:\n", _PG)
    return LGl, R_basis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With real markers, $m_3 - m_1$ would never be exactly perpendicular to $m_2 - m_1$. In a plane we do not even need the third marker: $\hat{\mathbf{e}}_y$ is $\hat{\mathbf{e}}_x$ turned by $90^\circ$, $[-e_{x,Y},\; e_{x,X}]$, as the next sections do. (In three dimensions, the cross product plays this role; see [Determination of a coordinate system](https://github.com/BMClab/BMC/blob/master/notebooks/CoordinateSystem.ipynb).)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Determination of the unknown angle of rotation

    If we did not know the angle of rotation between the two coordinate systems, which is the usual situation in motion analysis, we would equate one of the terms of the rotation matrix in its algebraic form to its numerical value in the matrix we calculated:

    $$
    \mathbf{R} = \begin{bmatrix}
    \cos\alpha & -\sin\alpha \\
    \sin\alpha & \hphantom{-}\cos\alpha
    \end{bmatrix}
    $$

    so the angle $\alpha$ can be found by

    $$
    \alpha = \arctan\left(\frac{R[1,0]}{R[0,0]}\right)
    $$

    For instance, the first term of the rotation matrix above, $\cos\alpha = 0$, implies that $\alpha$ is $90^\circ$ or $270^\circ$; combined with another term, $\sin\alpha = 1$, it implies that $\alpha=90^\circ$. We can solve this in one step with the tangent $(\sin\alpha/\cos\alpha)$ of two terms of the rotation matrix, using the function `arctan2(y, x)`, which keeps the signs of both terms and so finds the right quadrant (see [Angular kinematics in a plane](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb)):
    """)
    return


@app.cell
def _(R_basis, np):
    _ang = np.arctan2(R_basis[1, 0], R_basis[0, 0]) * 180 / np.pi

    print("The angle is:", _ang)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This procedure would be repeated for each segment and for each instant of the movement to find the rotation of each segment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Joint angle as a sequence of rotations of adjacent segments

    In [Angular kinematics in a plane](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb), we calculated segment and joint angles with simple trigonometric relations. We can also calculate them with rotation matrices.

    The segment angle is given by the matrix of the rotation from the laboratory coordinate system ($G$) to a coordinate system attached to the segment, and the joint angle by the matrix of the rotation from one segment's coordinate system ($l_1$) to the other's ($l_2$). So we compute two bases, one for each segment, and the joint angle comes from the product of the two rotation matrices.

    The description of a point P in the Global basis, given its description in the $l_1$ basis, is

    $$
    \begin{bmatrix}
    x_p\\
    y_p
    \end{bmatrix} = R_{Gl_1}
    \begin{bmatrix}
    x_{p_1}\\
    y_{p_1}
    \end{bmatrix}
    $$

    and the description of the same point P in the Global basis, given its description in the $l_2$ basis, is

    $$
    \begin{bmatrix}
    x_p\\
    y_p
    \end{bmatrix} = R_{Gl_2}
    \begin{bmatrix}
    x_{p_2}\\
    y_{p_2}
    \end{bmatrix}
    $$

    So the description of P in the $l_1$ basis, given its description in the $l_2$ basis, is

    $$
    R_{Gl_1}
    \begin{bmatrix}
    x_{p_1}\\
    y_{p_1}
    \end{bmatrix} =
    R_{Gl_2}
    \begin{bmatrix}
    x_{p_2}\\
    y_{p_2}
    \end{bmatrix}
    \quad \Longrightarrow \quad
    \begin{bmatrix}
    x_{p_1}\\
    y_{p_1}
    \end{bmatrix} =
    \underbrace{R_{Gl_1}^{-1}R_{Gl_2}}_{R_{l_1l_2}}
    \begin{bmatrix}
    x_{p_2}\\
    y_{p_2}
    \end{bmatrix}
    $$

    and the rotation matrix from $l_2$ to $l_1$ is

    $$
    R_{l_1l_2} = R_{Gl_1}^{-1}R_{Gl_2} = R_{Gl_1}^{T}R_{Gl_2}
    $$

    The rotation matrices of the two bases are

    $$
    R_{Gl_1} = \begin{bmatrix}
    \cos(\theta_1) &-\sin(\theta_1)\\
    \sin(\theta_1) &\hphantom{-}\cos(\theta_1)
    \end{bmatrix}, \quad
    R_{Gl_2} = \begin{bmatrix}
    \cos(\theta_2) &-\sin(\theta_2)\\
    \sin(\theta_2) &\hphantom{-}\cos(\theta_2)
    \end{bmatrix}
    $$

    so the rotation matrix from $l_2$ to $l_1$ is

    $$
    \begin{aligned}
    R_{l_1l_2} &= \begin{bmatrix}
    \cos(\theta_1) &\sin(\theta_1)\\
    -\sin(\theta_1) &\cos(\theta_1)
    \end{bmatrix}\begin{bmatrix}
    \cos(\theta_2) &-\sin(\theta_2)\\
    \sin(\theta_2) &\hphantom{-}\cos(\theta_2)
    \end{bmatrix} \\
    &= \begin{bmatrix}
    \cos(\theta_1)\cos(\theta_2)+\sin(\theta_1)\sin(\theta_2) &\cos(\theta_2)\sin(\theta_1)-\cos(\theta_1)\sin(\theta_2)\\
    \cos(\theta_1)\sin(\theta_2)-\cos(\theta_2)\sin(\theta_1)&\cos(\theta_1)\cos(\theta_2)+\sin(\theta_1)\sin(\theta_2)
    \end{bmatrix} \\
    &= \begin{bmatrix}
    \cos(\theta_2-\theta_1) &-\sin(\theta_2-\theta_1)\\
    \sin(\theta_2-\theta_1) &\hphantom{-}\cos(\theta_2-\theta_1)
    \end{bmatrix}
    \end{aligned}
    $$

    The angle $\theta_{l_1l_2} = \theta_2-\theta_1$ is the angle between the two coordinate systems, and it is found from

    $$
    \theta_{l_1l_2} = \arctan\left(\frac{R_{l_1l_2}[1,0]}{R_{l_1l_2}[0,0]}\right)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is an example with the two segments of [Angular kinematics in a plane](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb): segment 1 from $(0, 0)$ to $(1, 1)$, and segment 2 from $(2.1, 0)$ to $(1.1, 1)$. Each segment gives one axis of its basis. For the second axis we need a point on a line perpendicular to the segment, and there is a neat way to get one: turn the vector of the segment, $(\Delta x, \Delta y)$, by $90^\circ$, which gives $(-\Delta y, \Delta x)$. That is the rotation matrix of $90^\circ$ at work.
    """)
    return


@app.cell
def _():
    x1, y1, x2, y2 = 0, 0, 1, 1  # points on segment 1
    x3, y3, x4, y4 = 1.1, 1, 2.1, 0  # points on segment 2

    # A point on the line through the first point of each segment, perpendicular to it
    xl1 = x1 - (y2 - y1); yl1 = y1 + (x2 - x1)  # perpendicular to segment 1
    xl2 = x4 - (y3 - y4); yl2 = y4 + (x3 - x4)  # perpendicular to segment 2
    return x1, x2, x3, x4, xl1, xl2, y1, y2, y3, y4, yl1, yl2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With these points, we can create a basis for each segment and the corresponding rotation matrices:
    """)
    return


@app.cell
def _(np, x1, x2, x3, x4, xl1, xl2, y1, y2, y3, y4, yl1, yl2):
    b1x = np.array([x2 - x1, y2 - y1])
    b1x = b1x / np.linalg.norm(b1x)  # versor x of basis 1
    b1y = np.array([xl1 - x1, yl1 - y1])
    b1y = b1y / np.linalg.norm(b1y)  # versor y of basis 1
    b2x = np.array([x3 - x4, y3 - y4])
    b2x = b2x / np.linalg.norm(b2x)  # versor x of basis 2
    b2y = np.array([xl2 - x4, yl2 - y4])
    b2y = b2y / np.linalg.norm(b2y)  # versor y of basis 2

    RGl1 = np.array([b1x, b1y]).T  # rotation matrix from segment 1 to the laboratory
    RGl2 = np.array([b2x, b2y]).T  # rotation matrix from segment 2 to the laboratory

    print("Rotation matrix from segment 1 to Global:\n", RGl1)
    print("Rotation matrix from segment 2 to Global:\n", RGl2)
    return RGl1, RGl2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the segment and joint angles are simple matrix operations.

    **Before you run the next cell**, predict the three angles: of segment 1, of segment 2, and of the joint between them.
    """)
    return


@app.cell
def _(RGl1, RGl2, np):
    print("Rotation matrix for segment 1:\n", RGl1)
    print("\nRotation angle of segment 1:", np.arctan2(RGl1[1, 0], RGl1[0, 0]) * 180 / np.pi)
    print("\nRotation matrix for segment 2:\n", RGl2)
    print("\nRotation angle of segment 2:", np.arctan2(RGl2[1, 0], RGl2[0, 0]) * 180 / np.pi)

    _Rl1l2 = RGl1.T @ RGl2  # Rl1l2 = Rl1G RGl2

    print("\nJoint rotation matrix (Rl1l2 = Rl1G RGl2):\n", _Rl1l2)
    print("\nJoint angle:", np.arctan2(_Rl1l2[1, 0], _Rl1l2[0, 0]) * 180 / np.pi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same $45^\circ$, $135^\circ$ and $90^\circ$ as obtained with trigonometry in [Angular kinematics in a plane](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb). (Python prints $-0.$ for a number that is zero but carries a negative sign from rounding; it is harmless.)

    Why go to the trouble of matrices for something a subtraction of angles already did? Because the matrix method does not care how the bases were built, needs no rule about which end of a segment points where once the bases are defined, and carries over unchanged to three dimensions, where angles can no longer simply be subtracted.

    **Guiding questions 1.**

    1. Compute $R_{l_2l_1}$ instead of $R_{l_1l_2}$. What angle does it give, and why?
    2. Reverse the direction of segment 2's x axis, from marker 3 to marker 4 instead of from marker 4 to marker 3. What happens to the joint angle? What happens to it in the trigonometric method?
    3. The joint angle came from $R_{Gl_1}^T R_{Gl_2}$. Would $R_{Gl_2} R_{Gl_1}^T$ give the same answer in a plane? Do you expect that to hold in three dimensions?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: a knee angle that jumps by 360°

    The angles above were computed for a single instant. In a real analysis they are computed for every frame, so let's write the recipe as a function that works on whole arrays of frames: the local $x$ axis points from one joint (or marker) to the next, and the local $y$ axis is $x$ turned $90^\circ$ counterclockwise, as in the example above.
    """)
    return


@app.function
def segment_rotation_2d(origin, axis_point):
    """Rotation matrix R_Gl of a planar segment from two of its markers.

    The local x axis points from `origin` to `axis_point`, and the local y axis
    is x turned by 90 degrees counterclockwise. The versors are the columns of
    R_Gl. Works for one frame or for arrays of frames (one position per row).
    """
    import numpy as np

    d = np.asarray(axis_point, dtype=float) - np.asarray(origin, dtype=float)
    ex = d / np.linalg.norm(d, axis=-1, keepdims=True)
    ey = np.stack((-ex[..., 1], ex[..., 0]), axis=-1)
    return np.stack((ex, ey), axis=-1)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is a movement in which the difference between the two methods stops being academic. A kick, seen from the side, simulated at 100 Hz for one second: the knee flexes by up to $130^\circ$ while the hip flexes by up to $15^\circ$. With $X$ pointing forward, the thigh starts straight down, at $-90^\circ$, and the flexing knee carries the shank backward, so the shank's angle *decreases*, down to $-75^\circ - 130^\circ = -205^\circ$ at the peak.

    **Before you run the next cell**, think about that number. What angle will `arctan2`, whose answers lie between $-180^\circ$ and $180^\circ$, report for a shank pointing at $-205^\circ$? And what will the subtraction of the two segment angles give for the knee?
    """)
    return


@app.cell
def _(np, plt):
    _t = np.linspace(0, 1, 101)  # s, 100 Hz
    _thigh = np.radians(-90 + 15 * np.sin(np.pi * _t))  # the hip flexes up to 15 degrees
    _flexion = np.radians(130 * np.sin(np.pi * _t))  # the knee flexes up to 130 degrees
    _hip = np.column_stack((np.zeros_like(_t), np.full_like(_t, 0.9)))
    _knee = _hip + 0.42 * np.column_stack((np.cos(_thigh), np.sin(_thigh)))
    _ankle = _knee + 0.40 * np.column_stack((np.cos(_thigh - _flexion), np.sin(_thigh - _flexion)))

    def _segment_angle(R):
        return np.degrees(np.arctan2(R[..., 1, 0], R[..., 0, 0]))

    _R_thigh = segment_rotation_2d(_hip, _knee)  # one 2x2 matrix per frame
    _R_shank = segment_rotation_2d(_knee, _ankle)
    _by_subtraction = _segment_angle(_R_shank) - _segment_angle(_R_thigh)
    _by_matrices = _segment_angle(np.swapaxes(_R_thigh, -1, -2) @ _R_shank)  # R_l1G R_Gl2 in every frame

    _i = 50  # t = 0.5 s, the peak of the flexion
    print(
        f"At t = {_t[_i]:.1f} s, arctan2 gives the shank {_segment_angle(_R_shank)[_i]:.1f} degrees"
        f" and the thigh {_segment_angle(_R_thigh)[_i]:.1f} degrees"
    )
    print(f"Knee angle by subtraction: {_by_subtraction[_i]:.1f} degrees; with rotation matrices: {_by_matrices[_i]:.1f} degrees")
    print(f"Frames in which the two methods disagree: {np.sum(~np.isclose(_by_subtraction, _by_matrices))} of {_t.size}")

    _, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_t, _by_subtraction, color="tab:red", linewidth=2, linestyle="--", label="subtraction of segment angles")
    _ax.plot(_t, _by_matrices, color="tab:blue", linewidth=2, label=r"rotation matrices, $R_{l_1G}R_{Gl_2}$")
    _ax.axhline(0, color="0.7", linewidth=1)
    _ax.set_xlabel("Time [s]")
    _ax.set_ylabel("Knee angle [degrees]")
    _ax.set_title("Knee angle during a simulated kick")
    _ax.legend(loc="center right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `arctan2` reports the shank at $+155^\circ$, the same direction as $-205^\circ$ but on the other side of the $\pm 180^\circ$ cut. The subtraction then gives a knee bent by $230^\circ$ the wrong way, and it jumps by almost $360^\circ$ each time the shank passes straight backward. It is wrong in 43 of the 101 frames. The rotation matrix of the joint never looks at the two segment angles separately, only at the rotation between the segments, and it gives a smooth curve with a minimum of $-130^\circ$: the flexion we simulated, negative because flexing the knee turns the shank clockwise in this view.

    Note `np.swapaxes(R, -1, -2)` in the code. With one matrix per frame, `R.T` would transpose the whole array; swapping the last two axes transposes each matrix. The subtraction can also be repaired after the fact with `np.unwrap`, provided the frames are close enough together. The matrix method needs no repair as long as the joint itself turns by less than $180^\circ$.

    **Exercise 3.** In one frame of a walking trial, the hip, knee and ankle joint centres are at $(0, 1.00)$, $(0.25, 0.60)$ and $(0.05, 0.22)$ m, with $X$ forward and $Y$ up. Build the thigh's coordinate system from the hip to the knee and the shank's from the knee to the ankle with `segment_rotation_2d`, and calculate the angle of each segment and of the knee. Is the knee flexed, and by how much?
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _hip, _knee, _ankle = np.array([0, 1.00]), np.array([0.25, 0.60]), np.array([0.05, 0.22])
    _R_thigh = segment_rotation_2d(_hip, _knee)
    _R_shank = segment_rotation_2d(_knee, _ankle)
    _R_knee = _R_thigh.T @ _R_shank  # rotation of the shank relative to the thigh
    _thigh, _shank, _joint = (np.degrees(np.arctan2(_R[1, 0], _R[0, 0])) for _R in (_R_thigh, _R_shank, _R_knee))

    mo.accordion(
        {
            "Solution to Exercise 3": mo.md(
                "```python\n"
                "R_thigh = segment_rotation_2d(hip, knee)\n"
                "R_shank = segment_rotation_2d(knee, ankle)\n"
                "R_knee = R_thigh.T @ R_shank\n"
                "angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # for each R\n"
                "```\n\n"
                f"The thigh is at {_thigh:.1f}° and the shank at {_shank:.1f}°, and the rotation of the shank "
                f"relative to the thigh is {_joint:.1f}°. "
                "The sign is negative because the shank is turned clockwise from the thigh, which in this view, "
                f"with $X$ forward, is flexion: the knee is flexed by {abs(_joint):.1f}°. "
                f"Here the subtraction of the segment angles gives the same {_shank - _thigh:.1f}°, because "
                "neither segment points near the $\\pm 180°$ cut."
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Kinematic chain in a plane (2D)

    The fact that we simply multiplied rotation matrices to obtain the rotation of one segment relative to another is powerful, and it generalizes to any number of segments. Consider a serial kinematic chain with links $1, 2, \dots, n$, and call the base, or the laboratory, link $0$. With the notation of this notebook, where $\mathbf{R}_{ab}$ takes coordinates in $b$ to coordinates in $a$, the rotation matrix of the last link with respect to the base is

    $$
    \mathbf{R}_{0n} = \mathbf{R}_{01}\mathbf{R}_{12} \cdots \mathbf{R}_{n-1,n}
    $$

    where each matrix in the product is the rotation of one link with respect to the previous one. Its inverse, from the base to the last link, is the product of the transposes in the reverse order: $\mathbf{R}_{n0} = \mathbf{R}_{n,n-1}\mathbf{R}_{n-1,n-2} \cdots \mathbf{R}_{2,1}\mathbf{R}_{1,0}$.

    For instance, consider a kinematic chain with two links: link 1 is rotated by $\alpha_1$ with respect to the base (0), and link 2 is rotated by $\alpha_2$ with respect to link 1. With SymPy, the rotation matrices of link 2 with respect to link 1, $\mathbf{R}_{12}$, and of link 1 with respect to the base, $\mathbf{R}_{01}$, are:
    """)
    return


@app.cell
def _():
    import sympy as sym

    return (sym,)


@app.cell
def _(mo, sym):
    a1, a2 = sym.symbols("alpha1 alpha2")

    R12 = sym.Matrix([[sym.cos(a2), -sym.sin(a2)], [sym.sin(a2), sym.cos(a2)]])
    R01 = sym.Matrix([[sym.cos(a1), -sym.sin(a1)], [sym.sin(a1), sym.cos(a1)]])

    mo.md(rf"$$\mathbf{{R_{{12}}}} = {sym.latex(R12)}, \qquad \mathbf{{R_{{01}}}} = {sym.latex(R01)}$$")
    return R01, R12


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rotation matrix of link 2 with respect to the base, $\mathbf{R}_{02}$, is simply $\mathbf{R}_{01}\mathbf{R}_{12}$:
    """)
    return


@app.cell
def _(R01, R12, mo, sym):
    R02 = R01 * R12

    mo.md(rf"$$\mathbf{{R_{{02}}}} = {sym.latex(R02)}$$")
    return (R02,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    which simplifies to:
    """)
    return


@app.cell
def _(R02, mo, sym):
    mo.md(rf"$$\mathbf{{R_{{02}}}} = {sym.latex(sym.simplify(R02))}$$")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A rotation by $\alpha_1 + \alpha_2$, as expected: in a plane, rotations about the axis perpendicular to it simply add up. Which also means the order does not matter. The difference $\mathbf{R}_{01}\mathbf{R}_{12} - \mathbf{R}_{12}\mathbf{R}_{01}$ is:
    """)
    return


@app.cell
def _(R01, R12, sym):
    sym.simplify(R01 * R12 - R12 * R01)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Zero: in a plane, rotation matrices commute. Enjoy it while it lasts. In three-dimensional motion analysis, rotations happen about different axes, and the order of the product changes the result; that needs special care, as [Rigid-body transformations in 3D](https://github.com/BMClab/BMC/blob/master/notebooks/Transformation3D.ipynb) shows. [Kinematic chain](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicChain.ipynb) puts chains of links to work.

    **Challenge 1.** Hold a book in front of you. Turn it $90^\circ$ about a vertical axis and then $90^\circ$ about a horizontal axis pointing to your right. Start again from the same position and do the two rotations in the opposite order. Do you end up in the same place?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Translation and rotation

    Consider now a local coordinate system that is both translated and rotated in relation to the Global coordinate system, with a point described in both, as in the figure below (once again, this is equivalent to describing the translation and rotation between two rigid bodies).

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/transrot2D.png?raw=1" width=345 alt="translation and rotation 2D"/></center><figcaption><center><i>Figure. A point in two-dimensional space represented in two coordinate systems, with one system translated and rotated.</i></center></figcaption></figure>

    The position of point $\mathbf{P}$, originally described in the local coordinate system, described in the Global coordinate system in vector form is

    $$
    \mathbf{P_G} = \mathbf{L_G} + \mathbf{R_{Gl}}\mathbf{P_l}
    $$

    and in matrix form,

    $$
    \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y}
    \end{bmatrix} =
    \begin{bmatrix} \mathbf{L_{X}} \\ \mathbf{L_{Y}} \end{bmatrix} +
    \begin{bmatrix}
    \cos\alpha & -\sin\alpha \\
    \sin\alpha & \hphantom{-}\cos\alpha
    \end{bmatrix} \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y
    \end{bmatrix}
    $$

    This means that we first rotate the coordinates from the local to the Global orientation ("disrotate" the local coordinate system) and then correct for the translation between the two coordinate systems.

    **Before you run the next cell**, predict whether the order matters. Take the example of the three markers above, $\mathbf{L_{Gl}} = [1, 1]$ and a rotation of $90^\circ$, and the point $\mathbf{P_l}=[1,1]$. What would you get by adding the translation first and rotating afterwards?
    """)
    return


@app.cell
def _(LGl, R_basis, np):
    _Pl = np.array([[1, 1]]).T

    print("Rotate, then translate: LGl + RGl Pl   =", (LGl + R_basis @ _Pl).ravel())
    print("Translate, then rotate: RGl (Pl + LGl) =", (R_basis @ (_Pl + LGl)).ravel())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Different answers, and only the first is right. We cannot invert the order: the position of the point is expressed in the local coordinate system, and adding it to the translation, a vector expressed in the Global coordinate system, mixes two coordinate systems. Vectors can only be added once they are expressed in the same coordinate system.

    To find the position of a point in the local coordinate system, given its position in the Global coordinate system, the rotation matrix and the translation vector, we invert the expression above:

    $$
    \begin{aligned}
    \mathbf{P_G} = \mathbf{L_G} + \mathbf{R_{Gl}}\mathbf{P_l} & \implies
    \mathbf{R_{Gl}^{-1}}(\mathbf{P_G} - \mathbf{L_G}) = \mathbf{R_{Gl}^{-1}}\mathbf{R_{Gl}}\mathbf{P_l} \\
    & \implies \mathbf{P_l} = \mathbf{R_{Gl}^{-1}}\left(\mathbf{P_G}-\mathbf{L_G}\right) = \mathbf{R_{Gl}^T}\left(\mathbf{P_G}-\mathbf{L_G}\right)
    \quad \text{or} \quad \mathbf{P_l} = \mathbf{R_{lG}}\left(\mathbf{P_G}-\mathbf{L_G}\right)
    \end{aligned}
    $$

    To go from the Global to the local coordinate system, we first translate and then rotate: the subtraction happens while both vectors are still in Global coordinates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: where under the foot is the centre of pressure?

    A force plate reports the centre of pressure (COP), the point where the resultant ground reaction force acts, in the laboratory's coordinate system. On its own that says little about the foot: the same laboratory coordinates can be under the heel in one step and under the toes in the next. What we want is the COP *relative to the foot*, and that is a Global-to-local transformation.

    Look at the floor from above, with $X$ pointing in the direction of walking and $Y$ to the left. Markers on the heel and on the head of the second metatarsal of a right foot are at $(0.500, 0.100)$ and $(0.740, 0.140)$ m. The foot's coordinate system has its origin at the heel marker, its $x$ axis along the foot, towards the toe marker, and its $y$ axis turned $90^\circ$ counterclockwise from $x$, which for a right foot points medially. The force plate reports the COP at three instants of the stance phase:

    | instant | COP in the laboratory [m] |
    |---|---|
    | heel strike | (0.515, 0.098) |
    | midstance | (0.615, 0.112) |
    | push-off | (0.730, 0.146) |

    **Before you run the next cell**, look only at the laboratory coordinates. At midstance, is the COP on the medial or on the lateral side of the line from the heel marker to the toe marker?
    """)
    return


@app.cell
def _(np):
    _heel = np.array([0.500, 0.100])  # heel marker [m], seen from above
    _toe = np.array([0.740, 0.140])  # marker on the head of the second metatarsal [m]
    _R_foot = segment_rotation_2d(_heel, _toe)
    _length = np.linalg.norm(_toe - _heel)

    print(f"The foot is turned {np.degrees(np.arctan2(_R_foot[1, 0], _R_foot[0, 0])):.1f} degrees from the direction of walking")
    for _instant, _cop in [("heel strike", [0.515, 0.098]), ("midstance", [0.615, 0.112]), ("push-off", [0.730, 0.146])]:
        _x, _y = _R_foot.T @ (np.array(_cop) - _heel)  # Global to local
        print(
            f"{_instant:12s} COP at {100 * _x / _length:4.1f}% of the heel-toe distance,"
            f" {100 * abs(_y):.1f} cm {'medial' if _y > 0 else 'lateral'} to the foot's axis"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the laboratory, the midstance COP, at $Y = 0.112$ m, is to the left of the heel marker, which for a right foot is the medial side. In the foot's coordinate system it is 0.7 cm *lateral*. The foot is turned out by $9.5^\circ$, so its axis also runs to the left, and faster than the COP does. Over the stance phase, the COP moves from just lateral of the axis at the heel to just medial of it under the forefoot, the typical roll of the foot in walking, and that pattern only appears in the foot's coordinate system.

    **Exercise 4.** The same force plate measures the ground reaction force on a runner's foot, now in the sagittal plane: $\mathbf{F_G} = [120, 680]$ N, with $X$ forward and $Y$ up. The shank's coordinate system has its origin at the ankle, 9 cm above the floor, its $x$ axis along the shank from the ankle to the knee, inclined $20^\circ$ forward from the vertical (that is, at $70^\circ$ from $X$), and its $y$ axis turned $90^\circ$ counterclockwise from $x$. Find the components of the force along and across the shank. Does the position of the ankle enter the calculation?
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _F_G = np.array([120.0, 680.0])  # ground reaction force [N]
    _R_shank = rotation_2d(np.radians(70))
    _F_l = _R_shank.T @ _F_G  # Global to local: rotation only

    mo.accordion(
        {
            "Solution to Exercise 4": mo.md(
                r"A force is a vector, not a position: it has a magnitude and a direction, but no origin to subtract. "
                r"Only the rotation enters, $\mathbf{F_l} = \mathbf{R_{Gl}^T}\mathbf{F_G}$, and the 9 cm of the "
                "ankle play no role. (The translation cancels for any vector that is a difference of two positions, "
                "and velocities and accelerations are such differences divided by time.)\n\n"
                f"The force is {_F_l[0]:.1f} N along the shank, compressing it, and {_F_l[1]:.1f} N across it, "
                "towards the posterior side of the shank (the local $y$ axis points backward and up). "
                f"Its magnitude, {np.linalg.norm(_F_l):.1f} N, is the same in both coordinate systems, as it must be "
                "for a rotation. Almost all of the force runs along the shank because the force, at "
                f"{np.degrees(np.arctan2(_F_G[1], _F_G[0])):.1f}° from $X$, points within "
                f"{np.degrees(np.arctan2(_F_G[1], _F_G[0])) - 70:.0f}° of the shank's axis."
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transformation matrix

    The translation and the rotation can be combined in a single matrix, the transformation matrix (also called the homogeneous transformation matrix), by adding a third coordinate equal to 1 to every point:

    $$
    \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y} \\
    1
    \end{bmatrix} =
    \begin{bmatrix}
    \cos\alpha & -\sin\alpha & \mathbf{L_{X}} \\
    \sin\alpha & \hphantom{-}\cos\alpha  & \mathbf{L_{Y}} \\
    0 & 0 & 1
    \end{bmatrix} \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y \\
    1
    \end{bmatrix}
    $$

    or simply

    $$
    \mathbf{P_G} = \mathbf{T_{Gl}}\mathbf{P_l}
    $$

    Multiply it out and the first two rows give back $\mathbf{P_G} = \mathbf{L_G} + \mathbf{R_{Gl}}\mathbf{P_l}$, with the rotation applied first, as it must be; the third row just says $1 = 1$. These are the 3×3 matrices in the figure of affine transformations at the top.

    The inverse operation, expressing the position in the local coordinate system in terms of the Global coordinate system, is

    $$
    \mathbf{P_l} = \mathbf{T_{Gl}^{-1}}\mathbf{P_G}
    $$

    But $\mathbf{T_{Gl}}$ is not orthonormal, so its inverse is not its transpose. From the inverse derived above, $\mathbf{P_l} = \mathbf{R_{Gl}^T}\mathbf{P_G} - \mathbf{R_{Gl}^T}\mathbf{L_G}$, so in block form

    $$
    \begin{bmatrix}
    \mathbf{P}_x \\
    \mathbf{P}_y \\
    1
    \end{bmatrix} =
    \underbrace{\begin{bmatrix}
    \mathbf{R^{T}_{Gl}} & - \mathbf{R^{T}_{Gl}}\mathbf{L_{G}} \\
    0 \quad 0 & 1
    \end{bmatrix}}_{\mathbf{T_{Gl}^{-1}}} \begin{bmatrix}
    \mathbf{P_X} \\
    \mathbf{P_Y} \\
    1
    \end{bmatrix}
    $$

    where the top-left block is the 2×2 transposed rotation matrix and the top-right block is a 2×1 column. Let's write both as functions and check them with the example of the three markers:
    """)
    return


@app.function
def transformation_2d(R, L):
    """Homogeneous transformation matrix T_Gl from the rotation matrix R_Gl
    (2x2) and the translation L_G (the position of the local origin)."""
    import numpy as np

    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = np.ravel(L)
    return T


@app.function
def inverse_transformation_2d(T):
    """Inverse of a homogeneous transformation matrix, without a general
    matrix inversion: the rotation is transposed and the translation undone."""
    import numpy as np

    R = T[:2, :2]
    L = T[:2, 2]
    T_inv = np.eye(3)
    T_inv[:2, :2] = R.T
    T_inv[:2, 2] = -R.T @ L
    return T_inv


@app.cell
def _(LGl, R_basis, np):
    _T = transformation_2d(R_basis, LGl)
    _Pl = np.array([1, 1, 1])  # the point [1, 1] with the extra coordinate 1
    _PG = _T @ _Pl
    print("Transformation matrix TGl:\n", _T)
    print("\nPG = TGl Pl =", _PG)

    _T_inv = inverse_transformation_2d(_T)
    print("\nInverse from the formula:\n", _T_inv)
    print("Same as np.linalg.inv(TGl)?", np.allclose(_T_inv, np.linalg.inv(_T)))
    print("\nBack to local coordinates, T^-1 PG =", _T_inv @ _PG)
    print("With the transpose instead,   T^T PG =", _T.T @ _PG)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The formula agrees with the general matrix inverse, and takes the point back to $[1, 1]$. The transpose, which worked for the rotation matrix alone, sends it to $[2, 0]$ and spoils the extra coordinate, which should stay equal to 1.

    **Exercise 5.** A local coordinate system is rotated by $90^\circ$ counterclockwise and has its origin at $[2, 1]$ in the Global coordinate system. Write $\mathbf{T_{Gl}}$, find $\mathbf{T_{Gl}^{-1}}$ with the block formula above (no general matrix inversion), and use it to find the local coordinates of the Global point $[2, 3]$. Check your answer by transforming it back.

    /// details | Solution to Exercise 5
    With $\cos 90^\circ = 0$ and $\sin 90^\circ = 1$,

    $$
    \mathbf{T_{Gl}} = \begin{bmatrix} 0 & -1 & 2 \\ 1 & \hphantom{-}0 & 1 \\ 0 & \hphantom{-}0 & 1 \end{bmatrix}
    $$

    For the inverse, the rotation block is transposed, and the translation becomes

    $$
    -\mathbf{R_{Gl}^T}\mathbf{L_G} = -\begin{bmatrix} \hphantom{-}0 & 1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = -\begin{bmatrix} \hphantom{-}1 \\ -2 \end{bmatrix} = \begin{bmatrix} -1 \\ \hphantom{-}2 \end{bmatrix}
    \quad \Longrightarrow \quad
    \mathbf{T_{Gl}^{-1}} = \begin{bmatrix} \hphantom{-}0 & 1 & -1 \\ -1 & 0 & \hphantom{-}2 \\ \hphantom{-}0 & 0 & \hphantom{-}1 \end{bmatrix}
    $$

    The Global point $[2, 3, 1]$ goes to $[0 \cdot 2 + 1 \cdot 3 - 1,\; -1 \cdot 2 + 0 \cdot 3 + 2,\; 1] = [2, 0, 1]$: the local point $[2, 0]$, 2 along the local $x$ axis. Back again, $\mathbf{T_{Gl}}\,[2, 0, 1] = [0 - 0 + 2,\; 2 + 0 + 1,\; 1] = [2, 3, 1]$. In the sketch, the local $x$ axis points along the Global $Y$ axis from $[2, 1]$, and 2 along it is $[2, 3]$.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: where is the hand?

    Transformation matrices can be chained like rotation matrices, and this is where they earn their keep. Consider a planar model of the arm, seen from the side. The upper arm, 30 cm long, is turned about the shoulder by $\theta_1 = 30^\circ$ from the forward horizontal; the forearm, 25 cm long, is turned about the elbow by $\theta_2 = 60^\circ$ relative to the upper arm. Attach a coordinate system to each segment, with its origin at the proximal joint and its $x$ axis along the segment. Then:

    - the upper arm's coordinate system (1) is the shoulder's (0) rotated by $\theta_1$, with the same origin, so $\mathbf{T_{01}}$ has the rotation $\theta_1$ and the translation $[0, 0]$;
    - the forearm's coordinate system (2) is the upper arm's rotated by $\theta_2$, with its origin at the elbow, which is at $[0.30, 0]$ *in the upper arm's coordinates*, so $\mathbf{T_{12}}$ has the rotation $\theta_2$ and the translation $[0.30, 0]$;
    - the hand is at $[0.25, 0]$ in the forearm's coordinates.

    The hand, in the shoulder's coordinate system, is then

    $$
    \mathbf{P_0} = \mathbf{T_{01}}\,\mathbf{T_{12}}\,\mathbf{P_2}
    $$

    Each matrix needs only what is simple in its own coordinate system: the angle between neighbouring segments and the length of a segment.

    **Before you run the next cell**, predict where the hand is. (Hint: at what angle from the horizontal does the forearm point?)
    """)
    return


@app.cell
def _(np, plt):
    _T01 = transformation_2d(rotation_2d(np.radians(30)), [0, 0])  # upper arm in the shoulder's system
    _T12 = transformation_2d(rotation_2d(np.radians(60)), [0.30, 0])  # forearm in the upper arm's system
    _hand_2 = np.array([0.25, 0, 1])  # hand in the forearm's system, with the extra coordinate 1

    _T02 = _T01 @ _T12
    _elbow = _T01 @ [0.30, 0, 1]
    _hand = _T02 @ _hand_2
    _hand_wrong = _T12 @ _T01 @ _hand_2
    print("Elbow:", _elbow[:2].round(3), "m;  hand:", _hand[:2].round(3), "m")
    print("Check with trigonometry:", np.round([0.30 * np.cos(np.radians(30)) + 0.25 * np.cos(np.radians(90)),
                                               0.30 * np.sin(np.radians(30)) + 0.25 * np.sin(np.radians(90))], 3))
    print("Hand with the product in the wrong order, T12 T01:", _hand_wrong[:2].round(3), "m")
    print("Same rotation part in both orders?", np.allclose(_T02[:2, :2], (_T12 @ _T01)[:2, :2]))

    _, _ax = plt.subplots(figsize=(5, 5))
    _ax.plot([0, _elbow[0], _hand[0]], [0, _elbow[1], _hand[1]], "o-", color="0.5", linewidth=6, markersize=10, alpha=0.6)
    for _T in (np.eye(3), _T01, _T02):
        _origin = _T[:2, 2]
        _ax.quiver(*_origin, *(0.08 * _T[:2, 0]), angles="xy", scale_units="xy", scale=1, color="tab:red", width=0.008)
        _ax.quiver(*_origin, *(0.08 * _T[:2, 1]), angles="xy", scale_units="xy", scale=1, color="tab:green", width=0.008)
    _ax.plot(*_hand_wrong[:2], "x", color="tab:purple", markersize=12, mew=3, label="hand with the wrong order")
    _ax.plot([], [], color="tab:red", linewidth=3, label="local x axes")
    _ax.plot([], [], color="tab:green", linewidth=3, label="local y axes")
    _ax.set_aspect("equal")
    _ax.margins(0.2)
    _ax.set_xlabel("X [m]")
    _ax.set_ylabel("Y [m]")
    _ax.set_title("Shoulder, elbow and hand")
    _ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The forearm points straight up, at $30^\circ + 60^\circ = 90^\circ$ from the horizontal, so the hand is 25 cm directly above the elbow, at $(0.260, 0.400)$ m, which trigonometry confirms. The figure shows the three coordinate systems we used: two at the shoulder (the laboratory's and the upper arm's) and one at the elbow (the forearm's).

    Now the product in the wrong order. Its rotation part is the same, a rotation by $90^\circ$, because in a plane rotations commute. But the hand lands at $(0.300, 0.250)$ m, 39 cm from the shoulder instead of 48 cm. *Transformation* matrices do not commute, even in a plane, because each translation is written in the coordinate system of the segment before it: $[0.30, 0]$ means "30 cm along the upper arm" only when it is applied after $\mathbf{T_{01}}$ has turned the axes. [Kinematic chain](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicChain.ipynb) builds on exactly this product.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back to the knee

    We now have everything needed to locate the knee. The plan has two steps, one in each direction:

    1. **Calibration (Global to local).** In the standing trial, build the thigh's coordinate system from its two markers, and express the knee joint centre in it: $\mathbf{P_l} = \mathbf{R_{Gl}^T}(\mathbf{P_G} - \mathbf{L_G})$. Because the thigh is rigid, these local coordinates never change.
    2. **Reconstruction (local to Global).** In every frame of the movement, build the thigh's coordinate system again from the same two markers, and bring the knee back to the laboratory: $\mathbf{P_G} = \mathbf{L_G} + \mathbf{R_{Gl}}\mathbf{P_l}$.

    The thigh's coordinate system has its origin at the trochanter marker, its $x$ axis pointing from there to the thigh marker, and its $y$ axis turned $90^\circ$ counterclockwise from $x$: exactly the recipe of `segment_rotation_2d`, written for the kick example above.
    """)
    return


@app.cell
def _(np):
    # Calibration trial, standing [m]
    m1_cal = np.array([0.000, 0.900])  # marker on the greater trochanter
    m2_cal = np.array([0.020, 0.700])  # marker on the lateral thigh
    knee_cal = np.array([0.010, 0.480])  # knee joint centre, located in this trial only

    R_cal = segment_rotation_2d(m1_cal, m2_cal)
    knee_local = R_cal.T @ (knee_cal - m1_cal)  # Global to local

    print("Thigh angle in the calibration:", round(float(np.degrees(np.arctan2(R_cal[1, 0], R_cal[0, 0]))), 1), "degrees")
    print("Knee in the thigh's coordinate system:", knee_local, "m")
    return R_cal, knee_cal, knee_local, m1_cal, m2_cal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The thigh points almost straight down, at $-84.3^\circ$ from the forward horizontal, and the knee is 41.9 cm along the thigh's $x$ axis and 3.2 cm to the negative side of its $y$ axis: slightly behind the line through the two markers, which sit on the side of the thigh, not on the femur.

    Now the movement. To have a known answer to check against, the movement markers in the table at the top were generated from the calibration: the thigh was turned by $60^\circ$ counterclockwise (hip flexion, with $X$ pointing forward), the trochanter marker was moved to $(0.150, 0.800)$, and the thigh marker was rounded to the millimetre, as a motion capture system would report it. The same transformation gives the true position of the knee, which we will now pretend not to know.
    """)
    return


@app.cell
def _(knee_cal, m1_cal, m2_cal, np):
    _R_true = rotation_2d(np.radians(60))
    m1_mov = np.array([0.150, 0.800])
    m2_mov = np.round(m1_mov + _R_true @ (m2_cal - m1_cal), 3)  # reported to the millimetre
    knee_true = m1_mov + _R_true @ (knee_cal - m1_cal)  # the answer we pretend not to know

    print("Markers in the movement trial:", m1_mov, m2_mov)
    return knee_true, m1_mov, m2_mov


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Before you run the next cell**, go back to your guess in Guiding question 0.3, and make it a number: where is the knee?
    """)
    return


@app.cell
def _(R_cal, knee_cal, knee_local, knee_true, m1_cal, m1_mov, m2_mov, np):
    R_mov = segment_rotation_2d(m1_mov, m2_mov)
    knee_mov = m1_mov + R_mov @ knee_local  # local to Global

    _R_rel = R_cal.T @ R_mov  # rotation of the thigh from the calibration to the movement
    print("Thigh rotation since the calibration:", round(float(np.degrees(np.arctan2(_R_rel[1, 0], _R_rel[0, 0]))), 1), "degrees")
    print("Reconstructed knee:", knee_mov.round(3), "m")
    print(f"Distance from the true knee: {1000 * np.linalg.norm(knee_mov - knee_true):.1f} mm")
    print(
        f"Knee to trochanter marker: {np.linalg.norm(knee_mov - m1_mov):.3f} m"
        f" (standing: {np.linalg.norm(knee_cal - m1_cal):.3f} m)"
    )
    return R_mov, knee_mov


@app.cell
def _(R_cal, R_mov, knee_cal, knee_mov, m1_cal, m1_mov, m2_cal, m2_mov, plt):
    _, _ax = plt.subplots(figsize=(7, 6))
    for _m1, _m2, _knee, _R, _label, _color in [
        (m1_cal, m2_cal, knee_cal, R_cal, "standing (calibration)", "0.5"),
        (m1_mov, m2_mov, knee_mov, R_mov, "movement (reconstructed)", "tab:blue"),
    ]:
        _ax.plot([_m1[0], _knee[0]], [_m1[1], _knee[1]], color=_color, linewidth=8, alpha=0.3, solid_capstyle="round")
        _ax.plot([_m1[0], _m2[0]], [_m1[1], _m2[1]], "o", color=_color, markersize=9)
        _ax.plot(*_knee, "*", color=_color, markersize=20, label=f"knee, {_label}")
        _ax.quiver(*_m1, *(0.12 * _R[:, 0]), angles="xy", scale_units="xy", scale=1, color="tab:red", width=0.006)
        _ax.quiver(*_m1, *(0.12 * _R[:, 1]), angles="xy", scale_units="xy", scale=1, color="tab:green", width=0.006)
    _ax.plot([], [], color="tab:red", linewidth=3, label=r"$\hat{\mathbf{e}}_x$ of the thigh")
    _ax.plot([], [], color="tab:green", linewidth=3, label=r"$\hat{\mathbf{e}}_y$ of the thigh")
    _ax.set_aspect("equal")
    _ax.margins(0.15)
    _ax.set_xlabel("X [m]")
    _ax.set_ylabel("Y [m]")
    _ax.set_title("Thigh markers (dots) and knee joint centre (stars)")
    _ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The knee is at $(0.518, 0.598)$ m: 37 cm in front of the trochanter marker and 20 cm below it, with the thigh turned by $59.9^\circ$. The reconstruction misses the true position by less than a millimetre, and that small miss comes entirely from rounding the thigh marker to the millimetre. The distance from the knee to the trochanter marker is the same 42 cm as in the standing trial, as it must be for a rigid transformation.

    Two transformations, one in each direction, located a point that no camera could see. This is how the anatomical coordinate systems of [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb) are rebuilt in every frame of a movement from technical markers, and in three dimensions the procedure is the same, with 3×3 rotation matrices.

    **Guiding questions 2.**

    1. The thigh's coordinate system had its origin at the trochanter marker. If the origin were at the midpoint between the two markers, which numbers would change: the local coordinates of the knee, the reconstructed knee, or both?
    2. The recipe used only two markers. Why is that enough in a plane? What would a third marker add?
    3. The rotation of the thigh came out as $59.9^\circ$, not $60^\circ$. Where did the missing $0.1^\circ$ go?

    **Exercise 6.** In another frame of the same movement, the trochanter marker is at $(0.40, 0.85)$ m and the thigh marker at $(0.55, 0.72)$ m. Where is the knee, and how much has the thigh rotated since the calibration? Measure the distance between the two markers and compare it with the standing trial. What could the difference mean, and does it change the reconstructed knee?
    """)
    return


@app.cell(hide_code=True)
def _(R_cal, knee_local, m1_cal, m2_cal, mo, np):
    _m1, _m2 = np.array([0.40, 0.85]), np.array([0.55, 0.72])
    _R = segment_rotation_2d(_m1, _m2)
    _knee = _m1 + _R @ knee_local  # local to Global
    _R_rel = R_cal.T @ _R  # rotation of the thigh since the calibration
    _rotation = np.degrees(np.arctan2(_R_rel[1, 0], _R_rel[0, 0]))
    _d_now = np.linalg.norm(_m2 - _m1)
    _d_cal = np.linalg.norm(m2_cal - m1_cal)

    mo.accordion(
        {
            "Solution to Exercise 6": mo.md(
                "Build the thigh's rotation matrix from the two markers of this frame and bring the knee's local "
                r"coordinates, found in the calibration, back to the laboratory: "
                r"$\mathbf{P_G} = \mathbf{L_G} + \mathbf{R_{Gl}}\mathbf{P_l}$, with `knee_local` as $\mathbf{P_l}$. "
                f"The knee is at ({_knee[0]:.3f}, {_knee[1]:.3f}) m, and the thigh has turned by {_rotation:.1f}° "
                "since the calibration.\n\n"
                f"The markers are {100 * _d_now:.2f} cm apart, against {100 * _d_cal:.2f} cm standing: "
                f"{1000 * (_d_cal - _d_now):.1f} mm shorter. A rigid thigh cannot shorten, so the markers have moved "
                "on the skin, or the measurement is off. `segment_rotation_2d` uses only the *direction* from one "
                "marker to the other, so the change of length goes unnoticed, and the knee is still "
                f"{100 * np.linalg.norm(_knee - _m1):.1f} cm from the trochanter marker. But a marker that slides "
                "along the thigh may also slide across it, and that would tilt the axes and move the knee, as the "
                "simulation below shows. A change in the distance between markers is a warning sign worth checking "
                "in every frame."
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How far can you trust the reconstructed knee?

    Real markers are not off by a rounding error; they are off by a millimetre or so in every frame, and the noise of each frame is different. Let's add random noise with a standard deviation of 1 mm to each coordinate of the two markers in the movement trial, reconstruct the knee 5000 times, and measure how far it lands from the true knee. Then repeat with the thigh marker placed halfway, 10 cm from the trochanter marker instead of 20 cm.

    **Before you run the next cell**, predict: with 1 mm of noise in each marker, is the typical error of the knee smaller than 1 mm, about 1 mm, or larger? And with the markers closer together?
    """)
    return


@app.cell
def _(knee_cal, knee_true, m1_cal, m1_mov, m2_cal, np):
    _rng = np.random.default_rng(0)
    _n, _sigma = 5000, 0.001  # number of simulated frames, 1 mm of noise
    _R_true = rotation_2d(np.radians(60))

    for _label, _fraction in [("20 cm", 1.0), ("10 cm", 0.5)]:
        _m2c = m1_cal + _fraction * (m2_cal - m1_cal)  # thigh marker in the calibration
        _knee_local = segment_rotation_2d(m1_cal, _m2c).T @ (knee_cal - m1_cal)
        _m2m = m1_mov + _R_true @ (_m2c - m1_cal)  # its true position in the movement
        _m1_noisy = m1_mov + _rng.normal(0, _sigma, (_n, 2))
        _m2_noisy = _m2m + _rng.normal(0, _sigma, (_n, 2))
        _knee = _m1_noisy + segment_rotation_2d(_m1_noisy, _m2_noisy) @ _knee_local
        _err = 1000 * np.linalg.norm(_knee - knee_true, axis=1)  # mm
        _err_marker = 1000 * np.linalg.norm(_m2_noisy - _m2m, axis=1)
        print(
            f"Markers {_label} apart: knee error median {np.median(_err):.1f} mm,"
            f" 95th percentile {np.percentile(_err, 95):.1f} mm"
            f" (error of each marker: median {np.median(_err_marker):.1f} mm)"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The knee is reconstructed worse than the markers are measured: a typical error of 2 mm with the markers 20 cm apart, against 1.2 mm for each marker, and almost twice that, 3.8 mm, with the markers 10 cm apart. With the closer markers, the error exceeds 1 cm in one frame in twenty.

    Two things are at work. The noise tilts the thigh's $x$ axis a little, by an angle of the order of the marker error divided by the distance between the markers; halve that distance and the tilt doubles. And the knee sits 42 cm from the origin, so a small tilt of the axes swings it through an arc proportional to that lever arm. A point reconstructed far from its markers inherits their errors magnified. That is one more reason to spread the markers of a segment well apart, as [Determination of a coordinate system](https://github.com/BMClab/BMC/blob/master/notebooks/CoordinateSystem.ipynb) found for the axes themselves.

    **Challenge 2.** Move the origin of the thigh's coordinate system to the midpoint of the two markers (Guiding question 2.1), and repeat the simulation. Does the error of the knee change? Explain why, in terms of the lever arm.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to the coin of Challenge 0. When you turned the paper, which coordinates of the coin changed, those on the paper's axes or those on the table's? Which rotation matrix, $\mathbf{R_{Gl}}$ or $\mathbf{R_{lG}}$, converts the paper's coordinates into the table's?
    2. A camera measures a marker in the laboratory, and you want its coordinates in the segment's coordinate system. Do you multiply by $\mathbf{R_{Gl}}$ or by its transpose? Do you subtract the translation before or after?
    3. Why is the inverse of a rotation matrix its transpose, while the inverse of a transformation matrix is not?
    4. The columns of $\mathbf{R_{Gl}}$ are the local versors in Global coordinates. What are its rows?
    5. Two rotations in a plane give the same result in either order. Give a physical reason why that cannot be true in three dimensions.
    6. The reconstructed knee was about twice as uncertain as each marker. Name two choices in the design of an experiment that would reduce that error.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. A local coordinate system is rotated 30° clockwise in relation to the Global reference system.<br>
       A. Determine the matrices for rotating one coordinate system to the other (two-dimensional).<br>
       B. What are the coordinates of the point [1, 1] (local coordinate system) in the Global coordinate system?<br>
       C. And if this point is in the Global coordinate system and we want its coordinates in the local coordinate system?<br>
       D. Consider that the local coordinate system, besides the rotation, is also translated by [2, 2]. What are the matrices for rotation, translation, and transformation from one coordinate system to the other (two-dimensional)?<br>
       E. Repeat B and C considering this translation.

    2. Consider a local coordinate system U rotated 45° clockwise in relation to the Global reference system, and another local coordinate system V rotated 45° clockwise in relation to the local reference system U.<br>
       A. Determine the rotation matrices of all possible transformations between the coordinate systems.<br>
       B. For the point [1, 1] in the coordinate system U, what are its coordinates in the coordinate system V and in the Global coordinate system?

    3. Using the rotation matrix, deduce the new coordinates of a square with vertices [0, 0], [1, 0], [1, 1] and [0, 1] when rotated by 0°, 45°, 90°, 135° and 180° (always clockwise).

    4. Solve problem 2 of [Angular kinematics in a plane (2D)](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicsAngular2D.ipynb), but now using the concept of two-dimensional transformations.

    5. Write Python code to solve the problem in [https://leetcode.com/problems/rotate-image/](https://leetcode.com/problems/rotate-image/).

    6. Rotate a triangle with vertices A(0, 0), B(1, 1) and C(5, 2) by an angle of 45° with respect to the origin. (From a [numerical example at studyresearch.in](https://web.archive.org/web/20250223024131/https://studyresearch.in/2019/12/14/numerical-example-of-rotation-in-2d-transformation/), archived copy.)

    7. Rotate a triangle with vertices A(0, 0), B(1, 1) and C(5, 2) by an angle of 45° with respect to the point P(−1, −1). (From the same [numerical example](https://web.archive.org/web/20250223024131/https://studyresearch.in/2019/12/14/numerical-example-of-rotation-in-2d-transformation/).)

    8. In the knee example, define the thigh's coordinate system with its origin at the midpoint between the two markers, keeping the same axes. Calculate the local coordinates of the knee in the calibration and the reconstructed knee in the movement. Which numbers change, and which do not?

    9. Show that, if the axes of a segment's coordinate system are tilted by a small angle $\delta\theta$ (in radians), a point reconstructed at a distance $d$ from the origin moves by about $d\,\delta\theta$. Use this to estimate, without simulation, how the error of the reconstructed knee changes when the distance between the two thigh markers is halved, and compare with the simulation in this notebook.

    10. In the centre-of-pressure example, the same person walks again with the foot turned out further: the heel marker is at (0.500, 0.100) m and the toe marker at (0.729, 0.183) m. The COP at the same three instants is (0.512, 0.098), (0.605, 0.130) and (0.718, 0.190) m.<br>
        a. Calculate the angle of the foot and the COP in the foot's coordinate system.<br>
        b. Compare with the example. Which description, in the laboratory or in the foot, would you use to compare the two trials, and why?

    11. A planar model of the leg, seen from the side, has a thigh of 0.42 m, a shank of 0.40 m and a foot of 0.20 m from the ankle to the toes. At one instant, the thigh is at $-70^\circ$ from the forward horizontal, the shank is turned by $-40^\circ$ relative to the thigh, and the foot by $+100^\circ$ relative to the shank.<br>
        a. With one transformation matrix per segment, as in the example of the arm, find the positions of the knee, the ankle and the toes relative to the hip, and check them with trigonometry.<br>
        b. Use the inverse of the product of the matrices to find the position of the hip in the foot's coordinate system.

    12. Gait data of the two sides are often plotted together by mirroring one side, flipping the sign of $X$. Mirror the thigh markers of the knee example, calculate the thigh's angle with `segment_rotation_2d`, and compare it with the original $-84.3^\circ$.<br>
        a. Show that mirroring turns a segment angle $\theta$ into $180^\circ - \theta$, and that it changes the sign of the angle between two segments.<br>
        b. Propose how to report joint angles so that the mirrored and the original side can be compared directly.

    13. Repeat the simulation of the kick example, and reconstruct in every frame the position of a point of the foot, treated as rigidly attached to the shank, at the local coordinates $[0.45, 0.10]$ m in the shank's coordinate system (origin at the knee). Plot its path in the laboratory and find the frame in which it is highest. Use arrays of matrices, without a loop over the frames.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - Read pages 751–758 of chapter 16 of [Ruina and Pratap's book](http://ruina.tam.cornell.edu/Book/index.html) about the rotation matrix.
    - Read section 2.8 of Rade's book about the rotation matrix applied to kinematic chains.
    - [Rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) - Wikipedia.
    - [Rigid-body transformations in 3D](https://github.com/BMClab/BMC/blob/master/notebooks/Transformation3D.ipynb): the same ideas in three dimensions, where rotations no longer commute.
    - [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb) and [Determination of a coordinate system](https://github.com/BMClab/BMC/blob/master/notebooks/CoordinateSystem.ipynb): how the bases used here are built from markers, and how marker errors propagate to them.
    - [Kinematic chain](https://github.com/BMClab/BMC/blob/master/notebooks/KinematicChain.ipynb): products of transformations along a chain of links.

    ### Video lectures on the Internet

    - [Linear transformation examples: Rotations in R2](https://www.khanacademy.org/math/linear-algebra/matrix-transformations/lin-trans-examples/v/linear-transformation-examples-rotations-in-r2) - Khan Academy
    - [Linear transformations and matrices | Chapter 3, Essence of linear algebra](https://youtu.be/kYB8IZa5AuE) - 3Blue1Brown
    - [Linear Transformations and Their Matrices](https://youtu.be/Ts3o2I8_Mxc) - MIT OpenCourseWare
    - [Rotation Matrices](https://youtu.be/4srS0s1d9Yw)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Rade D (2017) [Cinemática e Dinâmica para Engenharia](https://www.grupogen.com.br/e-book-cinematica-e-dinamica-para-engenharia). Grupo GEN.
    - Robertson G, Caldwell G, Hamill J, Kamen G (2013) [Research Methods in Biomechanics](http://books.google.com.br/books?id=gRn8AAAAQBAJ). 2nd Edition. Human Kinetics.
    - Ruina A, Pratap R (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.
    - Winter DA (2009) [Biomechanics and Motor Control of Human Movement](http://books.google.com.br/books?id=_bFHL08IWfwC). 4th edition. Hoboken, USA: Wiley.
    - Zatsiorsky VM (1997) [Kinematics of Human Motion](http://books.google.com.br/books/about/Kinematics_of_Human_Motion.html?id=Pql_xXdbrMcC&redir_esc=y). Champaign: Human Kinetics.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
