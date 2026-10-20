import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Determination of a coordinate system

    > Marcos Duarte,
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br),
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Put three markers on a body segment, record them with a motion capture system, and you get nine numbers per frame. Nine numbers are not an orientation. To say which way the thigh is pointing, or how far the shank has turned, you need a coordinate system attached to the segment: an origin and three perpendicular unit vectors that move with it.

    This notebook builds that coordinate system from three markers, and then asks the questions that decide whether you can trust it. Is it really orthonormal, and right-handed? Does it move with the segment, and only with the segment? And when every marker is off by a millimetre, as real markers are, how far off are its axes?

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it. Make the prediction; a wrong guess that the code then corrects teaches more than a right one.

    The **Exercises** come with solutions, folded away under each exercise. Open a solution only after you have an answer of your own; the solutions of the computational exercises are produced by cells whose code you can reveal and run.

    You will need the vector operations from [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb): the norm, the dot product and the cross product. For the wider picture (what frames of reference are, why inertial frames matter, and how anatomical conventions fix the direction of each axis), see [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb).

    **Challenge 0.** Imagine that you have three markers to put on your own forearm. Before reading on, decide where you would put them and write down why. Keep your answer nearby; the checkpoint questions ask you to come back to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three markers on a shank

    A rigid plastic plate carrying three markers is strapped to the side of a runner's shank. This arrangement, called a *marker cluster*, is one of the most common in three-dimensional motion analysis. In every frame of the recording, the motion capture system reports nine numbers: the X, Y and Z coordinates of each marker in the laboratory.

    None of those nine numbers is the orientation of the shank. Every one of them changes when the shank turns, but also when the runner simply moves forward, and no single one of them says which way the shank is pointing. To get the orientation, we have to turn the three positions into a coordinate system that belongs to the shank.

    **Guiding questions 0.**

    1. Why three markers? Would two be enough to know the orientation of the shank? Which rotation of the shank would two markers miss?
    2. Why must the three markers not lie on a straight line?
    3. If you could make the plate any size you liked, would you make it small or large? What stops you from making it as large as possible?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the numerical work and Matplotlib draws the figures. The function `CCS`, from the `functions` folder of this repository, draws coordinate systems in three dimensions; it is loaded where it is first needed.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frames of reference and coordinate systems

    In mechanics, a [frame of reference](http://en.wikipedia.org/wiki/Frame_of_reference) is what we choose to describe the motion of a body with respect to. In this frame of reference we define a [coordinate system](http://en.wikipedia.org/wiki/Coordinate_system), a set of axes along which we measure the motion of the body (although frame of reference and coordinate system are often used interchangeably).

    In biomechanics, we use different coordinate systems for convenience, and call them global, laboratory, local, anatomical or technical reference frames or coordinate systems. For example, in a standard gait analysis we define a global or laboratory coordinate system, and a different coordinate system for each segment of the body, so that the motion of a segment can be described in relation to the anatomical axes of another segment. To define this anatomical coordinate system, we place markers on anatomical landmarks of each segment. We may also use other markers on the segment, technical markers, to improve the motion analysis, and then we also have to define a technical coordinate system for each segment.

    As we perceive the surrounding space as three-dimensional, a convenient coordinate system to use is the [Cartesian coordinate system](http://en.wikipedia.org/wiki/Cartesian_coordinate_system), with three orthogonal axes in [Euclidean space](http://en.wikipedia.org/wiki/Euclidean_space). From [linear algebra](http://en.wikipedia.org/wiki/Linear_algebra), a set of linearly independent unit vectors that can represent any vector via [linear combination](http://en.wikipedia.org/wiki/Linear_combination) is called a <a href="http://en.wikipedia.org/wiki/Basis_(linear_algebra)">basis</a>; when the vectors are also mutually orthogonal, it is an orthonormal basis. The figure below shows a point, its position vector in the Cartesian coordinate system, and the versors (unit vectors) of the basis of this coordinate system.

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/vector3Dijk.png?raw=1" width=350 alt="versors of the Cartesian basis"/></center><figcaption><center><i>Figure. Representation of a point $\mathbf{P}$ and its position vector $\overrightarrow{\mathbf{a}}$ in a Cartesian coordinate system. The versors $\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}$ form a basis for this coordinate system and are usually drawn in the colour sequence RGB (red, green, blue) for easier visualization.</i></center></figcaption></figure>

    In the Cartesian coordinate system itself, the versors of this basis have the coordinates

    $$
    \hat{\mathbf{i}} = \begin{bmatrix}1\\0\\0 \end{bmatrix}, \quad \hat{\mathbf{j}} = \begin{bmatrix}0\\1\\0 \end{bmatrix}, \quad \hat{\mathbf{k}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
    $$

    and, in the notation of the figure, the position vector is

    $$
    \overrightarrow{\mathbf{a}} = a_x\hat{\mathbf{i}} + a_y\hat{\mathbf{j}} + a_z\hat{\mathbf{k}}
    $$

    Every marker position a motion capture system reports is a vector like this one, written in the laboratory's basis. What we want is a second basis, attached to the segment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Definition of a basis

    The mathematical problem of determining a coordinate system is to find a basis and an origin for it; a basis on its own is only a set of directions, with no origin. In three-dimensional space, given the coordinates of three noncollinear points (points that do not all lie on the same line), $\overrightarrow{\mathbf{m}}_1, \overrightarrow{\mathbf{m}}_2, \overrightarrow{\mathbf{m}}_3$, which would be the positions of markers captured in a motion analysis session, a basis can be found in four steps:

    1. First axis, $\overrightarrow{\mathbf{v}}_1$: the vector $\overrightarrow{\mathbf{m}}_2-\overrightarrow{\mathbf{m}}_1$ (or any other difference between two of the points).
    2. Second axis, $\overrightarrow{\mathbf{v}}_2$: the cross product between $\overrightarrow{\mathbf{v}}_1$ and $\overrightarrow{\mathbf{m}}_3-\overrightarrow{\mathbf{m}}_1$ (or $\overrightarrow{\mathbf{m}}_3-\overrightarrow{\mathbf{m}}_2$).
    3. Third axis, $\overrightarrow{\mathbf{v}}_3$: the cross product between $\overrightarrow{\mathbf{v}}_1$ and $\overrightarrow{\mathbf{v}}_2$.
    4. Make every axis a versor by dividing it by its norm.

    Each step uses the cross product for what it is good at: producing a vector perpendicular to two others. The second axis is perpendicular to the plane of the three points, and the third lies in that plane, perpendicular to the first.

    The positions of the points used to construct a coordinate system have, by definition, to be specified in relation to an already existing coordinate system. In motion analysis, this is the coordinate system of the motion capture system, established in the calibration phase. In that phase, the positions of markers placed on an object with perpendicular axes and known distances between the markers are captured and used as the reference (laboratory) coordinate system.

    For example, given the positions $\overrightarrow{\mathbf{m}}_1 = [1,2,0],\; \overrightarrow{\mathbf{m}}_2 = [0,1,3],\; \overrightarrow{\mathbf{m}}_3 = [1,0,1]$, a basis can be found:
    """)
    return


@app.cell
def _(np):
    m1 = np.array([1, 2, 0])
    m2 = np.array([0, 1, 3])
    m3 = np.array([1, 0, 1])

    v1 = m2 - m1  # first axis
    v2 = np.cross(v1, m3 - m1)  # second axis
    v3 = np.cross(v1, v2)  # third axis

    # Vector normalization
    e1 = v1 / np.linalg.norm(v1)
    e2 = v2 / np.linalg.norm(v2)
    e3 = v3 / np.linalg.norm(v3)

    print("Versors:", "\ne1 =", e1, "\ne2 =", e2, "\ne3 =", e3)
    return e1, e2, e3, m1, m2, m3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Without a computer: three markers are at $\overrightarrow{\mathbf{m}}_1 = [0, 0, 0]$, $\overrightarrow{\mathbf{m}}_2 = [2, 0, 0]$ and $\overrightarrow{\mathbf{m}}_3 = [0, 3, 0]$ (in cm). Find $\hat{\mathbf{e}}_1$, $\hat{\mathbf{e}}_2$ and $\hat{\mathbf{e}}_3$ with the four steps above. Does $\hat{\mathbf{e}}_3$ point towards the third marker?

    /// details | Solution to Exercise 1

    $\overrightarrow{\mathbf{v}}_1 = [2, 0, 0]$, so $\hat{\mathbf{e}}_1 = [1, 0, 0]$. Then

    $$
    \overrightarrow{\mathbf{v}}_2 = [2, 0, 0] \times [0, 3, 0] = [0, 0, 6], \qquad
    \overrightarrow{\mathbf{v}}_3 = [2, 0, 0] \times [0, 0, 6] = [0, -12, 0]
    $$

    so $\hat{\mathbf{e}}_2 = [0, 0, 1]$ and $\hat{\mathbf{e}}_3 = [0, -1, 0]$.

    No: the third marker is on the positive $y$ side of the first axis, and $\hat{\mathbf{e}}_3$ points to the negative side. That always happens with this recipe. With $\overrightarrow{\mathbf{w}} = \overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_1$, the vector triple product gives $\overrightarrow{\mathbf{v}}_1 \times (\overrightarrow{\mathbf{v}}_1 \times \overrightarrow{\mathbf{w}}) = -\Vert\overrightarrow{\mathbf{v}}_1\Vert^2\,\overrightarrow{\mathbf{w}}^\bot$, where $\overrightarrow{\mathbf{w}}^\bot$ is the part of $\overrightarrow{\mathbf{w}}$ perpendicular to $\overrightarrow{\mathbf{v}}_1$: the third axis points *away* from the third marker. If you want it to point towards the marker, compute the second axis as $\overrightarrow{\mathbf{w}} \times \overrightarrow{\mathbf{v}}_1$ instead. Then $\hat{\mathbf{e}}_2 = [0, 0, -1]$ and $\hat{\mathbf{e}}_3 = [0, 1, 0]$, and the basis is still right-handed.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Checking the basis

    A coordinate system you have not tested is a guess. An orthonormal, right-handed basis has three properties, so there are three things to check:

    - **Every versor has norm 1.**
    - **Every pair of versors is perpendicular.** Test it with the dot product, which should be zero. Testing it with the cross product, whose norm should be one, also works in principle, but for two versors $\Vert\hat{\mathbf{a}}\times\hat{\mathbf{b}}\Vert = \sin\theta$, and the sine is flat near 90°: two versors 89° apart give a cross product of norm 0.99985, while their dot product, $\cos\theta = 0.017$, gives the error away at once. [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb) works through this example.
    - **The basis is right-handed**: $\hat{\mathbf{e}}_1 \times \hat{\mathbf{e}}_2 = \hat{\mathbf{e}}_3$. Equivalently, the determinant of the matrix with the versors as its rows is $+1$; it is $-1$ for a left-handed basis, which passes the first two tests just as well.
    """)
    return


@app.cell
def _(e1, e2, e3, np):
    print(
        "Norm of each versor:",
        "\n||e1|| =", np.linalg.norm(e1),
        "\n||e2|| =", np.linalg.norm(e2),
        "\n||e3|| =", np.linalg.norm(e3),
    )
    print(
        "\nTest of orthogonality (dot product between versors):",
        f"\ne1 . e2: {np.dot(e1, e2):.1e}",
        f"\ne1 . e3: {np.dot(e1, e3):.1e}",
        f"\ne2 . e3: {np.dot(e2, e3):.1e}",
    )
    print("\nTest of handedness:")
    print("e1 x e2 - e3 =", np.cross(e1, e2) - e3)
    print(f"det([e1, e2, e3]) = {np.linalg.det(np.vstack((e1, e2, e3))):g}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The norms are 1 and the dot products are zero, to within about $10^{-16}$: that is floating-point rounding, not a flaw in the method. And the basis is right-handed, as the construction guarantees, because $\overrightarrow{\mathbf{v}}_3 = \overrightarrow{\mathbf{v}}_1 \times \overrightarrow{\mathbf{v}}_2$ is by definition the third axis of a right-handed set.

    **Exercise 2.** The order of the markers matters. If you give the recipe the markers in the order $\overrightarrow{\mathbf{m}}_2, \overrightarrow{\mathbf{m}}_1, \overrightarrow{\mathbf{m}}_3$ instead of $\overrightarrow{\mathbf{m}}_1, \overrightarrow{\mathbf{m}}_2, \overrightarrow{\mathbf{m}}_3$, what happens to each of the three versors? Is the new basis still right-handed? Work it out from the properties of the cross product before checking with the code.

    /// details | Solution to Exercise 2

    The first axis becomes $\overrightarrow{\mathbf{m}}_1 - \overrightarrow{\mathbf{m}}_2 = -\overrightarrow{\mathbf{v}}_1$. For the second, $\overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_2 = (\overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_1) - \overrightarrow{\mathbf{v}}_1$, and the cross product of a vector with itself is zero, so

    $$
    (-\overrightarrow{\mathbf{v}}_1) \times \left[(\overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_1) - \overrightarrow{\mathbf{v}}_1\right] = -\overrightarrow{\mathbf{v}}_1 \times (\overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_1) = -\overrightarrow{\mathbf{v}}_2
    $$

    and the third is $(-\overrightarrow{\mathbf{v}}_1) \times (-\overrightarrow{\mathbf{v}}_2) = \overrightarrow{\mathbf{v}}_3$. So $\hat{\mathbf{e}}_1$ and $\hat{\mathbf{e}}_2$ reverse and $\hat{\mathbf{e}}_3$ does not change: the coordinate system has turned by 180° about $\hat{\mathbf{e}}_3$. Two reversed axes keep the determinant at $+1$, so the basis is still right-handed, and no test of the basis can tell that anything happened. With the markers of this notebook in the order $\overrightarrow{\mathbf{m}}_2, \overrightarrow{\mathbf{m}}_1, \overrightarrow{\mathbf{m}}_3$, the recipe gives $\hat{\mathbf{e}}_1 = [0.3015, 0.3015, -0.9045]$ and $\hat{\mathbf{e}}_2 = [-0.9129, -0.1826, -0.3651]$, while $\hat{\mathbf{e}}_3$ stays $[-0.2752, 0.9358, 0.2202]$.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The origin

    To define a coordinate system using the calculated basis, we also need to define an origin. In principle, any point could be the origin, but if the coordinate system should follow anatomical conventions, for example with its origin at a joint centre, the basis and the origin have to be calculated according to the standards used in motion analysis (see [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb)).

    If the coordinate system is a technical one and not anatomically based, a common procedure in motion analysis is to place the origin at the centroid (average position) of the markers. Averaging the positions of several markers potentially reduces the effect of noise, for example from soft tissue artifacts, on the calculation.

    For the markers in the example above, the origin of the coordinate system is:
    """)
    return


@app.cell
def _(m1, m2, m3, np):
    origin = np.mean((m1, m2, m3), axis=0)

    print("Origin:", origin)
    return (origin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization

    Let's plot the coordinate system with the function `CCS`, from the file `CCS.py` in the `functions` folder of this repository. The first cell adds that folder to Python's search path.
    """)
    return


@app.function
def repo_file(path):
    """Local path of a file of the BMC repository, e.g. "data/walk.trc".

    Where only the notebook is available (e.g., on molab), the file is
    downloaded from GitHub to a temporary folder.
    """
    import tempfile
    import urllib.request
    from pathlib import Path

    import marimo as mo

    local = mo.notebook_dir() / ".." / path
    if local.exists():
        return local
    cached = Path(tempfile.gettempdir()) / "BMC" / path
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/BMClab/BMC/master/{path}", cached)
    return cached


@app.cell
def _():
    import sys

    sys.path.insert(0, str(repo_file("functions/CCS.py").parent))  # add to pythonpath
    from CCS import CCS

    return (CCS,)


@app.cell
def _(e1, e2, e3, m1, m2, m3, np):
    markers = np.vstack((m1, m2, m3))
    basis = np.vstack((e1, e2, e3))
    return basis, markers


@app.cell
def _(CCS, basis, markers, origin):
    _ccs = CCS(xyz=[], Oijk=origin, ijk=basis, point=markers, vector=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The black arrows are the axes of the laboratory, the yellow dots are the three markers, and the red, green and blue arrows, which `CCS` labels i, j and k, are the versors $\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{e}}_3$, drawn from the centroid. The red versor points from the first marker towards the second, and the green one is perpendicular to the triangle the markers form.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A coordinate system that travels with the segment

    Why go to all this trouble? Because the origin and the basis are built from the markers alone, and the markers are fixed to the segment, so the coordinate system goes wherever the segment goes.

    The way to see this is to describe each marker in the segment's own coordinate system. Subtract the origin from the marker's position, and project what is left on each versor with the dot product. That gives the marker's **local coordinates**:

    $$
    \overrightarrow{\mathbf{m}}_{local} = \Big[\, (\overrightarrow{\mathbf{m}} - \overrightarrow{\mathbf{O}})\cdot\hat{\mathbf{e}}_1,\;\; (\overrightarrow{\mathbf{m}} - \overrightarrow{\mathbf{O}})\cdot\hat{\mathbf{e}}_2,\;\; (\overrightarrow{\mathbf{m}} - \overrightarrow{\mathbf{O}})\cdot\hat{\mathbf{e}}_3 \,\Big]
    $$

    Let's collect the recipe in two functions. The first returns the origin and the basis, with the versors as the rows of a matrix. It works for one frame or, as we will need later, for many frames at once, with one marker position per row. The second computes local coordinates. A third function, for comparing directions, measures the angle between two vectors from the dot product.
    """)
    return


@app.function
def segment_frame(m1, m2, m3):
    """Origin and basis of the coordinate system defined by three markers.

    Each marker is a position [x, y, z] or an array of positions, one frame per
    row (the two can be mixed). Returns the origin (the centroid of the markers)
    and the basis, with the versors e1, e2, e3 as the rows of the last two
    dimensions.
    """
    import numpy as np

    m1, m2, m3 = np.broadcast_arrays(*(np.asarray(m, dtype=float) for m in (m1, m2, m3)))
    v1 = m2 - m1
    v2 = np.cross(v1, m3 - m1)
    v3 = np.cross(v1, v2)
    basis = np.stack([v / np.linalg.norm(v, axis=-1, keepdims=True) for v in (v1, v2, v3)], axis=-2)
    origin = (m1 + m2 + m3) / 3
    return origin, basis


@app.function
def local_coordinates(points, origin, basis):
    """Coordinates of `points` (one per row) in the coordinate system with the
    given `origin` and `basis` (versors as rows)."""
    import numpy as np

    return (np.asarray(points, dtype=float) - origin) @ np.asarray(basis).T


@app.function
def angle_between(u, w):
    """Angle in degrees between the vectors `u` and `w` (along the last axis)."""
    import numpy as np

    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    cos = np.sum(u * w, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


@app.cell
def _(markers):
    _origin, _basis = segment_frame(*markers)

    print("Local coordinates of the markers (one marker per row):")
    print(local_coordinates(markers, _origin, _basis).round(4) + 0.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's move the segment: turn it by 30° about the laboratory's Z axis and shift it by $[0.5, -0.2, 0.1]$. The rotation is done with a rotation matrix; how rotation matrices work is the subject of [Rigid-body transformations in 3D](https://github.com/BMClab/BMC/blob/master/notebooks/Transformation3D.ipynb), but here it is simply a way to move the three markers as one rigid body.

    **Before you run the next cell**, predict: which numbers change, the laboratory coordinates of the markers, the versors, the local coordinates? And does every versor turn by 30°?
    """)
    return


@app.cell
def _(basis, markers, np):
    _theta = np.radians(30)
    _Rz = np.array(
        [
            [np.cos(_theta), -np.sin(_theta), 0],
            [np.sin(_theta), np.cos(_theta), 0],
            [0, 0, 1],
        ]
    )
    _moved = markers @ _Rz.T + np.array([0.5, -0.2, 0.1])
    _origin, _basis = segment_frame(*_moved)

    print("Laboratory coordinates of the moved markers:")
    print(_moved.round(4))
    print("\nLocal coordinates of the moved markers:")
    print(local_coordinates(_moved, _origin, _basis).round(4) + 0.0)
    print("\nHow much each versor turned [degrees]:", angle_between(_basis, basis).round(1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every laboratory coordinate changed, and so did every versor, but the local coordinates are exactly the same as before. That is what a segment coordinate system buys you: numbers that describe the segment and are blind to where the segment happens to be. When the local coordinates of a marker change from one frame to the next, either the segment is not rigid or the marker has moved on it.

    Two details are worth a second look.

    - The second local coordinate of every marker is zero. That is no accident: $\hat{\mathbf{e}}_2$ is perpendicular to the plane of the markers, so no marker has any position along it.
    - The segment turned by 30°, but its versors turned by 12.7°, 27.9° and 29.2°. Only a vector perpendicular to the axis of rotation turns by the full angle. $\hat{\mathbf{e}}_1$ is the versor closest to the Z axis (its Z component is 0.90), so it turns the least.

    **Guiding questions 1.**

    1. Which rotation of the segment would leave $\hat{\mathbf{e}}_1$ completely unchanged? Would the local coordinates change?
    2. A fourth marker on the same plate has a local coordinate of 2 cm along $\hat{\mathbf{e}}_2$ in one frame and 2.6 cm in the next. Give two possible explanations.
    3. Why must the origin be calculated from the markers too, and not taken as a fixed point of the laboratory?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** A segment's coordinate system has its origin at $\overrightarrow{\mathbf{O}} = [1, 2, 0]$ and the versors $\hat{\mathbf{e}}_1 = [0, 1, 0]$, $\hat{\mathbf{e}}_2 = [-1, 0, 0]$ and $\hat{\mathbf{e}}_3 = [0, 0, 1]$.<br>
    a) What are the local coordinates of the point $\overrightarrow{\mathbf{P}} = [1, 3, 0.5]$?<br>
    b) Which point of the laboratory has the local coordinates $[2, 1, 0]$?

    /// details | Solution to Exercise 3

    a) $\overrightarrow{\mathbf{P}} - \overrightarrow{\mathbf{O}} = [0, 1, 0.5]$, and its dot products with the three versors are $1$, $0$ and $0.5$: the local coordinates are $[1, 0, 0.5]$.

    b) Local coordinates say how much of each versor to add to the origin:

    $$
    \overrightarrow{\mathbf{O}} + 2\,\hat{\mathbf{e}}_1 + 1\,\hat{\mathbf{e}}_2 + 0\,\hat{\mathbf{e}}_3 = [1, 2, 0] + [0, 2, 0] + [-1, 0, 0] = [0, 4, 0]
    $$

    Check it by going back: $[0, 4, 0] - \overrightarrow{\mathbf{O}} = [-1, 2, 0]$, whose dot products with the versors are $2$, $1$ and $0$.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: a landmark without a marker

    Local coordinates do more than prove that a segment is rigid: they let you follow points that carry no marker at all. Suppose you need the lateral malleolus to locate the ankle joint centre, but a marker on the malleolus would slide with the skin, or get knocked off, during walking. The usual solution is to locate the landmark once, in a standing calibration trial, while the shank cluster is recorded, and to store its local coordinates in the cluster's coordinate system. In every frame of the movement, the landmark is then rebuilt from the cluster alone, by running Exercise 3 b) backwards:

    $$
    \overrightarrow{\mathbf{p}} = \overrightarrow{\mathbf{O}} + x_{local}\,\hat{\mathbf{e}}_1 + y_{local}\,\hat{\mathbf{e}}_2 + z_{local}\,\hat{\mathbf{e}}_3
    $$

    Here is that formula as a function, the inverse of `local_coordinates`, and a function that builds rotation matrices, which we need only to generate the movement:
    """)
    return


@app.function
def global_coordinates(local, origin, basis):
    """Laboratory coordinates of the point(s) with the given `local` coordinates
    in the coordinate system with `origin` and `basis` (versors as rows); the
    inverse of local_coordinates."""
    import numpy as np

    return origin + np.asarray(local, dtype=float) @ np.asarray(basis)


@app.function
def rotation_matrix(axis, angle):
    """Matrix that rotates vectors by `angle` degrees about the laboratory's
    "x", "y" or "z" axis, counterclockwise seen from the positive end of the axis."""
    import numpy as np

    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The laboratory's axes point forward ($X$), up ($Y$) and to the right ($Z$), and we follow the right leg. In the calibration trial, the cluster's three markers sit on the lateral side of the shank, about 7 to 11 cm apart, and the lateral malleolus is located with a pointer:
    """)
    return


@app.cell
def _(np):
    # Standing calibration trial [m]
    knee = np.array([0.000, 0.500, 0.100])  # knee joint centre
    cluster_static = np.array(
        [
            [0.040, 0.330, 0.150],  # c1
            [-0.030, 0.345, 0.148],  # c2
            [0.005, 0.245, 0.156],  # c3
        ]
    )
    lm_static = np.array([-0.010, 0.070, 0.135])  # lateral malleolus, located in this trial only

    _origin, _basis = segment_frame(*cluster_static)
    lm_local = local_coordinates(lm_static, _origin, _basis)

    print("Lateral malleolus in the cluster's coordinate system:", lm_local.round(4), "m")
    print(f"Distance from the cluster's origin: {np.linalg.norm(lm_local):.3f} m")
    return cluster_static, knee, lm_local, lm_static


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The malleolus is about 24 cm from the cluster's origin. Now a frame of the movement, in which the motion capture system reports only the cluster, to the millimetre. To have a known answer to check against, the frame was generated from the calibration: the knee flexed by 45°, the shank turned 8° outwards about its long axis, and the whole leg moved 0.75 m forward. The same movement gives the true position of the malleolus, which we will now pretend not to know.

    **Before you run the next cells**, predict: how far from its true position will the rebuilt malleolus be? And how much will each versor of the cluster turn?
    """)
    return


@app.cell
def _(cluster_static, knee, lm_static, np):
    _R = rotation_matrix("z", -45) @ rotation_matrix("y", -8)  # flexion after the outward turn
    _move = lambda p: knee + (p - knee) @ _R.T + np.array([0.75, 0.02, 0.0])

    cluster_mov = np.round(_move(cluster_static), 3)  # reported to the millimetre
    lm_true = _move(lm_static)  # the answer we pretend not to know

    print("Cluster markers in the movement trial [m]:\n", cluster_mov)
    return cluster_mov, lm_true


@app.cell
def _(cluster_mov, cluster_static, lm_local, lm_true, np):
    _origin, _basis = segment_frame(*cluster_mov)
    _, _basis_static = segment_frame(*cluster_static)
    lm_rebuilt = global_coordinates(lm_local, _origin, _basis)

    print("Rebuilt lateral malleolus:", lm_rebuilt.round(4), "m")
    print(f"Distance from the true position: {1000 * np.linalg.norm(lm_rebuilt - lm_true):.2f} mm")
    print("How much each versor of the cluster turned [degrees]:", angle_between(_basis, _basis_static).round(1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The malleolus is rebuilt less than half a millimetre from where it really is, and that small error comes entirely from rounding the cluster's markers to the millimetre, which shifts the cluster's origin and tilts its axes slightly.

    The versors turned by 45.6°, 5.7° and 45.4°. The knee flexed by 45° about the laboratory's $Z$ axis, and $\hat{\mathbf{e}}_2$, perpendicular to a plate on the lateral side of the shank, points almost along $Z$. A versor parallel to the axis of rotation does not turn at all, which is Guiding question 1.1 seen in real data; the few degrees $\hat{\mathbf{e}}_2$ did turn come from its not being exactly parallel to $Z$ and from the outward turn of the shank.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When the markers are real

    Everything so far used exact positions. Real markers are located with an error: around a millimetre for a good optoelectronic system, and much more if you count the movement of the skin over the bone. How does an error in the markers become an error in the axes?

    Let's simulate it. The function below places three markers at the corners of an equilateral triangle with a given side, like the markers of a cluster. We add random noise with a standard deviation of 1 mm to every coordinate of every marker, compute the basis, and measure the angle between each noisy versor and the true one. Each run has 5000 simulated frames.
    """)
    return


@app.function
def equilateral_cluster(side):
    """Positions of three markers at the corners of an equilateral triangle
    with the given side, in the plane z = 0 and centred on the origin."""
    import numpy as np

    angles = np.radians([90, 210, 330])
    radius = side / np.sqrt(3)
    return [np.array([radius * np.cos(a), radius * np.sin(a), 0.0]) for a in angles]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Before you run the next cell**, predict the typical angular error of the axes for a cluster with markers 10 cm apart: 0.1°, 1° or 10°? And for a cluster half that size?
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(0)
    _sigma, _n = 0.001, 5000  # 1 mm of noise, number of simulated frames

    print("Side    median error of e1, e2, e3    95th percentile")
    for _side in [0.05, 0.10, 0.15]:
        _markers = equilateral_cluster(_side)
        _, _basis = segment_frame(*_markers)
        _, _noisy = segment_frame(*[m + _rng.normal(0, _sigma, (_n, 3)) for m in _markers])
        _err = angle_between(_noisy, _basis)
        print(
            f"{100 * _side:3.0f} cm   {np.median(_err, axis=0).round(2)} degrees"
            f"   {np.percentile(_err, 95, axis=0).round(2)} degrees"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With markers 10 cm apart, a millimetre of noise turns each axis by about 1° in a typical frame, and by 2° in one frame in twenty. Halve the size of the cluster and the errors double. Each axis is a direction computed from differences between markers, and an error of 1 mm in a difference of 5 cm tilts that difference twice as much as the same error in a difference of 10 cm.

    Let's check that the error really goes as the inverse of the spacing, for sides from 2 to 20 cm. This time we record, in every frame, the worst of the three axes:
    """)
    return


@app.cell
def _(np, plt):
    _rng = np.random.default_rng(1)
    _sides = np.linspace(0.02, 0.20, 10)
    _worst = []
    for _side in _sides:
        _markers = equilateral_cluster(_side)
        _, _basis = segment_frame(*_markers)
        _, _noisy = segment_frame(*[m + _rng.normal(0, 0.001, (5000, 3)) for m in _markers])
        _worst.append(np.median(angle_between(_noisy, _basis).max(axis=1)))
    _worst = np.array(_worst)
    _k = np.mean(_worst * _sides)  # error x side, constant if the error goes as 1/side

    _d = np.linspace(0.018, 0.21, 200)
    _, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.plot(100 * _d, _k / _d, color="0.6", linewidth=2, label=f"{_k:.3f} degree·m / side")
    _ax.plot(100 * _sides, _worst, "o", color="tab:red", markersize=8, label="simulation")
    _ax.set_xlabel("Side of the cluster [cm]")
    _ax.set_ylabel("Error of the worst axis [degrees]")
    _ax.set_title("Median error of the worst axis, 1 mm of noise")
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulation follows the inverse of the spacing closely: doubling the distance between the markers halves the error. Halving the noise would have the same effect, which gives you two levers for the same result, and spreading the markers is usually the cheaper one. Cappozzo et al. (1997) turned considerations like these into design criteria for marker clusters.

    Why not, then, make the cluster as large as possible? Because the cluster sits on skin. The larger it is, the more skin it spans, and skin moves relative to the bone, differently from place to place. The best cluster is a compromise between the two kinds of error.

    **Exercise 4.** A cluster has its markers 10 cm apart, each with 1 mm of noise, and is used to rebuild a landmark 30 cm from the cluster's origin, in the plane of the markers, as in the example of the malleolus. Before simulating, estimate the typical error of (a) the origin and (b) the rebuilt landmark. Then check your estimates with 5000 simulated frames, using `equilateral_cluster`, `segment_frame` and `global_coordinates`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _rng = np.random.default_rng(3)
    _markers = equilateral_cluster(0.10)
    _origin, _basis = segment_frame(*_markers)
    _p_local = np.array([0, 0, -0.30])  # 30 cm from the origin, in the plane of the markers
    _p_true = global_coordinates(_p_local, _origin, _basis)

    _origin_n, _basis_n = segment_frame(*[m + _rng.normal(0, 0.001, (5000, 3)) for m in _markers])
    _p_n = global_coordinates(_p_local, _origin_n, _basis_n)
    _err_origin = 1000 * np.linalg.norm(_origin_n - _origin, axis=1)
    _err_p = 1000 * np.linalg.norm(_p_n - _p_true, axis=1)
    _err_axis = np.median(angle_between(_basis_n, _basis))
    _lever = 300 * np.radians(_err_axis)

    mo.accordion(
        {
            "Solution to Exercise 4": mo.md(
                f"""
    **(a) The origin.** It is the average of three markers, so each of its coordinates has a standard deviation of 1 mm divided by √3, or {1 / np.sqrt(3):.2f} mm. In three dimensions that makes a typical error of {np.median(_err_origin):.2f} mm (median of the simulation), less than the error of any single marker.

    **(b) The landmark.** Its error is dominated by the tilt of the axes. The axes are off by {_err_axis:.2f}° in a typical frame (as in the simulation above), and a tilt of that size moves a point 30 cm away by 0.30 m × {np.radians(_err_axis):.4f} rad = {_lever:.1f} mm. The simulation gives a median error of {np.median(_err_p):.1f} mm, and {np.percentile(_err_p, 95):.1f} mm in one frame in twenty.

    The landmark is rebuilt about {np.median(_err_p) / np.median(_err_origin):.0f} times worse than the origin. Angular errors grow with distance, which is why a cluster should sit close to the landmarks it is used to rebuild, and why a joint centre far from a small cluster is the least reliable point of a model.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Nearly collinear markers

    The recipe requires noncollinear points: if the three markers lie on a line, $\overrightarrow{\mathbf{v}}_1$ and $\overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_1$ are parallel, their cross product is zero, and there is no basis at all. Real markers are never exactly in a line, but what happens when they almost are?

    Let's keep two markers 10 cm apart and move the third one towards the line between them, from 5 cm away to 2 cm and then 5 mm.

    **Before you run the next cell**, predict: which of the three versors suffers?
    """)
    return


@app.cell
def _(np):
    _rng = np.random.default_rng(2)
    _sigma, _n = 0.001, 5000

    print("Offset    median error of e1, e2, e3    95th percentile")
    for _h in [0.05, 0.02, 0.005]:
        _markers = [np.array([-0.05, 0, 0]), np.array([0.05, 0, 0]), np.array([0, _h, 0])]
        _, _basis = segment_frame(*_markers)
        _, _noisy = segment_frame(*[m + _rng.normal(0, _sigma, (_n, 3)) for m in _markers])
        _err = angle_between(_noisy, _basis)
        print(
            f"{1000 * _h:3.0f} mm   {np.median(_err, axis=0).round(2)} degrees"
            f"   {np.percentile(_err, 95, axis=0).round(2)} degrees"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first versor does not care: it depends only on the first two markers, which stay 10 cm apart, and its error stays at about 1°. The other two degrade quickly. With the third marker 5 mm from the line, their typical error is about 10°, and in one frame in twenty it is close to 30°: a coordinate system that wobbles by tens of degrees while the segment stands still.

    The reason is in step 2. The norm of $\overrightarrow{\mathbf{v}}_1 \times (\overrightarrow{\mathbf{m}}_3 - \overrightarrow{\mathbf{m}}_1)$ is the length of $\overrightarrow{\mathbf{v}}_1$ times the distance of the third marker from the line. When that distance is 5 mm, a 1 mm error in the third marker is a large fraction of it, and the direction of the cross product, which defines $\hat{\mathbf{e}}_2$, swings wildly. $\hat{\mathbf{e}}_3$ is computed from $\hat{\mathbf{e}}_2$ and inherits the error.

    **Challenge 1.** For the 5 mm case, the first axis is built from the two markers that are 10 cm apart. Pass the same markers to `segment_frame` in a different order, so that the first axis is built from one of the outer markers and the marker near the middle, and compare the errors. What rule would you adopt for choosing which markers define the first axis?

    **Exercise 5.** In the 5 mm case, all three markers are noisy. Which marker's noise does the most damage to $\hat{\mathbf{e}}_2$? Add the noise to one marker at a time and find out. Then explain the result with the distance of each marker from the line through the other two.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _rng = np.random.default_rng(4)
    _markers = [np.array([-0.05, 0, 0]), np.array([0.05, 0, 0]), np.array([0, 0.005, 0])]
    _, _basis = segment_frame(*_markers)

    _rows, _e2 = [], []
    for _k in range(3):
        _noisy = [m + _rng.normal(0, 0.001, (5000, 3)) if _i == _k else m for _i, m in enumerate(_markers)]
        _, _basis_n = segment_frame(*_noisy)
        _err = np.median(angle_between(_basis_n, _basis), axis=0)
        _e2.append(_err[1])
        _a, _b = [_markers[_i] for _i in range(3) if _i != _k]
        _dist = np.linalg.norm(np.cross(_b - _a, _markers[_k] - _a)) / np.linalg.norm(_b - _a)
        _rows.append(f"| m{_k + 1} | {1000 * _dist:.1f} mm | {_err[0]:.2f}° | {_err[1]:.2f}° | {_err[2]:.2f}° |")

    _, _basis_all = segment_frame(*[m + _rng.normal(0, 0.001, (5000, 3)) for m in _markers])
    _e2_all = np.median(angle_between(_basis_all, _basis), axis=0)[1]
    _table = "\n    ".join(_rows)  # same indentation as the text around it

    mo.accordion(
        {
            "Solution to Exercise 5": mo.md(
                f"""
    Median error of each versor with 1 mm of noise on one marker only:

    | noisy marker | distance from the line through the other two | e1 | e2 | e3 |
    |---|---|---|---|---|
    {_table}

    The third marker does the most damage, but the outer ones are far from harmless: their noise turns ê2 by about half as much. The plane of the markers tilts about the line through the other two markers, by roughly the noisy marker's error out of the plane divided by its distance from that line. That distance is 5 mm for the third marker and only about 1 cm for each of the outer ones: in a thin triangle, *every* marker is close to the line through the other two.

    The three contributions add roughly like the sides of a right triangle: √({_e2[0]:.1f}² + {_e2[1]:.1f}² + {_e2[2]:.1f}²) = {np.sqrt(np.sum(np.square(_e2))):.1f}°, close to the {_e2_all:.1f}° of a simulation with all three markers noisy.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: a label swap in a recording

    A motion capture system sees anonymous dots; software decides which dot is which marker, frame after frame. Now and then it gets two of them the wrong way round for a few frames, especially when markers are close or briefly hidden. Nothing in the coordinates looks wrong, but the segment's coordinate system does something dramatic.

    Let's simulate two seconds of the shank cluster from the malleolus example, recorded at 100 Hz with 1 mm of noise, while the knee swings back and forth and the body walks forward at 1.2 m/s. Between 0.80 and 0.95 s, the labels of $c_1$ and $c_2$ are swapped. Pretend you do not know that, and look for it.
    """)
    return


@app.cell
def _(cluster_static, knee, np):
    _rng = np.random.default_rng(5)
    t_rec = np.arange(0, 2, 0.01)  # 2 s at 100 Hz
    _flexion = -15 + 25 * np.sin(2 * np.pi * 0.9 * t_rec)  # degrees about Z
    _true = np.array(
        [
            knee + (cluster_static - knee) @ rotation_matrix("z", _a).T + [1.2 * _t, 0, 0]
            for _t, _a in zip(t_rec, _flexion)
        ]
    )
    _true = _true + _rng.normal(0, 0.001, _true.shape)  # 1 mm of noise

    recording = _true.copy()  # frames x markers x coordinates
    _swap = (t_rec >= 0.80) & (t_rec < 0.95)
    recording[_swap, 0], recording[_swap, 1] = _true[_swap, 1], _true[_swap, 0]
    return recording, t_rec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two quantities are worth plotting: the distances between the markers, which a rigid plate keeps constant whatever the segment does, and how much each versor turns from one frame to the next.

    **Before you run the next cell**, predict: which of the three distances change during the swap? Which versors jump, and by how much? (If you solved Exercise 2, you already know.)
    """)
    return


@app.cell
def _(np, plt, recording, t_rec):
    _d12 = np.linalg.norm(recording[:, 0] - recording[:, 1], axis=1)
    _d13 = np.linalg.norm(recording[:, 0] - recording[:, 2], axis=1)
    _d23 = np.linalg.norm(recording[:, 1] - recording[:, 2], axis=1)
    _, _basis = segment_frame(recording[:, 0], recording[:, 1], recording[:, 2])
    _turn = angle_between(_basis[1:], _basis[:-1])  # turn of each versor between consecutive frames

    _, _axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for _d, _label in [(_d12, "$c_1 - c_2$"), (_d13, "$c_1 - c_3$"), (_d23, "$c_2 - c_3$")]:
        _axs[0].plot(t_rec, 100 * _d, linewidth=1.5, label=_label)
    _axs[0].set_ylabel("Distance [cm]")
    _axs[0].set_title("Distances between the cluster's markers")
    _axs[0].set_ylim(6.5, 11.5)
    _axs[0].legend(loc="center", bbox_to_anchor=(0.5, 0.34), ncol=3)  # in the empty band
    for _k, _color in enumerate(["tab:red", "tab:green", "tab:blue"]):
        _axs[1].plot(t_rec[1:], _turn[:, _k], color=_color, linewidth=1.5, label=rf"$\hat{{\mathbf{{e}}}}_{_k + 1}$")
    _axs[1].set_ylabel("Turn per frame [degrees]")
    _axs[1].set_xlabel("Time [s]")
    _axs[1].set_title("Rotation of each versor from one frame to the next")
    _axs[1].legend(loc="center left")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The distance between $c_1$ and $c_2$ does not notice the swap: it is the same distance whichever marker is called which. The other two exchange values, because during the swap "$c_1$" is really $c_2$. A cluster whose triangle is equilateral would hide even that, which is one reason clusters are deliberately built asymmetric.

    The versors are more dramatic. $\hat{\mathbf{e}}_1$ and $\hat{\mathbf{e}}_2$ turn by nearly 180° in a single frame, at the start and again at the end of the swap, while $\hat{\mathbf{e}}_3$ hardly moves: the 180° turn about $\hat{\mathbf{e}}_3$ of Exercise 2. Any joint angle computed from this shank would jump by the same amount, and a low-pass filter would smear the jump into something that looks like a real movement.

    Detection and repair take a few lines. During the swap, the distances $c_1 - c_3$ and $c_2 - c_3$ move away from their typical values by about 1.4 cm, the difference between them, while the noise moves them by only a few millimetres. A threshold halfway between, 7 mm, separates the two:
    """)
    return


@app.cell
def _(np, recording, t_rec):
    _d13 = np.linalg.norm(recording[:, 0] - recording[:, 2], axis=1)
    _d23 = np.linalg.norm(recording[:, 1] - recording[:, 2], axis=1)
    _swapped = (np.abs(_d13 - np.median(_d13)) > 0.007) | (np.abs(_d23 - np.median(_d23)) > 0.007)

    _fixed = recording.copy()
    _fixed[_swapped, 0], _fixed[_swapped, 1] = recording[_swapped, 1], recording[_swapped, 0]
    _, _basis_before = segment_frame(*recording.transpose(1, 0, 2))
    _, _basis_after = segment_frame(*_fixed.transpose(1, 0, 2))

    print(
        f"Frames flagged: {_swapped.sum()}, from t = {t_rec[_swapped][0]:.2f} s to t = {t_rec[_swapped][-1]:.2f} s"
    )
    print(
        "Largest turn per frame of e1, e2, e3 before the repair:",
        angle_between(_basis_before[1:], _basis_before[:-1]).max(axis=0).round(1),
        "degrees",
    )
    print(
        "Largest turn per frame of e1, e2, e3 after the repair: ",
        angle_between(_basis_after[1:], _basis_after[:-1]).max(axis=0).round(1),
        "degrees",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The threshold finds exactly the 15 swapped frames, and after swapping the labels back, no versor turns by more than about 6° between frames. That remaining turn is mostly noise: with 1 mm of error on markers 7 to 11 cm apart, the versors jitter by a degree or two from frame to frame, more than the true rotation of the shank, which never exceeds 25° × 2π × 0.9 Hz ≈ 141°/s, or 1.4° per frame.

    A real pipeline would use more robust statistics than the median of the whole trial, and would check every pair of markers, but the idea is the one shown here: a rigid cluster carries its own test of whether its labels can be trusted.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other ways to calculate a basis

    An equivalent way to calculate a basis from vectors is the [Gram–Schmidt process](http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process), built on the dot product. It is implemented in [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb), and [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb) shows a trap to avoid when using it for a coordinate system: it must be fed differences between markers, not their positions. For an anatomical coordinate system, we have to handle the vectors explicitly in either method, to obtain the axes in the directions the convention prescribes.

    For a technical coordinate system, it is common to use more than three markers on the segment, to reduce the effect of noise and soft tissue artifacts. With more markers than the minimum, the basis is found by least squares, with yet another procedure: the [singular value decomposition (SVD) algorithm](https://github.com/BMClab/BMC/blob/master/notebooks/SVDalgorithm.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to the three markers you placed on your forearm in Challenge 0. Would you move them now? How far apart would you try to put them, and what stops you from spreading them further?
    2. A colleague tests the orthogonality of a basis by checking that $\Vert\hat{\mathbf{e}}_1 \times \hat{\mathbf{e}}_2\Vert = 1$, and reports success. What could they have missed, and what would you check instead?
    3. A basis passes every test of length and orthogonality, but its determinant is $-1$. What went wrong in its construction, and why does it matter?
    4. Why is the centroid of the markers a better origin for a technical coordinate system than one of the markers?
    5. You can either buy cameras that halve the noise of every marker, or double the spacing of your markers. Which improves the axes more?
    6. In the simulation with moved markers, the local coordinates were exactly the same before and after. In real data they never are. List two reasons.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Given the points in 3D space m1 = [2, 2, 0], m2 = [0, 1, 1], m3 = [1, 2, 0], find an orthonormal basis.

    2. Determine whether the following points form a basis in 3D space: m1 = [2, 2, 0], m2 = [1, 1, 1], m3 = [1, 1, 0].

    3. For the points of problem 1, calculate the origin of the coordinate system and the local coordinates of the three points. Which local coordinate is zero for all three, and why?

    4. Using `segment_frame` and the simulation in this notebook, find the smallest side of an equilateral cluster for which the median error of every axis is below 0.5° when the markers have 0.5 mm of noise. Before simulating, predict the answer from the results for 1 mm.

    5. Write a function that receives the positions of three markers and returns a number between 0 and 1 that measures how far they are from lying on a line, with 0 for collinear markers. (Hint: the norm of a cross product, divided by something.) Apply it to the points of problems 1 and 2 and to the clusters of this notebook.

    6. In the calibration trial of the example "A landmark without a marker", the medial malleolus was at [−0.005, 0.075, 0.065] m. Rebuild it in the movement trial, calculate the ankle joint centre as the midpoint between the two malleoli in both trials, and check that the distance between the malleoli is the same in both trials.

    7. Repeat the example "A label swap in a recording" with the labels of $c_1$ and $c_3$ swapped instead of $c_1$ and $c_2$. Before running the code, predict which distances change and what happens to each versor; then check your predictions.

    8. A cluster's markers are at the corners of a triangle with sides of 8, 10 and 12 cm. Calculate the distance of each marker from the line through the other two and, using the reasoning of Exercise 5, rank the markers by how much their noise disturbs $\hat{\mathbf{e}}_2$. Check the ranking with a simulation.

    9. The whole plate of a marker cluster slides 1 cm down the shank during a run. Explain why neither the distances between its markers nor their local coordinates can reveal it, and propose a measurement that could.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Frame of reference](https://github.com/BMClab/BMC/blob/master/notebooks/ReferenceFrame.ipynb): frames of reference, inertial frames, the Gram–Schmidt process, and an anatomical coordinate system for the femur.
    - [Scalar and vector](https://github.com/BMClab/BMC/blob/master/notebooks/ScalarVector.ipynb): the norm, the dot product and the cross product used here.
    - [SVD algorithm](https://github.com/BMClab/BMC/blob/master/notebooks/SVDalgorithm.ipynb): the best-fitting basis for more than three markers.
    - [Rigid-body transformations in 3D](https://github.com/BMClab/BMC/blob/master/notebooks/Transformation3D.ipynb): rotation matrices, which take coordinates from one coordinate system to another.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Cappozzo A, Cappello A, Della Croce U, Pensalfini F (1997) [Surface-marker cluster design criteria for 3-D bone movement reconstruction](https://doi.org/10.1109/10.649988). IEEE Transactions on Biomedical Engineering, 44(12), 1165–1174.
    - [Standards - International Society of Biomechanics](https://isbweb.org/activities/standards).
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
