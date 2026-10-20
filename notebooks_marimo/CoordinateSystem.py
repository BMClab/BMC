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


@app.cell
def _(mo):
    import sys

    sys.path.insert(0, str(mo.notebook_dir() / ".." / "functions"))  # add to pythonpath
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
    row. Returns the origin (the centroid of the markers) and the basis, with
    the versors e1, e2, e3 as the rows of the last two dimensions.
    """
    import numpy as np

    m1, m2, m3 = (np.asarray(m, dtype=float) for m in (m1, m2, m3))
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
    """)
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
