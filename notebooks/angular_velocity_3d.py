# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14", "sympy>=1.13", "pandas>=2.2", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Angular velocity in 3D movements

    > Renato Naville Watanabe, Marcos Duarte<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    In a plane, angular velocity is easy: one angle, one derivative, one number with a sign. In three dimensions a body segment can flex, tilt and twist at the same time, and the familiar recipe of differentiating an angle stops working. Three angles describe the orientation, but their three derivatives are *not* the angular velocity.

    This notebook finds the angular velocity of a body in 3D from what a motion capture system actually gives you: a basis attached to the body, frame after frame. It starts with the axis about which a body has turned, shows why finite rotations cannot be added like vectors while angular velocities can, derives a formula that needs nothing but dot products, checks it on motions with known answers, and applies it to a forearm, a pendulum and the shank of a person walking.

    Read it in order and run each cell as you reach it. Where you find a **Challenge** or a set of **Guiding questions**, stop and answer on a scratchpad before moving on. Several of them ask you to predict a number *before* the code prints it. Make the prediction; a wrong guess that the code then corrects teaches more than a right one. The **Exercises** come with collapsed solutions: open one only after you have tried the exercise yourself.

    You will need the planar angular velocity and $\vec{\mathbf{v}} = \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}$ from [Kinematics of a rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py), rotation matrices and Cardan angles from [Rigid-body transformations in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py), and a basis attached to a segment, built from markers as in [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py).

    **Challenge 0.** Before you begin, pick up a pen. Spin it about its long axis with your fingers while you swing your arm in an arc. At any one instant, is there a single line about which the pen is rotating? Or two rotations going on at once? Write down your answer; the checkpoint questions ask you to come back to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How fast is the shank turning?

    In [Rigid-body transformations in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py), the orientation of the right shank during a stride was described by three Cardan angles. The largest of them, $\gamma$, the inclination of the shank in the sagittal plane, goes from about $-65^\circ$ to $+36^\circ$ during the stride, and in late swing it changes at almost $380^\circ$/s. The other two angles change too, by tens of degrees, while the shank swings.

    So how fast is the shank turning? Is it $380^\circ$/s? Is the angular velocity a vector with the three angle rates as its components?

    **Guiding questions 0.**

    1. In a plane, the angular velocity is the derivative of the angle. Why might that stop being true when there are three angles?
    2. The same orientations described with another Cardan sequence give different angles, and so different angle rates. Should two people using different sequences get different angular velocities for the same shank?
    3. The angular velocity is a vector. What would its direction mean physically?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy does the numerical work, Matplotlib draws the figures and SymPy derives a formula symbolically. The functions `CCSbasis` and `read_trc`, from the `functions` folder of this repository, draw coordinate systems and read motion capture files; they are loaded where they are first needed.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import sympy as sym

    matplotlib.rc("axes", labelsize=13, titlesize=14)
    matplotlib.rc("xtick", labelsize=11)
    matplotlib.rc("ytick", labelsize=11)
    matplotlib.rc("legend", fontsize=11)
    return np, plt, sym


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A basis attached to the body

    A usual problem in biomechanics, and in mechanics in general, is to find the angular velocity of an object. We consider that a basis $\hat{\mathbf{e}}_1$, $\hat{\mathbf{e}}_2$ and $\hat{\mathbf{e}}_3$ is attached to the body and is known at every instant. To learn how to find such a basis from the positions of markers, see [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py) and [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py).

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/3DbodyrefMove.png?raw=1" width=600 alt="a body and its basis at two instants"/></center><figcaption><center><i>Figure. A body with its basis $\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{e}}_3$ at two instants, described in the laboratory's basis $\hat{\mathbf{i}}, \hat{\mathbf{j}}, \hat{\mathbf{k}}$.</i></center></figcaption></figure>

    Everything about the body's rotation is in how this basis turns. The position of the body is a separate matter, carried by the origin of the basis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Axis of rotation

    As in planar movement, the angular velocity is a vector perpendicular to the plane of the rotation. The line in the direction of the angular velocity vector is known as the axis of rotation.

    The rotation between two frames of reference is characterized by the rotation matrix $\mathbf{R}$ obtained by stacking the versors $\hat{\mathbf{e}}_1$, $\hat{\mathbf{e}}_2$ and $\hat{\mathbf{e}}_3$, written in the laboratory's basis, in the columns of the matrix (for a review of rotation matrices, see [Rigid-body transformations in a plane](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_2d.py) and [Rigid-body transformations in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py)).

    A vector in the direction of the axis of rotation is a vector that does not change after the rotation. That is:

    $$
    \vec{\mathbf{v}} = \mathbf{R}\,\vec{\mathbf{v}}
    $$

    This vector is the eigenvector of the rotation matrix $\mathbf{R}$ with eigenvalue equal to one.

    Below, the yellow arrow shows the axis of rotation between the reference frame $\hat{\mathbf{i}}, \hat{\mathbf{j}}, \hat{\mathbf{k}}$ (coloured arrows) and the reference frame $\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{e}}_3$ (black arrows), obtained by rotating the first one by $\gamma$ about $\mathbf{Z}$, then by $\alpha$ about $\mathbf{X}$, then by $\beta$ about $\mathbf{Y}$, all about the fixed axes of the laboratory. Use the sliders to change the angles.
    """)
    return


@app.cell
def _(sym):
    _a, _b, _g = sym.symbols("alpha, beta, gamma")

    # Elemental rotation matrices of xyz in relation to XYZ:
    _RX = sym.Matrix([[1, 0, 0], [0, sym.cos(_a), -sym.sin(_a)], [0, sym.sin(_a), sym.cos(_a)]])
    _RY = sym.Matrix([[sym.cos(_b), 0, sym.sin(_b)], [0, 1, 0], [-sym.sin(_b), 0, sym.cos(_b)]])
    _RZ = sym.Matrix([[sym.cos(_g), -sym.sin(_g), 0], [sym.sin(_g), sym.cos(_g), 0], [0, 0, 1]])

    # Rotation matrix of xyz in relation to XYZ:
    rotation_matrix = sym.lambdify((_a, _b, _g), _RY @ _RX @ _RZ, "numpy")
    return (rotation_matrix,)


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(-180, 180, step=5, value=0, label="α, about X [degrees]")
    beta_slider = mo.ui.slider(-180, 180, step=5, value=0, label="β, about Y [degrees]")
    gamma_slider = mo.ui.slider(-180, 180, step=5, value=45, label="γ, about Z [degrees]")
    mo.vstack([alpha_slider, beta_slider, gamma_slider])
    return alpha_slider, beta_slider, gamma_slider


@app.function
def repo_file(path, *, base_url="https://raw.githubusercontent.com/BMClab/BMC/master/"):
    """Find a repository file beside this notebook, or fetch a temporary copy.

    Local files take precedence. Only file acquisition falls back to GitHub;
    parsing errors remain visible to the calling notebook. urllib3 selects its
    browser transport automatically without platform-specific imports.
    """
    import hashlib
    import tempfile
    from pathlib import Path, PurePosixPath
    from urllib.parse import quote

    import marimo as mo
    import urllib3

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise ValueError("Use a repository-relative path such as data/walk.trc")
    location = mo.notebook_dir()
    if location is not None:
        directory = Path(location).resolve()
        for base in (directory, *directory.parents):
            local = base.joinpath(*relative.parts)
            if local.is_file():
                return local
    url = base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")
    key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]
    cached = (Path(tempfile.gettempdir()) / "bmc-notebooks-v2" / key).joinpath(*relative.parts)
    if not cached.is_file():
        response = urllib3.request("GET", url, timeout=30.0, retries=2)
        if response.status != 200:
            raise OSError(f"Cannot load {path}: HTTP {response.status} from {url}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as download:
            temporary = Path(download.name)
            download.write(response.data)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    return cached


@app.cell
def _():
    import sys

    _functions = [repo_file(f"functions/{_m}.py") for _m in ("CCSbasis", "read_trc")]
    sys.path.insert(0, str(_functions[0].parent))  # add to pythonpath
    from CCSbasis import CCSbasis
    from read_trc import read_trc

    return CCSbasis, read_trc


@app.cell
def _(CCSbasis, alpha_slider, beta_slider, gamma_slider, np, rotation_matrix):
    _R = rotation_matrix(
        np.radians(alpha_slider.value), np.radians(beta_slider.value), np.radians(gamma_slider.value)
    )

    _e1 = np.array([[1, 0, 0]])
    _e2 = np.array([[0, 1, 0]])
    _e3 = np.array([[0, 0, 1]])

    _basis = np.vstack((_e1, _e2, _e3))
    _basis_rot = _R @ _basis
    _lv, _v = np.linalg.eig(_R)

    print("Eigenvalues of R:", _lv.round(4))
    if np.isclose(_lv, 1).sum() == 3:  # no rotation: every direction is unchanged
        print("No rotation: R is the identity matrix, and there is no axis to find.")
        _ccs = CCSbasis(Oijk=np.array([0, 0, 0]), Oxyz=np.array([0, 0, 0]), ijk=_basis.T, xyz=_basis_rot.T)
    else:
        _axis_of_rotation = [np.real(np.squeeze(_v[:, np.isclose(_lv, 1)]))]
        print("Axis of rotation (eigenvector of R with eigenvalue 1):", _axis_of_rotation[0].round(4))
        _ccs = CCSbasis(
            Oijk=np.array([0, 0, 0]),
            Oxyz=np.array([0, 0, 0]),
            ijk=_basis.T,
            xyz=_basis_rot.T,
            vector=True,
            point=_axis_of_rotation,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the default angles, a rotation of $45^\circ$ about $\mathbf{Z}$ alone, the axis is $\mathbf{Z}$ itself, $[0, 0, 1]$. Try the sliders: whatever the three angles, the yellow arrow is a single line. Notice also the other two eigenvalues: they are complex.

    **Exercise 1.** Show that a rotation by an angle $\theta$ about $\mathbf{Z}$ has the eigenvalues $1$ and $\cos\theta \pm i\sin\theta$, and check them against the eigenvalues printed for the default $45^\circ$. Then use the fact that the trace of a matrix is the sum of its eigenvalues to find $\operatorname{tr}\mathbf{R}$ in terms of $\theta$.

    /// details | Solution to Exercise 1

    For a rotation about $\mathbf{Z}$,

    $$
    \mathbf{R} - \lambda\mathbf{I} = \begin{bmatrix}\cos\theta - \lambda & -\sin\theta & 0 \\ \sin\theta & \cos\theta - \lambda & 0 \\ 0 & 0 & 1 - \lambda\end{bmatrix},
    \qquad
    \det(\mathbf{R} - \lambda\mathbf{I}) = (1 - \lambda)\left[(\cos\theta - \lambda)^2 + \sin^2\theta\right]
    $$

    The determinant is zero for $\lambda = 1$ (the axis) and for $(\cos\theta - \lambda)^2 = -\sin^2\theta$, that is, $\lambda = \cos\theta \pm i\sin\theta = e^{\pm i\theta}$. For $\theta = 45^\circ$ that is $0.7071 \pm 0.7071i$, as printed. Only directions along the axis are left unchanged; no other real direction is.

    The trace is the sum of the eigenvalues, so

    $$
    \operatorname{tr}\mathbf{R} = 1 + e^{i\theta} + e^{-i\theta} = 1 + 2\cos\theta
    $$

    Every rotation about any axis is a rotation about $\mathbf{Z}$ written in another basis, and a change of basis does not change the eigenvalues, so this holds for every rotation matrix. It is the formula used for the angle in the next cell.

    ///

    The eigenvector gives the line of the axis, but not which way along it the body turned: $-\vec{\mathbf{v}}$ is an eigenvector as well, and NumPy may return either. To get a direction and an angle that belong together, use the matrix itself. The angle comes from the trace, and the axis, pointing so that the rotation is counterclockwise by the right-hand rule, from the antisymmetric part of $\mathbf{R}$:

    $$
    \theta = \arccos\left(\frac{\operatorname{tr}\mathbf{R} - 1}{2}\right),
    \qquad
    \hat{\mathbf{u}} = \frac{1}{2\sin\theta}\begin{bmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{bmatrix}
    $$

    valid for $0 < \theta < 180^\circ$. Conversely, Rodrigues' formula builds the rotation matrix from an axis and an angle. Here are both, and an elemental rotation about one of the fixed axes, as reusable functions:
    """)
    return


@app.function
def axis_angle(R):
    """Axis (versor) and angle [degrees] of the rotation matrix R, for 0 < angle < 180."""
    import numpy as np

    R = np.asarray(R, dtype=float)
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    u = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return u, np.degrees(theta)


@app.function
def rodrigues(u, angle):
    """Rotation matrix for a rotation by `angle` [rad] about the axis `u`."""
    import numpy as np

    u = np.asarray(u, dtype=float) / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


@app.function
def elemental(axis, angle):
    """Rotation matrix for a rotation by `angle` [degrees] about the fixed axis 'X', 'Y' or 'Z'."""
    import numpy as np

    u = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[axis]
    return rodrigues(u, np.radians(angle))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One rotation is always enough

    That the yellow arrow is always a single line is [Euler's rotation theorem](https://en.wikipedia.org/wiki/Euler%27s_rotation_theorem): however a rigid body has been turned, its final orientation can be reached from the initial one by a single rotation about a single axis. Three Cardan angles and one axis with one angle are two descriptions of the same change of orientation.

    **Before you run the next cell**, predict: a body is turned by $45^\circ$ about $\mathbf{Z}$, then $45^\circ$ about $\mathbf{X}$, then $45^\circ$ about $\mathbf{Y}$. What is the angle of the single rotation that does the same? Is it $135^\circ$?
    """)
    return


@app.cell
def _(np, rotation_matrix):
    _R3 = rotation_matrix(np.radians(45), np.radians(45), np.radians(45))  # Z, then X, then Y
    _u, _theta = axis_angle(_R3)
    print(f"Z, then X, then Y: {_theta:.1f} degrees about {_u.round(3)}")
    print("Rodrigues' formula gives back the same matrix:", np.allclose(rodrigues(_u, np.radians(_theta)), _R3))

    _R3_reversed = elemental("Z", 45) @ elemental("X", 45) @ elemental("Y", 45)  # Y, then X, then Z
    _u, _theta = axis_angle(_R3_reversed)
    print(f"Y, then X, then Z: {_theta:.1f} degrees about {_u.round(3)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three rotations of $45^\circ$ amount to a single rotation of only $64.7^\circ$, about an axis close to $\mathbf{X}$. Done in the reverse order, the same three rotations give $85.8^\circ$ about a quite different axis. Angles of finite rotations about different axes do not simply add up.

    **Exercise 2.** A forearm held horizontally is turned by $90^\circ$ about the laboratory's $\mathbf{X}$ axis and then by $90^\circ$ about the laboratory's $\mathbf{Y}$ axis. Find the axis and the angle of the single rotation that takes it from the first orientation to the last. Before computing, guess whether the angle is smaller or larger than $90^\circ + 90^\circ$.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _R = elemental("Y", 90) @ elemental("X", 90)  # X first, then Y (fixed axes)
    _u, _theta = axis_angle(_R)
    _R_check = rodrigues(_u, np.radians(_theta))

    mo.accordion(
        {
            "Solution to Exercise 2": mo.vstack(
                [
                    mo.md(
                        rf"""
    Each new rotation about a fixed axis multiplies the matrix from the left, so $\mathbf{{R}} = \mathbf{{R_Y}}(90^\circ)\,\mathbf{{R_X}}(90^\circ)$, and `axis_angle` gives

    $$
    \theta = {_theta:.1f}^\circ \quad \text{{about}} \quad \hat{{\mathbf{{u}}}} = [{_u[0]:.4f},\; {_u[1]:.4f},\; {_u[2]:.4f}] = \frac{{1}}{{\sqrt{{3}}}}[1,\; 1,\; -1]
    $$

    Rodrigues' formula with this axis and angle gives back the same matrix: {np.allclose(_R_check, _R)}. Two quarter turns about perpendicular axes are a single turn of only $120^\circ$, a third of a full turn, about a diagonal of the cube defined by the axes. Repeat it three times and the body is back where it started.
    """
                    ),
                    mo.md(
                        r"$$\mathbf{R} = \mathbf{R_Y}(90^\circ)\,\mathbf{R_X}(90^\circ) = \begin{bmatrix}"
                        + r" \\ ".join(" & ".join(f"{_v:g}" for _v in _row) for _row in np.round(_R, 4) + 0.0)
                        + r"\end{bmatrix}$$"
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finite rotations are not vectors

    A tempting idea is to represent a rotation by a vector: along the axis, with a length equal to the angle. For rotations about the same axis this works perfectly, since $30^\circ$ followed by $15^\circ$ is $45^\circ$, which is why it works in a plane (see [Angular kinematics in a plane](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py)). Let's see what happens with two different axes, for large and for small angles.
    """)
    return


@app.cell
def _():
    def _rotation_vector(R):
        _u, _theta = axis_angle(R)
        return _u * _theta  # degrees

    for _deg in [45, 1]:
        _composed = elemental("Y", _deg) @ elemental("X", _deg)  # X first, then Y
        _naive_sum = _rotation_vector(elemental("X", _deg)) + _rotation_vector(elemental("Y", _deg))
        print(f"{_deg:2d} degrees about X, then {_deg} about Y:")
        print("   rotation vector of the result [degrees]:", _rotation_vector(_composed).round(4))
        print("   sum of the two rotation vectors:        ", _naive_sum.round(4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For $45^\circ$ the result is far from the sum: a different angle about each axis, and a $-17.7^\circ$ component about $\mathbf{Z}$ that neither rotation had. For $1^\circ$ the sum is almost right; the error, $-0.0087^\circ$, is smaller than the angles by a factor of about a hundred. The smaller the rotations, the better they add, and the order in which they are done stops mattering.

    That is what makes angular velocity a vector. It describes the infinitesimal rotation of a body in an infinitesimal time, $d\theta/dt$, and infinitesimal rotations add like vectors. Angular velocities about different axes can be added, and the order does not matter; angles about different axes cannot.

    **Guiding questions 1.**

    1. Why does the order of two rotations stop mattering when the rotations are small?
    2. The three Cardan angles are angles of *finite* rotations about different axes. What does that suggest about adding up their derivatives?
    3. Go back to your pen in Challenge 0. If the spin and the swing each have an angular velocity, what is the angular velocity of the pen?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computing the angular velocity

    The angular velocity $\vec{\boldsymbol{\omega}}$ is in the direction of the axis of rotation (hence it is parallel to the axis of rotation) and can be described in the basis fixed in the body:

    $$
    \vec{\boldsymbol{\omega}} = \omega_1\hat{\mathbf{e}}_1 + \omega_2\hat{\mathbf{e}}_2 + \omega_3\hat{\mathbf{e}}_3
    $$

    So, we must find $\omega_1$, $\omega_2$ and $\omega_3$, and we will do it using only the versors of the basis and their derivatives.

    Recall that the angular velocity is described as a vector perpendicular to the plane of the rotation: $\vec{\boldsymbol{\omega}}_1 = \frac{d\theta_1}{dt}\hat{\mathbf{e}}_1$, $\vec{\boldsymbol{\omega}}_2 = \frac{d\theta_2}{dt}\hat{\mathbf{e}}_2$ and $\vec{\boldsymbol{\omega}}_3 = \frac{d\theta_3}{dt}\hat{\mathbf{e}}_3$, where $\theta_1$, $\theta_2$ and $\theta_3$ are the infinitesimal angles the body turns about each of its axes.

    Note also that the derivative of the angle $\theta_1$ can be described as the projection of the vector $\frac{d\hat{\mathbf{e}}_2}{dt}$ on the vector $\hat{\mathbf{e}}_3$: as the body turns about $\hat{\mathbf{e}}_1$, the tip of $\hat{\mathbf{e}}_2$ moves towards $\hat{\mathbf{e}}_3$. This can be written with the scalar product between these vectors:

    $$
    \frac{d\theta_1}{dt} = \frac{d\hat{\mathbf{e}}_2}{dt}\cdot \hat{\mathbf{e}}_3
    $$

    <figure><center><img src="https://github.com/BMClab/BMC/blob/master/images/derivVersor.png?raw=1" width=350 alt="derivatives of the versors"/></center><figcaption><center><i>Figure. The versors of the body's basis (black), the same versors a moment later (red), and their derivatives (blue). The rotation $\theta_1$ about $\hat{\mathbf{e}}_1$ moves $\hat{\mathbf{e}}_2$ towards $\hat{\mathbf{e}}_3$.</i></center></figcaption></figure>

    Similarly, the same is valid for the angles in the other two directions:

    $$
    \frac{d\theta_2}{dt} = \frac{d\hat{\mathbf{e}}_3}{dt}\cdot \hat{\mathbf{e}}_1,
    \qquad
    \frac{d\theta_3}{dt} = \frac{d\hat{\mathbf{e}}_1}{dt}\cdot \hat{\mathbf{e}}_2
    $$

    So, we can write the angular velocity as:

    $$
    \vec{\boldsymbol{\omega}} = \left(\frac{d\hat{\mathbf{e}}_2}{dt}\cdot \hat{\mathbf{e}}_3\right) \hat{\mathbf{e}}_1 + \left(\frac{d\hat{\mathbf{e}}_3}{dt}\cdot \hat{\mathbf{e}}_1\right) \hat{\mathbf{e}}_2 + \left(\frac{d\hat{\mathbf{e}}_1}{dt}\cdot \hat{\mathbf{e}}_2\right) \hat{\mathbf{e}}_3
    $$

    **Exercise 3.** A body rotates in the horizontal plane about the fixed $\mathbf{Z}$ axis, by an angle $\theta(t)$, so that $\hat{\mathbf{e}}_1 = [\cos\theta, \sin\theta, 0]$, $\hat{\mathbf{e}}_2 = [-\sin\theta, \cos\theta, 0]$ and $\hat{\mathbf{e}}_3 = [0, 0, 1]$. Apply the formula above and show that it gives back the planar angular velocity.

    /// details | Solution to Exercise 3

    With the chain rule, $\frac{d\hat{\mathbf{e}}_1}{dt} = \dot{\theta}\,[-\sin\theta, \cos\theta, 0] = \dot{\theta}\,\hat{\mathbf{e}}_2$, $\frac{d\hat{\mathbf{e}}_2}{dt} = \dot{\theta}\,[-\cos\theta, -\sin\theta, 0] = -\dot{\theta}\,\hat{\mathbf{e}}_1$ and $\frac{d\hat{\mathbf{e}}_3}{dt} = \vec{\mathbf{0}}$. Then

    $$
    \omega_1 = \frac{d\hat{\mathbf{e}}_2}{dt}\cdot\hat{\mathbf{e}}_3 = 0,
    \qquad
    \omega_2 = \frac{d\hat{\mathbf{e}}_3}{dt}\cdot\hat{\mathbf{e}}_1 = 0,
    \qquad
    \omega_3 = \frac{d\hat{\mathbf{e}}_1}{dt}\cdot\hat{\mathbf{e}}_2 = \dot{\theta}
    $$

    so $\vec{\boldsymbol{\omega}} = \dot{\theta}\,\hat{\mathbf{e}}_3 = \dot{\theta}\,\hat{\mathbf{k}}$: the angular velocity of planar motion, along the axis perpendicular to the plane, as in [Kinematics of a rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py). In a plane, and only there, the angular velocity is the derivative of one angle.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why the formula works

    The argument with the figure can be made exact. In [Kinematics of a rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py), the versors of a body rotating in a plane changed as $d\hat{\mathbf{i}}'/dt = \vec{\boldsymbol{\omega}} \times \hat{\mathbf{i}}'$. The same holds in 3D for every vector fixed in a rigid body, so for each versor of its basis:

    $$
    \frac{d\hat{\mathbf{e}}_i}{dt} = \vec{\boldsymbol{\omega}} \times \hat{\mathbf{e}}_i
    $$

    Then, with the scalar triple product, $(\vec{\mathbf{a}} \times \vec{\mathbf{b}})\cdot\vec{\mathbf{c}} = \vec{\mathbf{a}}\cdot(\vec{\mathbf{b}} \times \vec{\mathbf{c}})$, and $\hat{\mathbf{e}}_2 \times \hat{\mathbf{e}}_3 = \hat{\mathbf{e}}_1$ for a right-handed basis:

    $$
    \frac{d\hat{\mathbf{e}}_2}{dt}\cdot \hat{\mathbf{e}}_3 = \left(\vec{\boldsymbol{\omega}} \times \hat{\mathbf{e}}_2\right)\cdot \hat{\mathbf{e}}_3 = \vec{\boldsymbol{\omega}}\cdot\left(\hat{\mathbf{e}}_2 \times \hat{\mathbf{e}}_3\right) = \vec{\boldsymbol{\omega}}\cdot\hat{\mathbf{e}}_1 = \omega_1
    $$

    and likewise for $\omega_2$ and $\omega_3$. The formula needs the basis to be right-handed: with a left-handed basis every component comes out with the wrong sign.

    Note that the angular velocity $\vec{\boldsymbol{\omega}}$ found this way is expressed in the reference frame of the object. If you want it described as a linear combination of the versors of the global basis $\hat{\mathbf{i}}$, $\hat{\mathbf{j}}$ and $\hat{\mathbf{k}}$, multiply the vector of components by the rotation matrix formed by stacking each versor in a column:

    $$
    \vec{\boldsymbol{\omega}}_{G} = \mathbf{R}\,\vec{\boldsymbol{\omega}}_{l} = \omega_1\hat{\mathbf{e}}_1 + \omega_2\hat{\mathbf{e}}_2 + \omega_3\hat{\mathbf{e}}_3
    $$

    There is also a compact formula that gives the global components directly. Since $\hat{\mathbf{e}}_i \times (\vec{\boldsymbol{\omega}} \times \hat{\mathbf{e}}_i) = \vec{\boldsymbol{\omega}} - (\vec{\boldsymbol{\omega}}\cdot\hat{\mathbf{e}}_i)\,\hat{\mathbf{e}}_i$, summing over the three versors gives $3\vec{\boldsymbol{\omega}} - \vec{\boldsymbol{\omega}}$, so

    $$
    \vec{\boldsymbol{\omega}} = \frac{1}{2}\sum_{i=1}^{3} \hat{\mathbf{e}}_i \times \frac{d\hat{\mathbf{e}}_i}{dt}
    $$

    We will use it as an independent check. Here is the dot-product formula as a function, for versors sampled at a constant rate, one row per instant. The derivatives are central differences from `numpy.gradient`, with `edge_order=2` so that the first and last samples are accurate too:
    """)
    return


@app.function
def angular_velocity(e1, e2, e3, dt):
    """Components of the angular velocity in the body's own basis.

    `e1`, `e2`, `e3` hold the versors of a right-handed basis attached to the
    body, one row per instant, sampled every `dt` seconds. Returns an array
    with the components [omega_1, omega_2, omega_3], in rad/s, one row per
    instant.
    """
    import numpy as np

    de1dt, de2dt, de3dt = (np.gradient(e, dt, axis=0, edge_order=2) for e in (e1, e2, e3))
    omega1 = np.sum(de2dt * e3, axis=1)
    omega2 = np.sum(de3dt * e1, axis=1)
    omega3 = np.sum(de1dt * e2, axis=1)
    return np.column_stack((omega1, omega2, omega3))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A spin we know

    Before trusting the formula with measured data, let's give it a motion with a known answer. A body, initially turned by $30^\circ$ about $\mathbf{X}$, spins at one revolution per second, $\omega = 2\pi$ rad/s, about the fixed axis $\hat{\mathbf{u}} = [1, 1, 1]/\sqrt{3}$. We sample its basis at 100 Hz for one second.

    The angular velocity in the laboratory's basis is $2\pi\,\hat{\mathbf{u}} = [3.6276, 3.6276, 3.6276]$ rad/s at every instant.

    **Before you run the next cell**, predict: will the components in the *body's* basis be the same three numbers? Will they at least be constant?
    """)
    return


@app.cell
def _(np):
    _dt = 0.01  # s, 100 Hz
    _t = np.arange(0, 1 + _dt / 2, _dt)
    _u = np.array([1, 1, 1]) / np.sqrt(3)
    _omega = 2 * np.pi  # rad/s

    _R0 = rodrigues([1, 0, 0], np.radians(30))  # initial orientation
    _Rs = np.array([rodrigues(_u, _omega * _ti) @ _R0 for _ti in _t])  # versors in the columns
    _e1, _e2, _e3 = _Rs[:, :, 0], _Rs[:, :, 1], _Rs[:, :, 2]

    _omega_l = angular_velocity(_e1, _e2, _e3, _dt)  # body components
    _omega_G = np.einsum("nij,nj->ni", _Rs, _omega_l)  # R omega_l, instant by instant

    _derivs = [np.gradient(_e, _dt, axis=0, edge_order=2) for _e in (_e1, _e2, _e3)]
    _omega_G2 = 0.5 * sum(np.cross(_e, _d) for _e, _d in zip((_e1, _e2, _e3), _derivs))

    print("Body components at t = 0, 0.5 and 1 s [rad/s]:")
    for _i in [0, 50, 100]:
        print("  ", _omega_l[_i].round(4))
    print("Expected body components, R0^T (2 pi u):", (_R0.T @ (_omega * _u)).round(4))
    print(f"\nLargest error of the global components: {np.abs(_omega_G - _omega * _u).max():.4f} rad/s")
    print(f"Relative error: {np.abs(_omega_G - _omega * _u).max() / _omega:.1e}; (omega dt)^2 / 6 = {(_omega * _dt) ** 2 / 6:.1e}")
    print(f"Largest difference from the compact formula: {np.abs(_omega_G2 - _omega_G).max():.1e} rad/s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The body components are constant, $[3.63, 4.96, 1.33]$ rad/s, but they are not the laboratory's three equal numbers. For a rotation about a fixed axis, the axis stays put in the laboratory *and* in the body, since the body turns about it. Its direction in the body's basis is whatever it was at the start, $\mathbf{R}_0^T\hat{\mathbf{u}}$, and the body components keep that value throughout. Multiplying by $\mathbf{R}$ at each instant brings them back to the laboratory's $[3.6276, 3.6276, 3.6276]$.

    The largest error is 0.005 rad/s, a relative error of $8 \times 10^{-4}$. It comes from the finite differences: the tip of a versor moves on a circle, and a central difference over an arc of angle $\omega\,\Delta t$ underestimates its speed by a factor of about $1 - (\omega\Delta t)^2/6$. The compact formula agrees with the dot-product formula to within $2 \times 10^{-4}$ rad/s.

    **Challenge 1.** Halve the sampling interval and run the cell again. By what factor does the error change? Is that what $(\omega\,\Delta t)^2/6$ predicts?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: adding angular velocities in the forearm

    Now a motion with two rotations at once, the situation of Challenge 0. The upper arm is held still and vertical, the elbow is flexed at $90^\circ$, and the forearm points forward, along the laboratory's $\mathbf{X}$ axis, with $\mathbf{Y}$ upwards and $\mathbf{Z}$ to the right. For 0.3 s the elbow flexes at a constant $300^\circ$/s about its medio-lateral axis, $\mathbf{Z}$, raising the hand, while the forearm pronates at a constant $400^\circ$/s about its own long axis, $\hat{\mathbf{e}}_1$. The rates are round numbers chosen for the arithmetic, but of the order of those of quick everyday movements.

    The orientation of the forearm is the flexion about the fixed $\mathbf{Z}$ axis composed with the pronation about the forearm's own axis, $\mathbf{R}(t) = \mathbf{R_Z}(\theta_f)\,\mathbf{R}_{\hat{\mathbf{e}}_1}(\theta_p)$, and the dot-product formula needs nothing else.

    **Before you run the next cell**, predict the magnitude of the angular velocity of the forearm: $700^\circ$/s, $500^\circ$/s, or $100^\circ$/s?
    """)
    return


@app.cell
def _(np):
    _dt = 1 / 200  # s, 200 Hz
    t_forearm = np.arange(0, 0.3 + _dt / 2, _dt)
    _w_flex, _w_pron = np.radians(300), np.radians(400)  # rad/s

    # Flexion about the fixed Z axis, pronation about the forearm's long axis (its e1)
    _R = np.array([rodrigues([0, 0, 1], _w_flex * _ti) @ rodrigues([1, 0, 0], _w_pron * _ti) for _ti in t_forearm])
    e1_forearm, _e2, _e3 = _R[:, :, 0], _R[:, :, 1], _R[:, :, 2]

    _omega_l = angular_velocity(e1_forearm, _e2, _e3, _dt)
    omega_forearm = np.einsum("nij,nj->ni", _R, _omega_l)  # laboratory components
    _vector_sum = _w_flex * np.array([0, 0, 1]) + _w_pron * e1_forearm  # the two rotations, added as vectors

    _speed = np.degrees(np.linalg.norm(omega_forearm, axis=1))
    _cos = np.sum(omega_forearm * e1_forearm, axis=1) / np.linalg.norm(omega_forearm, axis=1)
    print(f"|omega|: from {_speed.min():.1f} to {_speed.max():.1f} deg/s")
    print(f"Largest |omega - (flexion + pronation)|: {np.degrees(np.abs(omega_forearm - _vector_sum)).max():.2f} deg/s")
    print(f"Angle between omega and the forearm's long axis: {np.degrees(np.arccos(_cos)).mean():.1f} degrees")
    print("Components in the forearm's basis at t = 0 and t = 0.3 s [deg/s]:")
    print("  ", np.degrees(_omega_l[0]).round(1), "\n  ", np.degrees(_omega_l[-1]).round(1))
    return e1_forearm, omega_forearm, t_forearm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $500^\circ$/s, at every instant. The two angular velocities add as vectors, $\vec{\boldsymbol{\omega}} = \dot{\theta}_f\,\hat{\mathbf{k}} + \dot{\theta}_p\,\hat{\mathbf{e}}_1$, to within $0.4^\circ$/s of finite-difference error, and since the elbow's axis and the forearm's long axis are perpendicular, the magnitudes combine like the sides of a 3-4-5 triangle: $\sqrt{300^2 + 400^2} = 500$. The forearm turns about a single axis at each instant, at $36.9^\circ$ from its long axis, an axis that is neither the elbow's nor the forearm's.

    This is the general rule. If body B turns relative to body A with $\vec{\boldsymbol{\omega}}_{B/A}$, and body C relative to body B with $\vec{\boldsymbol{\omega}}_{C/B}$, then C turns relative to A with

    $$
    \vec{\boldsymbol{\omega}}_{C/A} = \vec{\boldsymbol{\omega}}_{C/B} + \vec{\boldsymbol{\omega}}_{B/A}
    $$

    whatever the axes (Kane and Levinson, 1985). Here A is the upper arm, B a frame that flexes with the forearm but does not pronate, and C the forearm. Read backwards, the rule gives the angular velocity of a joint: the angular velocity of the shank relative to the thigh is $\vec{\boldsymbol{\omega}}_{shank} - \vec{\boldsymbol{\omega}}_{thigh}$, both expressed in the same basis. No such rule exists for the angles.

    In the forearm's own basis the components are $[400, 0, 300]^\circ$/s at the start and $[400, 260.2, -150]^\circ$/s at the end: the pronation part stays along $\hat{\mathbf{e}}_1$, but the flexion part, fixed to the upper arm, turns relative to the forearm as the forearm pronates by $120^\circ$.

    **Exercise 4.** The magnitude of the forearm's angular velocity is constant. Is its direction in the laboratory constant too? Find the angle between $\vec{\boldsymbol{\omega}}$ at $t = 0$ and at $t = 0.3$ s, first by reasoning and then from the computed `omega_forearm`.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, omega_forearm):
    _w0, _w1 = omega_forearm[0], omega_forearm[-1]
    _angle = np.degrees(np.arccos(np.dot(_w0, _w1) / (np.linalg.norm(_w0) * np.linalg.norm(_w1))))
    _predicted = np.degrees(np.arccos(300**2 / 500**2))

    mo.accordion(
        {
            "Solution to Exercise 4": mo.md(
                rf"""
    No. The flexion part, $300^\circ$/s along $\hat{{\mathbf{{k}}}}$, is fixed in the laboratory, but the pronation part, $400^\circ$/s along the forearm's long axis, turns with the forearm, and in 0.3 s the flexion turns the long axis by $90^\circ$, from $\hat{{\mathbf{{i}}}}$ to $\hat{{\mathbf{{j}}}}$. So $\vec{{\boldsymbol{{\omega}}}}(0) = 400\,\hat{{\mathbf{{i}}}} + 300\,\hat{{\mathbf{{k}}}}$ and $\vec{{\boldsymbol{{\omega}}}}(0.3) = 400\,\hat{{\mathbf{{j}}}} + 300\,\hat{{\mathbf{{k}}}}$ (in degrees per second), and

    $$
    \cos\phi = \frac{{\vec{{\boldsymbol{{\omega}}}}(0)\cdot\vec{{\boldsymbol{{\omega}}}}(0.3)}}{{500^2}} = \frac{{300^2}}{{500^2}} = 0.36
    \quad\Longrightarrow\quad \phi = {_predicted:.1f}^\circ
    $$

    From the computed angular velocities, the angle is ${_angle:.1f}^\circ$. A constant-speed rotation can still have an axis that moves: the angular velocity changes, so the forearm has an angular acceleration even though nothing speeds up.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: a 3D pendulum bar

    The file `3dPendulum.txt`, in the `data` folder of this repository, has 3 seconds of data of 3 points of a three-dimensional cylindrical pendulum, sampled at 100 Hz. The pendulum can move in every direction and has a motor at the upper part of the cylindrical bar producing torques to move the bar.

    - The point m1 is at the upper part of the cylinder and is the origin of the system.
    - The point m2 is at the center of mass of the cylinder.
    - The point m3 is a point at the surface of the cylinder.

    Below we compute its angular velocity. First we load the file:
    """)
    return


@app.cell
def _(np):
    np.set_printoptions(precision=4, suppress=True)

    data = np.loadtxt(repo_file("data/3dPendulum.txt"), skiprows=1, delimiter=",")
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And separate each marker in a variable:
    """)
    return


@app.cell
def _(data):
    t = data[:, 0]
    m1 = data[:, 1:4]
    m2 = data[:, 4:7]
    m3 = data[:, 7:]
    dt = t[1]
    return dt, m1, m2, m3, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we form the basis $\hat{\mathbf{e}}_1$, $\hat{\mathbf{e}}_2$ and $\hat{\mathbf{e}}_3$. The first versor is along the bar, from the top to the center of mass; the second is perpendicular to the plane of the three points; the third completes a right-handed basis:
    """)
    return


@app.cell
def _(m1, m2, m3, np):
    V1 = m2 - m1
    e1 = V1 / np.linalg.norm(V1, axis=1, keepdims=True)

    V2 = m3 - m2

    V3 = np.cross(V2, V1)
    e2 = V3 / np.linalg.norm(V3, axis=1, keepdims=True)

    e3 = np.cross(e1, e2)
    return e1, e2, e3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below, we compute the derivative of each of the versors:
    """)
    return


@app.cell
def _(dt, e1, e2, e3, np):
    de1dt = np.gradient(e1, dt, axis=0, edge_order=2)
    de2dt = np.gradient(e2, dt, axis=0, edge_order=2)
    de3dt = np.gradient(e3, dt, axis=0, edge_order=2)
    return de1dt, de2dt, de3dt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we compute each of the components $\omega_1$, $\omega_2$ and $\omega_3$ of the angular velocity $\vec{\boldsymbol{\omega}}$ by using the scalar product:
    """)
    return


@app.cell
def _(de1dt, de2dt, de3dt, e1, e2, e3, np):
    omega1 = np.sum(de2dt * e3, axis=1, keepdims=True)
    omega2 = np.sum(de3dt * e1, axis=1, keepdims=True)
    omega3 = np.sum(de1dt * e2, axis=1, keepdims=True)
    return omega1, omega2, omega3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, the angular velocity vector $\vec{\boldsymbol{\omega}}$ is formed by stacking the three components together. The function `angular_velocity` does the same three steps, and must agree:
    """)
    return


@app.cell
def _(dt, e1, e2, e3, np, omega1, omega2, omega3):
    omega = np.hstack((omega1, omega2, omega3))

    print("Same as angular_velocity()?", np.allclose(omega, angular_velocity(e1, e2, e3, dt)))
    return (omega,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Before you run the next cell**, predict: the pendulum can move in every direction, and its motor can push it any way. Which of the three components of $\vec{\boldsymbol{\omega}}$ will you see changing?
    """)
    return


@app.cell
def _(e1, e2, np, omega, plt, t):
    _R = np.stack((e1, e2, np.cross(e1, e2)), axis=2)  # versors in the columns
    omega_G = np.einsum("nij,nj->ni", _R, omega)
    bar_angle = np.degrees(np.arccos(np.clip(-e1[:, 2], -1, 1)))  # the bar hangs along -Z

    print("Range of each body component [rad/s]:")
    for _name, _col in zip(["omega_1", "omega_2", "omega_3"], omega.T):
        print(f"  {_name}: from {_col.min():8.4f} to {_col.max():8.4f}")
    _moving = np.linalg.norm(omega_G, axis=1) > 1e-3
    _dirs = omega_G[_moving] / np.linalg.norm(omega_G[_moving], axis=1, keepdims=True)
    _dirs = _dirs * np.sign(_dirs[:, :1])
    print("Direction of omega in the laboratory: from", _dirs.min(axis=0).round(4), "to", _dirs.max(axis=0).round(4))
    print(f"Angle of the bar from the vertical: from {bar_angle.min():.1f} to {bar_angle.max():.1f} degrees")

    _, _axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for _i, (_label, _color) in enumerate(
        zip([r"$\omega_1$", r"$\omega_2$", r"$\omega_3$"], ["tab:red", "tab:green", "tab:blue"])
    ):
        _axs[0].plot(t, omega[:, _i], color=_color, linewidth=2, label=_label)
    _axs[0].set_ylabel("Angular velocity [rad/s]")
    _axs[0].set_title("Pendulum bar, components in its own basis")
    _axs[0].legend(loc="lower right", ncol=3)
    _axs[1].plot(t, bar_angle, color="k", linewidth=2)
    _axs[1].set_ylabel("Angle from vertical [deg]")
    _axs[1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    return bar_angle, omega_G


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only $\omega_2$ changes. The spin about the bar's long axis, $\omega_1$, and $\omega_3$ are zero to within $10^{-14}$ rad/s (their two lines lie on top of each other). In the laboratory, the angular velocity always points along the same horizontal line, $\pm[0.5, -0.866, 0]$: in these three seconds the bar swings in a single vertical plane, between $15.3^\circ$ and $20.8^\circ$ from the vertical, with $\omega_2$ reaching $0.31$ rad/s ($17.7^\circ$/s). The motor's torques show up as a swing that changes direction at irregular intervals.

    The pendulum *could* have moved in 3D; it did not, and the formula found that out without being told. That is the kind of result to look for before drawing conclusions from 3D data: a planar motion described with three components.

    ### Checking with the velocity of a point

    The angular velocity predicts the velocity of every point of the bar. The top, m1, does not move, so for the point m3, $\vec{\mathbf{v}}_{m3} = \vec{\boldsymbol{\omega}} \times (\vec{\mathbf{r}}_{m3} - \vec{\mathbf{r}}_{m1})$, as in [Kinematics of a rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py). The prediction can be compared with the velocity of m3 differentiated directly from its positions:
    """)
    return


@app.cell
def _(dt, m1, m2, m3, np, omega_G):
    _v3 = np.gradient(m3, dt, axis=0, edge_order=2)
    _v3_pred = np.cross(omega_G, m3 - m1)

    print(f"Largest speed of m3: {np.linalg.norm(_v3, axis=1).max():.4f} m/s")
    print(f"Largest |v_m3 - omega x r|: {np.abs(_v3 - _v3_pred).max():.1e} m/s")
    for _a, _b, _name in [(m1, m2, "m1-m2"), (m1, m3, "m1-m3"), (m2, m3, "m2-m3")]:
        _d = np.linalg.norm(_b - _a, axis=1)
        print(f"Distance {_name}: from {_d.min():.4f} to {_d.max():.4f} m")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The prediction matches to within $1.5 \times 10^{-5}$ m/s, for a point that moves at up to $0.16$ m/s, and the distances between the three points are constant. The bar is a perfect rigid body, as befits data that come from a simulation. Real markers on a real segment will be less obliging.

    **Exercise 5.** In a plane, integrating the angular velocity over time gives the angle turned. Integrate $\omega_2$ of the pendulum (for example with `scipy.integrate.cumulative_trapezoid`) and compare the result with the change of the bar's angle from the vertical, `bar_angle`. Do they agree? Would the same work for the components of a genuinely 3D motion?
    """)
    return


@app.cell(hide_code=True)
def _(bar_angle, mo, np, omega, t):
    from scipy.integrate import cumulative_trapezoid

    _turned = np.degrees(cumulative_trapezoid(omega[:, 1], t, initial=0))
    _change = bar_angle - bar_angle[0]
    _mismatch = np.abs(_turned + _change).max()

    mo.accordion(
        {
            "Solution to Exercise 5": mo.md(
                rf"""
    The integral of $\omega_2$ goes from ${_turned.min():.2f}^\circ$ to ${_turned.max():.2f}^\circ$, and the change of the bar's angle from the vertical from ${_change.min():.2f}^\circ$ to ${_change.max():.2f}^\circ$. They agree to within ${_mismatch:.3f}^\circ$ at every instant, with opposite signs: with the direction chosen for $\hat{{\mathbf{{e}}}}_2$, a positive $\omega_2$ swings the bar back towards the vertical.

    It works here because the axis of rotation is fixed: $\hat{{\mathbf{{e}}}}_2$ points along the same line throughout, and a rotation about a fixed axis is a planar rotation. For a genuinely 3D motion, the integral of each component of $\vec{{\boldsymbol{{\omega}}}}$ is not an angle of anything: it would be adding up small rotations about axes that keep moving, and finite rotations, as we saw, do not add like vectors. The orientation has to be obtained by composing the small rotations, not by summing their components.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Back to the shank

    Now the question we started with. The file `walk.trc`, in the `data` folder of this repository, is the walking trial used in [Rigid-body transformations in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py), with the laboratory's $\mathbf{X}$ axis pointing forward, $\mathbf{Y}$ upwards and $\mathbf{Z}$ to the right. We build the basis of the right shank exactly as there: $y$ from the ankle to the knee joint center, $x$ perpendicular to $y$ and to the line through the malleoli (lateral minus medial, for the right leg), and $z = x \times y$.

    One detail matters here more than it did there. The `Time` column of the file is rounded to the millisecond, so the difference between its first two values is 0.007 s, while the true interval at 150 Hz is 0.00667 s. Used for the derivatives, that difference would inflate every angular velocity by 5%. The sampling rate in the header is the reliable number:
    """)
    return


@app.cell
def _(np, read_trc):
    _h, _walk = read_trc(str(repo_file("data/walk.trc")), fmt="uni", show_msg=False)
    t_walk = _walk["Time"].to_numpy()
    dt_walk = 1 / _h["data_rate"]
    print(f"First time stamps: {t_walk[:4]} s; header: {_h['data_rate']:.0f} Hz, dt = {dt_walk:.5f} s")

    def _marker(name):
        return _walk[[name + c for c in "xyz"]].to_numpy() / 10  # cm

    _lm, _mm = _marker("R.Ankle"), _marker("R.Ankle.Medial")
    _fh, _mc = _marker("R.HF"), _marker("R.Knee.Medial")
    ajc, kjc = (_lm + _mm) / 2, (_fh + _mc) / 2  # ankle and knee joint centers

    _y = kjc - ajc
    _x = np.cross(_y, _lm - _mm)  # lateral minus medial: right leg, Z to the right
    _z = np.cross(_x, _y)
    x_shank, y_shank, z_shank = (_v / np.linalg.norm(_v, axis=1, keepdims=True) for _v in (_x, _y, _z))
    R_lG = np.stack((x_shank, y_shank, z_shank), axis=1)  # versors in the rows
    return R_lG, ajc, dt_walk, kjc, t_walk, x_shank, y_shank, z_shank


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Cardan angles in the $xyz$ sequence come from the rotation matrix $\mathbf{R_{lG}}$ (the versors in its rows) with the formulas of Rigid-body transformations in 3D:
    """)
    return


@app.function
def cardan_xyz(R_lG):
    """Cardan angles [alpha, beta, gamma] in radians, xyz sequence, from R_lG (versors as rows)."""
    import numpy as np

    R = np.asarray(R_lG, dtype=float)
    alpha = np.arctan2(-R[2, 1], R[2, 2])
    beta = np.arctan2(R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    gamma = np.arctan2(-R[1, 0], R[0, 0])
    return np.array([alpha, beta, gamma])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the two candidates for the angular velocity of the shank: the derivatives of the three Cardan angles, and the components of $\vec{\boldsymbol{\omega}}$ in the shank's own basis from the dot-product formula, about its $x$, $y$ and $z$ axes.

    **Before you run the next cell**, go back to Guiding question 0.1 and commit to an answer: will the two agree?
    """)
    return


@app.cell
def _(R_lG, dt_walk, np, plt, t_walk, x_shank, y_shank, z_shank):
    angles_shank = np.array([cardan_xyz(_R) for _R in R_lG])
    rates_shank = np.gradient(np.unwrap(angles_shank, axis=0), dt_walk, axis=0, edge_order=2)
    omega_shank = angular_velocity(x_shank, y_shank, z_shank, dt_walk)

    _deg = np.degrees
    _speed = np.linalg.norm(omega_shank, axis=1)
    _i = np.argmax(_speed)
    _j = np.argmax(np.abs(rates_shank[:, 2]))
    print(f"Peak |omega|: {_deg(_speed[_i]):.0f} deg/s at t = {t_walk[_i]:.3f} s")
    print(
        f"Peak d(gamma)/dt: {_deg(rates_shank[_j, 2]):.0f} deg/s at t = {t_walk[_j]:.3f} s,"
        f" where omega_z = {_deg(omega_shank[_j, 2]):.0f} deg/s and |omega| = {_deg(_speed[_j]):.0f} deg/s"
    )
    print("Largest |omega - angle rate| for x/alpha, y/beta, z/gamma [deg/s]:", np.abs(_deg(omega_shank - rates_shank)).max(axis=0).round(0))

    _, _axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    _pairs = [("x", r"\alpha"), ("y", r"\beta"), ("z", r"\gamma")]
    for _k, ((_axis, _angle), _color) in enumerate(zip(_pairs, ["tab:red", "tab:green", "tab:blue"])):
        _axs[_k].plot(t_walk, _deg(omega_shank[:, _k]), color=_color, linewidth=2, label=rf"$\omega_{_axis}$")
        _axs[_k].plot(t_walk, _deg(rates_shank[:, _k]), color="k", linewidth=1.5, linestyle="--", label=rf"$d{_angle}/dt$")
        _axs[_k].set_ylabel("[deg/s]")
        _axs[_k].legend(loc="upper right", ncol=2)
    _axs[0].set_title("Right shank: angular velocity (shank's basis) vs. Cardan angle rates")
    _axs[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    return angles_shank, omega_shank, rates_shank


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    They do not agree. The derivative of $\gamma$ peaks at $379^\circ$/s in late swing, but at that instant the angular velocity of the shank about its own $z$ axis is $305^\circ$/s, and its magnitude, the actual rate at which the shank turns, is $329^\circ$/s. The fastest the shank turns during the whole trial is $330^\circ$/s. Component by component, the angle rates differ from the angular velocity by up to $92$, $119$ and $96^\circ$/s. (The spikes at the very last samples are edge effects: there the derivatives can only use samples on one side, and the markers are noisy.)

    Nothing is wrong with either calculation. The angle rates are rates of change of three *finite* rotations about axes that are themselves moving, and finite rotations, as we saw, are not vectors. Only the angular velocity is.

    ### From angle rates to angular velocity

    The two are related, and the relation can be derived from the rotation matrix of the $xyz$ sequence, $\mathbf{R_{lG}} = \mathbf{R_z}(\gamma)\,\mathbf{R_y}(\beta)\,\mathbf{R_x}(\alpha)$, with the elemental matrices of the local (rotating) axes. With the versors in the rows of $\mathbf{R_{lG}}$, the element in row $i$ and column $j$ of $\dot{\mathbf{R}}_{lG}\mathbf{R_{lG}}^T$ is $\frac{d\hat{\mathbf{e}}_i}{dt}\cdot\hat{\mathbf{e}}_j$, so the dot-product formula picks three of its elements. Differentiating with the chain rule, SymPy does the rest:
    """)
    return


@app.cell
def _(mo, sym):
    _a, _b, _g = sym.symbols("alpha beta gamma")
    _ad, _bd, _gd = sym.symbols(r"\dot{\alpha} \dot{\beta} \dot{\gamma}")
    _c, _s = sym.cos, sym.sin

    # Elemental rotation matrices of the local (rotating) axes:
    _Rx = sym.Matrix([[1, 0, 0], [0, _c(_a), _s(_a)], [0, -_s(_a), _c(_a)]])
    _Ry = sym.Matrix([[_c(_b), 0, -_s(_b)], [0, 1, 0], [_s(_b), 0, _c(_b)]])
    _Rz = sym.Matrix([[_c(_g), _s(_g), 0], [-_s(_g), _c(_g), 0], [0, 0, 1]])
    _R = _Rz * _Ry * _Rx  # R_lG for the xyz sequence

    _Rdot = _R.diff(_a) * _ad + _R.diff(_b) * _bd + _R.diff(_g) * _gd
    _W = sym.simplify(_Rdot * _R.T)
    _omega = sym.Matrix([_W[1, 2], _W[2, 0], _W[0, 1]])  # (de2.e3, de3.e1, de1.e2)
    E = sym.simplify(_omega.jacobian([_ad, _bd, _gd]))
    E_func = sym.lambdify((_a, _b, _g), E, "numpy")

    mo.md(
        rf"""
    $$
    \begin{{bmatrix}}\omega_x\\ \omega_y\\ \omega_z\end{{bmatrix}} = {sym.latex(E)}
    \begin{{bmatrix}}\dot{{\alpha}}\\ \dot{{\beta}}\\ \dot{{\gamma}}\end{{bmatrix}},
    \qquad \det \mathbf{{E}} = {sym.latex(sym.simplify(E.det()))}
    $$
    """
    )
    return (E_func,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The matrix $\mathbf{E}$ is the identity only when $\beta = \gamma = 0$; otherwise each component of $\vec{\boldsymbol{\omega}}$ mixes several angle rates. The last row, for instance, says $\omega_z = \sin\beta\;\dot{\alpha} + \dot{\gamma}$: with the shank turned by about $-30^\circ$ about its long axis, the derivative of $\gamma$ overstates the rotation about $z$ by half of $\dot{\alpha}$, which is exactly what happened in late swing. And $\det\mathbf{E} = \cos\beta$ vanishes at $\beta = \pm 90^\circ$: at gimbal lock no set of finite angle rates can represent some angular velocities, the same singularity seen in Rigid-body transformations in 3D.

    Let's check that $\mathbf{E}$ times the measured angle rates gives back the angular velocity from the versors:
    """)
    return


@app.cell
def _(E_func, angles_shank, np, omega_shank, rates_shank):
    _omega_from_rates = np.array([np.asarray(E_func(*_ang), dtype=float) @ _r for _ang, _r in zip(angles_shank, rates_shank)])

    print(f"Largest |E (angle rates) - omega|: {np.degrees(np.abs(_omega_from_rates - omega_shank)).max():.1f} deg/s")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Within $2.2^\circ$/s, against values of up to $330^\circ$/s; the small difference comes from differentiating two different nonlinear functions of the same noisy markers. Angle rates and angular velocity carry the same information, but only the angular velocity can be used directly in mechanics: in $\vec{\mathbf{v}} = \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}$, in the angular momentum, in the kinetic energy of rotation (see [Kinetics of a 3D rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinetics_3d_rigid_body.py)).

    ### Where the shank's angular velocity points

    In the laboratory's basis, $\vec{\boldsymbol{\omega}}_G = \mathbf{R_{lG}}^T\vec{\boldsymbol{\omega}}_l$ (the versors are in the rows of $\mathbf{R_{lG}}$, so its transpose has them in the columns). And the rigid-body prediction can be checked once more, with the velocity of the knee joint center relative to the ankle joint center:
    """)
    return


@app.cell
def _(R_lG, ajc, dt_walk, kjc, np, omega_shank):
    omega_shank_G = np.einsum("nji,nj->ni", R_lG, omega_shank)  # R_lG^T omega_l
    _speed = np.linalg.norm(omega_shank_G, axis=1)
    _fast = _speed > np.radians(30)
    print(
        "Median fraction of |omega| along the laboratory's Z axis (when |omega| > 30 deg/s):",
        f"{np.median(np.abs(omega_shank_G[_fast, 2]) / _speed[_fast]):.2f}",
    )
    print(f"Median angle between the shank's z axis and the laboratory's Z: {np.median(np.degrees(np.arccos(R_lG[:, 2, 2]))):.0f} degrees")

    _v_rel = np.gradient(kjc, dt_walk, axis=0, edge_order=2) - np.gradient(ajc, dt_walk, axis=0, edge_order=2)
    _err = np.linalg.norm(_v_rel - np.cross(omega_shank_G, kjc - ajc), axis=1)
    _r_hat = (kjc - ajc) / np.linalg.norm(kjc - ajc, axis=1, keepdims=True)
    print(
        f"\n|v_knee - v_ankle - omega x r|: median {np.median(_err):.1f} cm/s, largest {_err.max():.1f} cm/s"
        f" (|v_knee - v_ankle| median {np.median(np.linalg.norm(_v_rel, axis=1)):.0f} cm/s)"
    )
    print(f"Largest velocity along the line between the joint centers: {np.abs(np.sum(_v_rel * _r_hat, axis=1)).max():.1f} cm/s")
    return (omega_shank_G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the laboratory, the shank's angular velocity points almost along $\mathbf{Z}$, the medio-lateral axis: in the median instant, 97% of its magnitude. The shank rotates in the sagittal plane, as anyone watching a person walk would expect. In the shank's own basis that same rotation looks less tidy, because the shank's $z$ axis is turned about $24^\circ$ away from the laboratory's $\mathbf{Z}$ (the external rotation discussed in Rigid-body transformations in 3D): part of the sagittal rotation shows up as $\omega_x$, up to $137^\circ$/s. Components are always components *in some basis*, and the basis must be stated.

    The rigid-body prediction for the knee works well in the median instant, with an error of 2.5 cm/s on a relative velocity of 75 cm/s, but misses by up to 20 cm/s at worst. All of that miss is along the line between the joint centers, the one direction $\vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}$ can never produce: the distance between the joint centers is changing. The markers move on the skin, which is the rigidity test of [Kinematics of a rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py) failing in 3D.

    **Challenge 2.** Smooth the marker trajectories with a low-pass filter before building the basis (see [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py)), for example a fourth-order Butterworth filter at 6 Hz, and compute the shank's angular velocity again. How much does its peak change? And the miss along the line between the joint centers?

    **Exercise 6.** At the instant when the shank turns fastest, how far is its axis of rotation from the laboratory's medio-lateral axis $\mathbf{Z}$? Give the angle in degrees.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, omega_shank_G, t_walk):
    _i = np.argmax(np.linalg.norm(omega_shank_G, axis=1))
    _w = np.degrees(omega_shank_G[_i])
    _angle = np.degrees(np.arccos(abs(_w[2]) / np.linalg.norm(_w)))

    mo.accordion(
        {
            "Solution to Exercise 6": mo.md(
                rf"""
    The fastest instant is $t = {t_walk[_i]:.3f}$ s, in late swing, where the angular velocity in the laboratory's basis is $\vec{{\boldsymbol{{\omega}}}}_G = [{_w[0]:.1f},\; {_w[1]:.1f},\; {_w[2]:.1f}]^\circ$/s, with magnitude ${np.linalg.norm(_w):.1f}^\circ$/s. The axis of rotation is the direction of $\vec{{\boldsymbol{{\omega}}}}_G$, and its angle with $\mathbf{{Z}}$ comes from the dot product with $\hat{{\mathbf{{k}}}}$:

    $$
    \phi = \arccos\frac{{|\omega_Z|}}{{\Vert\vec{{\boldsymbol{{\omega}}}}_G\Vert}} = \arccos\frac{{{abs(_w[2]):.1f}}}{{{np.linalg.norm(_w):.1f}}} = {_angle:.1f}^\circ
    $$

    At its fastest, the shank turns about an axis within ${_angle:.0f}^\circ$ of the medio-lateral direction, much closer than its own $z$ axis, which is about $24^\circ$ away.
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example: a gyroscope on the shank

    Wearable inertial sensors are an alternative to cameras. Their gyroscopes measure the angular velocity directly, as the three components in the sensor's own basis, the same vector our formula produces from markers. What they cannot do is know how they were strapped on.

    Suppose two sensors are strapped to the same right shank, each turned by $20^\circ$ about the shank's long axis relative to the shank's anatomical basis, one inwards (towards the other leg) and one outwards. Both are attached to the same rigid body, so both experience the same angular velocity; each reports it in its own basis, $\vec{\boldsymbol{\omega}}_{sensor} = \mathbf{R}_{s/l}\,\vec{\boldsymbol{\omega}}_l$, where the rows of $\mathbf{R}_{s/l}$ are the sensor's versors written in the shank's basis.

    **Before you run the next cell**, predict: the shank's peak angular velocity about its own $z$ axis is $307^\circ$/s. Will the $z$ gyroscopes of the two sensors read more, less, or the same?
    """)
    return


@app.cell
def _(np, omega_shank):
    def _sensor_basis(theta_deg):
        """Versors (rows) of a sensor turned by theta about the shank's y axis, in the shank's basis."""
        _c, _s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
        return np.array([[_c, 0, -_s], [0, 1, 0], [_s, 0, _c]])

    print(f"Shank's own z axis:         peak omega_z = {np.degrees(omega_shank[:, 2]).max():5.1f} deg/s")
    for _theta, _name in [(20, "turned 20 deg inwards: "), (-20, "turned 20 deg outwards:")]:
        _omega_sensor = omega_shank @ _sensor_basis(_theta).T
        _deg = np.degrees(_omega_sensor)
        print(
            f"Sensor {_name} peak omega_z = {_deg[:, 2].max():5.1f} deg/s, omega_x from {_deg[:, 0].min():6.1f}"
            f" to {_deg[:, 0].max():5.1f}, peak |omega| = {np.linalg.norm(_deg, axis=1).max():.1f} deg/s"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Neither reads $307^\circ$/s. The sensor turned inwards reads a peak of $330.1^\circ$/s on its $z$ gyroscope, more than the shank's own $z$ component and almost the whole magnitude, while its $x$ gyroscope stays within $-36$ and $57^\circ$/s. The sensor turned outwards reads only $247.4^\circ$/s about $z$, and up to $221^\circ$/s about $x$. Both agree on the magnitude, $330.3^\circ$/s, the one quantity that does not depend on the basis.

    The inward turn happens to undo the shank basis's external rotation of about $24^\circ$, so it lines the sensor's $z$ axis up with the actual axis of rotation of walking, close to $\mathbf{Z}$. A sensor's components are only as meaningful as its alignment, which is why inertial sensors are calibrated with known postures or movements before their readings are labelled "flexion" or "rotation".

    **Guiding questions 2.**

    1. A sensor strapped turned by $90^\circ$ about the long axis would read the sagittal rotation on which gyroscope?
    2. Why can no mounting change the reading of the $y$ gyroscope in this example?
    3. If you know the magnitude of $\vec{\boldsymbol{\omega}}$ and its component along one axis, what do you know about its direction?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    Pause here before the problems.

    1. Go back to your pen in Challenge 0. Is there a single axis of rotation at each instant? What is the angular velocity of the pen in terms of the spin and the swing?
    2. Why can angular velocities about different axes be added as vectors, while angles of rotation about different axes cannot?
    3. The dot-product formula gives $\omega_1$, $\omega_2$, $\omega_3$. In which basis are they components, and what do you need to express $\vec{\boldsymbol{\omega}}$ in the laboratory's basis?
    4. What happens to the formula if the basis built from the markers is left-handed?
    5. A colleague reports the "angular velocity of the shank about its long axis" as the derivative of $\beta$. Under what conditions would that be correct?
    6. The pendulum's angular velocity always pointed along one horizontal line. What would you have to see in $\vec{\boldsymbol{\omega}}$ to conclude that a movement is genuinely three-dimensional?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Initially, the lateral malleolus, medial malleolus, fibular head and medial condyle of the leg have the following positions (described in the laboratory coordinate system with coordinates $x, y, z$ in cm, the $x$ axis pointing forward and the $y$ axis pointing upward): lateral malleolus (lm = [2.92, 10.10, 18.85]), medial malleolus (mm = [2.71, 10.22, 26.52]), fibular head (fh = [5.05, 41.90, 15.41]), and medial condyle (mc = [8.29, 41.88, 26.52]). After 0.05 seconds the markers have the following positions: lateral malleolus (lm = [2.95, 10.19, 18.41]), medial malleolus (mm = [3.16, 10.04, 26.10]), fibular head (fh = [4.57, 42.13, 15.97]), and medial condyle (mc = [8.42, 41.76, 26.90]). Estimate the average angular velocity of the leg over the interval. State the basis construction and finite-rotation or finite-difference convention; two frames do not identify its instantaneous time course.

    2. Differentiating $\hat{\mathbf{e}}_2\cdot\hat{\mathbf{e}}_3 = 0$, show that $\frac{d\hat{\mathbf{e}}_2}{dt}\cdot\hat{\mathbf{e}}_3 = -\frac{d\hat{\mathbf{e}}_3}{dt}\cdot\hat{\mathbf{e}}_2$, and write a second formula for each component of $\vec{\boldsymbol{\omega}}$. Compute the shank's angular velocity with both formulas: do they agree with measured data?

    3. A body rotates with a constant angular velocity $\vec{\boldsymbol{\omega}} = [0, 0, 1]$ rad/s in its own basis. Using the matrix $\mathbf{E}$ of the $xyz$ sequence, calculate the Cardan angle rates when $\beta = 0$, $60^\circ$ and $89^\circ$ (with $\gamma = 0$). What happens as $\beta$ approaches $90^\circ$?

    4. Derive the matrix $\mathbf{E}$ for the $zxy$ sequence, which puts the rotation about the medio-lateral axis first, and use it with the shank's angular velocity to obtain the angle rates. Compare the rate of the medio-lateral angle with $\omega_z$. Is the difference smaller than for the $xyz$ sequence?

    5. Compute the angular velocity of the left shank in `walk.trc`, with the markers `L.Ankle`, `L.Ankle.Medial`, `L.HF` and `L.Knee.Medial`. Which difference between the malleoli must you use for the $x$ axis to point forward? Compare the peak angular velocity of the two shanks, and explain the difference in their timing.

    6. Build a basis for the right thigh in `walk.trc` from the greater trochanter (`R.GTR`) and the lateral and medial knee markers (`R.Knee`, `R.Knee.Medial`), compute its angular velocity, and obtain the angular velocity of the shank relative to the thigh, $\vec{\boldsymbol{\omega}}_{shank} - \vec{\boldsymbol{\omega}}_{thigh}$, expressed in the thigh's basis. How much of it is about the thigh's medio-lateral axis? Does the knee behave like a hinge, and what in the marker set could make it look less like one than it is? (The virtual hip joint center in the file, `V_R.Hip_JC`, is empty.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Final answers to the problems

    1. The two frames determine an interval estimate, not a unique instantaneous angular velocity. With the right-shank basis used here and the rotation-vector logarithm over 0.05 s, the laboratory estimate is $(0.69218,1.09415,0.21852)$ rad/s. In the initial shank basis it is $(-0.56310,1.15621,-0.26479)$ rad/s. Different finite-difference estimators agree only in the small-step limit.
    2. $\omega_1=-\dot e_3\cdot e_2$, $\omega_2=-\dot e_1\cdot e_3$, $\omega_3=-\dot e_2\cdot e_1$. Exact differentiable orthonormal bases give identical formulas. Numerical differences of noisy sampled bases need not preserve the product rule exactly, so small discrepancies remain.
    3. With the stated $\omega=(0,0,1)$ and $\gamma=0$, the rates are $(\dot\alpha,\dot\beta,\dot\gamma)=(0,0,1)$ rad/s at all three specified angles. They do **not** diverge for this particular vector. At $\beta=90^\circ$, the rates are nonunique: $\dot\beta=0$ and $\dot\alpha+\dot\gamma=1$. By contrast, the input $(1,0,0)$ gives $(\sec\beta,0,-\tan\beta)$ away from the singularity.
    4. For intrinsic $zxy$ angles $(a,b,c)$ and body angular-velocity components ordered $(x,y,z)$, $E=[[-\sin c\cos b,\cos c,0],[\sin b,0,1],[\cos c\cos b,\sin c,0]]$. For this recording, RMS difference between the first $zxy$ angle rate and body $\omega_z$ is about 16.29°/s, versus 38.41°/s for the $xyz$ medio-lateral angle rate. It is smaller here, not guaranteed for every movement.
    5. Use medial minus lateral for the left shank’s temporary medio-lateral direction. Peak magnitudes are about 330.35°/s on the right at 0.293 s and 348.17°/s on the left at 0.980 s. Their timing reflects alternating limbs and the trial’s chosen start, with natural asymmetry and measurement error.
    6. In the thigh basis, the medio-lateral component has peak magnitude about 392.58°/s and RMS magnitude about 90.9% of the RMS total relative angular-speed magnitude. It is predominantly hinge-like, with other components present. Greater-trochanter placement, skin movement and the approximate thigh long axis can create apparent non-hinge motion.

    ///
    """)
    return









@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    - [Kinematics of a rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_of_rigid_body.py): the planar angular velocity, $\vec{\mathbf{v}} = \vec{\boldsymbol{\omega}} \times \vec{\mathbf{r}}$ and the rigidity test.
    - [Rigid-body transformations in 3D](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py): rotation matrices, Cardan angles, gimbal lock and the stride used here.
    - [Frame of reference](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py) and [Determination of a coordinate system](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/coordinate_system.py): building the body's basis from markers, and how marker errors become errors in its axes.
    - [Kinetics of a 3D rigid body](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinetics_3d_rigid_body.py): where the angular velocity is used, in the angular momentum and the Newton–Euler equations.
    - [Data filtering](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/data_filtering.py): what to do before differentiating measured positions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Video lectures on the Internet

    [Lynch and Park: rotation matrices](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-1-of-2/) explains orthogonality, right-handedness and what each matrix column represents. Relate the displayed axes to the marker-derived basis in this guide.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Kane T, Levinson D (1985) [Dynamics: Theory and Applications](https://ecommons.cornell.edu/handle/1813/638). McGraw-Hill, Inc.
    """)
    return


if __name__ == "__main__":
    app.run()
