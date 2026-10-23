# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Direct Linear Transformation (DLT)

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    Two cameras see a marker at different pixel coordinates. How do those two images become one three-dimensional position in metres or millimetres? We will calibrate a projective camera model from known points, then reconstruct an unknown point.

    First work through homogeneous coordinates and the number of independent parameters. Run the preserved two-camera example, and use the Exercises to distinguish calibration error, reconstruction error and geometric degeneracy. The Problems provide final answers only. You need matrices and least squares.

    **Opening challenge.** Can a single calibrated image tell whether a marker is nearby or far away along its viewing ray? What extra information would remove the ambiguity?

    ## Python setup

    NumPy supplies linear algebra. This file contains the attributed DLT implementation used when the optional `dltx` package is absent, so no installation cell is necessary. The example coordinates are embedded in the notebook.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Camera calibration and point reconstruction based on DLT

    The fundamental problem here is to find a mathematical relationship between the coordinates of a 3D point and its projection onto the image plane. DLT estimates a homogeneous projective mapping with linear algebra. The underlying geometric assumption is a distortion-free pinhole camera; the pixel coordinates require division by a depth-dependent homogeneous scale.<br>
    In simplistic terms, using a pinhole camera model, it can be found by similar triangles the following relation between the image coordinates (u,v) and the 3D point (X,Y,Z):

    $$
    s\begin{bmatrix} u \\ v\\ 1 \end{bmatrix} =
    \begin{bmatrix} L1 &  L2 & L3 & L4 \\\ L5 & L6 & L7 & L8 \\\ L9 & L10 & L11 & L12 \end{bmatrix}
    \begin{bmatrix} X \\\ Y \\\ Z \\\ 1 \end{bmatrix}
    $$

    The matrix $L$ is the camera projection matrix. The scale $s$ varies from point to point: divide the first two projected homogeneous coordinates by the third to recover pixel coordinates. This is the [pinhole model in the current OpenCV documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). For a 2D point (X,Y), this matrix is 3x3. The matrix is determined only up to a nonzero scale, so it has 11 independent parameters (8 for planar DLT). Fixing the last entry to one is a valid normalization only when that entry is nonzero.

    There are more accurate (but more complex) algorithms for camera calibration that also consider lens distortion. For example, OpenCV and Tsai softwares have been ported to Python. However, DLT is classic, simple, and effective (fast) for most applications.

    DLT is typically used in two steps: 1. Camera calibration. 2. Object (point) reconstruction.

    The camera calibration step consists in digitizing points with known coordinates in the real space and find the camera parameters.<br>
    At least 4 points are necessary for the calibration of a plane (2D DLT) and at least 6 points for the calibration of a volume (3D DLT). For the 2D DLT, at least one view of the object (points) must be entered. A single camera can be calibrated from known 3D points; reconstructing an unconstrained unknown 3D point requires at least two distinct calibrated views. Calibration points must also avoid degenerate configurations: a point count alone does not guarantee a unique fit.

    These coordinates (from the object and image(s)) are inputed to a DLT algorithm which estimates the camera parameters (8 for 2D DLT and 11 for 3D DLT).

    Usually it is used more points than the minimum necessary and the overdetermined linear system is solved by a least squares minimization algorithm or using singular value decomposition (SVD).

    With these camera parameters and with the camera(s) at the same position of the calibration step, we now can reconstruct the real position of any point inside the calibrated space (area for 2D DLT and volume for the 3D DLT) from the point position(s) viewed by the same fixed camera(s).

    See more about DLT at [https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html).
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** A camera maps a point to homogeneous image coordinates $(600,300,3)$. What are the pixel coordinates? Does multiplying all camera coefficients by 5 change them?

    /// details | Worked solution — Exercise 1

    Dividing by the third coordinate gives $(200,100)$ pixels. A common nonzero scale cancels in both ratios, so multiplying the matrix by 5 gives the same pixels.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python code for camera calibration and point reconstruction based on DLT

    > [dltx](https://pypi.org/project/dltx/): This package implements camera calibration and point reconstruction by direct linear transformation (DLT).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The executable example uses the embedded NumPy implementation if the optional `dltx` package is unavailable. Its original package attribution and example are retained below.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** How many scalar image equations come from six 3D calibration points? Why does that not prove that every six-point set works?

    /// details | Worked solution — Exercise 2

    Each point supplies two equations, hence twelve equations for eleven projective degrees of freedom. Their independence also matters: coplanar or otherwise degenerate configurations do not identify a general volumetric camera.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example from [dltx](https://github.com/joonaojapalo/dltx)
    """)
    return


@app.cell
def _(dlt_calibrate_1, dlt_reconstruct_1):
    try:
        from dltx import dlt_calibrate, dlt_reconstruct
    except ImportError:
        # dltx is not installed: use the copy of its code in the cell below
        print('dltx is not installed; using the copy of its code in this notebook.')
        dlt_calibrate, dlt_reconstruct = dlt_calibrate_1, dlt_reconstruct_1

    # Define locations of 6 or more points in real world.
    world_positions = [
        [0,     0,      2550],  # point 1
        [0,     0,      0],     # point 2
        [0,     2632,   0   ],  # point 3
        [4500,  0,      2550],  # point 4
        [5000,  0,      0   ],  # point 5
        [5660,  2620,   0   ]   # point 6
    ]

    # Define pixel coordinates of respective points seen by two or more cameras.
    cameras = [
        # Camera 1
        [
            [1810, 885],
            [1353, 786],
            [1362, 301],
            [455, 1010],
            [329, 832],
            [183, 180]
        ],
        # Camera 2
        [
            [1734, 952],
            [1528, 768],
            [1546, 135],
            [115, 834],
            [459, 719],
            [358, 202]
        ]
    ]

    # Calibrate cameras
    n_dims = 3
    L1, err = dlt_calibrate(n_dims, world_positions, cameras[0])
    L2, err = dlt_calibrate(n_dims, world_positions, cameras[1])
    camera_calibration = [L1, L2]

    # Find world coordinates for `query_point` visible in both cameras
    query_point = [
        [1810, 885], # cam 1
        [1734, 952]  # cam 2
        ]
    dlt_reconstruct(n_dims, len(cameras), camera_calibration, query_point)
    # coordinates in real world: [-1.31704156e-01,  8.71539661e-01,  2.54975288e+03]
    return


@app.cell
def _():
    __author__ = 'Marcos Duarte <duartexyz@gmail.com>'
    __version__ = 'dlt.py v.0.1.0 2023/01/16'
    import numpy as np

    def dlt_calibrate_1(n_dims, xyz, uv):
        """
        Camera calibration by DLT with known object points and image points
        This code performs 2D or 3D DLT camera calibration with any number of
            views (cameras).
        For 3D DLT, at least two views (cameras) are necessary.
        Inputs:
            n_dims is the number of dimensions of the object space: 3 for 3D DLT
            and 2 for 2D DLT.
            xyz are the coordinates in the object 3D or 2D space of the
            calibration points.
            uv are the coordinates in the image 2D space of these calibration
            points.
            The coordinates (x,y,z and u,v) are given as columns and the different
            points as rows.
            For the 2D DLT (object planar space), only the first 2 columns
            (x and y) are used.
            There must be at least 6 calibration points for the 3D DLT and 4
            for the 2D DLT.
        Outputs:
            L: array of the 8 or 11 parameters of the calibration matrix.
            err: error of the DLT (mean residual of the DLT transformation in units
            of camera coordinates).
        """
        xyz = np.asarray(xyz)
        uv = np.asarray(uv)
        n_points = xyz.shape[0]
        if uv.shape[0] != n_points:
            raise ValueError('xyz (%d points) and uv (%d points) have different number of points.' % (n_points, uv.shape[0]))
        if n_dims == 2 and xyz.shape[1] != 2 or (n_dims == 3 and xyz.shape[1] != 3):
            raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' % (xyz.shape[1], n_dims, n_dims))
        if n_dims == 3 and n_points < 6 or (n_dims == 2 and n_points < 4):
            raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' % (n_dims, 2 * n_dims, n_points))
        (Txyz, xyzn) = normalize(n_dims, xyz)
        (Tuv, uvn) = normalize(2, uv)
        A = []
        if n_dims == 2:
            for i in range(n_points):
                (x, y) = (xyzn[i, 0], xyzn[i, 1])
                (u, v) = (uvn[i, 0], uvn[i, 1])
                A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
                A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
        elif n_dims == 3:
            for i in range(n_points):
                (x, y, z) = (xyzn[i, 0], xyzn[i, 1], xyzn[i, 2])
                (u, v) = (uvn[i, 0], uvn[i, 1])
                A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
                A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
        A = np.asarray(A)
        (U, S, Vh) = np.linalg.svd(A)
        L = Vh[-1, :] / Vh[-1, -1]
        H = L.reshape(3, n_dims + 1)
        H = np.dot(np.dot(np.linalg.pinv(Tuv), H), Txyz)
        H = H / H[-1, -1]
        L = H.flatten()
        uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
        uv2 = uv2 / uv2[2, :]
        err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv) ** 2, 1)))
        return (L, err)

    def dlt_reconstruct_1(n_dims, n_cams, Ls, uvs):
        """
        Reconstruction of object point from image point(s) based on the DLT parameters.
        This code performs 2D or 3D DLT point reconstruction with any number of
            views (cameras).
        For 3D DLT, at least two views (cameras) are necessary.
        Inputs:
            n_dims is the number of dimensions of the object space: 3 for 3D DLT
            and 2 for 2D DLT.
            n_cams is the number of cameras (views) used.
            Ls (array type) are the camera calibration parameters of each camera
            (is the output of DLTcalib function). The Ls parameters are given
            as columns and the Ls for different cameras as rows.
            uvs are the coordinates of the point in the image 2D space of each camera.
            The coordinates of the point are given as columns and the different
            views as rows.
        Outputs:
            xyz: point coordinates in space.
        """
        Ls = np.asarray(Ls)
        if Ls.ndim == 1 and n_cams != 1:
            raise ValueError('Number of views (%d) and number of sets of camera calibration parameters (1) are different.' % n_cams)
        if Ls.ndim > 1 and n_cams != Ls.shape[0]:
            raise ValueError('Number of views (%d) and number of sets of camera calibration parameters (%d) are different.' % (n_cams, Ls.shape[0]))
        if n_dims == 3 and Ls.ndim == 1:
            raise ValueError('At least two sets of camera calibration parametersd are neede for 3D point reconstruction.')
        if n_cams == 1:
            Hinv = np.linalg.inv(Ls.reshape(3, 3))
            xyz = np.dot(Hinv, [uvs[0], uvs[1], 1])
            xyz = xyz[0:2] / xyz[2]
        else:
            M = []
            for i in range(n_cams):
                L = Ls[i, :]
                (u, v) = (uvs[i][0], uvs[i][1])
                if n_dims == 2:
                    M.append([L[0] - u * L[6], L[1] - u * L[7], L[2] - u * L[8]])
                    M.append([L[3] - v * L[6], L[4] - v * L[7], L[5] - v * L[8]])
                elif n_dims == 3:
                    M.append([L[0] - u * L[8], L[1] - u * L[9], L[2] - u * L[10], L[3] - u * L[11]])
                    M.append([L[4] - v * L[8], L[5] - v * L[9], L[6] - v * L[10], L[7] - v * L[11]])
            (U, S, Vh) = np.linalg.svd(np.asarray(M))
            xyz = Vh[-1, 0:-1] / Vh[-1, -1]
        return xyz

    def normalize(n_dims, x):
        """Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3)).
        Inputs:
            n_dims: number of dimensions (2 for 2D; 3 for 3D)
            x: the data to be normalized (directions at different columns and points at rows)
        Outputs:
            Tr: the transformation matrix (translation plus scaling)
            x: the transformed data
        """
        x = np.asarray(x)
        (m, s) = (np.mean(x, 0), np.std(x))
        if n_dims == 2:
            Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
        else:
            Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        Tr = np.linalg.inv(Tr)
        x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
        x = x[0:n_dims, :].T
        return (Tr, x)

    return dlt_calibrate_1, dlt_reconstruct_1


@app.cell
def _():
    import marimo as mo

    return (mo,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. What is lost by projecting a 3D point into one camera?
    2. Can moving a calibrated camera leave its old matrix valid?
    3. Is a low calibration residual an independent validation?

    /// details | Checkpoint answers

    1. Depth along the viewing ray. 2. No; its extrinsic parameters change. 3. No; test known points held out from fitting and assess reconstruction across the measurement volume.

    ///
    """)
    return






@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. For $P=[[1000,0,0,0],[0,1000,0,0],[0,0,1,0]]$, give the image of $(2,1,10)$ and $(4,2,20)$.
    2. A planar calibration uses four collinear points. Is the homography unique?
    3. Two cameras give nearly parallel viewing rays. What happens to depth uncertainty?

    /// details | Final answers to the problems

    1. Both project to $(200,100)$ pixels. 2. No. 3. Depth uncertainty grows greatly; a small pixel error can produce a large depth error.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Read [SVDalgorithm](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/svd_algorithm.py) for SVD and geometric conditioning.

    ### Video lectures on the Internet

    [Lynch and Park: rotation matrices](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-1-rotation-matrices-part-1-of-2/) explains rigid geometry and coordinate changes; compare those distance-preserving transformations with the fitted mappings in this guide.

    ## References

    - [OpenCV camera calibration documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), homogeneous camera model and distortion.

    - [The dltx project](https://github.com/joonaojapalo/dltx), source of the attributed implementation and example.
    """)
    return


if __name__ == "__main__":
    app.run()
