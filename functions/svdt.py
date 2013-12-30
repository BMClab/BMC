#!/usr/bin/env python

"""Calculates the transformation between two coordinate systems using SVD."""

__author__ = 'Marcos Duarte'
__version__ = 'svdt.py v.1 2013/12/23'

import numpy as np


def svdt(A, B, order='col'):
    """Calculates the transformation between two coordinate systems using SVD.

    This function determines the rotation matrix (R) and the translation vector
    (L) for a rigid body after the following transformation [1]_, [2]_:
    B = R*A + L + err.
    Where A and B represents the rigid body in different instants and err is an
    aleatory noise (which should be zero for a perfect rigid body). A and B are
    matrices with the marker coordinates at different instants (at least three
    non-collinear markers are necessary to determine the 3D transformation).

    The matrix A can be thought to represent a local coordinate system (but A
    it's not a basis) and matrix B the global coordinate system. The operation
    Pg = R*Pl + L calculates the coordinates of the point Pl (expressed in the
    local coordinate system) in the global coordinate system (Pg).

    A typical use of the svdt function is to calculate the transformation
    between A and B (B = R*A + L), where A is the matrix with the markers data
    in one instant (the calibration or static trial) and B is the matrix with
    the markers data for one or more instants (the dynamic trial).

    If the parameter order='row', the A and B parameters should have the shape
    (n, 3), i.e., n rows and 3 columns, where n is the number of markers.
    If order='col', A can be a 1D array with the shape (n*3, like
    [x1, y1, z1, ..., xn, yn, zn] and B a 1D array with the same structure of A
    or a 2D array with the shape (ni, n*3) where ni is the number of instants.
    The output R has the shape (ni, 3, 3), L has the shape (ni, 3), and RMSE
    has the shape (ni,). If ni is equal to one, the outputs will have the
    singleton dimension dropped.
    
    Part of this code is based on the programs written by Alberto Leardini,
    Christoph Reinschmidt, and Ton van den Bogert.

    Parameters
    ----------
    A   : Numpy array
        Coordinates [x,y,z] of at least three markers with two possible shapes:
        order='row': 2D array (n, 3), where n is the number of markers.
        order='col': 1D array (3*nmarkers,) like [x1, y1, z1, ..., xn, yn, zn].
    B   : 2D Numpy array
        Coordinates [x,y,z] of at least three markers with two possible shapes:
        order='row': 2D array (n, 3), where n is the number of markers.
        order='col': 2D array (ni, n*3), where ni is the number of instants.
        If ni=1, B is a 1D array like A.
    order : string
        'col': specifies that A and B are column oriented (default).
        'row': specifies that A and B are row oriented.

    Returns
    -------
    R   : Numpy array
        Rotation matrix between A and B with two possible shapes:
        order='row': (3, 3).
        order='col': (ni, 3, 3), where ni is the number of instants.
        If ni=1, R will have the singleton dimension dropped.
    L   : Numpy array
        Translation vector between A and B with two possible shapes:
        order='row': (3,) if order = 'row'.
        order='col': (ni, 3), where ni is the number of instants.
        If ni=1, L will have the singleton dimension dropped.
    RMSE : array
        Root-mean-squared error for the rigid body model: B = R*A + L + err
        with two possible shapes:
        order='row': (1,).
        order='col': (ni,), where ni is the number of instants.

    See Also
    --------
    numpy.linalg.svd

    Notes
    -----
    The singular value decomposition (SVD) algorithm decomposes a matrix M
    (which represents a general transformation between two coordinate systems)
    into three simple transformations [3]_: a rotation Vt, a scaling factor S
    along the  rotated axes and a second rotation U: M = U*S*Vt.
    The rotation matrix is given by: R = U*Vt.

    References
    ----------
    .. [1] Soderkvist, Kedin (1993) Journal of Biomechanics, 26, 1473-1477.
    .. [2] http://www.kwon3d.com/theory/jkinem/rotmat.html.
    .. [3] http://en.wikipedia.org/wiki/Singular_value_decomposition.


    Examples
    --------
    >>> import numpy as np
    >>> from svdt import svdt
    >>> A = np.array([0,0,0, 1,0,0,  0,1,0,  1,1,0])  # four markers
    >>> B = np.array([0,0,0, 0,1,0, -1,0,0, -1,1,0])  # four markers
    >>> R, L, RMSE = svdt(A, B)
    >>> B = np.vstack((B, B))  # simulate two instants (two rows)
    >>> R, L, RMSE = svdt(A, B)
    >>> A = np.array([[0,0,0], [1,0,0], [ 0,1,0], [ 1,1,0]])  # four markers
    >>> B = np.array([[0,0,0], [0,1,0], [-1,0,0], [-1,1,0]])  # four markers
    >>> R, L, RMSE = svdt(A, B, order='row')
    """
    
    A, B = np.asarray(A), np.asarray(B)
    if order == 'row' or B.ndim == 1:
        if B.ndim == 1:
            A = A.reshape(A.size/3, 3)
            B = B.reshape(B.size/3, 3)
        R, L, RMSE = _svd(A, B)
    else:
        A = A.reshape(A.size/3, 3)
        ni = B.shape[0]
        R = np.empty((ni, 3, 3))
        L = np.empty((ni, 3))
        RMSE = np.empty(ni)
        for i in range(ni):
            R[i, :, :], L[i, :], RMSE[i] = _svd(A, B[i, :].reshape(A.shape))

    return R, L, RMSE


def _svd(A, B):
    """Calculates the transformation between two coordinate systems using SVD.

    See the help of the svdt function.

    Parameters
    ----------
    A   : 2D Numpy array (n, 3), where n is the number of markers.
        Coordinates [x,y,z] of at least three markers
    B   : 2D Numpy array (n, 3), where n is the number of markers.
        Coordinates [x,y,z] of at least three markers

    Returns
    -------
    R    : 2D Numpy array (3, 3)
         Rotation matrix between A and B
    L    : 1D Numpy array (3,)
         Translation vector between A and B
    RMSE : float
         Root-mean-squared error for the rigid body model: B = R*A + L + err.

    See Also
    --------
    numpy.linalg.svd
    """

    Am = np.mean(A, axis=0)           # centroid of m1
    Bm = np.mean(B, axis=0)           # centroid of m2
    M = np.dot((B - Bm).T, (A - Am))  # considering only rotation
    # singular value decomposition
    U, S, Vt = np.linalg.svd(M)
    # rotation matrix
    R = np.dot(U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt))
    # translation vector
    L = B.mean(0)  - np.dot(R, A.mean(0))
    # RMSE
    err = 0
    for i in range(A.shape[0]):
        Bp = np.dot(R, A[i, :]) + L
        err += np.sum((Bp - B[i, :])**2)
    RMSE = np.sqrt(err/A.shape[0]/3)

    return R, L, RMSE
