#!/usr/bin/env python

"""Calculates the transformation between two coordinate systems using SVD."""

__author__ = 'Marcos Duarte <duartexyz@gmail.com>'
__version__ = 'svdt.py v.1 2013/12/23'

import numpy as np

def svdt(A, B):
    """Calculates the transformation between two coordinate systems using SVD.

    This function determines the rotation matrix (R) and the translation vector
    (L) for a rigid body after the following transformation [1]_, [2]_:
    B = R*A + L + err.
    Where A and B represents the rigid body in different instants and err is an
    aleatory noise. A and B are matrices with the marker coordinates at
    different instants (at least three non-collinear markers are necessary to
    determine the 3D transformation).
    
    The matrix A can be thought to represent a local coordinate system (but A
    it's not a basis) and matrix B the global coordinate system. The operation
    Pg = R*Pl + L calculates the coordinates of the point Pl (expressed in the
    local coordinate system) in the global coordinate system (Pg).

    Parameters
    ----------
    A   : 2D Numpy array (Nmarkers x 3)
        Coordinates [x,y,z] of at least three markers
    B   : 2D Numpy array (Nmarkers x 3)
        Coordinates [x,y,z] of at least three markers

    Returns
    -------
    R    : 2D Numpy array (3 x 3)
         Rotation matrix between A and B
    L    : 1D Numpy array (3)
         Translation vector between A and B
    RMSE : float
         Root-mean-squared error for the rigid body model: B = R*A + L + err
         For an ideal rigid body, RMSE is zero.

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
    >>> A = np.array([[0,0,0], [1,0,0], [ 0,1,0], [ 1,1,0]])  # four markers
    >>> B = np.array([[0,0,0], [0,1,0], [-1,0,0], [-1,1,0]])  # four markers
    >>> from svdt import svdt
    >>> R, L, RMSE = svdt(A, B)
    """
       
    Am = np.mean(A, axis=0)           # centroid of m1
    Bm = np.mean(B, axis=0)           # centroid of m2
    M = np.dot((B - Bm).T, (A - Am))  # considering only rotation
    # singular value decomposition
    U, S, Vt = np.linalg.svd(M)
    # rotation matrix
    R = np.dot(U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt))  
    # translation vector
    L = Bm - np.dot(R, Am)
    # RMSE
    err = 0
    for i in range(A.shape[0]):
        Bp = np.dot(R, A[i, :]) + L
        err += np.sum((Bp - B[i, :])**2)
    RMSE = np.sqrt(err/A.shape[0]/3)
    
    return R, L, RMSE
    
    
def svdts(A, B):
    """Function to call svdt with markers data of more than one instant.
    
    See the help of the svdt function.
    
    A typical use of the svdt function is to calculate the transformation
    between A and B (B = R*A + L), where A is the matrix with the markers data
    in one instant (the calibration or static trial) and B is the matrix with
    the markers data for more than one instant (the dynamic trial).
    
    Input A as a 1D array [x1,y1,z1,...,xn,yn,zn] where n is the number of
    markers and B as a 2D array with the different instants as rows (each row
    like in A). The output R has the shape (tn, 3, 3), where tn is the number
    of instants, L has the shape (tn, 3), and RMSE has the shape (tn). If tn
    is equal to one, the outputs have the same shape as in svdt (the last
    dimension of the outputs above is dropped).

    See Also
    --------
    svdt.svdt

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([0,0,0, 1,0,0,  0,1,0,  1,1,0])  # four markers
    >>> B = np.array([0,0,0, 0,1,0, -1,0,0, -1,1,0])  # four markers
    >>> B = np.vstack((B, B))  # simulate two instants (two rows)
    >>> from svdt import svdts
    >>> R, L, RMSE = svdts(A, B)
    """

    A = np.reshape(A, (len(A)/3, 3))
    B = np.asarray(B)
    if B.ndim == 1:
        B = np.reshape(B, (len(B)/3, 3))
        R, L, RMSE = svdt(A, B)
    else:
        n = B.shape[0]
        R = np.empty((n, 3, 3))
        L = np.empty((n, 3))
        RMSE = np.empty(n)
        for i in range(n):
            R[i,:,:], L[i,:], RMSE[i] = svdt(A, np.reshape(B[i, :], A.shape))
    
    return R, L, RMSE