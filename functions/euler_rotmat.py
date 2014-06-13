#!/usr/bin/env python

"""Euler rotation matrix given sequence, frame, and angles."""

from __future__ import division, print_function

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'euler_rotmat.py v.1 2014/03/10'


def euler_rotmat(order='xyz', frame='local', angles=None, unit='deg',
                 str_symbols=None, showA=True, showN=True):
    """Euler rotation matrix given sequence, frame, and angles.
    
    This function calculates the algebraic rotation matrix (3x3) for a given
    sequence ('order' argument) of up to three elemental rotations of a given
    coordinate system ('frame' argument) around another coordinate system, the
    Euler (or Eulerian) angles [1]_.

    This function also calculates the numerical values of the rotation matrix
    when numerical values for the angles are inputed for each rotation axis.
    Use None as value if the rotation angle for the particular axis is unknown.

    The symbols for the angles are: alpha, beta, and gamma for the first,
    second, and third rotations, respectively.
    The matrix product is calulated from right to left and in the specified
    sequence for the Euler angles. The first letter will be the first rotation.
    
    The function will print and return the algebraic rotation matrix and the
    numerical rotation matrix if angles were inputed.

    Parameters
    ----------
    order  : string, optional (default = 'xyz')
             Sequence for the Euler angles, any combination of the letters
             x, y, and z with 1 to 3 letters is accepted to denote the
             elemental rotations. The first letter will be the first rotation.

    frame  : string, optional (default = 'local')
             Coordinate system for which the rotations are calculated.
             Valid values are 'local' or 'global'.

    angles : list, array, or bool, optional (default = None)
             Numeric values of the rotation angles ordered as the 'order'
             parameter. Enter None for a rotation whith unknown value.

    unit   : str, optional (default = 'deg')
             Unit of the input angles.
    
    str_symbols : list of strings, optional (default = None)
             New symbols for the angles, for instance, ['theta', 'phi', 'psi']
             
    showA  : bool, optional (default = True)
             True (1) displays the Algebraic rotation matrix in rich format.
             False (0) to not display.

    showN  : bool, optional (default = True)
             True (1) displays the Numeric rotation matrix in rich format.
             False (0) to not display.
             
    Returns
    -------
    R     :  Matrix Sympy object
             Rotation matrix (3x3) in algebraic format.

    Rn    :  Numpy array or Matrix Sympy object (only if angles are inputed)
             Numeric rotation matrix (if values for all angles were inputed) or
             a algebraic matrix with some of the algebraic angles substituted
             by the corresponding inputed numeric values.

    Notes
    -----
    This code uses Sympy, the Python library for symbolic mathematics, to
    calculate the algebraic rotation matrix and shows this matrix in latex form
    possibly for using with the IPython Notebook, see [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/Transformation3D.ipynb

    Examples
    --------
    >>> # import function
    >>> from euler_rotmat import euler_rotmat
    >>> # Default options: xyz sequence, local frame and show matrix
    >>> R = euler_rotmat()
    >>> # XYZ sequence (around global (fixed) coordinate system)
    >>> R = euler_rotmat(frame='global')
    >>> # Enter numeric values for all angles and show both matrices
    >>> R, Rn = euler_rotmat(angles=[90, 90, 90])
    >>> # show what is returned
    >>> euler_rotmat(angles=[90, 90, 90])
    >>> # show only the rotation matrix for the elemental rotation at x axis
    >>> R = euler_rotmat(order='x')
    >>> # zxz sequence and numeric value for only one angle
    >>> R, Rn = euler_rotmat(order='zxz', angles=[None, 0, None])
    >>> # input values in radians:
    >>> import numpy as np
    >>> R, Rn = euler_rotmat(order='zxz', angles=[None, np.pi, None], unit='rad')
    >>> # shows only the numeric matrix
    >>> R, Rn = euler_rotmat(order='zxz', angles=[90, 0, None], showA='False')
    >>> # Change the angles' symbols
    >>> R = euler_rotmat(order='zxz', str_symbols=['theta', 'phi', 'psi'])
    >>> # Negativate the angles' symbols
    >>> R = euler_rotmat(order='zxz', str_symbols=['-theta', '-phi', '-psi'])
    >>> # all algebraic matrices for all possible sequences for the local frame
    >>> s=['xyz','xzy','yzx','yxz','zxy','zyx','xyx','xzx','yzy','yxy','zxz','zyz']
    >>> for seq in s: R = euler_rotmat(order=seq)
    >>> # all algebraic matrices for all possible sequences for the global frame
    >>> for seq in s: R = euler_rotmat(order=seq, frame='global')
    """

    import numpy as np
    import sympy as sym
    try:
        from IPython.core.display import Math, display
        ipython = True
    except:
        ipython = False

    angles = np.asarray(np.atleast_1d(angles), dtype=np.float64)
    if ~np.isnan(angles).all():        
        if len(order) != angles.size:
            raise ValueError("Parameters 'order' and 'angles' (when " + 
                             "different from None) must have the same size.")

    x, y, z = sym.symbols('x, y, z')
    sig = [1, 1, 1]
    if str_symbols is None:
        a, b, g = sym.symbols('alpha, beta, gamma')
    else:
        s = str_symbols
        if s[0][0] == '-': s[0] = s[0][1:]; sig[0] = -1
        if s[1][0] == '-': s[1] = s[1][1:]; sig[1] = -1
        if s[2][0] == '-': s[2] = s[2][1:]; sig[2] = -1        
        a, b, g = sym.symbols(s)

    var = {'x': x, 'y': y, 'z': z, 0: a, 1: b, 2: g}
    # Elemental rotation matrices for xyz (local)
    cos, sin = sym.cos, sym.sin
    Rx = sym.Matrix([[1, 0, 0], [0, cos(x), sin(x)], [0, -sin(x), cos(x)]])
    Ry = sym.Matrix([[cos(y), 0, -sin(y)], [0, 1, 0], [sin(y), 0, cos(y)]])
    Rz = sym.Matrix([[cos(z), sin(z), 0], [-sin(z), cos(z), 0], [0, 0, 1]])

    if frame.lower() == 'global':
        Rs = {'x': Rx.T, 'y': Ry.T, 'z': Rz.T}
        order = order.upper()
    else:
        Rs = {'x': Rx, 'y': Ry, 'z': Rz}
        order = order.lower()

    R = Rn = sym.Matrix(sym.Identity(3))
    str1 = r'\mathbf{R}_{%s}( ' %frame  # last space needed for order=''
    #str2 = [r'\%s'%var[0], r'\%s'%var[1], r'\%s'%var[2]]
    str2 = [1, 1, 1]        
    for i in range(len(order)):
        Ri = Rs[order[i].lower()].subs(var[order[i].lower()], sig[i] * var[i]) 
        R = Ri * R
        if sig[i] > 0:
            str2[i] = '%s:%s' %(order[i], sym.latex(var[i]))
        else:
            str2[i] = '%s:-%s' %(order[i], sym.latex(var[i]))
        str1 = str1 + str2[i] + ','
        if ~np.isnan(angles).all() and ~np.isnan(angles[i]):
            if unit[:3].lower() == 'deg':
                angles[i] = np.deg2rad(angles[i])
            Rn = Ri.subs(var[i], angles[i]) * Rn
            #Rn = sym.lambdify(var[i], Ri, 'numpy')(angles[i]) * Rn
            str2[i] = str2[i] + '=%.0f^o' %np.around(np.rad2deg(angles[i]), 0)
        else:
            Rn = Ri * Rn

    Rn = sym.simplify(Rn)  # for trigonometric relations

    try:
        # nsimplify only works if there are symbols
        Rn2 = sym.latex(sym.nsimplify(Rn, tolerance=1e-8).n(chop=True, prec=4))
    except:
        Rn2 = sym.latex(Rn.n(chop=True, prec=4))
        # there are no symbols, pass it as Numpy array
        Rn = np.asarray(Rn)
    
    if showA and ipython:
        display(Math(str1[:-1] + ') =' + sym.latex(R, mat_str='matrix')))

    if showN and ~np.isnan(angles).all() and ipython:
            str2 = ',\;'.join(str2[:angles.size])
            display(Math(r'\mathbf{R}_{%s}(%s)=%s' %(frame, str2, Rn2)))

    if np.isnan(angles).all():
        return R
    else:
        return R, Rn
