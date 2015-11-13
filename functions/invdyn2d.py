"""Two-dimensional inverse-dynamics calculations of one segment."""

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'invdyn2d.py v.2 2015/11/13'


def invdyn2d(rcm, rd, rp, acm, alpha, mass, Icm, Fd, Md):
    """Two-dimensional inverse-dynamics calculations of one segment

    Parameters
    ----------
    rcm   : array_like [x,y]
            center of mass position (y is vertical)
    rd    : array_like [x,y]
            distal joint position
    rp    : array_like [x,y]
            proximal joint position
    acm   : array_like [x,y]
            center of mass acceleration
    alpha : array_like [x,y]
            segment angular acceleration
    mass  : number
            mass of the segment   
    Icm   : number
            rotational inertia around the center of mass of the segment
    Fd    : array_like [x,y]
            force on the distal joint of the segment
    Md    : array_like [x,y]
            moment of force on the distal joint of the segment
    
    Returns
    -------
    Fp    : array_like [x,y]
            force on the proximal joint of the segment (y is vertical)
    Mp    : array_like [x,y]
            moment of force on the proximal joint of the segment

    Notes
    -----
    To use this function recursevely, the outputs [Fp, Mp] must be inputed as 
    [-Fp, -Mp] on the next call to represent [Fd, Md] on the distal joint of the
    next segment (action-reaction).
    
    This code was inspired by a similar code written by Ton van den Bogert [1]_.
    See this notebook [2]_.

    References
    ----------
    .. [1] http://isbweb.org/data/invdyn/index.html
    .. [2] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/GaitAnalysis2D.ipynb

    """
    
    from numpy import cross
     
    g = 9.80665  # m/s2, standard acceleration of free fall (ISO 80000-3:2006)
    # Force and moment of force on the proximal joint
    Fp = mass*acm - Fd - [0, -g*mass]
    Mp = Icm*alpha - Md - cross(rd-rcm, Fd) - cross(rp-rcm, Fp)
    
    return Fp, Mp