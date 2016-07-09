"""Force plate calibration algorithm based on Cedraro et al. (2008).

"""

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'fpcalibra.py v.1 2016/07/10'

import numpy as np
from numpy.linalg import pinv
from scipy.optimize import minimize
import time


def fpcalibra(Lfp, Flc, COP, threshold = 1e-10):
    """Force plate calibration algorithm based on Cedraro et al. (2008).
    
    For a force plate, FP, re-calibration, the relationship between the
    measured FP output, $\mathbf{L}$, and the known loads, $\mathbf{L}_I$,
    is approximated by: $\mathbf{L}_I = \mathbf{C}\mathbf{L} + \mathbf{E}$.  
    Where $\mathbf{C}$ is the 6-by-6 re-calibration matrix and $\mathbf{E}$
    is a gaussian, uncorrelated, zero mean noise six-by-one matrix.  

    The re-calibration matrix can be found by solving the equation above and
    then $\mathbf{C}$ can be later used to re-calibrate the FP output:
    $\mathbf{L}_C = \mathbf{C}\mathbf{L}$.  
    Where $\mathbf{L}_C$ is the re-calibrated FP output.

    Cedraro et al. (2008) propose to use a calibrated three-component load
    cell to measure the loads applied on the FP at known measurements sites
    and an algorithm for the re-calibration.
    
    This code implements this re-calibration algorithm, see [1]_
    
    Parameters
    ----------
    Lfp  : numpy 2-d array
        loads measured by the force plate at the measurements sites
    Flc  : numpy 2-d array
        loads measured by the load cell at the measurements sites
    COP  : numpy 2-d array
        positions of the load cell at the measurements sites
    threshold  : float, optional
        threshold to stop the optimization
    
    Returns
    -------
    C    : numpy 6-by-6 array
            force plate re-calibration matrix
    ang    : numpy 1-d array [ang0, ... angk]
            angles of rotation of the load cells at the measurment sites

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/ForcePlateCalibration.ipynb

    Example
    -------
    >>> from fpcalibra import fpcalibra
    >>> import numpy as np
    >>> from numpy.linalg import inv
    >>>
    >>> # simulated true re-calibration matrix
    >>> C = np.array([[ 1.0354, -0.0053, -0.0021, -0.0289, -0.0402,  0.0081],
    >>>               [ 0.0064,  1.0309, -0.0031,  0.0211,  0.0135, -0.0001],
    >>>               [ 0.0000, -0.0004,  1.0022, -0.0005, -0.0182,  0.0300],
    >>>               [-0.0012, -0.0385,  0.0002,  0.9328,  0.0007,  0.0017],
    >>>               [ 0.0347,  0.0003,  0.0008, -0.0002,  0.9325, -0.0024],
    >>>               [-0.0004, -0.0013, -0.0003, -0.0023,  0.0035,  1.0592]])
    >>> # simulated 5 measurements sites (in m)
    >>> COP = np.array([[   0,  112,  112, -112, -112],
    >>>                 [   0,  192, -192,  192, -192],
    >>>                 [-124, -124, -124, -124, -124]]).T/1000
    >>> nk = COP.shape[0]
    >>> # simulated forces measured by the load cell (in N) before rotation
    >>> samples = np.linspace(1, 6000, 6000)
    >>> ns = samples.shape[0]
    >>> Flc = np.array([100*np.sin(5*2*np.pi*samples/samples[-1]),
    >>>                 100*np.cos(5*2*np.pi*samples/samples[-1]),
    >>>                 samples/15 + 200])
    >>> Flc = np.tile(Flc, nk)
    >>> # function for the COP skew-symmetric matrix
    >>> Acop = lambda x,y,z : np.array([[.0, -z, y], [z, .0, -x], [-y, x, .0]])
    >>> # simulated loads measured by the force plate
    >>> Li = np.empty((6, ns*nk))
    >>> P = np.empty((6, 3, nk))
    >>> for k, cop in enumerate(COP):
    >>>     P[:, :, k] = np.vstack((np.eye(3), Acop(*cop)))
    >>>     Li[:, k*ns:(k+1)*ns] = P[:, :, k] @ Flc[:, k*ns:(k+1)*ns]
    >>> Lfp = inv(C) @  Li
    >>> # simulated angles of rotaton of the measurement sites
    >>> ang = np.array([20, -10, 0, 15, -5])/180*np.pi
    >>> # function for the rotation matrix
    >>> R = lambda a : np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [ 0, 0, 1]])
    >>> # simulated forces measured by the load cell after rotation
    >>> for k in range(nk):
    >>>     Flc[:, k*ns:(k+1)*ns] = R(ang[k]).T @ Flc[:, k*ns:(k+1)*ns]
    >>> 
    >>> C2, ang2 = fpcalibra(Lfp, Flc, COP)
    >>> 
    >>> e = np.sqrt(np.sum(C2-C)**2)
    >>> print('\nResidual between simulated and optimal re-calibration matrices:', e)
    >>> e = np.sqrt(np.sum(ang2-ang)**2)
    >>> print('\nResidual between simulated and optimal rotation angles:', e)
    
    """

    # number of sites
    nk = COP.shape[0]
    # number of samples
    ns = int(Lfp.shape[1]/nk)
    # function for the COP skew-symmetric matrix
    Acop = lambda x,y,z : np.array([[.0, -z, y], [z, .0, -x], [-y, x, .0]])
    P = np.empty((6, 3, nk))
    for k, cop in enumerate(COP):
        P[:, :, k] = np.vstack((np.eye(3), Acop(*cop)))    
    # function for the 2D rotation matrix
    R = lambda a : np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [ 0, 0, 1]])
    # Pseudoiverse of the loads measured by the force plate
    Lpinv = pinv(Lfp)

    # cost function for the optimization
    def costfun(ang, P, R, Flc, C, Lfp, nk, ns, E):
        for k in range(nk):
            E[:,k*ns:(k+1)*ns] = P[:,:,k] @ R(ang[k]) @ Flc[:,k*ns:(k+1)*ns] - C @ Lfp[:,k*ns:(k+1)*ns]
        return np.sum(E * E)
    # inequality constraints
    bnds = [(-np.pi/2, np.pi/2) for k in range(nk)]
    # some initialization
    ang0 = np.zeros(nk)
    E = np.empty((6, ns*nk))
    da = []
    delta_ang = 10*threshold
    Li = np.empty((6, ns*nk))
    start = time.time()

    while np.all(delta_ang > threshold):
        for k, cop in enumerate(COP):
            Li[:, k*ns:(k+1)*ns] = P[:, :, k] @ R(ang0[k]) @ Flc[:, k*ns:(k+1)*ns]
        C = Li @ Lpinv
        res = minimize(fun=costfun, x0=ang0, args=(P, R, Flc, C, Lfp, nk, ns, E),
                       bounds=bnds, method='TNC', options={'disp': False})
        delta_ang = np.abs(res.x - ang0)
        ang0 = res.x
        da.append(delta_ang.sum())

    tdelta = time.time() - start
    print('\nOptimization finished after %d steps in %.1f s.\n' %(len(da), tdelta))
    print('Optimal calibration matrix:\n', C)
    print('\nOptimal angles:\n', res.x*180/np.pi)

    return C, res.x