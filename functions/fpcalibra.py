"""Force plate calibration algorithm.
"""

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'fpcalibra.py v.1 2016/07/09'

import numpy as np
from numpy.linalg import pinv
from scipy.optimize import minimize
import time


def fpcalibra(Lfp, Flc, COP, threshold=1e-10):
    """Force plate calibration algorithm.
    
    For a force plate (FP) re-calibration, the relationship between the
    measured FP output (L) and the known loads (Li) is approximated by:
    Li = C@L + E (@ is the operator for matrix multiplication).  
    Where C is the 6-by-6 re-calibration matrix and E is a gaussian,
    uncorrelated, zero mean noise six-by-one matrix.  

    The re-calibration matrix can be found by solving the equation above and
    then C can be later used to re-calibrate the FP output: Lc = C@L.  
    Where Lc is the re-calibrated FP output.

    Cedraro et al. (2008) [1]_ proposed to use a calibrated three-component
    load cell to measure the forces applied on the FP at known measurement
    sites and an algorithm for the re-calibration.
    
    This code implements the re-calibration algorithm, see [2]_
    
    Parameters
    ----------
    Lfp : numpy 2-D array (6, nsamples*nksites)
        loads [Fx, Fy, Fz, Mx, My, Mz] (in N and Nm) measured by the force
        plate due to the corresponding forces applied at the measurement sites
    Flc : numpy 2-D array (3, nsamples*nksites)
        forces [Fx, Fy, Fz] (in N) measured by the load cell at the
        measurement sites
    COP : numpy 2-D array (3, nksites)
        positions [COPx, COPy, COPz] (in m) of the load cell at the
        measurement sites
    threshold  : float, optional
        threshold to stop the optimization (default 1e-10)
    
    Returns
    -------
    C   : numpy 2-D (6-by-6) array
        optimal force plate re-calibration matrix (in dimensionless units)
    ang : numpy 1-D array [ang0, ..., angk]
        optimal angles of rotation (in rad) of the load cells at the
        measurement sites

    References
    ----------
    .. [1] Cedraro A, Cappello A, Chiari L (2008) Gait & Posture, 28, 488–494. 
    .. [2] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/ForcePlateCalibration.ipynb

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
    >>> # simulated 5 measurement sites (in m)
    >>> COP = np.array([[   0,  112,  112, -112, -112],
    >>>                 [   0,  192, -192,  192, -192],
    >>>                 [-124, -124, -124, -124, -124]])/1000
    >>> nk = COP.shape[1]
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
    >>> for k, cop in enumerate(COP.T):
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
    >>> print('Residual between simulated and optimal re-calibration matrices:', e)
    >>> e = np.sqrt(np.sum(ang2-ang)**2)
    >>> print('Residual between simulated and optimal rotation angles:', e)
    
    """

    # number of sites
    nk = COP.shape[1]
    # number of samples
    ns = int(Lfp.shape[1]/nk)
    # function for the COP skew-symmetric matrix
    Acop = lambda x,y,z : np.array([[.0, -z, y], [z, .0, -x], [-y, x, .0]])
    P = np.empty((6, 3, nk))
    for k, cop in enumerate(COP.T):
        P[:, :, k] = np.vstack((np.eye(3), Acop(*cop)))
    # function for the 2D rotation matrix
    R = lambda a : np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [ 0, 0, 1]])
    # Pseudoiverse of the loads measured by the force plate
    Lpinv = pinv(Lfp)
    # cost function for the optimization
    def costfun(ang, P, R, Flc, CLfp, nk, ns, E):
        for k in range(nk):
            E[:,k*ns:(k+1)*ns] = (P[:,:,k] @ R(ang[k])) @ Flc[:,k*ns:(k+1)*ns] - CLfp[:,k*ns:(k+1)*ns]
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
    # the optimization
    while np.all(delta_ang > threshold):
        for k in range(nk):
            Li[:,k*ns:(k+1)*ns] = (P[:,:,k] @ R(ang0[k])) @ Flc[:,k*ns:(k+1)*ns]
        C = Li @ Lpinv
        CLfp = C @ Lfp
        res = minimize(fun=costfun, x0=ang0, args=(P, R, Flc, CLfp, nk, ns, E),
                       bounds=bnds, method='TNC', options={'disp': False})
        delta_ang = np.abs(res.x - ang0)
        ang0 = res.x
        da.append(delta_ang.sum())

    tdelta = time.time() - start
    print('\nOptimization finished in %.1f s after %d steps.\n' %(tdelta, len(da)))
    print('Optimal calibration matrix:\n', C)
    print('\nOptimal angles:\n', res.x*180/np.pi)
    print('\n')

    return C, res.x