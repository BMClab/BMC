#!/usr/bin/env python

"""Calculates an ellipse(oid) as prediction interval for multivariate data."""

from __future__ import division, print_function
import numpy as np

__author__ = 'Marcos Duarte'
__version__ = 'ellipseoid.py v.1 2013/12/30'


def ellipseoid(P, y=None, z=None, pvalue=.95, units=None, show=True):
    """Calculates an ellipse(oid) as prediction interval for multivariate data.

    The prediction ellipse (or ellipsoid) is a prediction interval for a sample
    of a bivariate (or trivariate) random variable and is such that there is
    pvalue*100% of probability that a new observation will be contained in the
    ellipse (or ellipsoid) (Chew, 1966). [1]_.

    The semi-axes of the prediction ellipse(oid) are found by calculating the
    eigenvalues of the covariance matrix of the data and adjust the size of the
    semi-axes to account for the necessary prediction probability. 

    Parameters
    ----------
    P : 1-D or 2-D array_like
        For a 1-D array, P is the abscissa values of the [x,y] or [x,y,z] data.
        For a 2-D array, P is the joined values of the [x,y] or [x,y,z] data.
        The shape of the 2-D array should be (n, 2) or (n, 3) where n is the
        number of observations.
    y : 1-D array_like, optional (default = None)
        Ordinate values of the [x, y] or [x,y,z] data.
    z : 1-D array_like, optional (default = None)
        Ordinate values of the [x, y] or [x,y,z] data.
    pvalue : float, optional (default = .95)
        Desired prediction probability of the ellipse(oid).
    units : str, optional (default = None)
        Units of the input data.
    show : bool, optional (default = True)
        True (1) plots data in a matplotlib figure, False (0) to not plot.

    Returns
    -------
    volume : float
        Area of the ellipse or volume of the ellipsoid according to the inputs.
    axes : 2-D array
        Lengths of the semi-axes ellipse(oid) (largest first).
    angles : 1-D array
        Angles of the semi-axes ellipse(oid). For the ellipsoid (3D adata), the
        angles are the Euler angles calculated in the XYZ sequence.
    center : 1-D array
        Centroid of the ellipse(oid).
    rotation : 2-D array
        Rotation matrix of the semi-axes of the ellipse(oid).

    Notes
    -----
    The directions and lengths of the semi-axes are found, respectively, as the
    eigenvectors and eigenvalues of the covariance matrix of the data using
    the concept of principal components analysis (PCA) [2]_ or singular value
    decomposition (SVD) [3]_.
    
    See [4]_ for a discussion about prediction and confidence intervals and
    their use in posturography.

    References
    ----------
    .. [1] http://www.jstor.org/stable/2282774.
    .. [2] http://en.wikipedia.org/wiki/Principal_component_analysis.
    .. [3] http://en.wikipedia.org/wiki/Singular_value_decomposition.
    .. [4] http://www.sciencedirect.com/science/article/pii/S0966636213005961.

    Examples
    --------
    >>> import numpy as np
    >>> from ellipseoid import ellipseoid
    >>> y = np.cumsum(np.random.randn(3000)) / 50
    >>> x = np.cumsum(np.random.randn(3000)) / 100
    >>> area, axes, angles, center, R = ellipseoid(x, y, units='cm', show=True)
    >>> P = np.random.randn(1000, 3)
    >>> P[:, 2] = P[:, 2] + P[:, 1]*.5
    >>> P[:, 1] = P[:, 1] + P[:, 0]*.5
    >>> volume, axes, angles, center, R = ellipseoid(P, units='cm', show=True)
    """

    from scipy import stats

    P = np.array(P, ndmin=2, dtype=float)
    if P.shape[0] == 1:
        P = P.T
    elif P.shape[1] > 3:
        P = P.T
    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=float)
        if y.shape[0] == 1:
            y = y.T
        P = np.concatenate((P, y), axis=1)
    if z is not None:
        z = np.array(z, copy=False, ndmin=2, dtype=float)
        if z.shape[0] == 1:
            z = z.T
        P = np.concatenate((P, z), axis=1)
    # covariance matrix
    cov = np.cov(P, rowvar=0)
    # singular value decomposition
    U, s, Vt = np.linalg.svd(cov)
    # semi-axes (largest first)
    p, n = s.size, P.shape[0]
    saxes = np.sqrt(s*stats.f.ppf(pvalue, p, dfd=n-p)*(n-1)*p*(n+1)/(n*(n-p)))
    volume = 4/3*np.pi*np.prod(saxes) if p == 3 else np.pi*np.prod(saxes)
    # rotation matrix
    R = Vt
    if s.size == 2:
        angles = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
        angles = np.array([angles, angles + 90])
    else:
        angles = rotXYZ(R, unit='deg')
    # centroid of the ellipse(oid)
    center = np.mean(P, axis=0)

    if show:
        _plot(P, volume, saxes, center, R, pvalue, units, fig=None, ax=None)

    return volume, saxes, angles, center, R


def _plot(P, volume, saxes, center, R, pvalue, units, fig, ax):
    """Plot results of the ellipseoid function, see its help."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        # code based on https://github.com/minillinim/ellipsoid:
        # parametric equations
        u = np.linspace(0, 2*np.pi, 100)
        if saxes.size == 2:
            x = saxes[0]*np.cos(u)
            y = saxes[1]*np.sin(u)
            # rotate data
            for i in range(len(x)):
                [x[i], y[i]] = np.dot([x[i], y[i]], R) + center
        else:
            v = np.linspace(0, np.pi, 100)
            x = saxes[0]*np.outer(np.cos(u), np.sin(v))
            y = saxes[1]*np.outer(np.sin(u), np.sin(v))
            z = saxes[2]*np.outer(np.ones_like(u), np.cos(v))
            # rotate data
            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i,j],y[i,j],z[i,j]]=np.dot([x[i,j],y[i,j],z[i,j]],R)+center

        if saxes.size == 2:
            if fig is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            # plot raw data
            ax.plot(P[:, 0], P[:, 1], '.-', color=[0, 0, 1, .5])
            # plot ellipse
            ax.plot(x, y, color=[0, 1, 0, 1], linewidth=2)
            # plot axes
            for i in range(saxes.size):
                # rotate axes
                a = np.dot(np.diag(saxes)[i], R).reshape(2, 1)
                # points for the axes extremities
                a = np.dot(a, np.array([-1, 1], ndmin=2)) + center.reshape(2, 1)
                ax.plot(a[0], a[1], color=[1, 0, 0, .6], linewidth=2)
                ax.text(a[0, 1], a[1, 1], '%d' % (i + 1),
                        fontsize=20, color='r')
            plt.axis('equal')
            plt.grid()
            title = r'Prediction ellipse (p=%4.2f): Area=' % pvalue
            if units is not None:
                units2 = ' [%s]' % units
                units = units + r'$^2$'
                title = title + r'%.2f %s' % (volume, units)
            else:
                units2 = ''
                title = title + r'%.2f' % volume
        else:
            from mpl_toolkits.mplot3d import Axes3D
            if fig is None:
                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_axes([0, 0, 1, 1], projection='3d')
            ax.view_init(20, 30)
            # plot raw data
            ax.plot(P[:, 0], P[:, 1], P[:, 2], '.-', color=[0, 0, 1, .4])
            # plot ellipsoid
            ax.plot_surface(x, y, z, rstride=5, cstride=5, color=[0, 1, 0, .1],
                            linewidth=1, edgecolor=[.1, .9, .1, .4])
            #ax.plot_wireframe(x, y, z, color=[0, 1, 0, .5], linewidth=1)
            #                  rstride=3, cstride=3, edgecolor=[0, 1, 0, .5])
            # plot axes
            for i in range(saxes.size):
                # rotate axes
                a = np.dot(np.diag(saxes)[i], R).reshape(3, 1)
                # points for the axes extremities
                a = np.dot(a, np.array([-1, 1], ndmin=2)) + center.reshape(3, 1)
                ax.plot(a[0], a[1], a[2], color=[1, 0, 0, .6], linewidth=2)
                ax.text(a[0, 1], a[1, 1], a[2, 1], '%d' % (i+1),
                        fontsize=20, color='r')
            lims = [np.min([P.min(), x.min(), y.min(), z.min()]),
                    np.max([P.max(), x.max(), y.max(), z.max()])]
            ax.set_xlim(lims); ax.set_ylim(lims); ax.set_zlim(lims)
            title = r'Prediction ellipse (p=%4.2f): Volume=' % pvalue
            if units is not None:
                units2 = ' [%s]' % units
                units = units + r'$^3$'
                title = title + r'%.2f %s' % (volume, units)
            else:
                units2 = ''
                title = title + r'%.2f' % volume
            ax.set_zlabel('Z' + units2, fontsize=18)

        ax.set_xlabel('X' + units2, fontsize=18)
        ax.set_ylabel('Y' + units2, fontsize=18)
        plt.title(title)
        plt.show()

        return fig, ax


def rotXYZ(R, unit='deg'):
    """ Compute Euler angles from matrix R using XYZ sequence."""

    angles = np.zeros(3)
    angles[0] = np.arctan2(R[2, 1], R[2, 2])
    angles[1] = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    angles[2] = np.arctan2(R[1, 0], R[0, 0])

    if unit[:3].lower() == 'deg':  # convert from rad to degree
        angles = np.rad2deg(angles)

    return angles
