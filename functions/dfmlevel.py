"""Create multi-level pandas dataframe for kinematic data in OpenSim style.

"""

__author__ = "Marcos Duarte, https://github.com/BMClab/"
__version__ = "1.0.0"
__license__ = "MIT"

import numpy as np
import pandas as pd



def dfmlevel(x, labels=None, index=None, n_ini=0, names=['Marker', 'Coordinate'], 
             order='XYZ'):
    """
    Create multi-level pandas dataframe for kinematic data in OpenSim style.
    
    Parameters
    ----------
    x : numpy array

    labels : list, optional (default = None)

    index : index for dataframe, optional (default = None)

    n_ini : integer, optional (default = 0)

    names : list, optional (default = ['Marker', 'Coordinate'])

    order : string, optional (default = 'XYZ')


    Returns
    -------
    df : pandas dataframe
        dataframe with multi-levels given by `names`.

    """

    if labels is None:
        labels = ['m' + str(i) for i in range(n_ini, n_ini + int(x.shape[1]/len(order)))]
    names.append(order)
    n = np.repeat(range(n_ini + 1, len(labels) + n_ini + 1), len(order)).tolist()
    labelsxyz = [m for m in labels for i in range(len(order))]
    coordinate = [a for a in list(order)*len(labels)]
    xyz = [a + str(b) for a, b in zip(coordinate, n)]
    df = pd.DataFrame(data=x, index=index, columns=[labelsxyz, coordinate, xyz])
    if index is not None:
        df.index.name = 'Time'
    df.columns.set_names(names=names, level=[0, 1, 2], inplace=True)
    
    return df
