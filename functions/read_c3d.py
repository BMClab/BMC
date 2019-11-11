"""Get data from c3d file using ezc3d library.

"""

__author__ = "Marcos Duarte, https://github.com/BMClab/"
__version__ = "1.0.0"
__license__ = "MIT"

import pandas as pd
import ezc3d
from dfmlevel import dfmlevel



def read_c3d(fname, analog='all', point='all', order='XYZ', resamp=False,
             short_label=True):
    """
    Get data from c3d file using ezc3d library.

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
    an : pandas dataframe
        dataframe with analog data if `analog` is not None.
    pt : pandas dataframe
        dataframe with points data if `point` is not None.
        dataframe is multi-level for kinematic data in OpenSim style. 

    """

    c3d = ezc3d.c3d(fname)
    info = [[analog, 'ANALOG', 'analogs'] if analog is not None else False,
            [ point,  'POINT',  'points'] if point  is not None else False]
    for signal in info:
        if signal:
            labels = c3d['parameters'][signal[1]]['LABELS']['value']
            if signal[0] == 'all':
                idx = list(range(len(labels)))
            elif isinstance(signal[0][0], str):
                idx = [labels.index(label) for label in signal[0]]
            elif isinstance(signal[0][0], int):
                idx = signal[0]
            else:
                raise ValueError('Values in {} must be "all" or list of int or list of str'.format(signal[2][:-1]))           

            labels = [labels[i] for i in idx]
            rate = c3d['parameters'][signal[1]]['RATE']['value'][0]
            if short_label:
                labels = [label.split(': ')[-1] for label in labels]
            if signal[1] == 'ANALOG':
                data = c3d['data'][signal[2]][0, idx, :].T
                if resamp:
                    pass
                an = pd.DataFrame(data=data, columns=labels)
                an.index = an.index/rate
                an.index.name = 'Time'
            elif signal[1] == 'POINT':
                data = c3d['data'][signal[2]][:3, idx, :].reshape((int(3*len(labels)), -1), order='F').T
                if resamp:
                    pass
                pt = dfmlevel(data, labels=labels, index=None, n_ini=0,
                              names=['Marker', 'Coordinate'], order=order)
                pt.index = pt.index/rate
                pt.index.name = 'Time'            
    
    if info[0] and info[1]:
        return an, pt
    elif info[0]:
        return an
    else:
        return pt
