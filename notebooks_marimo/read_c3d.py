import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Read c3D file using EZC3D library

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://demotu.org/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > EZC3D is an easy to use reader, modifier and writer for C3D format files.  
        > https://github.com/pyomeca/ezc3d
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    #%matplotlib notebook
    # '%matplotlib widget' command supported automatically in marimo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import sys, os
    import ezc3d

    sys.path.insert(1, r'./../functions')
    from dfmlevel import dfmlevel
    from read_c3d import read_c3d
    return os, read_c3d


@app.cell
def _(os):
    path2 = '/mnt/A/BMClab/Projects/FapespRunAge/Data/Cadence/s20/'
    fname = 'run100c.c3d'
    fname = os.path.join(path2, fname)
    return (fname,)


@app.cell
def _(fname, read_c3d):
    an, pt = read_c3d(fname, analog='all', point='all', short_label=True)
    return an, pt


@app.cell
def _(an):
    an
    return


@app.cell
def _(pt):
    pt
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
