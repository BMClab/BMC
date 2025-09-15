import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Resample data
        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://demotu.org/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    # '%matplotlib notebook' command supported automatically in marimo
    # tk qt notebook inline ipympl
    import matplotlib
    import matplotlib.pyplot as plt

    import sys, os
    sys.path.insert(1, r'./../functions')

    from resample import resample
    return np, pd, resample


@app.cell
def _(np, pd):
    y = np.random.randn(100, 3)
    freq_new = 100
    y = pd.DataFrame(data=y, columns=None)
    y.index = y.index/freq_new
    y.index.name = y.index.name
    y.head()
    return (y,)


@app.cell
def _(resample, y):
    y2 = resample(y, freq_old=100, freq_new=1000, limit=1000, method='resample_poly')
    y2
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
