import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Time normalization of data

        > Marcos Duarte  
        > Laboratory of Biomechanics and Motor Control ([http://demotu.org/](http://demotu.org/))  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Time normalization is usually employed for the temporal alignment of cyclic data obtained from different trials with different duration (number of points). The most simple and common procedure for time normalization used in Biomechanics and Motor Control is known as the normalization to percent cycle (although it might not be the most adequate procedure in certain cases ([Helwig et al., 2011](http://www.sciencedirect.com/science/article/pii/S0021929010005038)).

        In the percent cycle, a fixed number (typically a temporal base from 0 to 100%) of new equally spaced data is created based on the old data with a mathematical procedure known as interpolation.   
        **Interpolation** is the estimation of new data points within the range of known data points. This is different from **extrapolation**, the estimation of data points outside the range of known data points.   
        Time normalization of data using interpolation is a simple procedure and it doesn't matter if the original data have more or less data points than desired.

        The Python function `tnorma.py` from Python module `tnorma` implements the normalization to percent cycle procedure for time normalization. The function signature is:   
        ```python
        yn, tn, indie = tnorma(y, axis=0, step=1, k=3, smooth=0, mask=None,
                               nan_at_ext='delete', show=False, ax=None)
        ```   
        Let's see now how to perform interpolation and time normalization; first let's import the necessary Python libraries and configure the environment:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Installation

        ```bash
        pip install tnorma
        ```

        Or

        ```bash
        conda install -c duartexyz tnorma
        ```
        """
    )
    return


@app.cell
def _():
    # Import the necessary libraries
    import numpy as np
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For instance, consider the data shown next. The time normalization of these data to represent a cycle from 0 to 100%, with a step of 1% (101 data points) is:
        """
    )
    return


@app.cell
def _():
    y = [5,  4, 10,  8,  1, 10,  2,  7,  1,  3]
    print("y data:")
    y
    return (y,)


@app.cell
def _(np, y):
    t  = np.linspace(0, 100, len(y))  # time vector for the original data
    tn = np.linspace(0, 100, 101)     # new time vector for the new time-normalized data
    yn = np.interp(tn, t, y)          # new time-normalized data
    print("y data interpolated to 101 points:")
    yn
    return t, tn, yn


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The key is the Numpy `interp` function, from its help:   

        >interp(x, xp, fp, left=None, right=None)       
        >One-dimensional linear interpolation.   
        >Returns the one-dimensional piecewise linear interpolant to a function with given values at discrete data-points.

        A plot of the data will show what we have done:
        """
    )
    return


@app.cell
def _(plt, t, tn, y, yn):
    plt.figure(figsize=(9, 5))
    plt.plot(t, y, 'bo-', lw=2, label='original data')
    plt.plot(tn, yn, '.-', color=[1, 0, 0, .5], lw=2, label='time normalized')
    plt.legend(loc='best', framealpha=.5)
    plt.xlabel('Cycle [%]')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The function `tnorma.py` implements this kind of normalization with option for a different interpolation than the linear one used, deal with missing points in the data (if these missing points are not at the extremities of the data because the interpolation function can not extrapolate data), other things.   
        Let's see the `tnorma.py` examples:
        """
    )
    return


@app.cell
def _():
    from tnorma import tnorma
    return


app._unparsable_cell(
    r"""
    >>> # Default options: cubic spline interpolation passing through
        >>> # each datum, 101 points, and no plot
        >>> y = [5,  4, 10,  8,  1, 10,  2,  7,  1,  3]
        >>> tnorma(y)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    >>> # Linear interpolation passing through each datum
        >>> yn, tn, indie = tnorma(y, k=1, smooth=0, mask=None, show=True)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    >>> # Cubic spline interpolation with smoothing
        >>> yn, tn, indie = tnorma(y, k=3, smooth=1, mask=None, show=True)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    >>> # Cubic spline interpolation with smoothing and 50 points
        >>> x = np.linspace(-3, 3, 60)
        >>> y = np.exp(-x**2) + np.random.randn(60)/10
        >>> yn, tn, indie = tnorma(y, step=-50, k=3, smooth=1, show=True)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    >>> # Deal with missing data (use NaN as mask)
        >>> x = np.linspace(-3, 3, 100)
        >>> y = np.exp(-x**2) + np.random.randn(100)/10
        >>> y[:10] = np.NaN # first ten points are missing
        >>> y[30: 41] = np.NaN # make other 10 missing points
        >>> yn, tn, indie = tnorma(y, step=-50, k=3, smooth=1, show=True)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    >>> # Deal with missing data at the extremities replacing by first/last not-NaN
        >>> x = np.linspace(-3, 3, 100)
        >>> y = np.exp(-x**2) + np.random.randn(100)/10
        >>> y[0:10] = np.NaN # first ten points are missing
        >>> y[-10:] = np.NaN # last ten points are missing
        >>> yn, tn, indie = tnorma(y, step=-50, k=3, smooth=1, nan_at_ext='replace', show=True)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    >>> # Deal with missing data at the extremities replacing by first/last not-NaN
        >>> x = np.linspace(-3, 3, 100)
        >>> y = np.exp(-x**2) + np.random.randn(100)/10
        >>> y[0:10] = np.NaN # first ten points are missing
        >>> y[-10:] = np.NaN # last ten points are missing
        >>> yn, tn, indie = tnorma(y, step=-50, k=1, smooth=0, nan_at_ext='replace', show=True)
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    >>> # Deal with 2-D array
        >>> x = np.linspace(-3, 3, 100)
        >>> y = np.exp(-x**2) + np.random.randn(100)/10
        >>> y = np.vstack((y-1, y[::-1])).T
        >>> yn, tn, indie = tnorma(y, step=-50, k=3, smooth=1, show=True)
    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
