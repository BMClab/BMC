import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # detect_peaks.py

    A function from [detecta](https://pypi.org/project/detecta/) - Python module to detect events in data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One way to detect peaks (local maxima) or valleys (local minima) in data is to use the property that a peak (or valley) must be greater (or smaller) than its immediate neighbors. The function `detect_peaks.py` (code at the end of this notebook) detects peaks (or valleys) based on this feature and other characteristics. The function signature is:
    ```python
    ind = detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None, title=True)
    ```
    The parameters `mph`, `mpd`, and `threshold` follow the convention of the Matlab function `findpeaks.m`.<br>
    Let's see how to use `detect_peaks.py`; first let's import the necessary Python libraries and configure the environment:

    <!-- TEASER_END -->
    """)
    return


@app.cell
def _(np):
    from detecta import detect_peaks

    # detecta 0.0.5 still calls np.in1d, which NumPy 2.4 removed; np.isin is its replacement
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    return (detect_peaks,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Running the function examples:
    """)
    return


@app.cell
def _(detect_peaks, np):
    _x = np.random.randn(100)
    _x[60:81] = np.nan
    # detect all peaks and plot data
    _ind = detect_peaks(_x, show=True)
    print(_ind)
    return


@app.cell
def _(detect_peaks, np):
    _x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # set minimum peak height = 0 and minimum peak distance = 20
    detect_peaks(_x, mph=0, mpd=20, show=True)
    return


@app.cell
def _(detect_peaks):
    _x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    # set minimum peak distance = 2
    detect_peaks(_x, mpd=2, show=True)
    return


@app.cell
def _(detect_peaks, np):
    _x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # detection of valleys instead of peaks
    detect_peaks(_x, mph=-1.2, mpd=20, valley=True, show=True)
    return


@app.cell
def _(detect_peaks):
    _x = [0, 1, 1, 0, 1, 1, 0]
    # detect both edges
    detect_peaks(_x, edge='both', show=True)
    return


@app.cell
def _(detect_peaks):
    _x = [-2, 1, -2, 2, 1, 1, 3, 0]
    # set threshold = 2
    detect_peaks(_x, threshold = 2, show=True)
    return


@app.cell
def _(detect_peaks, plt):
    _x = [-2, 1, -2, 2, 1, 1, 3, 0]
    _fig, _axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    detect_peaks(_x, show=True, ax=_axs[0], threshold=0.5, title=False)
    detect_peaks(_x, show=True, ax=_axs[1], threshold=1.5, title=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function performance

    The function `detect_peaks.py` is relatively fast but the parameter minimum peak distance (mpd) slows down the function if the data has several peaks (>1000). Try to decrease the number of peaks by tuning the other parameters or smooth the data before calling this function with several peaks in the data.<br>
    Here is a simple test of its performance:
    """)
    return


@app.cell
def _(detect_peaks, np):
    x = np.random.randn(10000)
    ind = detect_peaks(x)
    print('Data with %d points and %d peaks\n' %(x.size, ind.size))
    print('Performance (without the minimum peak distance parameter):')
    print('detect_peaks(x)')
    # magic command not supported in marimo; please file an issue to add support
    # %timeit detect_peaks(x)
    print('\nPerformance (using the minimum peak distance parameter):')
    print('detect_peaks(x, mpd=10)')
    # magic command not supported in marimo; please file an issue to add support
    # %timeit detect_peaks(x, mpd=10)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
