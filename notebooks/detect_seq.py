import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Detect indices of sequential data identical to a value

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://demotu.org/](http://demotu.org/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The function `detect_seq.py` from Python module `detecta` detects initial and final indices of sequential data identical to parameter `value` in `x`.  
        Use parameter `min_seq` to set the minimum number of sequential values to detect.

        The signature of `detect_seq.py` is:

        ```python
        idx = detect_seq(x, value=np.nan, index=False, min_seq=1, max_alert=0, show=False, ax=None)
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Installation

        ```bash
        pip install detecta
        ```

        Or

        ```bash
        conda install -c duartexyz detecta
        ```
        """
    )
    return


@app.cell
def _():
    import numpy as np
    # '%matplotlib inline' command supported automatically in marimo

    from detecta import detect_seq
    return detect_seq, np


@app.cell
def _(detect_seq):
    help(detect_seq)
    return


@app.cell
def _():
    x = [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]
    return (x,)


@app.cell
def _(detect_seq, x):
    detect_seq(x, 0)
    return


@app.cell
def _(detect_seq, x):
    detect_seq(x, 0, index=True)
    return


@app.cell
def _(detect_seq, x):
    detect_seq(x, 0, index=True, min_seq=2)
    return


@app.cell
def _(detect_seq, x):
    detect_seq(x, 10)
    return


@app.cell
def _(detect_seq, x):
    detect_seq(x, 10, index=True)
    return


@app.cell
def _(detect_seq, x):
    detect_seq(x, 1, index=True, min_seq=2, show=True)
    return


@app.cell
def _(detect_seq, x):
    detect_seq(x, 1, index=False, min_seq=1, show=True)
    return


@app.cell
def _(detect_seq, x):
    detect_seq(x, 0, index=True, min_seq=2, show=True)
    return


@app.cell
def _(detect_seq, x):
    detect_seq(x, 0, index=True, max_alert=2)
    return


@app.cell
def _(detect_seq, np):
    x_1 = [1, 2, np.nan, np.nan, 5, 4, 5, np.nan, 2, 1, 2]
    detect_seq(x_1, np.nan, index=True, max_alert=2, show=True)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
