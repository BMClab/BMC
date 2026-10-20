import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Open AMTI .bsf file

    Marcos Duarte
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When using the [NetForce](http://www.amti.biz/NetForce.aspx) software to acquire data from [AMTI](http://www.amti.biz/) force platforms, the data is saved in a proprietary binary format with the extension .bsf, but the NetForce software has an option to export data to ASCII format. The open source software [b-tk Biomechanical ToolKit](https://code.google.com/p/b-tk/) can open a .bsf file, but it works only for the old file style of NetForce (verison 100) and the current style is version 105. In the new version, 105, the acquired data is saved as 8-byte doubles and already converted to engineering units, at least when the new AMTI amplifier is used and the full conditioned mode is selected in the AMTI configuration software. In version 100, the raw data from the analog-to-digital converter was saved as 2-byte integers. In the NetForce manual, AMTI provides the necessary information for anyone to write a code and extract all the data from .bsf files (although the manual still refers to .bsf file version 100).

    `AMTIbsf.py` (code at the end of this notebook) is a Python function to open .bsf files in the new style and extract the file metadata and data from force platforms. The function signature for its typical use is:
    ```Python
    from AMTIbsf import loadbsf
    data, mh, ih = loadbsf(filename, plot=1, axs=None)
    ```
    Where:
    - filename, plot, axs are: string with path and name of the file to be opened; option to plot the data; and option with the plot axes handle.
    - data, mh, ih are: numpy array with all the data; object Main header; and object with all Instrument headers (iH[0], iH[1], ...).

    Or from command line:
    ```
    python AMTIbsf.py filename
    ```
    This code can also open the 'shfile' memory-mapped file by NetForce, just use `shfile` as the filename.

    Let's see this function in action, first the customary imports and customization for the Python environment:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    import seaborn as sns
    sns.set_context("notebook", font_scale=1.2,
                    rc={"lines.linewidth": 2, "lines.markersize": 8, "axes.titlesize": 'x-large'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In my setup, functions are in a specific diretory, which I have to add it to Python's path:
    """)
    return


@app.function
def repo_file(path):
    """Local path of a file of the BMC repository, e.g. "data/walk.trc".

    Where only the notebook is available (e.g., on molab), the file is
    downloaded from GitHub to a temporary folder.
    """
    import tempfile
    import urllib.request
    from pathlib import Path

    import marimo as mo

    local = mo.notebook_dir() / ".." / path
    if local.exists():
        return local
    cached = Path(tempfile.gettempdir()) / "BMC" / path
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/BMClab/BMC/master/{path}", cached)
    return cached


@app.cell
def _():
    import sys
    sys.path.insert(0, str(repo_file('functions/AMTIbsf.py').parent))  # add to pythonpath

    from AMTIbsf import loadbsf
    return (loadbsf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Openning a .bsf file with acquired data of one force platform and with the option to plot the data:
    """)
    return


@app.cell
def _(loadbsf):
    data, mh, ih = loadbsf(filename=str(repo_file('data/AMTIdata.bsf')), plot=1)
    return ih, mh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The headers (metadata from the .bsf file) are objects of the class AMTIbsf. You can visualize all the information from the headers typing for instance `mh` or ih[0]. Each information can be accessed as a propertiy of the header typing for instance `mh.rate`. The headers can also be acessed as dictionaries typing for instance `mh.__dict__`.
    """)
    return


@app.cell
def _(mh):
    mh.rate
    return


@app.cell
def _(mh):
    mh
    return


@app.cell
def _(ih):
    ih[0].__dict__
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function `AMTIbsf.py`
    """)
    return


@app.cell
def _(mo):
    mo.ui.code_editor(repo_file("functions/AMTIbsf.py").read_text(), language="python", disabled=True)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
