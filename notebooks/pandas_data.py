# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # pandas Data

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A motion-capture table contains dozens of markers, each with X, Y, and Z coordinates. How can labels help us select the right trajectory?

    We will compare flat column names and a two-level pandas index, then select and plot coordinates in marimo.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The public `data/walk1.trc` recording replaces the unavailable private file used in the old version. The loader checks the repository first, then downloads the same file.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    Pandas keeps column and time labels; Matplotlib draws the trajectories. Marimo controls update the plot reactively.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


@app.function
def repo_file(path, *, base_url="https://raw.githubusercontent.com/BMClab/BMC/master/"):
    """Find a repository file beside this notebook, or fetch a temporary copy.

    Local files take precedence. Only file acquisition falls back to GitHub;
    parsing errors remain visible to the calling notebook. urllib3 selects its
    browser transport automatically without platform-specific imports.
    """
    import hashlib
    import tempfile
    from pathlib import Path, PurePosixPath
    from urllib.parse import quote

    import marimo as mo
    import urllib3

    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise ValueError("Use a repository-relative path such as data/walk.trc")
    location = mo.notebook_dir()
    if location is not None:
        directory = Path(location).resolve()
        for base in (directory, *directory.parents):
            local = base.joinpath(*relative.parts)
            if local.is_file():
                return local
    url = base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")
    key = hashlib.sha256(base_url.rstrip("/").encode()).hexdigest()[:16]
    cached = (Path(tempfile.gettempdir()) / "bmc-notebooks-v2" / key).joinpath(*relative.parts)
    if not cached.is_file():
        response = urllib3.request("GET", url, timeout=30.0, retries=2)
        if response.status != 200:
            raise OSError(f"Cannot load {path}: HTTP {response.status} from {url}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as download:
            temporary = Path(download.name)
            download.write(response.data)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
    return cached


@app.cell
def _():
    fname = repo_file("data/walk1.trc")
    return (fname,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pandas with one index
    """)
    return


@app.cell
def _(fname, pd):
    _lines = fname.read_text().splitlines()
    _markers = [m.replace(".", "") for m in _lines[3].split("\t")[2::3] if m]
    _raw = pd.read_csv(fname, sep="\t", skiprows=6, header=None)
    _values = _raw.iloc[:, 2:2 + 3 * len(_markers)].to_numpy(dtype=float)
    df = pd.DataFrame(_values, index=_raw.iloc[:, 1].to_numpy(),
                      columns=[m + c for m in _markers for c in "XYZ"])
    df.index.name = "Time [s]"
    df.head()
    return (df,)


@app.cell
def _(df, plt):
    _ax = df.plot(y='RASISX', figsize=(8, 3), title='A plot of kinematics')
    _ax.set_ylabel('Position [mm]')
    plt.tight_layout(pad=0, h_pad=0, rect=[0, 0, 1, 0.95])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** What does `df["RASISX"]` select, and what is the index measured in?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 1': mo.md(r"""
    It selects the X coordinate of the right anterior superior iliac spine marker. The index is acquisition time in seconds. The recorded coordinates are in millimetres, as specified by the TRC header.
    """)}, expanded=False)
    return


@app.cell
def _(df, mo):
    selected_columns = mo.ui.multiselect(options=list(df.columns), value=["RASISX"], label="Trajectories")
    selected_columns
    return (selected_columns,)


@app.cell
def _(df, plt, selected_columns):
    _fig, _ax = plt.subplots(figsize=(9, 3))
    if selected_columns.value:
        df.loc[:, selected_columns.value].plot(ax=_ax)
    _ax.set(xlabel="Time [s]", ylabel="Position [mm]")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pandas multiindex

    Data with hierarchical column index ([multiindex](https://pandas.pydata.org/docs/user_guide/advanced.html)) where columns have multiple levels.
    """)
    return


@app.cell
def _(df, pd):
    df_1 = df.copy()
    df_1.columns = pd.MultiIndex.from_tuples(
        [(name[:-1], name[-1]) for name in df.columns], names=["Marker", "Coordinate"]
    )
    df_1.head()
    return (df_1,)


@app.cell
def _(df_1):
    df_1.head()
    return


@app.cell
def _(df_1):
    df_1['RASIS'].head()
    return


@app.cell
def _(df_1):
    df_1.RASIS.X.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** A table has 28 markers and three coordinates each. How many columns should the full table have? How many remain after selecting X for every marker?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 2': mo.md(r"""
    The full table has $28\times3=84$ columns. Selecting one coordinate for all 28 markers gives 28 columns. The number of time rows is unchanged.
    """)}, expanded=False)
    return


@app.cell
def _(df_1):
    df_1.xs('X', level='Coordinate', axis=1).head()
    return


@app.cell
def _(df_1):
    df_1.loc[:, (slice(None), 'X')].head()
    return


@app.cell
def _(df_1):
    df_1.swaplevel(0, 1, axis=1).head()
    return


@app.cell
def _(df_1):
    _ax = df_1.plot(y=('RASIS', 'X'), subplots=True, figsize=(8, 2), rot=0)
    return


@app.cell
def _(df_1, plt):
    _ax = df_1.plot(y='RASIS', subplots=True, sharex=True, figsize=(8, 4), rot=0, title='A plot of kinematics')
    plt.tight_layout(pad=0, h_pad=0, rect=[0, 0, 1, 0.95])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** Does swapping the two column-index levels change the measured coordinates? What information is lost in `.values`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 3': mo.md(r"""
    Swapping levels changes the label hierarchy, not the values. Converting to a NumPy array with `.values` drops the pandas labels; retain or reconstruct those labels before interpreting columns.
    """)}, expanded=False)
    return


@app.cell
def _(df_1):
    values = df_1.reset_index(drop=False).values
    values[0, :5]
    return


@app.cell
def _(df_1):
    df_1.head()
    return


@app.cell
def _(df_1):
    x = df_1.swaplevel(0, 1, axis=1)
    return (x,)


@app.cell
def _(x):
    x2 = x.unstack(level=-1)
    x2.head()
    return


@app.cell
def _(x):
    x.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Does `head()` shorten the stored table?

    2. What does `xs("X", level="Coordinate", axis=1)` select?

    3. Why keep units with a trajectory?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Checkpoint answers': mo.md(r"""
    1. No; it only displays the first rows.
    2. The X coordinate for each marker.
    3. Labels alone do not distinguish metres from millimetres.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    Try these independently; open a panel only to check your final answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 1.** A table has 10 markers, 3 coordinates, and 200 frames. Give its shape before selecting a coordinate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 1': mo.md(r"""
    (200, 30).
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** Give a pandas expression selecting X for marker RASIS from the two-level table `df_1`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 2': mo.md(r"""
    `df_1.loc[:, ("RASIS", "X")]`.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Compare `xs`, `loc`, `swaplevel`, and `unstack` in the retained examples. Check both shape and labels after each operation.

    ### Video lectures on the Internet

    [Official pandas presentation archive](https://pandas.pydata.org/community/ecosystem.html) links learning resources; [PyData video archive](https://www.youtube.com/@PyDataTV) includes pandas indexing talks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [pandas: MultiIndex and advanced indexing](https://pandas.pydata.org/docs/user_guide/advanced.html).

    - [Marimo user-interface elements](https://docs.marimo.io/api/inputs/).
    """)
    return


if __name__ == "__main__":
    app.run()
