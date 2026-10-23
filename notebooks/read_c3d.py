# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "urllib3>=2.2", "c3d>=0.6"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read C3D data into labelled tables

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A motion-capture recording contains both marker positions and analog channels. Why do their tables have different row counts?

    We will read the same synchronized recording into two pandas tables and inspect units, labels, missing points, and sampling rates.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). The example is `data/Gait.c3d`, distributed with this repository. Local files take precedence; otherwise the same public recording is downloaded.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The pure-Python `c3d` package decodes the file, NumPy handles arrays, pandas labels them, and Matplotlib plots them. The readers are included in this notebook so a single-file cloud session can run it.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


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


@app.function
def load_c3d(path):
    """Return metadata, marker coordinates, and calibrated analog samples.

    Times are seconds relative to the first recorded point frame. Negative
    residuals mark invalid points; a zero coordinate is a valid measurement.
    Analog scaling is performed once by c3d.Reader. Analog channels are not
    assumed to be force components or rotated in arbitrary groups of three.
    """
    import c3d
    import numpy as np
    import pandas as pd

    def normalized_labels(reader, group, used):
        # USED defines data width; LABELS can legally be shorter or longer.
        labels = []
        suffix = ""
        while len(labels) < used:
            parameter = reader.get(f"{group}:LABELS{suffix}")
            if parameter is None:
                break
            labels.extend(parameter.string_array)
            suffix = 2 if suffix == "" else suffix + 1
        labels = [str(label).strip(" \t\r\n\x00") for label in labels[:used]]
        labels.extend([""] * (used - len(labels)))
        reserved = set(labels)
        seen = set()
        result = []
        for index, label in enumerate(labels):
            base = label or f"{group}_{index + 1}"
            candidate = base
            number = 2
            while candidate in seen or (candidate != label and candidate in reserved):
                candidate = f"{base}_{number}"
                number += 1
            result.append(candidate)
            seen.add(candidate)
        return result

    with open(path, "rb") as handle:
        reader = c3d.Reader(handle)
        frames = list(reader.read_frames())
        if not frames:
            raise ValueError("The C3D file contains no data frames")
        point_labels = normalized_labels(reader, "POINT", reader.point_used)
        analog_labels = normalized_labels(reader, "ANALOG", reader.analog_used)
        units = reader.get("POINT:UNITS")
        event_used = reader.get("EVENT:USED")
        events = reader.get("EVENT:LABELS")
        event_times = reader.get("EVENT:TIMES")
        info = {
            "point_rate": float(reader.point_rate),
            "analog_rate": float(reader.analog_rate),
            "first_frame": int(reader.first_frame),
            "frames": len(frames),
            "point_units": units.string_value.strip().strip("\x00")
            if units
            else "unspecified",
            "analog_units": list(reader.get("ANALOG:UNITS").string_array)
            if reader.get("ANALOG:UNITS")
            else [],
            "parameter_groups": {
                name: [key for key, _ in group.param_items()]
                for name, group in reader.group_items()
            },
            "events": list(events.string_array) if events else [],
            "event_times": event_times.float_array.tolist() if event_times else [],
            "event_count": int(event_used.int16_value) if event_used else 0,
        }
        all_points = np.stack([points for _, points, _ in frames])
        xyz = all_points[:, :, :3].copy()
        xyz[all_points[:, :, 3] < 0] = np.nan
        columns = pd.MultiIndex.from_product(
            [point_labels, list("XYZ")], names=["Marker", "Coordinate"]
        )
        points = pd.DataFrame(
            xyz.reshape(len(frames), -1),
            columns=columns,
            index=np.arange(len(frames)) / reader.point_rate,
        )
        if reader.analog_used:
            samples = np.concatenate([analog for _, _, analog in frames], axis=1).T
            analog = pd.DataFrame(
                samples,
                columns=analog_labels,
                index=np.arange(len(samples)) / reader.analog_rate,
            )
        else:
            analog = pd.DataFrame()
    points.index.name = analog.index.name = "Time [s]"
    return info, points, analog


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read one synchronized acquisition

    This guide replaces the old native EZC3D example and unavailable private path with a portable reader and a public fixture. `load_c3d` returns metadata, point coordinates, and analog samples. It keeps the calibrated analog values supplied by the reader; do not apply the scale factors a second time.
    """)
    return


@app.cell
def _():
    info, pt, an = load_c3d(repo_file("data/Gait.c3d"))
    info
    return an, info, pt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** For point rate 100 Hz and analog rate 1,000 Hz, how many analog samples belong to one point frame?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 1': mo.md(r"""
    There are $1000/100=10$ analog samples per point frame. Align using time, not equal row numbers.
    """)}, expanded=False)
    return


@app.cell
def _(pt):
    pt.head()
    return


@app.cell
def _(an):
    an.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select coordinates and channels

    The point columns have two levels: marker and coordinate. The analog columns retain channel names. Repeated labels receive a numeric suffix so selection is unambiguous. Positions use the point units in the metadata; analog units can differ by channel. A channel measured in volts is not automatically a force.
    """)
    return


@app.cell
def _(pt):
    pt.loc[:, ("RASI", "X")].head()
    return


@app.cell
def _(an):
    an[["F1Z", "F2Z"]].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** Should a point with X=0 be marked missing? What evidence does the reader use instead?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 2': mo.md(r"""
    No: the coordinate origin may pass through a valid point. A negative C3D residual flags invalid point data. The reader marks all three coordinates of that point NaN, while preserving valid zero values.
    """)}, expanded=False)
    return


@app.cell
def _(an, plt, pt):
    _fig, _axs = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    pt.loc[:, ("RASI", "X")].plot(ax=_axs[0])
    an[["F1Z", "F2Z"]].plot(ax=_axs[1])
    _axs[0].set_ylabel("Position [mm]")
    _axs[1].set_ylabel("Force [N]")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Why are force and marker rows not interchangeable?

    2. Do analog channels always come in XYZ triplets?

    3. What defines time zero in these tables?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Checkpoint answers': mo.md(r"""
    1. Their sample rates differ.
    2. No; they may contain forces, moments, EMG voltages, or other signals.
    3. The first recorded point frame.
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
    **Problem 1.** How many analog rows correspond to 487 point frames when the rate ratio is 10?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 1': mo.md(r"""
    4,870.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** What is the interval between marker rows at 100 Hz?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 2': mo.md(r"""
    0.01 s.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Use the [C3D format guide](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/open_c3d_file.py) to inspect the binary header and browse marker frames. The [pandas guide](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/pandas_data.py) expands on labelled selection.

    ### Video lectures on the Internet

    [OpenSim webinar archive](https://opensim.stanford.edu/support/event_details.php?id=30) provides motion-analysis context. [MIT Signals and Systems lectures](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/video_galleries/video-lectures/) explain the sampling-rate distinction used here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [C3D technical user guide](https://c3d.org/docs/C3D_User_Guide.pdf).

    - [Python c3d reader](https://github.com/EmbodiedCognition/py-c3d).

    - [C3D example data](https://www.c3d.org/sampledata.html).
    """)
    return


if __name__ == "__main__":
    app.run()
