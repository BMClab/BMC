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


@app.cell(hide_code=True)
def _():
    try:
        from notebook_resources import repo_file
    except ModuleNotFoundError as _error:
        if _error.name != "notebook_resources":
            raise
        raise ModuleNotFoundError(
            "This notebook needs notebook_resources.py. Download it from:\n"
            "https://raw.githubusercontent.com/BMClab/BMC/master/functions/notebook_resources.py\n"
            "Save it in the same folder as this notebook, then rerun this cell.",
            name="notebook_resources",
        ) from None
    return (repo_file,)


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
def _(repo_file):
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
    mo.md(r"""
    /// details | Solution to Exercise 1

    There are $1000/100=10$ analog samples per point frame. Align using time, not equal row numbers.
    ///
    """)
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
    mo.md(r"""
    /// details | Solution to Exercise 2

    No: the coordinate origin may pass through a valid point. A negative C3D residual flags invalid point data. The reader marks all three coordinates of that point NaN, while preserving valid zero values.
    ///
    """)
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
    mo.md(r"""
    /// details | Checkpoint answers

    1. Their sample rates differ.
    2. No; they may contain forces, moments, EMG voltages, or other signals.
    3. The first recorded point frame.
    ///
    """)
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
    mo.md(r"""
    /// details | Answer to Problem 1

    4,870.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** What is the interval between marker rows at 100 Hz?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// details | Answer to Problem 2

    0.01 s.
    ///
    """)
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
