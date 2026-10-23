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
    # Open C3D files

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    How can a binary file preserve marker trajectories, force channels, and the information needed to interpret them?

    We will inspect a header, read metadata and calibrated samples, and browse a 3D marker trajectory without a desktop-only animation backend.

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
    ## Text, bytes, and the C3D structure

    All files are bytes. A text format interprets bytes with an encoding such as UTF-8; understanding its columns still requires a schema and units. A binary format specifies numeric representation, byte order, and layout. Binary storage can be compact, but size and speed depend on the format and data.

    C3D stores synchronized 3D-point and analog data, standard and custom parameters, and optional events. Its sections use 512-byte blocks: a header, a parameter section, and data blocks. The parameter section identifies the processor representation, so a parser must not assume that every file uses the host machine's byte order.

    Let us inspect the supplied fixture's header. The short example below reads integer fields using the declared byte order. Floating-point representation also varies by processor; `c3d.Reader` handles that conversion for the complete file. The supplied recording uses DEC representation.
    """)
    return


@app.cell
def _():
    import struct
    with open(repo_file("data/Gait.c3d"), "rb") as _file:
        _header = _file.read(512)
        _parameter_block, _key = struct.unpack("BB", _header[:2])
        if _key != 80:
            raise ValueError("Missing C3D header key")
        _file.seek(512 * (_parameter_block - 1) + 3)
        _processor = _file.read(1)[0]
        if _processor not in (84, 85, 86):
            raise ValueError("Unknown C3D processor representation")
        _byteorder = ">" if _processor == 86 else "<"
        header_summary = {
            "points": struct.unpack_from(_byteorder + "H", _header, 2)[0],
            "first_frame": struct.unpack_from(_byteorder + "H", _header, 6)[0],
            "last_frame": struct.unpack_from(_byteorder + "H", _header, 8)[0],
            "processor": {84: "Intel", 85: "DEC", 86: "MIPS"}[_processor],
            "point_rate_bytes": _header[20:24].hex(),
        }
    header_summary
    return (header_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** The first frame is 1 and the last is 487. How many frames are stored? Why specify byte order in the numeric unpacking format?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 1': mo.md(r"""
    There are $487-1+1=487$ frames. An explicit byte order avoids an accidental dependence on the computer running the example. The file's processor byte determines that choice.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the complete recording

    The old notebook used platform-specific BTK binaries for Python 2 and skipped most examples when those binaries were unavailable. The pure-Python reader now executes the examples in all supported editors. It exposes parameter groups, channel labels, point units, analog units, and optional events.
    """)
    return


@app.cell
def _():
    info, pt, an = load_c3d(repo_file("data/Gait.c3d"))
    info
    return an, info, pt


@app.cell
def _(info, mo):
    mo.md(f"**{info['frames']} frames**, point rate **{info['point_rate']:g} Hz**, analog rate **{info['analog_rate']:g} Hz**. This file has **{info['event_count']} parameter-defined events**.")
    return


@app.cell
def _(an, info, pt):
    {"markers": list(pt.columns.get_level_values("Marker").unique()), "analog_channels": list(an.columns), "parameter_groups": info["parameter_groups"]}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** The file has 33 points and 487 frames. What is the shape of its XYZ coordinate array if rows are frames?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 2': mo.md(r"""
    The array shape is `(487, 33, 3)`. A flat table has shape `(487, 99)`. Invalid points are marked from their residuals, not by replacing every zero.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect calibrated force channels

    The reader applies C3D analog offsets and scale factors. The sample file labels F1Z and F2Z in newtons. Their sign follows the file's convention; identify the coordinate system before interpreting a force as upward or downward.
    """)
    return


@app.cell
def _(an, plt):
    _fig, _ax = plt.subplots(figsize=(9, 3))
    an[["F1Z", "F2Z"]].plot(ax=_ax)
    _ax.set_ylabel("Force [N]")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** If an analog channel is already calibrated in newtons, should we divide it by its stored scale factor again?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 3': mo.md(r"""
    No. Scaling a second time changes the calibrated values and produces an incorrect force. Inspect units and reader documentation before applying a conversion.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Browse the marker motion

    Move the frame control to inspect the recording. Fixed axis limits make displacement visible across frames. The slider replaces the original separate-window animation and automatic video-file writes.
    """)
    return


@app.cell
def _(mo, pt):
    frame_control = mo.ui.slider(start=0, stop=len(pt)-1, step=1, value=130, label="Frame offset", show_value=True)
    frame_control
    return (frame_control,)


@app.cell
def _(frame_control, info, np, plt, pt):
    _xyz = pt.to_numpy().reshape(len(pt), -1, 3)
    _frame = frame_control.value
    _fig = plt.figure(figsize=(7, 5))
    _ax = _fig.add_subplot(projection="3d")
    _ax.scatter(*_xyz[_frame].T)
    for _coordinate, _set in enumerate((_ax.set_xlim, _ax.set_ylim, _ax.set_zlim)):
        _set(np.nanmin(_xyz[:, :, _coordinate]), np.nanmax(_xyz[:, :, _coordinate]))
    _ax.set(xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]", title=f"Time {_frame / info['point_rate']:.2f} s")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Does a binary file explain its units without metadata?

    2. Why preserve zero coordinates?

    3. Does absence of event labels mean that no gait events occurred?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Checkpoint answers': mo.md(r"""
    1. No.
    2. Zero can be a valid coordinate.
    3. No; the events may simply not have been annotated.
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
    **Problem 1.** At 100 Hz, what relative time belongs to zero-based frame offset 130?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 1': mo.md(r"""
    1.30 s.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** How many bytes occupy three C3D blocks?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 2': mo.md(r"""
    1,536 bytes.
    """)}, expanded=False)
    return


@app.cell
def _(mo):
    mo.accordion({"Original walking animation": mo.image(src=repo_file("images/walking.gif"), width=500)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    The companion [table-based reader](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/read_c3d.py) focuses on pandas selection. Consult the C3D manual before extending the minimal header demonstration to other processor types or custom parameters.

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
