# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "seaborn>=0.13", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Open AMTI .bsf file

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this notebook

    Follow the recording from its binary container to its force and moment channels. Run the cells, inspect the headers, and make the time-axis calculation yourself before opening each worked solution. Keep the original file and its metadata together when exporting a derived analysis.

    ## Opening challenge

    A file contains 1000 rows. Is that one second of movement, two seconds, or something else? Could six smooth curves still have the wrong units? By the end you should be able to justify the time base, channel interpretation and limits of this reader.

    ## Setup

    The notebook uses NumPy, Matplotlib, seaborn and the repository's `AMTIbsf` reader. The bundled recording supplies real data; no acquisition hardware or commercial software is needed to run the example. Binary-format support remains specific to the recorded version.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This example reads a historical NetForce `.bsf` acquisition from an [AMTI](https://www.amti.biz/) force platform. NetForce can also export ASCII data. The format history discussed here distinguishes version 100 (16-bit raw converter values) from version 105 (8-byte floating-point values). These are file-format identifiers, not a claim about the latest NetForce release. In the fully conditioned version-105 setup used here, values are already in engineering units; the supplied reader converts force and moment to N and N m.

    The original notebook mentioned [Biomechanical ToolKit](https://github.com/Biomechanical-ToolKit/BTKCore) as a reader for older version-100 files. Its former Google Code address is obsolete. Consult the toolkit's supported formats and your acquisition documentation rather than assuming every `.bsf` variant has the same layout. The supplied `AMTIbsf.py` explicitly targets version 105; the acquisition manual is the authority for other formats.

    `AMTIbsf.py` (code at the end of this notebook) is a Python function to open .bsf files in version-105 format and extract the file metadata and data from force platforms. The function signature for its typical use is:
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

    Let's see this function in action, first load the scientific libraries:
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
    ### Load the version-specific reader

    The helper is supplied with this repository; the next cell locates it and imports `loadbsf`.
    """)
    return


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
    import sys
    sys.path.insert(0, str(repo_file('functions/AMTIbsf.py').parent))  # add to pythonpath

    from AMTIbsf import loadbsf
    return (loadbsf,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — inspect before decoding.** A format stores six channels as 8-byte doubles for 1000 samples. How many bytes belong to the sample payload, excluding all headers? Why is total file size not a reliable sample count by itself?

    /// details | Worked solution — Exercise 1 — inspect before decoding

    The payload is $1000\times6\times8=48000$ bytes. Headers, instrument records and any additional sections also occupy bytes. Read the format metadata before assigning sample boundaries.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Open the recording and inspect its channels

    Before interpreting a curve, predict how many columns one six-component platform should supply. The example opens the retained `AMTIdata.bsf` fixture with plotting enabled.
    """)
    return


@app.cell
def _(loadbsf):
    data, mh, ih = loadbsf(filename=str(repo_file('data/AMTIdata.bsf')), plot=1)
    return data, ih, mh




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — construct time.** Use `mh.rate` and `data.shape[0]` to construct sample times starting at zero. Distinguish the final timestamp from the duration represented by the sample count.

    /// details | Worked solution — Exercise 2 — construct time

    Use `np.arange(data.shape[0]) / mh.rate`. This file has 1000 samples at 500 Hz: the interval is 0.002 s, the last timestamp is 1.998 s, and $N/f_s=2$ s describes the record length in sample intervals. There are only 999 intervals between first and last sample centres.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The headers (metadata from the .bsf file) are objects of the class AMTIbsf. You can visualize all the information from the headers typing for instance `mh` or ih[0]. Each information can be accessed as a property of the header typing for instance `mh.rate`. The headers can also be accessed as dictionaries typing for instance `mh.__dict__`.
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
    ## Checkpoint questions

    - Which header field establishes time spacing? Does an instrument's own `rate=0` override the main recording rate?
    - Why should engineering-unit data not be multiplied by the factory calibration a second time?
    - What information would you retain when saving the array in another format?

    /// details | Checkpoint answers

    The supplied file uses the main header's `mh.rate=500`; inspect format semantics before substituting another header field. A second calibration would alter values already converted by the acquisition/reader. Retain sample rate, channel identities/order, units, reference frame, instrument information, file version and provenance alongside the samples.

    ///
    """)
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




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. For the supplied file, report array shape, sampling interval, final timestamp (time starts at zero), and peak value in column 2 when interpreted as $F_z$.
    2. An otherwise identical recording has 2400 samples at 1200 Hz. Report the record duration $N/f_s$ and final timestamp.
    3. You receive a version-100 raw-integer file. Is changing the filename in this example sufficient to guarantee engineering-unit output? State the necessary next step.

    /// details | Final answers to the problems

    1. `(1000, 6)`; 0.002 s; 1.998 s; approximately 737.969 N.
    2. 2 s; 1.999167 s.
    3. No. Use a version-100-capable reader and the documented calibration/channel metadata, or obtain a documented engineering-unit export from the acquisition software.

    ///

    ## Go deeper

    Compare a vendor ASCII export with this reader channel by channel, including sample count, units and signs. A numerical match is a stronger validation than two similar-looking plots. The historical `shfile` memory-mapped option in the reader is a separate acquisition mode; it is not used by this saved-file example.

    ### Video lectures on the Internet

    - [MIT 8.01SC Classical Mechanics](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/) — use the force and torque lessons to interpret the six physical channels. These lessons supply mechanics background, not instructions for the proprietary binary format.

    ## References

    - [BMC's AMTI version-105 reader](https://github.com/BMClab/BMC/blob/master/functions/AMTIbsf.py) — source and format assumptions used by this example.

    - [AMTI](https://www.amti.biz/) — obtain the documentation for the actual acquisition system and file version.

    - [Biomechanical ToolKit source](https://github.com/Biomechanical-ToolKit/BTKCore) and [documentation](https://biomechanical-toolkit.github.io/docs/) — preserved alternative-reader context from the original guide.
    """)
    return

if __name__ == "__main__":
    app.run()
