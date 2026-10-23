# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Postural sway: rambling and trembling

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    The centre of pressure moves even while a person tries to stand still. Can we describe that motion as a moving reference plus deviations around it?

    We will examine the instant-equilibrium-point hypothesis, perform the decomposition on a public recording, and compare it with a COP–COG estimate. The batch examples retain subject-level summaries and explicit trial selection.

    Predict the result before running each example. Try the **Exercises** before opening their worked solutions. Finish with the **Checkpoint questions** and **Problems**; the latter provide final answers only.

    Open this file locally in marimo, on [marimo.app](https://marimo.app/), or in [molab](https://molab.marimo.io/notebooks). Data come from the public Santos–Duarte balance dataset. The selected trial and metadata are downloaded if they are not already available; no private computer paths or automatic population-wide processing remain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    NumPy and pandas hold the recordings; SciPy interpolates the zero-force points, and Matplotlib shows the components. The data loader and analysis functions are included in this file.
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


@app.function
def iep_decomp(time, force, cop, demean_force=True):
    """Return the valid sample domain, rambling/trembling, and zero-force knots.

    A cubic interpolant requires four distinct knots. End samples outside the
    knot range are excluded rather than extrapolated. A run of exact zeros
    contributes its midpoint. Removing the force mean follows the original
    tutorial convention; it is an explicit offset correction assumption.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline
    time, force, cop = [np.asarray(v, dtype=float) for v in (time, force, cop)]
    if any((v.ndim != 1 for v in (time, force, cop))) or not len(time) == len(force) == len(cop):
        raise ValueError('time, force, and cop must be equal-length one-dimensional arrays')
    if len(time) < 4 or not all((np.isfinite(v).all() for v in (time, force, cop))) or np.any(np.diff(time) <= 0):
        raise ValueError('Use finite samples with strictly increasing time')
    centered = force - force.mean() if demean_force else force.copy()
    if not np.any(centered):
        raise ValueError('A constant zero force does not identify discrete equilibrium points')
    before = np.flatnonzero(centered[:-1] * centered[1:] < 0)
    fraction = -centered[before] / (centered[before + 1] - centered[before])
    crossings = time[before] + fraction * (time[before + 1] - time[before])
    zero_edges = np.diff(np.r_[False, centered == 0, False].astype(int))
    starts = np.flatnonzero(zero_edges == 1)
    stops = np.flatnonzero(zero_edges == -1) - 1
    knots = np.unique(np.r_[crossings, (time[starts] + time[stops]) / 2])
    if len(knots) < 4:
        raise ValueError('Cubic rambling decomposition requires at least four zero-force knots')
    knot_cop = np.interp(knots, time, cop)
    selected = (time >= knots[0]) & (time <= knots[-1])
    rambling = CubicSpline(knots, knot_cop, extrapolate=False)(time[selected])
    return {'time': time[selected], 'force': centered[selected], 'cop': cop[selected], 'rambling': rambling, 'trembling': cop[selected] - rambling, 'knot_time': knots, 'knot_cop': knot_cop}


@app.function
def cogve(COP, freq, mass, height, show=False, ax=None):
    """COGv estimation using COP data based on the inverted pendulum model.

    This function estimates the center of gravity vertical projection (COGv)
    displacement from the center of pressure (COP) displacement at the
    anterior-posterior direction during quiet upright standing. COP and COGv
    displacements are measurements useful to quantify the postural sway of a
    person while standing.

    The COGv displacement is estimated by low-pass filtering the COP
    displacement in the frequency domain according to the person's moment
    of rotational inertia as a single inverted pendulum [1]_.

    Parameters
    ----------
    COP    : 1D array_like
             center of pressure data [cm]
    freq   : float
             sampling frequency of the COP data
    mass   : float
             body mass of the subject [kg]
    height : float
             height of the subject [cm]
    show   : bool, optional (default = False)
             True (1) plots data and results in a matplotlib figure
             False (0) to not plot
    ax     : matplotlib.axes.Axes instance, optional (default = None)

    Returns
    -------
    COGv   : 1D array
             center of gravity vertical projection data [cm]

    References
    ----------
    .. [1] https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ip_model.py

    Examples
    --------
    >>> from cogve import cogve
    >>> y = np.cumsum(np.random.randn(3000))/50
    >>> cogv = cogve(y, freq=100, mass=70, height=170, show=True)
    """
    import numpy as np
    from scipy.signal._arraytools import odd_ext
    import scipy.fftpack
    COP = np.asarray(COP)
    height = height / 100
    g = 9.8
    hcog = 0.56 * height - 0.039 * height
    I = mass * 0.0533 * height ** 2 + mass * hcog ** 2
    w02 = mass * g * hcog / I
    COP = odd_ext(COP, n=freq)
    COPm = np.mean(COP)
    COP = COP - COPm
    N = COP.size
    COPfft = scipy.fftpack.fft(COP, n=N) / N
    w = 2 * np.pi * scipy.fftpack.fftfreq(n=N, d=1 / freq)
    TF = w02 / (w02 + w ** 2)
    COGv = np.real(scipy.fftpack.ifft(TF * COPfft) * N)
    COGv = COGv[0:N]
    COP, COGv = (COP + COPm, COGv + COPm)
    COP, COGv = (COP[freq:-freq], COGv[freq:-freq])
    if show:
        plot_cogve(COP, COGv, freq, ax)
    return COGv


@app.function
def plot_cogve(COP, COGv, freq, ax):
    """Plot results of the cogve function, see its help."""
    import numpy as np
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        time = np.linspace(0, COP.size / freq, COP.size)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(time, COP, color=[0, 0, 1, 0.8], lw=2, label='COP')
        ax.plot(time, COGv, color=[1, 0, 0, 0.8], lw=2, label='COGv')
        ax.legend(fontsize=14, loc='best', framealpha=0.5, numpoints=1)
        ax.set_xlabel('Time [s]', fontsize=14)
        ax.set_ylabel('Amplitude [cm]', fontsize=14)
        ax.set_title('COGv estimation using the COP data', fontsize=16)
        ax.set_xlim(time[0], time[-1])
        plt.grid()
        plt.show()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A hypothesis about balance control

    The rambling–trembling hypothesis describes a slowly migrating reference position and deviations around it. Gurfinkel and colleagues' slow support-surface tilts, and observations of sway during prolonged standing, motivated this interpretation. It is a model of control, not a direct measurement of two independently observed neural subsystems.

    The original illustrations show force vectors, prolonged standing, and the scale structure of sway:
    """)
    return


@app.cell
def _(mo):
    mo.vstack([mo.image(src=repo_file("images/forcevectors.png"), width=600), mo.md('Force-vector illustration during quiet standing.')])
    return


@app.cell
def _(mo):
    mo.vstack([mo.image(src=repo_file("images/prolongedstanding.png"), width=600), mo.md('Illustration of prolonged-standing sway.')])
    return


@app.cell
def _(mo):
    mo.vstack([mo.image(src=repo_file("images/fractal.png"), width=600), mo.md('Illustration of the scale structure of sway.')])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the decomposition

    At an instant when horizontal ground-reaction force is zero, horizontal centre-of-mass acceleration is zero under the assumed force balance. This does **not** require zero velocity or full static equilibrium. The COP positions at these instants are called instant equilibrium points (IEPs). A cubic spline through them estimates **rambling**; **trembling = COP − rambling**.

    The control interpretation proposes an intended body position, its migration, deviations from it, and restoring actions associated with apparent stiffness. The computed components alone do not establish that mechanism.
    """)
    return


@app.cell
def _(mo):
    mo.image(src=repo_file("images/rambtremb.png"), width=600)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** At one instant the COP is 2.0 cm and the estimated rambling is 1.4 cm. What is trembling? Does zero horizontal force imply zero horizontal velocity?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 1': mo.md(r"""
    Trembling is $2.0-1.4=0.6$ cm. Zero horizontal force implies zero horizontal acceleration under the model, not zero velocity.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find zero-force instants between samples

    The earlier code chose the sample immediately before each sign change. Here we interpolate the crossing time within that sampling interval and interpolate COP at that time. Exact-zero runs contribute their midpoint. We keep only the domain between the first and last knot, avoiding extrapolation. Four distinct knots are required for the cubic interpolation.

    As in the original analysis, the force mean is removed before locating crossings. This treats the mean as an offset; assess that assumption for a recording rather than silently interpreting the centred force as the raw measurement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** At times 0 and 0.01 s, force changes from −1 N to +3 N. With linear interpolation and no offset correction, when is the zero crossing?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 2': mo.md(r"""
    The zero is one quarter of the way through the interval: $0.0025$ s. Choosing the earlier sample would shift the knot by 2.5 ms.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the public trial information

    The Santos and Duarte dataset includes repeated trials under different visual and support-surface conditions. Select trials deliberately: each selection downloads a recording. The default is the single trial used in the original walkthrough, BDS00038.
    """)
    return


@app.cell
def _(mo, pd):
    dataset_base = "https://raw.githubusercontent.com/demotu/datasets/master/"
    metadata = pd.read_csv(repo_file("BDS/BDSinfo.txt", base_url=dataset_base), sep="\t")
    trial_choices = mo.ui.multiselect(options=metadata["Trial"].tolist(), value=["BDS00038"], label="Trials to analyse")
    trial_choices
    return dataset_base, metadata, trial_choices


@app.cell
def _(dataset_base, metadata, mo, np, pd, trial_choices):
    selected_results = {}
    for _trial in trial_choices.value:
        _table = pd.read_csv(repo_file(f"BDS/data/{_trial}.txt", base_url=dataset_base), sep="\t")
        _table.columns = ["Time", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy"]
        _components = iep_decomp(_table.Time, _table.Fx, _table.COPx)
        _person = metadata.loc[metadata.Trial == _trial].iloc[0]
        _frequency = int(round(1 / np.mean(np.diff(_table.Time))))
        _cog = cogve(_components["cop"], freq=_frequency, mass=float(_person.Weight), height=float(_person.Height), show=False)
        selected_results[_trial] = {"recording": _table, "components": _components, "person": _person, "cog": _cog}
    mo.md(f"Analysed **{len(selected_results)}** selected trial(s).")
    return (selected_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect the recording and its components

    The first selected trial is shown below. Compare horizontal forces, vertical force, the two COP coordinates, and the force-crossing knots before interpreting the decomposition. The COP units in this dataset are centimetres.
    """)
    return


@app.cell
def _(mo, plt, selected_results):
    if selected_results:
        _item = next(iter(selected_results.values()))
        _data = _item["recording"]
        _fig, _axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        _data.plot(x="Time", y=["Fx", "Fy"], ax=_axs[0])
        _data.plot(x="Time", y="Fz", ax=_axs[1])
        _data.plot(x="Time", y=["COPx", "COPy"], ax=_axs[2])
        _axs[0].set_ylabel("Force [N]")
        _axs[1].set_ylabel("Force [N]")
        _axs[2].set_ylabel("COP [cm]")
        _display = _fig
    else:
        _display = mo.md("Select at least one trial to display its recording.")
    _display
    return


@app.cell
def _(mo, np, plt, selected_results):
    if selected_results:
        _item = next(iter(selected_results.values()))
        _c = _item["components"]
        _fig, _axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        for _ax, _key in zip(_axs, ["force", "cop", "rambling", "trembling"]):
            _ax.plot(_c["time"], _c[_key])
            _ax.set_ylabel(_key + (" [N]" if _key == "force" else " [cm]"))
        _axs[0].plot(_c["knot_time"], np.zeros(len(_c["knot_time"])), ".")
        _axs[2].plot(_c["knot_time"], _c["knot_cop"], ".")
        _axs[-1].set_xlabel("Time [s]")
        _display = _fig
    else:
        _display = mo.md("Select a trial to display the decomposition.")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** Why should the spline not be extended to the recording endpoints when no IEP knots bracket those endpoints?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 3': mo.md(r"""
    Beyond the knot range the curve is an extrapolation controlled by spline boundary behavior, not by surrounding IEPs. Restricting the analysis to the knot range avoids treating that extension as measured evidence.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare COP and estimated COG projection

    The inverted-pendulum model offers a different decomposition: estimated COG projection and COP minus that estimate. It relies on small-sway, single-link assumptions and anthropometric parameters. We use each selected subject's recorded height and mass, rather than the previous fixed height and approximate weight-to-mass conversion. Each trial uses its own valid time interval; indices from another trial are never reused.
    """)
    return


@app.cell
def _(mo, plt, selected_results):
    if selected_results:
        _item = next(iter(selected_results.values()))
        _c = _item["components"]
        _fig, _ax = plt.subplots(figsize=(10, 3))
        _ax.plot(_c["time"], _c["cop"], label="COP")
        _ax.plot(_c["time"], _item["cog"], label="estimated COG projection")
        _ax.set(xlabel="Time [s]", ylabel="Position [cm]")
        _ax.legend()
        _display = _fig
    else:
        _display = mo.md("Select a trial to compare COP and estimated COG projection.")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summarize repeated trials without counting them as independent subjects

    We first calculate within-trial sample SDs, then take the median across repeated trials for each subject and condition. Group descriptions use those subject-level summaries. The default single trial cannot support a population comparison. Excluding participants merely to balance group counts can bias an analysis; define exclusions from a justified study question before inspecting results.
    """)
    return


@app.cell
def _(mo, np, pd, selected_results):
    _rows = []
    for _trial, _item in selected_results.items():
        _c = _item["components"]
        _p = _item["person"]
        _rows.append({"Trial": _trial, "Subject": _p.Subject, "Vision": _p.Vision,
                      "Surface": _p.Surface, "AgeGroup": _p.AgeGroup,
                      "COP_sd": np.std(_c["cop"], ddof=1),
                      "Rambling_sd": np.std(_c["rambling"], ddof=1),
                      "Trembling_sd": np.std(_c["trembling"], ddof=1),
                      "COGv_sd": np.std(_item["cog"], ddof=1),
                      "COP_COGv_sd": np.std(_c["cop"]-_item["cog"], ddof=1)})
    summary = pd.DataFrame(_rows)
    if len(summary):
        subject_summary = summary.groupby(["Subject", "Vision", "Surface", "AgeGroup"], as_index=False).median(numeric_only=True)
        _display = mo.vstack([summary, subject_summary, mo.download(summary.to_csv(index=False).encode(), filename="balance_summary.csv", label="Download selected-trial summary")])
    else:
        subject_summary = pd.DataFrame()
        _display = mo.md("Select trials to compute subject summaries.")
    _display
    return subject_summary, summary


@app.cell
def _(mo, subject_summary):
    if len(subject_summary):
        _description = subject_summary.groupby(["AgeGroup", "Vision", "Surface"])[["COP_sd", "Rambling_sd", "Trembling_sd", "COGv_sd", "COP_COGv_sd"]].agg(["count", "mean", "std"])
    else:
        _description = mo.md("No selected subject summaries.")
    _description
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare selected subject summaries

    The original two sets of group-comparison plots are retained as an optional display. Each dot represents one subject/condition summary, using the median of that subject's selected repeated trials. Panels separate support surfaces, colours identify age groups, and counts state how many subjects contributed. These are descriptive comparisons; selecting one trial does not establish a population effect.
    """)
    return


@app.cell
def _(mo):
    comparison_plots = mo.ui.checkbox(value=False, label="Show selected-subject comparison plots")
    comparison_plots
    return (comparison_plots,)


@app.cell
def _(comparison_plots, mo, np, plt, subject_summary):
    if comparison_plots.value and len(subject_summary):
        _surfaces = list(subject_summary["Surface"].unique())
        _visions = list(subject_summary["Vision"].unique())
        _ages = list(subject_summary["AgeGroup"].unique())
        _figures = []
        for _metrics in [("COP_sd", "Rambling_sd", "Trembling_sd"), ("COP_sd", "COGv_sd", "COP_COGv_sd")]:
            _fig, _axs = plt.subplots(3, len(_surfaces), squeeze=False, sharex=True, sharey="row", figsize=(5*len(_surfaces), 7))
            for _column, _surface in enumerate(_surfaces):
                _surface_data = subject_summary.loc[subject_summary.Surface == _surface]
                for _row, _metric in enumerate(_metrics):
                    _ax = _axs[_row, _column]
                    for _colour, _age in enumerate(_ages):
                        _subset = _surface_data.loc[_surface_data.AgeGroup == _age]
                        _x = np.array([_visions.index(v) for v in _subset.Vision])
                        _offset = (_colour - (len(_ages)-1)/2)*0.12
                        _ax.scatter(_x+_offset, _subset[_metric], label=str(_age), color=f"C{_colour}")
                    _ax.set_ylabel(_metric.replace("_sd", " SD")+" [cm]")
                    _ax.set_xticks(range(len(_visions)), _visions)
                    _ax.set_title(f"{_surface}: n={_surface_data.Subject.nunique()} subject(s)")
                    if _row == 0:
                        _ax.legend(title="Age group")
            _fig.tight_layout()
            _figures.append(_fig)
        _display = mo.vstack(_figures)
    else:
        _display = mo.md("Enable the comparison display after choosing trials. The default single trial is a demonstration, not a group study.")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** A subject contributes three repeated trials. How many independent people are represented? Why summarize within subject first?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 4': mo.md(r"""
    There is one person. Summarizing the repeats avoids treating repeated measurements from that person as three independent subjects; between-subject inference still requires an appropriate design.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Component amplitudes do not add

    The original caution about RMS is essential. Although COP = rambling + trembling pointwise, RMS amplitudes do not generally add. Mean-square values contain a cross term; variances contain twice the covariance. In the synthetic example below, two complete-cycle sinusoids are orthogonal, so their mean squares add, but their RMS values do not.
    """)
    return


@app.cell
def _(np, plt):
    _t = np.arange(100) / 100
    _r = np.sin(2*np.pi*_t)
    _q = 0.1*np.sin(20*np.pi*_t)
    _c = _r + _q
    rms_example = {"COP": np.sqrt(np.mean(_c**2)), "rambling": np.sqrt(np.mean(_r**2)), "trembling": np.sqrt(np.mean(_q**2))}
    _fig, _ax = plt.subplots(figsize=(9, 3))
    for _values, _name in [(_c, "COP"), (_r, "rambling"), (_q, "trembling")]:
        _ax.plot(_t, _values, label=f"{_name}: RMS={rms_example[_name]:.3f}")
    _ax.set(xlabel="Time [s]", ylabel="Synthetic amplitude [a.u.]")
    _ax.legend()
    _fig
    return (rms_example,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 5.** If two uncorrelated, zero-mean components have variances 4 and 1, what is the variance and SD of their sum?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Solution to Exercise 5': mo.md(r"""
    The variance is $4+1=5$, while the SD is $\sqrt5\approx2.236$. Adding the component SDs would give 3 and is incorrect. If covariance is nonzero, include twice that covariance.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Is rambling a direct recording of a neural reference signal?

    2. Does zero horizontal force mean the body is motionless?

    3. Does decomposition imply independent components?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Checkpoint answers': mo.md(r"""
    1. No; it is a model-derived estimate.
    2. No.
    3. No; correlation must be assessed.
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
    **Problem 1.** COP is −1.2 cm and rambling is −0.8 cm. Find trembling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 1': mo.md(r"""
    −0.4 cm.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 2.** Two zero-mean orthogonal sinusoids have amplitudes 1 and 0.1. Give the RMS of their sum.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 2': mo.md(r"""
    $\sqrt{0.505}\approx0.71063$.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Problem 3.** A recording has only three distinct zero-force knots. Does the supplied cubic decomposition run?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Answer to Problem 3': mo.md(r"""
    No; it raises a ValueError requiring at least four knots.
    """)}, expanded=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Compare preprocessing, centring, crossing conventions, and boundary trimming before comparing studies. The [inverted-pendulum guide](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ip_model.py) explains the COP–COG model.

    ### Video lectures on the Internet

    [MIT Signals and Systems lectures](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/video_galleries/video-lectures/) give the interpolation and signal-decomposition background.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Gurfinkel et al. (1995), Kinesthetic reference for human orthograde posture. Neuroscience 68, 229–243.

    - Santos and Duarte (2016), [A public data set of human balance evaluations](https://doi.org/10.7717/peerj.2648).

    - Zatsiorsky and Duarte (1999), [Instant equilibrium point and its migration in standing tasks](https://doi.org/10.1123/mcj.3.1.28).

    - Zatsiorsky and Duarte (2000), [Rambling and trembling in quiet standing](https://doi.org/10.1123/mcj.4.2.185).

    - [Public balance dataset and documentation](https://github.com/demotu/datasets/tree/master/BDS).
    """)
    return


if __name__ == "__main__":
    app.run()
