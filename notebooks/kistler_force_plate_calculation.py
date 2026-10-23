# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kistler force plate calculation

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this notebook

    Start with voltages and predict the force before looking at the plot. Keep track of channel order, amplifier range, coordinates and units through each calculation. The worked exercises explain the transformations; the final problems let you check them independently.

    ## Opening challenge

    Two laboratories record the same force using different amplifier ranges. Their voltages differ. Should their calibrated forces differ too? What additional information turns eight voltage channels into a force, a centre of pressure and a free moment?

    ## Setup

    The example uses NumPy, Matplotlib, the bundled `kistler_fp_data.txt` recording and `kistler_fp_cal`. The first data column is time; the remaining eight columns follow the sensor convention in the retained figure. The example selects horizontal range 0 and vertical range 1 from the supplied historical calibration. Check those choices against the actual acquisition whenever you use new data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to get proper signals of a Kistler force plate, we have to calibrate the acquired data according to the factory calibration (i.e., transform the recorded values in volts of the 8 channels to values in newtons for fx12, fx34, fy14, fy23, fz1, fz2, fz3, fz4) and then calculate the resultant forces (Fx, Fy, Fz), resultant moments of force (Mx, My, Mz), and center of pressure (COPx, COPy) quantities (see figure below for the convention used).

    <div class='center-align'><figure aria-describedby="figure-caption-1"><img src="https://github.com/BMClab/BMC/blob/master/images/KistlerForcePlate.png?raw=1"alt="Kistler force plate and its coordinate system convention"/></figure></div>

    <span id="figure-caption-1" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Kistler force plate and its coordinate system convention (from [Kistler Force Plate Formulae](https://isbweb.org/software/movanal/vaughan/kistler.pdf)).</span>

    The function `kistler_fp_cal.py` performs such calculations for the Kistler force plates of the BMClab, and its defaults belong to one specific plate/amplifier setup. Another plate requires its own channel order, sensitivity table, amplifier ranges, geometry and applicable COP correction; these values are not interchangeable. The function signature is:<br>
    ```python
    grf, cop = kistler_fp_cal(data, S_matrix=None, fxfy_range=0, fz_range=0, origin=None,
                              fp_type=3, show=False, axs=None)
    ```
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — sensitivity.** A channel has sensitivity 2 mV/N and records 0.020 V. What force does it represent? What goes wrong if volts are treated as millivolts?

    /// details | Worked solution — Exercise 1 — sensitivity

    Convert first: $0.020\,\mathrm{V}=20\,\mathrm{mV}$, so $F=20/2=10$ N. Omitting the conversion gives a result 1000 times too small. The helper applies `1000 * volts / sensitivity`.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Follow the mechanics

    The four vertical channels add to $F_z$. Differences between those channels generate $M_x$ and $M_y$; horizontal forces also contribute when the measurement origin is below the surface. At the surface, the uncorrected COP is $x=-M_y/F_z$, $y=M_x/F_z$.

    The returned `grf` columns are **`Fx, Fy, Fz, Mx, My, Tz`**: the last column is the **free moment at the COP**, $T_z=M_z-F_y x+F_x y$. The historical code calls this column `Mz`, but it is not the original moment about the plate origin. Forces are in N, moments in N m, and COP in m. With negligible $F_z$, COP and free moment inferred through COP are not reliable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The legacy implementation's COP polynomial evaluation has been corrected: the manufacturer's coefficients use **millimetres**, and both corrections use the original measured $(x,y)$. The public result is converted back to metres. Applying those coefficients directly to metres, or feeding corrected $x$ into the $y$ correction, changes the result. The correction is model- and mounting-specific; see section 8.4.3 of the [Kistler manual](https://assets.kistler.com.cn/SAPCommerce_Download_original/002-462e.pdf).
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
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    return (np,)


@app.cell
def _(np):
    # load data file
    data = np.loadtxt(repo_file('data/kistler_fp_data.txt'), delimiter='\t', skiprows=1)
    data.shape
    return (data,)


@app.cell
def _(np):
    """Kistler force plate calibration."""


    __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version__ = 'kistler_fp_cal.py v.1 2014/12/12'


    def kistler_fp_cal(data, S_matrix=None, fxfy_range=0, fz_range=0, origin=None,
                       fp_type=3, show=False, axs=None):
        """Kistler force plate calibration

        In order to get proper signals of a Kistler force plate, we have to
        calibrate the acquired data according to the factory calibration (i.e.,
        transform the readed values in volts of the 8 channels to values in newtons
        for fx12, fx34, fy14, fy23, fz1, fz2, fz3, fz4) and then calculate the
        resultant forces (Fx, Fy, Fz), resultant moments of force (Mx, My, Mz),
        and center of pressure (COPx, COPy) quantities. See [1]_ and [2]_

        Parameters
        ----------
        data  : array_like [fx12, fx34, fy14, fy23, fz1, fz2, fz3, fz4]
            Kistler force plate data (8 columns, in Volts)
        S_matrix  : array_like
            sensitivity matrix for Kistler force plate (8 columns) and at least
            one row
        fxfy_range  : number [0, 1, 2, 3], optional
            Fx/Fy amplifier range used in the data acquisition
            Fx/Fy [kN]: 0.25, 1.25, 2.5, 5
            This is only used if the sensitivity matrix has more than one row
            and the first four columns of the corresponding row are selected.
        fz_range  : number [0, 1, 2, 3], optional
            Fz amplifier range used in the data acquisition
            Fz [kN]: 1, 5, 10, 20
            This is only used if the sensitivity matrix has more than one row
            and the last four columns of the corresponding row are selected.
        origin  : array_like [a, b, az0]
            coordinates of the force plate origin (az0 is negative, in meters)
        fp_type  : number [0, 1, 2, 3, 4]
            type of force plate to be used in the COP correction
            0: no COP correction
            1: type 9281E11
            2: type 9281E and 9281EA (SN: <= 616901)
            3: type 9281E and 9281EA (SN: >= 616902)
            4: type 9287C, 9287CA
        show  : bool, optional (default = False)
            if True (1), plot data in matplotlib figure.
        axs  : a matplotlib.axes.Axes instance, optional (default = None).

        Returns
        -------
        grf    : numpy array [Fx, Fy, Fz, Mx, My, Tz]
                Forces in N, surface-origin Mx/My and free moment Tz in Nm.
        cop    : numpy array [COPx, COPy]
                center of pressure data

        References
        ----------
        .. [1] http://isbweb.org/software/movanal/vaughan/kistler.pdf
        .. [2] kistler_force_plate_calculation.py

        """

        if S_matrix is None:
            # sensitivity matrix for Kistler force plate 9281EA Serial no. 4402018
            # calibrated range
            # Fx/Fy [kN]: 0.25, 1.25, 2.5, 5
            # Fz [kN]:    1, 5, 10, 20
            S_matrix = np.array([[38.117, 37.723, 38.062, 38.008, 19.492, 19.445, 19.426, 19.457],
                                 [ 7.623,  7.545,  7.212,  7.602,  3.898,  3.889,  3.885,  3.891],
                                 [ 3.803,  3.761,  3.800,  3.796,  1.951,  1.945,  1.944,  1.948],
                                 [ 1.901,  1.881,  1.900,  1.898,  0.976,  0.973,  0.972,  0.974]])
        S_matrix = np.atleast_2d(S_matrix)
        if S_matrix.shape[0] == 1:
            S_matrix = S_matrix[0]
        else:
            S_matrix = np.hstack((S_matrix[fxfy_range, 0:4], S_matrix[fz_range, 4:8]))

        if origin is None:
            # origin for Kistler force plate 9281EA Serial no. 4402018
            origin = np.array([0.120, 0.200, -0.048])
        a, b, az0 = np.hsplit(np.asarray(origin), 3)

        # COP correction coefficients
        if fp_type == 1:    # Type 9281E11
            Px = [2.51997E-15, -2.18826E-10, -2.69254E-07, -4.85912E-11, 4.55731E-6, -4.18892E-2]
            Py = [2.83750E-15, -1.00051E-10, -2.05349E-06, -1.16374E-10, 4.71553E-6, 6.89265E-2]
        elif fp_type == 2:  # Type 9281E and 9281EA (SN: <= 616901)
            Px = [1.1604E-14, -8.39091E-10, -1.44293E-6, -2.85927E-10, 2.05575E-5, -0.113525]
            Py = [1.27251E-14, -3.13238E-10, -3.33888E-6, -6.49641E-10, 1.53937E-5, 1.12624E-1]
        elif fp_type == 3:  # Type 9281E and 9281EA (SN: >= 616902)
            Px = [7.92063E-15, -5.821E-10, -2.77102E-6, -1.76083E-10, 1.29695E-5, -0.0413979]
            Py = [8.82869E-15, -2.48554E-10, -1.76282E-6, -4.22186E-10, 1.2091E-5, 5.16279E-2]
        elif fp_type == 4:  # Type 9287C, 9287CA
            Px = [1.72454E-16, -4.82275E-11, 3.30016E-7, -9.46569E-12, 2.78736E-6, -8.20399E-3]
            Py = [2.20428E-16, -1.80864E-11, -7.30249E-7, -3.03080E-11, 2.64974E-6, 5.41166E-2]
        else:
            Px = []
            Py = []

        # Calibration
        data = 1000*data/S_matrix

        fx12, fx34, fy14, fy23, fz1, fz2, fz3, fz4 = np.hsplit(data, 8)

        # Medio-lateral force
        Fx = fx12 + fx34
        # Anterior-posterior force
        Fy = fy14 + fy23
        # Vertical force
        Fz = fz1 + fz2 + fz3 + fz4
        # Plate moment about X-axis
        Mx = b * (fz1 + fz2 - fz3 - fz4)
        # Plate moment about Y-axis
        My = a * (-fz1 + fz2 + fz3 - fz4)
        # Plate moment about Z-axis
        Mz = b * (-fx12 + fx34) + a * (fy14 - fy23)
        # Plate moment about top plate surface
        Mx = Mx + Fy*az0
        # Plate moment about top plate surface
        My = My - Fx*az0
        # X-Coordinate of force application point (COP)
        ax = -My / Fz
        # Y-Coordinate of force application point (COP)
        ay = Mx / Fz
        # Coefficient of Friction x-component
        #COFx = Fx / Fz
        # Coefficient of Friction y-component
        #COFy = Fy / Fz
        # The manufacturer's polynomial table uses measured COP in millimetres.
        # Evaluate both corrections from the original coordinates, then return metres.
        # Kistler 9287C_002-462e-03.11, section 8.4.3.
        axm, aym = 1000 * ax, 1000 * ay
        if Px:
            Dax = (Px[0]*aym**4 + Px[1]*aym**2 + Px[2])*axm**3 + (Px[3]*aym**4 + Px[4]*aym**2 + Px[5])*axm
            ax = ax - Dax / 1000
        if Py:
            Day = (Py[0]*axm**4 + Py[1]*axm**2 + Py[2])*aym**3 + (Py[3]*axm**4 + Py[4]*axm**2 + Py[5])*aym
            ay = ay - Day / 1000
        # Free moment
        Mz = Mz - Fy*ax + Fx*ay

        grf = np.hstack((Fx, Fy, Fz, Mx, My, Mz))
        cop = np.hstack((ax, ay))

        if show:
            plot_kistler(grf, cop, axs)

        return grf, cop


    def plot_kistler(grf, cop, axs):
        """Plot results of the detect_peaks function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            grf = np.hstack((grf, cop))
            if axs is None:
                _, axs = plt.subplots(4, 2, figsize=(8, 6), sharex=True)

            axs = axs.flatten()
            ylabel = ['Fx [N]', 'Fy [N]', 'Fz [N]',
                      'Mx [Nm]', 'My [Nm]', 'Mz [Nm]', 'COPx [m]', 'COPy [m]']
            for i, axi in enumerate(axs):
                axi.plot(grf[:, i], 'b', lw=1)
                axi.set_ylabel(ylabel[i], fontsize=14)
                axi.yaxis.set_major_locator(plt.MaxNLocator(4))
                axi.yaxis.set_label_coords(-.2, 0.5)

            axs[6].set_xlabel('Sample #', fontsize=14)
            axs[7].set_xlabel('Sample #', fontsize=14)
            plt.suptitle('Ground reaction force data', y=1.02, fontsize=16)
            plt.tight_layout(h_pad=.1)
            # plt.grid()
            plt.show()
    return kistler_fp_cal, plot_kistler


@app.cell
def _(data, kistler_fp_cal):
    grf, cop = kistler_fp_cal(data[:, 1:], fxfy_range=0, fz_range=1, show=True)
    return grf, cop








@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — resultant and pressure point.** With four equal vertical sensor forces of 100 N, zero horizontal forces and a symmetric plate, predict $F_z$ and COP. Then use surface moments $M_x=10$ N m, $M_y=-20$ N m with $F_z=500$ N to find the uncorrected COP.

    /// details | Worked solution — Exercise 2 — resultant and pressure point

    The equal forces give $F_z=400$ N and zero moment differences, so COP is at the origin. In the second case, $x=-(-20)/500=0.040$ m and $y=10/500=0.020$ m. A plate-specific polynomial correction is a further calibration step, separate from these balance equations.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function kistler_fp_cal.py
    """)
    return


@app.cell
def _(kistler_fp_cal, mo):
    import inspect
    mo.ui.code_editor(inspect.getsource(kistler_fp_cal), language="python", disabled=True)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — low-force sensitivity.** A 1 N m uncertainty affects $M_y$. Compare its contribution to COP uncertainty at $F_z=500$ N and 5 N, holding $F_z$ fixed.

    /// details | Worked solution — Exercise 3

    From $x=-M_y/F_z$, the uncertainty magnitude is $1/|F_z|$: 0.002 m versus 0.2 m. The same moment error causes a hundredfold larger COP error near unloading. Choose a contact threshold from acquisition noise and analysis needs; a universal threshold is not established by this demonstration.

    ///

    ## Checkpoint questions

    - Which quantities must match the plate serial number and acquisition settings?
    - Why does the surface-origin offset affect moments when shear force is present?
    - How could a plausible COP plot conceal a sign or unit error?

    /// details | Checkpoint answers

    Use the actual sensitivities, amplifier ranges, geometry, channel convention and applicable correction. Moving the origin changes moment by a position cross force term. Validate signs with known loading locations, check metres versus millimetres, and compare a known static load; smoothness alone is insufficient.

    ///

    ## Problems

    1. Four vertical channels each record 0.050 V with sensitivity 2 mV/N. Horizontal channels are zero. Find the total vertical force and uncorrected COP on a symmetric plate.
    2. At the plate surface, $F=(20,30,600)$ N, $M_x=12$ N m, $M_y=-24$ N m and origin moment $M_z=3$ N m. Find the uncorrected COP and free moment.
    3. If $F_z=0$, should the COP be reported as the origin, infinity, or undefined?
    4. A sensitivity of 4 mV/N is mistakenly replaced by 2 mV/N. What happens to the inferred force from that channel?

    /// details | Final answers to the problems

    1. 100 N; COP $(0,0)$ m.
    2. COP $(0.040,0.020)$ m; $T_z=2.2$ N m.
    3. Undefined.
    4. The inferred force doubles.

    ///

    ## Go deeper

    Use a known load at several positions to check force balance, COP signs and free moment. Review the sensitivity table for your serial number and distinguish factory calibration from an in-situ re-calibration. The companion [force-plate calibration guide](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/force_plate_calibration.py) develops the latter.

    ### Video lectures on the Internet

    - [MIT 8.01SC Classical Mechanics](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/) — review torque and equilibrium, then explain why the COP moment equations divide by vertical force.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Kistler Force Plate Formulae](https://isbweb.org/software/movanal/vaughan/kistler.pdf)

    - [Qualisys Kistler force-plate settings](https://docs.qualisys.com/qtm/content/project_options/force_plate_settings_kistler.htm) — primary integration documentation on calibration, ranges and geometry.
    """)
    return


if __name__ == "__main__":
    app.run()
