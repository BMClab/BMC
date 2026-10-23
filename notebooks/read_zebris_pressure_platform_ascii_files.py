# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read Zebris pressure platform ASCII files

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this notebook

    Read the APD header before looking at the foot image. Then compare the raw grid, display interpolation, spline surface and resized steps. Each representation answers a different question. Predict a numerical quantity or a visual change before expanding a worked solution.

    ## Opening challenge

    A colourful footprint looks like a pressure snapshot. Is it? If each sensor stores its maximum over an entire step, what happens when you sum those values and call the result peak force?

    ## Setup

    NumPy reads the ASCII arrays; Matplotlib draws them; SciPy demonstrates spline fitting and resizing. The retained `MDwalk2` APD files are bundled examples. Their header distances are in mm and pressures in N/cm². The figures convert distances to cm. No platform connection is needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Zebris Medical GmbH (https://www.zebris.de/en/) builds devices for measuring pressure on the foot/platform interface.<br>
    The BMClab (https://bmclab.pesquisa.ufabc.edu.br/) originally recorded the following setup notes for its two Zebris FDM 1.5 platforms. These are historical laboratory notes, not specifications for every model or firmware:
    - Measuring principle: capacitive force measurement
    - Dimensions: 158 x 60.5 x 2.1 cm (L x W x H)
    - Sensor area: 149 x 54.2 cm (L x W)
    - Number of sensors: 11264
    - Physical resolution: 1.4 sensors /cm2  (0.714 cm2)
    - Sampling frequency: 100 Hz
    - Measuring Range: 1 - 120 N/cm2
    - Accuracy of the calibrated measuring range: (1 – 80 N/cm2), ±5% (FS)
    - Hysteresis: < 3 % (FS)

    The [manufacturer's 2015 hardware manual](https://www.zebris.de/fileadmin/Editoren/zebris-PDF-Manuals/Medizin/Hardware/Alte_Versionen/FDM_Hardware-Manual_151203_en.pdf) lists a 1440 × 560 mm sensor area and 120 Hz standard sampling for its FDM 1.5 version, which differs from some historical notes above. Use the manual for your model and the acquisition metadata. The sample APD files provide sensor spacing but no pressure time series.

    The two pressure platforms can be synchronized and used as a single 3-m platform.

    The proprietary software to operate the pressure device saves files in ASCII and binary formats with the pressure data. Here are functions for reading most of the files saved in ASCII format. These files have headers with metadata about the patient and acquisition conditions and the data of pressure, force, or center of pressure  depending on the type of acquisition and chosen option to export the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## APD file

    The .apd file in ASCII contains the metadata and the maximum values of pressure over the trial in the exported foot region. The original notes stated a nominal 1 N/cm² threshold, but the supplied file also contains 0.5 N/cm² entries; do not impose that historical threshold as a property of every export. This file can be used for making insoles.<br>
    Here is a typical .apd file:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    return np, plt


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
    filename = str(repo_file('data/MDwalk2.apd'))
    with open(file=filename, mode='rt', newline='') as f:
        print(f.read())
    return (filename,)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — inspect the header.** The first file has `SensCountX=32`, `SensCountY=15` and spacing 8.46591 mm in each direction. What shape and footprint bounds should the raw array have?

    /// details | Worked solution — Exercise 1 — inspect the header

    The array has 32 rows and 15 columns. Spacing is 0.846591 cm, so the rectangular bounds are 27.090912 cm in X and 12.698865 cm in Y. The rectangle includes unoccupied sensors; its area is not necessarily contact area. The reader maps the export sentinel −1 to zero for these displays, so distinguish unreported cells from measured zero pressure in later analyses.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is a function for reading an .apd file from Zebris pressure platform:
    """)
    return


@app.cell
def _(np):
    def read_zebris_apd(filename):
        """Reads Zebris pressure platform ASCII files .apd.
        """
        import pprint
        sections = ['General', 'Customer', 'Technical', 'Data']
        _s = 0
        info = {}
        with open(file=filename, mode='rt', newline='') as f:
            for linea in f:
                line = linea.strip('\r[]\n')
                if line == sections[_s]:
                    info[sections[_s]] = {}
                    _s += 1
                elif line:
                    info[sections[_s - 1]][line.split('=')[0]] = line.split('=')[1]
                elif _s == 3:
                    break
            f.readline()
            data = np.loadtxt(f, delimiter='\t')
            data[data == -1] = 0
            print('File %s successfully open.' % filename)
            print('Data has %d rows and %d columns.' % data.shape)
        return (info, data)

    return (read_zebris_apd,)


@app.cell
def _(filename, read_zebris_apd):
    info, data = read_zebris_apd(filename)
    return data, info


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simple 2D plot of the raw data
    """)
    return


@app.cell
def _(data, info, np, plt):
    from matplotlib import cm
    dx = float(info['Technical']['LDistX']) / 10
    dy = float(info['Technical']['LDistY']) / 10
    x = np.arange(1 / 2, data.shape[0] + 1 / 2, 1) * dx
    y = np.arange(1 / 2, data.shape[1] + 1 / 2, 1) * dy
    (X, Y) = np.meshgrid(y, x)
    print('Shapes:')
    print('X:', X.shape, 'Y:', Y.shape, 'data:', data.shape)
    (_fig, _ax) = plt.subplots(figsize=(6, 7))
    _img = _ax.pcolormesh(X, Y, data, cmap=cm.jet)
    _ax.set_aspect('equal')
    _fig.colorbar(_img, label='Pressure (N/cm$^2$)')
    _ax.set_xlabel('Y (cm)')
    _ax.set_ylabel('X (cm)')
    _ax.set_title('Plantar pressure during walking')
    plt.show()
    return X, Y, cm, dx, dy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2D display interpolation

    Let's use the matplotlib function [`imshow`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html) to compare nearest-neighbour and bilinear display interpolation. This changes how the same array is drawn, not the stored pressure measurements.<br>
    The `extent` argument maps the image bounds to physical centimetres; without it, labelled axes would still count array indices.
    """)
    return


@app.cell
def _(cm, data, dx, dy, plt):
    (_fig, _ax) = plt.subplots(1, 2, figsize=(12, 7))
    _extent = (0, data.shape[1]*dy, 0, data.shape[0]*dx)
    _img0 = _ax[0].imshow(data, extent=_extent, vmin=0, vmax=40, cmap=cm.jet, aspect='equal', origin='lower', interpolation='nearest')
    _ax[0].set_xlabel('Y (cm)')
    _ax[0].set_ylabel('X (cm)')
    _img1 = _ax[1].imshow(data, extent=_extent, cmap=cm.jet, aspect='equal', origin='lower', interpolation='bilinear', vmin=0, vmax=40)
    _ax[1].set_xlabel('Y (cm)')
    _ax[1].set_ylabel('X (cm)')
    _fig.colorbar(_img1, ax=list(_ax), label='Pressure (N/cm$^2$)')
    _fig.suptitle('Plantar pressure during walking', fontsize=16)
    plt.show()
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — peak map versus force.** Two 1 cm² sensors have pressures (10,0) N/cm² at one instant and (0,10) at another. What is the peak actual force, and what would summing an APD maximum map give?

    /// details | Worked solution — Exercise 2 — peak map versus force

    Both instantaneous forces are 10 N. The per-sensor maximum map is (10,10), whose area-weighted sum is 20 N. Maxima at different times cannot be added as though they were simultaneous. APD data alone also cannot recover impulse, contact duration or cadence.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3D plots
    """)
    return


@app.cell
def _(X, Y, cm, data, plt):
    from mpl_toolkits.mplot3d import Axes3D
    _fig = plt.figure(figsize=(14, 6))
    _ax = _fig.add_subplot(projection='3d')
    _surf = _ax.plot_surface(X, Y, data, cmap=cm.jet, rcount=data.shape[0], ccount=data.shape[1], linewidth=0, antialiased=True)
    _ax.view_init(60, 200)
    _fig.colorbar(_surf, orientation='vertical', label='Pressure (N/cm$^2$)')
    _ax.set_xlabel('Y (cm)')
    _ax.set_ylabel('X (cm)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3D spline surface

    This fits a smoother surface on a denser grid. It does not add measured spatial resolution. Inspect overshoot, including negative values, before interpreting an interpolated peak.
    """)
    return


@app.cell
def _(X, Y, data, dx, dy, np):
    from scipy import interpolate
    # interpolate data over a four times denser grid
    x2 = np.arange(1/8, data.shape[0] + 1/8, 1/4)*dx
    y2 = np.arange(1/8, data.shape[1] + 1/8, 1/4)*dy
    X2, Y2 = np.meshgrid(y2, x2)
    tck = interpolate.bisplrep(X, Y, data)
    data2 = interpolate.bisplev(X2[0,:], Y2[:,0], tck).T
    print('Shapes:')
    print('X2:', X2.shape, 'Y2:', Y2.shape, 'data2:', data2.shape)
    return X2, Y2, data2


@app.cell
def _(X2, Y2, cm, data2, plt):
    _fig = plt.figure(figsize=(14, 6))
    _ax = _fig.add_subplot(projection='3d')
    _surf = _ax.plot_surface(X2, Y2, data2, cmap=cm.jet, rcount=data2.shape[0], ccount=data2.shape[1], linewidth=0, antialiased=False)
    _ax.view_init(60, 200)
    _fig.colorbar(_surf, orientation='vertical', label='Pressure (N/cm$^2$)')
    _ax.set_xlabel('Y (cm)')
    _ax.set_ylabel('X (cm)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loading several steps of a trial
    """)
    return


@app.cell
def _(cm):
    def plot_apd(info, data, ax, title):
        dx = float(info['Technical']['LDistX']) / 10
        dy = float(info['Technical']['LDistY']) / 10
        x0 = int(info['Technical']['StartSensY']) * dy
        y0 = int(info['Technical']['StartSensX']) * dx
        xlen = int(info['Technical']['SensCountY'])
        ylen = int(info['Technical']['SensCountX'])
        _img = ax.imshow(data, cmap=cm.jet, aspect='auto', origin='lower', extent=[x0, x0 + xlen * dy, y0, y0 + ylen * dx], interpolation='nearest', vmin=0, vmax=40)
        ax.set_title(title)
        ax.set_xlabel('Y (cm)')
        return _img

    return (plot_apd,)


@app.cell
def _(plot_apd, plt, read_zebris_apd):
    steps = ['MDwalk2' + _step for _step in ['', '_1', '_2', '_3', '_4']]
    (infos, datas) = ({}, {})
    (_fig, _axs) = plt.subplots(1, len(steps), figsize=(14, 7))
    for (_s, _step) in enumerate(steps):
        (infos[_step], datas[_step]) = read_zebris_apd(str(repo_file('data/' + _step + '.apd')))
        _img = plot_apd(infos[_step], datas[_step], _axs[_s], _step)
    _axs[0].set_ylabel('X (cm)')
    _fig.colorbar(_img, ax=list(_axs), label='Pressure (N/cm$^2$)', orientation='horizontal', pad=0.1, aspect=40)
    plt.show()
    return datas, infos, steps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It seems the Zebris software when exporting the pressure data to .apd files of a trial with a sequence of steps, saves first the data of the right foot and then of the left foot.

    The origin of the pressure platform was at the far right corner in relation to the walking direction of the subject in this trial (see the values at the x and y axes). So, the order of the subject's foot steps were: MDwalk2_4, MDwalk2_1, MDwalk2_3, MDwalk2, and MDwalk2_2.

    The size of the pressure data across steps is not constant, the width varies from 12 to 15 columns and the length varies from 30 to 32 rows of data. Multiply these numbers by 0.846591 cm to have the size of the pressure data in centimeters. Possible reasons for this variation are: 1. foot rotation, 2. differences at how the foot is supported at the ground at each step, and 3. perhaps measurement noise. So, one can't directly compare the images (the pressures), for example, we can't average the data in order to get the mean foot pressure (or other statistics related to the positions of the steps); we will have first to align the data (each step) and account for the different sizes. In image processing, this procedure is part of what is known as [image registration](https://en.wikipedia.org/wiki/Image_registration). For the application of image registration to foot plantar pressure, see Pataky et al. (2008), Oliveira et al. (2010).

    For now, given that there are only few steps and they seemed to be similar, we will only transform the images to have the same size.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — registration.** Two foot images have the same dimensions after resizing, but one toe is shifted five pixels. Can the pixels now be averaged anatomically? What does the maximum-preserving rescaling in the next cell guarantee?

    /// details | Worked solution — Exercise 3 — registration

    Equal dimensions only make array arithmetic possible; corresponding pixels need not represent corresponding anatomy. Registration must establish that spatial correspondence. Multiplying the resized image by its original maximum divided by its new maximum restores that peak value only; it does not conserve pressure integral, force, area or anatomy.

    ///
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Image resize
    """)
    return


@app.cell
def _(datas, infos, plot_apd, plt, steps):
    from scipy.ndimage import zoom
    datas2 = {}
    (_fig, _axs) = plt.subplots(1, len(steps), figsize=(14, 7))
    for (_s, _step) in enumerate(steps):
        maxdata = datas[_step].max()
        datas2[_step] = zoom(datas[_step], (120 / datas[_step].shape[0], 60 / datas[_step].shape[1]), order=1)  # bilinear
        datas2[_step] = maxdata * (datas2[_step] / datas2[_step].max())
        print('%s has %d rows and %d columns.' % (_step, *datas2[_step].shape))
        _img = plot_apd(infos[_step], datas2[_step], _axs[_s], _step)
        _axs[_s].set_aspect('equal')
    _axs[0].set_ylabel('X (cm)')
    _fig.colorbar(_img, ax=list(_axs), label='Pressure (N/cm$^2$)', orientation='horizontal', pad=0.1, aspect=40)
    plt.show()
    return (datas2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now calculate for instance the left and right steps with the maxium pressure of all corresponding steps:
    """)
    return


@app.cell
def _(cm, datas2, np, plt, steps):
    max_step_r = np.max(np.stack((datas2[steps[0]], datas2[steps[1]]), 2), 2)
    max_step_l = np.max(np.stack((datas2[steps[2]], datas2[steps[3]], datas2[steps[4]]), 2), 2)
    (_fig, _axs) = plt.subplots(1, 2, figsize=(12, 7))
    _img0 = _axs[0].imshow(max_step_l, cmap=cm.jet, aspect='equal', origin='lower', interpolation='nearest', vmin=0, vmax=40)
    _axs[0].set_xlabel('Y')
    _axs[0].set_ylabel('X')
    _img1 = _axs[1].imshow(max_step_r, cmap=cm.jet, aspect='equal', origin='lower', interpolation='nearest', vmin=0, vmax=40)
    _axs[1].set_xlabel('Y')
    _fig.colorbar(_img1, ax=list(_axs), label='Pressure (N/cm$^2$)')
    _fig.suptitle('Plantar pressure during walking (maximum values across steps)', fontsize=16)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It doesn't work!<br>
    We need to perform image registration...
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    - Which coordinates correspond to array rows and columns in these figures?
    - Does bilinear display interpolation increase sensor resolution?
    - Why is the final left/right maximum image presented as an unsuccessful anatomical comparison?

    /// details | Checkpoint answers

    Rows follow the file's X direction and columns its Y direction; plotted horizontal coordinates are Y. Display interpolation does not create measurements. The final example preserves the original warning: resized steps can still have different orientation and anatomical alignment, so their elementwise maxima need not refer to matching foot locations.

    ///

    ## Problems

    1. Convert a pressure of 20 N/cm² over one square sensor of side 0.846591 cm into force, assuming that pressure applies at one instant across the sensor.
    2. For the first supplied APD file, state the array shape and maximum recorded pressure. Can you determine when that maximum occurred?
    3. A 3×2 array has X spacing 2 mm and Y spacing 3 mm. With zero local origin, give the `imshow` extent in cm using the notebook's row/column convention.
    4. Which operation is sufficient to make anatomical pixelwise comparison valid: changing a colour map, resizing to equal array dimensions, or establishing anatomical registration?

    /// details | Final answers to the problems

    1. Approximately 14.3343 N.
    2. `(32, 15)`; 36.5 N/cm²; its time is unavailable from this APD file.
    3. `(0, 0.6, 0, 0.6)`.
    4. Anatomical registration, with its alignment accuracy assessed.

    ///

    ## Go deeper

    Read Pataky et al. and Oliveira et al. below for registration approaches. Compare a raw pressure time series with its per-sensor maximum image and show which temporal information is lost. Keep original physical spacing separate from any resized display grid.

    ### Video lectures on the Internet

    - [MIT 8.01SC Classical Mechanics](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/) — the force and impulse lessons provide the mechanical background for distinguishing instantaneous force from a map of maxima.
    """)
    return





@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - [Pataky TC, Goulermas JY, Crompton RH. (2008) A comparison of seven methods of within-subjects rigid-body pedobarographic image registration. J Biomech., 20;41(14):3085-9. doi: 10.1016/j.jbiomech.2008.08.001](https://pubmed.ncbi.nlm.nih.gov/18790481/).

    - [Oliveira FP, Pataky TC, Tavares JM (2010) Registration of pedobarographic image data in the frequency domain. Comput Methods Biomech Biomed Engin., 13(6):731-40. doi: 10.1080/10255840903573020](https://pubmed.ncbi.nlm.nih.gov/20526916/). [DOI](https://doi.org/10.1080/10255840903573020)

    - [Zebris FDM hardware manual, 2015](https://www.zebris.de/fileadmin/Editoren/zebris-PDF-Manuals/Medizin/Hardware/Alte_Versionen/FDM_Hardware-Manual_151203_en.pdf) — historical model specifications; consult the [manufacturer's manuals index](https://www.zebris.de/service/downloads) for your hardware.

    - [Matplotlib `imshow`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html) — coordinate extent and interpolation semantics.
    """)
    return


if __name__ == "__main__":
    app.run()
