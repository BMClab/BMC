# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "scipy>=1.14", "psd2>=0.0.5", "urllib3>=2.2"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Measurements in stabilography

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    We will inspect a public force-platform recording, remove its mean, and compute variability, path length, speed, area and spectral measures. Distinguish COP from body COM, and keep duration, sampling and filtering consistent before comparing trials.

    Try the distributed **Exercises** before opening their collapsed worked solutions. Then use the **Checkpoint questions** and attempt the **Problems** independently; their panels contain final answers only. Predict the effect of a parameter change before running it, and check units and limiting cases.

    **Opening challenge.** Two COP recordings have the same standard deviation. Can one travel a much longer total path, and would that alone prove worse balance?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The dependency declaration at the top of this file lists the packages used by this guide. Marimo runs cells in dependency order. Run the notebook in an editor to use its displays and controls; the computational examples below explain their inputs before using them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Posturography is a general term for all techniques concerned with quantifying postural sway of a standing person.

    Typically in posturography, instead of measuring the actual sway of each segment, measurements of the whole body sway have been used. The displacements of center of pressure (COP) and of the vertical projection (on the horizontal plane) of the center of gravity (COGv) are the most common measures of body sway (however, keep in mind that the COP displacement is not an actual measurement of postural sway of the body or of any of its segments). While the COP displacement can be easily measured with a force plate, the direct measurement of the COGv is more complicated and typically subject to a larger error. The measurement of the COGv is computed by measuring the position of each body segment combined with the mass of each segment. More commonly, the COGv displacement is determined indirectly from the COP displacement and different methods are available for such, which produce similar results (Lafond et al. 2004).  See the notebook [The inverted pendulum model of the human standing posture](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/ip_model.py) for a description and code of one of these methods.

    It is possible to quantify a certain property of a phenomenon, like the postural control during upright standing, without any underlying scientific theory on what generated that property and what the implication of that measurement for the understanding of the phenomenon is. Sometimes, this is the only approach to start with. However, as our knowledge on the phenomenon advances, the need for a scientific theory or a hypothesis to interpret the results in a more meaningful way becomes evident. It is relatively easy to perform many measurements of the body sway during upright standing; far more difficult is to interpret those measurements to understand what they mean.

    The most widespread empirical interpretation of typical measurements of posture sway is that more sway means more instability which is seen as an indication of a deteriorated postural control system. This rationale is based on many experiments on aging and pathological conditions that, indeed, have observed increased sway in those conditions. However, bear in mind that it might not always be the case.

    Selecting which measurement will be used to quantify a certain characteristic of the postural control depends on which experiment we design and which task the individual under evaluation is performing. In this sense, posturography has been divided into two experimental paradigms: dynamic and static posturography. In static posturography, the individual is typically trying to stay as quiet as possible while his or her sway is measured, and no perturbation is applied during the task. In dynamic posturography, a momentary perturbation is applied and how the subject responded to that perturbation is measured.

    Let's show some of the most common measures of postural sway, more typically employed in static posturography.

    The data are a trial of the public data set of human balance evaluations of Santos and Duarte (2016): the center of pressure of a 27-year-old woman standing still for 60 s with her eyes open on a rigid surface (trial BDS00013), measured with a force platform at 100 Hz. The full data set is available at [PhysioNet](https://physionet.org/content/hbedb/1.0.0/) and at [GitHub](https://github.com/demotu/datasets/tree/master/BDS). The COP in the anterior-posterior direction (ap) is positive to the front, and in the medio-lateral direction (ml), positive to the right.
    """)
    return


@app.cell
def _():
    import numpy as np
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(np):
    import pandas as pd  # use Pandas to read data from a website
    fileUrl = repo_file('BDS/data/BDS00013.txt', base_url="https://raw.githubusercontent.com/demotu/datasets/master/")
    data = pd.read_csv(fileUrl, sep='\t')  # Pandas dataframe
    COP = data[['COPx[cm]', 'COPy[cm]']].to_numpy()  # ap and ml directions, in cm
    freq = 1 / np.mean(np.diff(data['Time[s]']))  # sampling frequency, Hz
    print('COP shape: ', COP.shape)
    print(f'Sampling frequency: {freq:.0f} Hz, first-to-last duration: {(COP.shape[0] - 1) / freq:.2f} s')
    return COP, freq


@app.cell
def _(np, plt):
    def cop_plot(freq, COP, units='cm'):
        '''
        Plot COP data from postural sway measurement.
        '''
        import matplotlib.gridspec as gridspec
        t = np.arange(COP.shape[0]) / freq
        plt.rc('axes', labelsize=16,  titlesize=16)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
        ax1 = plt.subplot(gs[0])
        ax1.plot(t, COP[:,0], lw=2, color=[0, 0, 1, 1], label='ap')
        ax1.plot(t, COP[:,1], lw=2, color=[1, 0, 0, 1], label='ml')
        ax1.set_xlim([t[0], t[-1]])
        ax1.grid()
        ax1.locator_params(axis='both', nbins=5)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('COP [%s]' %units)
        ax1.set_title('Stabilogram')
        ax1.legend(fontsize=12, loc='best', framealpha=.5)
        ax2 = plt.subplot(gs[1])
        ax2.plot(COP[:,1], COP[:,0], lw=2, color='g')
        ax2.set_xlabel('COP ml [%s]' %units)
        ax2.set_ylabel('COP ap [%s]' %units)
        ax2.set_title('Statokinesigram')
        if 0:  # plot the axes with the same colors of the COP data
            ax2.xaxis.label.set_color('red')
            ax2.spines['bottom'].set_color('red')
            ax2.tick_params(axis='x', colors='red')
            ax2.yaxis.label.set_color('blue')
            ax2.spines['left'].set_color('blue')
            ax2.tick_params(axis='y', colors='blue')
        ax2.grid()
        ax2.locator_params(axis='both', nbins=5)
        plt.tight_layout()
        plt.show()

    return (cop_plot,)


@app.cell
def _(COP, cop_plot, freq):
    # plot data
    cop_plot(freq, COP)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data detrend

    The mean value of the COP (or COGv) displacement is dependent where the individual stood on the force plate or in the space and usually has no particular interest to the understanding of postural sway. So, a typical procedure in the analysis of postural sway is to remove the mean value of the data. Related to that, the presence of a trend (a slow fluctuation) in the signal might also affect some of the measurements. Someone might argue that the trend itself could give valuable information about the signal (see Duarte and Zatsiorsky, 2000), but the problem is that most of the measurements one uses to describe a signal assumes that the signal is stationary. A signal is stationary if its statistical properties (such as mean and variance) do not change across time. So, to detrend the data might be necessary in certain cases. Another way to remove a trend in the data is to apply a high pass filter to the data with a cut-off frequency related to the period of data acquisition (Witt et al., 1998).

    Removing the mean or a trend does not guarantee stationarity. Weak stationarity also requires finite variance and an autocovariance that depends on lag rather than absolute time. Inspect the record and state the preprocessing used before comparing sway measures.

    Let's see a simple function for data detrend in Python and use it to remove the mean of the data.
    """)
    return


@app.cell
def _(COP):
    from scipy.signal import detrend
    COP_1 = detrend(COP, axis=0, type='constant')
    return (COP_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — Remove an offset.** A COP series is $(10,11,9)$ cm. Find its demeaned values and range. Does subtracting the mean change its path length?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 1': mo.md(r"""
    The mean is 10 cm, so the demeaned values are $(0,1,-1)$ cm and range remains 2 cm. Path length is $|1-0|+|-1-1|=3$ cm before and after demeaning: constant translations do not affect differences.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Measurements of spatial variability

    Standard deviation, RMS, range (or amplitude), and total path (or total displacement) are commonly employed to describe the spatial variability of some measurement of postural sway. For the data represented by $x$, these variables are defined as:

    $$
    \bar{x} \; (mean) = \frac{1}{N}\sum_{i=1}^{N} x_i
    $$

    $$
    SD = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N} (x_i - \bar{x})^2}
    $$

    $$
    RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
    $$

    $$
    Range = max[x] - min[x]
    $$

    $$
    Total \: path = \sum_{i=1}^{N-1} | x_{i+1}-x_i |
    $$
    """)
    return


@app.cell
def _(COP_1, np):
    m = np.mean(COP_1, axis=0)
    sd = np.std(COP_1, axis=0, ddof=1)
    rms = np.sqrt(np.mean(COP_1 ** 2, axis=0))
    rang = np.max(COP_1, axis=0) - np.min(COP_1, axis=0)
    tpath = np.sum(np.abs(np.diff(COP_1, axis=0)), axis=0)
    unit = 'cm'
    print('Measurements of spatial variability')
    print('{0:12} {1:^16}'.format('Variable', 'Direction'))
    print('{0:12} {1:^8} {2:^5}'.format('', 'ap', 'ml'))
    print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('Mean:', m[0], m[1], unit))
    print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('SD:', sd[0], sd[1], unit))
    print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('RMS:', rms[0], rms[1], unit))
    print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('Range:', rang[0], rang[1], unit))
    print('{0:12} {1:>6.2f} {2:>6.2f} {3:>3}'.format('Total path:', tpath[0], tpath[1], unit))
    return (unit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — Distinguish RMS and sample SD.** For demeaned samples $(0,1,-1)$ cm, calculate RMS and sample SD.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 2': mo.md(r"""
    RMS uses denominator N: $\sqrt{2/3}=0.8165$ cm. Sample SD uses N−1: $\sqrt{2/2}=1$ cm. They are close for large N, but are not identical under these definitions.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean velocity (or mean speed)

    The mean speed variable expresses the the average velocity of the COP displacement computed simply as the total path variable (the total displacement) divided by the total period. This variable is usually referred as velocity but as it a scalar, it should in fact be named speed.<br>
    The mean resultant speed is the speed calculated in the vectorial form considering each direction (the square root of the sum of the squared speed in each direction). For the data represented by *x* and *y*, these variables are defined as:

    $$
    Mean \: speed = \frac{1}{T}\sum_{i=1}^{N-1} | x_{i+1}-x_i |
    $$

    $$
    Mean \: resultant \: speed = \frac{1}{T}\sum_{i=1}^{N-1} \sqrt{( x_{i+1}-x_i )^2 + ( y_{i+1}-y_i )^2}
    $$

    where $T=(N-1)/f_s$ is the observed first-to-last sample duration. The norm of the two directional mean speeds is generally smaller than the mean resultant speed: averaging and taking a vector norm do not commute when movement direction varies.
    """)
    return


@app.cell
def _(COP_1, freq, np, unit):
    T = (COP_1.shape[0] - 1) / freq  # span of the N-1 observed increments, s
    mvel = np.sum(np.abs(np.diff(COP_1, axis=0)), axis=0) / T
    mvelr = np.sum(np.abs(np.sqrt(np.sum(np.diff(COP_1, axis=0) ** 2, axis=1))), axis=0) / T
    print('{0:15} {1:^16}'.format('Variable', 'Direction'))
    print('{0:15} {1:^8} {2:^5}'.format('', 'ap', 'ml'))
    print('{0:15} {1:>6.2f} {2:>6.2f} {3:>5}'.format('Mean velocity:', mvel[0], mvel[1], unit + '/s'))
    print('')
    print('{0:22} {1:>6.2f} {2:>5}'.format('Mean resultant velocity:', mvelr, unit + '/s'))
    return (mvel,)


@app.cell
def _(mvel, np, unit):
    print('Euclidean norm of directional mean speeds (not mean resultant speed):')
    print('{0:>6.2f} {1:>5}'.format(np.sqrt(np.sum(mvel ** 2)), unit+'/s'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — Read speed from an out-and-back path.** At times 0, 1 and 2 s, COP positions are $(0,0),(1,0),(0,0)$ cm. Find mean resultant speed and mean velocity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 3': mo.md(r"""
    Total path is 2 cm over 2 s, so mean speed is 1 cm/s. Net displacement is zero, so mean velocity is $(0,0)$ cm/s. Repeated reversals can make path large without increasing the overall spread.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Area

    Another measurement of postural sway is to compute the area that encompasses the COPap versus COPml data using a curve in a plane (e.g., a circle or ellipse) or a polygon (e.g., a rectangle). A common method for such is to compute a prediction ellipse, which is found by fitting an ellipse to the data using concepts from the statistical procedure known as principal component analysis. Under the bivariate-normal model and independent-sampling assumptions used in its construction, a 95% prediction ellipse targets 95% coverage for a new observation. COP samples are temporally correlated, so the fitted ellipse is usually best treated as a descriptive spatial measure; nominal coverage is not guaranteed for this time series. For more details, see [Prediction ellipse and ellipsoid](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/prediction_ellipse_ellipsoid.py).

    The function `hyperellipsoid.py` (see [Prediction ellipse and ellipsoid](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/prediction_ellipse_ellipsoid.py)) calculates the ellipse area, some related parameters, and plots the results for a given multivariate random variable.
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
    sys.path.insert(0, str(repo_file('functions/hyperellipsoid.py').parent))  # add to pythonpath
    from hyperellipsoid import hyperellipsoid

    return (hyperellipsoid,)


@app.cell
def _(COP_1, hyperellipsoid):
    (area, axes, angles, center, R) = hyperellipsoid(COP_1[:, 1], COP_1[:, 0], units='cm')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4 — Interpret an ellipse.** Two ellipses have semiaxes (2,1) cm and (4,0.5) cm. Compare their areas and explain why area alone is incomplete.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 4': mo.md(r"""
    Both have area $\pi ab=2\pi$ cm². Their anisotropy differs: one is much more elongated. Area does not retain orientation, direction-specific variability or temporal ordering.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequency analysis

    Frequency analysis refers to estimate the frequency content of a signal. Let's use standard Fourier analysis and the related power spectral analysis to estimate some frequency characteristics of the COP displacement. The function `psd2` from the Python package [psd2](https://pypi.org/project/psd2/) (see how it works in the notebook [Fourier transform](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/fourier_transform.py)) estimates the power spectral density with Welch's method, using the `scipy.signal.welch` function, estimates some frequency characteristics, and plots the results.

    Let's use `psd2` to estimate the frequency characteristics of the COP data.
    """)
    return


@app.cell
def _(COP_1, freq):
    from psd2 import psd2
    (fp_ap, mf_ap, fmax_ap, Ptot_ap, _F, P_ap) = psd2(COP_1[:, 0], fs=freq, scales='linear', xlim=[0, 2], units='cm')
    return (psd2,)


@app.cell
def _(COP_1, freq, psd2):
    (fp_ml, mf_ml, fmax_ml, Ptot_ml, _F, P_ml) = psd2(COP_1[:, 1], fs=freq, xlim=[0, 2], units='cm')
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. What does demeaning preserve?
    2. Which duration matches N samples separated by 1/fs?
    3. Does a nominal 95% ellipse guarantee 95% future coverage for autocorrelated nonstationary COP?
    4. Why can high-frequency noise strongly increase path speed?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Checkpoint answers': mo.md(r"""
    1. Differences, path length, range and covariance. 2. First-to-last timestamp duration is (N−1)/fs. 3. No; the probability interpretation depends on distributional and sampling assumptions. 4. Many rapid sample-to-sample changes accumulate into path length.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. A trajectory moves through (0,0), (3,4), (6,0) cm at 0, 1 and 2 s. Give path length, mean resultant speed and mean velocity.

    2. An ellipse has semiaxes 1.5 and 0.8 cm. Give area.

    3. A 100 Hz record contains 6000 samples. Give its first-to-last duration.

    4. Two traces have equal SD but different sampling rates and filtering. Can their raw COP path speeds be directly ranked as balance quality?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Final answers to Problems': mo.md(r"""
    1. 10 cm; 5 cm/s; (3,0) cm/s.

    2. $1.2\pi\approx3.770$ cm².

    3. 59.99 s.

    4. No; acquisition and processing must be made comparable, and clinical or functional meaning requires the task context.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    Use [KineticsFundamentalConcepts](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinetics_fundamental_concepts.py) to connect this guide with the surrounding mechanics and modeling material. Return to the assumptions behind each equation when applying it to a new experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Video lectures on the Internet

    [MIT OpenCourseWare](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/video_galleries/video-lectures/) — Signals and Systems: the Fourier-series, filtering and sampling lectures explain the frequency analysis and processing choices used here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Duarte M, Zatsiorsky VM (2000) [On the fractal properties of natural human standing](https://pubmed.ncbi.nlm.nih.gov/10754215). Neuroscience Letters, 283, 173-176.

    - Lafond, D., M. Duarte, et al. (2004). [Comparison of three methods to estimate the center of mass during balance assessment](https://pubmed.ncbi.nlm.nih.gov/15275850). J. Biomech. 37(9): 1421-1426.

    - Santos DA, Duarte M (2016) [A public data set of human balance evaluations](https://doi.org/10.7717/peerj.2648). PeerJ, 4, e2648.

    - Witt A, Kurths J, Pikovsky A (1998) [Testing stationarity in time series](http://journals.aps.org/pre/abstract/10.1103/PhysRevE.58.1800). Physical Review E, 58, 1800-1810.
    """)
    return


if __name__ == "__main__":
    app.run()
