# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "numpy>=2.0", "matplotlib>=3.9", "pandas>=2.2", "urllib3>=2.2", "ipython"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Body segment parameters

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    We will read the Dempster and de Leva tables, locate segment centers of mass, and combine segment inertias. You need weighted averages and distances between coordinates. Treat the table as an estimate for a population, and keep its segment landmarks and reference axes attached to every number.

    Try the distributed **Exercises** before opening their collapsed worked solutions. Then use the **Checkpoint questions** and attempt the **Problems** independently; their panels contain final answers only. Predict the effect of a parameter change before running it, and check units and limiting cases.

    **Opening challenge.** Two people have the same total mass and height. Must their leg moments of inertia be equal? Predict which measurements an anthropometric table can supply and which it cannot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    The dependency declaration at the top of this file lists the packages used by this guide. Marimo runs cells in dependency order. Run the notebook in an editor to use its displays and controls; the computational examples below explain their inputs before using them.
    """)
    return


@app.function(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    _vitruvian_image = mo.image(
        repo_file("images/vitruvian_man.jpg"),
        width=240,
        alt="Leonardo da Vinci's Vitruvian Man",
        style={"float": "right", "margin": "20px 0 0 50px"},
    )
    mo.md(rf"""
    <figure>{_vitruvian_image.text}</figure>
    <br><br>
    <i>"Le proporzioni del corpo umano secondo Vitruvio"</i>, also known as the <a href="https://en.wikipedia.org/wiki/Vitruvian_Man" target="_blank">Vitruvian Man</a>, drawing by <a href="https://en.wikipedia.org/wiki/Leonardo_da_Vinci" target="_blank">Leonardo da Vinci</a> circa 1490 based on the work of <a href="https://en.wikipedia.org/wiki/Vitruvius" target="_blank">Marcus Vitruvius Pollio</a> (1st century BC), depicting a man in supposedly ideal human proportions. Image: <a href="https://commons.wikimedia.org/wiki/File:Da_Vinci_Vitruve_Luc_Viatour.jpg">Wikimedia Commons</a>, digitization by <a href="https://lucnix.be/">Luc Viatour / Lucnix</a>.
    <br><br>
    In fact, Leonardo's drawing does not follow the proportions according Vitruvius, but rather the proportions he found after his own anthropometrical studies of the human body. Leonardo was unable to fit a human body inside a circle and a square with the same center, one of the Vitruvius' claims.
    <br><br>
    This is a remarkable historical evidence of not complying with established common knowledge and relying on experimental data for acquiring knowledge about nature, a <i>tour de force</i> for the scientific method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimation of body segment parameters

    Body segment parameters (BSP) of the human body, such as length, area, volume, mass, density, center of mass, moment of inertia, and center of volume, are fundamental for the application of mechanics to the understanding of human movement. Anthropometry is the field concerned with the study of such measurements of the human body. Frequently, one cannot measure most of these parameters of each segment of an individual and these quantities are estimated by indirect methods. The main indirect methods are based in data of cadavers (e.g. Dempster's model), body image scanning of living subjects (e.g., Zatsiorsky-Seluyanov's model), and geometric measurements (e.g., Hanavan's model).

    For reviews available online of the different methods employed in the estimation of BSP, see [Drills et al. (1964)](http://www.oandplibrary.org/al/1964_01_044.asp) and [Bjørnstrup (1995)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.5223).

    Let's look on how to estimate some of the BSP using the anthropometric model of Dempster (1955) with some parameters adapted by Winter (2009) and the model of Zatsiorsky and Seluyanov (Zatsiorsky, 2002), from now on, Zatsiorsky, with parameters adjusted by de Leva (1996). There is at least one Python library for the calculation of human body segment parameters, see Dembia et al. (2014), it implements the Yeadon human inertia geometric model, but we will not use it here.

    For a table with BSP values, also referred as anthropometric table, typically:

    + The mass of each segment is given as fraction of the total body mass.
    + The center of mass (CM) position in the sagittal plane of each segment is given as fraction of the segment length with respect to the proximal or distal joint position.
    + The radius of gyration (Rg) around the transverse axis (rotation at the sagittal plane) and around other axes of each segment is given as fraction of the segment length with respect to (w.r.t.) the center of mass or w.r.t. the proximal or w.r.t. the distal joint position.

    For a formal description of these parameters, see the notebook [Center of Mass and Moment of Inertia](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/center_of_mass_and_moment_of_inertia.py).
    """)
    return


@app.cell
def _():
    # Import the necessary libraries
    import numpy as np
    import pandas as pd
    from IPython.display import display, Math, Latex
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    pd.set_option('max_colwidth', 100)
    return Latex, display, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dempster's model adapted by Winter
    """)
    return


@app.cell
def _(Latex, display, pd):
    BSP_Dmarks = pd.read_csv(repo_file('data/BSP_DempsterWinter.txt'), sep='\t')
    display(Latex('\\text{BSP segments from Dempster\'s model adapted by Winter (2009):}'))
    display(BSP_Dmarks)
    return


@app.cell
def _(Latex, display, pd):
    bsp_D = pd.read_csv(repo_file('data/BSP_DempsterWinter.txt'), index_col=0, sep='\t')
    display(Latex('\\text{BSP values from Dempster\'s model adapted by Winter (2009):}'))
    display(bsp_D)
    return (bsp_D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1 — Read a mass fraction.** A shank has mass fraction 0.0465 in a chosen table. For a 70 kg person, find its mass. Would you use 4.65 or 0.0465 in Python?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 1': mo.md(r"""
    The table fraction is dimensionless: $m=70(0.0465)=3.255$ kg. Using 4.65 would give 325.5 kg, already larger than the whole person. This simple total-mass check catches a percent/fraction error.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zatsiorsky's model adjusted by de Leva

    The segments defined in the Zatsiorsky's model (Zatsiorsky, 2002) adjusted by de Leva (1996) are illustrated in the next figure.

    <figure aria-describedby="figure-caption-1"><center><img src='https://github.com/BMClab/BMC/blob/master/images/BSP_ZdeLeva.png?raw=1' alt='Zatsiorsky anthropometric model'/></center></figure>

    <span id="figure-caption-1" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Segment definition employed in the anthropometric model of Zatsiorsky and Seluyanov (Zatsiorsky, 2002) adjusted by de Leva (1996).<br>Image from a <a href="http://motionanalysis.com/" target="_blank">Motion Analysis Corporation</a> manual.</span>
    """)
    return


@app.cell
def _(Latex, display, pd):
    BSP_Zmarks = pd.read_csv(repo_file('data/BSPlandmarks_ZdeLeva.txt'), sep='\t')
    display(Latex('\\text{BSP landmarks from Zatsiorsky\'s model adjusted by de Leva (1996):}'))
    display(BSP_Zmarks)
    return


@app.cell
def _(Latex, display, pd):
    bsp_Zf = pd.read_csv(repo_file('data/BSPfemale_ZdeLeva.txt'), index_col=0, sep='\t')
    display(Latex('\\text{BSP female values from Zatsiorsky\'s model adjusted by de Leva (1996):}'))
    display(bsp_Zf)
    return (bsp_Zf,)


@app.cell
def _(Latex, display, pd):
    bsp_Zm = pd.read_csv(repo_file('data/BSPmale_ZdeLeva.txt'), index_col=0, sep='\t')
    display(Latex('\\text{BSP male values from Zatsiorsky\'s model adjusted by de Leva (1996):}'))
    display(bsp_Zm)
    return (bsp_Zm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Differences between the anthropometric models from Dempster and Zatsiorsky

    The anthropometric models from Dempster and Zatsiorsky are different in many aspects; regarding the subjetcs investigated in the studies, Dempster's model is based on the data of 8 cadavers of older male individuals (but two of the individuals were of unknown age) analyzed in the United States. Zatsiorsky's model is based on image scanning of 100 young men and 15 young women, at the time all students of a military school in the former Soviet Union.

    The difference between models for some segments is large (see table below): the mass fraction of the thigh segment for Zatsiorsky's model is more than 40% larger than for the Dempster's model, inversely, the trunk segment has about 15% lower mass fraction for Zatisorsky's model. Also, note that some of the  segments don't have the same definition in the two models.
    """)
    return


@app.cell
def _(Latex, bsp_D, bsp_Zf, bsp_Zm, display, np, pd):
    m_D = bsp_D.loc[['Foot', 'Leg', 'Thigh', 'Pelvis', 'Abdomen', 'Thorax', 'Trunk',
                       'Upper arm', 'Forearm', 'Hand', 'Head neck'], 'Mass']
    m_Zf = bsp_Zf.loc[['Foot', 'Shank', 'Thigh', 'Lower trunk', 'Middle trunk', 'Upper trunk',
                       'Trunk', 'Upper arm', 'Forearm', 'Hand', 'Head'], 'Mass']
    m_Zm = bsp_Zm.loc[['Foot', 'Shank', 'Thigh', 'Lower trunk', 'Middle trunk', 'Upper trunk',
                       'Trunk', 'Upper arm', 'Forearm', 'Hand', 'Head'], 'Mass']
    m_D.index = m_Zf.index  # because of different names for some segments

    display(Latex("\\text{Mass fraction difference (in %) of Zatsiorsky's model w.r.t. Dempster's model}"))
    d = pd.DataFrame({'Females': np.around(100 * (m_Zf - m_D) / m_D), \
                      'Males': np.around(100 * (m_Zm - m_D) / m_D)})
    display(d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Center of mass

    See the notebook [Center of Mass and Moment of Inertia](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/center_of_mass_and_moment_of_inertia.py) for a description of center of mass.

    Using the data of the body segment parameters table, the center of mass of a single segment $i$ is (see figure below):

    $$
    r_{i} = r_{i,p} + \text{bsp[i,cmp]} \cdot (r_{i,d}-r_{i,p})
    $$

    Where $r_{i,p}$ and $\:r_{i,d}$ are the positions of the proximal and distal landmarks used to define the $i$ segment.<br>
    Note that $r$ is a vector and may have more than one dimension. The equation for the center of mass is valid in each direction and the calculations are performed independently in each direction. In addition, there is no need to include the mass of the segment in the equation above; the mass of the segment is used only when there is more than one segment.

    For example, given the following coordinates ($x, y$) for the MT2, ankle, knee and hip joints:
    """)
    return


@app.cell
def _(display, np):
    r = np.array([[101.1, 1.3], [84.9, 11.0], [86.4, 54.9], [72.1, 92.8]])/100
    display(np.around(r, 3))
    return (r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The position of the center of mass of each segment and of the lower limb are:
    """)
    return


@app.cell
def _(bsp_D, np, r):
    M = bsp_D.loc[['Foot', 'Leg', 'Thigh'], 'Mass'].sum()
    rcm_foot  = r[1] + bsp_D.loc['Foot', 'CM prox']*(r[0]-r[1])
    rcm_leg   = r[2] + bsp_D.loc['Leg',  'CM prox']*(r[1]-r[2])
    rcm_thigh = r[3] + bsp_D.loc['Thigh','CM prox']*(r[2]-r[3])
    rcm = (bsp_D.loc['Foot','Mass']*rcm_foot + bsp_D.loc['Leg','Mass']*rcm_leg + \
           bsp_D.loc['Thigh','Mass']*rcm_thigh)/M
    print('Foot CM: ', np.around(rcm_foot, 3), 'm')
    print('Leg CM: ', np.around(rcm_leg, 3), 'm')
    print('Thigh CM: ', np.around(rcm_thigh, 3), 'm')
    print('Lower limb CM: ', np.around(rcm, 3), 'm')
    return rcm, rcm_foot, rcm_leg, rcm_thigh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And here is a geometric representation of part of these calculations:
    """)
    return


@app.cell
def _(plt, r, rcm, rcm_foot, rcm_leg, rcm_thigh):
    plt.rc('axes',  labelsize=14, linewidth=1.5)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('lines', markersize=8)
    hfig, hax = plt.subplots(1, 1, figsize=(10, 5))

    # bones and joints
    plt.plot(r[:,0], r[:,1], 'b-')
    plt.plot(r[:,0], r[:,1], 'ko', label='joint')
    # center of mass of each segment
    plt.plot(rcm_foot[0], rcm_foot[1], 'go', label='segment center of mass')
    plt.plot(rcm_leg[0], rcm_leg[1], 'go', rcm_thigh[0], rcm_thigh[1], 'go')
    # total center of mass
    plt.plot(rcm[0], rcm[1], 'ro', label='total center of mass')
    hax.legend(frameon=False, loc='upper left', fontsize=12, numpoints=1)
    plt.arrow(0, 0, r[3,0], r[3,1], color='b', head_width=0.02, overhang=.5, fc="k", ec="k",
              lw=2, length_includes_head=True)
    plt.arrow(r[3,0], r[3,1], rcm_thigh[0] - r [3,0], rcm_thigh[1] - r[3,1], head_width=0.02,
          overhang=.5, fc="b", ec="b", lw=2, length_includes_head=True)
    plt.arrow(0, 0, rcm_thigh[0], rcm_thigh[1], head_width=0.02, overhang=.5, fc="g", ec="g",
              lw=2, length_includes_head=True)
    plt.text(0.30, .5, r'$\mathbf{r}_{thigh,p}$', rotation=38, fontsize=16)
    plt.text(0.77, .85, r'$bsp_{thigh,cmp}*(\mathbf{r}_{i,d}-\mathbf{r}_{i,p})$',
             fontsize=16, color='b')
    plt.text(0.15, .05,
             r'$\mathbf{r}_{thigh,cm}=\mathbf{r}_{i,p}+bsp_{i,cmp}*' +
             r'(\mathbf{r}_{i,d}-\mathbf{r}_{i,p})$',
             rotation=25, fontsize=16, color='g')
    hax.set_xlim(0,1.1)
    hax.set_ylim(0,1.05)
    hax.set_xlabel('x [m]')
    hax.set_ylabel('y [m]')
    hax.set_title('Determination of center of mass', fontsize=16)
    hax.grid()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2 — Locate a segment center.** A proximal joint is at $(0,0)$ m and a distal joint at $(0.4,0.3)$ m. The center lies 43% of the length from proximal to distal. Where is it, and what is the segment length?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 2': mo.md(r"""
    Use the same fraction on the displacement vector: $\mathbf r_C=\mathbf r_P+0.43(\mathbf r_D-\mathbf r_P)=(0.172,0.129)$ m. The length is $\sqrt{0.4^2+0.3^2}=0.5$ m. The percentage is a position along the segment, not an absolute laboratory coordinate.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Moment of inertia

    ### Radius of gyration

    See the notebook [Center of Mass and Moment of Inertia](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/center_of_mass_and_moment_of_inertia.py) for a description of moment of inertia and radius of gyration.

    The radius of gyration (as a fraction of the segment length) is the quantity that is given in the table of body segment parameters. Because of that, we don't need to sum each element of mass of the segment to calculate its moment of inertia; we just need to take the mass of the segment times the radius or gyration squared.

    Using the body segment parameters, the moment of inertia of a single segment $i$ rotating around its own center of mass is (see figure below):

    $$
    I_{i,cm} = M \cdot \text{bsp[i,mass]} \cdot \left(\text{bsp[i,rgcm]} \cdot ||r_{i,d}-r_{i,p}||\right)^2
    $$

    Where $M$ is the total body mass of the subject and $||r_{i,d}-r_{i,p}||$ is the length of the segment $i$.

    For example, the moment of inertia of each segment of the lower limb around each corresponding segment center of mass considering the coordinates (x, y) for the MT2, ankle, knee and hip joints given above are:
    """)
    return


@app.cell
def _(bsp_D, np, r):
    norm = np.linalg.norm
    M_1 = 100
    Icm_foot = M_1 * bsp_D.loc['Foot', 'Mass'] * (bsp_D.loc['Foot', 'Rg CM'] * norm(r[0] - r[1])) ** 2
    Icm_leg = M_1 * bsp_D.loc['Leg', 'Mass'] * (bsp_D.loc['Leg', 'Rg CM'] * norm(r[1] - r[2])) ** 2
    Icm_thigh = M_1 * bsp_D.loc['Thigh', 'Mass'] * (bsp_D.loc['Thigh', 'Rg CM'] * norm(r[2] - r[3])) ** 2
    print('Icm foot: ', np.around(Icm_foot, 3), 'kgm2')
    print('Icm leg: ', np.around(Icm_leg, 3), 'kgm2')
    print('Icm thigh: ', np.around(Icm_thigh, 3), 'kgm2')
    return Icm_foot, Icm_leg, Icm_thigh, M_1, norm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3 — Turn a radius into inertia.** For a 3 kg, 0.4 m segment, the radius of gyration about a specified center-of-mass axis is 30% of the segment length. Find the inertia and its units.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 3': mo.md(r"""
    The dimensional radius is $k=0.30(0.4)=0.12$ m; hence $I_C=mk^2=3(0.12)^2=0.0432$ kg m². Squaring the fraction alone would miss the length scale.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parallel axis theorem

    See the notebook [Center of Mass and Moment of Inertia](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/center_of_mass_and_moment_of_inertia.py) for a description of parallel axis theorem.

    For example, using the parallel axis theorem the moment of inertia of the lower limb around its center of mass is:
    """)
    return


@app.cell
def _(
    Icm_foot,
    Icm_leg,
    Icm_thigh,
    M_1,
    bsp_D,
    norm,
    np,
    rcm,
    rcm_foot,
    rcm_leg,
    rcm_thigh,
):
    Icmll = Icm_foot + M_1 * bsp_D.loc['Foot', 'Mass'] * norm(rcm - rcm_foot) ** 2 + Icm_leg + M_1 * bsp_D.loc['Leg', 'Mass'] * norm(rcm - rcm_leg) ** 2 + Icm_thigh + M_1 * bsp_D.loc['Thigh', 'Mass'] * norm(rcm - rcm_thigh) ** 2
    print('Icm lower limb: ', np.around(Icmll, 3), 'kgm2')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To calculate the moment of inertia of the lower limb around the hip, we use again the parallel axis theorem:
    """)
    return


@app.cell
def _(
    Icm_foot,
    Icm_leg,
    Icm_thigh,
    M_1,
    bsp_D,
    norm,
    np,
    r,
    rcm_foot,
    rcm_leg,
    rcm_thigh,
):
    Ihipll = Icm_foot + M_1 * bsp_D.loc['Foot', 'Mass'] * norm(r[3] - rcm_foot) ** 2 + Icm_leg + M_1 * bsp_D.loc['Leg', 'Mass'] * norm(r[3] - rcm_leg) ** 2 + Icm_thigh + M_1 * bsp_D.loc['Thigh', 'Mass'] * norm(r[3] - rcm_thigh) ** 2
    print('Ihip lower limb: ', np.around(Ihipll, 3), 'kgm2')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that for the correct use of the parallel axis theorem we have to input the moment of inertia around the center of mass of each body. For example, we CAN NOT calculate the moment of inertia around the hip with the moment of inertia of the entire lower limb:
    """)
    return


@app.cell
def _(Icm_foot, Icm_leg, Icm_thigh, M_1, bsp_D, norm, np, r, rcm):
    I = Icm_foot + M_1 * bsp_D.loc['Foot', 'Mass'] * norm(r[3] - rcm) ** 2 + Icm_leg + M_1 * bsp_D.loc['Leg', 'Mass'] * norm(r[3] - rcm) ** 2 + Icm_thigh + M_1 * bsp_D.loc['Thigh', 'Mass'] * norm(r[3] - rcm) ** 2
    print('Icm lower limb: ', np.around(I, 3), 'kgm2. THIS IS WRONG!')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4 — Move the axis.** The preceding segment center is 0.17 m from a parallel proximal axis. Find the proximal inertia. Why must it exceed the center-of-mass inertia?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Solution to Exercise 4': mo.md(r"""
    $I_P=I_C+md^2=0.0432+3(0.17)^2=0.1299$ kg m². The added term is nonnegative. This statement applies to parallel axes; an axis with a different direction requires the inertia tensor.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Center of buoyancy

    [Center of buoyancy](https://en.wikipedia.org/wiki/Buoyancy) is the center of the volume of water which the submerged part of an object displaces. Center of buoyancy is to center of volume as center of gravity is to center of mass.

    For the human body submerged in water, because different parts of the body have different densities, the center of buoyancy is at a different place than the center of gravity, see for example [Yanai and Wilson (2008)](https://onlinelibrary.wiley.com/doi/pdf/10.1002/jst.23).

    <figure aria-describedby="figure-caption-2"><center><img src='https://github.com/BMClab/BMC/blob/master/images/buoyance.png?raw=1' width=400 alt='Center of gravity and center of buoyancy.'/></center></figure>

    <span id="figure-caption-2" class="figure-caption" style="display:block;text-align:center;font-style:italic">Figure. Forces of gravity and buoyancy acting respectively on the center of gravity and center of buoyancy in a submerged human body (figure from Yanai and Wilson (2008)).</span>
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Why can two anthropometric tables give different estimates for the same person?
    2. What distinguishes a center-of-mass fraction from a radius-of-gyration fraction?
    3. Can a planar model determine all six independent components of a 3D inertia tensor?
    4. Revisit the opening challenge: what remains unknown after mass and height are measured?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Checkpoint answers': mo.md(r"""
    1. Population, measurement method, and anatomical segment definitions differ.
    2. The first locates a point; the second determines an axis-specific inertia after squaring its dimensional radius.
    3. No; out-of-plane geometry or additional assumptions are needed.
    4. Segment mass distribution, lengths, landmarks, and axis-specific inertias remain individual quantities.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    1. Take a picture at the frontal plane of somebody standing on one foot on tiptoes with the arms and the other leg abducted at the horizontal.<br>
      1. Estimate the body center of mass of this person. Hint: for simplicity, consider the center of mass of each segment to be located at the middle of the segment and measure these positions using a image digitizer, e.g., [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/).<br>
      2. If the person is almost standing still, through which part of the body a vertical line through the center of mass should necessarily pass? Have you obtained this result? Comment on possible differences between the expected and obtained results.

    2. Consider the following positions of markers placed on a leg (described in the laboratory coordinate system with coordinates $x, y, z$ in cm, the $x$ axis points forward and the $y$ axes points upward): lateral malleolus (**lm** = [2.92, 10.10, 18.85]), medial malleolus (**mm** = [2.71, 10.22, 26.52]), fibular head (**fh** = [5.05, 41.90, 15.41]), and medial condyle (**mc** = [8.29, 41.88, 26.52]). Define the ankle joint center as the centroid between the **lm** and **mm** markers and the knee joint center as the centroid between the **fh** and **mc** markers (same data as in the "Example with actual motion analysis data" of the notebook [Rigid-body transformations in three dimensions](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/transformation_3d.py)). Consider that the principal axes of the leg are aligned with the axes of the respective anatomical coordinate system.<br>
      1. Determine the center of mass position of the leg at the anatomical and laboratory coordinate systems.<br>
      2. Determine the inertia tensor of the leg for a rotation around its proximal joint and around its center of mass.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {'Final answers to original Problems': mo.md(r"""
    1. Your photograph determines the numerical result: $\mathbf r_C=\sum_i m_i\mathbf r_{C_i}/\sum_i m_i$. For quasi-static balance without external support, its vertical projection must lie inside the supporting foot’s contact area. Landmark, perspective and segment-model errors can explain disagreement.

    2. Ankle center: $(2.815,10.160,22.685)$ cm; knee center: $(6.670,41.890,20.965)$ cm; length: 0.320096 m. **Using the male de Leva shank parameters explicitly**, $m_s=0.0433M$, $c=0.4395$, and laboratory center: $(4.97573,27.94467,21.72094)$ cm. With an anatomical origin at the knee and longitudinal axis pointing proximally, the center is $(0,-0.140682,0)$ m. Principal center inertias are $m_sL^2(0.251^2,0.246^2,0.102^2)$ in the table’s sagittal/transverse/longitudinal order. In any declared orthonormal anatomical basis, $I_P=I_C+m_s[(r^Tr)\mathbf1-rr^T]$; laboratory tensors are $RI R^T$. A numerical tensor requires body mass and the full anatomical-axis convention; the markers alone do not specify these choices.
    """)},
        expanded=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Go deeper

    **Optional Winter-table assignment.** Consider the kinematic data from table A.1 of the Winter's book (Winter, 2009) used in problem 2 of the notebook [Angular kinematics in a plane (2D)](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/kinematics_angular_2d.py).<br>
      1. Calculate the center of mass position for each segment and for the whole body (beware that no data are given for the head and arms segments) using the Dempster's and Zatsiorsky's models.<br>
      2. Perform these calculations also for the moment of inertia (of each segment and of the whole body around the corresponding centers of mass).

    The external table is needed for numerical work. Without head and arm data, report partial-body quantities; whole-body center and inertia are underdetermined.

    - [Drill et a. (1964)](http://www.oandplibrary.org/al/1964_01_044.asp) and [Bjørnstrup (1995)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.5223) for a historical overview on the estimation of body segment parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Video lectures on the Internet

    - [Body segments parameters in Biomechanics](https://youtu.be/8hyPiha-lFU)
    - [Estimating and Visualizing the Inertia of the Human Body with Python](https://youtu.be/H9AK65ZY-Vw)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Additional verified lecture

    [MIT OpenCourseWare](https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/pages/week-5-momentum-and-impulse/17-1-definition-of-the-center-of-mass/) — Center of mass: compare the weighted-average definition with a physical balance argument.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Bjørnstrup J (1995) [Estimation of Human Body Segment Parameters - Historical Background](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.5223). Technical Report.

    - de Leva P (1996) [Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters](http://ebm.ufabc.edu.br/wp-content/uploads/2013/12/Leva-1996.pdf). Journal of Biomechanics, 29, 9, 1223-1230.

    - Dembia C, Moore JK, Hubbard M (2014) [An object oriented implementation of the Yeadon human inertia model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4329601/). F1000Research. 2014;3:223. doi:10.12688/f1000research.5292.2.

    - Drills R, Contini R, Bluestein M (1964) [Body segment parameters: a survey of measurement techniques](http://www.oandplibrary.org/al/1964_01_044.asp). Artificial Limbs, 8, 44-66.

    - Kwon Y-H (1998) [BSP Estimation Methods](http://kwon3d.com/theory/bsp.html).

    - Ruina A, Rudra P (2013) [Introduction to Statics and Dynamics](https://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.

    - Winter DA (2009) [Biomechanics and motor control of human movement](https://books.google.com.br/books?id=_bFHL08IWfwC). 4 ed. Hoboken, EUA: Wiley.

    - Yanai T, Wilson BD (2008) [How does buoyancy influence front-crawl performance? Exploring the assumptions](https://onlinelibrary.wiley.com/doi/pdf/10.1002/jst.23). Sports Technol., 1 2–3, 89–99.

    - Zatsiorsky VM (2002) [Kinetics of human motion](https://books.google.com.br/books?id=wp3zt7oF8a0C&lpg=PA571&ots=Kjc17DAl19&dq=ZATSIORSKY%2C%20Vladimir%20M.%20Kinetics%20of%20human%20motion&hl=pt-BR&pg=PP1#v=onepage&q&f=false). Champaign, IL: Human Kinetics.

    - Some of the original works on body segment parameters:

    - Contini R (1972) [Body Segment Parameters, Part II](http://www.oandplibrary.org/al/1972_01_001.asp). Artificial Limbs, 16, 1-19.

    - Dempster WT (1955) [Space requirements of the seated operator: geometrical, kinematic, and mechanical aspects of the body, with special reference to the limbs](http://deepblue.lib.umich.edu/handle/2027.42/4540). WADC Technical Report 55-159, AD-087-892, Wright-Patterson Air Force Base, Ohio.

    - Hanavan EP (1964). [A mathematical model of the human body](http://www.dtic.mil/cgi-bin/GetTRDoc?AD=AD0608463). AMRL-TR-64-102, AD-608-463. Aerospace Medical Research Laboratories, Wright-Patterson Air Force Base, Ohio.
    """)
    return


if __name__ == "__main__":
    app.run()
