import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Postural Sway: Rambling and Trembling decomposition of the COP

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Import-necessary-Python-libraries-and-configure-the-environment" data-toc-modified-id="Import-necessary-Python-libraries-and-configure-the-environment-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import necessary Python libraries and configure the environment</a></span></li><li><span><a href="#Hypothesis-about-the-human-postural-control-(Zatsiorsky)" data-toc-modified-id="Hypothesis-about-the-human-postural-control-(Zatsiorsky)-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Hypothesis about the human postural control (Zatsiorsky)</a></span><ul class="toc-item"><li><span><a href="#Findings-supporting-a-two-subsystems-control" data-toc-modified-id="Findings-supporting-a-two-subsystems-control-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Findings supporting a two-subsystems control</a></span></li><li><span><a href="#Force-fields-during-quiet-standing" data-toc-modified-id="Force-fields-during-quiet-standing-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Force fields during quiet standing</a></span></li><li><span><a href="#Prolonged-Standing" data-toc-modified-id="Prolonged-Standing-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Prolonged Standing</a></span></li><li><span><a href="#Postural-sway:-a-fractal-process" data-toc-modified-id="Postural-sway:-a-fractal-process-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Postural sway: a fractal process</a></span></li></ul></li><li><span><a href="#Rambling-and-Trembling-decomposition-of-the-COP" data-toc-modified-id="Rambling-and-Trembling-decomposition-of-the-COP-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Rambling and Trembling decomposition of the COP</a></span><ul class="toc-item"><li><span><a href="#Rambling-and-Trembling-during-quiet-standing" data-toc-modified-id="Rambling-and-Trembling-during-quiet-standing-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Rambling and Trembling during quiet standing</a></span></li><li><span><a href="#Balance-maintenance-according-to-the-Rambling-and-Trembling-hypothesis" data-toc-modified-id="Balance-maintenance-according-to-the-Rambling-and-Trembling-hypothesis-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Balance maintenance according to the Rambling and Trembling hypothesis</a></span></li><li><span><a href="#Decomposition-based-on-the-instant-equilibrium-point-hypothesis" data-toc-modified-id="Decomposition-based-on-the-instant-equilibrium-point-hypothesis-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Decomposition based on the instant-equilibrium-point hypothesis</a></span></li><li><span><a href="#Python-functions" data-toc-modified-id="Python-functions-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Python functions</a></span></li><li><span><a href="#Load-some-postural-sway-data" data-toc-modified-id="Load-some-postural-sway-data-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Load some postural sway data</a></span></li><li><span><a href="#Running-the-rambling-trembling-decomposition" data-toc-modified-id="Running-the-rambling-trembling-decomposition-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Running the rambling-trembling decomposition</a></span></li><li><span><a href="#Run-the-rambling-trembling-decomposition-for-a-sample-of-subjects" data-toc-modified-id="Run-the-rambling-trembling-decomposition-for-a-sample-of-subjects-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>Run the rambling-trembling decomposition for a sample of subjects</a></span><ul class="toc-item"><li><span><a href="#The-results-were-saved-locally,-we-can-load-it-after-the-first-run" data-toc-modified-id="The-results-were-saved-locally,-we-can-load-it-after-the-first-run-3.7.1"><span class="toc-item-num">3.7.1&nbsp;&nbsp;</span>The results were saved locally, we can load it after the first run</a></span></li></ul></li><li><span><a href="#Data-description" data-toc-modified-id="Data-description-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>Data description</a></span></li><li><span><a href="#Data-visualization" data-toc-modified-id="Data-visualization-3.9"><span class="toc-item-num">3.9&nbsp;&nbsp;</span>Data visualization</a></span></li></ul></li><li><span><a href="#The-COP-COGv-decomposition" data-toc-modified-id="The-COP-COGv-decomposition-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>The COP-COGv decomposition</a></span><ul class="toc-item"><li><span><a href="#Run-the-COP-COGv-decomposition-for-the-same-subjects" data-toc-modified-id="Run-the-COP-COGv-decomposition-for-the-same-subjects-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Run the COP-COGv decomposition for the same subjects</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Careful-using-RMS-as-measurement-of-the-components-amplitude" data-toc-modified-id="Careful-using-RMS-as-measurement-of-the-components-amplitude-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Careful using RMS as measurement of the components amplitude</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Import necessary Python libraries and configure the environment
        """
    )
    return


@app.cell
def _():
    import sys, os
    sys.path.insert(1, r'./../functions')
    import numpy as np
    import pandas as pd
    pd.set_option('precision', 4)
    # matplotlib configuration:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    # '%matplotlib inline' command supported automatically in marimo
    import seaborn as sns
    sns.set_context("notebook", font_scale=1.4,
                    rc={'font.size': 16, 'lines.linewidth': 2,\
                        'lines.markersize': 10, 'axes.titlesize': 'x-large'})
    matplotlib.rc('legend', numpoints=1, fontsize=16)
    # IPython:
    from IPython.display import display, Image, IFrame
    import ipywidgets
    from ipywidgets import FloatProgress, interactive
    return (
        FloatProgress,
        IFrame,
        Image,
        display,
        gridspec,
        np,
        os,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Hypothesis about the human postural control (Zatsiorsky)

        The control system for equilibrium of the human body includes two subsystems:  
         1. the first one determining a reference position with respect to which the body equilibrium is maintained.  
         2. the second one maintaining equilibrium about the pre-selected reference point.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Findings supporting a two-subsystems control

        > Gurfinkel et al. (1995) Kinesthetic reference for human orthograde posture. Neuroscience, 68, 229.  

        The supporting surface where the subjects stood was tilted slowly (0.04$^o/s$).  
        During the tilting, **small high frequency oscillations** of the body were superimposed on a **large slow body movements**.  
        The usual process of stabilization of the body continued, but the **instant equilibrium** was maintained relative to a **slowly changing position**, rather than around a fixed set point.

        > See Zatsiorsky & Duarte (2000), for a review about findings supporting a two-subsystems control.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Force fields during quiet standing
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/forcevectors.png?raw=1" width="500" alt="Force vectors"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Prolonged Standing
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/prolongedstanding.png?raw=1" width="600" alt="Prolonged standing"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Postural sway: a fractal process
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/fractal.png?raw=1" width="500" alt="Fractal"/></figure>

        Duarte & Zatsiorsky (2000)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Rambling and Trembling decomposition of the COP

        The reference point migration is called **rambling** and the COP migration around the reference was coined **trembling**.

        - Zatsiorsky VM, Duarte M (1999) Instant equilibrium point and its migration in standing tasks: Rambling and trembling components of the stabilogram. Motor Control, 3, 28. [PDF](http://demotu.org/pubs/mc99.pdf).  
        - Zatsiorsky VM, Duarte M (2000) Rambling and trembling in quiet standing. Motor Control, 2, 185. [PDF](http://demotu.org/pubs/e00.pdf).  
        """
    )
    return


@app.cell
def _(IFrame):
    IFrame('http://pesquisa.ufabc.edu.br/bmclab/pubs/mc99b.pdf', width='100%', height='500px')
    return


@app.cell
def _(IFrame):
    IFrame('http://pesquisa.ufabc.edu.br/bmclab/pubs/mc00.pdf', width='100%', height='500px')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Rambling and Trembling during quiet standing
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/rambtremb.png?raw=1" width="600" alt="Rambling and Trembling"/></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
         ### Balance maintenance according to the Rambling and Trembling hypothesis
 
         1. The CNS specifies an intended position of the body. The intended position is specified by a reference point on the supporting surface with respect to which body equilibrium is instantly maintained.  
         2. The reference point migrates and can be considered a moving attracting point.  
         3. The body sways because of two reasons: the migration of the reference point and the deviation away from the reference point.  
         4. When the deflection is not too large, the restoring force is due to the ‘apparent intrinsic stiffness’ of the muscles. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Decomposition based on the instant-equilibrium-point hypothesis

        The rambling and trembling components of the COP trajectory were computed in the following way:  
         1. The instants when$F_{hor}$changes its sign from positive (negative) to negative (positive) are determined.  
         2. The COP positions at these instants (the instant equilibrium points, IEP, or zero-force points) are determined.  
         3. To obtain an estimate of the rambling trajectory, the IEP discrete positions are interpolated by cubic spline functions.  
         4. To obtain the trembling trajectory, the deviation of the COP from the rambling trajectory is determined.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Python functions
        """
    )
    return


@app.cell
def _(np):
    def detect_zerocross(y, method='no_zero'):
        """
        Zero-crossing detection, index before a zero cross.
        """

    
        y = np.asarray(y)
    
        if method == 'no_zero':
            # doesn't detect if a value is exactly zero:
            inds0 = np.where(y[:-1] * y[1:] < 0)[0]
        else:
            # detects if a value is exactly zero:
            inds0 = np.where(np.diff(np.signbit(y)))[0]  
        
        return inds0
    return (detect_zerocross,)


@app.cell
def _(detect_zerocross, np):
    def iep_decomp(time, force, cop):
        """
        Center of pressure decomposition based on the IEP hypothesis.
        """
    
        from scipy.interpolate import UnivariateSpline
    
        force = force - np.mean(force)
    
        # detect zeros
        inds0 = detect_zerocross(force)
    
        # select data between first and last zeros
        time = time[inds0[0]:inds0[-1]+1]
        force = force[inds0[0]:inds0[-1]+1]
        cop = cop[inds0[0]:inds0[-1]+1]
    
        # IEP0 & IEP:
        iep0 = inds0 - inds0[0]
        spl = UnivariateSpline(time[iep0], cop[iep0], k=3, s=0)
        rambling = spl(time)
        trembling = cop - rambling
    
        return rambling, trembling, iep0, inds0
    return (iep_decomp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Load some postural sway data
        """
    )
    return


@app.cell
def _(os, pd):
    path2 = './../../../X/BDB/'
    _filename = os.path.join(path2, 'BDS00038.txt')
    grf = pd.read_csv(_filename, delimiter='\t', skiprows=1, header=None, names=['Time', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'COPx', 'COPy'], engine='c')
    (t, Fx, _Fy, Fz, _Mx, _My, _Mz, COPx, _COPy) = [_ for _ in grf.values.T]
    return COPx, Fx, grf, t


@app.cell
def _(gridspec, plt):
    def plot_grf(grf):
        (t, Fx, _Fy, Fz, _Mx, _My, _Mz, COPx, _COPy) = [_ for _ in grf.values.T]
        (Funits, Munits, COPunits) = ('N', 'Nm', 'cm')
        plt.figure(figsize=(12, 7))
        gs1 = gridspec.GridSpec(3, 1)
        gs1.update(bottom=0.52, top=0.96, hspace=0.12, wspace=0.2)
        (ax1, ax2, ax3) = (plt.subplot(gs1[0]), plt.subplot(gs1[1]), plt.subplot(gs1[2]))
        gs2 = gridspec.GridSpec(3, 3)
        gs2.update(bottom=0.08, top=0.42, wspace=0.3)
        (ax4, ax5) = (plt.subplot(gs2[:, :-1]), plt.subplot(gs2[:, 2]))
        ax1.set_ylabel('Fx (%s)' % Funits)
        (ax1.set_xticklabels([]), ax1.locator_params(axis='y', nbins=4))
        ax1.yaxis.set_label_coords(-0.07, 0.5)
        ax2.set_ylabel('Fy (%s)' % Funits)
        (ax2.set_xticklabels([]), ax2.locator_params(axis='y', nbins=4))
        ax2.yaxis.set_label_coords(-0.07, 0.5)
        ax3.set_ylabel('Fz (%s)' % Funits)
        ax3.locator_params(axis='y', nbins=4)
        ax3.yaxis.set_label_coords(-0.07, 0.5)
        ax3.set_xlabel('Time (s)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('COP (%s)' % COPunits)
        ax5.set_xlabel('COPml (%s)' % COPunits)
        ax5.set_ylabel('COPap (%s)' % COPunits)
        (ax1.plot(t, Fx), ax2.plot(t, _Fy), ax3.plot(t, Fz))
        ax4.plot(t, COPx, 'b', label='COP ap')
        ax4.plot(t, _COPy, 'r', label='COP ml')
        ax4.yaxis.set_label_coords(-0.1, 0.5)
        ax4.legend(fontsize=12, loc='best', framealpha=0.5)
        ax5.plot(_COPy, COPx)
        ax5.locator_params(axis='both', nbins=5)
        plt.suptitle('Ground reaction force data during quiet standing', fontsize=20, y=1)
        plt.show()
    return (plot_grf,)


@app.cell
def _(grf, plot_grf):
    plot_grf(grf)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Running the rambling-trembling decomposition
        """
    )
    return


@app.cell
def _(COPx, Fx, iep_decomp, np, t):
    rambling, trembling, iep0, inds0 = iep_decomp(t, Fx, COPx)
    t2 = t[inds0[0]:inds0[-1]+1]
    Fx2 = Fx[inds0[0]:inds0[-1]+1] - np.mean(Fx)
    COPx2 = COPx[inds0[0]:inds0[-1]+1]
    return COPx2, Fx2, iep0, inds0, rambling, t2, trembling


@app.cell
def _(gridspec, plt):
    def plot_rambtremb(t, Fx, COPx, rambling, trembling, iep0, inds0):
        Funits, Munits, COPunits = 'N', 'Nm', 'cm'
        plt.figure(figsize=(12, 6))
        gs1 = gridspec.GridSpec(4, 1)
        gs1.update(bottom=0.01, top=0.96, hspace=0.12, wspace=.15)
        ax1 = plt.subplot(gs1[0])
        ax2 = plt.subplot(gs1[1])
        ax3 = plt.subplot(gs1[2])
        ax4 = plt.subplot(gs1[3])
    
        ax1.set_ylabel('F (%s)' %Funits)
        ax1.set_xticklabels([]), ax1.locator_params(axis='y', nbins=4)
        ax1.yaxis.set_label_coords(-.05, 0.5)
        ax2.set_ylabel('COP (%s)' %COPunits)
        ax2.set_xticklabels([]), ax2.locator_params(axis='y', nbins=4)
        ax2.yaxis.set_label_coords(-.05, 0.5)
        ax3.set_ylabel('Rambling'), ax3.locator_params(axis='y', nbins=4)
        ax3.set_xticklabels([]), ax3.locator_params(axis='y', nbins=4)
        ax3.yaxis.set_label_coords(-.05, 0.5)
        ax4.set_ylabel('Trembling'), ax4.locator_params(axis='y', nbins=4)
        ax4.yaxis.set_label_coords(-.05, 0.5)
        ax4.set_xlabel('Time (s)')

        ax1.plot(t, Fx)
        ax1.plot(t[iep0], Fx[iep0], 'ro', markersize=5)
        ax2.plot(t, COPx, linewidth=3)
        ax3.plot(t, rambling, linewidth=3)
        ax3.plot(t, COPx, 'k', linewidth=.5)
        ax4.plot(t, trembling, linewidth=3)
    
        plt.suptitle('Rambling & Trembling decomposition during quiet standing', fontsize=18, y=1)

        plt.show()
    return (plot_rambtremb,)


@app.cell
def _(COPx2, Fx2, iep0, inds0, plot_rambtremb, rambling, t2, trembling):
    plot_rambtremb(t2, Fx2, COPx2, rambling, trembling, iep0, inds0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Run the rambling-trembling decomposition for a sample of subjects

        > Santos DA, Duarte M. (2016) A public data set of human balance evaluations. PeerJ4:e2648 https://doi.org/10.7717/peerj.2648.

        The data set comprises signals from the force platform (raw data for the force, moments of forces, and centers of pressure) of 163 subjects plus one file with information about the subjects and balance conditions and the results of the other evaluations.   

        Subject’s balance was evaluated by posturography during standing still for 60 s in four different conditions where vision and the standing surface were manipulated.
        """
    )
    return


@app.cell
def _(os, pd):
    path2_1 = './../../../X/BDB/'
    _fname = os.path.join(path2_1, 'BDSinfo.txt')
    BDSinfo = pd.read_csv(_fname, sep='\t', header=0, index_col=None, engine='c', encoding='utf-8')
    print('Information of %s subjects loaded (%s rows, %s columns).' % (len(pd.unique(BDSinfo.Subject)), BDSinfo.shape[0], BDSinfo.shape[1]))
    return BDSinfo, path2_1


@app.cell
def _(BDSinfo, FloatProgress, display, iep_decomp, np, os, path2_1, pd):
    _fp = FloatProgress(min=0, max=len(BDSinfo.Trial) - 1)
    display(_fp)
    _freq = 100
    for (_i, _fname) in enumerate(BDSinfo.Trial):
        _filename = os.path.join(path2_1, _fname + '.txt')
        _fp.description = 'Reading data from file %s (%s/%s)/n' % (os.path.basename(_filename), _i + 1, len(BDSinfo.Trial))
        _fp.value = _i
        grf_1 = pd.read_csv(_filename, delimiter='\t', skiprows=1, header=None, names=['Time', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'COPx', 'COPy'], engine='c')
        (t_1, Fx_1, _Fy, Fz_1, _Mx, _My, _Mz, COPx_1, _COPy) = [_ for _ in grf_1.values.T]
        (rambling_1, trembling_1, iep0_1, inds0_1) = iep_decomp(t_1, Fx_1, COPx_1)
        t2_1 = t_1[inds0_1[0]:inds0_1[-1] + 1]
        Fx2_1 = Fx_1[inds0_1[0]:inds0_1[-1] + 1] - np.mean(Fx_1)
        COPx2_1 = COPx_1[inds0_1[0]:inds0_1[-1] + 1]
        (_COPsd, Rsd, Tsd) = (np.std(COPx2_1), np.std(rambling_1), np.std(trembling_1))
        BDSinfo.loc[_i, 'COP_sd'] = _COPsd
        BDSinfo.loc[_i, 'Rambling_sd'] = Rsd
        BDSinfo.loc[_i, 'Trembling_sd'] = Tsd
    BDSinfo.to_csv(os.path.join(path2_1, 'BDSinfoIEP.txt'), sep='\t', encoding='utf-8', index=False)
    print('Data from %d files were processed.' % len(BDSinfo.Trial))
    return COPx2_1, Fz_1, inds0_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The results were saved locally, we can load it after the first run
        """
    )
    return


@app.cell
def _(os, path2_1, pd):
    _fname = os.path.join(path2_1, 'BDSinfoIEP.txt')
    BDSinfo_1 = pd.read_csv(_fname, sep='\t', header=0, index_col=None, engine='c', encoding='utf-8')
    print('Information from %s files successfully loaded (total of %s subjects).' % (len(BDSinfo_1), len(pd.unique(BDSinfo_1.Subject))))
    return (BDSinfo_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because we have 3 trials per condition for each subject, let's take the median across trials for each subject as representative of the subject:
        """
    )
    return


@app.cell
def _(BDSinfo_1, pd):
    BDSinfo_2 = BDSinfo_1.groupby(['Subject', 'Vision', 'Surface', 'Illness', 'Disability', 'AgeGroup'], as_index=False).median()
    print('%s subjects.' % len(pd.unique(BDSinfo_2.Subject)))
    return (BDSinfo_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We may choose to not consider the subjects with disabilities because their numbers are unbalanced in the age groups:
        """
    )
    return


@app.cell
def _(BDSinfo_2, display, pd):
    print('Before selection: %s subjects.' % len(pd.unique(BDSinfo_2.Subject)))
    display(BDSinfo_2.drop_duplicates(subset='Subject')[['AgeGroup', 'Subject']].groupby(['AgeGroup']).count())
    print('After selection: %s subjects.' % len(pd.unique(BDSinfo_2.Subject)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Data description
        """
    )
    return


@app.cell
def _(BDSinfo_2, np, pd):
    pd.set_option('precision', 2)
    BDSinfo_2.groupby(['AgeGroup', 'Vision', 'Surface'])[['COP_sd', 'Rambling_sd', 'Trembling_sd']].agg([np.mean, np.std])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Data visualization
        """
    )
    return


@app.cell
def _(np, plt, sns):
    def plot_summary(data):
        g0 = sns.catplot(x='Vision', y='COP_sd', hue='AgeGroup', order=['Open', 'Closed'],
                         data=data, estimator=np.mean, ci=95, col='Surface',
                         kind='point', dodge=True, sharey=False, height=3, aspect=1.6)
        g0.set_xticklabels(''), g0.set_axis_labels('', 'COP (cm)')
        g1 = sns.catplot(x='Vision', y='Rambling_sd', hue='AgeGroup', order=['Open', 'Closed'],
                         data=data, estimator=np.mean, ci=95, col='Surface',
                         kind='point', dodge=True, sharey=False, height=3, aspect=1.6)
        g1.set_axis_labels('', 'Rambling (cm)')
        g1.set_titles('',''), g1.set_xticklabels('')
        g2 = sns.catplot(x='Vision', y='Trembling_sd', hue='AgeGroup', order=['Open', 'Closed'],
                         data=data, estimator=np.mean, ci=95, col='Surface',
                         kind='point', dodge=True, sharey=False, height=3, aspect=1.6)
        g2.set_axis_labels('Vision', 'Trembling (cm)'), g2.set_titles('','')
        plt.show()
    return (plot_summary,)


@app.cell
def _(BDSinfo_2, plot_summary):
    plot_summary(BDSinfo_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The COP-COGv decomposition

        It's possible to estimate the COG vertical projection (COGv or GL) from the COP displacement based on the inverted-pendulum model of the body on quiet standing.  

        See the notebook [The inverted pendulum model of the human standing posture](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/IP_Model.ipynb)

        Let's do some analysis with that.
        """
    )
    return


@app.cell
def _(COPx2_1, Fz_1, np, plt):
    from cogve import cogve
    (_fig, ax) = plt.subplots(1, 1, figsize=(12, 6))
    _cogv = cogve(COPx2_1, freq=100, mass=np.mean(Fz_1) / 10, height=170, ax=ax, show=True)
    cop_cogv = COPx2_1 - _cogv
    return ax, cogve


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Run the COP-COGv decomposition for the same subjects
        """
    )
    return


@app.cell
def _(display, os, path2_1, pd):
    _fname = os.path.join(path2_1, 'BDSinfo.txt')
    BDSinfo_3 = pd.read_csv(_fname, sep='\t', header=0, index_col=None, engine='c', encoding='utf-8')
    print('Information of %s subjects loaded (%s rows, %s columns).' % (len(pd.unique(BDSinfo_3.Subject)), BDSinfo_3.shape[0], BDSinfo_3.shape[1]))
    display(BDSinfo_3.drop_duplicates(subset='Subject')[['AgeGroup', 'Subject']].groupby(['AgeGroup']).count())
    print('After selection: %s subjects.' % len(pd.unique(BDSinfo_3.Subject)))
    return (BDSinfo_3,)


@app.cell
def _(
    BDSinfo_3,
    FloatProgress,
    ax,
    cogve,
    display,
    inds0_1,
    np,
    os,
    path2_1,
    pd,
):
    _fp = FloatProgress(min=0, max=len(BDSinfo_3.Trial) - 1)
    display(_fp)
    _freq = 100
    for (_i, _fname) in enumerate(BDSinfo_3.Trial):
        _filename = os.path.join(path2_1, _fname + '.txt')
        _fp.description = 'Reading data from file %s (%s/%s)/n' % (os.path.basename(_filename), _i + 1, len(BDSinfo_3.Trial))
        _fp.value = _i
        grf_2 = pd.read_csv(_filename, delimiter='\t', skiprows=1, header=None, names=['Time', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'COPx', 'COPy'], engine='c')
        (t_2, Fx_2, _Fy, Fz_2, _Mx, _My, _Mz, COPx_2, _COPy) = [_ for _ in grf_2.values.T]
        t2_2 = t_2[inds0_1[0]:inds0_1[-1] + 1]
        Fx2_2 = Fx_2[inds0_1[0]:inds0_1[-1] + 1] - np.mean(Fx_2)
        COPx2_2 = COPx_2[inds0_1[0]:inds0_1[-1] + 1]
        _cogv = cogve(COPx2_2, freq=100, mass=np.mean(Fz_2) / 10, height=170, ax=ax, show=False)
        copcogv = COPx2_2 - _cogv
        (_COPsd, COGVsd, COPCOGVsd) = (np.std(COPx2_2), np.std(_cogv), np.std(copcogv))
        BDSinfo_3.loc[_i, 'COP2_sd'] = _COPsd
        BDSinfo_3.loc[_i, 'COGv_sd'] = COGVsd
        BDSinfo_3.loc[_i, 'COPCOGv_sd'] = COPCOGVsd
    BDSinfo_3.to_csv(os.path.join(path2_1, 'BDSinfoCOPCOGv.txt'), sep='\t', encoding='utf-8', index=False)
    print('Data from %d files were processed.' % len(BDSinfo_3.Trial))
    return


@app.cell
def _(np, plt, sns):
    def plot_summary2(data):
        g0 = sns.catplot(x='Vision', y='COP2_sd', hue='AgeGroup', order=['Open', 'Closed'],
                         data=data, estimator=np.mean, ci=95, col='Surface',
                         kind='point', dodge=True, sharey=False, height=3, aspect=1.6)
        g0.set_xticklabels(''), g0.set_axis_labels('', 'COP (cm)')
        g1 = sns.catplot(x='Vision', y='COGv_sd', hue='AgeGroup', order=['Open', 'Closed'],
                         data=data, estimator=np.mean, ci=95, col='Surface',
                         kind='point', dodge=True, sharey=False, height=3, aspect=1.6)
        g1.set_axis_labels('', 'COGv (cm)')
        g1.set_titles('',''), g1.set_xticklabels('')
        g2 = sns.catplot(x='Vision', y='COPCOGv_sd', hue='AgeGroup', order=['Open', 'Closed'],
                         data=data, estimator=np.mean, ci=95, col='Surface',
                         kind='point', dodge=True, sharey=False, height=3, aspect=1.6)
        g2.set_axis_labels('Vision', 'COP-COGv (cm)'), g2.set_titles('','')
        plt.show()
    return (plot_summary2,)


@app.cell
def _(BDSinfo_3, plot_summary2):
    plot_summary2(BDSinfo_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Gurfinkel et al. (1995) Kinesthetic reference for human orthograde posture. Neuroscience, 68, 229.  
        - Santos DA, Duarte M. (2016) A public data set of human balance evaluations. PeerJ4:e2648 https://doi.org/10.7717/peerj.2648.  
        - Zatsiorsky VM, Duarte M (1999) Instant equilibrium point and its migration in standing tasks: Rambling and trembling components of the stabilogram. Motor Control, 3, 28. [PDF](http://demotu.org/pubs/mc99.pdf).  
        - Zatsiorsky VM, Duarte M (2000) Rambling and trembling in quiet standing. Motor Control, 2, 185. [PDF](http://demotu.org/pubs/e00.pdf).  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Obrigado
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Careful using RMS as measurement of the components amplitude
        """
    )
    return


@app.cell
def _(np, plt):
    from scipy import signal
    _freq = 100
    t_3 = np.arange(0, 1, 0.01)
    w = 2 * np.pi * 1
    rambling_2 = np.sin(w * t_3)
    trembling_2 = 0.1 * np.sin(10 * w * t_3)
    cop = rambling_2 + trembling_2
    rms = lambda x: np.sqrt(np.mean(x * x))
    RMScop = rms(cop)
    RMSrmb = rms(rambling_2)
    RMStmb = rms(trembling_2)
    (_fig, ax1) = plt.subplots(1, 1, figsize=(12, 6))
    ax1.plot(t_3, cop, 'b', linewidth=3, label='COP:          RMS=%.3f' % RMScop)
    ax1.plot(t_3, rambling_2, 'g', linewidth=3, label='Rambling:  RMS=%.3f' % RMSrmb)
    ax1.plot(t_3, trembling_2, 'r', linewidth=3, label='Trembling: RMS=%.3f' % RMStmb)
    ax1.legend(frameon=False, fontsize=18)
    ax1.set_title("Fictitious COP, Rambling and Trembling and their 'amplitudes'", fontsize=18)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
