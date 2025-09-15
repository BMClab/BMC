import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Inverse dynamics (2D) for gait analysis 

        > Marcos Duarte, Renato Naville Watanabe  
        > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1><br>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Forward-and-inverse-dynamics" data-toc-modified-id="Forward-and-inverse-dynamics-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Forward and inverse dynamics</a></span></li><li><span><a href="#Estimation-of-joint-force-and-moments-of-force-by-inverse-dynamics" data-toc-modified-id="Estimation-of-joint-force-and-moments-of-force-by-inverse-dynamics-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Estimation of joint force and moments of force by inverse dynamics</a></span><ul class="toc-item"><li><span><a href="#Free-body-diagrams" data-toc-modified-id="Free-body-diagrams-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Free body diagrams</a></span></li><li><span><a href="#Equations-of-motion" data-toc-modified-id="Equations-of-motion-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Equations of motion</a></span></li><li><span><a href="#The-recursive-approach-for-inverse-dynamics-of-multi-body-systems" data-toc-modified-id="The-recursive-approach-for-inverse-dynamics-of-multi-body-systems-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>The recursive approach for inverse dynamics of multi-body systems</a></span></li><li><span><a href="#Python-function-invdyn2d.py" data-toc-modified-id="Python-function-invdyn2d.py-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Python function <code>invdyn2d.py</code></a></span></li><li><span><a href="#Experimental-data" data-toc-modified-id="Experimental-data-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Experimental data</a></span></li><li><span><a href="#Load-data-file" data-toc-modified-id="Load-data-file-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Load data file</a></span></li><li><span><a href="#Data-filtering" data-toc-modified-id="Data-filtering-3.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>Data filtering</a></span></li><li><span><a href="#Data-selection" data-toc-modified-id="Data-selection-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>Data selection</a></span></li><li><span><a href="#Plot-file-data" data-toc-modified-id="Plot-file-data-3.9"><span class="toc-item-num">3.9&nbsp;&nbsp;</span>Plot file data</a></span></li><li><span><a href="#Body-segment-parameters" data-toc-modified-id="Body-segment-parameters-3.10"><span class="toc-item-num">3.10&nbsp;&nbsp;</span>Body-segment parameters</a></span></li><li><span><a href="#Kinematic-calculations" data-toc-modified-id="Kinematic-calculations-3.11"><span class="toc-item-num">3.11&nbsp;&nbsp;</span>Kinematic calculations</a></span></li><li><span><a href="#Plot-joint-angles" data-toc-modified-id="Plot-joint-angles-3.12"><span class="toc-item-num">3.12&nbsp;&nbsp;</span>Plot joint angles</a></span></li><li><span><a href="#Inverse-dynamics-calculations" data-toc-modified-id="Inverse-dynamics-calculations-3.13"><span class="toc-item-num">3.13&nbsp;&nbsp;</span>Inverse dynamics calculations</a></span></li><li><span><a href="#Load-files-with-true-joint-forces-and-moments-of-force" data-toc-modified-id="Load-files-with-true-joint-forces-and-moments-of-force-3.14"><span class="toc-item-num">3.14&nbsp;&nbsp;</span>Load files with true joint forces and moments of force</a></span></li><li><span><a href="#Plot-calculated-variables-and-their-true-values" data-toc-modified-id="Plot-calculated-variables-and-their-true-values-3.15"><span class="toc-item-num">3.15&nbsp;&nbsp;</span>Plot calculated variables and their true values</a></span></li></ul></li><li><span><a href="#Contribution-of-each-term-to-the-joint-force-and-moment-of-force" data-toc-modified-id="Contribution-of-each-term-to-the-joint-force-and-moment-of-force-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Contribution of each term to the joint force and moment of force</a></span><ul class="toc-item"><li><span><a href="#Quasi-static-analysis" data-toc-modified-id="Quasi-static-analysis-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Quasi-static analysis</a></span></li><li><span><a href="#Neglecting-the-acceleration-and-mass-(weight)-of-the-segments" data-toc-modified-id="Neglecting-the-acceleration-and-mass-(weight)-of-the-segments-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Neglecting the acceleration and mass (weight) of the segments</a></span></li><li><span><a href="#WARNING:-the-calculated-resultant-joint-force-is-not-the-actual-joint-reaction-force!" data-toc-modified-id="WARNING:-the-calculated-resultant-joint-force-is-not-the-actual-joint-reaction-force!-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>WARNING: the calculated resultant joint force is not the actual joint reaction force!</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></span></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Python setup
        """
    )
    return


@app.cell
def _():
    # import the necessary libraries
    import numpy as np
    from numpy import cross
    from IPython.display import IFrame
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('notebook', font_scale=1.2, rc={"lines.linewidth": 2})
    import sys
    sys.path.insert(1, r'./../functions')
    return IFrame, cross, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Forward and inverse dynamics

        With respect to the equations of motion to determine the dynamics of a system, there are two general approaches: forward (or direct) and inverse dynamics. For example, consider the solution of Newton's second law for a particle. If we know the force(s) and want to find the trajectory, this is **forward dynamics**. If instead, we know the trajectory and want to find the force(s), this is **inverse dynamics**:

        <figure><img src="./../images/dynamics.png" alt="Forward and inverse dynamics." width=220/><figcaption><i><center>Figure. The equation of motion and the forward and inverse dynamics approaches.</center></i></figcaption></figure>

        In Biomechanics, in a typical movement analysis of the human body using inverse dynamics, we would measure the positions of the segments and measure the external forces, calculate the segments' linear and angular acceleration, and find the internal net force and moment of force at the joint using the equations of motion. In addition, we could estimate the muscle forces (if we solve the redundancy problem of having more muscles than joints).  
        Using forward dynamics, the muscle forces would be the inputs and the trajectories of the segments would be the outputs. The figure below compares the forward and inverse dynamics approaches.

        <figure><img src="./../images/InvDirDyn.png" alt="Direct and inverse dynamics."/><figcaption><i><center>Figure. Inverse dynamics and Forward (or Direct) dynamics approaches for movement analysis (adapted from Zajac and Gordon, 1989).</center></i></figcaption></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Estimation of joint force and moments of force by inverse dynamics

        Let's estimate the joint force and moments of force at the lower limb during locomotion using the inverse dynamics approach.   
        We will model the lower limbs at the right side as composed by three rigid bodies (foot, leg, and thigh) articulated by three hinge joints (ankle, knee, and hip) and perform a two-dimensional analysis.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Free body diagrams

        The [free body diagrams](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/FreeBodyDiagram.ipynb) of the lower limbs are:  
        <br>
        <figure><img src="./../images/fbdgaitb.png" width=640 alt="FBD lowerlimb"/><figcaption><center><i>Figure. Free body diagrams of the lower limbs for a gait analysis. <b>GRF</b> is the resultant ground reaction force applied on the foot at the center of pressure, <b>COP</b>, position.</i></center></figcaption></figure>    
  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
  
        ### Equations of motion
    
        The equilibrium equations for the forces and moments of force around the center of mass are:   

        For body 1 (foot):$\begin{array}{l l}
        \mathbf{F}_1 + m_1\mathbf{g} + \mathbf{GRF} = m_1\mathbf{a}_1 \\
        \mathbf{M}_1 + \mathbf{r}_{cmp1}\times\mathbf{F}_1 + \mathbf{r}_{cmCOP}\times\mathbf{GRF} = I_1\mathbf{\alpha}_1 
        
        \end{array}$For body 2 (leg):$\begin{array}{l}
        \mathbf{F}_2 + m_2\mathbf{g} - \mathbf{F}_1 = m_2\mathbf{a}_2 \\
        \mathbf{M}_2 + \mathbf{r}_{cmp2}\times\mathbf{F}_2 + \mathbf{r}_{cmd2}\times-\mathbf{F}_{1} - \mathbf{M}_1 = I_2\mathbf{\alpha}_2
        
        \end{array}$For body 3 (thigh):$\begin{array}{l l}
        \mathbf{F}_3 + m_3\mathbf{g} - \mathbf{F}_2 = m_3\mathbf{a}_3 \\
        \mathbf{M}_3 + \mathbf{r}_{cmp3}\times\mathbf{F}_3 + \mathbf{r}_{cmd3}\times-\mathbf{F}_{2} - \mathbf{M}_2 = I_3\mathbf{\alpha}_3
        
        \end{array}$Where$p$and$d$stands for proximal and distal joints (with respect to the fixed extremity),$\mathbf{r}_{cmji}$is the position vector from the center of mass of body$i$to the joint$j$,$COP$is the center of pressure, the position of application of the resultant ground reaction force (GRF),$\mathbf{\alpha}$is the angular acceleration, and$g$is the acceleration of gravity

        Note that the pattern of the equations is the same for the three segments: distal and proximal forces and moments of force and the weight force are present in all segments.   
        The only exception is with the foot in contact with the ground. As the ground only pushes the foot, it can not generate a moment of force over the foot. Because of that we model the interaction foot-ground as a resultant ground reaction force (GRF) applied on the foot at the COP position.   

        Both GRF and COP quantities are measured with a force platform and are assumed as known quantities.    
        Because of that the system of equations above is only solvable if we start by the body 1, from bottom to top.   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The system of equations above is simple and straightforward to solve, it is just a matter of being systematic.   
        We start by segment 1, find$\mathbf{F}_1$and$\mathbf{M}_1$, substitute these values on the equations for segment 2, find$\mathbf{F}_2$and$\mathbf{M}_2$, substitute them in the equations for segment 3 and find$\mathbf{F}_3$and$\mathbf{M}_3\:$:  

        For body 1 (foot):$\begin{array}{l l}
        \mathbf{F}_1 &=& m_1\mathbf{a}_1 - m_1\mathbf{g} - \mathbf{GRF} \\
        \mathbf{M}_1 &=& I_1\mathbf{\alpha}_1 - \mathbf{r}_{cmp1}\times\big(m_1\mathbf{a}_1 - m_1\mathbf{g} - \mathbf{GRF}\big) - \mathbf{r}_{cmCOP}\times\mathbf{GRF}
        
        \end{array}$For body 2 (leg):$\begin{array}{l l}
        \mathbf{F}_2 &=& m_1\mathbf{a}_1 + m_2\mathbf{a}_2 - (m_1+m_2)\mathbf{g} - \mathbf{GRF} \\
        \mathbf{M}_2 &=& I_1\mathbf{\alpha}_1 + I_2\mathbf{\alpha}_2 - \mathbf{r}_{cmp2}\times\big(m_1\mathbf{a}_1 + m_2\mathbf{a}_2 - (m_1+m_2)\mathbf{g} - \mathbf{GRF}\big)\, + \\
        &\phantom{=}& \mathbf{r}_{cgd2}\times\big(m_1\mathbf{a}_1 - m_1\mathbf{g} - \mathbf{GRF}\big) - \mathbf{r}_{cmp1}\times\big(m_1\mathbf{a}_1 - m_1\mathbf{g} - \mathbf{GRF}\big)\, - \\
        &\phantom{=}& \mathbf{r}_{cmCOP}\times\mathbf{GRF}
        
        \end{array}$For body 3 (thigh):$\begin{array}{l l}
        \mathbf{F}_3 &=& m_1\mathbf{a}_1 + m_2\mathbf{a}_2 + m_3\mathbf{a}_3 - (m_1+m_2+m_3)\mathbf{g} - \mathbf{GRF} \\
        \mathbf{M}_3 &=& I_1\mathbf{\alpha}_1 + I_2\mathbf{\alpha}_2 + I_3\mathbf{\alpha}_3\, - \\
        &\phantom{=}& \mathbf{r}_{cmp3}\times\big(m_1\mathbf{a}_1 + m_2\mathbf{a}_2 + m_3\mathbf{a}_3\, - (m_1+m_2+m_3)\mathbf{g} - \mathbf{GRF}\big)\, + \\
        &\phantom{=}& \mathbf{r}_{cmd3}\times\big(m_1\mathbf{a}_1 + m_2\mathbf{a}_2 - (m_1+m_2)\mathbf{g} - \mathbf{GRF}\big)\, - \\
        &\phantom{=}& \mathbf{r}_{cmp2}\times\big(m_1\mathbf{a}_1 + m_2\mathbf{a}_2 - (m_1+m_2)\mathbf{g} - \mathbf{GRF}\big)\, + \\
        &\phantom{=}& \mathbf{r}_{cmd2}\times\big(m_1\mathbf{a}_1 - m_1\mathbf{g} - \mathbf{GRF}\big)\, - \\
        &\phantom{=}& \mathbf{r}_{cmp1}\times\big(m_1\mathbf{a}_1 - m_1\mathbf{g} - \mathbf{GRF}\big)\, - \\
        &\phantom{=}& \mathbf{r}_{cmCOP}\times\mathbf{GRF}
        
        \end{array}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The recursive approach for inverse dynamics of multi-body systems

        The calculation above is tedious, error prone, useless, and probably it's wrong.  

        To make some use of it, we can clearly see that forces act on far segments, which are not directly in contact with these forces. In fact, this is true for all stuff happening on a segment: note that$\mathbf{F}_1$and$\mathbf{M}_1$are present in the expression for$\mathbf{F}_3$and$\mathbf{M}_3$and that the acceleration of segment 1 matters for the calculations of segment 3.   

        Instead, we can use the power of computer programming (like this one right now!) and solve these equations recursively hence they have the same pattern. Let's do that.

        For body 1 (foot):$\begin{array}{l l}
        \mathbf{F}_1 = m_1\mathbf{a}_1 - m_1\mathbf{g} -  \mathbf{GRF} \\
        \mathbf{M}_1 = I_1\mathbf{\alpha}_1 - \mathbf{r}_{cmp1}\times\mathbf{F}_1 - \mathbf{r}_{cmCOP}\times\mathbf{GRF}
        
        \end{array}$For body 2 (leg):$\begin{array}{l}
        \mathbf{F}_2  = m_2\mathbf{a}_2 -  m_2\mathbf{g} + \mathbf{F}_1\\
        \mathbf{M}_2 = I_2\mathbf{\alpha}_2 - \mathbf{r}_{cmp2}\times\mathbf{F}_2 +\mathbf{r}_{cmd2}\times\mathbf{F}_{1} + \mathbf{M}_1
        
        \end{array}$For body 3 (thigh):$\begin{array}{l l}
        \mathbf{F}_3  = m_3\mathbf{a}_3 - m_3\mathbf{g} + \mathbf{F}_2\\
        \mathbf{M}_3  = I_3\mathbf{\alpha}_3 - \mathbf{r}_{cmp3}\times\mathbf{F}_3 + \mathbf{r}_{cmd3}\times\mathbf{F}_{2} + \mathbf{M}_2
        
        \end{array}$"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Python function `invdyn2d.py`

        We could write a function that it would have as inputs the body-segment parameters, the kinematic data, and the distal joint force and moment of force and output the proximal joint force and moment of force.   
        Then, we would call this function for each segment, starting with the segment that has a free extremity or that has the force and moment of force measured by some instrument (i,e, use a force plate for the foot-ground interface).  
        This function would be called in the following manner:   

        ```python
            Fp, Mp = invdyn2d(rcm, rd, rp, acm, alfa, mass, Icm, Fd, Md)
        ```   

        So, here is such function:
        """
    )
    return


@app.cell
def _(cross):
    # %load ./../functions/invdyn2d.py
    """Two-dimensional inverse-dynamics calculations of one segment."""

    __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    __version__ = 'invdyn2d.py v.2 2015/11/13'


    def invdyn2d(rcm, rd, rp, acm, alpha, mass, Icm, Fd, Md):
        """Two-dimensional inverse-dynamics calculations of one segment

        Parameters
        ----------
        rcm   : array_like [x,y]
                center of mass position (y is vertical)
        rd    : array_like [x,y]
                distal joint position
        rp    : array_like [x,y]
                proximal joint position
        acm   : array_like [x,y]
                center of mass acceleration
        alpha : array_like [x,y]
                segment angular acceleration
        mass  : number
                mass of the segment   
        Icm   : number
                rotational inertia around the center of mass of the segment
        Fd    : array_like [x,y]
                force on the distal joint of the segment
        Md    : array_like [x,y]
                moment of force on the distal joint of the segment
    
        Returns
        -------
        Fp    : array_like [x,y]
                force on the proximal joint of the segment (y is vertical)
        Mp    : array_like [x,y]
                moment of force on the proximal joint of the segment

        Notes
        -----
        To use this function recursevely, the outputs [Fp, Mp] must be inputed as 
        [-Fp, -Mp] on the next call to represent [Fd, Md] on the distal joint of the
        next segment (action-reaction).
    
        This code was inspired by a similar code written by Ton van den Bogert [1]_.
        See this notebook [2]_.

        References
        ----------
        .. [1] http://isbweb.org/data/invdyn/index.html
        .. [2] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/GaitAnalysis2D.ipynb

        """
    
     
        g = 9.80665  # m/s2, standard acceleration of free fall (ISO 80000-3:2006)
        # Force and moment of force on the proximal joint
        Fp = mass*acm - Fd - [0, -g*mass]
        Mp = Icm*alpha - Md - cross(rd-rcm, Fd) - cross(rp-rcm, Fp)
    
        return Fp, Mp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The inverse dynamics calculations are implemented in only two lines of code at the end, the first part of the code is the help on how to use the function. The help is long because it's supposed to be helpful :), see the [style guide for NumPy/SciPy documentation](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). 

        The real problem is to measure or estimate the experimental variables: the body-segment parameters, the ground reaction forces, and the kinematics of each segment. For such, it is necessary some expensive equipments, but they are typical in a biomechanics laboratory, such the the [BMClab](http://pesquisa.ufabc.edu.br/bmclab).   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Experimental data

        Let's work with some data of kinematic position of the segments and ground reaction forces in order to compute the joint forces and moments of force.   
        The data we will work are in fact from a computer simulation of running created by Ton van den Bogert. The nice thing about these data is that as a simulation, the true joint forces and moments of force are known and we will be able to compare our estimation with these true values.   
        All the data can be downloaded from a page at the [ISB website](http://isbweb.org/data/invdyn/index.html):
        """
    )
    return


@app.cell
def _(IFrame):
    IFrame('https://isbweb.org/data/invdyn/index.html', width='100%', height=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Load data file
        """
    )
    return


@app.cell
def _(np):
    # load file with ground reaction force data
    grf = np.loadtxt('./../data/all.frc') # [Fx, Fy, COPx]
    # load file with kinematic data
    kin = np.loadtxt('./../data/all.kin') # [Hip(x,y), knee(x,y), ankle(x,y), toe(x,y)] 
    freq = 10000
    time = np.linspace(0, grf.shape[0]/freq, grf.shape[0])
    return freq, grf, kin, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Data filtering
        """
    )
    return


@app.cell
def _(freq, grf, kin, np):
    # this is simulated data with no noise, filtering doesn't matter
    if False:
        # filter data
        from scipy.signal import butter, filtfilt
        # Butterworth filter
        b, a = butter(2, (10/(freq/2)))
        for col in np.arange(grf.shape[1]-1):
            grf[:, col]  = filtfilt(b, a, grf[:, col])
        b, a = butter(2, (10/(freq/2)))
        for col in np.arange(kin.shape[1]):    
            kin[:, col] = filtfilt(b, a, kin[:, col])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Data selection
        """
    )
    return


@app.cell
def _(freq, grf, kin, time):
    time_1 = time[3001 - int(freq / 40):-int(freq / 20)]
    grf_1 = grf[3001 - int(freq / 40):-int(freq / 20), :]
    kin_1 = kin[3001 - int(freq / 40):-int(freq / 20), :]
    return grf_1, kin_1, time_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Plot file data
        """
    )
    return


@app.cell
def _(grf_1, kin_1, plt, time_1):
    (_hfig, hax) = plt.subplots(2, 2, sharex=True, squeeze=True, figsize=(9, 5))
    hax[0, 0].plot(time_1, grf_1[:, [0, 1]], linewidth=2)
    hax[0, 0].legend(('Fx', 'Fy'), frameon=False)
    hax[0, 0].set_ylabel('Force [N]')
    hax[0, 1].plot(time_1, grf_1[:, 2], linewidth=2)
    hax[0, 1].legend(['COPx'], frameon=False)
    hax[0, 1].set_ylabel('Amplitude [m]')
    hax[1, 0].plot(time_1, kin_1[:, 0::2], linewidth=2)
    hax[1, 0].legend(('Hip x', 'Knee x', 'Ankle x', 'Toe x'), frameon=False)
    hax[1, 0].set_ylabel('Amplitude [m]')
    hax[1, 1].plot(time_1, kin_1[:, 1::2], linewidth=2)
    hax[1, 1].legend(('Hip y', 'Knee y', 'Ankle y', 'Toe y'), frameon=False)
    hax[1, 1].set_ylabel('Amplitude [m]')
    (hax[1, 0].set_xlabel('Time [s]'), hax[1, 1].set_xlabel('Time [s]'))
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Body-segment parameters 
        """
    )
    return


@app.cell
def _():
    # body-segment parameters [thigh, shank, foot]
    mass = [6.85, 2.86, 1.00]                 # mass [kg]
    Icm  = [0.145361267, 0.042996389, 0.0200] # rotational inertia [kgm2]
    cmpr = [0.4323725, 0.4334975, 0.0]        # CM [m] wrt. prox. joint [frac. segment len]
    return Icm, cmpr, mass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Kinematic calculations
        """
    )
    return


@app.cell
def _(cmpr, freq, kin_1, np):
    rcm = np.hstack((kin_1[:, [0, 1]] + cmpr[0] * (kin_1[:, [2, 3]] - kin_1[:, [0, 1]]), kin_1[:, [2, 3]] + cmpr[1] * (kin_1[:, [4, 5]] - kin_1[:, [2, 3]]), kin_1[:, [4, 5]] + cmpr[2] * (kin_1[:, [6, 7]] - kin_1[:, [4, 5]])))
    acm = np.diff(rcm, n=2, axis=0) * freq * freq
    acm = np.vstack((acm, acm[-1, :], acm[-1, :]))
    ang = np.vstack((np.arctan2(kin_1[:, 1] - kin_1[:, 3], kin_1[:, 0] - kin_1[:, 2]), np.arctan2(kin_1[:, 3] - kin_1[:, 5], kin_1[:, 2] - kin_1[:, 4]), np.arctan2(kin_1[:, 5] - kin_1[:, 7], kin_1[:, 4] - kin_1[:, 6]))).T
    angj = np.vstack((-(ang[:, 0] - ang[:, 1]), np.unwrap(ang[:, 1] - ang[:, 2] + np.pi / 2))).T * 180 / np.pi
    aang = np.diff(ang, n=2, axis=0) * freq * freq
    aang = np.vstack((aang, aang[-1, :], aang[-1, :]))
    return aang, acm, angj, rcm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Plot joint angles
        """
    )
    return


@app.cell
def _(angj, plt, time_1):
    (_hfig, (hax1, hax2)) = plt.subplots(2, 1, sharex=True, squeeze=True, figsize=(10, 5))
    hax1.plot(time_1, angj[:, 0], linewidth=2, label='Knee')
    (hax1.legend(frameon=False, loc='upper left'), hax1.grid())
    hax2.plot(time_1, angj[:, 1], linewidth=2, label='Ankle')
    (hax2.legend(frameon=False, loc='upper left'), hax2.grid())
    hax1.set_ylabel('Joint angle$[^o]$')
    hax2.set_ylabel('Joint angle$[^o]$')
    hax2.set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Inverse dynamics calculations
        """
    )
    return


@app.cell
def _(Icm, aang, acm, grf_1, invdyn2d_1, kin_1, mass, np, rcm):
    from invdyn2d import invdyn2d
    (Fa, Ma) = invdyn2d_1(rcm[:, (4, 5)], grf_1[:, (2, 2)] * [1, 0], kin_1[:, (4, 5)], acm[:, (4, 5)], aang[:, 2], mass[2], Icm[2], grf_1[:, (0, 1)], 0)
    (Fk, Mk) = invdyn2d_1(rcm[:, (2, 3)], kin_1[:, (4, 5)], kin_1[:, (2, 3)], acm[:, (2, 3)], aang[:, 1], mass[1], Icm[1], -Fa, -Ma)
    (Fh, Mh) = invdyn2d_1(rcm[:, (0, 1)], kin_1[:, (2, 3)], kin_1[:, (0, 1)], acm[:, (0, 1)], aang[:, 0], mass[0], Icm[0], -Fk, -Mk)
    Fam = np.sqrt(np.sum(np.abs(Fa) ** 2, axis=-1))
    Fkm = np.sqrt(np.sum(np.abs(Fk) ** 2, axis=-1))
    Fhm = np.sqrt(np.sum(np.abs(Fh) ** 2, axis=-1))
    return Fam, Fhm, Fkm, Ma, Mh, Mk


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Load files with true joint forces and moments of force
        """
    )
    return


@app.cell
def _(freq, np):
    # load file with true joint forces and moments of force
    forces  = np.loadtxt('./../data/all.fmg') # [Hip, knee, ankle]
    moments = np.loadtxt('./../data/all.mom') # [Hip, knee, ankle]
    #heel strike occurs at sample 3001
    forces  = forces[3001-int(freq/40):-int(freq/20), :]
    moments = moments[3001-int(freq/40):-int(freq/20), :]
    return forces, moments


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Plot calculated variables and their true values

        Let's plot these data but because later we will need to plot similar plots, let's create a function for the plot to avoid repetition of code:
        """
    )
    return


@app.cell
def _(Fam, Fhm, Fkm, Ma, Mh, Mk, forces, moments, plt, time_1):
    def plotdata(time, Fh, Fk, Fa, Mh, Mk, Ma, forces, moments, stitle):
        (_hfig, hax) = plt.subplots(3, 2, sharex=True, squeeze=True, figsize=(11, 6))
        (hax[0, 0].plot(time, Fh, label='invdyn'), hax[0, 0].set_title('Hip'))
        (hax[1, 0].plot(time, Fk), hax[1, 0].set_title('Knee'))
        (hax[2, 0].plot(time, Fa), hax[2, 0].set_title('Ankle'))
        hax[1, 0].set_ylabel('Joint force [N]')
        hax[2, 0].set_xlabel('Time [s]')
        (hax[0, 1].plot(time, Mh), hax[0, 1].set_title('Hip'))
        (hax[1, 1].plot(time, Mk), hax[1, 1].set_title('Knee'))
        (hax[2, 1].plot(time, Ma), hax[2, 1].set_title('Ankle'))
        hax[1, 1].set_ylabel('Moment of Force [Nm]')
        hax[2, 1].set_xlabel('Time [s]')
        hax[0, 0].plot(time, forces[:, 0], 'r--', label='True')
        hax[0, 0].legend(frameon=False)
        hax[1, 0].plot(time, forces[:, 1], 'r--')
        hax[2, 0].plot(time, forces[:, 2], 'r--')
        hax[0, 1].plot(time, moments[:, 0], 'r--')
        hax[1, 1].plot(time, moments[:, 1], 'r--')
        hax[2, 1].plot(time, moments[:, 2], 'r--')
        plt.suptitle(stitle, fontsize=16)
        for x in hax.flat:
            x.locator_params(nbins=5)
            x.grid()
        plt.show()
    plotdata(time_1, Fhm, Fkm, Fam, Mh, Mk, Ma, forces, moments, 'Inverse dynamics: estimated versus true values')
    return (plotdata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The results are very similar; only a small part of the moments of force is different because of some noise.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Contribution of each term to the joint force and moment of force 

        Let's see what happens with the joint forces and moments of force when we neglect the contribution of some terms in the inverse dynamics analysis of these data. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Quasi-static analysis
        Consider the case where the segment acceleration is neglected:
        """
    )
    return


@app.cell
def _(
    Icm,
    aang,
    acm,
    forces,
    grf_1,
    invdyn2d_1,
    kin_1,
    mass,
    moments,
    np,
    plotdata,
    rcm,
    time_1,
):
    (Fast, Mast) = invdyn2d_1(rcm[:, (4, 5)], grf_1[:, (2, 2)] * [1, 0], kin_1[:, (4, 5)], acm[:, (4, 5)] * 0, aang[:, 2] * 0, mass[2], Icm[2], grf_1[:, (0, 1)], 0)
    (Fkst, Mkst) = invdyn2d_1(rcm[:, (2, 3)], kin_1[:, (4, 5)], kin_1[:, (2, 3)], acm[:, (2, 3)] * 0, aang[:, 1] * 0, mass[1], Icm[1], -Fast, -Mast)
    (Fhst, Mhst) = invdyn2d_1(rcm[:, (0, 1)], kin_1[:, (2, 3)], kin_1[:, (0, 1)], acm[:, (0, 1)] * 0, aang[:, 0] * 0, mass[0], Icm[0], -Fkst, -Mkst)
    Fastm = np.sqrt(np.sum(np.abs(Fast) ** 2, axis=-1))
    Fkstm = np.sqrt(np.sum(np.abs(Fkst) ** 2, axis=-1))
    Fhstm = np.sqrt(np.sum(np.abs(Fhst) ** 2, axis=-1))
    plotdata(time_1, Fhstm, Fkstm, Fastm, Mhst, Mkst, Mast, forces, moments, 'Inverse dynamics: quasis-static approach versus true values')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is not a pure static analysis because part of the ground reaction forces still reflects the body accelerations (were the body completely static, the ground reaction force should be equal to the body weight in magnitude).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Neglecting the acceleration and mass (weight) of the segments

        Consider the case where besides the acceleration, the body-segment parameters are also neglected.   
        This means that the joint loads are due only to the ground reaction forces (which implicitly include contributions due to the acceleration and the body-segment weights).  
        """
    )
    return


@app.cell
def _(
    aang,
    acm,
    forces,
    grf_1,
    invdyn2d_1,
    kin_1,
    moments,
    np,
    plotdata,
    rcm,
    time_1,
):
    (Fagrf, Magrf) = invdyn2d_1(rcm[:, (4, 5)], grf_1[:, (2, 2)] * [1, 0], kin_1[:, (4, 5)], acm[:, (4, 5)] * 0, aang[:, 2] * 0, 0, 0, grf_1[:, (0, 1)], 0)
    (Fkgrf, Mkgrf) = invdyn2d_1(rcm[:, (2, 3)], kin_1[:, (4, 5)], kin_1[:, (2, 3)], acm[:, (2, 3)] * 0, aang[:, 1] * 0, 0, 0, -Fagrf, -Magrf)
    (Fhgrf, Mhgrf) = invdyn2d_1(rcm[:, (0, 1)], kin_1[:, (2, 3)], kin_1[:, (0, 1)], acm[:, (0, 1)] * 0, aang[:, 0] * 0, 0, 0, -Fkgrf, -Mkgrf)
    Fagrfm = np.sqrt(np.sum(np.abs(Fagrf) ** 2, axis=-1))
    Fkgrfm = np.sqrt(np.sum(np.abs(Fkgrf) ** 2, axis=-1))
    Fhgrfm = np.sqrt(np.sum(np.abs(Fhgrf) ** 2, axis=-1))
    plotdata(time_1, Fhgrfm, Fkgrfm, Fagrfm, Mhgrf, Mkgrf, Magrf, forces, moments, 'Inverse dynamics: ground-reaction-force approach versus true values')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Neglecting all the accelerations and the weight of the segments means that the only external force that actuates on the system is the ground reaction force, which although is only actuating at the foot-ground interface it will be transmitted to the other segments through the joint forces.  Because of that, the joint forces on the ankle, knee, and hip will simply be minus the ground reaction force. Note that the forces shown above for the three joints are the same and equal to:$\begin{array}{l}
        \sqrt{GRF_x^2+GRF_y^2}
        
        \end{array}$These simplifications also mean that the moments of force could have been simply calculated as the cross product between the vector position of the the COP in relation to the joint and the GRF vector:$\begin{array}{l}
        \mathbf{M_{a}} = -\mathbf{cross}(\mathbf{COP}-\mathbf{r_{a}},\,\mathbf{GRF}) \\
        \mathbf{M_{k}} = -\mathbf{cross}(\mathbf{COP}-\mathbf{r_{k}},\,\mathbf{GRF}) \\
        \mathbf{M_{h}} = -\mathbf{cross}(\mathbf{COP}-\mathbf{r_{h}},\,\mathbf{GRF})
        
        \end{array}$Where$\mathbf{r_{i}}\;$is the position vector of joint$i$.

        Let's calculate the variables in this way:
        """
    )
    return


@app.cell
def _(forces, grf_1, kin_1, moments, np, plotdata, time_1):
    Fhgrfm2 = Fkgrfm2 = Fagrfm2 = np.sqrt(np.sum(np.abs(-grf_1[:, (0, 1)]) ** 2, axis=-1))
    Magrf2 = -np.cross(grf_1[:, (2, 2)] * [1, 0] - kin_1[:, (4, 5)], grf_1[:, (0, 1)])
    Mkgrf2 = -np.cross(grf_1[:, (2, 2)] * [1, 0] - kin_1[:, (2, 3)], grf_1[:, (0, 1)])
    Mhgrf2 = -np.cross(grf_1[:, (2, 2)] * [1, 0] - kin_1[:, (0, 1)], grf_1[:, (0, 1)])
    plotdata(time_1, Fhgrfm2, Fkgrfm2, Fagrfm2, Mhgrf2, Mkgrf2, Magrf2, forces, moments, 'Inverse dynamics: ground-reaction-force approach versus true values II')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### WARNING: the calculated resultant joint force is not the actual joint reaction force!

        In the Newton-Euler equations based on the free body diagrams we represented the consequences of all possible muscle forces on a joint as a net muscle torque and all forces acting on a joint as a resultant joint reaction force. That is, all forces between segments were represented as a resultant force that doesn't generate torque and a moment of force that only generates torque.  
        This is an important principle in mechanics of rigid bodies as we saw before.  
        However, this principle creates the unrealistic notion that the sum of forces is applied directly on the joint (which has no further implication for a rigid body), but it is inaccurate for the understanding of the local effects on the joint. So, if we are trying to understand the stress on the joint or mechanisms of joint injury, the forces acting on the joint and on the rest of the segment must be considered individually.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Conclusion

        For these data set of 'running' (remember this is simulated data), in the estimation of the forces and moments of force at the hip, knee, and ankle joints in a two-dimensional analysis, to not consider the segment acceleration and/or the mass of the segments had no effect on the ankle variables, a small effect on the knee, and a large effect on the hip.    
        This is not surprising; during the support phase, ankle and knee have small movements and the mass of the segments only start to have a significant contribution for more proximal and heavy segments such as the thigh.   

        Don't get disappointed thinking that all this work for drawing the complete FBDs and their correspondent equations was a waste of time.    
        Nowadays, the state of the art and the demand for higher accuracy in biomechanics is such that such simplifications are usually not accepted.   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        - [Gait Analysis on Wikipedia](https://en.wikipedia.org/wiki/Gait_analysis)  
        - [Gait analysis: clinical facts](https://www.ncbi.nlm.nih.gov/pubmed/27618499)  
        - [Gait Analysis Methods: An Overview of Wearable and Non-Wearable Systems, Highlighting Clinical Applications](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3958266/)  
        - [Avaliação Biomecânica da Corrida no BMClab (in Portuguese)](http://pesquisa.ufabc.edu.br/bmclab/servicos/rba-2/)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

        - Understanding & Analyzing Gait For The Clinician - series: [Introduction](https://youtu.be/x1JoaGgyKX0), [Patient Assessment](https://youtu.be/Z0QNkLshQUk), [Intro To Computer-Based 3-D Analysis](https://youtu.be/g0OcCLTQM_Y), [Basic Musculoskeletal Biomechanics](https://youtu.be/KsdrmyxOyxM), [The Gait Cycle](https://youtu.be/96nLX6sm9Yw)  
        - [How to benefit from a Gait Analysis | Runners Need](https://youtu.be/rxkX7qGtIEI)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Search the Internet for actual experimental data from a gait analysis of a runner and compare with the simulated data used in this notebook.  
        2. Collect or search for some experimental data from a movement analysis and perform inverse dynamics to determine joint forces and torques.  
        3. Imagine that you have to perform a similar analysis but of the upper limb during throwing a ball. What would have to change in the approach described in this notebook?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.    
        - Winter DA (2009) [Biomechanics and motor control of human movement](http://books.google.com.br/books?id=_bFHL08IWfwC). 4 ed. Hoboken, EUA: Wiley. 
        - Zajac FE, Gordon ME (1989) [Determining muscle's force and action in multi-articular movement](https://github.com/BMClab/BMC/blob/master/refs/zajac89.pdf). Exercise and Sport Sciences Reviews, 17, 187-230.  
        - Zatsiorsky VM (2202) [Kinetics of human motion](http://books.google.com.br/books?id=wp3zt7oF8a0C&lpg=PA571&ots=Kjc17DAl19&dq=ZATSIORSKY%2C%20Vladimir%20M.%20Kinetics%20of%20human%20motion&hl=pt-BR&pg=PP1#v=onepage&q&f=false). Champaign, IL: Human Kinetics.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
