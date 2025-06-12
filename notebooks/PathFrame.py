import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Path frame

        > Renato Naville Watanabe, Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Cartesian-coordinate-system" data-toc-modified-id="Cartesian-coordinate-system-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Cartesian coordinate system</a></span></li><li><span><a href="#Determination-of-a-coordinate-system" data-toc-modified-id="Determination-of-a-coordinate-system-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Determination of a coordinate system</a></span></li><li><span><a href="#Time-varying-basis" data-toc-modified-id="Time-varying-basis-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Time varying basis</a></span><ul class="toc-item"><li><span><a href="#Tangential-versor" data-toc-modified-id="Tangential-versor-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Tangential versor</a></span></li><li><span><a href="#Normal-versor" data-toc-modified-id="Normal-versor-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Normal versor</a></span></li><li><span><a href="#Binormal-versor" data-toc-modified-id="Binormal-versor-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Binormal versor</a></span></li></ul></li><li><span><a href="#Velocity-and-Acceleration-in-a-time-varying-frame" data-toc-modified-id="Velocity-and-Acceleration-in-a-time-varying-frame-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Velocity and Acceleration in a time-varying frame</a></span><ul class="toc-item"><li><span><a href="#Velocity" data-toc-modified-id="Velocity-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Velocity</a></span></li><li><span><a href="#Acceleration" data-toc-modified-id="Acceleration-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Acceleration</a></span></li></ul></li><li><span><a href="#Example" data-toc-modified-id="Example-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Example</a></span><ul class="toc-item"><li><span><a href="#Solving-numerically" data-toc-modified-id="Solving-numerically-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Solving numerically</a></span></li><li><span><a href="#Symbolic-solution-(extra-reading)" data-toc-modified-id="Symbolic-solution-(extra-reading)-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Symbolic solution (extra reading)</a></span></li></ul></li><li><span><a href="#Further-Reading" data-toc-modified-id="Further-Reading-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Further Reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>References</a></span></li></ul></div>
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
    import numpy as np
    import sympy as sym
    from sympy.vector import CoordSys3D
    import matplotlib.pyplot as plt
    sym.init_printing()
    from sympy.plotting import plot_parametric
    from sympy.physics.mechanics import ReferenceFrame, Vector, dot
    from matplotlib.patches import FancyArrowPatch
    plt.rcParams.update({'figure.figsize':(8, 5), 'lines.linewidth':2})
    return FancyArrowPatch, np, plot_parametric, plt, sym


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Cartesian coordinate system

        As we perceive the surrounding space as three-dimensional, a convenient coordinate system is the [Cartesian coordinate system](http://en.wikipedia.org/wiki/Cartesian_coordinate_system) in the [Euclidean space](http://en.wikipedia.org/wiki/Euclidean_space) with three orthogonal axes as shown below. The axes directions are commonly defined by the [right-hand rule](http://en.wikipedia.org/wiki/Right-hand_rule) and attributed the letters X, Y, Z. The orthogonality of the Cartesian coordinate system is convenient for its use in classical mechanics, most of the times the structure of space is assumed having the [Euclidean geometry](http://en.wikipedia.org/wiki/Euclidean_geometry) and as consequence, the motion in different directions are independent of each other.  
        <br>
        <figure><img src="./../images/CCS.png" width=350/><figcaption><center><i>Figure. Representation of a point and its position vector in a Cartesian coordinate system.</i></center></figcaption></figure>  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Determination of a coordinate system

        In Biomechanics, we may use different coordinate systems for convenience and refer to them as global, laboratory, local, anatomical, or technical reference frames or coordinate systems. 

        From [linear algebra](http://en.wikipedia.org/wiki/Linear_algebra), a set of unit linearly independent vectors (orthogonal in the Euclidean space and each with norm (length) equals to one) that can represent any vector via [linear combination](http://en.wikipedia.org/wiki/Linear_combination) is called a <a href="http://en.wikipedia.org/wiki/Basis_(linear_algebra)">basis</a> (or **orthonormal basis**). The figure below shows a point and its position vector in the Cartesian coordinate system and the corresponding versors (**unit vectors**) of the basis for this coordinate system. See the notebook [Scalar and vector](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/ScalarVector.ipynb) for a description on vectors.  
        <figure><img src="./../images/vector3Dijk.png" width=350/><figcaption><center><i>Figure. Representation of a point$\mathbf{P}$and its position vector$\overrightarrow{\mathbf{r}}$in a Cartesian coordinate system. The versors <span class="notranslate">$\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}\,$</span> form a basis for this coordinate system and are usually represented in the color sequence RGB (red, green, blue) for easier visualization.</i></center></figcaption></figure>  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One can see that the versors of the basis shown in the figure above have the following coordinates in the Cartesian coordinate system:
        <br>  
        <span class="notranslate">$\hat{\mathbf{i}} = \begin{bmatrix}1\\0\\0 \end{bmatrix}, \quad \hat{\mathbf{j}} = \begin{bmatrix}0\\1\\0 \end{bmatrix}, \quad \hat{\mathbf{k}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$</span>

        Using the notation described in the figure above, the position vector$\overrightarrow{\mathbf{r}}$can be expressed as:
        <br>  
        <span class="notranslate">$\overrightarrow{\mathbf{r}} = x\hat{\mathbf{i}} + y\hat{\mathbf{j}} + z\hat{\mathbf{k}}$</span>

        However, to use a fixed basis can lead to very complex expressions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Path basis
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Consider that we have the position vector of a particle, moving in the path described by the parametric curve$s(t)$, described in a fixed reference frame as:
        <br>  
        <span class="notranslate">${\bf\hat{r}}(t) = {x}{\bf\hat{i}}+{y}{\bf\hat{j}} + {z}{\bf\hat{k}}$</span>

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/velRefFrame.png?raw=1" width=500/><figcaption><center><i>Figure. Position vector of a moving particle in relation to a coordinate system.</i></center></figcaption></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Tangential versor

        Often we describe all the kinematic variables in this fixed reference frame. However, it is often useful to define a path basis, attached to some point of interest. In this case, what is usually done is to choose as one of the basis vector a unitary vector in the direction of the velocity of the particle. Defining this vector as:

        <span class="notranslate">${\bf\hat{e}_t} = \frac{{\bf\vec{v}}}{\Vert{\bf\vec{v}}\Vert}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Normal versor

        For the second vector of the basis, we define first a vector of curvature of the path (the meaning of this curvature vector will be seeing in another notebook):
        <br>  
        <span class="notranslate">${\bf\vec{C}} = \frac{d{\bf\hat{e}_t}}{ds}$</span>

        Note that$\bf\hat{e}_t$is a function of the path$s(t)$. So, by the chain rule:
        <br>  
        <span class="notranslate">$\frac{d{\bf\hat{e}_t}}{dt} = \frac{d{\bf\hat{e}_t}}{ds}\frac{ds}{dt} \longrightarrow \frac{d{\bf\hat{e}_t}}{ds} = \frac{\frac{d{\bf\hat{e}_t}}{dt}}{\frac{ds}{dt}} \longrightarrow {\bf\vec{C}} = \frac{\frac{d{\bf\hat{e}_t}}{dt}}{\frac{ds}{dt}}\longrightarrow {\bf\vec{C}} = \frac{\frac{d{\bf\hat{e}_t}}{dt}}{\Vert{\bf\vec{v}}\Vert}$</span>

        Now we can define the second vector of the basis,${\bf\hat{e}_n}$:
        <br>  
        <span class="notranslate">${\bf\hat{e}_n} = \frac{{\bf\vec{C}}}{\Vert{\bf\vec{C}}\Vert}$</span>

        <figure><img src="./../images/velRefFrameeten.png" width=500/><figcaption><center><i>Figure. A moving particle and a corresponding path basis.</i></center></figcaption></figure> 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Binormal versor

        The third vector of the basis is obtained by the cross product between${\bf\hat{e}_n}$and${\bf\hat{e}_t}$:
        <br>  
        <span class="notranslate">${\bf\hat{e}_b} = {\bf\hat{e}_t} \times {\bf\hat{e}_n}$</span>

        Note that the vectors <span class="notranslate">${\bf\hat{e}_t}$</span>, <span class="notranslate">${\bf\hat{e}_n}$</span> and <span class="notranslate">${\bf\hat{e}_b}$</span> vary together with the particle movement. This basis is also called as **Frenet-Serret frame**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Velocity and Acceleration in a path frame
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Velocity 

        Given the expression of$r(t)$in a fixed frame we can write the velocity <span class="notranslate">${\bf\vec{v}(t)}$</span> as a function of the fixed frame of reference <span class="notranslate">${\bf\hat{i}}$</span>, <span class="notranslate">${\bf\hat{j}}$</span> and <span class="notranslate">${\bf\hat{k}}$</span> (see http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/KinematicsParticle.ipynb)).
        <br>  
        <span class="notranslate">${\bf\vec{v}}(t) = \dot{x}{\bf\hat{i}}+\dot{y}{\bf\hat{j}}+\dot{z}{\bf\hat{k}}$</span>

        However, this can lead to very complex functions. So it is useful to use the basis find previously <span class="notranslate">${\bf\hat{e}_t}$</span>, <span class="notranslate">${\bf\hat{e}_n}$</span> and <span class="notranslate">${\bf\hat{e}_b}$</span>.

        The velocity <span class="notranslate">${\bf\vec{v}}$</span> of the particle is, by the definition of <span class="notranslate">${\bf\hat{e}_t}$</span>, in the direction of <span class="notranslate">${\bf\hat{e}_t}$</span>:
        <br>  
        <span class="notranslate">${\bf\vec{v}}={\Vert\bf\vec{v}\Vert}.{\bf\hat{e}_t}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Acceleration

        The acceleration can be written in the fixed frame of reference as:
        <br>  
        <span class="notranslate">${\bf\vec{a}}(t) = \ddot{x}{\bf\hat{i}}+\ddot{y}{\bf\hat{j}}+\ddot{z}{\bf\hat{k}}$</span>

        But for the same reasons of the velocity vector, it is useful to describe the acceleration vector in the path basis. We know that the acceleration is the time derivative of the velocity:
        <br>  
        <span class="notranslate">
        \begin{align}
        {\bf\vec{a}} =& \frac{{d\bf\vec{v}}}{dt}=\\
            =&\frac{{d({\Vert\bf\vec{v}\Vert}{\bf\hat{e}_t}})}{dt}=\\
            =&\dot{\Vert\bf\vec{v}\Vert}{\bf\hat{e}_t}+{\Vert\bf\vec{v}\Vert}\dot{{\bf\hat{e}_t}}=\\
            =&\dot{\Vert\bf\vec{v}\Vert}{\bf\hat{e}_t}+{\Vert\bf\vec{v}\Vert}\frac{d{\bf\hat{e}_t}}{ds}\frac{ds}{dt}=\\
            =&\dot{\Vert\bf\vec{v}\Vert}{\bf\hat{e}_t}+{\Vert\bf\vec{v}\Vert}^2\frac{d{\bf\hat{e}_t}}{ds}=\\
            =&\dot{\Vert\bf\vec{v}\Vert}{\bf\hat{e}_t}+{\Vert\bf\vec{v}\Vert}^2\Vert{\bf\vec{C}} \Vert {\bf\hat{e}_n}
        \label{eq_12}
        \end{align}
        </span>

        The inverse of the size of the curvature vector$\frac{1}{\Vert\bf\vec{C}\Vert}$is know as the curvature radius of the path$\rho$.

        <span class="notranslate">${\bf\vec{a}} =\dot{\Vert\bf\vec{v}\Vert}{\bf\hat{e}_t}+\frac{{\Vert\bf\vec{v}\Vert}^2}{\rho}{\bf\hat{e}_n}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example
        For example, consider that a particle follows the path described by the parametric curve below:
        <br>  
        <span class="notranslate">$\vec{r}(t) = (10t+100){\bf{\hat{i}}} + \left(-\frac{9,81}{2}t^2+50t+100\right){\bf{\hat{j}}}$</span>

        This curve could be, for example, from a projectile motion. See http://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/ProjectileMotion.ipynb for an explanation on projectile motion.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Solving numerically

        Now we will obtain the path basis numerically. This method is useful when it is not available a mathematical expression of the path. This often happens when you have available data collected experimentally (most of the cases in Biomechanics). 

        First, data will be obtained from the expression of$r(t)$. This is done to replicate the example above. You could use data collected experimentally, for example.
        """
    )
    return


@app.cell
def _(np):
    t = np.linspace(0, 10, 1000).reshape(-1,1)
    r = np.hstack((10*t + 100, -9.81/2*t**2 + 50*t + 100))
    return r, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, to obtain the <span class="notranslate">$\bf{\hat{e_t}}$</span> versor, we can use Equation (4).
        """
    )
    return


@app.cell
def _(np, r, t):
    dt = t[1]
    v = np.diff(r,axis=0)/dt
    vNorm  = np.linalg.norm(v, axis=1, keepdims=True)

    et = v/vNorm
    return dt, et, vNorm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And to obtain the versor <span class="notranslate">$\bf{\hat{e_n}}$</span>, we can use Equation (8).
        """
    )
    return


@app.cell
def _(dt, et, np, vNorm):
    C = np.diff(et,axis=0)/dt
    C = C/vNorm[1:]

    CNorm = np.linalg.norm(C, axis=1, keepdims=True)
    en = C/CNorm
    return CNorm, en


@app.cell
def _(FancyArrowPatch, en, et, np, plt, r, t):
    _fig = plt.figure()
    plt.plot(r[:, 0], r[:, 1], '.')
    _ax = _fig.add_axes([0, 0, 1, 1])
    _time = np.linspace(0, 10, 10)
    for _i in np.arange(0, len(t) - 2, 50):
        _vec1 = FancyArrowPatch(r[_i, :], r[_i, :] + 10 * et[_i, :], mutation_scale=20, color='r')
        _vec2 = FancyArrowPatch(r[_i, :], r[_i, :] + 10 * en[_i, :], mutation_scale=20, color='g')
        _ax.add_artist(_vec1)
        _ax.add_artist(_vec2)
    plt.xlim((80, 250))
    plt.ylim((80, 250))
    plt.legend([_vec1, _vec2], ['$\\vec{e_t}$', '$\\vec{e_{n}}$'])
    plt.show()
    return


@app.cell
def _(CNorm, dt, en, et, np, vNorm):
    v_1 = vNorm * et
    vNormDot = np.diff(vNorm, axis=0) / dt
    a = vNormDot * et[1:, :] + CNorm * en * vNorm[1:] ** 2
    return a, v_1


@app.cell
def _(FancyArrowPatch, a, plt, r, t, v_1):
    _fig = plt.figure()
    plt.plot(r[:, 0], r[:, 1], '.')
    _ax = _fig.add_axes([0, 0, 1, 1])
    for _i in range(0, len(t) - 2, 50):
        _vec1 = FancyArrowPatch(r[_i, :], r[_i, :] + v_1[_i, :], mutation_scale=10, color='r')
        _vec2 = FancyArrowPatch(r[_i, :], r[_i, :] + a[_i, :], mutation_scale=10, color='g')
        _ax.add_artist(_vec1)
        _ax.add_artist(_vec2)
    plt.xlim((80, 250))
    plt.ylim((80, 250))
    plt.legend([_vec1, _vec2], ['$\\vec{v}$', '$\\vec{a}$'])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Symbolic solution (extra reading)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The computation here will be performed symbolically, with the symbolic math package of Python, Sympy. Below,a reference frame, called O,  and a varible for time (t) are defined.
        """
    )
    return


@app.cell
def _(sym):
    O = sym.vector.CoordSys3D(' ')
    t_1 = sym.symbols('t')
    return O, t_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Below the vector$r(t)$is defined symbolically.
        """
    )
    return


@app.cell
def _(O, t_1):
    r_1 = (10 * t_1 + 100) * O.i + (-9.81 / 2 * t_1 ** 2 + 50 * t_1 + 100) * O.j + 0 * O.k
    r_1
    return (r_1,)


@app.cell
def _(O, plot_parametric, r_1, t_1):
    p1 = plot_parametric(r_1.dot(O.i), r_1.dot(O.j), (t_1, 0, 10), line_width=4, aspect_ratio=(2, 1))
    return


@app.cell
def _(r_1, sym):
    v_2 = sym.diff(r_1)
    v_2
    return (v_2,)


@app.cell
def _(sym, v_2):
    et_1 = v_2 / sym.sqrt(v_2.dot(v_2))
    et_1
    return (et_1,)


@app.cell
def _(et_1, sym, v_2):
    C_1 = sym.diff(et_1) / sym.sqrt(v_2.dot(v_2))
    C_1
    return (C_1,)


@app.cell
def _(C_1, sym):
    en_1 = C_1 / sym.sqrt(C_1.dot(C_1))
    sym.simplify(en_1)
    return (en_1,)


@app.cell
def _(FancyArrowPatch, O, en_1, et_1, np, plt, r_1, t_1):
    _fig = plt.figure()
    _ax = _fig.add_axes([0, 0, 1, 1])
    _time = np.linspace(0, 10, 30)
    for _instant in _time:
        _vt = FancyArrowPatch([float(r_1.dot(O.i).subs(t_1, _instant)), float(r_1.dot(O.j).subs(t_1, _instant))], [float(r_1.dot(O.i).subs(t_1, _instant)) + 10 * float(et_1.dot(O.i).subs(t_1, _instant)), float(r_1.dot(O.j).subs(t_1, _instant)) + 10 * float(et_1.dot(O.j).subs(t_1, _instant))], mutation_scale=20, arrowstyle='->', color='r', label='${\\hat{e_t}}$', lw=2)
        _vn = FancyArrowPatch([float(r_1.dot(O.i).subs(t_1, _instant)), float(r_1.dot(O.j).subs(t_1, _instant))], [float(r_1.dot(O.i).subs(t_1, _instant)) + 10 * float(en_1.dot(O.i).subs(t_1, _instant)), float(r_1.dot(O.j).subs(t_1, _instant)) + 10 * float(en_1.dot(O.j).subs(t_1, _instant))], mutation_scale=20, arrowstyle='->', color='g', label='${\\hat{e_n}}$', lw=2)
        _ax.add_artist(_vn)
        _ax.add_artist(_vt)
    plt.xlim((90, 250))
    plt.ylim((90, 250))
    plt.xlabel('x')
    plt.legend(handles=[_vt, _vn], fontsize=20)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can find the vectors${\bf\vec{v}}$and${\bf\vec{a}}$described in the path frame.
        """
    )
    return


@app.cell
def _(et_1, sym, v_2):
    v_3 = sym.sqrt(v_2.dot(v_2)) * et_1
    return (v_3,)


@app.cell
def _(C_1, en_1, et_1, sym, v_3):
    a_1 = sym.diff(sym.sqrt(v_3.dot(v_3))) * et_1 + v_3.dot(v_3) * sym.sqrt(C_1.dot(C_1)) * en_1
    sym.simplify(sym.simplify(a_1))
    return (a_1,)


@app.cell
def _(FancyArrowPatch, O, a_1, np, plt, r_1, t_1, v_3):
    _fig = plt.figure()
    _ax = _fig.add_axes([0, 0, 1, 1])
    _time = np.linspace(0, 10, 10)
    for _instant in _time:
        _vt = FancyArrowPatch([float(r_1.dot(O.i).subs(t_1, _instant)), float(r_1.dot(O.j).subs(t_1, _instant))], [float(r_1.dot(O.i).subs(t_1, _instant)) + float(v_3.dot(O.i).subs(t_1, _instant)), float(r_1.dot(O.j).subs(t_1, _instant)) + float(v_3.dot(O.j).subs(t_1, _instant))], mutation_scale=20, arrowstyle='->', color='r', label='${{v}}$', lw=2)
        _vn = FancyArrowPatch([float(r_1.dot(O.i).subs(t_1, _instant)), float(r_1.dot(O.j).subs(t_1, _instant))], [float(r_1.dot(O.i).subs(t_1, _instant)) + float(a_1.dot(O.i).subs(t_1, _instant)), float(r_1.dot(O.j).subs(t_1, _instant)) + float(a_1.dot(O.j).subs(t_1, _instant))], mutation_scale=20, arrowstyle='->', color='g', label='${{a}}$', lw=2)
        _ax.add_artist(_vn)
        _ax.add_artist(_vt)
    plt.xlim((60, 250))
    plt.ylim((60, 250))
    plt.legend(handles=[_vt, _vn], fontsize=20)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further Reading

         - Read pages 932-971 of the 18th chapter of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html) about path basis vectors.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

         - [Path Vectors](https://eaulas.usp.br/portal/video?idItem=7800)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Obtain the vectors <span class="notranslate">$\hat{e_n}$</span> and <span class="notranslate">$\hat{e_t}$</span> for the problem 18.1.1 from Ruina and Rudra's book.
        2. Solve the problem 18.1.9 from Ruina and Rudra's book.
        3. Write a Python program to solve the problem 18.1.10 (only the part of <span class="notranslate">$\hat{e_n}$</span> and <span class="notranslate">$\hat{e_t}$</span>).
        4. Solve the problems 1.13, 1.15, and 1.16 from the Rade's book.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        + Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press. 
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
