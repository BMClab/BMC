import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Kinematics of particle

        > Marcos Duarte, Renato Naville Watanabe  
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <div style="background-color:#F1F1F1;border:1px solid black;padding:10px;">
            <figure><img src="https://github.com/BMClab/BMC/blob/master/images/usain_bolt_berlin2009.png?raw=1" width=700 alt="Momentary velocity vs location for Usain Bolt"/><figcaption><center><br><i><b>Figure. Momentary velocity vs location for Usain Bolt in the men’s 100m final at the IAAF World Championships in Athletics, Berlin 2009. This measurement represents the velocity of the body (considered as a particle) and was measured with a laser radar. From <a href="http://www.meathathletics.ie/devathletes/pdf/Biomechanics%20of%20Sprints.pdf">Graubner and Nixdorf (2011)</a>.</b></i></center></figcaption></figure>
         </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Biomechanics-&amp;-Mechanics" data-toc-modified-id="Biomechanics-&amp;-Mechanics-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Biomechanics &amp; Mechanics</a></span></li><li><span><a href="#Kinematics" data-toc-modified-id="Kinematics-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Kinematics</a></span><ul class="toc-item"><li><span><a href="#Vectors-in-Kinematics" data-toc-modified-id="Vectors-in-Kinematics-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Vectors in Kinematics</a></span></li></ul></li><li><span><a href="#Position" data-toc-modified-id="Position-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Position</a></span></li><li><span><a href="#Basis" data-toc-modified-id="Basis-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Basis</a></span></li><li><span><a href="#Displacement" data-toc-modified-id="Displacement-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Displacement</a></span></li><li><span><a href="#Velocity" data-toc-modified-id="Velocity-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Velocity</a></span></li><li><span><a href="#Acceleration" data-toc-modified-id="Acceleration-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Acceleration</a></span></li><li><span><a href="#The-antiderivative" data-toc-modified-id="The-antiderivative-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>The antiderivative</a></span></li><li><span><a href="#Some-cases-of-motion-of-a-particle" data-toc-modified-id="Some-cases-of-motion-of-a-particle-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Some cases of motion of a particle</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Particle-at-rest" data-toc-modified-id="Particle-at-rest-10.0.1"><span class="toc-item-num">10.0.1&nbsp;&nbsp;</span>Particle at rest</a></span></li><li><span><a href="#Particle-at-constant-speed" data-toc-modified-id="Particle-at-constant-speed-10.0.2"><span class="toc-item-num">10.0.2&nbsp;&nbsp;</span>Particle at constant speed</a></span></li><li><span><a href="#Particle-at-constant-acceleration" data-toc-modified-id="Particle-at-constant-acceleration-10.0.3"><span class="toc-item-num">10.0.3&nbsp;&nbsp;</span>Particle at constant acceleration</a></span></li></ul></li><li><span><a href="#Values-for-the-mechanical-quantities" data-toc-modified-id="Values-for-the-mechanical-quantities-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Values for the mechanical quantities</a></span></li><li><span><a href="#Visual-representation-of-these-cases" data-toc-modified-id="Visual-representation-of-these-cases-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>Visual representation of these cases</a></span></li></ul></li><li><span><a href="#Symbolic-programming" data-toc-modified-id="Symbolic-programming-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Symbolic programming</a></span></li><li><span><a href="#Kinematics-of-human-movement" data-toc-modified-id="Kinematics-of-human-movement-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Kinematics of human movement</a></span><ul class="toc-item"><li><span><a href="#Kinematics-of-the-100-m-race" data-toc-modified-id="Kinematics-of-the-100-m-race-12.1"><span class="toc-item-num">12.1&nbsp;&nbsp;</span>Kinematics of the 100-m race</a></span></li></ul></li><li><span><a href="#More-examples" data-toc-modified-id="More-examples-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>More examples</a></span></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-16"><span class="toc-item-num">16&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-17"><span class="toc-item-num">17&nbsp;&nbsp;</span>References</a></span></li></ul></div>
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
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Biomechanics & Mechanics

        **A good knowledge of Mechanics is a necessary condition, although not sufficient!, to master Biomechanics**

        For this reason, we will review principles of Classical Mechanics in the context of Biomechanics.  

        The book [*Introduction to Statics and Dynamics*](http://ruina.tam.cornell.edu/Book/index.html), written by Andy Ruina and Rudra Pratap, is an excellent reference (a rigorous and yet didactic presentation of Mechanics for undergraduate students) on Classical Mechanics. The preface and first chapter of the book are a good read on how someone should study Mechanics. You should read them!

        Most of the content in this notebook is covered in chapter 12 of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html) and in chapter 1 of the Rade's book.

        As we argued in the notebook [Biomechanics](https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Biomechanics.ipynb), we will start with a branch of Classical Mechanics that is simpler to measure its related quantities on biological systems, the Kinematics.  

        There are some relevant cases in the study of human movement where modeling the human body or one of its segments as a particle might be all we need to explore the phenomenon. The concept of kinematics of a particle, for instance, can be applied to study the performance in the 100-m race; to describe spatial and temporal characteristics of a movement pattern, and to conjecture about how voluntary movements are planned (the minimum jerk hypothesis).  

        Now, let's review the concept of kinematics of a particle and later apply to the study of human movement.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Kinematics

        **Kinematics** is the branch of Classical Mechanics that describes the motion of objects without consideration of the causes of motion ([Wikipedia](http://en.wikipedia.org/wiki/Kinematics)).  

        Kinematics of a particle is the description of the motion when the object is considered a particle.  

        A particle as a physical object does not exist in nature; it is a simplification to understand the motion of a body or it is a conceptual definition such as the center of mass of a system of objects.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Vectors in Kinematics

        Some mechanical quantities in Kinematics (position and its derivatives) are represented as vectors and others, such as time and distance, are scalars.  
        A vector in Mechanics is a physical quantity with magnitude, direction, and satisfies some elementary vector arithmetic, whereas a scalar is a physical quantity that is fully expressed by a magnitude (a number) only.  

        For a review about scalars and vectors, see chapter 1 of [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html).  

         For how to use Python to work with scalars and vectors, see the notebook [Scalar and Vector](http://nbviewer.jupyter.org/github/BMCLab/BMC/blob/master/notebooks/ScalarVector.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Position

        Consider a point in the three-dimensional Euclidean space described in a Cartesian coordinate system (see the notebook [Frame of reference](http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/ReferenceFrame.ipynb) for an introduction on coordinate systems in Mechanics and Biomechanics):  
        <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/vector3Dijk.png?raw=1" width=350/><figcaption><center><i>Figure. Representation of a point$\mathbf{P}$and its position vector$\overrightarrow{\mathbf{r}}$in a Cartesian coordinate system. The versors <span class="notranslate">$\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}\,$</span> form a basis for this coordinate system and are usually represented in the color sequence RGB (red, green, blue) for easier visualization.</i></center></figcaption></figure>  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The position of this point in space can be represented as a triple of values each representing the coordinate at each axis of the Cartesian coordinate system following the$\mathbf{X, Y, Z}$convention order (which is omitted):

        <span class="notranslate">$(x,\, y,\, z)$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The position of a particle in space can also be represented by a vector in the Cartesian coordinate system, with the origin of the vector at the origin of the coordinate system and the tip of the vector at the point position:

        <span class="notranslate">$\overrightarrow{\mathbf{r}}(t) = x\,\hat{\mathbf{i}} + y\,\hat{\mathbf{j}} + z\,\hat{\mathbf{k}}$</span>

        Where <span class="notranslate">$\hat{\mathbf{i}},\, \hat{\mathbf{j}},\, \hat{\mathbf{k}}\,$</span> are unit vectors in the directions of the axes$\mathbf{X, Y, Z}$.

        For a review on vectors, see the notebook [Scalar and vector](http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/ScalarVector.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With this new notation, the coordinates of a point representing the position of a particle that vary with time would be expressed by the following position vector$\overrightarrow{\mathbf{r}}(t)$:   

        <span class="notranslate">$\overrightarrow{\mathbf{r}}(t) = x(t)\,\hat{\mathbf{i}} + y(t)\,\hat{\mathbf{j}} + z(t)\,\hat{\mathbf{k}}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A vector can also be represented in matrix form:

        <span class="notranslate">$\overrightarrow{\mathbf{r}}(t) = \begin{bmatrix} x(t) \\y(t) \\z(t) \end{bmatrix}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the unit vectors in each Cartesian coordinate in matrix form are given by:

        <span class="notranslate">$\hat{\mathbf{i}} = \begin{bmatrix}1\\0\\0 \end{bmatrix},\; \hat{\mathbf{j}} = \begin{bmatrix}0\\1\\0 \end{bmatrix},\; \hat{\mathbf{k}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Basis

        In [linear algebra](http://en.wikipedia.org/wiki/Linear_algebra), a set of unit linearly independent vectors as the three vectors above (orthogonal in the Euclidean space) that can represent any vector via [linear combination](http://en.wikipedia.org/wiki/Linear_combination) is called a basis. A basis is the foundation of creating a reference frame and we will study how to do that other time.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Displacement

        The shortest distance between two positions of a particle.

        As the difference between two vectors, displacement is also a vector quantity.

        <span class="notranslate">$\mathbf{\overrightarrow{d}} = \mathbf{\overrightarrow{r}_2} - \mathbf{\overrightarrow{r}_1}$</span>

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/displacement.png?raw=1" width=450/><figcaption><center><i>Figure. Representation of the displacement vector$\mathbf{\overrightarrow{d}}$between two positions$\mathbf{\overrightarrow{r}_1}$and$\mathbf{\overrightarrow{r}_2}$.</i></center></figcaption></figure>  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Velocity

        Velocity is the rate (with respect to time) of change of the position of a particle.  

        The average velocity between two instants is:

        <span class="notranslate">$\overrightarrow{\mathbf{v}}(t) = \frac{\overrightarrow{\mathbf{r}}(t_2)-\overrightarrow{\mathbf{r}}(t_1)}{t_2-t_1} = \frac{\Delta \overrightarrow{\mathbf{r}}}{\Delta t}$</span>    

        The instantaneous velocity of the particle is obtained when$\Delta t$approaches to zero, which from calculus is the first-order [derivative](http://en.wikipedia.org/wiki/Derivative) of the position vector:

        <span class="notranslate">$\overrightarrow{\mathbf{v}}(t) = \lim_{\Delta t \to 0} \frac{\Delta \overrightarrow{\mathbf{r}}}{\Delta t} = \lim_{\Delta t \to 0} \frac{\overrightarrow{\mathbf{r}}(t+\Delta t)-\overrightarrow{\mathbf{r}}(t)}{\Delta t} = \frac{\mathrm{d}\overrightarrow{\mathbf{r}}}{dt}$</span>

        For the movement of a particle described with respect to an [inertial Frame of reference](http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/ReferenceFrame.ipynb), the derivative of a vector is obtained by differentiating each vector component of the Cartesian coordinates (since the base versors <span class="notranslate">$\hat{\mathbf{i}}, \hat{\mathbf{j}}, \hat{\mathbf{k}}$</span> are constant):   

        <span class="notranslate">$\overrightarrow{\mathbf{v}}(t) = \frac{\mathrm{d}\overrightarrow{\mathbf{r}}(t)}{dt} = \frac{\mathrm{d}x(t)}{\mathrm{d}t}\hat{\mathbf{i}} + \frac{\mathrm{d}y(t)}{\mathrm{d}t}\hat{\mathbf{j}} + \frac{\mathrm{d}z(t)}{\mathrm{d}t}\hat{\mathbf{k}}$</span>

        Or in matrix form (and using the Newton's notation for differentiation):

        <span class="notranslate">$\overrightarrow{\mathbf{v}}(t) = \begin{bmatrix}
        \dot x(t) \\
        \dot y(t) \\
        \dot z(t)
        \end{bmatrix}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Acceleration  

        Acceleration is the rate (with respect to time) of change of the velocity of a particle, which can also be given by the second-order rate of change of the position.

        The average acceleration between two instants is:

        <span class="notranslate">$\overrightarrow{\mathbf{a}}(t) = \frac{\overrightarrow{\mathbf{v}}(t_2)-\overrightarrow{\mathbf{v}}(t_1)}{t_2-t_1} = \frac{\Delta \overrightarrow{\mathbf{v}}}{\Delta t}$</span>

        Likewise, instantaneous acceleration is the first-order derivative of the velocity or the second-order derivative of the position vector:   

        <span class="notranslate">$\overrightarrow{\mathbf{a}}(t) = \frac{\mathrm{d}\overrightarrow{\mathbf{v}}(t)}{\mathrm{d}t} = \frac{\mathrm{d}^2\overrightarrow{\mathbf{r}}(t)}{\mathrm{d}t^2} = \frac{\mathrm{d}^2x(t)}{\mathrm{d}t^2}\hat{\mathbf{i}} + \frac{\mathrm{d}^2y(t)}{\mathrm{d}t^2}\hat{\mathbf{j}} + \frac{\mathrm{d}^2z(t)}{\mathrm{d}t^2}\hat{\mathbf{k}}$</span>

        And in matrix form:

        <span class="notranslate">$\overrightarrow{\mathbf{a}}(t) = \begin{bmatrix}
        \ddot x(t) \\
        \ddot y(t) \\
        \ddot z(t)
        \end{bmatrix}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For curiosity, see [Notation for differentiation](https://en.wikipedia.org/wiki/Notation_for_differentiation) on the origin of the different notations for differentiation.

        When the base versors change in time, for instance when the basis is attached to a rotating frame or reference, the components of the vector’s derivative is not the derivatives of its components; we will also have to consider the derivative of the basis with respect to time.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The antiderivative

        As the acceleration is the derivative of the velocity which is the derivative of position, the inverse mathematical operation is the [antiderivative](http://en.wikipedia.org/wiki/Antiderivative) (or integral):

        <span class="notranslate">$\begin{array}{l l}
        \mathbf{r}(t) = \mathbf{r}_0 + \int \mathbf{v}(t) \:\mathrm{d}t \\
        \mathbf{v}(t) = \mathbf{v}_0 + \int \mathbf{a}(t) \:\mathrm{d}t
        \end{array}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Some cases of motion of a particle

        To deduce some trivial cases of motion of a particle (at rest, at constant speed, and at constant acceleration), we can start from the equation for its position and differentiate it to obtain expressions for the velocity and acceleration or the inverse approach, start with the equation for acceleration, and then integrate it to obtain the velocity and position of the particle. Both approachs are valid in Mechanics. For the present case, it probaly makes more sense to start with the expression for acceleration.

        #### Particle at rest

        <span class="notranslate">$\begin{array}{l l}
        \overrightarrow{\mathbf{a}}(t) = 0 \\
        \overrightarrow{\mathbf{v}}(t) = 0 \\
        \overrightarrow{\mathbf{r}}(t) = \overrightarrow{\mathbf{r}}_0
        \end{array}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Particle at constant speed

        <span class="notranslate">$\begin{array}{l l}
        \overrightarrow{\mathbf{a}}(t) = 0 \\
        \overrightarrow{\mathbf{v}}(t) = \overrightarrow{\mathbf{v}}_0 \\
        \overrightarrow{\mathbf{r}}(t) = \overrightarrow{\mathbf{r}}_0 + \overrightarrow{\mathbf{v}}_0t
        \end{array}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Particle at constant acceleration

        <span class="notranslate">$\begin{array}{l l}
        \overrightarrow{\mathbf{a}}(t) = \overrightarrow{\mathbf{a}}_0 \\
        \overrightarrow{\mathbf{v}}(t) = \overrightarrow{\mathbf{v}}_0 + \overrightarrow{\mathbf{a}}_0t \\
        \overrightarrow{\mathbf{r}}(t) = \overrightarrow{\mathbf{r}}_0 + \overrightarrow{\mathbf{v}}_0t +
        \frac{1}{2}\overrightarrow{\mathbf{a}}_0 t^2
        \end{array}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Values for the mechanical quantities
        """
    )
    return


@app.cell
def _(np):
    t = np.linspace(0, 2, 101)
    print('t:', t)

    r0 = 1
    v0 = 2
    a0 = 4
    r = r0 + v0*t + 1/2*a0*t**2
    print('r:', r)
    return a0, r, r0, t, v0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Visual representation of these cases
        """
    )
    return


@app.cell
def _(plt, r, t):
    # a simple plot
    plt.plot(t, r);
    return


@app.cell
def _(a0, np, plt, r0, t, v0):
    # a more decorated plot
    plt.rc('axes',  labelsize=14,  titlesize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    f, axarr = plt.subplots(3, 3, sharex = True, sharey = True, figsize=(14,7))
    plt.suptitle('Scalar kinematics of a particle', fontsize=20);

    tones = np.ones(np.size(t))

    axarr[0, 0].set_title('at rest', fontsize=14);
    axarr[0, 0].plot(t, r0*tones, 'g', linewidth=4, label='$r(t)=1$')
    axarr[1, 0].plot(t,  0*tones, 'b', linewidth=4, label='$v(t)=0$')
    axarr[2, 0].plot(t,  0*tones, 'r', linewidth=4, label='$a(t)=0$')
    axarr[0, 0].set_ylabel('r(t) [m]')
    axarr[1, 0].set_ylabel('v(t) [m/s]')
    axarr[2, 0].set_ylabel('a(t) [m/s$^2$]')

    axarr[0, 1].set_title('at constant speed');
    axarr[0, 1].plot(t, r0*tones+v0*t, 'g', linewidth=4, label='$r(t)=1+2t$')
    axarr[1, 1].plot(t, v0*tones,      'b', linewidth=4, label='$v(t)=2$')
    axarr[2, 1].plot(t,  0*tones,      'r', linewidth=4, label='$a(t)=0$')

    axarr[0, 2].set_title('at constant acceleration');
    axarr[0, 2].plot(t, r0*tones+v0*t+1/2.*a0*t**2,'g', linewidth=4,
                     label='$r(t)=1+2t+\\frac{1}{2}4t^2$')
    axarr[1, 2].plot(t, v0*tones+a0*t,             'b', linewidth=4,
                     label='$v(t)=2+4t$')
    axarr[2, 2].plot(t, a0*tones,                  'r', linewidth=4,
                     label='$a(t)=4$')

    for i in range(3):
        axarr[2, i].set_xlabel('Time [s]');
        for j in range(3):
            axarr[i,j].set_ylim((-.2, 10))
            axarr[i,j].legend(loc = 'upper left', frameon=True, framealpha = 0.9, fontsize=14)

    plt.subplots_adjust(hspace=0.09, wspace=0.07)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Symbolic programming

        We can use [Sympy](http://www.sympy.org/en/index.html), a Python library for symbolic mathematics, to deduce the expressions for the cases of motion of a particle we just visualized.  
        Let's show how to integrate with Sympy for the case of particle with constant acceleration:
        """
    )
    return


@app.cell
def _():
    import sympy as sym
    sym.init_printing(use_latex='mathjax')
    t_1 = sym.symbols('t', real=True)
    (r0_1, v0_1, a0_1) = sym.symbols('r0, v0, a0', real=True, constant=True)
    return a0_1, r0_1, sym, t_1, v0_1


@app.cell
def _(a0_1, sym, t_1, v0_1):
    v = sym.integrate(a0_1, t_1) + v0_1
    v
    return (v,)


@app.cell
def _(r0_1, sym, t_1, v):
    r_1 = sym.integrate(v, t_1) + r0_1
    r_1
    return (r_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can also plot a symbolic expression in a given range after we substitute the symbolic variables with numbers:
        """
    )
    return


@app.cell
def _(a0_1, r0_1, r_1, t_1, v0_1):
    from sympy.plotting import plot
    r_n = r_1.subs({r0_1: 1, v0_1: 2, a0_1: 4})
    plot(r_n, (t_1, 0, 2), xlim=(0, 2), ylim=(0, 10), axis_center=(0, 0), line_color='g', xlabel='Time [s]', ylabel='r(t) [m]', legend=True, title='Scalar kinematics of a particle at constant acceleration', backend='matplotlib', size=(5, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Kinematics of human movement
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Kinematics of the 100-m race

        An example where the analysis of some aspects of the human body movement can be reduced to the analysis of a particle is the study of the Biomechanics of the 100-m race.

        A technical report with the kinematic data for the 100-m world record by Usain Bolt can be downloaded from the [website for Research Projects](http://www.iaaf.org/development/research) from the International Association of Athletics Federations.  
        [Here is a direct link for that report](http://www.iaaf.org/download/download?filename=76ade5f9-75a0-4fda-b9bf-1b30be6f60d2.pdf&urlSlug=1-biomechanics-report-wc-berlin-2009-sprint). In particular, the following table shows the data for the three medalists in that race:  
        <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/Berlin2009_100m.png?raw=1" width=700 alt="partial times of the 100m-race at Berlin 2009"/><figcaption><center><i>Figure. Data from the three medalists of the 100-m dash in Berlin, 2009 (<a href="http://www.iaaf.org/download/download?filename=76ade5f9-75a0-4fda-b9bf-1b30be6f60d2.pdf&urlSlug=1-biomechanics-report-wc-berlin-2009-sprint)">IAAF report</a>).</i></center></figcaption></figure>

        The column **RT** in the table above refers to the reaction time of each athlete. The IAAF has a very strict rule about reaction time: any athlete with a reaction time less than 100 ms is disqualified from the competition! See the website [Reaction Times and Sprint False Starts](http://condellpark.com/kd/reactiontime.htm) for a discussion about this rule.

        You can measure your own reaction time in a simple way visiting this website: [http://www.humanbenchmark.com/tests/reactiontime](http://www.humanbenchmark.com/tests/reactiontime).

        The article [A Kinematics Analysis Of Three Best 100 M Performances Ever](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3661886/) by Krzysztof and Mero presents a detailed kinematic analysis of 100-m races.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## More examples

         - From Ruina's book, study the samples 12.2 and 12.3.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

         - Read the preface and first chapter of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html) about how someone should study Mechanics.  
         - See the notebook [Spatial and temporal characteristics](http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/SpatialTemporalCharacteristcs.ipynb) about how the simple measurement of spatial and temporal kinematic variables can be very useful to describe the human gait.  
         - See the notebook [The minimum jerk hypothesis](http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/MinimumJerkHypothesis.ipynb) about the conjecture that movements are performed (organized) with the smoothest trajectory possible.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

         - Khan Academy: [One-dimensional motion](https://www.khanacademy.org/science/ap-physics-1/ap-one-dimensional-motion)  
         - [Powers of 10, Units, Dimensions, Uncertainties, Scaling Arguments](https://youtu.be/GtOGurrUPmQ)  
         - [1D Kinematics - Speed, Velocity, Acceleration](https://youtu.be/q9IWoQ199_o)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Answer the 12 questions of the [Khan Academy's test on one-dimensional motion](https://www.khanacademy.org/science/ap-physics-1/ap-one-dimensional-motion/test/ap-one-dimensional-motion-unit-test?modal=1).  

        2. Consider the data for the three medalists of the 100-m dash in Berlin, 2009, shown previously.  
           a. Calculate the average velocity and acceleration.   
           b. Plot the graphs for the displacement, velocity, and acceleration versus time.   
           c. Plot the graphs velocity and acceleration versus partial distance (every 20m).   
           d. Calculate the average velocity and average acceleration and the instants and values of the peak velocity and peak acceleration.  

        3. The article "Biomechanical Analysis of the Sprint and Hurdles Events at the 2009 IAAF World Championships in Athletics" by Graubner and Nixdorf lists the 10-m split times for the three medalists of the 100-m dash in Berlin, 2009 and is shown below.  
           a. Repeat the same calculations performed in problem 2 and compare the results.
        <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/Berlin2009_100m_10.png?raw=1" width=600 alt="partial times of the 100m-race at Berlin 2009"/></figure>  
        <br>   

        4. On an Olympic running track, runners A and B start running on the first lane (a.k.a., the inside lane) from the same position on the track, but in opposite directions. If$\lVert v_A \rVert=4$m/s and$\lVert v_B \rVert=6$m/s, how far from the starting line will the runners meet?

        5. A body attached to a spring has its position (in cm) described by the equation$x(t) = 2\sin(4\pi t + \pi/4)$.   
           a. Calculate the equation for the body velocity and acceleration.   
           b. Plot the position, velocity, and acceleration in the interval [0, 1] s.
   
        6. The rectilinear motion of a particle is given by$x(t) = -12t^3 + 15t^2 + 5t + 2$[s; m]. Calculate:  
           a. Velocity and acceleration of the particle as a function of time. Solution:$v(t)=36t^2+30t+5$[s; m/s],$a(t)=-72t+30$[s; m/s2].  
           b. Total distance traveled by the particle in the interval$0 \leq t \leq 4$s. Solution:$\Delta_{0-4}=524.02$m.  
           c. Maximum value of the velocity module reached by the particle in the interval$0 \leq t \leq 4$s. Solution:$\lVert v_{max} \rVert=451.0$m/s.  
           d. Plots of the position, velocity and acceleration of the particle in the interval$0 \leq t \leq 4$s.
   
        7. A stone is released from the opening of a well, and the noise from its fall to the bottom is heard 4 seconds later. Knowing that the speed of sound in air is 340 m/s, determine the depth of the well. Solution:$h=70.55$m.  

        8. The position of a particle is given by$\overrightarrow{\mathbf{r}}(t)=(t^2\hat{\mathbf{i}}+e^{t}\hat{\mathbf{j}})$.    
            a. Calculate the velocity and acceleration of the particle as functions of time.   
            b. Draw the path of the particle and show the vectors$\overrightarrow{\mathbf{v}}(t)$and$\overrightarrow{\mathbf{a}}(t)$at$t=1$s.

        9. Sometimes all we have access to is the image with plotted data that interests us. For example, the first figure shown in this Notebook contains velocity versus position data for Usain Bolt's 100m sprint at a much higher resolution than the numerical data shown here. If we want to access numerical data from an image, it is possible to extract this information using some software to automatically identify points of interest. If points cannot be extracted automatically, you may need to do it manually. The [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) software is one such tool. Try using it to extract the data from the quoted figure. Note: it won't be easy; you may have to do this manually. Anyway, know that such a tool exists.

        10. There are some nice free software that can be used for the kinematic analysis of human motion. Some of them are: [Kinovea](http://www.kinovea.org/), [Tracker](http://www.cabrillo.edu/~dbrown/tracker/), and [SkillSpector](http://video4coach.com/index.php?option=com_content&task=view&id=13&Itemid=45). Visit their websites and explore these software to understand in which biomechanical applications they could be used.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Graubner R, Nixdorf E (2011) [Biomechanical Analysis of the Sprint and Hurdles Events at the 2009 IAAF World Championships in Athletics ](http://www.meathathletics.ie/devathletes/pdf/Biomechanics%20of%20Sprints.pdf). [New Studies in Athletics](http://www.iaaf.org/development/new-studies-in-athletics), 1/2, 19-53.
        - Krzysztof M, Antti Mero A (2013) [A Kinematics Analysis Of Three Best 100 M Performances Ever](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3661886/). Journal of Human Kinetics, 36, 149–160.
        - [Research Projects](http://www.iaaf.org/development/research) from the International Association of Athletics Federations.  
        - Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.  
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
