import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a href="https://colab.research.google.com/github/BMClab/BMC/blob/master/notebooks/KinematicChain.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Kinematic chain in a plane (2D)

        > Marcos Duarte, Renato Naville Watanabe   
        > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Properties-of-kinematic-chains" data-toc-modified-id="Properties-of-kinematic-chains-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Properties of kinematic chains</a></span></li><li><span><a href="#The-kinematics-of-one-link-system" data-toc-modified-id="The-kinematics-of-one-link-system-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>The kinematics of one-link system</a></span><ul class="toc-item"><li><span><a href="#Forward-and-inverse-kinematics" data-toc-modified-id="Forward-and-inverse-kinematics-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Forward and inverse kinematics</a></span></li><li><span><a href="#Matrix-representation-of-the-kinematics" data-toc-modified-id="Matrix-representation-of-the-kinematics-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Matrix representation of the kinematics</a></span></li></ul></li><li><span><a href="#Differential-kinematics" data-toc-modified-id="Differential-kinematics-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Differential kinematics</a></span><ul class="toc-item"><li><span><a href="#Linear-velocity-of-the-endpoint" data-toc-modified-id="Linear-velocity-of-the-endpoint-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Linear velocity of the endpoint</a></span></li><li><span><a href="#Linear-acceleration-of-the-endpoint" data-toc-modified-id="Linear-acceleration-of-the-endpoint-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Linear acceleration of the endpoint</a></span><ul class="toc-item"><li><span><a href="#Tangential-acceleration" data-toc-modified-id="Tangential-acceleration-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Tangential acceleration</a></span></li><li><span><a href="#Centripetal-acceleration" data-toc-modified-id="Centripetal-acceleration-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>Centripetal acceleration</a></span></li></ul></li><li><span><a href="#Simulation" data-toc-modified-id="Simulation-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Simulation</a></span></li><li><span><a href="#Jacobian-matrix" data-toc-modified-id="Jacobian-matrix-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Jacobian matrix</a></span></li><li><span><a href="#Derivative-of-a-vector-valued-function-using-the-Jacobian-matrix" data-toc-modified-id="Derivative-of-a-vector-valued-function-using-the-Jacobian-matrix-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Derivative of a vector-valued function using the Jacobian matrix</a></span></li><li><span><a href="#Jacobian-matrix-in-the-context-of-kinematic-chains" data-toc-modified-id="Jacobian-matrix-in-the-context-of-kinematic-chains-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Jacobian matrix in the context of kinematic chains</a></span><ul class="toc-item"><li><span><a href="#Jacobian-matrix-of-one-link-chain" data-toc-modified-id="Jacobian-matrix-of-one-link-chain-4.6.1"><span class="toc-item-num">4.6.1&nbsp;&nbsp;</span>Jacobian matrix of one-link chain</a></span></li></ul></li></ul></li><li><span><a href="#The-kinematics-of-a-two-link-chain" data-toc-modified-id="The-kinematics-of-a-two-link-chain-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>The kinematics of a two-link chain</a></span><ul class="toc-item"><li><span><a href="#Joint-and-segment-angles" data-toc-modified-id="Joint-and-segment-angles-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Joint and segment angles</a></span></li><li><span><a href="#Inverse-kinematics" data-toc-modified-id="Inverse-kinematics-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Inverse kinematics</a></span></li></ul></li><li><span><a href="#Differential-kinematics" data-toc-modified-id="Differential-kinematics-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Differential kinematics</a></span><ul class="toc-item"><li><span><a href="#Simulation" data-toc-modified-id="Simulation-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Simulation</a></span></li><li><span><a href="#Calculation-of-each-type-of-acceleration-of-the-endpoint-for-the-numerical-example-of-the-two-link-system" data-toc-modified-id="Calculation-of-each-type-of-acceleration-of-the-endpoint-for-the-numerical-example-of-the-two-link-system-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Calculation of each type of acceleration of the endpoint for the numerical example of the two-link system</a></span></li><li><span><a href="#And-the-corresponding-plots" data-toc-modified-id="And-the-corresponding-plots-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>And the corresponding plots</a></span></li></ul></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#Problems" data-toc-modified-id="Problems-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Problems</a></span></li><li><span><a href="#References" data-toc-modified-id="References-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction

        <figure><a href="https://en.wikipedia.org/wiki/Kinematic_chain"><img align="right"  src='https://upload.wikimedia.org/wikipedia/commons/2/2c/Modele_cinematique_corps_humain.svg' width="240"   style="margin: 0 10px 0 10px"/></a></figure>  

        **Kinematic chain refers to an assembly of rigid bodies (links) connected by joints that is the mathematical model for a mechanical system which in turn can represent a biological system such as the human body** ([Wikipedia](https://en.wikipedia.org/wiki/Kinematic_chain)).   

        The term chain refers to the fact that the links are constrained by their connections (typically, by a hinge joint which is also called pin joint or revolute joint) to other links. As consequence of this constraint, a kinematic chain in a plane is an example of circular motion of a rigid object.

        Chapter 16 of Ruina and Rudra's book and chapter 2 of the Rade's book are a good formal introduction on the topic of circular motion of a rigid object anmd sections 2.7 and 2.8 of Rade's book covers specifically kinematic chains. However, in this notebook we will not employ the mathematical formalism introduced in those chapters - the concept of a rotating reference frame and the related rotation matrix - we cover these subjects in the notebook [Rigid-body transformations (2D)](https://nbviewer.jupyter.org/github/BMClab/BMC/blob/master/notebooks/Transformation2D.ipynb).  

        Here, we will describe the kinematics of a chain in a Cartesian coordinate system using solely trigonometry and calculus. This approach is simpler and more intuitive but it gets too complicated for a kinematic chain with many links or in the 3D space. For such more complicated problems, indeed it would be recommended using rigid transformations (see for example, Siciliano et al. (2009)).  

        We will deduce the kinematic properties of kinematic chains algebraically using [Sympy](http://sympy.org/), a Python library for symbolic mathematics. In Sympy, we could have used the [mechanics module](http://docs.sympy.org/latest/modules/physics/mechanics/index.html), a specific module for creation of symbolic equations of motion for multibody systems, but let's deduce most of the stuff by ourselves to understand the details.  

        <p style="text-align: right;"><i>Figure. The human body modeled as a kinematic chain (<a href="https://en.wikipedia.org/wiki/Kinematic_chain">image from Wikipedia</a>).</i></p>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Properties of kinematic chains

        For a kinematic chain, the **base** is the extremity (origin) of a kinematic chain which is typically considered attached to the ground, body or fixed. The **endpoint** is the other extremity (end) of a kinematic chain and typically can move. In robotics, the term **end-effector** is used and usually refers to a last link (rigid body) in this chain.

        In topological terms, a kinematic chain is termed **open** when there is only one sequence of links connecting the two ends of the chain. Otherwise it's termed **closed** and in this case a sequence of links forms a loop. A kinematic chain can be classified as **serial** or **parallel** or a **mixed** of both. In a serial chain the links are connected in a serial order. A serial chain is an open chain, otherwise it is a parallel chain or a branched chain (e.g., hand and fingers).  

        Although the definition above is clear and classic in mechanics, it is not the definition used by health professionals (clinicians and athletic trainers) when describing human movement. They refer to human joints and segments as a closed or open kinematic (or kinetic) chain simply if the distal segment (typically the foot or hand) is fixed (closed chain) or not (open chain). This difference in definition sometimes will result in different classifications. For example, a person standing on one leg is an open kinematic chain in mechanics, but closed according to the latter definition. In this text we will be consistent with mechanics, but keep in mind this difference when interacting with clinicians and athletic trainers.

        Another important term to characterize a kinematic chain is **<a href="https://en.wikipedia.org/wiki/Degrees_of_freedom_(mechanics)">degree of freedom (DOF)</a>**. In mechanics, the degree of freedom of a mechanical system is the number of independent parameters that define its configuration or that determine the state of a physical system. A particle in the 3D space has three DOFs because we need three coordinates to specify its position. A rigid body in the 3D space has six DOFs because we need three coordinates of one point at the body to specify its position and three angles to to specify its orientation in order to completely define the configuration of the rigid body. For a link attached to a fixed body by a hinge joint in a plane, all we need to define the configuration of the link is one angle and then this link has only one DOF. A kinematic chain with two links in a plane has two DOFs, and so on.

        The **mobility** of a kinematic chain is its total number of degrees of freedom. The **redundancy** of a kinematic chain is its mobility minus the number of degrees of freedom of the endpoint.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The kinematics of one-link system

        First, let's study the case of a system composed by one planar hinge joint and one link, which technically it's not a chain but it will be useful to review (or introduce) key concepts.  
        <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/onelink.gif?raw=1" width=350 alt="onelink"/><figcaption><center><i>Figure. One link attached to a fixed body by a hinge joint in a plane.</i></center></figcaption> </figure>

        First, let's import the necessary libraries from Python and its ecosystem:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2,
                                                    "lines.markersize": 10})
    from IPython.display import display, Math
    from sympy import Symbol, symbols, Function, Matrix, simplify, lambdify, expand, latex
    from sympy import diff, cos, sin, sqrt, acos, atan2, atan, Abs
    from sympy.vector import CoordSys3D
    from sympy.physics.mechanics import dynamicsymbols, mlatex, init_vprinting
    init_vprinting()
    return (
        CoordSys3D,
        Math,
        Matrix,
        Symbol,
        acos,
        atan,
        atan2,
        cos,
        diff,
        display,
        dynamicsymbols,
        expand,
        lambdify,
        latex,
        mlatex,
        np,
        plt,
        simplify,
        sin,
        sqrt,
        symbols,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We need to define a Cartesian coordinate system and the symbolic variables,$t$,$\ell$,$\theta$(and make$\theta$a function of time):
        """
    )
    return


@app.cell
def _(CoordSys3D, Symbol, dynamicsymbols):
    G = CoordSys3D('')
    t = Symbol('t')
    l = Symbol('ell', real=True, positive=True)
    # type \theta and press tab for the Greek letter θ:
    θ = dynamicsymbols('theta', real=True)  # or Function('theta')(t)
    return G, l, t, θ


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using trigonometry, the endpoint position in terms of the joint angle and link length is:
        """
    )
    return


@app.cell
def _(G, cos, l, sin, θ):
    r_p = l*cos(θ)*G.i + l*sin(θ)*G.j + 0*G.k
    r_p
    return (r_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With the components:
        """
    )
    return


@app.cell
def _(r_p):
    r_p.components
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Forward and inverse kinematics

        Computing the configuration of a link or a chain (including the endpoint location) from the joint parameters (joint angles and link lengths) as we have done is called [**forward or direct kinematics**](https://en.wikipedia.org/wiki/Forward_kinematics).

        If the linear coordinates of the endpoint position are known (for example, if they are measured with a motion capture system) and one wants to obtain the joint angle(s), this process is known as [**inverse kinematics**](https://en.wikipedia.org/wiki/Inverse_kinematics). For the one-link system above:
        <br>
        <span class="notranslate">$\theta = \arctan\left(\frac{y_P}{x_P}\right)$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Matrix representation of the kinematics

        The mathematical manipulation will be easier if we use the matrix formalism (and let's drop the explicit dependence on <span class="notranslate">$t$</span>):
        """
    )
    return


@app.cell
def _(G, Matrix, r_p):
    r = Matrix((r_p.dot(G.i), r_p.dot(G.j)))
    r
    return (r,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using the matrix formalism will simplify things, but we will loose some of the Sympy methods for vectors (for instance, the variable `r_p` has a method `magnitude` and the variable `r` does not.   
        If you prefer, you can keep the pure vector representation and just switch to matrix representation when displaying a variable:
        """
    )
    return


@app.cell
def _(G, r_p):
    r_p.to_matrix(G)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The third element of the matrix above refers to the <span class="notranslate">$\hat{\mathbf{k}}$</span> component which is zero for the present case (planar movement).  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Differential kinematics

        Differential kinematics gives the relationship between the joint velocities and the corresponding endpoint linear velocity. This mapping is described by a matrix, termed [**Jacobian matrix**](http://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant), which depends on the kinematic chain configuration and it is of great use in the study of kinematic chains.  
        First, let's deduce the endpoint velocity without using the Jacobian and then we will see how to calculate the endpoint velocity using the Jacobian matrix.

        The velocity of the endpoint can be obtained by the first-order derivative of the position vector. The derivative of a vector is obtained by differentiating each vector component:
        <br>  
        <span class="notranslate">$\frac{\mathrm{d}\overrightarrow{\mathbf{r}}}{\mathrm{d}t} =
        \large
        \begin{bmatrix}
        \frac{\mathrm{d}x_P}{\mathrm{d}t} \\
        \frac{\mathrm{d}y_P}{\mathrm{d}t} \\
        \end{bmatrix}$</span>

        Note that the derivative is with respect to time but <span class="notranslate">$x_P$</span> and <span class="notranslate">$y_P$</span> depend explicitly on <span class="notranslate">$\theta$</span> and it's <span class="notranslate">$\theta$</span> that depends on <span class="notranslate">$t$($x_P$</span> and <span class="notranslate">$y_P$</span> depend implicitly on <span class="notranslate">$t$</span>). To calculate this type of derivative we will use the [chain rule](http://en.wikipedia.org/wiki/Chain_rule).  
        <br/>  

        <div style="background-color:#FBFBEF;border:1px solid black;padding:10px;">
        <b><a href="http://en.wikipedia.org/wiki/Chain_rule">Chain rule</a></b>   
        <br />
        For variable <span class="notranslate">$f$</span> which is function of variable <span class="notranslate">$g$</span> which in turn is function of variable <span class="notranslate">$t$,$f(g(t))$</span> or <span class="notranslate">$(f\circ g)(t)$</span>, the derivative of <span class="notranslate">$f$</span>  with respect to <span class="notranslate">$t$</span> is (using <a href="http://en.wikipedia.org/wiki/Notation_for_differentiation">Lagrange's notation</a>):   
        <br />
        <span class="notranslate">$(f\circ g)^{'}(t) = f'(g(t)) \cdot g'(t)$</span>

        Or using what is known as <a href="http://en.wikipedia.org/wiki/Notation_for_differentiation">Leibniz's notation</a>:   
        <br />
        <span class="notranslate">$\frac{\mathrm{d}f}{\mathrm{d}t} = \frac{\mathrm{d}f}{\mathrm{d}g} \cdot \frac{\mathrm{d}g}{\mathrm{d}t}$</span>

        If <span class="notranslate">$f$</span> is function of two other variables which both are function of <span class="notranslate">$t$,$f(x(t),y(t))$</span>, the chain rule for this case is:   
        <br />
        <span class="notranslate">$\frac{\mathrm{d}f}{\mathrm{d}t} = \frac{\partial f}{\partial x} \cdot \frac{\mathrm{d}x}{\mathrm{d}t} + \frac{\partial f}{\partial y} \cdot \frac{\mathrm{d}y}{\mathrm{d}t}$</span>
    
        Where <span class="notranslate">$df/dt$</span> represents the <a href="http://en.wikipedia.org/wiki/Total_derivative">total derivative</a> and <span class="notranslate">$\partial f / \partial x$</span> represents the <a href="http://en.wikipedia.org/wiki/Partial_derivative">partial derivative</a> of a function.   
        <br />
        <b><a href="http://en.wikipedia.org/wiki/Product_rule">Product rule</a></b>   
        The derivative of the product of two functions is:   
        <br />
        <span class="notranslate">$(f \cdot g)' = f' \cdot g + f \cdot g'$</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Linear velocity of the endpoint

        For the planar one-link case, the linear velocity of the endpoint is:
        """
    )
    return


@app.cell
def _(r, t):
    v = r.diff(t)
    v
    return (v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Where we used the [Newton's notation](http://en.wikipedia.org/wiki/Notation_for_differentiation) for differentiation.  
        Note that <span class="notranslate">$\dot{\theta}$</span> represents the unknown angular velocity of the joint; this is why the derivative of <span class="notranslate">$\theta$</span> is not explicitly solved.  
        The magnitude or [Euclidian norm](http://en.wikipedia.org/wiki/Vector_norm) of the vector <span class="notranslate">$\overrightarrow{\mathbf{v}}$</span> is:  
        <br>
        <span class="notranslate">$||\overrightarrow{\mathbf{v}}||=\sqrt{v_x^2+v_y^2}$</span>
        """
    )
    return


@app.cell
def _(simplify, sqrt, v):
    simplify(sqrt(v[0]**2 + v[1]**2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Which is <span class="notranslate">$\ell\dot{\theta}$</span>.<br>
        We could have used the function `norm` of Sympy, but the output does not simplify nicely:
        """
    )
    return


@app.cell
def _(simplify, v):
    simplify(v.norm())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The direction of <span class="notranslate">$\overrightarrow{\mathbf{v}}$</span> is tangent to the circular trajectory of the endpoint as can be seen in the figure below where its components are also shown.

        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/onelink_vel.gif?raw=1" width=350 alt="onelinkVel"/><figcaption><center><i>Figure. Endpoint velocity of one link attached to a fixed body by a hinge joint in a plane.</i></center></figcaption></figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Linear acceleration of the endpoint

        The acceleration of the endpoint position can be given by the second-order derivative of the position or by the first-order derivative of the velocity.  
        Using the chain and product rules for differentiation, the linear acceleration of the endpoint is:
        """
    )
    return


@app.cell
def _(t, v):
    acc = v.diff(t, 1)
    acc
    return (acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Examining the terms of the expression for the linear acceleration, we see there are two types of them: the term (in each direction) proportional to the angular acceleration <span class="notranslate">$\ddot{\theta}$</span> and other term proportional to the square of the angular velocity <span class="notranslate">$\dot{\theta}^{2}$</span>.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Tangential acceleration

        The term proportional to angular acceleration, <span class="notranslate">$a_t$</span>, is always tangent to the trajectory of the endpoint (see figure below) and it's magnitude or Euclidean norm is:
        """
    )
    return


@app.cell
def _(acc, expand, simplify, sqrt, t, θ):
    _A = θ.diff(t, 2)
    simplify(sqrt(expand(acc[0]).coeff(_A) ** 2 + expand(acc[1]).coeff(_A) ** 2)) * _A
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Centripetal acceleration

        The term proportional to angular velocity, <span class="notranslate">$a_c$</span>, always points to the joint, the center of the circular motion (see figure below), because of that this term is termed [centripetal acceleration](http://en.wikipedia.org/wiki/Centripetal_acceleration#Tangential_and_centripetal_acceleration). Its magnitude is:
        """
    )
    return


@app.cell
def _(acc, expand, simplify, sqrt, t, θ):
    _A = θ.diff(t) ** 2
    simplify(sqrt(expand(acc[0]).coeff(_A) ** 2 + expand(acc[1]).coeff(_A) ** 2)) * _A
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This means that there will be a linear acceleration even if the angular acceleration is zero because although the magnitude of the linear velocity is constant in this case, its direction varies (due to the centripetal acceleration).  
        <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/onelink_acc.gif?raw=1" width=350 alt="onelinkAcc"/><figcaption><center><i>Figure. Endpoint tangential and centripetal acceleration terms of one link attached to a fixed body by a hinge joint in a plane.</i></center></figcaption> </figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Simulation

        Let's plot some simulated data to have an idea of the one-link kinematics.  
        Consider <span class="notranslate">$\ell=1\:m,\theta_i=0^o,\theta_f=90^o$</span>, and <span class="notranslate">$1\:s$</span> of movement duration, and that it is a [minimum-jerk movement](https://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/MinimumJerkHypothesis.ipynb).
        """
    )
    return


@app.cell
def _(acc, l, lambdify, np, r, symbols, t, v, θ):
    (θ_i, θ_f, _d) = (0, np.pi / 2, 1)
    ts = np.arange(0.01, 1.01, 0.01)
    _mjt = θ_i + (θ_f - θ_i) * (10 * (t / _d) ** 3 - 15 * (t / _d) ** 4 + 6 * (t / _d) ** 5)
    ang = lambdify(t, _mjt, 'numpy')
    ang = ang(ts)
    vang = lambdify(t, _mjt.diff(t, 1), 'numpy')
    vang = vang(ts)
    aang = lambdify(t, _mjt.diff(t, 2), 'numpy')
    aang = aang(ts)
    jang = lambdify(t, _mjt.diff(t, 3), 'numpy')
    jang = jang(ts)
    (b, c, _d, e) = symbols('b c d e')
    _dicti = {l: 1, θ: b, θ.diff(t, 1): c, θ.diff(t, 2): _d, θ.diff(t, 3): e}
    r2 = r.subs(_dicti)
    rxfu = lambdify(b, r2[0], modules='numpy')
    ryfu = lambdify(b, r2[1], modules='numpy')
    v2 = v.subs(_dicti)
    vxfu = lambdify((b, c), v2[0], modules='numpy')
    vyfu = lambdify((b, c), v2[1], modules='numpy')
    acc2 = acc.subs(_dicti)
    axfu = lambdify((b, c, _d), acc2[0], modules='numpy')
    ayfu = lambdify((b, c, _d), acc2[1], modules='numpy')
    jerk = r.diff(t, 3)
    jerk2 = jerk.subs(_dicti)
    jxfu = lambdify((b, c, _d, e), jerk2[0], modules='numpy')
    jyfu = lambdify((b, c, _d, e), jerk2[1], modules='numpy')
    return (
        aang,
        ang,
        axfu,
        ayfu,
        jang,
        jxfu,
        jyfu,
        rxfu,
        ryfu,
        ts,
        vang,
        vxfu,
        vyfu,
    )


@app.cell
def _(
    aang,
    ang,
    axfu,
    ayfu,
    jang,
    jxfu,
    jyfu,
    np,
    plt,
    rxfu,
    ryfu,
    ts,
    vang,
    vxfu,
    vyfu,
):
    (_fig, _hax) = plt.subplots(2, 4, sharex=True, figsize=(14, 7))
    _hax[0, 0].plot(ts, ang * 180 / np.pi, linewidth=3)
    _hax[0, 0].set_title('Angular displacement [$^o$]')
    _hax[0, 0].set_ylabel('Joint')
    _hax[0, 1].plot(ts, vang * 180 / np.pi, linewidth=3)
    _hax[0, 1].set_title('Angular velocity [$^o/s$]')
    _hax[0, 2].plot(ts, aang * 180 / np.pi, linewidth=3)
    _hax[0, 2].set_title('Angular acceleration [$^o/s^2$]')
    _hax[0, 3].plot(ts, jang * 180 / np.pi, linewidth=3)
    _hax[0, 3].set_title('Angular jerk [$^o/s^3$]')
    _hax[1, 0].plot(ts, rxfu(ang), 'r', linewidth=3, label='x')
    _hax[1, 0].plot(ts, ryfu(ang), 'k', linewidth=3, label='y')
    _hax[1, 0].set_title('Linear displacement [$m$]')
    _hax[1, 0].legend(loc='best').get_frame().set_alpha(0.8)
    _hax[1, 0].set_ylabel('Endpoint')
    _hax[1, 1].plot(ts, vxfu(ang, vang), 'r', linewidth=3)
    _hax[1, 1].plot(ts, vyfu(ang, vang), 'k', linewidth=3)
    _hax[1, 1].set_title('Linear velocity [$m/s$]')
    _hax[1, 2].plot(ts, axfu(ang, vang, aang), 'r', linewidth=3)
    _hax[1, 2].plot(ts, ayfu(ang, vang, aang), 'k', linewidth=3)
    _hax[1, 2].set_title('Linear acceleration [$m/s^2$]')
    _hax[1, 3].plot(ts, jxfu(ang, vang, aang, jang), 'r', linewidth=3)
    _hax[1, 3].plot(ts, jyfu(ang, vang, aang, jang), 'k', linewidth=3)
    _hax[1, 3].set_title('Linear jerk [$m/s^3$]')
    _fig.suptitle('Minimum jerk trajectory kinematics of one-link system', fontsize=20)
    for (_i, _hax2) in enumerate(_hax.flat):
        _hax2.locator_params(nbins=5)
        _hax2.grid(True)
        if _i > 3:
            _hax2.set_xlabel('Time [s]')
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Jacobian matrix    
        <br>
        <div style="background-color:#FBFBEF;border:1px solid black;padding:10px;">
        The <b><a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">Jacobian matrix</a></b> is the matrix of all first-order partial derivatives of a vector-valued function <span class="notranslate">$F$</span>:  
    
        <br>
        <span class="notranslate">$F(q_1,...q_n) = \begin{bmatrix}F_{1}(q_1,...q_n)\\
        \vdots\\
        F_{m}(q_1,...q_n)\\
        \end{bmatrix}$</span>

        In a general form, the Jacobian matrix of the function <span class="notranslate">$F$</span> is:   
        <br>
        <span class="notranslate">$\mathbf{J}=
        \large
        \begin{bmatrix}
        \frac{\partial F_{1}}{\partial q_{1}} & ... & \frac{\partial F_{1}}{\partial q_{n}} \\
        \vdots  & \ddots  & \vdots \\
        \frac{\partial F_{m}}{\partial q_{1}} & ... & \frac{\partial F_{m}}{\partial q_{n}} \\
        \end{bmatrix}$</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Derivative of a vector-valued function using the Jacobian matrix
        <br>
        <div style="background-color:#FBFBEF;border:1px solid black;padding:10px;">
        The time-derivative of a vector-valued function <span class="notranslate">$F$</span> can be computed using the Jacobian matrix:  
        <br>  
        <span class="notranslate">$\frac{dF}{dt} = \mathbf{J} \cdot \begin{bmatrix}\frac{d q_1}{dt}\\
        \vdots\\
        \frac{d q_n}{dt}\\
        \end{bmatrix}$</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Jacobian matrix in the context of kinematic chains

        In the context of kinematic chains, the Jacobian is a matrix of all first-order partial derivatives of the linear position vector of the endpoint with respect to the angular position vector. The Jacobian matrix for a kinematic chain relates differential changes in the joint angle vector with the resulting differential changes in the linear position vector of the endpoint.  

        For a kinematic chain, the function <span class="notranslate">$F_{i}$</span> is the expression of the endpoint position in <span class="notranslate">$m$</span> coordinates and the variable <span class="notranslate">$q_{i}$</span> is the angle of each <span class="notranslate">$n$</span> joints.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Jacobian matrix of one-link chain

        For the planar one-link case, the Jacobian matrix of the position vector of the endpoint <span class="notranslate">$r_P$</span> with respect to the angular position vector <span class="notranslate">$q_1=\theta$</span> is:
        <br>  
        <span class="notranslate">$\mathbf{J}=
        \large
        \begin{bmatrix}
        \frac{\partial x_P}{\partial \theta} \\
        \frac{\partial y_P}{\partial \theta} \\
        \end{bmatrix}$</span>

        Which evaluates to:
        """
    )
    return


@app.cell
def _(r, θ):
    J = r.diff(θ)
    J
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And Sympy has a function to calculate the Jacobian:
        """
    )
    return


@app.cell
def _(r, θ):
    J_1 = r.jacobian([θ])
    J_1
    return (J_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can recalculate the kinematic expressions using the Jacobian matrix, which can be useful for simplifying the deduction.

        The linear velocity of the end-effector is given by the product between the Jacobian of the kinematic link and the angular velocity:
        <br>  
        <span class="notranslate">$\overrightarrow{\mathbf{v}} = \mathbf{J} \cdot \dot{\theta}$</span>

        Where:
        """
    )
    return


@app.cell
def _(t, θ):
    ω = θ.diff(t)
    ω
    return (ω,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The angular velocity is also a vector; it's direction is perpendicular to the plane of rotation and using the [right-hand rule](http://en.wikipedia.org/wiki/Right-hand_rule) this direction is the same as of the versor <span class="notranslate">$\hat{\mathbf{k}}$</span> coming out of the screen (paper).   

        Then:
        """
    )
    return


@app.cell
def _(J_1, ω):
    velJ = J_1 * ω
    velJ
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the linear acceleration of the endpoint is given by the derivative of this product:
        <br>  
        <span class="notranslate">$\overrightarrow{\mathbf{a}} = \dot{\mathbf{J}} \cdot \overrightarrow{\mathbf{\omega}} + \mathbf{J} \cdot \dot{\overrightarrow{\mathbf{\omega}}}$</span>

        Let's calculate this derivative:
        """
    )
    return


@app.cell
def _(J_1, t, ω):
    accJ = J_1.diff(t) * ω + J_1 * ω.diff(t)
    accJ
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These two expressions derived with the Jacobian are the same as the direct derivatives of the equation for the endpoint position.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The kinematics of a two-link chain

        We now will look at the case of a planar kinematic chain with two links, as shown below. The deduction will be similar to the case with one link we just saw.  
        <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/twolinks.gif?raw=1" width=400 alt="twolinks"/><figcaption><center><i>Figure. Kinematics of a two-link chain with hinge joints in a plane.</i></center></figcaption> </figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We need to define a Cartesian coordinate system and the symbolic variables <span class="notranslate">$t,\:\ell_1,\:\ell_2,\:\theta_1,\:\theta_2$</span> (and make <span class="notranslate">$\theta_1$</span> and <span class="notranslate">$\theta_2$</span> function of time):
        """
    )
    return


@app.cell
def _(CoordSys3D, Symbol, dynamicsymbols, symbols):
    G_1 = CoordSys3D('')
    t_1 = Symbol('t')
    (l1, l2) = symbols('ell_1 ell_2', positive=True)
    (θ1, θ2) = dynamicsymbols('theta1 theta2')
    return G_1, l1, l2, t_1, θ1, θ2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The position of the endpoint in terms of the joint angles and link lengths is:
        """
    )
    return


@app.cell
def _(G_1, cos, l1, l2, sin, θ1, θ2):
    r2_p = (l1 * cos(θ1) + l2 * cos(θ1 + θ2)) * G_1.i + (l1 * sin(θ1) + l2 * sin(θ1 + θ2)) * G_1.j
    r2_p
    return (r2_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With the components:
        """
    )
    return


@app.cell
def _(r2_p):
    r2_p.components
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And in matrix form:
        """
    )
    return


@app.cell
def _(G_1, Matrix, r2_p):
    r2_1 = Matrix((r2_p.dot(G_1.i), r2_p.dot(G_1.j)))
    r2_1
    return (r2_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Joint and segment angles

        Note that <span class="notranslate">$\theta_2$</span> is a joint angle (referred as measured in the **joint space**); the angle of the segment 2 with respect to the horizontal is <span class="notranslate">$\theta_1+\theta_2$</span> and is referred as an angle in the **segment space**.  
        Joint and segment angles are also referred as relative and absolute angles, respectively.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Inverse kinematics

        Using the [cosine rule](http://en.wikipedia.org/wiki/Law_of_cosines), in terms of the endpoint position, the angle <span class="notranslate">$\theta_2$</span> is:  
        <br>
        <span class="notranslate">$x_P^2 + y_P^2 = \ell_1^2+\ell_2^2 - 2\ell_1 \ell_2 cos(\pi-\theta_2)$</span>

        <span class="notranslate">$\theta_2 = \arccos\left(\frac{x_P^2 + y_P^2 - \ell_1^2 - \ell_2^2}{2\ell_1 \ell_2}\;\;\right)$</span>

        To find the angle <span class="notranslate">$\theta_1$</span>, if we now look at the triangle in red in the figure below, its angle <span class="notranslate">$\phi$</span> is:  
        <br>
        <span class="notranslate">$\phi = \arctan\left(\frac{\ell_2 \sin(\theta_2)}{\ell_1 + \ell_2 \cos(\theta_2)}\right)$</span>

        And the angle of its hypotenuse with the horizontal is:  
        <br>
        <span class="notranslate">$\theta_1 + \phi = \arctan\left(\frac{y_P}{x_P}\right)$</span>

        Then, the angle <span class="notranslate">$\theta_1$</span> is:  
        <br>
        <span class="notranslate">$\theta_1 = \arctan\left(\frac{y_P}{x_P}\right) - \arctan\left(\frac{\ell_2 \sin(\theta_2)}{\ell_1+\ell_2 \cos(\theta_2)}\right)$</span>

        Note that there are two possible sets of <span class="notranslate">$(\theta_1, \theta_2)$</span> angles for the same <span class="notranslate">$(x_P, y_P)$</span> coordinate that satisfy the equations above. The figure below shows in orange another possible configuration of the kinematic chain with the same endpoint coordinate. The other solution is <span class="notranslate">$\theta_2'=2\pi - \theta_2$</span>, but <span class="notranslate">$\sin(\theta_2')=-sin(\theta_{2})$</span> and then the <span class="notranslate">$arctan()$</span> term in the last equation becomes negative.   
        Even for a simple two-link chain we already have a problem of redundancy, there is more than one joint configuration for the same endpoint position; this will be much more problematic for chains with more links (more degrees of freedom).  
        <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/twolinks_ik.gif?raw=1" width=350 alt="twolinks_ik"/><figcaption><center><i>Figure. Indetermination in the inverse kinematics approach to determine one of the joint angles for a two-link chain with hinge joints in a plane.</i></center></figcaption> </figure>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Differential kinematics

        The linear velocity of the endpoint is:
        """
    )
    return


@app.cell
def _(r2_1, t_1):
    vel2 = r2_1.diff(t_1)
    vel2
    return (vel2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The linear velocity of the endpoint is the sum of the velocities at each joint, i.e., it is the velocity of the endpoint in relation to joint 2, for instance, <span class="notranslate">$\ell_2cos(\theta_1 + \theta_2)\dot{\theta}_1$</span>, plus the velocity of joint 2 in relation to joint 1, for instance, <span class="notranslate">$\ell_1\dot{\theta}_1 cos(\theta_1)$</span>, and this last term we already saw for the one-link example. In classical mechanics this is known as [relative velocity](http://en.wikipedia.org/wiki/Relative_velocity), an example of [Galilean transformation](http://en.wikipedia.org/wiki/Galilean_transformation).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The linear acceleration of the endpoint is:
        """
    )
    return


@app.cell
def _(r2_1, t_1):
    acc2_1 = r2_1.diff(t_1, 2)
    acc2_1
    return (acc2_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can separate the equation above for the linear acceleration in three types of terms: proportional to <span class="notranslate">$\ddot{\theta}$</span> and to <span class="notranslate">$\dot{\theta}^2$</span>, as we already saw for the one-link case, and a new term, proportional to <span class="notranslate">$\dot{\theta}_1\dot{\theta}_2$</span>:
        """
    )
    return


@app.cell
def _(Math, Matrix, acc2_1, display, latex, mlatex, t_1, θ1, θ2):
    acc2_2 = acc2_1.expand()
    _A = θ1.diff(t_1, 2)
    B = θ2.diff(t_1, 2)
    tg = _A * Matrix((acc2_2[0].coeff(_A), acc2_2[1].coeff(_A))) + B * Matrix((acc2_2[0].coeff(B), acc2_2[1].coeff(B)))
    _A = θ1.diff(t_1) ** 2
    B = θ2.diff(t_1) ** 2
    ct = _A * Matrix((acc2_2[0].coeff(_A), acc2_2[1].coeff(_A))) + B * Matrix((acc2_2[0].coeff(B), acc2_2[1].coeff(B)))
    _A = θ1.diff(t_1) * θ2.diff(t_1)
    co = _A * Matrix((acc2_2[0].coeff(_A), acc2_2[1].coeff(_A)))
    display(Math('\\text{Tangential:} \\:\\,' + latex(tg)))
    print()
    display(Math('\\text{Centripetal:}' + mlatex(ct)))
    print()
    display(Math('\\text{Coriolis:} \\quad\\,' + mlatex(co)))
    return acc2_2, co, ct, tg


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This new term is called the [Coriolis acceleration](http://en.wikipedia.org/wiki/Coriolis_effect); it is 'felt' by the endpoint when its distance to the instantaneous center of rotation varies, due to the links' constraints, and as consequence the endpoint motion is deflected (its direction is perpendicular to the relative linear velocity of the endpoint with respect to the linear velocity at the second joint, <span class="notranslate">$\mathbf{v} - \mathbf{v}_{joint2}$</span>.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's now deduce the Jacobian for this planar two-link chain:  
        <br>
        <span class="notranslate">$\mathbf{J} =
        \large
        \begin{bmatrix}
        \frac{\partial x_P}{\partial \theta_{1}} & \frac{\partial x_P}{\partial \theta_{2}} \\
        \frac{\partial y_P}{\partial \theta_{1}} & \frac{\partial y_P}{\partial \theta_{2}} \\
        \end{bmatrix}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We could manually run:  
        ```python
        J = Matrix([[r2[0].diff(theta1), r2[0].diff(theta2)], [r2[1].diff(theta1), r2[1].diff(theta2)]])
        ```
        But it's shorter with the Jacobian function from Sympy:
        """
    )
    return


@app.cell
def _(r2_1, θ1, θ2):
    J2 = r2_1.jacobian([θ1, θ2])
    J2
    return (J2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using the Jacobian, the linear velocity of the endpoint is:   
        <br>
        <span class="notranslate">$\mathbf{v_J} = \mathbf{J} \cdot \begin{bmatrix}\dot{\theta_1} \\
            \dot{\theta_2}\\
            \end{bmatrix}$</span>

        Where:
        """
    )
    return


@app.cell
def _(Matrix, t_1, θ1, θ2):
    ω2 = Matrix((θ1, θ2)).diff(t_1)
    ω2
    return (ω2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Then:
        """
    )
    return


@app.cell
def _(J2, ω2):
    vel2J = J2*ω2
    vel2J
    return (vel2J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This expression derived with the Jacobian is the same as the first-order derivative  of the equation for the endpoint position. We can show this equality by comparing the two expressions with Sympy:
        """
    )
    return


@app.cell
def _(vel2, vel2J):
    vel2.expand() == vel2J.expand()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once again, the linear acceleration of the endpoint is given by the derivative of the product between the Jacobian and the angular velocity:  
        <br>
        <span class="notranslate">$\mathbf{a} = \dot{\mathbf{J}} \cdot \mathbf{\omega} + \mathbf{J} \cdot \dot{\mathbf{\omega}}$</span>

        Let's calculate this derivative:
        """
    )
    return


@app.cell
def _(J2, t_1, ω2):
    acc2J = J2.diff(t_1) * ω2 + J2 * ω2.diff(t_1)
    acc2J
    return (acc2J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once again, the expression above is the same as the second-order derivative of the equation for the endpoint position:
        """
    )
    return


@app.cell
def _(acc2J, acc2_2):
    acc2_2.expand() == acc2J.expand()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Simulation

        Let's plot some simulated data to have an idea of the two-link kinematics.  
        Consider 1 s of movement duration, <span class="notranslate">$\ell_1=\ell_2=0.5m, \theta_1(0)=\theta_2(0)=0$</span>, <span class="notranslate">$\theta_1(1)=\theta_2(1)=90^o$</span>, and that the endpoint trajectory is a [minimum-jerk movement](https://nbviewer.jupyter.org/github/BMClab/bmc/blob/master/notebooks/MinimumJerkHypothesis.ipynb).   

        First, the simulated trajectories:
        """
    )
    return


@app.cell
def _(
    acos,
    atan,
    atan2,
    cos,
    diff,
    dynamicsymbols,
    l1,
    l2,
    lambdify,
    np,
    sin,
    symbols,
):
    (t_2, p0, pf, _d) = symbols('t p0 pf d')
    rx = dynamicsymbols('rx', real=True)
    ry = dynamicsymbols('ry', real=True)
    _mjt = p0 + (pf - p0) * (10 * (t_2 / _d) ** 3 - 15 * (t_2 / _d) ** 4 + 6 * (t_2 / _d) ** 5)
    rfu = lambdify((t_2, p0, pf, _d), _mjt, 'numpy')
    vfu = lambdify((t_2, p0, pf, _d), diff(_mjt, t_2, 1), 'numpy')
    afu = lambdify((t_2, p0, pf, _d), diff(_mjt, t_2, 2), 'numpy')
    jfu = lambdify((t_2, p0, pf, _d), diff(_mjt, t_2, 3), 'numpy')
    (_d, L1, L2) = (1, 0.5, 0.5)
    (p0, pf) = ([-0.5, 0.5], [0, 0.5 * np.sqrt(2)])
    ts_1 = np.arange(0.01, 1.01, 0.01)
    x = rfu(ts_1, p0[0], pf[0], _d)
    y = rfu(ts_1, p0[1], pf[1], _d)
    vx = vfu(ts_1, p0[0], pf[0], _d)
    vy = vfu(ts_1, p0[1], pf[1], _d)
    ax = afu(ts_1, p0[0], pf[0], _d)
    ay = afu(ts_1, p0[1], pf[1], _d)
    jx = jfu(ts_1, p0[0], pf[0], _d)
    jy = jfu(ts_1, p0[1], pf[1], _d)
    ang2b = np.arccos((x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2))
    ang1b = np.arctan2(y, x) - np.arctan2(L2 * np.sin(ang2b), L1 + L2 * np.cos(ang2b))
    ang2 = acos((rx ** 2 + ry ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
    ang2fu = lambdify((rx, ry, l1, l2), ang2, 'numpy')
    ang2 = ang2fu(x, y, L1, L2)
    ang1 = atan2(ry, rx) - atan(l2 * sin(acos((rx ** 2 + ry ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))) / (l1 + l2 * cos(acos((rx ** 2 + ry ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)))))
    ang1fu = lambdify((rx, ry, l1, l2), ang1, 'numpy')
    ang1 = ang1fu(x, y, L1, L2)
    ang2b = acos((rx ** 2 + ry ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
    ang1b = atan2(ry, rx) - atan(l2 * sin(acos((rx ** 2 + ry ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))) / (l1 + l2 * cos(acos((rx ** 2 + ry ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)))))
    (X, Y, Xd, Yd, Xdd, Ydd, Xddd, Yddd) = symbols('X Y Xd Yd Xdd Ydd Xddd Yddd')
    _dicti = {rx: X, ry: Y, rx.diff(t_2, 1): Xd, ry.diff(t_2, 1): Yd, rx.diff(t_2, 2): Xdd, ry.diff(t_2, 2): Ydd, rx.diff(t_2, 3): Xddd, ry.diff(t_2, 3): Yddd, l1: L1, l2: L2}
    vang1 = diff(ang1b, t_2, 1)
    vang1 = vang1.subs(_dicti)
    vang1fu = lambdify((X, Y, Xd, Yd, l1, l2), vang1, 'numpy')
    vang1 = vang1fu(x, y, vx, vy, L1, L2)
    vang2 = diff(ang2b, t_2, 1)
    vang2 = vang2.subs(_dicti)
    vang2fu = lambdify((X, Y, Xd, Yd, l1, l2), vang2, 'numpy')
    vang2 = vang2fu(x, y, vx, vy, L1, L2)
    aang1 = diff(ang1b, t_2, 2)
    aang1 = aang1.subs(_dicti)
    aang1fu = lambdify((X, Y, Xd, Yd, Xdd, Ydd, l1, l2), aang1, 'numpy')
    aang1 = aang1fu(x, y, vx, vy, ax, ay, L1, L2)
    aang2 = diff(ang2b, t_2, 2)
    aang2 = aang2.subs(_dicti)
    aang2fu = lambdify((X, Y, Xd, Yd, Xdd, Ydd, l1, l2), aang2, 'numpy')
    aang2 = aang2fu(x, y, vx, vy, ax, ay, L1, L2)
    jang1 = diff(ang1b, t_2, 3)
    jang1 = jang1.subs(_dicti)
    jang1fu = lambdify((X, Y, Xd, Yd, Xdd, Ydd, Xddd, Yddd, l1, l2), jang1, 'numpy')
    jang1 = jang1fu(x, y, vx, vy, ax, ay, jx, jy, L1, L2)
    jang2 = diff(ang2b, t_2, 3)
    jang2 = jang2.subs(_dicti)
    jang2fu = lambdify((X, Y, Xd, Yd, Xdd, Ydd, Xddd, Yddd, l1, l2), jang2, 'numpy')
    jang2 = jang2fu(x, y, vx, vy, ax, ay, jx, jy, L1, L2)
    return (
        L1,
        L2,
        aang1,
        aang2,
        ang1,
        ang2,
        ax,
        ay,
        jang1,
        jang2,
        jx,
        jy,
        t_2,
        ts_1,
        vang1,
        vang2,
        vx,
        vy,
        x,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the plots for the trajectories:
        """
    )
    return


@app.cell
def _(
    aang1,
    aang2,
    ang1,
    ang2,
    ax,
    ay,
    jang1,
    jang2,
    jx,
    jy,
    np,
    plt,
    ts_1,
    vang1,
    vang2,
    vx,
    vy,
    x,
    y,
):
    (_fig, _hax) = plt.subplots(2, 4, sharex=True, figsize=(14, 7))
    _hax[0, 0].plot(ts_1, x, 'r', linewidth=3, label='x')
    _hax[0, 0].plot(ts_1, y, 'k', linewidth=3, label='y')
    _hax[0, 0].set_title('Linear displacement [$m$]')
    _hax[0, 0].legend(loc='best').get_frame().set_alpha(0.8)
    _hax[0, 0].set_ylabel('Endpoint')
    _hax[0, 1].plot(ts_1, vx, 'r', linewidth=3)
    _hax[0, 1].plot(ts_1, vy, 'k', linewidth=3)
    _hax[0, 1].set_title('Linear velocity [$m/s$]')
    _hax[0, 2].plot(ts_1, ax, 'r', linewidth=3)
    _hax[0, 2].plot(ts_1, ay, 'k', linewidth=3)
    _hax[0, 2].set_title('Linear acceleration [$m/s^2$]')
    _hax[0, 3].plot(ts_1, jx, 'r', linewidth=3)
    _hax[0, 3].plot(ts_1, jy, 'k', linewidth=3)
    _hax[0, 3].set_title('Linear jerk [$m/s^3$]')
    _hax[1, 0].plot(ts_1, ang1 * 180 / np.pi, 'b', linewidth=3, label='Ang1')
    _hax[1, 0].plot(ts_1, ang2 * 180 / np.pi, 'g', linewidth=3, label='Ang2')
    _hax[1, 0].set_title('Angular displacement [$^o$]')
    _hax[1, 0].legend(loc='best').get_frame().set_alpha(0.8)
    _hax[1, 0].set_ylabel('Joint')
    _hax[1, 1].plot(ts_1, vang1 * 180 / np.pi, 'b', linewidth=3)
    _hax[1, 1].plot(ts_1, vang2 * 180 / np.pi, 'g', linewidth=3)
    _hax[1, 1].set_title('Angular velocity [$^o/s$]')
    _hax[1, 2].plot(ts_1, aang1 * 180 / np.pi, 'b', linewidth=3)
    _hax[1, 2].plot(ts_1, aang2 * 180 / np.pi, 'g', linewidth=3)
    _hax[1, 2].set_title('Angular acceleration [$^o/s^2$]')
    _hax[1, 3].plot(ts_1, jang1 * 180 / np.pi, 'b', linewidth=3)
    _hax[1, 3].plot(ts_1, jang2 * 180 / np.pi, 'g', linewidth=3)
    _hax[1, 3].set_title('Angular jerk [$^o/s^3$]')
    _tit = _fig.suptitle('Minimum jerk trajectory kinematics of a two-link chain', fontsize=20)
    for (_i, _hax2) in enumerate(_hax.flat):
        _hax2.locator_params(nbins=5)
        _hax2.grid(True)
        if _i > 3:
            _hax2.set_xlabel('Time [$s$]')
    plt.subplots_adjust(hspace=0.15, wspace=0.25)
    (_fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(x, y, 'r', linewidth=3)
    ax1.set_xlabel('Displacement in x [$m$]')
    ax1.set_ylabel('Displacement in y [$m$]')
    ax1.set_title('Endpoint space', fontsize=14)
    ax1.axis('equal')
    ax1.grid(True)
    ax2.plot(ang1 * 180 / np.pi, ang2 * 180 / np.pi, 'b', linewidth=3)
    ax2.set_xlabel('Displacement in joint 1 [$^o$]')
    ax2.set_ylabel('Displacement in joint 2 [$^o$]')
    ax2.set_title('Joint sapace', fontsize=14)
    ax2.axis('equal')
    ax2.grid(True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Calculation of each type of acceleration of the endpoint for the numerical example of the two-link system
        """
    )
    return


@app.cell
def _(
    L1,
    L2,
    aang1,
    aang2,
    ang1,
    ang2,
    co,
    ct,
    l1,
    l2,
    lambdify,
    symbols,
    t_2,
    tg,
    vang1,
    vang2,
    θ1,
    θ2,
):
    (A1, A2, A1d, A2d, A1dd, A2dd) = symbols('A1 A2 A1d A2d A1dd A2dd')
    _dicti = {θ1: A1, θ2: A2, θ1.diff(t_2, 1): A1d, θ2.diff(t_2, 1): A2d, θ1.diff(t_2, 2): A1dd, θ2.diff(t_2, 2): A2dd, l1: L1, l2: L2}
    tg2 = tg.subs(_dicti)
    tg2fu = lambdify((A1, A2, A1dd, A2dd), tg2, 'numpy')
    tg2n = tg2fu(ang1, ang2, aang1, aang2)
    tg2n = tg2n.reshape((2, 100)).T
    ct2 = ct.subs(_dicti)
    ct2fu = lambdify((A1, A2, A1d, A2d), ct2, 'numpy')
    ct2n = ct2fu(ang1, ang2, vang1, vang2)
    ct2n = ct2n.reshape((2, 100)).T
    co2 = co.subs(_dicti)
    co2fu = lambdify((A1, A2, A1d, A2d), co2, 'numpy')
    co2n = co2fu(ang1, ang2, vang1, vang2)
    co2n = co2n.reshape((2, 100)).T
    acc_tot = tg2n + ct2n + co2n
    return acc_tot, co2n, ct2n, tg2n


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### And the corresponding plots
        """
    )
    return


@app.cell
def _(acc_tot, co2n, ct2n, plt, tg2n, ts_1):
    (_fig, _hax) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 5))
    _hax[0].plot(ts_1, acc_tot[:, 0], color=(1, 0, 0, 0.3), linewidth=5, label='x total')
    _hax[0].plot(ts_1, acc_tot[:, 1], color=(0, 0, 0, 0.3), linewidth=5, label='y total')
    _hax[0].plot(ts_1, tg2n[:, 0], 'r', linewidth=2, label='x')
    _hax[0].plot(ts_1, tg2n[:, 1], 'k', linewidth=2, label='y')
    _hax[0].set_title('Tangential')
    _hax[0].set_ylabel('Endpoint acceleration [$m/s^2$]')
    _hax[0].set_xlabel('Time [$s$]')
    _hax[1].plot(ts_1, acc_tot[:, 0], color=(1, 0, 0, 0.3), linewidth=5, label='x total')
    _hax[1].plot(ts_1, acc_tot[:, 1], color=(0, 0, 0, 0.3), linewidth=5, label='y total')
    _hax[1].plot(ts_1, ct2n[:, 0], 'r', linewidth=2, label='x')
    _hax[1].plot(ts_1, ct2n[:, 1], 'k', linewidth=2, label='y')
    _hax[1].set_title('Centripetal')
    _hax[1].set_xlabel('Time [$s$]')
    _hax[1].legend(loc='best').get_frame().set_alpha(0.8)
    _hax[2].plot(ts_1, acc_tot[:, 0], color=(1, 0, 0, 0.3), linewidth=5, label='x total')
    _hax[2].plot(ts_1, acc_tot[:, 1], color=(0, 0, 0, 0.3), linewidth=5, label='y total')
    _hax[2].plot(ts_1, co2n[:, 0], 'r', linewidth=2, label='x')
    _hax[2].plot(ts_1, co2n[:, 1], 'k', linewidth=2, label='y')
    _hax[2].set_title('Coriolis')
    _hax[2].set_xlabel('Time [$s$]')
    _tit = _fig.suptitle('Acceleration terms for the minimum jerk trajectory of a two-link chain', fontsize=16)
    for _hax2 in _hax:
        _hax2.locator_params(nbins=5)
        _hax2.grid(True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

         - Read pages 477-494 of the 10th chapter of the [Ruina and Rudra's book](http://ruina.tam.cornell.edu/Book/index.html) for a review of differential equations and kinematics.  
         - Read section 2.8 of Rade's book about using the concept of rotation matrix applied to kinematic chains.  
         - Read [Computational Motor Control: Kinematics](https://gribblelab.org/teaching/compneuro2012/4_Computational_Motor_Control_Kinematics.html) about the concept of kinematic chain for modeling the human upper limb.   
         - For more about the Jacobian matrix, read [Jacobian matrix and determinant](https://www.algebrapracticeproblems.com/jacobian-matrix-determinant/).  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

         - Khan Academy: [Differential Calculus Review](https://khanacademy.org/math/differential-calculus)
         - Khan Academy: [Chain Rule Review](https://khanacademy.org/math/differential-calculus/dc-chain)
         - [Multivariate Calculus – Jacobian applied](https://www.youtube.com/watch?v=57q-2YxIZss)  
         - [Kinematics of the Two Link Arm](https://youtu.be/3WAk8ABj-kg)  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problems

        1. Calculate the Jacobian matrix for the following function:  
        <span class="notranslate">$f(x, y) = \begin{bmatrix}
        x^2 y \\
        5 x + \sin y \end{bmatrix}$</span>  

        2. For the two-link chain, calculate and interpret the Jacobian and the expressions for the position, velocity, and acceleration of the endpoint for the following cases:   
         a) When the first joint (the joint at the base) is fixed at <span class="notranslate">$0^o$</span>.   
         b) When the second joint is fixed at <span class="notranslate">$0^o$</span>.  

        3. Deduce the equations for the kinematics of a two-link pendulum with the angles in relation to the vertical.  

        4. Deduce the equations for the kinematics of a two-link system using segment angles and compare with the deduction employing joint angles.  

        5. For the two-link chain, a special case of movement occurs when the endpoint moves along a line passing through the first joint (the joint at the base). A system with this behavior is known as a polar manipulator (Mussa-Ivaldi, 1986). For simplicity, consider that the lengths of the two links are equal to$\ell$. In this case, the two joint angles are related by: <span class="notranslate">$2\theta_1+\theta_2=\pi$</span>.  
         a) Calculate the Jacobian for this polar manipulator and compare it with the Jacobian for the  standard two-link chain. Note the difference between the off-diagonal terms.  
         b) Calculate the expressions for the endpoint position, velocity, and acceleration.   
         c) For the endpoint acceleration of the polar manipulator, identify the tangential, centrifugal, and Coriolis components and compare them with the expressions for the standard two-link chain.  

        6. [Ex. 2.22, Rade's book] In an internal combustion engine, crank$AB$, shown in figure below, has a constant rotation of 1000 rpm, counterclockwise. For position$\theta=30^o$, determine: a) the angular velocity of connecting rod$BD$; b) the speed of piston$P$. Solution: a)$\omega_{BD}=-44.88$rad/s. b)$v_P=-8.976$m/s. <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/motor_2rods.png?raw=1" width=450/></figure>
        <br>  

        7. [Ex. 2.30, Rade's book] The robot shown in the figure below has a plane motion, with velocity$\overrightarrow{v}_0$and acceleration$\overrightarrow{a}_0$in relation to the floor. Arm$O_1O_2$rotates with angular velocity$\omega_1=\dot{\theta}_1$and angular acceleration$\alpha_1=\ddot{\theta}_1$relative to the robot's body, and arm$O_2P$rotates with angular velocity$\omega_2=\dot{\theta}_2$and angular acceleration$\alpha_2=\ddot{\theta}_2$relative to the arm$O_1O_2$, with the directions indicated. Designating by$\ell_1$the length of the arm$O_1O_2$and by$\ell_2$the length of the arm$O_2P$, obtain the expressions for the velocity$\overrightarrow{v}_P$and the acceleration$\overrightarrow{a}_P$of the endpoint$P$, with respect to the floor, as function of the indicated parameters.
        <br>
        <figure><img src="https://github.com/BMClab/BMC/blob/master/images/robot_arm.png?raw=1" width=450/></figure>
        <br>
        Solution for$\overrightarrow{v}$:  
        <span class="notranslate">$\overrightarrow{v}_{P, OXY} = \begin{bmatrix}
        v_0+\dot{\theta}_1\ell_1\cos(\theta_1)+(\dot{\theta}_1+\dot{\theta}_2)\ell_2\cos(\theta_1+\theta_2) \\
        \dot{\theta}_1\ell_1\sin(\theta_1)+(\dot{\theta}_1+\dot{\theta}_2)\ell_2\sin(\theta_1+\theta_2) \end{bmatrix}$</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Mussa-Ivaldi FA (1986) Compliance.  In: Morasso P Tagliasco V (eds), [Human Movement Understanding: from computational geometry to artificial Intelligence](http://books.google.com.br/books?id=ZlZyLKNoAtEC). North-Holland, Amsterdam.  
        - Rade D (2017) [Cinemática e Dinâmica para Engenharia](https://www.grupogen.com.br/e-book-cinematica-e-dinamica-para-engenharia). Grupo GEN.  
        - Ruina A, Rudra P (2019) [Introduction to Statics and Dynamics](http://ruina.tam.cornell.edu/Book/index.html). Oxford University Press.  
        - Siciliano B et al. (2009) [Robotics - Modelling, Planning and Control](http://books.google.com.br/books/about/Robotics.html?hl=pt-BR&id=jPCAFmE-logC). Springer-Verlag London.
        - Zatsiorsky VM (1998) [Kinematics of Human Motions](http://books.google.com.br/books/about/Kinematics_of_Human_Motion.html?id=Pql_xXdbrMcC&redir_esc=y). Champaign, Human Kinetics.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
